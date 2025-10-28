import datetime
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from lib.agent_n_p3o import NP3OAgent
from lib.buffer_n_p3o import NP3OBuffer
from lib.symmetry import HumanoidSymmetry, augment_batch_with_symmetry
from lib.utils import parse_args_ppo, make_env, log_video


def n_p3o_update(agent, optimizer, scaler, batch_obs, batch_actions, batch_returns,
                 batch_old_log_probs, batch_reward_adv, batch_cost_adv, 
                 batch_old_values, batch_old_cost_values,
                 clip_epsilon, vf_coef, ent_coef, kappa, gamma):
    """
    N-P3O update with constraint penalty
    """
    agent.train()
    
    optimizer.zero_grad()
    with torch.amp.autocast(str(device)):
        # Get new predictions
        _, new_log_probs, entropies, new_reward_values, new_cost_values = \
            agent.get_action_and_value(batch_obs, batch_actions)
        
        ratio = torch.exp(new_log_probs - batch_old_log_probs)
        
        # Approximate KL
        kl = ((batch_old_log_probs - new_log_probs) / batch_actions.size(-1)).mean()
        
        # === REWARD OBJECTIVE (PPO clipped) ===
        surr1_reward = ratio * batch_reward_adv
        surr2_reward = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_reward_adv
        L_clip_R = torch.min(surr1_reward, surr2_reward).mean()
        
        # === COST OBJECTIVE (PPO clipped, but MAX for cost) ===
        surr1_cost = ratio * batch_cost_adv
        surr2_cost = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_cost_adv
        L_clip_C = torch.max(surr1_cost, surr2_cost).mean()
        
        # === CONSTRAINT VIOLATION TERM ===
        mu_C = batch_cost_adv.mean()
        sigma_C = batch_cost_adv.std() + 1e-8
        mean_cost = batch_old_cost_values.mean()  # Approximate
        constraint_term = (1 - gamma) * mean_cost + mu_C / sigma_C
        L_viol = L_clip_C + constraint_term
        
        # === N-P3O POLICY LOSS ===
        policy_loss = -(L_clip_R - kappa * torch.max(torch.tensor(0.0, device=device), L_viol))
        
        # === VALUE LOSSES (with clipping for stability) ===
        # Reward value loss
        value_pred_clipped = batch_old_values + torch.clamp(
            new_reward_values.squeeze(1) - batch_old_values,
            -clip_epsilon, clip_epsilon
        )
        value_loss_unclipped = nn.MSELoss()(new_reward_values.squeeze(1), batch_returns)
        value_loss_clipped = nn.MSELoss()(value_pred_clipped, batch_returns)
        reward_value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
        
        # Cost value loss
        cost_value_loss = nn.MSELoss()(new_cost_values.squeeze(1), batch_old_cost_values)
        
        # === TOTAL LOSS ===
        entropy = entropies.mean()
        loss = (policy_loss + 
                vf_coef * reward_value_loss + 
                vf_coef * cost_value_loss - 
                ent_coef * entropy)
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
    scaler.step(optimizer)
    scaler.update()
    
    return (loss.item(), policy_loss.item(), reward_value_loss.item(), 
            cost_value_loss.item(), entropy.item(), kl.item(), 
            L_viol.item(), grad_norm.item())


if __name__ == "__main__":
    args = parse_args_ppo()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    # Create folders
    current_dir = os.path.dirname(__file__)
    folder_name = f"n_p3o_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    videos_dir = os.path.join(current_dir, "videos", folder_name)
    os.makedirs(videos_dir, exist_ok=True)
    checkpoint_dir = os.path.join(current_dir, "checkpoints", folder_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # TensorBoard
    log_dir = os.path.join(current_dir, "logs", folder_name)
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Create environments
    envs = gym.vector.AsyncVectorEnv(
        [lambda: make_env(args.env, reward_scaling=args.reward_scale) for _ in range(args.n_envs)])
    test_env = make_env(args.env, reward_scaling=args.reward_scale, render=True)
    obs_dim = envs.single_observation_space.shape
    act_dim = envs.single_action_space.shape
    
    # Create agent
    agent = NP3OAgent(obs_dim[0], act_dim[0]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Learning rate scheduler
    def lr_lambda(epoch):
        warmup_epochs = 10
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            T_cur = epoch - warmup_epochs
            T_total = args.n_epochs - warmup_epochs
            return 0.5 * (1 + np.cos(np.pi * T_cur / T_total))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(str(device))
    
    # Symmetry
    symmetry = HumanoidSymmetry()
    use_symmetry = True  # Set to False to disable
    
    # N-P3O specific parameters
    kappa = 0.1  # Initial penalty weight
    kappa_max = 5.0
    kappa_growth = 1.0002
    
    # Buffer
    buffer = NP3OBuffer(obs_dim, act_dim, args.n_steps, args.n_envs, device, 
                        args.gamma, args.gae_lambda)
    
    # Training loop
    global_step_idx = 0
    best_mean_reward = -np.inf
    start_time = time.time()
    next_obs = torch.tensor(np.array(envs.reset()[0], dtype=np.float32), device=device)
    next_terminateds = torch.zeros(args.n_envs, dtype=torch.float32, device=device)
    next_truncateds = torch.zeros(args.n_envs, dtype=torch.float32, device=device)
    reward_list = []
    cost_list = []
    
    print(f"🚀 Starting N-P3O training with symmetry={use_symmetry}")
    
    try:
        for epoch in range(1, args.n_epochs + 1):
            
            # === PHASE 1: COLLECT TRAJECTORIES ===
            for _ in tqdm(range(0, args.n_steps), desc=f"Epoch {epoch}: Collecting"):
                global_step_idx += args.n_envs
                obs = next_obs
                terminateds = next_terminateds
                truncateds = next_truncateds
                
                # Sample actions
                with torch.no_grad():
                    actions, logprobs, _, reward_values, cost_values = \
                        agent.get_action_and_value(obs)
                    reward_values = reward_values.reshape(-1)
                    cost_values = cost_values.reshape(-1)
                
                # Step environment
                next_obs, rewards, next_terminateds, next_truncateds, infos = \
                    envs.step(actions.cpu().numpy())
                
                # Extract costs from info
                costs = np.array([info.get('cost', 0.0) for info in infos])
                
                # Convert to tensors
                next_obs = torch.tensor(np.array(next_obs, dtype=np.float32), device=device)
                reward_list.extend(rewards)
                cost_list.extend(costs)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                costs = torch.tensor(costs, dtype=torch.float32, device=device)
                next_terminateds = torch.as_tensor(next_terminateds, dtype=torch.float32, device=device)
                next_truncateds = torch.as_tensor(next_truncateds, dtype=torch.float32, device=device)
                
                # Store in buffer
                buffer.store(obs, actions, rewards, costs, reward_values, cost_values,
                           terminateds, truncateds, logprobs)
            
            # === PHASE 2: CALCULATE ADVANTAGES ===
            with torch.no_grad():
                next_reward_value = agent.get_value(next_obs).reshape(1, -1)
                next_cost_value = agent.get_cost_value(next_obs).reshape(1, -1)
                next_terminateds = next_terminateds.reshape(1, -1)
                next_truncateds = next_truncateds.reshape(1, -1)
                
                traj_reward_adv, traj_reward_ret, traj_cost_adv = \
                    buffer.calculate_advantages(next_reward_value, next_cost_value,
                                               next_terminateds, next_truncateds)
            
            # Get trajectories
            traj_obs, traj_act, traj_logprob, traj_rew, traj_cost = buffer.get()
            
            # Flatten
            traj_obs = traj_obs.view(-1, *obs_dim)
            traj_act = traj_act.view(-1, *act_dim)
            traj_logprob = traj_logprob.view(-1)
            traj_reward_adv = traj_reward_adv.view(-1)
            traj_cost_adv = traj_cost_adv.view(-1)
            traj_reward_ret = traj_reward_ret.view(-1)
            traj_old_values = buffer.val_buf.view(-1)
            traj_old_cost_values = buffer.cost_val_buf.view(-1)
            
            # Normalize advantages (CRITICAL!)
            traj_reward_adv = (traj_reward_adv - traj_reward_adv.mean()) / (traj_reward_adv.std() + 1e-8)
            traj_cost_adv = (traj_cost_adv - traj_cost_adv.mean()) / (traj_cost_adv.std() + 1e-8)
            
            # === PHASE 3: UPDATE POLICY ===
            dataset_size = args.n_steps * args.n_envs
            traj_indices = np.arange(dataset_size)
            
            losses_policy = []
            losses_value = []
            losses_cost_value = []
            entropies = []
            losses_total = []
            kl_list = []
            constraint_violations = []
            grad_norms = []
            
            kl_early_stop = False
            
            for train_iter in tqdm(range(args.train_iters), desc=f"Epoch {epoch}: Training"):
                np.random.shuffle(traj_indices)
                
                for start_idx in range(0, dataset_size, args.batch_size):
                    end_idx = start_idx + args.batch_size
                    batch_indices = traj_indices[start_idx:end_idx]
                    
                    # Get batch
                    batch_obs = traj_obs[batch_indices]
                    batch_actions = traj_act[batch_indices]
                    batch_returns = traj_reward_ret[batch_indices]
                    batch_old_log_probs = traj_logprob[batch_indices]
                    batch_reward_adv = traj_reward_adv[batch_indices]
                    batch_cost_adv = traj_cost_adv[batch_indices]
                    batch_old_values = traj_old_values[batch_indices]
                    batch_old_cost_values = traj_old_cost_values[batch_indices]
                    
                    # === SYMMETRY AUGMENTATION ===
                    if use_symmetry:
                        batch_obs, batch_actions, batch_old_log_probs, \
                        batch_reward_adv, batch_cost_adv, batch_returns, \
                        batch_old_values, batch_old_cost_values = \
                            augment_batch_with_symmetry(
                                batch_obs, batch_actions, batch_old_log_probs,
                                batch_reward_adv, batch_cost_adv, batch_returns,
                                batch_old_values, batch_old_cost_values, symmetry
                            )
                    
                    # Update
                    loss, policy_loss, value_loss, cost_value_loss, entropy, kl, \
                    constraint_viol, grad_norm = n_p3o_update(
                        agent, optimizer, scaler, batch_obs, batch_actions, batch_returns,
                        batch_old_log_probs, batch_reward_adv, batch_cost_adv,
                        batch_old_values, batch_old_cost_values,
                        args.clip_ratio, args.vf_coef, args.ent_coef, kappa, args.gamma
                    )
                    
                    losses_policy.append(policy_loss)
                    losses_value.append(value_loss)
                    losses_cost_value.append(cost_value_loss)
                    entropies.append(entropy)
                    losses_total.append(loss)
                    kl_list.append(kl)
                    constraint_violations.append(constraint_viol)
                    grad_norms.append(grad_norm)
                    
                    # Early stopping
                    if kl > args.target_kl:
                        kl_early_stop = True
                        break
                
                if kl_early_stop:
                    print(f"  Early stop at iter {train_iter} due to KL={np.mean(kl_list):.6f}")
                    break
            
            # Update kappa
            kappa = min(kappa_max, kappa * kappa_growth)
            
            # === PHASE 4: LOGGING ===
            mean_reward = float(np.mean(reward_list) / args.reward_scale)
            mean_cost = float(np.mean(cost_list))
            mean_constraint_viol = np.mean(constraint_violations)
            
            writer.add_scalar("loss/total", np.mean(losses_total), epoch)
            writer.add_scalar("loss/policy", np.mean(losses_policy), epoch)
            writer.add_scalar("loss/reward_value", np.mean(losses_value), epoch)
            writer.add_scalar("loss/cost_value", np.mean(losses_cost_value), epoch)
            writer.add_scalar("loss/entropy", np.mean(entropies), epoch)
            writer.add_scalar("metrics/kl", np.mean(kl_list), epoch)
            writer.add_scalar("metrics/kappa", kappa, epoch)
            writer.add_scalar("metrics/constraint_violation", mean_constraint_viol, epoch)
            writer.add_scalar("metrics/grad_norm", np.mean(grad_norms), epoch)
            writer.add_scalar("metrics/learning_rate", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("reward/mean", mean_reward, epoch)
            writer.add_scalar("cost/mean", mean_cost, epoch)
            
            reward_list = []
            cost_list = []
            
            print(f"Epoch {epoch} | {time.time() - start_time:.1f}s | "
                  f"R={mean_reward:.2f} C={mean_cost:.2f} κ={kappa:.3f} "
                  f"KL={np.mean(kl_list):.4f} GradNorm={np.mean(grad_norms):.3f}")
            start_time = time.time()
            
            # Save best model
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                torch.save(agent.state_dict(), os.path.join(checkpoint_dir, "best.pt"))
                print(f"  🏆 New best: {mean_reward:.2f}")
            
            # Save last
            torch.save(agent.state_dict(), os.path.join(checkpoint_dir, "last.pt"))
            
            # Video
            if epoch % args.render_epoch == 0:
                log_video(test_env, agent, device, os.path.join(videos_dir, f"epoch_{epoch}.mp4"))
            
            # Update LR
            scheduler.step()
    
    finally:
        envs.close()
        test_env.close()
        writer.close()
        print("✓ Training complete!")