# Import Required External Libraries
import os
import _20_model

import torch
import torch.nn as nn
import torch.optim as optim

# Import Required Internal Libraries
from _20_model import ppo


class PPO:
    def __init__(self, conf, policy_name_for_play=None):
        """================================================================================================
        ## Parameters for PPO
        ================================================================================================"""
        self.conf = conf
        self.train_conf = self.get_train_configuration()

        self.gamma = float(self.train_conf["gamma"])
        self.gae_lambda = float(self.train_conf.get("gae_lambda", 0.95))
        self.clip_epsilon = float(self.train_conf["clip_epsilon"])
        self.value_clip_epsilon = float(self.train_conf.get("value_clip_epsilon", 0.0))
        if self.value_clip_epsilon < 0.0:
            self.value_clip_epsilon = 0.0
        self.vf_coefficient = float(self.train_conf.get("vf_coefficient", 0.5))
        if self.vf_coefficient < 0.0:
            self.vf_coefficient = 0.0
        self.update_epochs = int(self.train_conf["update_epochs"])
        self.minibatch_size = int(self.train_conf.get("minibatch_size", 0))
        if self.minibatch_size < 1:
            self.minibatch_size = 0
        self.target_kl = float(self.train_conf.get("target_kl", 0.0))
        if self.target_kl < 0.0:
            self.target_kl = 0.0
        self.entropy_coefficient_start = float(
            self.train_conf.get("entropy_coefficient", 0.01)
        )
        self.entropy_coefficient_end = float(
            self.train_conf.get("entropy_coefficient_end", self.entropy_coefficient_start)
        )
        self.entropy_decay_episodes = int(
            self.train_conf.get("entropy_decay_episodes", 0)
        )
        self.entropy_coefficient = float(self.entropy_coefficient_start)
        self.max_grad_norm = float(
            self.train_conf.get("max_grad_norm", 0.5)
        )
        self.advantage_epsilon = float(
            self.train_conf.get("advantage_epsilon", 1e-8)
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.learning_rate_actor = float(
            self.train_conf["learning_rate_actor"])
        self.learning_rate_critic = float(
            self.train_conf["learning_rate_critic"])

        self.state_dim = int(ppo._03_state_design.get_state_dim())
        self.dim_action = len(ppo._04_action_space_design.action_mask())
        self.hidden_dim = int(self.train_conf["hidden_dim"])
        self.hidden_layer_count = int(self.train_conf["hidden_layer_count"])

        self.loss_function = nn.MSELoss()
        self.rollout_states = []
        self.rollout_states_next = []
        self.rollout_action_indices = []
        self.rollout_log_probs_old = []
        self.rollout_rewards = []
        self.rollout_dones = []
        self.training_episode = 0
        self.last_approx_kl = None
        self.last_update_epochs = 0

        if policy_name_for_play is not None:
            self.policy_name = str(policy_name_for_play).strip()
        else:
            self.policy_name = str(self.conf.train_policy).strip()

        self.actor_path = os.path.join(
            _20_model.get_model_policy_dir(self.conf, self),
            self.policy_name + '.pth',
        )
        self.policy_path = self.actor_path
        self.critic_path = os.path.join(
            _20_model.get_model_policy_dir(self.conf, self),
            self.policy_name + '_critic' + '.pth',
        )

        self.actor = ppo._02_network.create_actor_nn(
            self.state_dim,
            self.dim_action,
            self.hidden_dim,
            self.hidden_layer_count,
        ).to(self.device)
        self.critic = ppo._02_network.create_critic_nn(
            self.state_dim,
            self.hidden_dim,
            self.hidden_layer_count,
        ).to(self.device)
        self.actor_old = ppo._02_network.create_actor_nn(
            self.state_dim,
            self.dim_action,
            self.hidden_dim,
            self.hidden_layer_count,
        ).to(self.device)

        if self.conf.train_rewrite is not True:
            actor_exists = os.path.exists(self.actor_path)
            if actor_exists:
                self.actor.load_state_dict(torch.load(
                    self.actor_path,
                    map_location=self.device,
                    weights_only=True,
                ))
            else:
                mode_name = str(getattr(self.conf, "mode", "")).strip().lower()
                if mode_name == "play":
                    raise FileNotFoundError(
                        f"PPO policy checkpoint not found: {self.actor_path}"
                    )
            if os.path.exists(self.critic_path):
                self.critic.load_state_dict(torch.load(
                    self.critic_path,
                    map_location=self.device,
                    weights_only=True,
                ))

        self.actor_old.load_state_dict(self.actor.state_dict())

        self.optimizer_for_actor = optim.Adam(
            self.actor.parameters(), lr=self.learning_rate_actor)
        self.optimizer_for_critic = optim.Adam(
            self.critic.parameters(), lr=self.learning_rate_critic)

    def get_transition(self, env, state_mat):
        """====================================================================================================
        ## Get Transition by Algorithm
        ===================================================================================================="""
        state = self.map_to_designed_state(state_mat)

        # - Collect rollouts with old policy
        action_mat, action_idx, log_prob_old = \
            ppo._06_algorithm.stochastic_action_selection(
                policy=self.actor_old,
                state=state,
            )
        action = self.map_to_designed_action(action_mat)

        score, state_next_mat, reward_next_mat, done = env.run(
            player=self.conf.train_side, run_type='ai', action=action)

        state_next = self.map_to_designed_state(state_next_mat)
        reward_next = self.map_to_designed_reward(reward_next_mat)

        transition = (
            state,
            action_idx,
            log_prob_old,
            state_next,
            reward_next,
            done,
            score,
        )
        return transition, state_next_mat

    def update(self, transition):
        """================================================================================================
        ## Accumulate Rollout and Update PPO in Batch
        ================================================================================================"""
        state, action_idx, log_prob_old, state_next, reward, done, _ = transition

        self.rollout_states.append(state)
        self.rollout_states_next.append(state_next)
        self.rollout_action_indices.append(action_idx)
        self.rollout_log_probs_old.append(float(log_prob_old))
        self.rollout_rewards.append(float(reward))
        self.rollout_dones.append(float(done))

        if done:
            self.update_rollout()

    def update_rollout(self):
        """================================================================================================
        ## PPO Update from Collected Rollout
        ================================================================================================"""
        if len(self.rollout_states) == 0:
            return

        states = torch.as_tensor(
            self.rollout_states, dtype=torch.float32, device=self.device)
        states_next = torch.as_tensor(
            self.rollout_states_next, dtype=torch.float32, device=self.device)
        action_indices = torch.as_tensor(
            self.rollout_action_indices, dtype=torch.long, device=self.device)
        log_probs_old = torch.as_tensor(
            self.rollout_log_probs_old, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(
            self.rollout_rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(
            self.rollout_dones, dtype=torch.float32, device=self.device)

        # - Build GAE advantages and value targets.
        with torch.no_grad():
            values_old = self.critic(states).squeeze(-1)
            values_next = self.critic(states_next).squeeze(-1)
            deltas = rewards + self.gamma * values_next * (1.0 - dones) - values_old

            advantages = torch.empty_like(rewards)
            gae = torch.zeros((), dtype=torch.float32, device=self.device)
            for step_idx in range(rewards.shape[0] - 1, -1, -1):
                gae = (
                    deltas[step_idx]
                    + self.gamma
                    * self.gae_lambda
                    * (1.0 - dones[step_idx])
                    * gae
                )
                advantages[step_idx] = gae
            returns = advantages + values_old
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + self.advantage_epsilon
                )

        sample_count = int(states.shape[0])
        batch_size = sample_count
        if self.minibatch_size > 0:
            batch_size = min(self.minibatch_size, sample_count)

        epochs_ran = 0
        update_approx_kl = None
        for _ in range(self.update_epochs):
            epochs_ran += 1
            epoch_approx_kl_sum = 0.0
            epoch_approx_kl_count = 0
            permutation = torch.randperm(sample_count, device=self.device)
            for start_idx in range(0, sample_count, batch_size):
                batch_indices = permutation[start_idx:start_idx + batch_size]

                batch_states = states[batch_indices]
                batch_action_indices = action_indices[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                logits = self.actor(batch_states)
                log_probs = torch.log_softmax(logits, dim=-1)
                action_probs = torch.softmax(logits, dim=-1)
                selected_log_probs = log_probs.gather(
                    1, batch_action_indices.unsqueeze(1)).squeeze(1)
                log_ratio = selected_log_probs - batch_log_probs_old
                ratios = torch.exp(log_ratio)
                with torch.no_grad():
                    approx_kl = ((ratios - 1.0) - log_ratio).mean()
                    epoch_approx_kl_sum += float(approx_kl.item())
                    epoch_approx_kl_count += 1

                # - PPO actor loss:
                #   min(ratio * advantage, clip(ratio) * advantage)
                clipped_ratios = torch.clamp(
                    ratios,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                )
                surrogate_unclipped = ratios * batch_advantages
                surrogate_clipped = clipped_ratios * batch_advantages
                entropy = -(action_probs * log_probs).sum(dim=-1).mean()
                loss_actor = -torch.minimum(
                    surrogate_unclipped,
                    surrogate_clipped,
                ).mean() - self.entropy_coefficient * entropy
                self.optimizer_for_actor.zero_grad(set_to_none=True)
                loss_actor.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer_for_actor.step()

                # - Critic fits the return target.
                values = self.critic(batch_states).squeeze(-1)
                if self.value_clip_epsilon > 0.0:
                    batch_values_old = values_old[batch_indices]
                    values_clipped = batch_values_old + torch.clamp(
                        values - batch_values_old,
                        -self.value_clip_epsilon,
                        self.value_clip_epsilon,
                    )
                    value_loss_unclipped = (values - batch_returns).pow(2)
                    value_loss_clipped = (values_clipped - batch_returns).pow(2)
                    value_loss = torch.maximum(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = self.loss_function(values, batch_returns)
                loss_critic = self.vf_coefficient * value_loss
                self.optimizer_for_critic.zero_grad(set_to_none=True)
                loss_critic.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_for_critic.step()

            if epoch_approx_kl_count > 0:
                update_approx_kl = epoch_approx_kl_sum / float(epoch_approx_kl_count)
            if (
                self.target_kl > 0.0
                and update_approx_kl is not None
                and update_approx_kl > self.target_kl
            ):
                break

        self.actor_old.load_state_dict(self.actor.state_dict())
        self.last_approx_kl = update_approx_kl
        self.last_update_epochs = int(epochs_ran)
        self.rollout_states = []
        self.rollout_states_next = []
        self.rollout_action_indices = []
        self.rollout_log_probs_old = []
        self.rollout_rewards = []
        self.rollout_dones = []

    def get_train_configuration(self):
        return ppo._01_params.get_train_params()

    def map_to_designed_state(self, state_mat):
        state_custom = ppo._03_state_design.calculate_state_key(state_mat)
        return tuple(state_custom)

    def map_to_designed_action(self, action_mat):
        action_custom = action_mat * ppo._04_action_space_design.action_mask()
        return action_custom

    def map_to_designed_reward(self, reward_mat):
        reward_custom = ppo._05_reward_design.calculate_reward(reward_mat)
        return reward_custom

    def select_action(self, state_mat, epsilon=0.0, deterministic=False):
        del epsilon
        state = self.map_to_designed_state(state_mat)
        if deterministic:
            action_mat, _, _ = ppo._06_algorithm.deterministic_action_selection(
                policy=self.actor,
                state=state,
            )
        else:
            action_mat, _, _ = ppo._06_algorithm.stochastic_action_selection(
                policy=self.actor,
                state=state,
            )
        action = self.map_to_designed_action(action_mat)
        return action

    def set_training_progress(self, episodes_completed):
        self.training_episode = max(0, int(episodes_completed))
        if self.entropy_decay_episodes <= 0:
            self.entropy_coefficient = float(self.entropy_coefficient_start)
            return

        progress = min(1.0, float(self.training_episode) / float(self.entropy_decay_episodes))
        self.entropy_coefficient = (
            float(self.entropy_coefficient_start)
            + (float(self.entropy_coefficient_end) - float(self.entropy_coefficient_start)) * progress
        )

    def save(self):
        self.policy_path = self.actor_path
        ppo._02_network.save_nn(self.actor, self.actor_path)
        ppo._02_network.save_nn(self.critic, self.critic_path)
