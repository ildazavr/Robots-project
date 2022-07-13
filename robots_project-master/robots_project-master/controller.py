import torch
import numpy as np
from ppo import PPO
import wandb
import time
from utils import make_gif


class Controller:
    def __init__(self):
        self.model = PPO()
        self.device = self.model.device
        # self.mediator = nn_mediator

        # self.device = cfg.device

        self.dtype = torch.float32
        self.batch_size = 64
        self.ppo_epochs = 5

    def _tensorify(self, inp):
        ret = torch.tensor(inp, device=self.device, dtype=self.dtype).unsqueeze(0).permute(0, 3, 1, 2)

        return ret

    def sample_episode(self, env, test=False):
        obs = env.reset()
        done = 0
        trajectory = []

        # For evaluation only
        pick_mediator = []
        all_rewards = []
        next_obs_rgb_glob = []

        full_obs = []
        joint_obs = []

        while not done:
            # Agents' moves
            obs_ag = self._tensorify(obs)

            act_to_env = []
            with torch.no_grad():
                for i in range(4):
                    act = self.model.step(obs_ag, i)
                    act_to_env.append(act)

            next_obs, rewards, done = env.step(act_to_env)

            trajectory.append((obs, act_to_env, rewards, next_obs, [done]))
            obs = next_obs

            all_rewards.append(rewards)
            full_obs.append(env.get_full_obs())
            joint_obs.append(env.get_joint_obs())
            # pick_mediator.append(np.sum(coalition) / 2)

        if test:
            return full_obs, joint_obs

        return trajectory, all_rewards

    def _get_batch(self, trajectories):
        transitions = [t for traj in trajectories for t in traj]
        idx = np.random.randint(0, len(transitions), self.batch_size * self.ppo_epochs)
        transitions = [transitions[i] for i in idx]
        # batch = map(lambda x: torch.tensor(x, device=self.device, dtype=self.dtype), zip(*transitions))
        f = lambda x: torch.stack(x, dim=0) if torch.is_tensor(x[0]) else np.stack(x, axis=0)
        batch = list(map(f, zip(*transitions)))

        obs, act, rewards, next_obs, done = batch

        obs = torch.tensor(obs, device=self.device, dtype=self.dtype).permute(0, 3, 1, 2)
        act = torch.tensor(act, device=self.device, dtype=self.dtype)
        rewards = torch.tensor(rewards, device=self.device, dtype=self.dtype).reshape(-1, 1)
        next_obs = torch.tensor(next_obs, device=self.device, dtype=self.dtype).permute(0, 3, 1, 2)
        done = torch.tensor(done, device=self.device, dtype=self.dtype)

        return [obs, act, rewards, next_obs, done]

    def update(self, trajectories):
        ts = time.time()
        batch = self._get_batch(trajectories)

        obs, act, rewards, next_obs, done = batch
        print('got batch')

        for i in range(4):
            self.model.compute_ppo_stats(obs, act, rewards, next_obs, done, i)
        print('calculated statistics')

        for _ in range(self.ppo_epochs):
            idx = np.random.randint(0, self.batch_size * self.ppo_epochs, self.batch_size)
            for i in range(4):
                self.model.update_ppo(obs[idx], act[idx], [0] * 16, idx, i)

        print(f'update is done in {time.time() - ts:.2f} sec')

    def train(self, env):
        # get batch of trajectories
        for i in range(10_000):
            trajectories = []
            rewards = []
            steps_ctn = 0

            ts = time.time()
            while steps_ctn < 1_000:
                traj, rew = self.sample_episode(env)
                steps_ctn += len(traj)
                trajectories.append(traj)
                rewards.append(rew)

            print(f'trajectories collected in {time.time() - ts:.2f}')
            # update
            self.update(trajectories)

            rewards = np.vstack(rewards).sum(0)

            wandb.log(data={
                'general_reward_mean': np.mean(rewards),
                'general_reward_std': np.std(rewards),
                'sum_reward': np.sum(rewards),
                'min_reward': np.min(rewards),
                'max_reward': np.max(rewards),
            }, step=i)

            if i % 100 == 0:
                full, joint = self.sample_episode(env, test=True)
                make_gif(full, joint)
                wandb.log(data={
                    'video': wandb.Video('test_anim.gif')
                }, step=i)
