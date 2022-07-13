import torch
import loss
import numpy as np

from model import BigModel


class PPO:
    def __init__(self):
        """
        Class for building agents or mediator

        :param input_size: iterable(n_actor, n_critic), number of input features to NNs
        :param agent_nn: iterable, Actor and Critic NN
        :param cfg_agent: config file for agents/mediator
        :param cfg_env: config file for environment
        :param agent_i: int, "-1" if mediator, positive integers otherwise
        """
        optimizer = torch.optim.Adam
        self.lr = 5e-4
        self.gamma = 0.99
        self.entropy_coef = 1
        self.dtype = torch.float32

        self.entropy_coef_decrease = 0.001

        self.device = 'cpu'
        self.model = BigModel(4, 4).to(self.device)
        self.actor = lambda x, i: self.model(x, i)[:2]
        self.critic = lambda x: self.model(x, 0)[-1]

        self.opt = optimizer(self.model.parameters(), lr=self.lr)

        self.rets = [[], [], [], []]
        self.adv = [[], [], [], []]
        self.log_old = [[], [], [], []]

    def _tensorify(self, list_of_non_tensors):
        tensors = []
        for item in list_of_non_tensors:
            converted = torch.tensor(item, device=self.device, dtype=self.dtype)

            if converted.dim() == 0:
                converted = converted.reshape(-1, 1)
            elif converted.dim() == 1:
                converted = converted.unsqueeze(0)

            tensors.append(converted)

        return tensors

    def _calc_advantage(self, obs, next_obs, reward, done):
        """
        Calculating advantage

        :param obs: iterable, of everything connected to observation (i.e. in_coalition)
        :param next_obs: iterable, same as previous converning next_state
        :param reward: real value
        :param done: Boolean
        :return: advantage
        """
        advantage = reward + (1 - done) * self.gamma * self.critic(next_obs) - self.critic(obs)
        assert advantage.dim() == 2, f'{advantage.shape=}'

        return advantage

    def get_policy(self, obs, i):
        _, pi_dist = self.actor(obs, i)

        return pi_dist

    def step(self, obs, i):
        action, _ = self.actor(obs, i)

        return action.squeeze(0).cpu().detach().numpy()

    def compute_ppo_stats(self, obs, action, rewards, next_obs, done, i):
        with torch.no_grad():
            self.rets[i] = rewards + (1 - done) * self.gamma * self.critic(next_obs)
            self.adv[i] = self.rets[i] - self.critic(obs)

            policy = self.get_policy(obs, i)
            self.log_old[i] = policy.log_prob(action[:, i]).unsqueeze(-1)

        self.entropy_coef = np.maximum(0.01, self.entropy_coef - self.entropy_coef_decrease)

    def update_ppo(self, obs, action, under, idx, i):
        """
        rets, adv, log_old -- отсоединены от графа вычислений!
        """
        # rets, adv_old, rho = self.ppo_stats(policy)
        rets, adv, log_old = self.rets[i][idx].detach(), self.adv[i][idx].detach(), self.log_old[i][idx].detach()

        adv = loss.advantage(adv)

        # agent_critic_loss = mse(self.critic([m_obs, other_obs]), rets)
        agent_critic_loss = loss.valueLoss(self.critic(obs), rets)

        policy = self.get_policy(obs, i)

        log_prob = policy.log_prob(action[:, i]).unsqueeze(-1)
        #####
        # log_old[under == 1.] = 0.
        #####
        rho = torch.exp(log_prob - log_old)

        entropy = policy.entropy().mean()

        # agent_actor_loss = adv * rho - self.entropy_coef * entropy
        agent_actor_loss = loss.ppo_loss(adv, rho) - self.entropy_coef * entropy

        self.opt.zero_grad()
        agent_actor_loss.backward()
        agent_critic_loss.backward()
        self.opt.step()

    def critic_loss(self, state, next_state, reward, done):
        baseline = self._calc_advantage(state, next_state, reward, done)

        assert baseline.shape == (reward.shape[0], 1)
        critic_loss = baseline.pow(2).mean()

        return critic_loss, baseline
