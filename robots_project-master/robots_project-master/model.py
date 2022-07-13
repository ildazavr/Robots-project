import torch
import torch.nn as nn


class BigModel(nn.Module):
    def __init__(self, action_size, n_actors):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 2, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2, 2, 1),
        )

        self.policy_heads = []
        for i in range(n_actors):
            head = nn.Sequential(
                nn.Linear(32 * 8 * 8, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 4)
            )

            self.policy_heads.append(head)

        self.head_value = nn.Linear(32 * 8 * 8, 1)

    def return_policy(self, emb, i):
        logits = self.policy_heads[i](emb).reshape(emb.size(0), 4)

        pi_dist = torch.distributions.Categorical(logits=logits)
        action = pi_dist.sample()

        return action, pi_dist

    def forward(self, obs, i):
        emb = self.cnn(obs).reshape(obs.size(0), -1)

        action, pi_dist = self.return_policy(emb, i)
        value = self.head_value(emb)

        return action, pi_dist, value
