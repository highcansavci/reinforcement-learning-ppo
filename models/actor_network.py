import torch
import torch.nn as nn
import os


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=512, fc2_dims=256, chkpt_dir="tmp/ppo"):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, "actor_torch_ppo")
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = torch.distributions.Categorical(dist)
        return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
