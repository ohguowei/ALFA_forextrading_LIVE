# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, input_size=7, decision_dim=16, hidden_size=128, num_actions=3, num_lstm_layers=4):
        super(ActorCritic, self).__init__()
        # Actor branch with a deeper LSTM (2 layers)
        self.actor_lstm = nn.LSTM(input_size, hidden_size, num_layers=num_lstm_layers, batch_first=True)
        self.actor_fc1 = nn.Linear(hidden_size, 32)
        self.actor_fc2 = nn.Linear(32 + decision_dim, 64)
        self.actor_fc3 = nn.Linear(64, 64)
        self.actor_output = nn.Linear(64, num_actions)
        
        # Critic branch with a deeper LSTM (2 layers)
        self.critic_lstm = nn.LSTM(input_size, hidden_size, num_layers=num_lstm_layers, batch_first=True)
        self.critic_fc1 = nn.Linear(hidden_size, 32)
        self.critic_fc2 = nn.Linear(32 + decision_dim, 64)
        self.critic_fc3 = nn.Linear(64, 64)
        self.critic_output = nn.Linear(64, 1)
        
    def forward(self, state, decisions):
        # Actor forward pass
        actor_out, _ = self.actor_lstm(state)
        actor_last = actor_out[:, -1, :]  # Take output of the final time step
        x = F.relu(self.actor_fc1(actor_last))
        x = torch.cat([x, decisions], dim=1)
        x = F.relu(self.actor_fc2(x))
        x = F.relu(self.actor_fc3(x))
        policy_logits = self.actor_output(x)
        
        # Critic forward pass
        critic_out, _ = self.critic_lstm(state)
        critic_last = critic_out[:, -1, :]
        y = F.relu(self.critic_fc1(critic_last))
        y = torch.cat([y, decisions], dim=1)
        y = F.relu(self.critic_fc2(y))
        y = F.relu(self.critic_fc3(y))
        value = self.critic_output(y)
        
        return policy_logits, value
