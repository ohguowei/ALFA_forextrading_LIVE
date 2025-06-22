# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """Combined actor-critic network."""

    def __init__(
        self,
        input_size: int = 8,
        decision_history_len: int = 16,
        hidden_size: int = 128,
        num_actions: int = 3,
        num_lstm_layers: int = 4,
    ) -> None:
        """Initialize the network."""
        super().__init__()
        self.decision_history_len = decision_history_len
        self.num_actions = num_actions
        decision_dim = decision_history_len * num_actions
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
        
    def forward(self, state: torch.Tensor, decisions: torch.Tensor):
        """Return policy logits and state-value estimate."""
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


def load_checkpoint(model: nn.Module, path: str) -> None:
    """Load a checkpoint into ``model`` handling minor shape mismatches."""
    state = torch.load(path)
    model_state = model.state_dict()
    patched_state = {}
    for name, param in state.items():
        if name not in model_state:
            continue
        if param.shape == model_state[name].shape:
            patched_state[name] = param
        elif (
            name.endswith("weight_ih_l0")
            and param.shape[0] == model_state[name].shape[0]
            and param.shape[1] + 1 == model_state[name].shape[1]
        ):
            new_param = torch.zeros_like(model_state[name])
            new_param[:, : param.shape[1]] = param
            patched_state[name] = new_param
        else:
            print(
                f"Skipping parameter {name}: {param.shape} vs "
                f"{model_state[name].shape}"
            )
    model_state.update(patched_state)
    model.load_state_dict(model_state)
