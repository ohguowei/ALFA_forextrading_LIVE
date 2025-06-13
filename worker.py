import torch
import torch.nn.functional as F
import numpy as np
import traceback
from threading import Lock

from simulated_env import SimulatedOandaForexEnv
from models import ActorCritic
from config import TradingConfig


def worker(worker_id: int,
           global_model: ActorCritic,
           optimizer: torch.optim.Optimizer,
           optimizer_lock: Lock,
           max_steps: int = 20,
           currency_config=None,
           barrier=None,
           gamma: float = 0.99,
           accumulate_returns: bool = False):
    if currency_config is None:
        raise ValueError("Currency config is required for worker")
    if barrier is None:
        raise ValueError("Barrier is required for synchronization")
    
    # Instantiate the simulated environment using the provided currency configuration.
    env = SimulatedOandaForexEnv(
        currency_config,
        candle_count=TradingConfig.CANDLE_COUNT,
        granularity=TradingConfig.GRANULARITY
    )
    
    # Get the initial state and create an initial decisions vector.
    state = env.reset()  # Expected shape: (time_window, features)
    decisions = np.zeros((1, 16), dtype=np.float32)
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    decisions_t = torch.tensor(decisions, dtype=torch.float32)
    
    step_count = 0
    returns = torch.tensor([[0.0]], dtype=torch.float32)
    while step_count < max_steps:
        try:
            # Lock the entire training iteration to avoid concurrent modifications.
            with optimizer_lock:
                # Forward pass and action selection.
                policy_logits, value = global_model(state_t, decisions_t)
                probs = torch.softmax(policy_logits, dim=1)
                action = torch.multinomial(probs, num_samples=1)
                action_idx = action.item()
                
                # Perform the environment step while holding the lock.
                # (This may slow training but prevents the in-place modification error.)
                next_state, reward, done, _ = env.step(action_idx)

                reward_t = torch.tensor([[reward]], dtype=torch.float32)

                # Estimate value of next state for bootstrapping
                if done or next_state is None:
                    next_value = torch.tensor([[0.0]], dtype=torch.float32)
                else:
                    next_state_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        _, next_value = global_model(next_state_t, decisions_t)

                if accumulate_returns:
                    returns[:] = reward_t + gamma * returns
                    advantage = returns - value
                else:
                    advantage = reward_t + gamma * next_value - value

                policy_loss = -torch.log(probs[0, action_idx] + 1e-8) * advantage.detach()
                value_loss = advantage.pow(2)
                loss = policy_loss + value_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Release the lock and update the state.
            if done or next_state is None:
                state = env.reset()
                returns[:] = 0
            else:
                state = next_state
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            step_count += 1
        
        except Exception as e:
            print(f"[Worker {worker_id}] Exception at step {step_count}: {e}")
            traceback.print_exc()
            break
    
        # Optionally, print progress or debugging info here.
    
    print(f"[Worker {worker_id}] Finished training cycle at step {step_count}.")
    
    # Wait at the barrier to signal completion to the main thread.
    barrier.wait()
