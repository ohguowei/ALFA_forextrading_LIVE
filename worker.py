import torch
import numpy as np
import traceback
from collections import deque

from preprocessing import encode_decision_history

from simulated_env import SimulatedOandaForexEnv
from models import ActorCritic
from config import TradingConfig


def worker(
    worker_id: int,
    global_model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    max_steps: int = 20,
    currency_config=None,
    barrier=None,
    gamma: float = 0.99,
    accumulate_returns: bool = False,
    action_counts=None,
    action_lock=None,
    model_lock=None,
    entropy_weight: float = 0.01,
):
    """Run a training loop on a simulated environment.

    Parameters
    ----------
    worker_id : int
        Identifier for the worker thread.
    global_model : ActorCritic
        Shared model updated by all workers.
    optimizer : torch.optim.Optimizer
        Optimizer used to update ``global_model``.
    max_steps : int, optional
        Number of steps to run in this training cycle.
    currency_config : CurrencyConfig
        Configuration object for the currency pair.
    barrier : threading.Barrier
        Barrier used to synchronize workers at the end.
    gamma : float, optional
        Discount factor for returns.
    accumulate_returns : bool, optional
        If ``True``, use accumulated returns.
    action_counts : list of int, optional
        Shared list counting how often each action is taken.
    action_lock : threading.Lock, optional
        Lock guarding ``action_counts`` updates.
    model_lock : threading.Lock, optional
        Lock guarding updates to ``global_model``.
    entropy_weight : float, optional
        Weight for policy entropy regularization. Defaults to ``0.01``.
    """
    if currency_config is None:
        raise ValueError("Currency config is required for worker")
    if barrier is None:
        raise ValueError("Barrier is required for synchronization")
    
    # Instantiate the simulated environment with the provided currency
    # configuration.
    env = SimulatedOandaForexEnv(
        currency_config,
        candle_count=TradingConfig.CANDLE_COUNT,
        granularity=TradingConfig.GRANULARITY
    )
    
    # Get the initial state and initialize decision history.
    state = env.reset()  # Expected shape: (time_window, features)
    decision_history = deque([2] * 16, maxlen=16)
    decisions_t = encode_decision_history(decision_history)
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    step_count = 0
    returns = torch.tensor([[0.0]], dtype=torch.float32)
    while step_count < max_steps:
        try:
            decisions_t = encode_decision_history(decision_history)
            with model_lock:
                policy_logits, value = global_model(state_t, decisions_t)
                probs = torch.softmax(policy_logits, dim=1)
                action = torch.multinomial(probs, num_samples=1)
                action_idx = action.item()

                if action_counts is not None:
                    if action_lock is not None:
                        with action_lock:
                            action_counts[action_idx] += 1
                    else:
                        action_counts[action_idx] += 1

                next_state, reward, done, _ = env.step(action_idx)
                decision_history.append(action_idx)
                decisions_next = encode_decision_history(decision_history)

                reward_t = torch.tensor([[reward]], dtype=torch.float32)

                if done or next_state is None:
                    next_value = torch.tensor([[0.0]], dtype=torch.float32)
                else:
                    next_state_t = torch.tensor(
                        next_state, dtype=torch.float32
                    ).unsqueeze(0)
                    with torch.no_grad():
                        _, next_value = global_model(
                            next_state_t, decisions_next
                        )

                if accumulate_returns:
                    returns[:] = reward_t + gamma * returns
                    advantage = returns - value
                else:
                    advantage = reward_t + gamma * next_value - value

                log_probs = torch.log(probs + 1e-8)
                policy_loss = -log_probs[0, action_idx] * advantage.detach()
                value_loss = advantage.pow(2)
                entropy = -(probs * log_probs).sum()
                loss = policy_loss + value_loss - entropy_weight * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update the state and history.
            if done or next_state is None:
                state = env.reset()
                decision_history = deque([2] * 16, maxlen=16)
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
    
    print(
        f"[Worker {worker_id}] Finished training cycle at step {step_count}."
    )

    if env.position_open:
        profit = env.simulated_close_position()
        if profit is not None:
            final_reward = float(np.clip(profit, -1.0, 1.0))
            decisions_t = encode_decision_history(decision_history)
            reward_t = torch.tensor([[final_reward]], dtype=torch.float32)
            with model_lock:
                _, value = global_model(state_t, decisions_t)
                if accumulate_returns:
                    returns[:] = reward_t + gamma * returns
                    advantage = returns - value
                else:
                    advantage = reward_t - value
                value_loss = advantage.pow(2)
                optimizer.zero_grad()
                value_loss.backward()
                optimizer.step()
    
    # Wait at the barrier to signal completion to the main thread.
    barrier.wait()
