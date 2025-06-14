import torch
from typing import Tuple
from collections import deque

from models import ActorCritic
from simulated_env import SimulatedOandaForexEnv
from config import TradingConfig


def evaluate_model(
    model: ActorCritic,
    currency_config,
    episodes: int = 3,
    candle_count: int = 5000,
) -> Tuple[float, float, float]:
    """Evaluate a model in a simulated environment.

    Runs ``episodes`` episodes using ``candle_count`` candles. Actions are chosen
    greedily from the policy. A detailed report including profit factor and
    maximum drawdown is printed.

    Returns average reward, average profit and win rate.
    """
    env = SimulatedOandaForexEnv(
        currency_config,
        candle_count=candle_count,
        granularity=TradingConfig.GRANULARITY,
    )

    model.eval()
    total_reward = 0.0
    total_trades = 0
    profits = []
    action_counts = [0, 0, 0]

    for _ in range(episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
        decision_history = deque([0] * 16, maxlen=16)
        decisions = torch.tensor(decision_history, dtype=torch.float32).unsqueeze(0)
        episode_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                logits, _ = model(state, decisions)
                probs = torch.softmax(logits, dim=1)
                action = torch.argmax(probs, dim=1).item()

            action_counts[action] += 1
            next_state, reward, done, _ = env.step(action)
            decision_history.append(action)
            decisions = torch.tensor(decision_history, dtype=torch.float32).unsqueeze(0)
            episode_reward += reward

            if done or next_state is None:
                break

            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        if env.position_open:
            env.simulated_close_position()

        total_reward += episode_reward
        profits.extend(t.profit for t in env.trade_log)
        total_trades += len(env.trade_log)

    avg_reward = total_reward / episodes
    if profits:
        avg_profit = sum(profits) / len(profits)
        win_rate = sum(p > 0 for p in profits) / len(profits)
        positive = sum(p for p in profits if p > 0)
        negative = abs(sum(p for p in profits if p < 0))
        profit_factor = positive / negative if negative else float("inf")
        cumulative = torch.tensor(profits).cumsum(0)
        max_cum = torch.maximum(cumulative, cumulative.clone().cummax(0)[0])
        drawdown = (max_cum - cumulative).max().item()
    else:
        avg_profit = 0.0
        win_rate = 0.0
        profit_factor = 0.0
        drawdown = 0.0

    total_actions = sum(action_counts)
    if total_actions > 0:
        action_dist = [c / total_actions for c in action_counts]
    else:
        action_dist = [0.0, 0.0, 0.0]

    print(f"Evaluation result for {currency_config.instrument}:")
    print(f"  Avg reward per episode: {avg_reward:.4f}")
    print(
        f"  Trades: {total_trades}, Win rate: {win_rate*100:.1f}%, "
        f"Avg profit: {avg_profit:.4f}"
    )
    print(
        "  Action distribution: "
        f"long {action_dist[0]*100:.1f}%, "
        f"short {action_dist[1]*100:.1f}%, "
        f"neutral {action_dist[2]*100:.1f}%"
    )
    print(f"  Profit factor: {profit_factor:.2f}")
    print(f"  Max drawdown: {drawdown:.4f}")

    return avg_reward, avg_profit, win_rate


def feature_importance(model: ActorCritic, currency_config, episodes: int = 3):
    """Return normalized input feature importances for a trained model.

    Runs ``episodes`` episodes in a :class:`SimulatedOandaForexEnv` and
    greedily selects actions from the policy. For each step the gradient of the
    selected action's logit with respect to the input state is computed. The
    absolute gradients are accumulated and averaged across all steps and
    episodes. The resulting importances for the seven input features are
    normalized to sum to one.
    """

    env = SimulatedOandaForexEnv(
        currency_config,
        candle_count=5000,
        granularity=TradingConfig.GRANULARITY,
    )

    model.eval()
    importance = torch.zeros(7)
    step_count = 0

    for _ in range(episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
        decision_history = deque([0] * 16, maxlen=16)
        decisions = torch.tensor(decision_history, dtype=torch.float32).unsqueeze(0)
        done = False

        while not done:
            state.requires_grad_(True)
            logits, _ = model(state, decisions)
            action = torch.argmax(logits, dim=1)
            logit = logits[0, action]

            model.zero_grad()
            if state.grad is not None:
                state.grad.zero_()
            logit.backward()

            grad = state.grad.detach().abs().sum(dim=1).squeeze(0)
            importance += grad
            step_count += 1

            next_state, _, done, _ = env.step(action.item())
            decision_history.append(action.item())
            decisions = torch.tensor(decision_history, dtype=torch.float32).unsqueeze(0)
            if done or next_state is None:
                break
            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        if env.position_open:
            env.simulated_close_position()

    if step_count > 0:
        importance /= step_count
        importance /= importance.sum() if importance.sum() != 0 else 1.0

    print(
        "Feature importance:",
        ", ".join(f"f{i+1}:{imp:.3f}" for i, imp in enumerate(importance)),
    )

    return importance


def models_are_equal(path_a: str, path_b: str) -> bool:
    """Return True if two saved ActorCritic models have identical parameters."""
    model_a = ActorCritic()
    model_b = ActorCritic()
    model_a.load_state_dict(torch.load(path_a))
    model_b.load_state_dict(torch.load(path_b))
    for p1, p2 in zip(model_a.parameters(), model_b.parameters()):
        if not torch.equal(p1, p2):
            return False
    return True
