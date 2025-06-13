import os
import threading
import torch
import torch.optim as optim

from models import ActorCritic
from worker import worker
from config import CURRENCY_CONFIGS, TradingConfig
from simulated_env import SimulatedOandaForexEnv


def evaluate_model(model, currency_config, episodes: int = 3, steps: int = 50):
    """Run a short evaluation in a simulated environment and print
    detailed metrics about the model's quality."""

    env = SimulatedOandaForexEnv(
        currency_config,
        candle_count=TradingConfig.CANDLE_COUNT,
        granularity=TradingConfig.GRANULARITY,
    )

    model.eval()
    total_reward = 0.0
    total_trades = 0
    profits = []
    action_counts = [0, 0, 0]  # long, short, neutral

    for _ in range(episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
        decisions = torch.zeros((1, 16), dtype=torch.float32)
        episode_reward = 0.0

        for _ in range(steps):
            with torch.no_grad():
                logits, _ = model(state, decisions)
                probs = torch.softmax(logits, dim=1)
                action = torch.multinomial(probs, num_samples=1).item()

            action_counts[action] += 1
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            if done or next_state is None:
                break

            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        total_reward += episode_reward
        profits.extend([t.profit for t in env.trade_log])
        total_trades += len(env.trade_log)

    avg_reward = total_reward / episodes
    if profits:
        avg_profit = sum(profits) / len(profits)
        win_rate = sum(p > 0 for p in profits) / len(profits)
    else:
        avg_profit = 0.0
        win_rate = 0.0

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

    return avg_reward, avg_profit, win_rate

# Directory to save models per currency.
MODEL_DIR = "./models/"

def main():
    # Set training parameters for the weekend training script.
    num_workers = 200       # Use 100 worker threads.
    train_steps = 121      # Each worker runs 121 training steps.

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Initialize (or load) models for each currency.
    models = {}
    for currency, currency_config in CURRENCY_CONFIGS.items():
        model_path = os.path.join(MODEL_DIR, f"{currency}.pt")
        currency_model = ActorCritic()
        if os.path.exists(model_path):
            currency_model.load_state_dict(torch.load(model_path))
            print(f"Loaded existing model for {currency}.")
        else:
            print(f"Initializing new model for {currency}.")
        currency_model.share_memory()
        models[currency] = currency_model

    # Run the training cycle for each currency once.
    for currency, currency_config in CURRENCY_CONFIGS.items():
        print(f"\n--- Training cycle for {currency} ---")
        model = models[currency]
        optimizer = optim.Adam(model.parameters(), lr=0.00004)
        optimizer_lock = threading.Lock()
        barrier = threading.Barrier(num_workers + 1)

        workers = []
        for i in range(num_workers):
            t = threading.Thread(
                target=worker,
                args=(i, model, optimizer, optimizer_lock, train_steps, currency_config, barrier),
                daemon=True
            )
            workers.append(t)
            t.start()

        # Wait for all workers to complete their training steps.
        print(f"\n--- waiting for workers for {currency} ---")
        barrier.wait()
        for t in workers:
            t.join()

        # Save the updated model.
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{currency}.pt"))
        print(f"--- Finished training cycle for {currency} ---")

        # Evaluate the trained model on a simulated environment
        evaluate_model(model, currency_config)

if __name__ == "__main__":
    main()
