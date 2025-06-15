import os
import argparse
import datetime
import time
import threading
import multiprocessing
import torch
import torch.optim as optim
from collections import deque

from preprocessing import encode_decision_history

import tg_bot  # Contains Telegram bot logic and global "last_trade_status"
from models import ActorCritic, load_checkpoint
from worker import worker
from live_env import LiveOandaForexEnv
from config import TradingConfig, CURRENCY_CONFIGS, set_global_seed
from evaluation import evaluate_model, models_are_equal


def summarize_trade_log(trades):
    """Print a simple quality summary for a list of trades."""
    if not trades:
        print("No trades executed during this cycle.")
        return
    profits = [t.profit for t in trades]
    win_rate = sum(p > 0 for p in profits) / len(profits)
    avg_profit = sum(profits) / len(profits)
    print(
        f"Trade summary: {len(trades)} trades, "
        f"win rate {win_rate*100:.1f}%, "
        f"avg profit {avg_profit:.4f}"
    )

MODEL_DIR = "./models/"

def wait_for_trading_window():
    """
    Block until the current local time is within the trading window:
    Monday ≥6 AM to Saturday <6 AM.
    """
    while True:
        now = datetime.datetime.now()
        wd, hr = now.weekday(), now.hour
        if not (wd == 6 or (wd == 0 and hr < 6) or (wd == 5 and hr >= 6)):
            return
        print("Outside trading window. Sleeping 60 seconds...")
        time.sleep(60)

def calculate_next_trigger_time():
    now = datetime.datetime.now()
    next_minute = now.minute + 1
    if next_minute >= 60:
        next_trigger = now.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
    else:
        next_trigger = now.replace(minute=next_minute, second=0, microsecond=0)
    return next_trigger

def wait_until_next_trigger():
    next_trigger = calculate_next_trigger_time()
    now = datetime.datetime.now()
    wait_time = (next_trigger - now).total_seconds()
    if wait_time > 0:
        print(f"Waiting {wait_time:.0f} seconds until {next_trigger.strftime('%H:%M:%S')}...")
        time.sleep(wait_time)
    else:
        print("Next trigger time is in the past. Triggering immediately.")
    return next_trigger

def trade_live(currency_model, live_env, num_steps=10):
    """
    Runs a live trading cycle.
    Updates the Telegram bot's global last_trade_status variable with the most recent trade.
    """
    currency_model.eval()
    state = live_env.reset()
    decision_history = deque([2] * 16, maxlen=16)
    decisions = encode_decision_history(decision_history)
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    for step in range(num_steps):
        with torch.no_grad():
            policy_logits, _ = currency_model(state, decisions)
            probs = torch.softmax(policy_logits, dim=1)
            action = torch.multinomial(probs, num_samples=1).item()
        print(f"[Trading] Step {step}, Action: {action}")
        next_state, reward, done, _ = live_env.step(action)
        decision_history.append(action)
        decisions = encode_decision_history(decision_history)
        print(f"[Trading] Reward: {reward}")
        if not done and next_state is not None:
            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        else:
            break
    print("[Trading] Finished trading cycle.")
    summarize_trade_log(live_env.trade_log)
    if live_env.trade_log:
        last_trade = live_env.trade_log[-1]
        tg_bot.last_trade_status = str(last_trade)
        print("Updated last trade status:", tg_bot.last_trade_status)
    else:
        tg_bot.last_trade_status = "No trades executed."

def run_training_cycle(models, num_workers, train_steps, training_label):
    """
    Runs a training cycle over all currencies.
    This function is intended to be run in a separate process.
    """
    for currency, currency_config in CURRENCY_CONFIGS.items():
        print(f"\n--- {training_label} Training cycle for {currency} ---")
        model = models[currency]
        optimizer = optim.Adam(model.parameters(), lr=0.00004)
        barrier = threading.Barrier(num_workers + 1)
        action_counts = [0, 0, 0]
        action_lock = threading.Lock()
        model_lock = threading.Lock()

        workers = []
        for i in range(num_workers):
            t = threading.Thread(
                target=worker,
                args=(
                    i,
                    model,
                    optimizer,
                    train_steps,
                    currency_config,
                    barrier,
                ),
                kwargs={
                    "action_counts": action_counts,
                    "action_lock": action_lock,
                    "model_lock": model_lock,
                    "accumulate_returns": True,
                    "entropy_weight": 0.01,
                },
                daemon=True
            )
            workers.append(t)
            t.start()

        barrier.wait()
        for t in workers:
            t.join()
        total_actions = sum(action_counts)
        if total_actions:
            dist = [c / total_actions for c in action_counts]
            print(
                "Training action distribution: "
                f"long {dist[0]*100:.1f}%, "
                f"short {dist[1]*100:.1f}%, "
                f"neutral {dist[2]*100:.1f}%"
            )
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{currency}.pt"))
        print(f"--- Finished {training_label} Training cycle for {currency} ---")

def training_process(models, num_workers_full, train_steps_full):
    """Launch a full training cycle in a lower priority process."""
    try:
        # Increase niceness by 10 to lower priority (macOS uses os.nice)
        os.nice(10)
        print("Training process niceness increased; lower CPU priority assigned.")
    except Exception as e:
        print("Error setting process niceness:", e)
    run_training_cycle(models, num_workers_full, train_steps_full, "FULL")

def trading_loop(train_steps_full=121):
    """Main trading loop coordinating training and live trading."""
    # Full training (60-minute) settings.
    num_workers_full = 100      # 100 workers

    trade_steps = 1             # Trading steps per trading cycle

    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Initialize (or load) models for each currency.
    models = {}
    for currency, currency_config in CURRENCY_CONFIGS.items():
        model_path = os.path.join(MODEL_DIR, f"{currency}.pt")
        currency_model = ActorCritic()
        if os.path.exists(model_path):
            try:
                load_checkpoint(currency_model, model_path)
                print(f"Loaded existing model for {currency}.")
            except Exception as e:
                print(
                    f"Failed to load model for {currency}: {e}. "
                    "Initializing a new model instead."
                )
        else:
            print(f"Initializing new model for {currency}.")
        currency_model.share_memory()
        models[currency] = currency_model

    while True:
        try:
            # Block until trading is active.
            tg_bot.trading_event.wait()
            wait_for_trading_window()
            next_trigger = wait_until_next_trigger()

            # If it's minute 0, launch full training in a separate process with lowered priority.
            if next_trigger.minute == 0:
                print(f"\n=== Trigger at {next_trigger.strftime('%H:%M:%S')}: Launching FULL TRAINING cycle in a separate process ===")
                p = multiprocessing.Process(
                    target=training_process,
                    args=(models, num_workers_full, train_steps_full),
                    daemon=True
                )
                p.start()

            # Always perform the trading cycle.
            print(f"\n=== Trigger at {next_trigger.strftime('%H:%M:%S')}: Running TRADING cycle ===")
            for currency, currency_config in CURRENCY_CONFIGS.items():
                try:
                    # Reload the latest saved model weights for the currency.
                    model_path = os.path.join(MODEL_DIR, f"{currency}.pt")
                    if os.path.exists(model_path):
                        try:
                            load_checkpoint(models[currency], model_path)
                            print(f"Reloaded latest model for {currency}.")
                        except Exception as e:
                            print(
                                f"Failed to reload model for {currency}: {e}. "
                                "Continuing with in-memory weights."
                            )
                    else:
                        print(
                            f"No saved model file for {currency}; using current in-memory model."
                        )

                    print(f"\n--- Trading cycle for {currency} ---")
                    model = models[currency]
                    model.eval()
                    live_env = LiveOandaForexEnv(
                        currency_config,
                        candle_count=TradingConfig.CANDLE_COUNT,
                        granularity=TradingConfig.GRANULARITY
                    )
                    trade_live(model, live_env, num_steps=trade_steps)
                    print(f"--- Finished trading cycle for {currency} ---")
                except Exception as e:
                    print(f"Error during trading cycle for {currency}: {e}")
            
            print("\nCycle complete. Waiting for the next trigger...\n")
        except Exception as e:
            print(f"Error in trading loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        help="Training steps for each worker during full training",
    )
    args, _ = parser.parse_known_args()
    seed = args.seed
    if seed is None:
        env_seed = os.getenv("SEED")
        if env_seed is not None:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = None
    if seed is not None:
        set_global_seed(seed)

    train_steps_full = args.train_steps
    if train_steps_full is None:
        env_steps = os.getenv("TRAIN_STEPS")
        if env_steps is not None:
            try:
                train_steps_full = int(env_steps)
            except ValueError:
                train_steps_full = None
    if train_steps_full is None:
        train_steps_full = 121

    # Start the trading loop in a background thread.
    trading_thread = threading.Thread(
        target=trading_loop,
        kwargs={"train_steps_full": train_steps_full},
        daemon=True,
    )
    trading_thread.start()

    # Run the Telegram bot in the main thread.
    tg_bot.run_telegram_bot()
