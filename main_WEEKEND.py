import os
import argparse
import threading
import torch
import torch.optim as optim

from tg_bot import send_telegram_message

from models import ActorCritic
from worker import worker
from config import CURRENCY_CONFIGS, set_global_seed
from evaluation import evaluate_model



# Directory to save models per currency.
MODEL_DIR = "./models/"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
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
        barrier = threading.Barrier(num_workers + 1)

        workers = []
        for i in range(num_workers):
            t = threading.Thread(
                target=worker,
                args=(i, model, optimizer, train_steps, currency_config, barrier),
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
    send_telegram_message("Weekend training cycle completed.")
