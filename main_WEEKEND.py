import os
import threading
import torch
import torch.optim as optim

from models import ActorCritic
from worker import worker
from config import CURRENCY_CONFIGS

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

if __name__ == "__main__":
    main()
