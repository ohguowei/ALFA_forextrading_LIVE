import os
import threading

import optuna
import torch
import torch.optim as optim

from models import ActorCritic
from worker import worker
from evaluation import evaluate_model
from config import CURRENCY_CONFIGS


MODEL_DIR = "./models/"


def train_once(model: ActorCritic, currency_config, steps: int = 100) -> float:
    """Train and evaluate ``model`` for a single currency.

    Parameters
    ----------
    model : ActorCritic
        The model to train.
    currency_config : CurrencyConfig
        Configuration for the currency pair.
    steps : int, optional
        Training steps to run. Defaults to ``100``.

    Returns
    -------
    float
        Average profit from :func:`evaluate_model`.
    """
    optimizer = optim.Adam(model.parameters(), lr=0.00004)
    barrier = threading.Barrier(2)
    action_counts = [0, 0, 0]
    action_lock = threading.Lock()
    model_lock = threading.Lock()

    t = threading.Thread(
        target=worker,
        args=(0, model, optimizer, steps, currency_config, barrier),
        kwargs={
            "action_counts": action_counts,
            "action_lock": action_lock,
            "model_lock": model_lock,
        },
        daemon=True,
    )
    t.start()
    barrier.wait()
    t.join()

    _, avg_profit, _ = evaluate_model(model, currency_config, episodes=1)
    return avg_profit


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function."""
    decision_len = trial.suggest_int("decision_history_len", 8, 32)
    hidden_size = trial.suggest_int("hidden_size", 64, 256)
    num_layers = trial.suggest_int("num_lstm_layers", 1, 4)
    steps = trial.suggest_int("train_steps", 50, 200)

    profits = []
    for currency, cfg in CURRENCY_CONFIGS.items():
        model = ActorCritic(
            input_size=8,
            decision_history_len=decision_len,
            hidden_size=hidden_size,
            num_actions=3,
            num_lstm_layers=num_layers,
        )
        profit = train_once(model, cfg, steps)
        profits.append(profit)

        path = os.path.join(MODEL_DIR, f"{currency}_trial.pt")
        torch.save(model.state_dict(), path)

    return -float(sum(profits) / len(profits))


def run_optuna(study_name: str = "model_search", n_trials: int = 20) -> None:
    """Launch an Optuna hyperparameter search."""
    storage = f"sqlite:///{study_name}.db"
    study = optuna.create_study(direction="minimize", study_name=study_name,
                                storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    for key, value in trial.params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    run_optuna()
