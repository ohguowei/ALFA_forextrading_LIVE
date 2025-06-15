import numpy as np


def compute_rsi(closes, period=15):
    """Return the Relative Strength Index for ``closes``.

    Parameters
    ----------
    closes : array-like
        Sequence of close prices.
    period : int, optional
        Number of periods to use for RSI calculation. Defaults to ``15``.

    Returns
    -------
    np.ndarray
        Array of RSI values aligned with ``closes``.
    """
    closes = np.asarray(closes, dtype=np.float64)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    rsi = np.full(closes.shape, 50.0, dtype=np.float64)
    if closes.size <= period:
        return rsi

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi[period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period + 1, closes.size):
        delta = deltas[i - 1]
        gain = max(delta, 0.0)
        loss = -min(delta, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi

def compute_features(data):
    """Return feature matrix derived from OHLCV data.

    For each time step ``i`` (starting at 1) the following features are
    computed:

    ``x1``  Percentage change in close price.
    ``x2``  Percentage change in high price.
    ``x3``  Percentage change in low price.
    ``x4``  Relative difference between high and close.
    ``x5``  Relative difference between close and low.
    ``x6``  Actual spread computed from bid/ask prices.
    ``x7``  15-period RSI of the close price.

    Parameters
    ----------
    data : array-like
        Array of candles ``[open, high, low, close, volume, spread]``.

    Returns
    -------
    np.ndarray
        Array with shape ``(len(data)-1, 7)`` containing the features.
    """

    closes = [c[3] for c in data]
    rsi_values = compute_rsi(closes, period=15)

    features = []
    for i in range(1, len(data)):
        _, h_prev, l_prev, c_prev, _, _ = data[i - 1]
        _, h, l, c, _, spread = data[i]
        x1 = (c - c_prev) / c_prev
        x2 = (h - h_prev) / h_prev
        x3 = (l - l_prev) / l_prev
        x4 = (h - c) / c if c != 0 else 0.0
        x5 = (c - l) / c if c != 0 else 0.0
        x6 = spread
        x7 = rsi_values[i]
        features.append([x1, x2, x3, x4, x5, x6, x7])

    return np.array(features)
