import numpy as np

def compute_features(data):
    """
    Compute features for each time step from the OHLC data with actual spread.
    For each time step i (starting at 1):
      x1 = (c_i - c_{i-1}) / c_{i-1}      : % change in close price
      x2 = (h_i - h_{i-1}) / h_{i-1}      : % change in high price
      x3 = (l_i - l_{i-1}) / l_{i-1}      : % change in low price
      x4 = (h_i - c_i) / c_i              : Relative difference between high and close
      x5 = (c_i - l_i) / c_i              : Relative difference between close and low
      x6 = actual spread                  : Spread computed from bid/ask prices
    
    Returns:
      A NumPy array of shape (len(data)-1, 6)
    """
    features = []
    for i in range(1, len(data)):
        # Use previous candle for price comparisons; ignore volume and spread.
        _, h_prev, l_prev, c_prev, _, _ = data[i-1]
        o, h, l, c, _, spread = data[i]
        x1 = (c - c_prev) / c_prev
        x2 = (h - h_prev) / h_prev
        x3 = (l - l_prev) / l_prev
        x4 = (h - c) / c if c != 0 else 0
        x5 = (c - l) / c if c != 0 else 0
        x6 = spread
        features.append([x1, x2, x3, x4, x5, x6])
    return np.array(features)
