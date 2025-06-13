# config.py
# Base trading configuration
import os
import random
from typing import Optional

import numpy as np
import torch

class TradingConfig:
    GRANULARITY = "H1"
    # CANDLE_COUNT = 5000
    CANDLE_COUNT = 136

try:
    from local_config import (
        OANDA_ACCOUNT_ID as DEFAULT_ACCOUNT_ID,
        OANDA_ACCESS_TOKEN as DEFAULT_ACCESS_TOKEN,
        OANDA_ENVIRONMENT as DEFAULT_ENVIRONMENT,
    )
except ImportError:
    DEFAULT_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
    DEFAULT_ACCESS_TOKEN = os.getenv("OANDA_ACCESS_TOKEN")
    DEFAULT_ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")


def set_global_seed(seed: int) -> None:
    """Seed random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class CurrencyConfig:
    def __init__(self, instrument, live_units, simulated_units, spread, account_id, access_token, environment):
        self.instrument = instrument
        self.live_units = live_units
        self.simulated_units = simulated_units
        self.spread = spread
        self.account_id = account_id
        self.access_token = access_token
        self.environment = environment  # e.g., "live" or "practice"

CURRENCY_CONFIGS = {
    "EUR_USD": CurrencyConfig(
        instrument="EUR_USD",
        #live_units=1419,
        live_units=1000,
        simulated_units=1000,
        spread=0.0002,
        account_id=DEFAULT_ACCOUNT_ID,
        access_token=DEFAULT_ACCESS_TOKEN,
        environment=DEFAULT_ENVIRONMENT
     #   account_id="001-003-255162-003",
     #   access_token="c33734921cd0b7b68c721fc18e2019c2-8cfd11c75b7df0c81301e2cf58846540",
     #   environment="live"
     ),
     "AUD_USD": CurrencyConfig(
         instrument="AUD_USD",
       # live_units=1589,
         live_units=1000,
         simulated_units=1000,
         spread=0.0003,
        account_id=DEFAULT_ACCOUNT_ID,
        access_token=DEFAULT_ACCESS_TOKEN,
        environment=DEFAULT_ENVIRONMENT
      #   account_id="001-003-255162-003",
      #   access_token="c33734921cd0b7b68c721fc18e2019c2-8cfd11c75b7df0c81301e2cf58846540",
      #   environment="live"
     ),
     "EUR_AUD": CurrencyConfig(
         instrument="EUR_AUD",
         #live_units=945,
         live_units=1000,
         simulated_units=1000,
         spread=0.0004,
        account_id=DEFAULT_ACCOUNT_ID,
        access_token=DEFAULT_ACCESS_TOKEN,
        environment=DEFAULT_ENVIRONMENT
     #    account_id="001-003-255162-003",
     #    access_token="c33734921cd0b7b68c721fc18e2019c2-8cfd11c75b7df0c81301e2cf58846540",
     #    environment="live"
     ),     
     "GBP_USD": CurrencyConfig(
         instrument="GBP_USD",
         live_units=1000,
         simulated_units=1000,
         spread=0.0002,
        account_id=DEFAULT_ACCOUNT_ID,
        access_token=DEFAULT_ACCESS_TOKEN,
        environment=DEFAULT_ENVIRONMENT
        
    #    #account_id="001-003-255162-002",
    #    #access_token="c33734921cd0b7b68c721fc18e2019c2-8cfd11c75b7df0c81301e2cf58846540",
    #    #environment="live"
     ),
     "GBP_AUD": CurrencyConfig(
         instrument="GBP_AUD",
         live_units=1000,
         simulated_units=1000,
         spread=0.00055,
        account_id=DEFAULT_ACCOUNT_ID,
        access_token=DEFAULT_ACCESS_TOKEN,
        environment=DEFAULT_ENVIRONMENT
        
    #    #account_id="001-003-255162-002",
    #    #access_token="c33734921cd0b7b68c721fc18e2019c2-8cfd11c75b7df0c81301e2cf58846540",
    #    #environment="live"
     ),
     "EUR_GBP": CurrencyConfig(
         instrument="EUR_GBP",
         live_units=1000,
         simulated_units=1000,
         spread=0.00015,
        account_id=DEFAULT_ACCOUNT_ID,
        access_token=DEFAULT_ACCESS_TOKEN,
        environment=DEFAULT_ENVIRONMENT
        
    #    #account_id="001-003-255162-002",
    #    #access_token="c33734921cd0b7b68c721fc18e2019c2-8cfd11c75b7df0c81301e2cf58846540",
    #    #environment="live"
     ),    
    # # Add more currencies as needed...
}
