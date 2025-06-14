import time
import numpy as np
from oanda_api import fetch_candle_data
from feature_extractor import compute_features
from config import TradingConfig
from normalization import RunningStandardScaler

class Trade:
    """
    A class to represent a trade with structured information.
    """
    def __init__(self, side, entry_price, exit_price, profit, timestamp):
        self.side = side
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.profit = profit
        self.timestamp = timestamp

    def __repr__(self):
        return (f"Trade(side={self.side}, entry={self.entry_price}, "
                f"exit={self.exit_price}, profit={self.profit:.4f})")

class SimulatedOandaForexEnv:
    def __init__(self, currency_config, candle_count=TradingConfig.CANDLE_COUNT, granularity=TradingConfig.GRANULARITY):
        self.currency_config = currency_config
        self.instrument = currency_config.instrument
        self.units = currency_config.simulated_units
        # With the new approach, the actual spread is computed from candle data.
        # Thus, we no longer use a fixed spread from the config.
        
        self.account_id = currency_config.account_id
        self.access_token = currency_config.access_token
        self.environment = currency_config.environment
        
        self.granularity = granularity
        self.candle_count = candle_count

        # Fetch initial historical data and compute features.
        self.data = self._fetch_initial_data()  # Expecting each candle to have 6 values: [o, h, l, c, volume, spread]
        self.features = compute_features(self.data)  # Returns a (n, 6) array
        self.scaler = RunningStandardScaler(self.features.shape[1])
        self.scaler.update(self.features)
        self.current_index = 16

        self.position_open = False
        self.position_side = None
        self.entry_price = None
        self.trade_log = []
        self.just_closed_profit = None

    def _fetch_initial_data(self):
        attempts = 0
        max_attempts = 5
        while attempts < max_attempts:
            try:
                data = np.array(fetch_candle_data(
                    self.instrument, 
                    self.granularity, 
                    self.candle_count, 
                    access_token=self.access_token, 
                    environment=self.environment
                ))
                if len(data) == 0:
                    raise ValueError("No data returned from OANDA API.")
                print(f"Successfully fetched {len(data)} candles for {self.instrument}.")
                return data
            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts} - Error fetching initial data: {e}")
                time.sleep(30)
        print("Max attempts reached. Using fallback data.")
        # Fallback candle includes a spread value (defaulting to 0)
        fallback_candle = [1.0, 1.0, 1.0, 1.0, 0, 0]
        fallback_data = np.array([fallback_candle] * self.candle_count)
        return fallback_data

    def reset(self):
        """
        Reset the environment to its initial state.
        Returns a state with shape (16, 7):
          - 6 features from compute_features (x1–x6)
          - 1 additional column for P/L (set to 0)
        """
        self.current_index = 16
        self.position_open = False
        self.position_side = None
        self.entry_price = None
        self.trade_log = []
        self.just_closed_profit = None

        self.scaler.update(self.features)
        base_features = self.features[self.current_index - 16 : self.current_index]
        base_features = self.scaler.normalize(base_features)
        current_pl = 0.0
        pl_column = np.full((base_features.shape[0], 1), current_pl)  # Shape: (16, 1)
        state_with_pl = np.hstack((base_features, pl_column))  # Final shape: (16, 7)
        return state_with_pl

    def update_live_data(self):
        try:
            new_candle = fetch_candle_data(
                self.instrument, 
                self.granularity, 
                candle_count=1, 
                access_token=self.access_token, 
                environment=self.environment
            )[0]
            if len(new_candle) != 6:  # Expecting [o, h, l, c, volume, spread]
                raise ValueError("Invalid candle data returned from OANDA API.")
            self.data = np.vstack((self.data, new_candle))
            new_features = compute_features(np.vstack((self.data[-2:],)))
            self.features = np.vstack((self.features, new_features))
            self.scaler.update(new_features)
            self.current_index += 1
        except Exception as e:
            print(f"Error updating live data: {e}")

    def compute_reward(self, action):
        """
        Compute reward in a manner consistent with the live environment.
        1) If a trade was closed this step, return that realized profit.
        2) If a position is open, return mark-to-market profit using the
           current candle's close price.
        3) Otherwise return 0.
        """
        if self.just_closed_profit is not None:
            reward = self.just_closed_profit
            self.just_closed_profit = None
            return float(np.clip(reward, -1.0, 1.0))

        if self.position_open:
            current_price = self.data[self.current_index][3]
            if self.position_side == "long":
                reward = (current_price - self.entry_price) / self.entry_price
                return float(np.clip(reward, -1.0, 1.0))
            else:
                reward = (self.entry_price - current_price) / self.entry_price
                return float(np.clip(reward, -1.0, 1.0))

        return 0.0

    def simulated_open_position(self, side):
        """
        Simulate opening a position.
        """
        if not self.position_open:
            self.position_open = True
            self.position_side = side
            # Use the current candle's close price as the entry price.
            self.entry_price = self.data[self.current_index][3]

    def simulated_close_position(self):
        """
        Simulate closing a position.
        """
        if self.position_open:
            exit_price = self.data[self.current_index][3]
            if self.position_side == "long":
                profit = (exit_price - self.entry_price) / self.entry_price
            else:
                profit = (self.entry_price - exit_price) / self.entry_price
            trade = Trade(
                side=self.position_side,
                entry_price=self.entry_price,
                exit_price=exit_price,
                profit=profit,
                timestamp=time.time()
            )
            self.trade_log.append(trade)
            self.position_open = False
            self.position_side = None
            self.entry_price = None
            return profit
        return None
    def step(self, action):
        """
        Execute one step in the simulated trading environment.
        Returns:
          next_state: (16, 7) array,
          reward: float,
          done: bool,
          info: dict
        """
        if action == 0:  # long
            if not self.position_open or self.position_side != "long":
                if self.position_open:
                    profit = self.simulated_close_position()
                    if profit is not None:
                        self.just_closed_profit = profit
                self.simulated_open_position("long")
        elif action == 1:  # short
            if not self.position_open or self.position_side != "short":
                if self.position_open:
                    profit = self.simulated_close_position()
                    if profit is not None:
                        self.just_closed_profit = profit
                self.simulated_open_position("short")
        elif action == 2:  # neutral (close any open position)
            if self.position_open:
                profit = self.simulated_close_position()
                if profit is not None:
                    self.just_closed_profit = profit

        reward = self.compute_reward(action)
        self.current_index += 1

        done = (self.current_index >= len(self.features))
        if done:
            return None, reward, done, {}

        next_features = self.features[self.current_index - 16 : self.current_index]
        next_features = self.scaler.normalize(next_features)
        if self.position_open:
            current_price = self.data[self.current_index][3]
            if self.position_side == "long":
                current_pl = (current_price - self.entry_price) / self.entry_price
            else:
                current_pl = (self.entry_price - current_price) / self.entry_price
        else:
            current_pl = self.trade_log[-1].profit if self.trade_log else 0.0

        pl_column = np.full((next_features.shape[0], 1), current_pl)
        next_state = np.hstack((next_features, pl_column))
        return next_state, reward, done, {}
