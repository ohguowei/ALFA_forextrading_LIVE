import time
import numpy as np
from feature_extractor import compute_features

from oanda_api import (
    fetch_candle_data,
    open_position,
    close_position,
    get_open_positions
)

from config import TradingConfig  # Import the centralized trading config


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
        self.just_closed_profit = None  # <--- NEW

    def __repr__(self):
        return (
            f"Trade(side={self.side}, entry={self.entry_price}, "
            f"exit={self.exit_price}, profit={self.profit:.4f}, "
            f"timestamp={self.timestamp})"
        )

class LiveOandaForexEnv:
    def __init__(self, currency_config, candle_count=TradingConfig.CANDLE_COUNT, granularity=TradingConfig.GRANULARITY):
        self.account_id = currency_config.account_id
        self.access_token = currency_config.access_token
        self.environment = currency_config.environment
        self.currency_config = currency_config
        
        self.instrument = currency_config.instrument
        self.units = currency_config.live_units
        self.account_id = currency_config.account_id  # Use this when calling API functions
        self.granularity = granularity
        self.candle_count = candle_count

        # Fetch initial data and compute features
        self.data = self._fetch_initial_data()
        self.features = compute_features(self.data)
        self.current_index = 16

        # Trade state variables
        self.position_open = False
        self.position_side = None
        self.entry_price = None
        self.trade_log = []
        self.just_closed_profit = None
        self._sync_oanda_position_state()
        

    def _sync_oanda_position_state(self):
        """
        Check OANDA for any open positions in our 'instrument'.
        If found, set self.position_open, self.position_side, and self.entry_price.
        """
        positions_response = get_open_positions(self.account_id, self.access_token, self.environment)
        if not positions_response or "positions" not in positions_response:
            print("No positions found or error in position check.")
            return
    
        # positions_response["positions"] is typically a list of dicts
        for pos in positions_response["positions"]:
            if pos["instrument"] == self.instrument:
                # OANDA splits positions into 'long' and 'short' subfields
                long_units = float(pos["long"]["units"])
                short_units = float(pos["short"]["units"])
    
                if long_units > 0:
                    self.position_open = True
                    self.position_side = "long"
                    self.entry_price = float(pos["long"]["averagePrice"])
                    print(f"Detected existing LONG position at price {self.entry_price}.")
                    return
                elif short_units < 0:
                    self.position_open = True
                    self.position_side = "short"
                    self.entry_price = float(pos["short"]["averagePrice"])
                    print(f"Detected existing SHORT position at price {self.entry_price}.")
                    return
    
        # If no position in that instrument is found
        print(f"No existing position found for {self.instrument} in OANDA.")


    # live_env.py
    def _fetch_initial_data(self):
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
            return data
        except Exception as e:
            print(f"Error fetching initial data: {e}")
            raise
    
    def reset(self):
        self.current_index = 16
        self.position_open = False
        self.position_side = None
        self.entry_price = None
        self.trade_log = []
        
        self.data = self._fetch_initial_data()
        self.features = compute_features(self.data)  # Updated call.
        
        initial_features = self.features[self.current_index - 16 : self.current_index]
        current_pl = 0.0
        pl_column = np.full((initial_features.shape[0], 1), current_pl)
        initial_state = np.hstack((initial_features, pl_column))
        return initial_state
    
    def update_live_data(self):
        try:            
            new_candle = fetch_candle_data(
                self.instrument, 
                self.granularity, 
                candle_count=1, 
                access_token=self.access_token, 
                environment=self.environment
            )[0]
            
            if len(new_candle) != 6:
                raise ValueError("Invalid candle data returned from OANDA API.")
            
            self.data = np.vstack((self.data, new_candle))
            new_features = compute_features(np.vstack((self.data[-2:],)))
            self.features = np.vstack((self.features, new_features))
            self.current_index += 1
        except Exception as e:
            print(f"Error updating live data: {e}")

    def live_open_position(self, side):
        if self.position_open:
            print(f"Live: Position already open ({self.position_side}). Cannot open a new position.")
            return
    
        try:        
            response = open_position(instrument=self.instrument,
                                     account_id=self.account_id, 
                                     units=self.units, side=side, 
                                     access_token=self.access_token, 
                                     environment=self.environment)

            if response is not None:
                self.position_open = True
                self.position_side = side
                # Use the actual fill price from the API response if available
                if "orderFillTransaction" in response:
                    self.entry_price = float(response["orderFillTransaction"]["price"])
                elif "longOrderFillTransaction" in response:
                    self.entry_price = float(response["longOrderFillTransaction"]["price"])
                elif "shortOrderFillTransaction" in response:
                    self.entry_price = float(response["shortOrderFillTransaction"]["price"])
                else:
                    # Fallback: use local candle close price
                    self.entry_price = self.data[self.current_index][3]
                print(f"Live: Opened {side} position on {self.instrument} at {self.entry_price}.")
            else:
                print("Live: Order failed.")
        except Exception as e:
            print(f"Error opening position: {e}")
    
    
    def live_close_position(self):
        if not self.position_open:
            print("Live: No open position to close.")
            return
    
        try:            
            response = close_position(account_id=self.account_id, 
                                      instrument=self.instrument, 
                                      position_side=self.position_side, 
                                      access_token=self.access_token, 
                                      environment=self.environment)

            if response is not None:
                # Parse exit_price from 'orderFillTransaction'
                if "orderFillTransaction" in response:
                    exit_price = float(response["orderFillTransaction"]["price"])
                elif "longOrderFillTransaction" in response:
                    exit_price = float(response["longOrderFillTransaction"]["price"])
                elif "shortOrderFillTransaction" in response:
                    exit_price = float(response["shortOrderFillTransaction"]["price"])
                else:
                    exit_price = self.data[self.current_index][3]
    
                # Realized profit
                if self.position_side == "long":
                    profit = (exit_price - self.entry_price) / self.entry_price
                else:  # short
                    profit = (self.entry_price - exit_price) / self.entry_price
    
                # Record the closed trade
                trade = Trade(
                    side=self.position_side,
                    entry_price=self.entry_price,
                    exit_price=exit_price,
                    profit=profit,
                    timestamp=time.time()
                )
                self.trade_log.append(trade)
                print(f"Live: Closed {self.position_side} position on {self.instrument} at {exit_price}, profit={profit:.4f}")
                
                # <--- STORE the just-closed profit here so compute_reward() can return it.
                self.just_closed_profit = profit

                # Reset open-position flags
                self.position_open = False
                self.position_side = None
                self.entry_price = None
            else:
                print("Live: Close request failed or returned None from OANDA.")
        except Exception as e:
            print(f"Error closing position: {e}")

    def compute_reward(self, action):
        """
        Compute reward based on actual P&L:
          1) If we closed a trade this step, return that trade’s realized profit.
          2) Else if we have an open position, return mark-to-market profit.
          3) Otherwise return 0.
        """
        # 1) If a position was closed this step, immediately return that realized P&L.
        if self.just_closed_profit is not None:
            reward = self.just_closed_profit
            self.just_closed_profit = None
            return reward
        
        # 2) If a position is still open, compute mark-to-market.
        if self.position_open:
            current_price = self.data[self.current_index][3]
            if self.position_side == "long":
                return (current_price - self.entry_price) / self.entry_price
            else:  # short
                return (self.entry_price - current_price) / self.entry_price
        
        # 3) No open position and no just-closed position => zero reward
        return 0.0
    
    def step(self, action):
        # Sync with OANDA to update local state based on actual open positions
        self._sync_oanda_position_state()
        
        # Execute trade action based on the new signal
        if action == 0:  # long signal
            if not self.position_open or self.position_side != "long":
                if self.position_open:
                    self.live_close_position()
                self.live_open_position("long")
        
        elif action == 1:  # short signal
            if not self.position_open or self.position_side != "short":
                if self.position_open:
                    self.live_close_position()
                self.live_open_position("short")
        
        elif action == 2:  # neutral signal: close any open trade
            if self.position_open:
                self.live_close_position()
        
        # Compute reward using the updated method
        reward = self.compute_reward(action)
        
        # Update live data (this updates self.data and self.features, and increments self.current_index)
        self.update_live_data()
        
        # Prepare the next state: a sliding window of 16 rows of features (each originally 12 dimensions)
        next_features = self.features[self.current_index-16:self.current_index]
        
        # Calculate the current P/L:
        if self.position_open:
            # Using the current candle's close as a proxy for current market price.
            current_price = self.data[self.current_index][3]
            if self.position_side == "long":
                current_pl = (current_price - self.entry_price) / self.entry_price
            elif self.position_side == "short":
                current_pl = (self.entry_price - current_price) / self.entry_price
        else:
            # Use the last realized P/L if available, or 0 if no trades have occurred
            current_pl = self.trade_log[-1].profit if self.trade_log else 0.0
    
        # Append the P/L as an extra feature column to each row in the sliding window
        # This creates a column with shape (16, 1) filled with current_pl.
        pl_column = np.full((next_features.shape[0], 1), current_pl)
        # Concatenate horizontally to obtain an updated state with 13 features per timestep.
        next_state = np.hstack((next_features, pl_column))
        
        done = False  # Live trading typically runs continuously
        return next_state, reward, done, {}
