# config.py
# Base trading configuration
class TradingConfig:
    GRANULARITY = "H1"
#    CANDLE_COUNT = 5000
    CANDLE_COUNT = 136

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
        account_id="101-001-26348919-001",
        access_token="68ff286dfb6bc058031e66ddcdc72d64-138d97e64d2976820a19a4b179cdcf09",
        environment="practice"
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
         account_id="101-001-26348919-001",
         access_token="68ff286dfb6bc058031e66ddcdc72d64-138d97e64d2976820a19a4b179cdcf09",
         environment="practice"
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
         account_id="101-001-26348919-001",
         access_token="68ff286dfb6bc058031e66ddcdc72d64-138d97e64d2976820a19a4b179cdcf09",
         environment="practice" 
     #    account_id="001-003-255162-003",
     #    access_token="c33734921cd0b7b68c721fc18e2019c2-8cfd11c75b7df0c81301e2cf58846540",
     #    environment="live"
     ),     
     "GBP_USD": CurrencyConfig(
         instrument="GBP_USD",
         live_units=1000,
         simulated_units=1000,
         spread=0.0002,
         account_id="101-001-26348919-001",
         access_token="68ff286dfb6bc058031e66ddcdc72d64-138d97e64d2976820a19a4b179cdcf09",
         environment="practice"
        
    #    #account_id="001-003-255162-002",
    #    #access_token="c33734921cd0b7b68c721fc18e2019c2-8cfd11c75b7df0c81301e2cf58846540",
    #    #environment="live"
     ),
     "GBP_AUD": CurrencyConfig(
         instrument="GBP_AUD",
         live_units=1000,
         simulated_units=1000,
         spread=0.00055,
         account_id="101-001-26348919-001",
         access_token="68ff286dfb6bc058031e66ddcdc72d64-138d97e64d2976820a19a4b179cdcf09",
         environment="practice"
        
    #    #account_id="001-003-255162-002",
    #    #access_token="c33734921cd0b7b68c721fc18e2019c2-8cfd11c75b7df0c81301e2cf58846540",
    #    #environment="live"
     ),
     "EUR_GBP": CurrencyConfig(
         instrument="EUR_GBP",
         live_units=1000,
         simulated_units=1000,
         spread=0.00015,
         account_id="101-001-26348919-001",
         access_token="68ff286dfb6bc058031e66ddcdc72d64-138d97e64d2976820a19a4b179cdcf09",
         environment="practice"
        
    #    #account_id="001-003-255162-002",
    #    #access_token="c33734921cd0b7b68c721fc18e2019c2-8cfd11c75b7df0c81301e2cf58846540",
    #    #environment="live"
     ),    
    # # Add more currencies as needed...
}
