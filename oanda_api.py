import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions

def get_client(access_token, environment):
    return API(access_token=access_token, environment=environment)

# oanda_api.py

def fetch_candle_data(instrument, granularity="H1", candle_count=500, access_token=None, environment=None):
    if not access_token or not environment:
        raise ValueError("Access token and environment must be provided")
    client = get_client(access_token, environment)
    params = {
        "granularity": granularity,
        "count": candle_count,
        "price": "BA"  # Request Bid and Ask prices.
    }
    r = instruments.InstrumentsCandles(instrument, params=params)
    response = client.request(r)
    if "candles" not in response:
        raise ValueError("Invalid API response: 'candles' field missing.")
    candles = response["candles"]
    data = []
    for candle in candles:
        if "bid" not in candle or "ask" not in candle or "volume" not in candle:
            print(f"Skipping invalid candle: {candle}")
            continue
        try:
            bid = candle["bid"]
            ask = candle["ask"]
            # Compute mid prices from bid and ask values.
            o_bid = float(bid["o"])
            o_ask = float(ask["o"])
            h_bid = float(bid["h"])
            h_ask = float(ask["h"])
            l_bid = float(bid["l"])
            l_ask = float(ask["l"])
            c_bid = float(bid["c"])
            c_ask = float(ask["c"])
            
            o = (o_bid + o_ask) / 2
            h = (h_bid + h_ask) / 2
            l = (l_bid + l_ask) / 2
            c = (c_bid + c_ask) / 2
            
            # Calculate the actual spread using the close prices.
            spread = c_ask - c_bid
            
            v = int(candle["volume"])
            # Return six values per candle: [open, high, low, close, volume, spread]
            data.append([o, h, l, c, v, spread])
        except (KeyError, ValueError) as e:
            print(f"Skipping invalid candle due to error: {e}")
            continue
    if not data:
        raise ValueError("No valid candles found in the API response.")
    return data

def open_position(account_id, instrument, units, side, access_token, environment):
    client = get_client(access_token, environment)
    order_data = {
        "order": {
            "units": str(units if side == "long" else -units),
            "instrument": instrument,
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }
    }
    r = orders.OrderCreate(account_id, data=order_data)
    try:
        response = client.request(r)
        print(f"Opened {side} position on {instrument}: {response}")
        return response
    except Exception as e:
        print(f"Order failed: {e}")
        return None

def close_position(account_id, instrument, position_side, access_token, environment):
    client = get_client(access_token, environment)
    try:
        if position_side == "long":
            data = {"longUnits": "ALL"}
        elif position_side == "short":
            data = {"shortUnits": "ALL"}
        else:
            raise ValueError("position_side must be 'long' or 'short'")
        r = positions.PositionClose(accountID=account_id, instrument=instrument, data=data)
        response = client.request(r)
        print(f"Position closed for {instrument}: {response}")
        return response
    except Exception as e:
        print(f"Error closing position: {e}")
        return None

def get_open_positions(account_id, access_token, environment):
    client = get_client(access_token, environment)
    try:
        r = positions.OpenPositions(accountID=account_id)
        response = client.request(r)
        return response
    except Exception as e:
        print(f"Error retrieving open positions: {e}")
        return None
