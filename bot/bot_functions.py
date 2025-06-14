import MetaTrader5 as mt5
import datetime
import time
import pytz
import numpy as np
import pandas as pd
import os
import ta

# 15 porque é preciso dados de uma hora atrás
def get_latest_data(symbol, timeframe, n):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def build_dataset(df):
    # ---- 1. Load raw M1 ASCII -----------------------------------------------
    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'tick_volume': float})
    df = df.rename(columns={'time': 'Date_Time', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'})
    df.index = df['Date_Time']
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df['Volume'] = 0
    
    m5 = df
    # ---- 3. M5 Technical indicators -------------------------------------------
    m5["rsi"]  = ta.momentum.RSIIndicator(m5["Close"], 14).rsi()
    macd = ta.trend.MACD(m5["Close"])
    m5["macd"] = macd.macd_diff()
    m5["body"] = m5["Close"] - m5["Open"]
    m5["vol_local"] = m5["High"] - m5["Low"]

    # ---- 3b. Add Rolling M15 (3 x 5m) features --------------------------------
    m5["roll_m15_high"]  = m5["High"].rolling(3).max()
    m5["roll_m15_low"]   = m5["Low"].rolling(3).min()
    m5["roll_m15_close"] = m5["Close"].rolling(3).apply(lambda x: x.iloc[-1], raw=False)
    m5["roll_m15_trend"] = m5["Close"] - m5["Close"].rolling(3).mean()
    m5["roll_m15_body"]  = (m5["Close"] - m5["Open"]).rolling(3).mean()

    # ---- 3c. Add Rolling H1 (12 x 5m) features --------------------------------
    m5["roll_h1_high"]  = m5["High"].rolling(12).max()
    m5["roll_h1_low"]   = m5["Low"].rolling(12).min()
    m5["roll_h1_close"] = m5["Close"].rolling(12).apply(lambda x: x.iloc[-1], raw=False)
    m5["roll_h1_vol"]   = m5["roll_h1_high"] - m5["roll_h1_low"]
    m5["roll_h1_relpos"] = (m5["Close"] - m5["roll_h1_low"]) / (m5["roll_h1_high"] - m5["roll_h1_low"] + 1e-6)

    
    # -7. Add temporal features --------------------------------------------
    m5["hour"]          = m5.index.hour
    m5["dayofweek"]     = m5.index.dayofweek
    m5["mins_into_m15"] = m5.index.minute % 15
    m5["frac_into_m15"] = m5["mins_into_m15"] / 15

    # ---- 8. Drop NaNs caused by rolling / label shift ------------------------

    return m5

def scale(df, scaler):
    scaler.fit(df)
    df_scaled = scaler.transform(df)
    return df_scaled

def calculate_lot_size_multiplier(current_sl_pips, running_balance_in_window, risk_per_trade_percentage):
    monetary_risk_for_this_trade = running_balance_in_window * risk_per_trade_percentage

    calculated_lot_size_multiplier = monetary_risk_for_this_trade / (current_sl_pips * 10)

    min_broker_lot_size = 0.01 # Example: Minimum micro lot
    max_broker_lot_size = 50.0 # Example: Maximum standard lots allowed
        
    calculated_lot_size_multiplier = min(max_broker_lot_size, calculated_lot_size_multiplier)
    calculated_lot_size_multiplier = round(calculated_lot_size_multiplier, 2)
    if calculated_lot_size_multiplier < min_broker_lot_size:
        return None
    return calculated_lot_size_multiplier

def execute_trade(sl, tp, direction, lot_size_multiplier, symbol, deviation):
    tick = mt5.symbol_info_tick(symbol)
    if direction == 1:
        order = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size_multiplier,
            "type": mt5.ORDER_TYPE_BUY,
            "price": tick.ask,
            "sl": tick.ask - sl * 0.0001,
            "tp": tick.ask + tp * 0.0001,
            "deviation": deviation,
            "magic": 123456,
            "comment": "ML Bot Buy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(order)
    elif direction == -1:
        order = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size_multiplier,
            "type": mt5.ORDER_TYPE_SELL,
            "price": tick.bid,
            "sl": tick.bid + sl * 0.0001,
            "tp": tick.bid - tp * 0.0001,
            "deviation": deviation,
            "magic": 123456,
            "comment": "ML Bot Sell",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(order)