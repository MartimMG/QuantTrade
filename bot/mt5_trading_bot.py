import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import time
import joblib
import MetaTrader5 as mt5
from bot_functions import *

# Conectar com MT5

# Display data on the MetaTrader 5 package
print("MetaTrader5 package author: ", mt5.__author__)
print("MetaTrader5 package version: ", mt5.__version__)

# Establish connection to the MetaTrader 5 terminal
# You can specify a path to the terminal executable if it's not in the default location
mt5.initialize(path="C:\Program Files\MetaTrader 5 IC Markets EU\terminal64.exe")
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()

# Get basic terminal info
print(mt5.terminal_info())

# Get account info (useful to check if logged in)
account_info = mt5.account_info()
if account_info:
    print("\nAccount Info:")
    print(f"  Login: {account_info.login}")
    print(f"  Balance: {account_info.balance}")
    print(f"  Equity: {account_info.equity}")
    print(f"  Free Margin: {account_info.margin_free}")
else:
    print("Failed to get account info, error code =", mt5.last_error())

# At the end of your script or when done, shut down the connection
# mt5.shutdown()

# ir buscar o modelo
loaded_bundle = joblib.load('eurusd_model.joblib')
print("Bundle loaded.")
# Access individual components from the loaded bundle
model = loaded_bundle['model']
sl_tp_map = loaded_bundle['sl_tp_map']
avg_duration_by_class = loaded_bundle['avg_duration_by_class']
scaler = loaded_bundle['scaler']

#put parameters
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
mt5.symbol_select(SYMBOL, True)
class_to_direction = {0: -1, 1: -1, 2: 0, 3: 1, 4: 1}
# ver isto...................
DEVIATION = 10
risk_per_trade_percentage = 0.01

# run the model
while True:
    df = get_latest_data(SYMBOL, TIMEFRAME, 50)
    X = build_dataset(df)
    X_scaled = scale(X, scaler)
    last_candle = X_scaled[-1:].copy()    
    prediction = model.predict(last_candle)
    signal = np.argmax(prediction)
    sltp = sl_tp_map.get(signal, {'sl': None, 'tp': None})
    sl = sltp['sl']
    tp = sltp['tp']
    balance = account_info.equity
    direction = class_to_direction.get(signal, 0)
    print(f"Predição: {signal}")
    if signal != 2:
        lot_size_multiplier = calculate_lot_size_multiplier(sl, balance, risk_per_trade_percentage)
        execute_trade(sl, tp, direction, lot_size_multiplier, SYMBOL, DEVIATION)
    time.sleep(300)  # espera 5 minutos