import os
import pandas as pd
import ta  

def build_eurusd_dataset(filename, hours_ahead=1):
    # ---- 1. Load raw M1 ASCII -----------------------------------------------
    file_path = os.path.join(os.getcwd(), filename)
    df = pd.read_csv(
        file_path,
        sep=',',
        header=None,
        names=['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    )
    df = df.astype({'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': float})
    df.index = pd.to_datetime(df['Date_Time'], format='%Y%m%d %H%M%S')
    df = df.drop(columns=['Date_Time'])

    # ---- 2. Resample ----------------------------------------------------------
    def resample_ohlc(data, rule):
        return data.resample(rule).agg({
            "Open": "first",
            "High": "max",
            "Low":  "min",
            "Close":"last",
            "Volume":"sum"
        }).dropna()

    m5  = resample_ohlc(df,  '5min')
    # m15 = resample_ohlc(df, '15min')
    # h1  = resample_ohlc(df,  '1h')

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

    # ---- 5. Build future-range label -----------------------------------------
    # Assumes 5-min candles
    max_steps = 8  # max candles to look ahead, put six to b more conservative
    entry = m5['Close'].values
    highs = m5['High'].values
    lows = m5['Low'].values

    labels = []

    for i in range(len(m5)):
        label = 2  # default: no trade
        weak_label = None

        for step in range(1, max_steps + 1):
            if i + step >= len(m5):
                break

            future_high = highs[i + step]
            future_low = lows[i + step]
            up_pips = (future_high - entry[i]) * 10_000
            down_pips = (entry[i] - future_low) * 10_000

            # Strong signals - exit immediately
            if down_pips > 20:
                label = 0  # strong short
                break
            elif up_pips > 20:
                label = 4  # strong long
                break

            # Capture first weak signal only
            if weak_label is None:
                if down_pips > 10:
                    weak_label = 1  # weak short
                elif up_pips > 10:
                    weak_label = 3  # weak long

        if label == 2 and weak_label is not None:
            label = weak_label

        labels.append(label)

    m5['label'] = labels

    
    # -7. Add temporal features --------------------------------------------
    m5["hour"]          = m5.index.hour
    m5["dayofweek"]     = m5.index.dayofweek
    m5["mins_into_m15"] = m5.index.minute % 15
    m5["frac_into_m15"] = m5["mins_into_m15"] / 15

    # ---- 8. Drop NaNs caused by rolling / label shift ------------------------
    m5 = m5.dropna()
    m5_outside_overlap = m5.drop(m5.between_time("12:00", "16:00").index)

    return m5_outside_overlap