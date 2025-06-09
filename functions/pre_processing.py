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
    hours_ahead = 1
    steps = int(hours_ahead * 60 / 5)  # e.g., 12 steps = 1 hour ahead

    future_high = m5['High'].rolling(window=steps).max().shift(-steps)
    future_low = m5['Low'].rolling(window=steps).min().shift(-steps)
    entry = m5['Close']

    # Calculate max potential return up or down
    up_pips = (future_high - entry) * 10_000
    down_pips = (entry - future_low) * 10_000

    def classify_from_extremes(up, down):
        if down > 20: return 0  # strong short
        elif down > 10: return 1  # weak short
        elif up > 20: return 4  # strong long
        elif up > 10: return 3  # weak long
        else: return 2  # no trade

    m5['label'] = [classify_from_extremes(u, d) for u, d in zip(up_pips, down_pips)]
    
    # ---- 7. Add temporal features --------------------------------------------
    m5["hour"]          = m5.index.hour
    m5["dayofweek"]     = m5.index.dayofweek
    m5["mins_into_m15"] = m5.index.minute % 15
    m5["frac_into_m15"] = m5["mins_into_m15"] / 15

    # ---- 8. Drop NaNs caused by rolling / label shift ------------------------
    m5 = m5.dropna()

    return m5
