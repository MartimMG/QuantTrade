import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.utils import class_weight

def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def simulate_trade(entry_price, highs, lows, direction, sl, tp):
    for i, (high, low) in enumerate(zip(highs, lows)):
        if direction == 1:
            if (high - entry_price) * 10_000 >= tp:
                return tp, i + 1
            elif (entry_price - low) * 10_000 >= sl:
                return -sl, i + 1
        elif direction == -1:
            if (entry_price - low) * 10_000 >= tp:
                return tp, i + 1
            elif (high - entry_price) * 10_000 >= sl:
                return -sl, i + 1
    final_price = highs[-1] if direction == 1 else lows[-1]
    result = (final_price - entry_price) * 10_000 * direction
    return result, len(highs)

def optimize_sl_tp_per_class(y, close_prices, highs, lows, sl_values, tp_values, class_to_direction, cost_per_trade):
    sl_tp_map = {}
    for cls in [0, 1, 3, 4]:
        best_profit = -np.inf
        best_pair = (12, 20)
        for sl in sl_values:
            for tp in tp_values:
                temp_profit = 0
                count = 0
                for idx, pred in enumerate(y):
                    if pred != cls:
                        continue
                    direction = class_to_direction.get(pred, 0)
                    if direction == 0:
                        continue
                    # Make sure we have enough future data
                    if idx + 7 > len(close_prices):
                        continue
                    entry = close_prices[idx]
                    highs_seq = highs[idx:idx+7]
                    lows_seq = lows[idx:idx+7]
                    outcome, _ = simulate_trade(entry, highs_seq, lows_seq, direction, sl, tp)
                    temp_profit += outcome
                    temp_profit -= cost_per_trade
                    count += 1
                if count > 0 and temp_profit > best_profit:
                    best_profit = temp_profit
                    best_pair = (sl, tp)
        sl_tp_map[cls] = {'sl': best_pair[0], 'tp': best_pair[1]}
    sl_tp_map[2] = {'sl': None, 'tp': None}  # no-trade
    return sl_tp_map

def estimate_avg_duration_per_class(y, close_prices, highs, lows, sl_tp_map, class_to_direction):
    duration_by_class = {0: [], 1: [], 3: [], 4: []}

    for idx, pred in enumerate(y):
        direction = class_to_direction.get(pred, 0)
        if direction == 0:
            continue
        sltp = sl_tp_map.get(pred, {'sl': None, 'tp': None})
        if sltp['sl'] is None or sltp['tp'] is None:
            continue
        # Make sure we have enough future data
        if idx + 12 > len(close_prices):
            continue
        entry = close_prices[idx]
        highs_seq = highs[idx:idx+12]
        lows_seq = lows[idx:idx+12]
        _, duration = simulate_trade(entry, highs_seq, lows_seq, direction, sltp['sl'], sltp['tp'])
        duration_by_class[pred].append(duration)

    avg_duration_by_class = {
        k: round(np.mean(v)) if v else 7 for k, v in duration_by_class.items()
    }
    return avg_duration_by_class


# not being used yet

def relabel_data(df, sl_tp_map, avg_duration_by_class, class_to_direction):
    relabeled = []
    for i in range(len(df)):
        row = df.iloc[i]
        label = row['label']
        direction = class_to_direction.get(label, 0)
        if direction == 0:
            relabeled.append(2)  # No-trade
            continue

        sltp = sl_tp_map.get(label, {'sl': None, 'tp': None})
        horizon = avg_duration_by_class.get(label)
        if sltp['sl'] is None or sltp['tp'] is None:
            relabeled.append(2)
            continue

        highs_seq = df['High'].iloc[i:i+horizon].values
        lows_seq = df['Low'].iloc[i:i+horizon].values
        if len(highs_seq) < horizon or len(lows_seq) < horizon:
            relabeled.append(2)
            continue

        result, _ = simulate_trade(row['Close'], highs_seq, lows_seq, direction, sltp['sl'], sltp['tp'])
        if result > 0:
            relabeled.append(label)
        else:
            relabeled.append(2)  # No-trade if neither hit
    return np.array(relabeled)


# see this last part of the result > 0 because then it's sus