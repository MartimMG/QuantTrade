import numpy as np

def simulate_trade(entry, highs, lows, direction, sl, tp):
    for i, (h, l) in enumerate(zip(highs, lows)):
        if direction == 1 and (h - entry) * 10_000 >= tp:
            return 'TP', i+1
        if direction == 1 and (entry - l) * 10_000 >= sl:
            return 'SL', i+1
        if direction == -1 and (entry - l) * 10_000 >= tp:
            return 'TP', i+1
        if direction == -1 and (h - entry) * 10_000 >= sl:
            return 'SL', i+1
    return 'NONE', len(highs)

def optimize_sl_tp_per_class(y, close_prices, highs, lows, sl_values, tp_values, window_size, val_size, step, class_to_direction, cost_per_trade=1.5):
    sl_tp_map = {}
    for cls in [0, 1, 3, 4]:
        best_profit = -np.inf
        best_pair = (12, 20)
        for sl in sl_values:
            for tp in tp_values:
                temp_profit = 0
                count = 0
                for start in range(0, len(y) - window_size - val_size, step):
                    val_y = y[start+window_size:start+window_size+val_size]
                    val_start = start + window_size
                    max_len = min(val_size, len(y) - val_start)

                    entry_prices = close_prices[val_start:val_start + max_len]
                    future_highs_seq = [highs[t:t+12] for t in range(val_start, val_start + max_len)]
                    future_lows_seq = [lows[t:t+12] for t in range(val_start, val_start + max_len)]
                    val_preds = val_y[:max_len]

                    for pred, entry, highs_seq, lows_seq in zip(val_preds, entry_prices, future_highs_seq, future_lows_seq):
                        if pred != cls:
                            continue
                        direction = class_to_direction[pred]
                        outcome, _ = simulate_trade(entry, highs_seq, lows_seq, direction, sl, tp)
                        if outcome == 'TP':
                            temp_profit += tp
                        elif outcome == 'SL':
                            temp_profit -= sl
                        temp_profit -= cost_per_trade
                        count += 1
                if count > 0 and temp_profit > best_profit:
                    best_profit = temp_profit
                    best_pair = (sl, tp)
        sl_tp_map[cls] = {'sl': best_pair[0], 'tp': best_pair[1]}
    sl_tp_map[2] = {'sl': None, 'tp': None}  # no-trade
    return sl_tp_map

def estimate_avg_duration_per_class(y, close_prices, highs, lows, sl_tp_map, window_size, val_size, step, class_to_direction):
    duration_by_class = {0: [], 1: [], 3: [], 4: []}
    for start in range(0, len(y) - window_size - val_size, step):
        val_y = y[start+window_size:start+window_size+val_size]
        val_start = start + window_size
        max_len = min(val_size, len(y) - val_start)

        entry_prices = close_prices[val_start:val_start + max_len]
        future_highs_seq = [highs[t:t+12] for t in range(val_start, val_start + max_len)]
        future_lows_seq = [lows[t:t+12] for t in range(val_start, val_start + max_len)]
        val_preds = val_y[:max_len]

        for pred, entry, highs_seq, lows_seq in zip(val_preds, entry_prices, future_highs_seq, future_lows_seq):
            direction = class_to_direction.get(pred, 0)
            if direction == 0:
                continue
            sltp = sl_tp_map.get(pred, {'sl': None, 'tp': None})
            if sltp['sl'] is None or sltp['tp'] is None:
                continue
            result, duration = simulate_trade(entry, highs_seq, lows_seq, direction, sltp['sl'], sltp['tp'])
            duration_by_class[pred].append(duration)

    avg_duration_by_class = {
        k: round(np.mean(v)) if v else 12 for k, v in duration_by_class.items()
    }
    return avg_duration_by_class

def relabel_data(df, sl_tp_map, avg_duration_by_class, class_to_direction, cost_per_trade=1.5):
    relabeled = []
    for i in range(len(df)):
        row = df.iloc[i]
        label = row['label']
        direction = class_to_direction.get(label, 0)
        if direction == 0:
            relabeled.append(2)  # No-trade
            continue

        sltp = sl_tp_map.get(label, {'sl': None, 'tp': None})
        horizon = avg_duration_by_class.get(label, 12)
        if sltp['sl'] is None or sltp['tp'] is None:
            relabeled.append(2)
            continue

        highs_seq = df['High'].iloc[i:i+horizon].values
        lows_seq = df['Low'].iloc[i:i+horizon].values
        if len(highs_seq) < horizon or len(lows_seq) < horizon:
            relabeled.append(2)
            continue

        result, _ = simulate_trade(row['Close'], highs_seq, lows_seq, direction, sltp['sl'], sltp['tp'])
        if result == 'TP':
            relabeled.append(label)
        elif result == 'SL':
            relabeled.append(label)
        else:
            relabeled.append(2)  # No-trade if neither hit
    return np.array(relabeled)
