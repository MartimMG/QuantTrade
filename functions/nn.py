import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.utils import class_weight

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os

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

# Has the limit in count
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


def generate_model_report_pdf(
    steps,
    extra_steps,
    window_indices,
    f1_per_window,
    acc_per_window,
    profit_monetary_per_window, # Ensure this list contains monetary profits ($)
    trades_per_window,           # Ensure this list contains total trades for each window
    initial_account_balance,
    # Parameters
    window_size,
    val_size,
    step,
    cost_per_trade,
    pip_value_per_standard_lot, # Corrected name for clarity
    risk_per_trade_percentage,        # Corrected name for clarity (e.g., 0.1 for mini lot)
    report_filename
):
    # --- 1. Calculate Overall Statistics ---
    total_profit = np.sum(profit_monetary_per_window)
    total_trades = np.sum(trades_per_window)
    avg_f1 = np.mean(f1_per_window)
    avg_acc = np.mean(acc_per_window)

    # Maximum drawdown calculation
    cumulative_balance = [initial_account_balance]
    for p in profit_monetary_per_window:
        cumulative_balance.append(cumulative_balance[-1] + p)
    cumulative_balance_arr = np.array(cumulative_balance)
    peak = cumulative_balance_arr[0]
    max_drawdown_percentage = 0
    # Ensure cumulative_balance_arr has more than one element to avoid errors
    if len(cumulative_balance_arr) > 1:
        for balance in cumulative_balance_arr:
            if balance > peak:
                peak = balance # Update peak if new high
            # Drawdown for current balance relative to peak
            # Ensure peak is not zero to avoid division by zero
            drawdown = (peak - balance) / peak if peak != 0 else 0
            if drawdown > max_drawdown_percentage:
                max_drawdown_percentage = drawdown

    # --- 2. Generate Plots as PNG images ---
    # Set a style for better visualization
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 10})

    # Plot 1: Performance Metrics per Window (2x2 grid)
    fig1, axes1 = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    fig1.suptitle('Rolling Window Backtest Performance Metrics', fontsize=16)

    axes1[0, 0].plot(window_indices, profit_monetary_per_window, marker='o', linestyle='-', color='green', markersize=4)
    axes1[0, 0].set_title('Profit per Window (Dollars)')
    axes1[0, 0].set_xlabel('Window Index')
    axes1[0, 0].set_ylabel('Total Profit ($)') # Corrected label to Dollars
    axes1[0, 0].grid(True)
    axes1[0, 0].axhline(0, color='gray', linestyle='--', linewidth=0.8)

    axes1[0, 1].plot(window_indices, trades_per_window, marker='o', linestyle='-', color='blue', markersize=4) # Changed 'trade_per_window' to 'trades_per_window'
    axes1[0, 1].set_title('Number of Trades per Window')
    axes1[0, 1].set_xlabel('Window Index')
    axes1[0, 1].set_ylabel('Number of Trades')
    axes1[0, 1].grid(True)

    axes1[1, 0].plot(window_indices, f1_per_window, marker='o', linestyle='-', color='purple', markersize=4)
    axes1[1, 0].set_title('F1-score per Window (Weighted)')
    axes1[1, 0].set_xlabel('Window Index')
    axes1[1, 0].set_ylabel('F1-score')
    axes1[1, 0].grid(True)
    axes1[1, 0].set_ylim(0, 1)

    axes1[1, 1].plot(window_indices, acc_per_window, marker='o', linestyle='-', color='red', markersize=4)
    axes1[1, 1].set_title('Accuracy per Window')
    axes1[1, 1].set_xlabel('Window Index')
    axes1[1, 1].set_ylabel('Accuracy')
    axes1[1, 1].grid(True)
    axes1[1, 1].set_ylim(0, 1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot1_path = 'temp_metrics_plot.png'
    fig1.savefig(plot1_path)
    plt.close(fig1) # Close the figure to free up memory

    # Plot 2: Cumulative Profit
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(window_indices, cumulative_balance_arr[1:], color='darkgreen', linewidth=2) # [1:] because initial balance is at index 0
    plt.title('Cumulative Account Balance Over Windows')
    plt.xlabel('Window Index')
    plt.ylabel(f'Cumulative Balance ($)')
    plt.grid(True)
    plt.axhline(initial_account_balance, color='orange', linestyle='--', linewidth=0.8, label='Initial Balance')
    plt.axhline(np.max(cumulative_balance_arr), color='blue', linestyle='--', linewidth=0.8, label='All-time High')
    plt.axhline(np.min(cumulative_balance_arr), color='red', linestyle='--', linewidth=0.8, label='All-time Low')
    plt.legend()
    plot2_path = 'temp_cumulative_plot.png'
    fig2.savefig(plot2_path)
    plt.close(fig2)

    # --- 3. Create PDF Report ---
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title Page
    pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 10, "Trading Model Performance Report", 0, 1, "C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, "C")
    pdf.ln(20)

    # Parameters Section
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "1. Backtest Parameters", 0, 1, "L")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 7, f"""
    - Window Size: {window_size} candles
    - Validation Size: {val_size} candles
    - Step Size: {step} candles
    - Cost per Trade: {cost_per_trade} pips
    - Pip Value (Standard Lot): ${pip_value_per_standard_lot}
    - Risk per trade (e.g., 0.1 for Mini Lot): {risk_per_trade_percentage}
    - Initial Account Balance: ${initial_account_balance:,.2f}
    - Extra Steps in the simulate trade: {extra_steps}
    - Total Steps in the prediction: {steps}
    """)
    pdf.ln(5)

    # Overall Summary Statistics
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "2. Overall Performance Summary", 0, 1, "L")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 7, f"""
    - Total Profit: ${total_profit:,.2f}
    - Total Trades: {total_trades:,}
    - Average F1-score (Weighted): {avg_f1:.3f}
    - Average Accuracy: {avg_acc:.3f}
    - Maximum Drawdown: {max_drawdown_percentage * 100:.2f}%
    - Final Account Balance: ${cumulative_balance_arr[-1]:,.2f}
    - All-time High Balance: ${np.max(cumulative_balance_arr):,.2f}
    - All-time Low Balance: ${np.min(cumulative_balance_arr):,.2f}
    """)
    pdf.ln(5)

    # Plots Section
    pdf.add_page() # Add a new page for plots
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "3. Performance Visualizations", 0, 1, "L")
    pdf.ln(5)

    # Add Plot 1
    pdf.image(plot1_path, x=10, y=pdf.get_y(), w=180)
    pdf.ln(fig1.get_size_inches()[1] * 10) # Move cursor down after image

    # Add Plot 2
    pdf.image(plot2_path, x=10, y=pdf.get_y(), w=180)
    pdf.ln(fig2.get_size_inches()[1] * 10) # Move cursor down after image

    # Output PDF
    pdf.output(report_filename)

    # --- 4. Clean up temporary image files ---
    os.remove(plot1_path)
    os.remove(plot2_path)

    print(f"\nReport generated successfully: {report_filename}")