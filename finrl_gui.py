

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkcalendar import DateEntry
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas_ta as ta
import pickle
from datetime import datetime
import os

# Import your FinRL agent class
from finrobot.agents.finrl_trading_agent import FinRLTradingAgent

# List of available technical indicators
TECH_INDICATORS = [
    'rsi', 'macd', 'bbands', 'ema', 'wma', 'hma', 'tema', 'cci', 'mfi', 'roc', 'stoch', 'atr'
]

def save_config():
    config = collect_config()
    if not config:
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])
    if file_path:
        with open(file_path, 'wb') as f:
            pickle.dump(config, f)
        messagebox.showinfo("Config Saved", f"Configuration saved to {file_path}")

def load_config():
    file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
    if file_path and os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            config = pickle.load(f)
        apply_config(config)
        messagebox.showinfo("Config Loaded", f"Configuration loaded from {file_path}")

def collect_config():
    try:
        return {
            "algorithm": algo_var.get(),
            "stocks": stocks_entry.get().upper().replace(" ", "").split(","),
            "train_start": train_start.get(),
            "train_end": train_end.get(),
            "test_start": test_start.get(),
            "test_end": test_end.get(),
            "initial_balance": float(balance_entry.get()),
            "transaction_cost": float(cost_entry.get()),
            "max_shares": int(hmax_entry.get()),
            "reward_scaling": float(reward_entry.get()),
            "timesteps": int(steps_entry.get()),
            "random_seed": int(seed_entry.get()) if seed_entry.get() else None,
            "tech_indicators": [TECH_INDICATORS[i] for i in tech_listbox.curselection()]
        }
    except Exception as e:
        messagebox.showerror("Input Error", f"Error in input fields: {e}")
        return None

def apply_config(config):
    if not config:
        return
    algo_var.set(config.get("algorithm", RL_ALGOS[0]))
    stocks_entry.delete(0, tk.END)
    stocks_entry.insert(0, ",".join(config.get("stocks", [])))
    train_start.set_date(config.get("train_start", datetime.now()))
    train_end.set_date(config.get("train_end", datetime.now()))
    test_start.set_date(config.get("test_start", datetime.now()))
    test_end.set_date(config.get("test_end", datetime.now()))
    balance_entry.delete(0, tk.END)
    balance_entry.insert(0, str(config.get("initial_balance", "")))
    cost_entry.delete(0, tk.END)
    cost_entry.insert(0, str(config.get("transaction_cost", "")))
    hmax_entry.delete(0, tk.END)
    hmax_entry.insert(0, str(config.get("max_shares", "")))
    reward_entry.delete(0, tk.END)
    reward_entry.insert(0, str(config.get("reward_scaling", "")))
    steps_entry.delete(0, tk.END)
    steps_entry.insert(0, str(config.get("timesteps", "")))
    seed_entry.delete(0, tk.END)
    seed_entry.insert(0, str(config.get("random_seed", "")))
    tech_listbox.selection_clear(0, tk.END)
    idxs = [TECH_INDICATORS.index(ti) for ti in config.get("tech_indicators", []) if ti in TECH_INDICATORS]
    for idx in idxs:
        tech_listbox.selection_set(idx)

def save_results(results_df):
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if file_path:
        results_df.to_csv(file_path, index=False)
        messagebox.showinfo("Results Saved", f"Results saved to {file_path}")

def fetch_and_plot():
    config = collect_config()
    if not config:
        return
    symbols = config["stocks"]
    start = config["train_start"]
    end = config["test_end"]  # show until end of testing
    indicators = config["tech_indicators"]

    for widget in chart_frame.winfo_children():
        widget.destroy()
    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start, end=end)
            if df.empty:
                raise ValueError("No data")
        except Exception as e:
            messagebox.showerror("Data Error", f"Could not fetch data for {symbol}: {e}")
            continue

        apds = []

        # Add overlays based on selected indicators
        if 'bbands' in indicators:
            bband = ta.bbands(df['Close'])
            df['BBL'], df['BBM'], df['BBU'] = bband['BBL_5_2.0'], bband['BBM_5_2.0'], bband['BBU_5_2.0']
            apds += [
                mpf.make_addplot(df['BBL'], color='blue', width=0.7),
                mpf.make_addplot(df['BBM'], color='grey', width=0.7),
                mpf.make_addplot(df['BBU'], color='blue', width=0.7),
            ]
        if 'rsi' in indicators:
            df['RSI'] = ta.rsi(df['Close'])
            apds.append(mpf.make_addplot(df['RSI'], panel=1, color='magenta', width=0.7))
        if 'macd' in indicators:
            macd = ta.macd(df['Close'])
            df['MACD'] = macd['MACD_12_26_9']
            df['MACDh'] = macd['MACDh_12_26_9']
            apds.append(mpf.make_addplot(df['MACD'], panel=2, color='green', width=0.7))
            apds.append(mpf.make_addplot(df['MACDh'], panel=2, color='red', type='bar'))

        fig = mpf.figure(style='yahoo', figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        mpf.plot(df, type='candle', ax=ax, addplot=apds, volume=True, returnfig=True)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def run_agent():
    config = collect_config()
    if not config:
        return
    try:
        agent = FinRLTradingAgent(stock_list=config["stocks"], initial_amount=config["initial_balance"])
        # Train model on training period
        trained_model, train_df = agent.train_and_trade(
            start_date=config["train_start"],
            end_date=config["train_end"]
        )
        # Prepare test data
        test_start = config["test_start"]
        test_end = config["test_end"]
        test_df = agent.prepare_data(test_start, test_end)
        # Paper trade on test period
        account_value, actions = agent.paper_trade(trained_model, test_df)
        # Show results in GUI
        results_text.delete(1.0, tk.END)
        results_text.insert(tk.END, f"Final portfolio value: ${account_value.iloc[-1][0]:.2f}\n\n")
        results_text.insert(tk.END, f"First few rows of actions:\n{actions.head()}\n")
        # Store results for saving
        results_text.results_df = account_value
        # Plot
        for widget in chart_frame.winfo_children():
            widget.destroy()
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(account_value['date'], account_value['account_value'])
        ax.set_title("Portfolio Value Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Account Value")
        fig.tight_layout()
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    except Exception as e:
        messagebox.showerror("Agent Error", f"Error running agent: {e}")

def save_results_button():
    if hasattr(results_text, "results_df"):
        save_results(results_text.results_df)
    else:
        messagebox.showinfo("No Results", "No results to save yet.")

# Supported RL algorithms (edit as needed)
RL_ALGOS = ["ppo", "a2c", "ddpg", "sac", "td3"]

# GUI
root = tk.Tk()
root.title("FinRL Trading Agent GUI")

mainframe = ttk.Frame(root, padding="10")
mainframe.pack(fill=tk.BOTH, expand=True)

row = 0
tk.Label(mainframe, text="RL Algorithm:").grid(row=row, column=0, sticky=tk.E)
algo_var = tk.StringVar(value=RL_ALGOS[0])
algo_menu = ttk.Combobox(mainframe, textvariable=algo_var, values=RL_ALGOS, state='readonly')
algo_menu.grid(row=row, column=1, sticky=tk.W)

row += 1
tk.Label(mainframe, text="Stock Symbols:").grid(row=row, column=0, sticky=tk.E)
stocks_entry = ttk.Entry(mainframe, width=40)
stocks_entry.insert(0, "AAPL,MSFT,NVDA")
stocks_entry.grid(row=row, column=1, sticky=tk.W)

row += 1
tk.Label(mainframe, text="Train Start Date:").grid(row=row, column=0, sticky=tk.E)
train_start = DateEntry(mainframe, width=12, background='darkblue', foreground='white', borderwidth=2)
train_start.set_date(datetime.now().replace(year=datetime.now().year-2))
train_start.grid(row=row, column=1, sticky=tk.W)

row += 1
tk.Label(mainframe, text="Train End Date:").grid(row=row, column=0, sticky=tk.E)
train_end = DateEntry(mainframe, width=12, background='darkblue', foreground='white', borderwidth=2)
train_end.set_date(datetime.now())
train_end.grid(row=row, column=1, sticky=tk.W)

row += 1
tk.Label(mainframe, text="Test Start Date:").grid(row=row, column=0, sticky=tk.E)
test_start = DateEntry(mainframe, width=12, background='darkblue', foreground='white', borderwidth=2)
test_start.set_date(datetime.now().replace(year=datetime.now().year-1))
test_start.grid(row=row, column=1, sticky=tk.W)

row += 1
tk.Label(mainframe, text="Test End Date:").grid(row=row, column=0, sticky=tk.E)
test_end = DateEntry(mainframe, width=12, background='darkblue', foreground='white', borderwidth=2)
test_end.set_date(datetime.now())
test_end.grid(row=row, column=1, sticky=tk.W)

row += 1
tk.Label(mainframe, text="Initial Balance:").grid(row=row, column=0, sticky=tk.E)
balance_entry = ttk.Entry(mainframe, width=12)
balance_entry.insert(0, "1000000")
balance_entry.grid(row=row, column=1, sticky=tk.W)

row += 1
tk.Label(mainframe, text="Transaction Cost:").grid(row=row, column=0, sticky=tk.E)
cost_entry = ttk.Entry(mainframe, width=12)
cost_entry.insert(0, "0.001")
cost_entry.grid(row=row, column=1, sticky=tk.W)

row += 1
tk.Label(mainframe, text="Max Shares/Trade (hmax):").grid(row=row, column=0, sticky=tk.E)
hmax_entry = ttk.Entry(mainframe, width=12)
hmax_entry.insert(0, "100")
hmax_entry.grid(row=row, column=1, sticky=tk.W)

row += 1
tk.Label(mainframe, text="Reward Scaling:").grid(row=row, column=0, sticky=tk.E)
reward_entry = ttk.Entry(mainframe, width=12)
reward_entry.insert(0, "1e-4")
reward_entry.grid(row=row, column=1, sticky=tk.W)

row += 1
tk.Label(mainframe, text="Total Timesteps:").grid(row=row, column=0, sticky=tk.E)
steps_entry = ttk.Entry(mainframe, width=12)
steps_entry.insert(0, "50000")
steps_entry.grid(row=row, column=1, sticky=tk.W)

row += 1
tk.Label(mainframe, text="Random Seed (optional):").grid(row=row, column=0, sticky=tk.E)
seed_entry = ttk.Entry(mainframe, width=12)
seed_entry.grid(row=row, column=1, sticky=tk.W)

row += 1
tk.Label(mainframe, text="Technical Indicators:").grid(row=row, column=0, sticky=tk.E)
tech_listbox = tk.Listbox(mainframe, selectmode=tk.MULTIPLE, exportselection=0, height=6)
for i, ind in enumerate(TECH_INDICATORS):
    tech_listbox.insert(i, ind)
tech_listbox.grid(row=row, column=1, sticky=tk.W)

row += 1
btn_frame = ttk.Frame(mainframe)
btn_frame.grid(row=row, column=0, columnspan=2, pady=8)
ttk.Button(btn_frame, text="Chart Stocks", command=fetch_and_plot).grid(row=0, column=0, padx=2)
ttk.Button(btn_frame, text="Run Agent", command=run_agent).grid(row=0, column=1, padx=2)
ttk.Button(btn_frame, text="Save Results", command=save_results_button).grid(row=0, column=2, padx=2)
ttk.Button(btn_frame, text="Save Config", command=save_config).grid(row=0, column=3, padx=2)
ttk.Button(btn_frame, text="Load Config", command=load_config).grid(row=0, column=4, padx=2)

row += 1
tk.Label(mainframe, text="Results:").grid(row=row, column=0, sticky=tk.NE)
results_text = tk.Text(mainframe, width=60, height=7)
results_text.grid(row=row, column=1, sticky=tk.W)

# Chart area
chart_frame = tk.Frame(root)
chart_frame.pack(fill=tk.BOTH, expand=True)

root.mainloop()

