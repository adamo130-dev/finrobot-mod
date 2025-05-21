import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrobot.functional.quantitative import BackTraderUtils
from finrobot.agents.finrl_trading_agent import FinRLTradingAgent

class TradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trading Strategy Platform")

        # Create main frames
        self.strategy_frame = ttk.LabelFrame(root, text="Strategy Selection")
        self.strategy_frame.pack(fill="x", padx=5, pady=5)

        self.screener_frame = ttk.LabelFrame(root, text="Stock Screener")
        self.screener_frame.pack(fill="x", padx=5, pady=5)

        self.analysis_frame = ttk.LabelFrame(root, text="Analysis Results")
        self.analysis_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.setup_strategy_section()
        self.setup_screener_section()
        self.setup_analysis_section()

    def setup_strategy_section(self):
        # Strategy selection
        strategies = ["Moving Average Crossover", "RSI Strategy", "MACD Strategy", "Custom Strategy"]
        self.strategy_var = tk.StringVar(value=strategies[0])
        strategy_menu = ttk.OptionMenu(self.strategy_frame, self.strategy_var, *strategies)
        strategy_menu.pack(side="left", padx=5)

        ttk.Button(self.strategy_frame, text="Create New Strategy", 
                  command=self.create_strategy).pack(side="right", padx=5)

    def setup_screener_section(self):
        # Screening criteria
        ttk.Label(self.screener_frame, text="Market Cap:").pack(side="left", padx=5)
        self.market_cap = ttk.Entry(self.screener_frame, width=15)
        self.market_cap.pack(side="left", padx=5)

        ttk.Button(self.screener_frame, text="Run Screener", 
                  command=self.run_screener).pack(side="right", padx=5)

    def setup_analysis_section(self):
        # Results display
        self.results_text = tk.Text(self.analysis_frame, height=10)
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)

        button_frame = ttk.Frame(self.analysis_frame)
        button_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(button_frame, text="Start Paper Trading", 
                  command=self.start_paper_trading).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Save Results", 
                  command=self.save_results).pack(side="right", padx=5)

    def create_strategy(self):
        strategy_window = tk.Toplevel(self.root)
        strategy_window.title("Create New Strategy")
        strategy_window.geometry("600x400")

        # Strategy name
        name_frame = ttk.Frame(strategy_window)
        name_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(name_frame, text="Strategy Name:").pack(side="left")
        name_entry = ttk.Entry(name_frame, width=30)
        name_entry.pack(side="left", padx=5)

        # Strategy parameters
        params_frame = ttk.LabelFrame(strategy_window, text="Parameters")
        params_frame.pack(fill="x", padx=5, pady=5)

        # Technical indicators
        indicator_var = tk.StringVar(value="SMA")
        ttk.Label(params_frame, text="Indicator:").grid(row=0, column=0, padx=5, pady=5)
        indicator_menu = ttk.OptionMenu(params_frame, indicator_var, "SMA", "SMA", "RSI", "MACD")
        indicator_menu.grid(row=0, column=1, padx=5, pady=5)

        # Timeframes
        timeframe_var = tk.StringVar(value="Daily")
        ttk.Label(params_frame, text="Timeframe:").grid(row=1, column=0, padx=5, pady=5)
        timeframe_menu = ttk.OptionMenu(params_frame, timeframe_var, "Daily", "Daily", "Weekly", "Monthly")
        timeframe_menu.grid(row=1, column=1, padx=5, pady=5)

        # Parameters specific to SMA
        sma_frame = ttk.Frame(params_frame)
        sma_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        ttk.Label(sma_frame, text="Fast Period:").grid(row=0, column=0, padx=5)
        fast_period = ttk.Entry(sma_frame, width=10)
        fast_period.insert(0, "10")
        fast_period.grid(row=0, column=1, padx=5)
        
        ttk.Label(sma_frame, text="Slow Period:").grid(row=1, column=0, padx=5)
        slow_period = ttk.Entry(sma_frame, width=10)
        slow_period.insert(0, "30")
        slow_period.grid(row=1, column=1, padx=5)

        # Rules frame
        rules_frame = ttk.LabelFrame(strategy_window, text="Trading Rules")
        rules_frame.pack(fill="x", padx=5, pady=5)
        
        # Entry rules
        entry_label = ttk.Label(rules_frame, text="Entry Rule:")
        entry_label.pack(anchor="w", padx=5, pady=2)
        entry_text = tk.Text(rules_frame, height=3, width=50)
        entry_text.insert("1.0", "Fast SMA crosses above Slow SMA")
        entry_text.pack(padx=5, pady=2)

        # Exit rules
        exit_label = ttk.Label(rules_frame, text="Exit Rule:")
        exit_label.pack(anchor="w", padx=5, pady=2)
        exit_text = tk.Text(rules_frame, height=3, width=50)
        exit_text.insert("1.0", "Fast SMA crosses below Slow SMA")
        exit_text.pack(padx=5, pady=2)

        def save_strategy():
            # Here we would save the strategy configuration
            strategy_config = {
                "name": name_entry.get(),
                "indicator": indicator_var.get(),
                "timeframe": timeframe_var.get(),
                "parameters": {
                    "fast_period": int(fast_period.get()),
                    "slow_period": int(slow_period.get())
                },
                "entry_rule": entry_text.get("1.0", "end-1c"),
                "exit_rule": exit_text.get("1.0", "end-1c")
            }
            # You would typically save this to a file or database
            self.strategy_var.set(name_entry.get())
            strategy_window.destroy()
            messagebox.showinfo("Success", "Strategy saved successfully!")

        # Save button
        save_button = ttk.Button(strategy_window, text="Save Strategy", command=save_strategy)
        save_button.pack(pady=10)

    def run_screener(self):
        # Implement screening logic
        pass

    def start_paper_trading(self):
        # Initialize FinRL agent and environment
        selected_stocks = self.get_selected_stocks()
        agent = FinRLTradingAgent(stock_list=selected_stocks, initial_amount=100000)
        # Assuming 'train_data' and 'test_data' are defined elsewhere or obtained through other means
        # These would typically be created using YahooDownloader or similar data loading techniques.
        # For example:
        # data = YahooDownloader(start_date = TRAIN_START_DATE,
        #                         end_date = TRADE_END_DATE,
        #                         ticker_list = HSI_TICKER).fetch_data()
        # train_data = data[(data.date >= TRAIN_START_DATE) & (data.date < TRAIN_END_DATE)]
        # test_data = data[(data.date >= TRADE_START_DATE) & (data.date <= TRADE_END_DATE)]

        # Example placeholder data (replace with actual data loading)
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()
        
        model = agent.train_agent(train_data)

        # Start paper trading simulation
        account_value, actions = agent.paper_trade(model, test_data)
        self.display_trading_results(account_value, actions)

    def save_results(self):
        # Save analysis and trading results
        pass

    def get_selected_stocks(self):
        # Get list of stocks from screener results
        return ["AAPL", "MSFT", "GOOGL"]  # Placeholder

    def display_trading_results(self, account_value, actions):
        # Plotting the account value
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(account_value.index, account_value['account_value'])  # Assuming 'account_value' is a DataFrame
        ax.set_title("Portfolio Value Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Account Value")
        fig.tight_layout()

        # Embedding the plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.analysis_frame)  # Use analysis_frame as the parent
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Displaying action summary in the text widget
        self.results_text.delete("1.0", tk.END)  # Clear previous results
        self.results_text.insert(tk.END, "Trading Actions Summary:\n")
        self.results_text.insert(tk.END, str(actions.head()))  # Display first few actions

def main():
    root = tk.Tk()
    app = TradingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
