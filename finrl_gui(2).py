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
        # Open strategy creation window
        strategy_window = tk.Toplevel(self.root)
        strategy_window.title("Create New Strategy")
        # Add strategy creation widgets here

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
