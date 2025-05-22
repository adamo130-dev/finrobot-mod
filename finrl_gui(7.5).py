import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
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

        # To store the current shortlist for performance-based screening
        self.filtered_stocks = []

    def setup_strategy_section(self):
        # Strategy selection
        strategies = ["Moving Average Crossover", "RSI Strategy", "MACD Strategy", "Custom Strategy"]
        self.strategy_var = tk.StringVar(value=strategies[0])
        strategy_menu = ttk.OptionMenu(self.strategy_frame, self.strategy_var, *strategies)
        strategy_menu.pack(side="left", padx=5)

        ttk.Button(self.strategy_frame, text="Create New Strategy", 
                  command=self.create_strategy).pack(side="right", padx=5)

    def setup_screener_section(self):
        # --- Fundamental filters ---
        ttk.Label(self.screener_frame, text="Market Cap (M):").pack(side="left", padx=5)
        self.market_cap = ttk.Entry(self.screener_frame, width=10)
        self.market_cap.pack(side="left", padx=2)
        self.market_cap.insert(0, "10000")  # Default 10B

        ttk.Label(self.screener_frame, text="P/E Max:").pack(side="left", padx=5)
        self.pe_ratio = ttk.Entry(self.screener_frame, width=7)
        self.pe_ratio.pack(side="left", padx=2)
        self.pe_ratio.insert(0, "40")

        ttk.Label(self.screener_frame, text="Min. Volume (M):").pack(side="left", padx=5)
        self.min_volume = ttk.Entry(self.screener_frame, width=10)
        self.min_volume.pack(side="left", padx=2)
        self.min_volume.insert(0, "1")

        # --- Technical indicator filters ---
        ttk.Label(self.screener_frame, text="RSI Min:").pack(side="left", padx=5)
        self.rsi_min = ttk.Entry(self.screener_frame, width=5)
        self.rsi_min.pack(side="left", padx=2)
        self.rsi_min.insert(0, "30")

        ttk.Label(self.screener_frame, text="RSI Max:").pack(side="left", padx=5)
        self.rsi_max = ttk.Entry(self.screener_frame, width=5)
        self.rsi_max.pack(side="left", padx=2)
        self.rsi_max.insert(0, "70")

        self.sma_var = tk.BooleanVar()
        ttk.Checkbutton(self.screener_frame, text="SMA", variable=self.sma_var).pack(side="left", padx=5)

        self.macd_var = tk.BooleanVar()
        ttk.Checkbutton(self.screener_frame, text="MACD", variable=self.macd_var).pack(side="left", padx=5)

        self.bb_var = tk.BooleanVar()
        ttk.Checkbutton(self.screener_frame, text="Bollinger Bands", variable=self.bb_var).pack(side="left", padx=5)

        # Run Screener Button (2-stage screening)
        ttk.Button(self.screener_frame, text="Quick Screener", 
                  command=self.run_quick_screener).pack(side="right", padx=5)
        ttk.Button(self.screener_frame, text="Strategy Screener", 
                  command=self.run_strategy_screener).pack(side="right", padx=5)

    def setup_analysis_section(self):
        # Results display
        self.results_text = tk.Text(self.analysis_frame, height=12)
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
            # Save the strategy configuration (stub: just sets current selection)
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
            self.strategy_var.set(name_entry.get())
            strategy_window.destroy()
            messagebox.showinfo("Success", "Strategy saved successfully!")

        save_button = ttk.Button(strategy_window, text="Save Strategy", command=save_strategy)
        save_button.pack(pady=10)

    def run_quick_screener(self):
        """
        Fast filter based on user technical/fundamental criteria.
        """
        self.results_text.delete(1.0, "end")
        self.results_text.insert("end", "Running quick screener...\n")

        # Get S&P 500 tickers (or a fixed list for demo)
        try:
            sp500 = yf.Ticker("^GSPC")
            tickers = sp500.constituents if hasattr(sp500, "constituents") else [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "BRK-B", "JPM", "V"
            ]
        except Exception:
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "BRK-B", "JPM", "V"]

        # Reduce to first 30 for demo/performance
        tickers = tickers[:30]
        screened = []

        for symbol in tickers:
            try:
                info = yf.Ticker(symbol).fast_info
                hist = yf.download(symbol, period="6mo", progress=False)
                if hist.empty or len(hist) < 50:
                    continue

                # Calculate indicators
                sma20 = hist['Close'].rolling(window=20).mean()
                rsi = self._calculate_rsi(hist['Close'], period=14)
                avg_volume = hist['Volume'].mean() / 1e6
                pe_ratio = float(info.get('trailingPE', float('inf')))
                market_cap = float(info.get('marketCap', 0)) / 1e6  # in millions

                # Apply filters
                if self.market_cap.get() and market_cap < float(self.market_cap.get()):
                    continue
                if self.pe_ratio.get() and pe_ratio > float(self.pe_ratio.get()):
                    continue
                if self.min_volume.get() and avg_volume < float(self.min_volume.get()):
                    continue
                if self.rsi_min.get() and rsi[-1] < float(self.rsi_min.get()):
                    continue
                if self.rsi_max.get() and rsi[-1] > float(self.rsi_max.get()):
                    continue

                # SMA filter
                if self.sma_var.get():
                    sma50 = hist['Close'].rolling(window=50).mean()
                    if sma20.iloc[-1] <= sma50.iloc[-1]:
                        continue

                # MACD filter
                if self.macd_var.get():
                    macd, macd_signal = self._calculate_macd(hist['Close'])
                    if macd[-1] <= macd_signal[-1]:
                        continue

                # Bollinger Bands filter
                if self.bb_var.get():
                    sma = hist['Close'].rolling(window=20).mean()
                    std = hist['Close'].rolling(window=20).std()
                    upper = sma + (2 * std)
                    lower = sma - (2 * std)
                    price = hist['Close'].iloc[-1]
                    if not (lower.iloc[-1] < price < upper.iloc[-1]):
                        continue

                screened.append({
                    'symbol': symbol,
                    'market_cap': market_cap,
                    'pe_ratio': pe_ratio,
                    'volume': avg_volume,
                    'rsi': rsi[-1]
                })

            except Exception as e:
                continue

        # Save shortlist for next stage
        self.filtered_stocks = [x["symbol"] for x in screened]
        self.results_text.delete(1.0, "end")
        if screened:
            self.results_text.insert("end", f"{len(screened)} stocks passed quick screening.\n\n")
            self.results_text.insert("end", f"{'Symbol':<8} {'MktCap(M)':<12} {'P/E':<10} {'Vol(M)':<10} {'RSI':<7}\n")
            self.results_text.insert("end", "-"*50+"\n")
            for stock in screened:
                self.results_text.insert(
                    "end",
                    f"{stock['symbol']:<8} {stock['market_cap']:<12.1f} {stock['pe_ratio']:<10.2f} {stock['volume']:<10.2f} {stock['rsi']:<7.2f}\n"
                )
        else:
            self.results_text.insert("end", "No stocks matched quick screener criteria.\n")

    def run_strategy_screener(self):
        """
        Takes the output of run_quick_screener (self.filtered_stocks) and runs a backtest-driven strategy screen.
        """
        self.results_text.delete(1.0, "end")
        self.results_text.insert("end", "Running strategy-based screener on shortlist...\n")

        if not self.filtered_stocks:
            self.results_text.insert("end", "Please run the Quick Screener first!\n")
            return

        strategy_name = self.strategy_var.get()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 6 months

        screened_stocks = []

        for symbol in self.filtered_stocks:
            try:
                # For demo: just call BackTraderUtils.back_test with stub params
                results = BackTraderUtils.back_test(
                    symbol,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    strategy=strategy_name,
                    cash=100000
                )
                # Analyze results (customize thresholds per strategy)
                if strategy_name == "Moving Average Crossover":
                    if results.get("Sharpe Ratio", {}).get("sharperatio", 0) > 1.0:
                        screened_stocks.append({
                            'symbol': symbol,
                            'sharpe': results["Sharpe Ratio"]["sharperatio"],
                            'returns': results["Returns"]["rtot"]
                        })
                elif strategy_name == "RSI Strategy":
                    if (results.get("Returns", {}).get("rtot", 0) > 0.1 and
                        results.get("DrawDown", {}).get("max", {}).get("drawdown", 1) < 0.15):
                        screened_stocks.append({
                            'symbol': symbol,
                            'returns': results["Returns"]["rtot"],
                            'max_drawdown': results["DrawDown"]["max"]["drawdown"]
                        })
                # Add more strategies as needed

            except Exception:
                continue

        # Display results
        self.results_text.delete(1.0, "end")
        if screened_stocks:
            df = pd.DataFrame(screened_stocks)
            df = df.sort_values(by=['returns'], ascending=False).head(10)
            self.results_text.insert("end", f"Top {len(df)} stocks by {strategy_name} strategy performance:\n\n")
            for _, row in df.iterrows():
                summary = f"Symbol: {row['symbol']}, Returns: {row['returns']:.2%}"
                if "sharpe" in row:
                    summary += f", Sharpe: {row['sharpe']:.2f}"
                if "max_drawdown" in row:
                    summary += f", Max Drawdown: {row['max_drawdown']:.2%}"
                self.results_text.insert("end", summary + "\n")
            # Update filtered_stocks for trading
            self.filtered_stocks = list(df['symbol'])
        else:
            self.results_text.insert("end", "No stocks passed the performance-based screen.\n")

    def start_paper_trading(self):
        # Initialize FinRL agent and environment
        selected_stocks = self.get_selected_stocks()
        agent = FinRLTradingAgent(stock_list=selected_stocks, initial_amount=100000)
        # Placeholder for demonstration: real implementation should provide train_data/test_data
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()
        model = agent.train_agent(train_data)
        account_value, actions = agent.paper_trade(model, test_data)
        self.display_trading_results(account_value, actions)

    def save_results(self):
        # Save analysis and trading results
        messagebox.showinfo("Save", "Results saved (stub).")

    def get_selected_stocks(self):
        # Use filtered_stocks if available, else default
        return getattr(self, "filtered_stocks", ["AAPL", "MSFT", "GOOGL"])

    def display_trading_results(self, account_value, actions):
        # Plotting the account value
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(account_value.index, account_value['account_value'])
        ax.set_title("Portfolio Value Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Account Value")
        fig.tight_layout()

        # Embedding the plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.analysis_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Displaying action summary in the text widget
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, "Trading Actions Summary:\n")
        self.results_text.insert(tk.END, str(actions.head()))

    # --- Helper indicator functions ---
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        avg_gain = up.rolling(window=period).mean()
        avg_loss = down.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal

def main():
    root = tk.Tk()
    app = TradingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()