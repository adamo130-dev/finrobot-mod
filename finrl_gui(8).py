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
        # Technical Analysis Frame
        tech_frame = ttk.LabelFrame(self.screener_frame, text="Technical Indicators")
        tech_frame.pack(fill="x", padx=5, pady=5)
        
        # RSI
        ttk.Label(tech_frame, text="RSI Range:").grid(row=0, column=0, padx=5)
        self.rsi_min = ttk.Entry(tech_frame, width=8)
        self.rsi_max = ttk.Entry(tech_frame, width=8)
        self.rsi_min.grid(row=0, column=1, padx=2)
        ttk.Label(tech_frame, text="-").grid(row=0, column=2)
        self.rsi_max.grid(row=0, column=3, padx=2)
        
        # MACD
        ttk.Label(tech_frame, text="MACD Signal:").grid(row=1, column=0, padx=5)
        self.macd_signal = ttk.Combobox(tech_frame, values=["Bullish", "Bearish"])
        self.macd_signal.grid(row=1, column=1, columnspan=3, padx=5)
        
        # Bollinger Bands
        ttk.Label(tech_frame, text="BB Position:").grid(row=2, column=0, padx=5)
        self.bb_position = ttk.Combobox(tech_frame, values=["Above Upper", "Below Lower", "Middle"])
        self.bb_position.grid(row=2, column=1, columnspan=3, padx=5)

        # Fundamental Analysis Frame
        fund_frame = ttk.LabelFrame(self.screener_frame, text="Fundamental Metrics")
        fund_frame.pack(fill="x", padx=5, pady=5)
        
        # Market Cap
        ttk.Label(fund_frame, text="Market Cap (M):").grid(row=0, column=0, padx=5)
        self.market_cap = ttk.Entry(fund_frame, width=15)
        self.market_cap.grid(row=0, column=1, padx=5)
        
        # P/E Ratio
        ttk.Label(fund_frame, text="P/E Range:").grid(row=1, column=0, padx=5)
        self.pe_min = ttk.Entry(fund_frame, width=8)
        self.pe_max = ttk.Entry(fund_frame, width=8)
        self.pe_min.grid(row=1, column=1, padx=2)
        ttk.Label(fund_frame, text="-").grid(row=1, column=2)
        self.pe_max.grid(row=1, column=3, padx=2)
        
        # Revenue Growth
        ttk.Label(fund_frame, text="Min Revenue Growth (%):").grid(row=2, column=0, padx=5)
        self.rev_growth = ttk.Entry(fund_frame, width=8)
        self.rev_growth.grid(row=2, column=1, padx=5)

        # Risk Metrics Frame
        risk_frame = ttk.LabelFrame(self.screener_frame, text="Risk Metrics")
        risk_frame.pack(fill="x", padx=5, pady=5)
        
        # Beta
        ttk.Label(risk_frame, text="Beta Range:").grid(row=0, column=0, padx=5)
        self.beta_min = ttk.Entry(risk_frame, width=8)
        self.beta_max = ttk.Entry(risk_frame, width=8)
        self.beta_min.grid(row=0, column=1, padx=2)
        ttk.Label(risk_frame, text="-").grid(row=0, column=2)
        self.beta_max.grid(row=0, column=3, padx=2)
        
        # Volatility
        ttk.Label(risk_frame, text="Max Volatility (%):").grid(row=1, column=0, padx=5)
        self.max_vol = ttk.Entry(risk_frame, width=8)
        self.max_vol.grid(row=1, column=1, padx=5)
        
        # Sharpe Ratio
        ttk.Label(risk_frame, text="Min Sharpe Ratio:").grid(row=2, column=0, padx=5)
        self.min_sharpe = ttk.Entry(risk_frame, width=8)
        self.min_sharpe.grid(row=2, column=1, padx=5)

        # Volume Analysis Frame
        vol_frame = ttk.LabelFrame(self.screener_frame, text="Volume Analysis")
        vol_frame.pack(fill="x", padx=5, pady=5)
        
        # Average Volume
        ttk.Label(vol_frame, text="Min Avg Volume (M):").grid(row=0, column=0, padx=5)
        self.min_volume = ttk.Entry(vol_frame, width=15)
        self.min_volume.grid(row=0, column=1, padx=5)
        
        # Volume Trend
        ttk.Label(vol_frame, text="Volume Trend:").grid(row=1, column=0, padx=5)
        self.volume_trend = ttk.Combobox(vol_frame, values=["Increasing", "Decreasing", "Stable"])
        self.volume_trend.grid(row=1, column=1, padx=5)

        # Run Button
        ttk.Button(self.screener_frame, text="Run Screener", 
                  command=self.run_screener).pack(side="right", padx=5, pady=10)

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
        from finrobot.functional.quantitative import BackTraderUtils
        from finrobot.data_source import FMPUtils, YFinanceUtils
        import pandas as pd
        from datetime import datetime, timedelta

        # Get universe of stocks
        sp500 = YFinanceUtils.get_stock_info("^GSPC")
        universe = sp500.history(period="1d").index

        screened_stocks = []
        strategy_name = self.strategy_var.get()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)

        for symbol in universe:
            try:
                # Technical Analysis
                stock_data = YFinanceUtils.get_stock_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if stock_data.empty:
                    continue

                # RSI Filter
                if self.rsi_min.get() and self.rsi_max.get():
                    rsi_val = self.calculate_rsi(stock_data['Close'])
                    if not (float(self.rsi_min.get()) <= rsi_val <= float(self.rsi_max.get())):
                        continue

                # Volume Filter
                if self.min_volume.get():
                    avg_volume = stock_data['Volume'].mean() / 1e6  # Convert to millions
                    if avg_volume < float(self.min_volume.get()):
                        continue

                # Fundamental Analysis
                metrics = FMPUtils.get_key_metrics(symbol)
                if metrics is None:
                    continue

                # Market Cap Filter
                if self.market_cap.get() and metrics['marketCap'] < float(self.market_cap.get()) * 1e6:
                    continue

                # P/E Filter
                if self.pe_min.get() and self.pe_max.get() and metrics['peRatio'] > 0:
                    if not (float(self.pe_min.get()) <= metrics['peRatio'] <= float(self.pe_max.get())):
                        continue

                # Revenue Growth Filter
                if self.rev_growth.get() and metrics['revenueGrowth'] < float(self.rev_growth.get()) / 100:
                    continue

                # Risk Analysis
                backtest_results = BackTraderUtils.back_test(
                    symbol,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    strategy=strategy_name,
                    cash=100000
                )

                sharpe_ratio = backtest_results["Sharpe Ratio"]["sharperatio"]
                returns = backtest_results["Returns"]["rtot"]
                max_drawdown = abs(backtest_results["DrawDown"]["max"]["drawdown"])

                # Sharpe Ratio Filter
                if self.min_sharpe.get() and sharpe_ratio < float(self.min_sharpe.get()):
                    continue

                # Beta Filter
                if self.beta_min.get() and self.beta_max.get():
                    beta = self.calculate_beta(stock_data['Close'], YFinanceUtils.get_stock_data('^GSPC', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))['Close'])
                    if not (float(self.beta_min.get()) <= beta <= float(self.beta_max.get())):
                        continue

                screened_stocks.append({
                    'symbol': symbol,
                    'sharpe': sharpe_ratio,
                    'returns': returns,
                    'max_drawdown': max_drawdown,
                    'metrics': metrics
                })

            except Exception as e:
                continue

        # Sort and display results
        if screened_stocks:
            screened_stocks = pd.DataFrame(screened_stocks)
            screened_stocks = screened_stocks.sort_values(by=['sharpe'], ascending=False).head(10)

            self.results_text.delete(1.0, "end")
            self.results_text.insert("end", "Top Screened Stocks:\n\n")
            for _, row in screened_stocks.iterrows():
                self.results_text.insert("end",
                    f"Symbol: {row['symbol']}\n"
                    f"Sharpe Ratio: {row['sharpe']:.2f}\n"
                    f"Returns: {row['returns']:.2%}\n"
                    f"Max Drawdown: {row['max_drawdown']:.2%}\n"
                    f"Market Cap: ${row['metrics']['marketCap']:,.0f}\n"
                    f"P/E Ratio: {row['metrics']['peRatio']:.2f}\n"
                    "-------------------\n"
                )
        else:
            self.results_text.delete(1.0, "end")
            self.results_text.insert("end", "No stocks matched the criteria")

        return screened_stocks.index.tolist() if not screened_stocks.empty else []

    def calculate_rsi(self, prices, periods=14):
        import numpy as np
        
        deltas = np.diff(prices)
        seed = deltas[:periods+1]
        up = seed[seed >= 0].sum()/periods
        down = -seed[seed < 0].sum()/periods
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:periods] = 100. - 100./(1. + rs)

        for i in range(periods, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (periods - 1) + upval) / periods
            down = (down * (periods - 1) + downval) / periods
            rs = up/down
            rsi[i] = 100. - 100./(1. + rs)

        return rsi[-1]

    def calculate_beta(self, stock_returns, market_returns):
        import numpy as np
        
        stock_returns = np.diff(stock_returns) / stock_returns[:-1]
        market_returns = np.diff(market_returns) / market_returns[:-1]
        
        covariance = np.cov(stock_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        
        return covariance / market_variance if market_variance != 0 else 1

    def start_paper_trading(self):
        from datetime import datetime, timedelta
        
        # Get selected stocks from screener
        selected_stocks = self.get_selected_stocks()
        if not selected_stocks:
            messagebox.showwarning("Warning", "Please run screener first to select stocks")
            return
            
        # Initialize trading agent
        agent = FinRLTradingAgent(stock_list=selected_stocks, initial_amount=100000)
        
        # Prepare training data (last 2 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        train_data = agent.prepare_data(start_date.strftime('%Y-%m-%d'), 
                                      end_date.strftime('%Y-%m-%d'))
        
        if train_data.empty:
            messagebox.showerror("Error", "Failed to fetch training data")
            return
            
        # Train the agent
        try:
            model = agent.train_agent(train_data)
            self.results_text.insert("end", "Model training completed\n")
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            return
            
        # Paper trade (using last 30 days)
        test_start = end_date - timedelta(days=30)
        test_data = agent.prepare_data(test_start.strftime('%Y-%m-%d'),
                                     end_date.strftime('%Y-%m-%d'))
                                     
        if test_data.empty:
            messagebox.showerror("Error", "Failed to fetch test data")
            return
            
        try:
            account_value, actions = agent.paper_trade(model, test_data)
            self.display_trading_results(account_value, actions)
            
            # Save results
            results_dir = "trading_results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            account_value.to_csv(f"{results_dir}/account_value_{timestamp}.csv")
            actions.to_csv(f"{results_dir}/actions_{timestamp}.csv")
            
            self.results_text.insert("end", f"\nResults saved to {results_dir}/\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Paper trading failed: {str(e)}")

    def save_results(self):
        # Save analysis and trading results
        pass

    def get_selected_stocks(self):
        # Get list of stocks from screener results
        return ["AAPL", "MSFT", "GOOGL"]  # Placeholder

    def display_trading_results(self, account_value, actions):
        # Clear previous results
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()
            
        # Create figures frame
        fig_frame = ttk.Frame(self.analysis_frame)
        fig_frame.pack(fill="both", expand=True)
        
        # Portfolio value plot
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot account value
        account_value['account_value'].plot(ax=ax1)
        ax1.set_title("Portfolio Value Over Time")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Account Value ($)")
        ax1.grid(True)
        
        # Plot returns
        returns = account_value['account_value'].pct_change()
        cumulative_returns = (1 + returns).cumprod()
        cumulative_returns.plot(ax=ax2)
        ax2.set_title("Cumulative Returns")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Return")
        ax2.grid(True)
        
        fig1.tight_layout()
        
        # Embed plot
        canvas1 = FigureCanvasTkAgg(fig1, master=fig_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        
        # Create text frame for metrics
        text_frame = ttk.Frame(self.analysis_frame)
        text_frame.pack(fill="both", expand=True)
        
        # Display metrics
        self.results_text = tk.Text(text_frame, height=10)
        self.results_text.pack(fill="both", expand=True)
        
        # Calculate metrics
        total_return = (account_value['account_value'].iloc[-1] / account_value['account_value'].iloc[0] - 1) * 100
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        max_drawdown = (account_value['account_value'] / account_value['account_value'].cummax() - 1).min() * 100
        
        metrics_text = f"""Trading Results Summary:
        
Total Return: {total_return:.2f}%
Sharpe Ratio: {sharpe_ratio:.2f}
Maximum Drawdown: {max_drawdown:.2f}%

Recent Trading Actions:
{actions.tail().to_string()}
        """
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, metrics_text)

def main():
    root = tk.Tk()
    app = TradingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
