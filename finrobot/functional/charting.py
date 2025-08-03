import os
import numpy as np
import mplfinance as mpf
import pandas as pd
from matplotlib import pyplot as plt
from typing import Annotated, List, Tuple
from pandas import DateOffset
from datetime import datetime, timedelta
from ..data_source.yfinance_utils import YFinanceUtils

class MplFinanceUtils:

    @staticmethod
    def plot_stock_price_chart(
        ticker_symbols: Annotated[
            List[str], "List of ticker symbols (e.g., ['AAPL', 'MSFT'])"
        ],
        start_date: Annotated[
            str, "Start date of the historical data in 'YYYY-MM-DD' format"
        ],
        end_date: Annotated[
            str, "End date of the historical data in 'YYYY-MM-DD' format"
        ],
        save_dir: Annotated[str, "Directory where the plots should be saved"],
        indicators: Annotated[
            List[str],
            "List of indicators to plot: 'bollinger', 'rsi', 'macd', 'custom'. Default: all.",
        ] = None,
        custom_pine: Annotated[
            str, "Pine Script code for a custom indicator (optional)"
        ] = None,
        verbose: Annotated[
            bool, "Whether to print stock data to console. Default to False."
        ] = False,
        type: Annotated[
            str,
            "Type of the plot, e.g., 'candle','ohlc','line'. Default to 'candle'",
        ] = "candle",
        style: Annotated[
            str,
            "Style of the plot, e.g., 'default','classic','yahoo'. Default to 'default'.",
        ] = "default",
        mav: Annotated[
            int | List[int] | Tuple[int, ...] | None,
            "Moving average window(s) to plot on the chart. Default to None.",
        ] = None,
        show_nontrading: Annotated[
            bool, "Whether to show non-trading days on the chart. Default to False."
        ] = False,
    ) -> List[str]:
        """
        Plot stock price charts for a list of tickers with selected indicators and save to files.
        Returns a list of file paths for the saved charts.
        """
        if indicators is None:
            indicators = ['bollinger', 'rsi', 'macd']

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        saved_files = []

        for ticker_symbol in ticker_symbols:
            stock_data = YFinanceUtils.get_stock_data(ticker_symbol, start_date, end_date)
            if verbose:
                print(stock_data.to_string())

            addplots = []
            panel_count = 1  # main panel is 0

            # Bollinger Bands
            if 'bollinger' in indicators:
                close = stock_data['Close']
                ma = close.rolling(window=20).mean()
                std = close.rolling(window=20).std()
                upper = ma + (std * 2)
                lower = ma - (std * 2)
                addplots += [
                    mpf.make_addplot(upper, color='g'),
                    mpf.make_addplot(lower, color='r')
                ]

            # RSI
            if 'rsi' in indicators:
                delta = stock_data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                addplots.append(mpf.make_addplot(rsi, panel=panel_count, color='b', ylabel='RSI'))
                panel_count += 1

            # MACD
            if 'macd' in indicators:
                exp12 = stock_data['Close'].ewm(span=12, adjust=False).mean()
                exp26 = stock_data['Close'].ewm(span=26, adjust=False).mean()
                macd = exp12 - exp26
                signal = macd.ewm(span=9, adjust=False).mean()
                addplots.append(mpf.make_addplot(macd, panel=panel_count, color='b', ylabel='MACD'))
                addplots.append(mpf.make_addplot(signal, panel=panel_count, color='r'))
                panel_count += 1

            # Custom Pine Script (placeholder)
            if 'custom' in indicators and custom_pine:
                # NOTE: Pine Script cannot be executed natively in Python.
                # You could use a Pine-to-Python converter or TradingView API.
                # Here, we just note that the code was received.
                print(f"Custom Pine Script for {ticker_symbol}:\n{custom_pine}")
                # Optionally, you could display a message on the chart or log it.

            save_path = os.path.join(save_dir, f"{ticker_symbol}_chart.png")
            params = {
                "type": type,
                "style": style,
                "title": f"{ticker_symbol} {type} chart",
                "ylabel": "Price",
                "volume": True,
                "ylabel_lower": "Volume",
                "mav": mav,
                "show_nontrading": show_nontrading,
                "savefig": save_path,
                "addplot": addplots if addplots else None,
                "panel_ratios": tuple([3] + [1]*(panel_count-1)) if panel_count > 1 else None,
            }
            filtered_params = {k: v for k, v in params.items() if v is not None}
            mpf.plot(stock_data, **filtered_params)
            saved_files.append(save_path)

        return saved_files
