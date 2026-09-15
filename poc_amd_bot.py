"""
POC/AMD Strategy - Independent Trading Bot
============================================
Fully standalone from any other bot. Own Alpaca account, own schedule, own
trade log. The only thing it shares is the strategy logic in
poc_amd_strategy.py (pure signal math, no side effects).

WHY A SEPARATE ALPACA ACCOUNT:
Alpaca's paper accounts are free and unlimited. Using a SECOND paper account
(not the one your MA-crossover/momentum bot uses) means its equity curve is
a clean, unambiguous read on whether THIS strategy alone makes money - no
need to untangle combined P&L or shared positions.

Create a second paper account at https://app.alpaca.markets (Alpaca allows
multiple accounts), generate its API key/secret, and set them as:
  POC_AMD_ALPACA_API_KEY
  POC_AMD_ALPACA_SECRET_KEY

ENV VARS REQUIRED
------------------
  POC_AMD_ALPACA_API_KEY
  POC_AMD_ALPACA_SECRET_KEY
  POC_AMD_SYMBOLS         (optional, comma-separated, default below)

WHAT IT DOES EACH RUN
-----------------------
  1. Pull recent 30-min bars per symbol from Alpaca
  2. Run PocAmdStrategy.generate_signals()
  3. If the LATEST bar has a fresh BUY/SELL signal and there's no open
     position in that symbol on this account -> submit a bracket order
     (entry + stop-loss + take-profit in one order, Alpaca manages the exit)
  4. Append every submitted trade to trade_log.csv (for later comparison
     against your other strategy's log)

This script is meant to be run on a schedule (see poc_amd_bot.yml) - it does
NOT loop or sleep; each invocation does one pass and exits.
"""

import csv
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

from poc_amd_strategy import PocAmdStrategy, PocAmdParams

DEFAULT_SYMBOLS = ["AAPL", "SPY", "QQQ", "GOOGL", "AMD"]
TRADE_LOG_PATH = os.path.join(os.path.dirname(__file__), "trade_log.csv")
LOOKBACK_BARS = 400  # enough history for swing/volume-profile detection
QTY_PER_TRADE = 1    # bump this once you've validated results; start small


def get_clients():
    api_key = os.environ["POC_AMD_ALPACA_API_KEY"]
    secret_key = os.environ["POC_AMD_ALPACA_SECRET_KEY"]
    trading_client = TradingClient(api_key, secret_key, paper=True)
    data_client = StockHistoricalDataClient(api_key, secret_key)
    return trading_client, data_client


def fetch_bars(data_client: StockHistoricalDataClient, symbol: str) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)  # 30 days of 30-min bars comfortably covers LOOKBACK_BARS
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(30, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=DataFeed.IEX,  # paper/free accounts only have access to IEX, not SIP
    )
    bars = data_client.get_stock_bars(req).df
    if bars.empty:
        return pd.DataFrame()
    bars = bars.xs(symbol, level="symbol") if "symbol" in bars.index.names else bars
    bars = bars.rename(columns={"open": "open", "high": "high", "low": "low",
                                 "close": "close", "volume": "volume"})
    bars = bars[["open", "high", "low", "close", "volume"]].tail(LOOKBACK_BARS)
    return bars


def has_open_position(trading_client: TradingClient, symbol: str) -> bool:
    try:
        trading_client.get_open_position(symbol)
        return True
    except Exception:
        return False  # no position exists -> Alpaca raises, which is expected


def log_trade(row: dict):
    file_exists = os.path.isfile(TRADE_LOG_PATH)
    with open(TRADE_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def submit_bracket_order(trading_client: TradingClient, symbol: str, side: str,
                          stop_price: float, target_price: float, qty: int):
    order_side = OrderSide.BUY if side == "BUY" else OrderSide.SELL
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=order_side,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=round(target_price, 2)),
        stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
    )
    return trading_client.submit_order(order)


def run():
    trading_client, data_client = get_clients()
    symbols = os.environ.get("POC_AMD_SYMBOLS", ",".join(DEFAULT_SYMBOLS)).split(",")
    strategy = PocAmdStrategy(PocAmdParams())

    for symbol in symbols:
        symbol = symbol.strip()
        print(f"[{symbol}] fetching bars...")
        df = fetch_bars(data_client, symbol)
        if df.empty or len(df) < 50:
            print(f"[{symbol}] not enough data, skipping")
            continue

        signals = strategy.generate_signals(df)
        latest = signals.iloc[-1]

        if latest["signal"] not in ("BUY", "SELL"):
            print(f"[{symbol}] no signal this bar")
            continue

        if has_open_position(trading_client, symbol):
            print(f"[{symbol}] signal={latest['signal']} but position already open, skipping")
            continue

        try:
            order = submit_bracket_order(
                trading_client, symbol, latest["signal"],
                stop_price=float(latest["stop"]),
                target_price=float(latest["target"]),
                qty=QTY_PER_TRADE,
            )
            print(f"[{symbol}] submitted {latest['signal']} order id={order.id}")
            log_trade({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "side": latest["signal"],
                "entry_close": float(latest["close"]),
                "stop": float(latest["stop"]),
                "target": float(latest["target"]),
                "order_id": str(order.id),
            })
        except Exception as e:
            print(f"[{symbol}] order submission FAILED: {e}", file=sys.stderr)


if __name__ == "__main__":
    run()
