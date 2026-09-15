"""
POC/AMD Strategy - Performance Report
========================================
Reads trade_log.csv (written by poc_amd_bot.py) and cross-references each
logged trade against Alpaca's actual order records to determine whether it
hit its stop or target, and the realized P&L - then prints a summary.

Run manually any time:  python poc_amd_report.py

Requires the same env vars as poc_amd_bot.py:
  POC_AMD_ALPACA_API_KEY
  POC_AMD_ALPACA_SECRET_KEY
"""

import csv
import os

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrderByIdRequest

TRADE_LOG_PATH = os.path.join(os.path.dirname(__file__), "trade_log.csv")


def load_trade_log():
    if not os.path.isfile(TRADE_LOG_PATH):
        print("No trade_log.csv found yet - nothing to report.")
        return []
    with open(TRADE_LOG_PATH, newline="") as f:
        return list(csv.DictReader(f))


def evaluate_trade(trading_client: TradingClient, row: dict) -> dict:
    """Look up the bracket order by id and figure out its current state:
    OPEN (still live), TARGET (won), STOP (lost), or UNKNOWN (couldn't
    determine - e.g. order was cancelled or errored)."""
    order_id = row["order_id"]
    entry_price = float(row["entry_close"])
    side = row["side"]  # 'BUY' or 'SELL'
    symbol = row["symbol"]

    try:
        order = trading_client.get_order_by_id(
            order_id, filter=GetOrderByIdRequest(nested=True)
        )
    except Exception as e:
        return {**row, "status": "LOOKUP_FAILED", "pnl": None, "note": str(e)}

    legs = order.legs or []
    filled_leg = next((leg for leg in legs if str(leg.status) == "OrderStatus.FILLED"), None)

    if filled_leg is None:
        return {**row, "status": "OPEN", "pnl": None, "note": ""}

    exit_price = float(filled_leg.filled_avg_price)
    qty = float(filled_leg.filled_qty)
    pnl = (exit_price - entry_price) * qty if side == "BUY" else (entry_price - exit_price) * qty

    # Which leg filled tells us win vs loss: the take-profit leg has a
    # limit_price close to the logged target; the stop leg has a stop_price
    # close to the logged stop. Compare against both to label it.
    target = float(row["target"])
    stop = float(row["stop"])
    dist_to_target = abs(exit_price - target)
    dist_to_stop = abs(exit_price - stop)
    result = "TARGET (win)" if dist_to_target < dist_to_stop else "STOP (loss)"

    return {**row, "status": result, "pnl": round(pnl, 2), "note": ""}


def main():
    rows = load_trade_log()
    if not rows:
        return

    api_key = os.environ["POC_AMD_ALPACA_API_KEY"]
    secret_key = os.environ["POC_AMD_ALPACA_SECRET_KEY"]
    trading_client = TradingClient(api_key, secret_key, paper=True)

    results = [evaluate_trade(trading_client, row) for row in rows]

    closed = [r for r in results if r["pnl"] is not None]
    open_trades = [r for r in results if r["status"] == "OPEN"]
    failed = [r for r in results if r["status"] == "LOOKUP_FAILED"]

    print(f"\n{'='*60}")
    print(f"POC/AMD STRATEGY - PERFORMANCE REPORT")
    print(f"{'='*60}")
    print(f"Total logged trades : {len(rows)}")
    print(f"Still open           : {len(open_trades)}")
    print(f"Lookup failed         : {len(failed)}")
    print(f"Closed (have result)  : {len(closed)}\n")

    if closed:
        wins = [r for r in closed if r["pnl"] > 0]
        total_pnl = sum(r["pnl"] for r in closed)
        print(f"Win rate      : {len(wins)}/{len(closed)} ({100*len(wins)/len(closed):.1f}%)")
        print(f"Total P&L     : ${total_pnl:.2f}")
        print(f"Avg P&L/trade : ${total_pnl/len(closed):.2f}\n")

        print(f"{'Symbol':<8}{'Side':<6}{'Entry':<10}{'Result':<15}{'P&L':<10}")
        print("-" * 55)
        for r in closed:
            print(f"{r['symbol']:<8}{r['side']:<6}{float(r['entry_close']):<10.2f}"
                  f"{r['status']:<15}${r['pnl']:<9.2f}")
    else:
        print("No closed trades yet - check back once some have hit stop or target.")

    if open_trades:
        print(f"\nStill open ({len(open_trades)}):")
        for r in open_trades:
            print(f"  {r['symbol']} {r['side']} @ {r['entry_close']} "
                  f"(stop {r['stop']}, target {r['target']})")

    if failed:
        print(f"\nCould not look up ({len(failed)}) - check manually in Alpaca dashboard:")
        for r in failed:
            print(f"  {r['symbol']} order_id={r['order_id']}: {r['note']}")

    print()


if __name__ == "__main__":
    main()
