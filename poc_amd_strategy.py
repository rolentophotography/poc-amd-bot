"""
POC / AMD Retest Strategy
==========================
Derived from analysis of TikTok trading videos (@bondbardtrade / @bondbardtradefx
/ @worldwidecpt). Core idea, in order:

  1. SWING     - find a clean directional move (swing low -> swing high, or
                 swing high -> swing low)
  2. PROFILE   - build a volume profile across that swing, find the POC
                 (price level where the most volume traded)
  3. MANIPULATE- watch for a sweep beyond the swing extreme (a stop-hunt /
                 liquidity grab) that quickly reverses - this is the trap,
                 NOT the entry signal
  4. STRUCTURE - require price to close back beyond a defined structure level,
                 confirming the real direction
  5. RETEST    - wait for price to pull back near the POC after the structure
                 break
  6. ENTRY     - enter in the original swing direction on the retest, stop
                 beyond the manipulation extreme, target = risk * reward_ratio

This module is self-contained (pandas/numpy only) so it can be dropped into
any bot. It exposes:
  - find_swings()
  - compute_volume_profile()
  - PocAmdStrategy.generate_signals(df) -> DataFrame with signal column
  - backtest(df, params) -> trades DataFrame + summary metrics

INPUT DATA FORMAT
------------------
`df` must be a pandas DataFrame, indexed by datetime, ascending, with columns:
    open, high, low, close, volume

PARAMETERS THAT WERE FUZZY IN THE SOURCE VIDEOS - NOW MADE EXPLICIT
----------------------------------------------------------------------
These three definitions were never precisely stated on camera. Defaults below
are reasonable starting points, NOT proven values - tune them via backtesting.

  swing_lookback      : bars on each side to confirm a swing high/low (fractal)
  manipulation_atr_mult: how far beyond the swing extreme (in ATR) counts as
                         a genuine "sweep" rather than noise
  manipulation_max_bars: sweep must reverse back within this many bars, or
                         it's treated as a real breakout instead (no trade)
  structure_break_atr_mult: how far price must close beyond the prior
                         counter-swing to confirm a real structure break
  retest_tolerance_atr: how close price must come back to the POC to count
                         as a valid "retest" (in ATR units)
  retest_max_bars     : how many bars after the structure break we'll wait
                         for a retest before giving up on the setup
  reward_risk_ratio   : target distance = risk distance * this multiple
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PocAmdParams:
    swing_lookback: int = 5
    volume_profile_bins: int = 24
    manipulation_atr_mult: float = 0.25
    manipulation_max_bars: int = 4
    structure_break_atr_mult: float = 0.15
    retest_tolerance_atr: float = 0.35
    retest_max_bars: int = 8
    reward_risk_ratio: float = 2.5
    atr_period: int = 14


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def find_swings(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Fractal swing detection: a bar is a swing high if its high is the max
    over the window [i-lookback, i+lookback], swing low = min likewise."""
    highs, lows = df["high"], df["low"]
    is_swing_high = highs == highs.rolling(2 * lookback + 1, center=True).max()
    is_swing_low = lows == lows.rolling(2 * lookback + 1, center=True).min()
    out = pd.DataFrame(index=df.index)
    out["swing_high"] = np.where(is_swing_high, highs, np.nan)
    out["swing_low"] = np.where(is_swing_low, lows, np.nan)
    return out


def compute_volume_profile(df: pd.DataFrame, start_idx, end_idx, bins: int) -> float:
    """Return the POC (price with the most traded volume) across
    df.loc[start_idx:end_idx]. Volume is split evenly across each bar's
    high-low range into `bins` price buckets."""
    segment = df.loc[start_idx:end_idx]
    if segment.empty:
        return np.nan
    lo, hi = segment["low"].min(), segment["high"].max()
    if hi <= lo:
        return segment["close"].iloc[-1]
    edges = np.linspace(lo, hi, bins + 1)
    vol_by_bin = np.zeros(bins)
    for _, row in segment.iterrows():
        bar_lo, bar_hi, bar_vol = row["low"], row["high"], row["volume"]
        if bar_hi <= bar_lo or bar_vol == 0:
            continue
        # which bins this bar's range overlaps
        lo_bin = np.searchsorted(edges, bar_lo, side="right") - 1
        hi_bin = np.searchsorted(edges, bar_hi, side="left")
        lo_bin = max(0, min(lo_bin, bins - 1))
        hi_bin = max(0, min(hi_bin, bins - 1))
        span = max(hi_bin - lo_bin + 1, 1)
        vol_by_bin[lo_bin:hi_bin + 1] += bar_vol / span
    poc_bin = int(np.argmax(vol_by_bin))
    return (edges[poc_bin] + edges[poc_bin + 1]) / 2


class PocAmdStrategy:
    """Direction-agnostic POC/AMD retest strategy."""

    def __init__(self, params: PocAmdParams = None):
        self.p = params or PocAmdParams()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns df with added columns:
          signal   : 'BUY' / 'SELL' / '' per bar (entry signal on that bar)
          stop     : stop-loss price for that signal
          target   : take-profit price for that signal
        """
        p = self.p
        df = df.copy()
        df["atr"] = _atr(df, p.atr_period)
        swings = find_swings(df, p.swing_lookback)
        df["swing_high"] = swings["swing_high"]
        df["swing_low"] = swings["swing_low"]

        df["signal"] = ""
        df["stop"] = np.nan
        df["target"] = np.nan

        swing_lows = df.index[df["swing_low"].notna()]
        swing_highs = df.index[df["swing_high"].notna()]

        # ---- Bullish setups: swing_low -> swing_high, expect pullback + reclaim ----
        for low_idx in swing_lows:
            later_highs = swing_highs[swing_highs > low_idx]
            if len(later_highs) == 0:
                continue
            high_idx = later_highs[0]
            self._eval_setup(df, p, direction="long",
                              start_idx=low_idx, extreme_idx=high_idx)

        # ---- Bearish setups: swing_high -> swing_low ----
        for high_idx in swing_highs:
            later_lows = swing_lows[swing_lows > high_idx]
            if len(later_lows) == 0:
                continue
            low_idx = later_lows[0]
            self._eval_setup(df, p, direction="short",
                              start_idx=high_idx, extreme_idx=low_idx)

        return df

    def _eval_setup(self, df, p, direction, start_idx, extreme_idx):
        loc_extreme = df.index.get_loc(extreme_idx)
        atr_at_extreme = df["atr"].iloc[loc_extreme]
        if pd.isna(atr_at_extreme) or atr_at_extreme == 0:
            return

        poc = compute_volume_profile(df, start_idx, extreme_idx, p.volume_profile_bins)
        if pd.isna(poc):
            return

        extreme_price = (df["low"].loc[extreme_idx] if direction == "long"
                          else df["high"].loc[extreme_idx])

        # 1) MANIPULATION: look for a sweep beyond extreme within manipulation_max_bars,
        #    that reverses back inside the range (not a clean continuation)
        window = df.iloc[loc_extreme + 1: loc_extreme + 1 + p.manipulation_max_bars]
        if window.empty:
            return

        sweep_thresh = p.manipulation_atr_mult * atr_at_extreme
        if direction == "long":
            swept = window[window["low"] < extreme_price - sweep_thresh]
        else:
            swept = window[window["high"] > extreme_price + sweep_thresh]
        if swept.empty:
            return  # no manipulation detected, skip this setup
        sweep_idx = swept.index[0]
        sweep_price = swept["low"].loc[sweep_idx] if direction == "long" else swept["high"].loc[sweep_idx]
        loc_sweep = df.index.get_loc(sweep_idx)

        # 2) STRUCTURE BREAK: after the sweep, require a close back beyond the
        #    extreme by structure_break_atr_mult, confirming reversal
        break_thresh = p.structure_break_atr_mult * atr_at_extreme
        struct_window = df.iloc[loc_sweep + 1: loc_sweep + 1 + p.retest_max_bars + 5]
        if direction == "long":
            broken = struct_window[struct_window["close"] > extreme_price + break_thresh]
        else:
            broken = struct_window[struct_window["close"] < extreme_price - break_thresh]
        if broken.empty:
            return
        break_idx = broken.index[0]
        loc_break = df.index.get_loc(break_idx)

        # 3) RETEST: within retest_max_bars after the break, price must come
        #    back close to the POC
        retest_thresh = p.retest_tolerance_atr * atr_at_extreme
        retest_window = df.iloc[loc_break + 1: loc_break + 1 + p.retest_max_bars]
        if retest_window.empty:
            return
        near_poc = (retest_window["low"] <= poc + retest_thresh) & \
                   (retest_window["high"] >= poc - retest_thresh)
        retest_candidates = retest_window[near_poc]
        if retest_candidates.empty:
            return
        entry_idx = retest_candidates.index[0]

        # avoid overwriting an existing signal on this bar
        if df.at[entry_idx, "signal"] != "":
            return

        risk = abs(poc - sweep_price)
        if risk <= 0:
            return
        target = poc + risk * p.reward_risk_ratio if direction == "long" \
            else poc - risk * p.reward_risk_ratio

        df.at[entry_idx, "signal"] = "BUY" if direction == "long" else "SELL"
        df.at[entry_idx, "stop"] = sweep_price
        df.at[entry_idx, "target"] = target


def backtest(df: pd.DataFrame, params: PocAmdParams = None) -> tuple[pd.DataFrame, dict]:
    """Simple one-position-at-a-time backtest: enters at signal bar's close,
    exits at stop or target (whichever hit first on subsequent bars), or at
    the last available bar if neither is hit."""
    strat = PocAmdStrategy(params)
    df = strat.generate_signals(df)

    trades = []
    open_trade = None

    for i, (idx, row) in enumerate(df.iterrows()):
        if open_trade is None:
            if row["signal"] in ("BUY", "SELL"):
                open_trade = {
                    "entry_time": idx,
                    "direction": row["signal"],
                    "entry_price": row["close"],
                    "stop": row["stop"],
                    "target": row["target"],
                }
            continue

        d = open_trade
        hit_stop = (row["low"] <= d["stop"]) if d["direction"] == "BUY" else (row["high"] >= d["stop"])
        hit_target = (row["high"] >= d["target"]) if d["direction"] == "BUY" else (row["low"] <= d["target"])

        if hit_stop or hit_target:
            exit_price = d["stop"] if hit_stop else d["target"]
            pnl = (exit_price - d["entry_price"]) if d["direction"] == "BUY" else (d["entry_price"] - exit_price)
            trades.append({**d, "exit_time": idx, "exit_price": exit_price,
                            "result": "STOP" if hit_stop else "TARGET", "pnl": pnl})
            open_trade = None

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, {"n_trades": 0}

    wins = trades_df[trades_df["pnl"] > 0]
    summary = {
        "n_trades": len(trades_df),
        "win_rate": len(wins) / len(trades_df),
        "avg_pnl": trades_df["pnl"].mean(),
        "total_pnl": trades_df["pnl"].sum(),
        "avg_win": wins["pnl"].mean() if not wins.empty else 0,
        "avg_loss": trades_df[trades_df["pnl"] <= 0]["pnl"].mean() if (trades_df["pnl"] <= 0).any() else 0,
    }
    return trades_df, summary


if __name__ == "__main__":
    # Quick self-test with synthetic data so the module is verified to run
    # end-to-end without needing live market data / network access.
    rng = np.random.default_rng(7)
    n = 400
    price = 100 + np.cumsum(rng.normal(0, 1, n))
    high = price + rng.uniform(0.2, 1.5, n)
    low = price - rng.uniform(0.2, 1.5, n)
    close = price + rng.normal(0, 0.3, n)
    open_ = price + rng.normal(0, 0.3, n)
    volume = rng.integers(1000, 10000, n)
    idx = pd.date_range("2026-01-01", periods=n, freq="30min")
    synth = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    trades, summary = backtest(synth)
    print("Self-test on synthetic data (sanity check only, not a real edge test):")
    print(summary)
