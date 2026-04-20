import argparse
import importlib
import json
import logging
import math
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data_fetcher import get_candles
from config import TICKERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FORWARD_DAYS = (1, 3, 5, 10, 20)
REPORTS_DIR = Path("reports_walkforward")
MIN_HISTORY_FOR_TRAIN = 120
MIN_TRAIN_SIGNALS_PER_IND = 12
MIN_TRAIN_SIGNALS_PER_TICKER_DIR = 6


@dataclass
class Window:
    idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _reload_backtest_module():
    import backtest
    return importlib.reload(backtest)


def fetch_history(ticker: str, days: int) -> pd.DataFrame:
    df = get_candles(ticker, days=days, interval=24)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_values("begin").reset_index(drop=True).copy()
    df["begin"] = pd.to_datetime(df["begin"])
    return df


def build_windows(all_dates: List[pd.Timestamp], train_days: int, test_days: int, step_days: int) -> List[Window]:
    if not all_dates:
        return []
    dates = pd.Series(sorted(pd.to_datetime(pd.Series(all_dates).dropna().unique())))
    windows: List[Window] = []
    start_idx = 0
    idx = 1
    while True:
        train_start = dates.iloc[start_idx]
        train_end_target = train_start + pd.Timedelta(days=train_days)
        test_end_target = train_end_target + pd.Timedelta(days=test_days)

        train_slice = dates[(dates >= train_start) & (dates < train_end_target)]
        test_slice = dates[(dates >= train_end_target) & (dates < test_end_target)]
        if len(train_slice) < 60 or len(test_slice) < 10:
            break

        windows.append(Window(
            idx=idx,
            train_start=train_slice.iloc[0],
            train_end=train_slice.iloc[-1],
            test_start=test_slice.iloc[0],
            test_end=test_slice.iloc[-1],
        ))
        idx += 1

        next_target = train_start + pd.Timedelta(days=step_days)
        next_idxs = dates[dates >= next_target]
        if next_idxs.empty:
            break
        start_idx = next_idxs.index[0]
        if start_idx >= len(dates) - 20:
            break
    return windows


def run_backtest_on_dataframes(module, data_by_ticker: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_records = []
    min_window = getattr(module, "MIN_WINDOW", 60)
    forward_days = getattr(module, "FORWARD_DAYS", [5, 10, 20])

    for ticker, df_full in data_by_ticker.items():
        if df_full.empty or len(df_full) < min_window + max(forward_days) + 2:
            continue
        df_full = df_full.sort_values("begin").reset_index(drop=True)
        closes = df_full["close"].values
        dates = df_full["begin"].values
        n = len(df_full)

        for i in range(min_window, n - max(forward_days)):
            window = df_full.iloc[i - min_window:i].copy().reset_index(drop=True)
            try:
                indscores = module.indicator_snapshot(window)
            except Exception:
                continue
            signal = module.signal_from_scores(indscores)
            if not signal:
                continue
            entry_price = closes[i]
            if not entry_price or entry_price <= 0:
                continue
            row = {
                "ticker": ticker,
                "date": pd.Timestamp(dates[i]).strftime("%Y-%m-%d"),
                "signal": signal,
                "score": int(sum(indscores.values())),
                "entry": round(float(entry_price), 4),
            }
            for fwd in forward_days:
                future_price = closes[i + fwd]
                ret_pct = (future_price - entry_price) / entry_price * 100
                row[f"ret_{fwd}d"] = round(float(ret_pct), 4)
                row[f"win_{fwd}d"] = bool((signal == "ЛОНГ" and ret_pct > 0) or (signal == "ШОРТ" and ret_pct < 0))
            for indname, indscore in indscores.items():
                row[f"ind_{indname}"] = indscore
            all_records.append(row)
    return pd.DataFrame(all_records)


def build_summary(df: pd.DataFrame, forward_days) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for ticker, grp in df.groupby("ticker"):
        for sig in ["ЛОНГ", "ШОРТ", "ALL"]:
            sub = grp if sig == "ALL" else grp[grp["signal"] == sig]
            if sub.empty:
                continue
            row = {"ticker": ticker, "signal": sig, "total_signals": len(sub)}
            for fwd in forward_days:
                wins = int(sub[f"win_{fwd}d"].sum())
                row[f"win_{fwd}d"] = wins
                row[f"acc_{fwd}d_%"] = round(wins / len(sub) * 100, 1)
                row[f"avg_ret_{fwd}d"] = round(sub[f"ret_{fwd}d"].mean(), 2)
            rows.append(row)
    return pd.DataFrame(rows)


def build_indicator_stats(df: pd.DataFrame, forward_days) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    ind_cols = [c for c in df.columns if c.startswith("ind_")]
    for col in ind_cols:
        name = col[4:]
        active = df[df[col] != 0].copy()
        if active.empty:
            continue
        bull_mask = active[col] > 0
        bear_mask = active[col] < 0
        for direction, mask in [("бычий (+)", bull_mask), ("медвежий (−)", bear_mask)]:
            sub = active[mask]
            if sub.empty:
                continue
            row = {
                "indicator": name,
                "direction": direction,
                "signals": len(sub),
                "avg_contrib": round(sub[col].mean(), 2),
            }
            for fwd in forward_days:
                if direction == "бычий (+)":
                    wins = int((sub[f"ret_{fwd}d"] > 0).sum())
                else:
                    wins = int((sub[f"ret_{fwd}d"] < 0).sum())
                row[f"acc_{fwd}d_%"] = round(wins / len(sub) * 100, 1)
                row[f"avg_ret_{fwd}d"] = round(sub[f"ret_{fwd}d"].mean(), 2)
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and "acc_10d_%" in out.columns:
        out = out.sort_values(["acc_10d_%", "signals"], ascending=[False, False]).reset_index(drop=True)
    return out


def fit_weights_from_train(ind_stats: pd.DataFrame, summary: pd.DataFrame) -> Dict:
    cfg = {
        "indicator_overrides": {},
        "ticker_signal_filter": {},
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "logic": "walk-forward train fit",
        },
    }

    if not ind_stats.empty:
        for _, r in ind_stats.iterrows():
            signals = int(r["signals"])
            acc10 = float(r.get("acc_10d_%", 0))
            avg10 = float(r.get("avg_ret_10d", 0))
            indicator = str(r["indicator"])
            direction = "bull" if "бычий" in str(r["direction"]) else "bear"
            if signals < MIN_TRAIN_SIGNALS_PER_IND:
                continue

            action = None
            score = None
            if acc10 >= 68 and ((direction == "bull" and avg10 > 0) or (direction == "bear" and avg10 < 0)):
                action = "boost"
                score = 1
            elif acc10 <= 48:
                action = "disable"
                score = 0
            elif acc10 <= 52:
                action = "weaken"
                score = 0

            if action:
                cfg["indicator_overrides"].setdefault(indicator, {})[direction] = {
                    "action": action,
                    "train_acc_10d": round(acc10, 1),
                    "train_avg_ret_10d": round(avg10, 2),
                    "signals": signals,
                    "delta": score,
                }

    if not summary.empty:
        sub = summary[summary["signal"].isin(["ЛОНГ", "ШОРТ"])].copy()
        for ticker, grp in sub.groupby("ticker"):
            allowed = []
            for sig in ["ЛОНГ", "ШОРТ"]:
                row = grp[grp["signal"] == sig]
                if row.empty:
                    continue
                total = int(row.iloc[0]["total_signals"])
                acc10 = float(row.iloc[0].get("acc_10d_%", 0))
                avg10 = float(row.iloc[0].get("avg_ret_10d", 0))
                if total < MIN_TRAIN_SIGNALS_PER_TICKER_DIR:
                    continue
                if sig == "ЛОНГ":
                    if acc10 >= 55 and avg10 > 0:
                        allowed.append(sig)
                else:
                    if acc10 >= 55 and avg10 < 0:
                        allowed.append(sig)
            if allowed and len(allowed) < 2:
                cfg["ticker_signal_filter"][ticker] = allowed
    return cfg


def save_fit_config(path: Path, fit_cfg: Dict):
    path.write_text(json.dumps(fit_cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def eval_window(module, win: Window, full_data: Dict[str, pd.DataFrame], out_dir: Path) -> Dict:
    train_data = {}
    test_data = {}
    for ticker, df in full_data.items():
        train_df = df[(df["begin"] >= win.train_start) & (df["begin"] <= win.train_end)].copy()
        test_df = df[(df["begin"] >= win.test_start - pd.Timedelta(days=220)) & (df["begin"] <= win.test_end + pd.Timedelta(days=25))].copy()
        if len(train_df) >= MIN_HISTORY_FOR_TRAIN:
            train_data[ticker] = train_df
        if len(test_df) >= 90:
            test_data[ticker] = test_df

    train_signals = run_backtest_on_dataframes(module, train_data)
    train_summary = build_summary(train_signals, module.FORWARD_DAYS)
    train_ind = build_indicator_stats(train_signals, module.FORWARD_DAYS)
    fit_cfg = fit_weights_from_train(train_ind, train_summary)
    save_fit_config(out_dir / f"window_{win.idx:02d}_fit.json", fit_cfg)

    test_signals = run_backtest_on_dataframes(module, test_data)
    test_signals["date"] = pd.to_datetime(test_signals["date"])
    test_signals = test_signals[(test_signals["date"] >= win.test_start) & (test_signals["date"] <= win.test_end)].copy()

    test_summary = build_summary(test_signals, module.FORWARD_DAYS)
    test_ind = build_indicator_stats(test_signals, module.FORWARD_DAYS)

    test_signals.to_csv(out_dir / f"window_{win.idx:02d}_signals.csv", index=False, encoding="utf-8-sig")
    test_summary.to_csv(out_dir / f"window_{win.idx:02d}_summary.csv", index=False, encoding="utf-8-sig")
    test_ind.to_csv(out_dir / f"window_{win.idx:02d}_indicator_stats.csv", index=False, encoding="utf-8-sig")

    row = {
        "window": win.idx,
        "train_start": win.train_start.strftime("%Y-%m-%d"),
        "train_end": win.train_end.strftime("%Y-%m-%d"),
        "test_start": win.test_start.strftime("%Y-%m-%d"),
        "test_end": win.test_end.strftime("%Y-%m-%d"),
        "train_signals": len(train_signals),
        "test_signals": len(test_signals),
    }
    for fwd in module.FORWARD_DAYS:
        if len(test_signals):
            row[f"acc_{fwd}d_%"] = round(test_signals[f"win_{fwd}d"].mean() * 100, 1)
            row[f"avg_ret_{fwd}d"] = round(test_signals[f"ret_{fwd}d"].mean(), 2)
        else:
            row[f"acc_{fwd}d_%"] = None
            row[f"avg_ret_{fwd}d"] = None
    return row


def build_equity_curve(df: pd.DataFrame, horizon: int, risk_per_trade_pct: float, start_capital: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "equity"])
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values("date")
    equity = start_capital
    rows = []
    for _, r in work.iterrows():
        ret = float(r[f"ret_{horizon}d"])
        pnl = equity * (risk_per_trade_pct / 100.0) * (ret / 10.0)
        equity += pnl
        rows.append({"date": r["date"].strftime("%Y-%m-%d"), "equity": round(equity, 2), "ret_pct": ret})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Walk-forward test for MOEX intraday/swing strategy")
    p.add_argument("--days", type=int, default=900)
    p.add_argument("--train-days", type=int, default=252)
    p.add_argument("--test-days", type=int, default=63)
    p.add_argument("--step-days", type=int, default=63)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--tickers", nargs="*", default=None)
    p.add_argument("--capital", type=float, default=100000.0)
    p.add_argument("--risk-pct", type=float, default=1.0)
    args = p.parse_args()

    module = _reload_backtest_module()
    tickers = args.tickers or TICKERS
    REPORTS_DIR.mkdir(exist_ok=True)
    run_dir = REPORTS_DIR / f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(exist_ok=True)

    logger.info("=" * 72)
    logger.info(f"Walk-forward: tickers={len(tickers)} days={args.days} train={args.train_days} test={args.test_days} step={args.step_days}")
    logger.info("=" * 72)

    full_data = {}
    all_dates = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_history, t, args.days): t for t in tickers}
        for fut in as_completed(futs):
            t = futs[fut]
            df = fut.result()
            if not df.empty:
                full_data[t] = df
                all_dates.extend(df["begin"].tolist())
                logger.info(f"{t}: {len(df)} bars")
            else:
                logger.warning(f"{t}: no data")

    windows = build_windows(all_dates, args.train_days, args.test_days, args.step_days)
    if not windows:
        raise SystemExit("No valid windows for walk-forward")

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tickers": tickers,
        "days": args.days,
        "train_days": args.train_days,
        "test_days": args.test_days,
        "step_days": args.step_days,
        "windows": len(windows),
        "forward_days": list(module.FORWARD_DAYS),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    all_test_signals = []
    for win in windows:
        logger.info(f"[window {win.idx}] train {win.train_start.date()}..{win.train_end.date()} | test {win.test_start.date()}..{win.test_end.date()}")
        row = eval_window(module, win, full_data, run_dir)
        rows.append(row)
        sig_path = run_dir / f"window_{win.idx:02d}_signals.csv"
        if sig_path.exists():
            df_sig = pd.read_csv(sig_path)
            if not df_sig.empty:
                df_sig["window"] = win.idx
                all_test_signals.append(df_sig)

    df_windows = pd.DataFrame(rows)
    df_windows.to_csv(run_dir / "walkforward_windows.csv", index=False, encoding="utf-8-sig")

    if all_test_signals:
        df_all = pd.concat(all_test_signals, ignore_index=True)
        df_all.to_csv(run_dir / "walkforward_all_test_signals.csv", index=False, encoding="utf-8-sig")
        agg_summary = build_summary(df_all, module.FORWARD_DAYS)
        agg_ind = build_indicator_stats(df_all, module.FORWARD_DAYS)
        agg_summary.to_csv(run_dir / "walkforward_summary.csv", index=False, encoding="utf-8-sig")
        agg_ind.to_csv(run_dir / "walkforward_indicator_stats.csv", index=False, encoding="utf-8-sig")
        for horizon in (1, 5, 10, 20):
            if f"ret_{horizon}d" in df_all.columns:
                eq = build_equity_curve(df_all, horizon, args.risk_pct, args.capital)
                eq.to_csv(run_dir / f"equity_curve_{horizon}d.csv", index=False, encoding="utf-8-sig")

    print("=" * 72)
    print("WALK-FORWARD READY")
    print(f"windows: {len(df_windows)}")
    print(f"folder : {run_dir}")
    if not df_windows.empty:
        cols = [c for c in ["window", "test_signals", "acc_5d_%", "acc_10d_%", "acc_20d_%", "avg_ret_10d"] if c in df_windows.columns]
        print(df_windows[cols].to_string(index=False))
    print("=" * 72)


if __name__ == "__main__":
    main()
