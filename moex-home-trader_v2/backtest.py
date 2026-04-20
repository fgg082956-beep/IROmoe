# backtest.py — ЧЕСТНЫЙ бэктест IROmoe (v2)
#
# Что это делает по-новому (и почему это важно):
#
#   1. ЗЕРКАЛИТ ПРОДАКШЕН. Для расчёта сигнала используется analyze_ticker()
#      из analyzer.py — с теми же весами, что и в боевом боте. Старая версия
#      считала свои собственные +1/+2, и это мерило «не ту» стратегию.
#
#   2. ИСПОЛНЕНИЕ БЕЗ LOOK-AHEAD. Сигнал виден после закрытия бара i.
#      Вход — по open бара i+1 (open next bar) + slippage.
#      Никаких «купил по close, того же на котором увидел сигнал».
#
#   3. КОМИССИИ + ПРОСКАЛЬЗЫВАНИЕ. По умолчанию 0.05% комиссия в одну сторону
#      и 0.05% slippage. MOEX ~0.1-0.3% round-trip — это реалистично.
#
#   4. ДВА РЕЖИМА РЕЗУЛЬТАТА:
#        • "horizon" — держим N дней и закрываем по open бара i+1+N.
#                      Считаем доходность, winrate, средний ret.
#        • "trade"   — полный трейд с ATR-стопом и ATR-целями из
#                      ideas_generator.calculate_trade_idea(). Бар-за-баром
#                      проверяем hit(stop) / hit(tp1) / hit(tp2) / timeout.
#                      Это максимально близко к реальной торговле.
#
#   5. МЕТРИКИ. Считаем winrate, profit factor, expectancy, средний RR,
#      max drawdown эквити, медианную и среднюю доходность сделки.
#
#   6. ВЕРНЫЙ WIN. Порог "победы" можно задать явно (MIN_WIN_RET_PCT).
#      +0.01% больше не считается победой.
#
#   7. ФИЛЬТР ЛИКВИДНОСТИ. Сигнал игнорируется, если средний оборот
#      (close*volume) за 20 баров ниже MIN_AVG_TURNOVER_RUB.
#
# Отчёты в reports/:
#   backtest_signals.csv       — все сигналы (horizon-режим)
#   backtest_trades.csv        — все сделки (trade-режим)
#   backtest_summary.csv       — сводка по тикеру и горизонту
#   backtest_trade_summary.csv — сводка по тикеру (trade-режим)
#   indicator_stats.csv        — точность каждого индикатора

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Импорты проекта ────────────────────────────────────────────────────────────
from data_fetcher import _get_candles_moex
from analyzer import (
    analyze_ticker,
    calculate_atr,
    find_support_resistance,
)

# Расчёт идеи используем из боевого генератора, чтобы мерять то, что реально торгуем
try:
    from ideas_generator import calculate_trade_idea  # type: ignore
except Exception:  # на случай циклических/конфиг-проблем
    calculate_trade_idea = None  # тогда trade-режим деградирует до horizon-только

from config import TICKERS
try:
    from config import TICKER_SIGNAL_FILTER
except Exception:
    TICKER_SIGNAL_FILTER = {}

# ── Настройки бэктеста ─────────────────────────────────────────────────────────
HISTORY_DAYS    = 365
MIN_WINDOW      = 60      # минимальная длина окна для analyze_ticker() (нужно EMA200)
FORWARD_DAYS    = [5, 10, 20]
MAX_FORWARD     = max(FORWARD_DAYS)
TRADE_MAX_BARS  = 20      # сколько баров максимум держим сделку в trade-режиме
MIN_WIN_RET_PCT = 0.3     # «победа» (horizon) только если |ret| >= 0.3%

FEE_PCT      = 0.05       # комиссия в одну сторону, %
SLIPPAGE_PCT = 0.05       # проскальзывание при входе/выходе, %
ROUND_TRIP   = 2 * (FEE_PCT + SLIPPAGE_PCT) / 100.0   # общие транз. издержки (доля)

MIN_AVG_TURNOVER_RUB = 5_000_000   # фильтр по ликвидности: 5 млн ₽ / день
MAX_WORKERS  = 6
REPORTS_DIR  = Path("reports")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Вспомогательные ────────────────────────────────────────────────────────────
def _apply_costs(ret_pct: float) -> float:
    """Вычитает полный round-trip (комиссия + slippage обоих сторон) из сырой доходности."""
    return ret_pct - ROUND_TRIP * 100.0


def _safe_analysis(window: pd.DataFrame) -> Optional[dict]:
    """Вызов analyze_ticker без новостей/недели/дивов — чистый техбалл, как в боте."""
    try:
        last = float(window["close"].iloc[-1])
        return analyze_ticker(window, last, news_sentiment_score=0)
    except Exception:
        return None


def _passes_liquidity(window: pd.DataFrame) -> bool:
    if "volume" not in window.columns:
        return True
    tail = window.tail(20)
    if tail.empty:
        return False
    avg_turnover = (tail["close"] * tail["volume"]).mean()
    return bool(avg_turnover and avg_turnover >= MIN_AVG_TURNOVER_RUB)


def _indicator_contributions(window: pd.DataFrame) -> dict:
    """
    Тонкая обёртка: достаём вклад каждого индикатора в итоговый score,
    чтобы посчитать indicator_stats. Используем причины из analyze_ticker.
    """
    a = _safe_analysis(window)
    if a is None:
        return {}
    # analyze_ticker возвращает reasons, но не «вклады поштучно». Для грубого
    # маркирования делаем так: ставим +1/-1/0 по ключевым словам в reasons.
    contrib = {
        "RSI": 0, "MACD": 0, "EMA_trend": 0,
        "OBV": 0, "RSI_div": 0, "Bollinger": 0,
        "Volume": 0, "Candles": 0, "Weekly": 0,
    }
    for r in a.get("reasons", []) or []:
        rr = str(r).lower()
        if "rsi дивергенц" in rr:
            contrib["RSI_div"] = 1 if "быч" in rr else -1
        elif rr.startswith("rsi") or "rsi" in rr[:6]:
            if "перепродан" in rr or "зона покуп" in rr:
                contrib["RSI"] = 1
            elif "перекуп" in rr:
                contrib["RSI"] = -1
        elif "macd" in rr:
            contrib["MACD"] = 1 if ("быч" in rr or "выше сигн" in rr) else -1
        elif "ema" in rr:
            contrib["EMA_trend"] = 1 if ("золот" in rr or "вверх" in rr) else -1
        elif "obv" in rr:
            contrib["OBV"] = 1 if "растёт" in rr or "быч" in rr else -1
        elif "боллиндж" in rr:
            contrib["Bollinger"] = 1 if "ниже нижн" in rr else -1
        elif "объём" in rr or "объем" in rr:
            contrib["Volume"] = 1 if "рост" in rr else -1
        elif "поглощ" in rr or "молот" in rr or "висел" in rr:
            contrib["Candles"] = 1 if "быч" in rr or "молот" in rr else -1
        elif "недельн" in rr:
            contrib["Weekly"] = 1 if "раст" in rr or "быч" in rr else -1
    return contrib


# ── Horizon-режим (close-to-open-next-bar, держим N дней) ──────────────────────
def _horizon_outcome(df: pd.DataFrame, i: int, direction: str, fwd: int) -> Optional[float]:
    """
    Входим по open[i+1], выходим по open[i+1+fwd]. Возвращает ЧИСТУЮ доходность, %.
    Если баров не хватает — None.
    """
    if i + 1 + fwd >= len(df):
        return None
    entry = float(df["open"].iloc[i + 1])
    exitp = float(df["open"].iloc[i + 1 + fwd])
    if entry <= 0 or exitp <= 0:
        return None
    gross = (exitp - entry) / entry * 100.0
    if direction == "ШОРТ":
        gross = -gross
    return _apply_costs(gross)


# ── Trade-режим (полный трейд с ATR-стопом и целями) ───────────────────────────
def _trade_outcome(df: pd.DataFrame, i: int, analysis: dict, direction: str) -> Optional[dict]:
    """
    Симулирует реальную сделку:
      entry = open[i+1] (с проскальзыванием)
      stop/tp = из calculate_trade_idea на момент сигнала
      Перебираем бары вперёд (до TRADE_MAX_BARS) и проверяем high/low:
        если пересёк стоп — стоп; если пересёк tp1 — tp1 и добираем до tp2;
        иначе по таймауту — close последнего бара.
    """
    if calculate_trade_idea is None:
        return None
    if i + 1 >= len(df):
        return None

    # idea строится на данных до бара i включительно
    window = df.iloc[: i + 1]
    # find_support_resistance уже внутри analyze_ticker, так что можно взять оттуда:
    atr_series = calculate_atr(window)
    atr_val = float(atr_series.iloc[-1]) if not atr_series.empty and not pd.isna(atr_series.iloc[-1]) else 0.0
    lvl = find_support_resistance(window)

    price_data = {"last": float(df["close"].iloc[i]), "change_pct": 0.0}

    # подсовываем analyze_ticker-совместимый analysis (часть полей уже есть)
    analysis_ex = dict(analysis)
    analysis_ex.setdefault("atr", atr_val)
    analysis_ex.setdefault("atr_pct", (atr_val / price_data["last"] * 100) if price_data["last"] else 0.0)
    analysis_ex.setdefault("support", lvl.get("support"))
    analysis_ex.setdefault("resistance", lvl.get("resistance"))
    analysis_ex.setdefault("score", analysis.get("score", 0))
    analysis_ex.setdefault("score_tech", analysis.get("score_tech", analysis.get("score", 0)))
    analysis_ex.setdefault("score_news", 0)
    analysis_ex.setdefault("rsi", 50)
    analysis_ex.setdefault("reasons", analysis.get("reasons", []))

    try:
        idea = calculate_trade_idea(
            "BTTEST", price_data, analysis_ex, style="swing",
            sector_adj=None, ob=None,
        )
    except Exception:
        return None
    if idea is None or idea.get("signal") != direction:
        return None

    # Фактический entry по open[i+1] с проскальзыванием
    raw_entry = float(df["open"].iloc[i + 1])
    slip = raw_entry * SLIPPAGE_PCT / 100.0
    entry = raw_entry + slip if direction == "ЛОНГ" else raw_entry - slip
    stop  = float(idea["stop_loss"])
    tp1   = float(idea["take_profit_1"])
    tp2   = float(idea["take_profit_2"])

    # Проверяем, что уровни в правильных знаках (защита от старых багов)
    if direction == "ЛОНГ" and not (stop < entry < tp1):
        return None
    if direction == "ШОРТ" and not (tp1 < entry < stop):
        return None

    rr_planned = abs(tp1 - entry) / max(abs(entry - stop), 1e-9)

    exit_price = None
    exit_reason = "timeout"
    for k in range(1, TRADE_MAX_BARS + 1):
        if i + 1 + k >= len(df):
            break
        bar = df.iloc[i + 1 + k]
        hi, lo = float(bar["high"]), float(bar["low"])
        if direction == "ЛОНГ":
            # Пессимистично: сначала проверяем стоп, потом тейк
            if lo <= stop:
                exit_price = stop
                exit_reason = "stop"
                break
            if hi >= tp2:
                exit_price = tp2
                exit_reason = "tp2"
                break
            if hi >= tp1:
                exit_price = tp1
                exit_reason = "tp1"
                # допускаем, что после tp1 не двигаем стоп; закрываем тут
                break
        else:
            if hi >= stop:
                exit_price = stop
                exit_reason = "stop"
                break
            if lo <= tp2:
                exit_price = tp2
                exit_reason = "tp2"
                break
            if lo <= tp1:
                exit_price = tp1
                exit_reason = "tp1"
                break

    if exit_price is None:
        last_idx = min(i + 1 + TRADE_MAX_BARS, len(df) - 1)
        exit_price = float(df["close"].iloc[last_idx])
        exit_reason = "timeout"

    # Чистая доходность с round-trip-комиссией
    gross = (exit_price - entry) / entry * 100.0
    if direction == "ШОРТ":
        gross = -gross
    net = _apply_costs(gross)

    return {
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "tp1": round(tp1, 4),
        "tp2": round(tp2, 4),
        "exit": round(exit_price, 4),
        "exit_reason": exit_reason,
        "rr_planned": round(rr_planned, 2),
        "ret_pct_net": round(net, 3),
    }


# ── Бэктест одного тикера ──────────────────────────────────────────────────────
def backtest_ticker(ticker: str, days: int = HISTORY_DAYS) -> dict:
    df = _get_candles_moex(ticker, days=days, interval=24)
    if df is None or df.empty or len(df) < MIN_WINDOW + MAX_FORWARD + 5:
        logger.warning(f"  {ticker}: недостаточно данных ({0 if df is None else len(df)} свечей)")
        return {"signals": [], "trades": []}

    df = df.sort_values("begin").reset_index(drop=True)
    dates = df["begin"]

    signals_rows: list[dict] = []
    trades_rows: list[dict] = []

    # Скользящее окно: сигнал на баре i, исполнение с i+1
    last_i = len(df) - 1 - MAX_FORWARD  # нужен хвост для horizon-режима
    for i in range(MIN_WINDOW, last_i):
        window = df.iloc[: i + 1].copy().reset_index(drop=True)
        if not _passes_liquidity(window):
            continue

        analysis = _safe_analysis(window)
        if analysis is None:
            continue
        signal = analysis.get("signal", "НЕЙТРАЛЬНО")
        if signal not in ("ЛОНГ", "ШОРТ"):
            continue
        # [BT3] Применяем TICKER_SIGNAL_FILTER так же, как это делается в
        # ideas_generator.calculate_trade_idea в продакшене. Если для тикера
        # задан whitelist направлений, а текущий сигнал в него не входит —
        # пропускаем сигнал (как будто бот его бы не выдал).
        allowed = TICKER_SIGNAL_FILTER.get(ticker)
        if allowed and signal not in allowed:
            continue

        base = {
            "ticker":   ticker,
            "date":     pd.Timestamp(dates.iloc[i]).strftime("%Y-%m-%d"),
            "signal":   signal,
            "score":    int(analysis.get("score", 0)),
            "rsi":      round(float(analysis.get("rsi", 0) or 0), 1),
            "close":    round(float(df["close"].iloc[i]), 2),
        }

        # --- Horizon outcomes ---
        row = dict(base)
        for fwd in FORWARD_DAYS:
            net = _horizon_outcome(df, i, signal, fwd)
            row[f"ret_{fwd}d"] = round(net, 3) if net is not None else None
            if net is None:
                row[f"win_{fwd}d"] = None
            else:
                # победа: знак совпал и модуль > MIN_WIN_RET_PCT
                row[f"win_{fwd}d"] = bool(net >= MIN_WIN_RET_PCT)
        # Вклады индикаторов (для indicator_stats)
        contribs = _indicator_contributions(window)
        for k, v in contribs.items():
            row[f"ind_{k}"] = v
        signals_rows.append(row)

        # --- Trade outcome ---
        tr = _trade_outcome(df, i, analysis, signal)
        if tr is not None:
            trades_rows.append({**base, **tr})

    logger.info(f"  {ticker}: сигналов={len(signals_rows)} сделок={len(trades_rows)}")
    return {"signals": signals_rows, "trades": trades_rows}


# ── Агрегации ──────────────────────────────────────────────────────────────────
def _build_signal_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for ticker, grp in df.groupby("ticker"):
        for sig in ["ЛОНГ", "ШОРТ", "ALL"]:
            sub = grp if sig == "ALL" else grp[grp["signal"] == sig]
            if sub.empty:
                continue
            row = {"ticker": ticker, "signal": sig, "total_signals": len(sub)}
            for fwd in FORWARD_DAYS:
                col_w = f"win_{fwd}d"
                col_r = f"ret_{fwd}d"
                valid = sub[sub[col_w].notna()]
                if valid.empty:
                    continue
                wins = int(valid[col_w].sum())
                row[f"win_{fwd}d"] = wins
                row[f"acc_{fwd}d_%"] = round(wins / len(valid) * 100, 1)
                row[f"avg_ret_{fwd}d"] = round(valid[col_r].mean(), 2)
            rows.append(row)
    return pd.DataFrame(rows)


def _build_trade_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for (ticker, sig), grp in trades.groupby(["ticker", "signal"]):
        wins_mask = grp["ret_pct_net"] > 0
        wins = grp[wins_mask]
        losses = grp[~wins_mask]
        gross_win = wins["ret_pct_net"].sum()
        gross_loss = -losses["ret_pct_net"].sum()
        pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
        wr = len(wins) / len(grp) * 100.0
        expectancy = grp["ret_pct_net"].mean()
        rows.append({
            "ticker": ticker,
            "signal": sig,
            "trades": len(grp),
            "winrate_%": round(wr, 1),
            "avg_ret_%": round(expectancy, 3),
            "median_ret_%": round(grp["ret_pct_net"].median(), 3),
            "profit_factor": round(pf, 2) if pf != float("inf") else "inf",
            "avg_RR": round(grp["rr_planned"].mean(), 2),
            "pct_stop": round((grp["exit_reason"] == "stop").mean() * 100, 1),
            "pct_tp1":  round((grp["exit_reason"] == "tp1").mean() * 100, 1),
            "pct_tp2":  round((grp["exit_reason"] == "tp2").mean() * 100, 1),
            "pct_timeout": round((grp["exit_reason"] == "timeout").mean() * 100, 1),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["winrate_%", "avg_ret_%"], ascending=[False, False])
    return out


def _build_indicator_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    ind_cols = [c for c in df.columns if c.startswith("ind_")]
    rows = []
    for col in ind_cols:
        name = col[4:]
        for sign, label in [(1, "бычий (+)"), (-1, "медвежий (−)")]:
            sub = df[df[col] == sign]
            if sub.empty:
                continue
            for fwd in FORWARD_DAYS:
                col_r = f"ret_{fwd}d"
                valid = sub[sub[col_r].notna()]
                if valid.empty:
                    continue
                if sign > 0:
                    wins = (valid[col_r] > MIN_WIN_RET_PCT).sum()
                else:
                    wins = (valid[col_r] < -MIN_WIN_RET_PCT).sum()
                rows.append({
                    "indicator": name,
                    "direction": label,
                    "horizon":   fwd,
                    "signals":   len(valid),
                    "acc_%":     round(wins / len(valid) * 100, 1),
                    "avg_ret_%": round(valid[col_r].mean(), 2),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["horizon", "acc_%"], ascending=[True, False])
    return out


def _equity_curve(trades: pd.DataFrame) -> dict:
    """По суммарному эквити (равные позиции) считаем max DD."""
    if trades.empty:
        return {"max_dd_%": 0.0, "total_return_%": 0.0}
    s = trades.sort_values("date")["ret_pct_net"].fillna(0).values / 100.0
    eq = np.cumprod(1.0 + s)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return {
        "max_dd_%": round(float(dd.min() * 100), 2),
        "total_return_%": round(float((eq[-1] - 1.0) * 100), 2),
    }


# ── Оркестратор ────────────────────────────────────────────────────────────────
def run_backtest(tickers: Optional[list[str]] = None,
                 days: int = HISTORY_DAYS,
                 workers: int = MAX_WORKERS):
    if tickers is None:
        tickers = list(TICKERS)

    logger.info("=" * 64)
    logger.info(f"BACKTEST v2 | tickers={len(tickers)} days={days} workers={workers}")
    logger.info(f"TICKER_SIGNAL_FILTER: {len(TICKER_SIGNAL_FILTER)} ограничений активно")
    logger.info(f"horizons={FORWARD_DAYS} fee={FEE_PCT}% slip={SLIPPAGE_PCT}% "
                f"min_win={MIN_WIN_RET_PCT}% min_liq={MIN_AVG_TURNOVER_RUB:,}₽")
    logger.info("=" * 64)

    all_signals, all_trades = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(backtest_ticker, t, days): t for t in tickers}
        for fut in as_completed(futs):
            r = fut.result()
            all_signals.extend(r["signals"])
            all_trades.extend(r["trades"])

    if not all_signals:
        logger.error("Нет сигналов — проверь соединение с MOEX ISS / кэш / фильтры")
        return None

    df_signals = pd.DataFrame(all_signals)
    df_trades = pd.DataFrame(all_trades)
    REPORTS_DIR.mkdir(exist_ok=True)

    df_signals.to_csv(REPORTS_DIR / "backtest_signals.csv",
                      index=False, encoding="utf-8-sig")
    if not df_trades.empty:
        df_trades.to_csv(REPORTS_DIR / "backtest_trades.csv",
                         index=False, encoding="utf-8-sig")
    df_sum = _build_signal_summary(df_signals)
    df_sum.to_csv(REPORTS_DIR / "backtest_summary.csv",
                  index=False, encoding="utf-8-sig")
    df_trade_sum = _build_trade_summary(df_trades)
    if not df_trade_sum.empty:
        df_trade_sum.to_csv(REPORTS_DIR / "backtest_trade_summary.csv",
                            index=False, encoding="utf-8-sig")
    df_ind = _build_indicator_stats(df_signals)
    df_ind.to_csv(REPORTS_DIR / "indicator_stats.csv",
                  index=False, encoding="utf-8-sig")

    # ── Печать в консоль ──────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  BACKTEST v2 — {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    print("=" * 64)
    print(f"  Тикеров: {len(tickers)}   Сигналов: {len(df_signals)}   Сделок: {len(df_trades)}")
    longs = (df_signals["signal"] == "ЛОНГ").sum()
    shorts = (df_signals["signal"] == "ШОРТ").sum()
    print(f"  Лонги: {longs}   Шорты: {shorts}")
    print()
    print(f"  Транз. издержки учтены: round-trip = {ROUND_TRIP*100:.2f}%")
    print(f"  Порог победы (horizon): {MIN_WIN_RET_PCT}% чистыми")
    print()
    print("  HORIZON-режим (вход по open next bar, держим N дней):")
    for fwd in FORWARD_DAYS:
        col_w = f"win_{fwd}d"
        col_r = f"ret_{fwd}d"
        valid = df_signals[df_signals[col_w].notna()]
        if valid.empty:
            continue
        acc = valid[col_w].mean() * 100
        avg = valid[col_r].mean()
        print(f"    +{fwd:>2}д  точность={acc:5.1f}%   ср. чистый доход={avg:+.2f}%")
    print()

    if not df_trades.empty:
        wins = (df_trades["ret_pct_net"] > 0).sum()
        wr = wins / len(df_trades) * 100
        gw = df_trades.loc[df_trades["ret_pct_net"] > 0, "ret_pct_net"].sum()
        gl = -df_trades.loc[df_trades["ret_pct_net"] <= 0, "ret_pct_net"].sum()
        pf = (gw / gl) if gl > 0 else float("inf")
        eq = _equity_curve(df_trades)
        print("  TRADE-режим (ATR-stop + ATR-targets из calculate_trade_idea):")
        print(f"    сделок={len(df_trades)}  winrate={wr:.1f}%  "
              f"profit_factor={pf:.2f}  expectancy={df_trades['ret_pct_net'].mean():+.3f}%")
        print(f"    max DD={eq['max_dd_%']}%   cumulative={eq['total_return_%']}%")
        print(f"    exit: stop={(df_trades['exit_reason']=='stop').mean()*100:.1f}%  "
              f"tp1={(df_trades['exit_reason']=='tp1').mean()*100:.1f}%  "
              f"tp2={(df_trades['exit_reason']=='tp2').mean()*100:.1f}%  "
              f"timeout={(df_trades['exit_reason']=='timeout').mean()*100:.1f}%")
        print()

    if not df_ind.empty:
        top = df_ind[df_ind["horizon"] == 10].head(8)
        print("  ТОП индикаторов (горизонт 10 дней):")
        print(top[["indicator", "direction", "signals", "acc_%", "avg_ret_%"]].to_string(index=False))
        print()
        bot = df_ind[df_ind["horizon"] == 10].tail(5)
        print("  ХУДШИЕ индикаторы (горизонт 10 дней):")
        print(bot[["indicator", "direction", "signals", "acc_%", "avg_ret_%"]].to_string(index=False))

    print(f"\n  Файлы в {REPORTS_DIR}/")
    print("=" * 64)
    return {
        "signals": df_signals,
        "trades": df_trades,
        "summary": df_sum,
        "trade_summary": df_trade_sum,
        "indicators": df_ind,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IROmoe backtest v2 (honest)")
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="Список тикеров (по умолчанию — все из config.TICKERS)")
    parser.add_argument("--days", type=int, default=HISTORY_DAYS,
                        help=f"Дней истории (по умолч. {HISTORY_DAYS})")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Потоков (по умолч. {MAX_WORKERS})")
    parser.add_argument("--fee", type=float, default=FEE_PCT,
                        help=f"Комиссия одной стороны, %% (по умолч. {FEE_PCT})")
    parser.add_argument("--slip", type=float, default=SLIPPAGE_PCT,
                        help=f"Проскальзывание одной стороны, %% (по умолч. {SLIPPAGE_PCT})")
    parser.add_argument("--min-win", type=float, default=MIN_WIN_RET_PCT,
                        help=f"Мин. чистая доходность, чтобы считать победой, %% "
                             f"(по умолч. {MIN_WIN_RET_PCT})")
    parser.add_argument("--min-liq", type=float, default=MIN_AVG_TURNOVER_RUB,
                        help=f"Мин. средний дневной оборот, ₽ (по умолч. {MIN_AVG_TURNOVER_RUB})")
    args = parser.parse_args()

    # Применяем CLI-оверрайды
    FEE_PCT = args.fee
    SLIPPAGE_PCT = args.slip
    ROUND_TRIP = 2 * (FEE_PCT + SLIPPAGE_PCT) / 100.0
    MIN_WIN_RET_PCT = args.min_win
    MIN_AVG_TURNOVER_RUB = args.min_liq

    run_backtest(
        tickers=args.tickers,
        days=args.days,
        workers=args.workers,
    )
