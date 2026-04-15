# market_regime.py — Фильтр рыночного режима [M]
# Определяет режим рынка по IMOEX: бычий / медвежий / боковик
# Результат используется в ideas_generator.py для фильтрации сигналов

import logging
import pandas as pd
from data_fetcher import get_index_value, _get_candles_moex
from config import (
    MARKET_REGIME_EMA_PERIOD,
    MARKET_REGIME_SIDEWAYS_PCT,
    MARKET_REGIME_SCORE_THRESHOLD_BULL,
    MARKET_REGIME_SCORE_THRESHOLD_BEAR,
    MARKET_REGIME_SIDEWAYS_EXTRA,
)

logger = logging.getLogger(__name__)

REGIME_BULL    = "bull"
REGIME_BEAR    = "bear"
REGIME_SIDE    = "sideways"

_EMOJI = {
    REGIME_BULL: "🟢",
    REGIME_BEAR: "🔴",
    REGIME_SIDE: "🟡",
}
_NAMES = {
    REGIME_BULL: "Бычий рынок",
    REGIME_BEAR: "Медвежий рынок",
    REGIME_SIDE: "Боковик",
}


def get_imoex_candles(days: int = 60) -> pd.DataFrame:
    """Загружает дневные свечи IMOEX через MOEX ISS."""
    try:
        return _get_candles_moex("IMOEX", days=days, interval=24)
    except Exception as e:
        logger.warning(f"IMOEX свечи: {e}")
        return pd.DataFrame()


def detect_regime(df_imoex: pd.DataFrame = None) -> dict:
    """
    Определяет текущий рыночный режим.

    Алгоритм:
    1. Считает EMA{MARKET_REGIME_EMA_PERIOD} на IMOEX
    2. Если цена выше EMA → потенциально бычий
    3. Если цена ниже EMA → потенциально медвежий
    4. Если разброс цен за последние 5 дней < SIDEWAYS_PCT% → боковик

    Возвращает dict с полями:
        regime: "bull" | "bear" | "sideways"
        emoji, name, imoex_last, ema, slope_pct,
        allowed_signals: список разрешённых типов ("ЛОНГ","ШОРТ")
        score_threshold: минимальный |балл| для выдачи идеи
    """
    if df_imoex is None or df_imoex.empty:
        df_imoex = get_imoex_candles(days=60)

    if df_imoex.empty or len(df_imoex) < MARKET_REGIME_EMA_PERIOD + 5:
        logger.warning("[M] Нет данных IMOEX — режим: нейтральный")
        return _neutral_regime()

    close = df_imoex["close"]
    ema   = close.ewm(span=MARKET_REGIME_EMA_PERIOD, adjust=False).mean()

    last      = close.iloc[-1]
    ema_last  = ema.iloc[-1]
    ema_prev  = ema.iloc[-2] if len(ema) > 1 else ema_last
    slope_pct = round((ema_last - ema_prev) / ema_prev * 100, 3) if ema_prev else 0

    # Боковик: разброс за 5 дней < SIDEWAYS_PCT%
    w5_hi = close.iloc[-5:].max()
    w5_lo = close.iloc[-5:].min()
    w5_range_pct = (w5_hi - w5_lo) / w5_lo * 100 if w5_lo > 0 else 0

    if w5_range_pct < MARKET_REGIME_SIDEWAYS_PCT:
        regime = REGIME_SIDE
    elif last > ema_last:
        regime = REGIME_BULL
    else:
        regime = REGIME_BEAR

    # Разрешённые сигналы
    if regime == REGIME_BULL:
        allowed = ["ЛОНГ"]
        thr_long  = MARKET_REGIME_SCORE_THRESHOLD_BULL
        thr_short = None   # шорты запрещены в бычьем рынке
    elif regime == REGIME_BEAR:
        allowed   = ["ШОРТ"]
        thr_long  = None
        thr_short = MARKET_REGIME_SCORE_THRESHOLD_BEAR
    else:   # sideways
        allowed   = ["ЛОНГ", "ШОРТ"]
        # В боковике повышаем порог на EXTRA
        thr_long  = MARKET_REGIME_SCORE_THRESHOLD_BULL  + MARKET_REGIME_SIDEWAYS_EXTRA
        thr_short = MARKET_REGIME_SCORE_THRESHOLD_BEAR  - MARKET_REGIME_SIDEWAYS_EXTRA

    result = {
        "regime":        regime,
        "emoji":         _EMOJI[regime],
        "name":          _NAMES[regime],
        "imoex_last":    round(last, 1),
        "ema":           round(ema_last, 1),
        "slope_pct":     slope_pct,
        "w5_range_pct":  round(w5_range_pct, 2),
        "allowed":       allowed,
        "thr_long":      thr_long,
        "thr_short":     thr_short,
    }
    logger.info(
        f"[M] Режим: {result['emoji']} {result['name']} | "
        f"IMOEX {result['imoex_last']} / EMA{MARKET_REGIME_EMA_PERIOD} {result['ema']} | "
        f"EMA slope {slope_pct:+.3f}% | 5d range {w5_range_pct:.2f}% | "
        f"Разрешено: {allowed}"
    )
    return result


def _neutral_regime() -> dict:
    return {
        "regime": REGIME_SIDE, "emoji": "⚪", "name": "Нет данных",
        "imoex_last": None, "ema": None, "slope_pct": 0, "w5_range_pct": 0,
        "allowed": ["ЛОНГ", "ШОРТ"],
        "thr_long":  MARKET_REGIME_SCORE_THRESHOLD_BULL,
        "thr_short": MARKET_REGIME_SCORE_THRESHOLD_BEAR,
    }


def passes_regime_filter(idea: dict, regime: dict) -> bool:
    """
    Проверяет, проходит ли идея через фильтр рыночного режима.
    Возвращает True если идею нужно включить, False — отбросить.
    """
    signal = idea.get("signal")
    score  = idea.get("score", 0)

    if signal not in regime["allowed"]:
        logger.debug(
            f"  [M] {idea['ticker']} {signal} отброшен "
            f"(режим {regime['name']}, разрешено {regime['allowed']})"
        )
        return False

    if signal == "ЛОНГ" and regime["thr_long"] is not None:
        if score < regime["thr_long"]:
            logger.debug(f"  [M] {idea['ticker']} ЛОНГ отброшен (балл {score} < {regime['thr_long']})")
            return False

    if signal == "ШОРТ" and regime["thr_short"] is not None:
        if score > regime["thr_short"]:
            logger.debug(f"  [M] {idea['ticker']} ШОРТ отброшен (балл {score} > {regime['thr_short']})")
            return False

    return True
