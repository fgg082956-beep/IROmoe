# sector_analyzer.py — Корреляция с сектором [P]
# Считает средний балл по сектору и применяет бонус/штраф к идее

import logging
from config import SECTOR_MAP, TICKER_TO_SECTOR, SECTOR_NAMES_RU

logger = logging.getLogger(__name__)


def compute_sector_scores(all_raw_scores: dict) -> dict:
    """
    Принимает: {ticker: score} — сырые баллы всех проанализированных тикеров
    Возвращает: {sector_key: {"avg": float, "count": int, "tickers": list}}
    [P]
    """
    sector_data = {}
    for sector, tickers in SECTOR_MAP.items():
        scores = [all_raw_scores[t] for t in tickers if t in all_raw_scores]
        if not scores:
            continue
        sector_data[sector] = {
            "avg":     round(sum(scores) / len(scores), 2),
            "count":   len(scores),
            "tickers": [t for t in tickers if t in all_raw_scores],
        }
    return sector_data


def get_sector_adj(ticker: str, sector_scores: dict) -> dict:
    """
    Для данного тикера возвращает корректировку балла на основе
    среднего по сектору.

    Логика:
      avg > +2  → сектор бычий   → +1 к лонгу, -1 к шорту
      avg < -2  → сектор медвежий → -1 к лонгу, +1 к шорту
      иначе     → нейтрально (0)

    Возвращает dict: {score_adj, sector_key, sector_name, sector_avg, label}
    [P]
    """
    sector_key = TICKER_TO_SECTOR.get(ticker)
    if not sector_key or sector_key not in sector_scores:
        return {"score_adj": 0, "sector_key": None,
                "sector_name": "нет данных", "sector_avg": None,
                "label": ""}

    info       = sector_scores[sector_key]
    avg        = info["avg"]
    sector_name = SECTOR_NAMES_RU.get(sector_key, sector_key)

    if avg > 2:
        adj   = 1
        trend = "🟢 бычий"
    elif avg < -2:
        adj   = -1
        trend = "🔴 медвежий"
    else:
        adj   = 0
        trend = "🟡 нейтральный"

    label = f"Сектор {sector_name}: {trend} (ср. балл {avg:+.1f})"
    return {
        "score_adj":   adj,
        "sector_key":  sector_key,
        "sector_name": sector_name,
        "sector_avg":  avg,
        "sector_trend": trend,
        "label":       label,
    }


def format_sector_line(sector_info: dict) -> str:
    """Форматирует строку для Telegram-сообщения."""
    if not sector_info or not sector_info.get("sector_key"):
        return ""
    return f"🏭 {sector_info['label']}"
