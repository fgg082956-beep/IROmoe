# ideas_generator.py — Генерация торговых идей (v7)
# [A]  Параллельный анализ через ThreadPoolExecutor (8 потоков)
# [TF] Недельные свечи передаются в analyze_ticker()
# [DG] Дивидендная проверка передаётся в analyze_ticker()

from concurrent.futures import ThreadPoolExecutor, as_completed
from config import DEPOSIT, MAX_RISK_PERCENT, TICKERS
from data_fetcher import (get_candles, get_current_price, get_index_value,
                          get_candles_weekly, is_near_dividend_gap)    # [TF][DG]
from analyzer import analyze_ticker
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    from config import MIN_AVG_VOLUME
except ImportError:
    MIN_AVG_VOLUME = 0

try:
    from news_sentiment import get_sentiment_addon, load_data as _ns_load
    NEWS_AVAILABLE = True
    def _get_news_headlines(ticker):
        data = _ns_load()
        if ticker in data and data[ticker]:
            return data[ticker][-1].get("headlines", [])
        return []
except ImportError:
    NEWS_AVAILABLE = False
    def get_sentiment_addon(t): return 0
    def _get_news_headlines(t): return []


def _passes_volume_filter(candles):
    if MIN_AVG_VOLUME <= 0 or candles.empty or "volume" not in candles.columns:
        return True, 0.0
    avg = candles["volume"].tail(20).mean()
    return (avg >= MIN_AVG_VOLUME, avg)


def calculate_trade_idea(ticker, price_data, analysis, style="swing"):
    current_price = price_data["last"]
    if not current_price or current_price == 0:
        return None
    signal = analysis["signal"]
    if signal == "НЕЙТРАЛЬНО":
        return None
    support    = analysis.get("support")
    resistance = analysis.get("resistance")
    atr        = analysis.get("atr")

    if signal == "ЛОНГ":
        entry = (round(support * 1.002, 2)
                 if support and abs(current_price - support) / current_price < 0.03
                 else current_price)
        if atr and atr > 0:
            atr_stop  = entry - 1.5 * atr
            stop_loss = round(min(support*0.995, atr_stop), 2) if support else round(atr_stop, 2)
        else:
            sp        = 0.015 if style == "intraday" else 0.03
            stop_loss = round(min(support*0.995, entry*(1-sp)), 2) if support else round(entry*(1-sp), 2)
        if resistance:
            tp1 = round(resistance * 0.998, 2); tp2 = round(resistance * 1.02, 2)
        elif atr and atr > 0:
            tp1 = round(entry + 2*atr, 2);      tp2 = round(entry + 3*atr, 2)
        else:
            tp1 = round(entry * 1.03, 2);        tp2 = round(entry * 1.05, 2)
    else:
        entry = (round(resistance * 0.998, 2)
                 if resistance and abs(current_price - resistance) / current_price < 0.03
                 else current_price)
        if atr and atr > 0:
            atr_stop  = entry + 1.5 * atr
            stop_loss = round(max(resistance*1.005, atr_stop), 2) if resistance else round(atr_stop, 2)
        else:
            sp        = 0.015 if style == "intraday" else 0.03
            stop_loss = round(max(resistance*1.005, entry*(1+sp)), 2) if resistance else round(entry*(1+sp), 2)
        if support:
            tp1 = round(support * 1.002, 2);  tp2 = round(support * 0.98, 2)
        elif atr and atr > 0:
            tp1 = round(entry - 2*atr, 2);    tp2 = round(entry - 3*atr, 2)
        else:
            tp1 = round(entry * 0.97, 2);      tp2 = round(entry * 0.95, 2)

    rps = abs(entry - stop_loss)
    if rps == 0:
        return None
    max_risk   = DEPOSIT * (MAX_RISK_PERCENT / 100)
    n_shares   = int(max_risk / rps)
    if n_shares == 0:
        return None
    risk_rub   = round(n_shares * rps, 0)
    risk_pct   = round(risk_rub / DEPOSIT * 100, 1)
    pot_profit = round(n_shares * abs(tp1 - entry), 0)
    rr         = round(abs(tp1 - entry) / rps, 1)
    if rr < 1.0:
        return None

    return {
        "ticker": ticker, "signal": signal, "style": style,
        "current_price": current_price, "entry": entry,
        "stop_loss": stop_loss, "take_profit_1": tp1, "take_profit_2": tp2,
        "risk_reward": f"1:{rr}", "num_shares": n_shares,
        "risk_rubles": int(risk_rub), "risk_pct": risk_pct,
        "potential_profit": int(pot_profit),
        "rsi": analysis["rsi"], "score": analysis["score"],
        "score_tech": analysis["score_tech"], "score_news": analysis["score_news"],
        "reasons": analysis["reasons"], "change_pct": price_data.get("change_pct", 0),
        "news_headlines": _get_news_headlines(ticker),
        "weekly_trend": analysis.get("weekly_trend", "нет данных"),
        "div_warning":  analysis.get("div_warning"),
        "ema50": analysis.get("ema50"), "ema200": analysis.get("ema200"),
        "obv_trend": analysis.get("obv_trend"), "candle_pats": analysis.get("candle_pats", []),
    }


def _analyze_one(ticker):
    try:
        # Дневные свечи
        candles = get_candles(ticker, days=60)
        if candles.empty:
            logger.info(f"  {ticker}: нет данных"); return None

        ok, avg_vol = _passes_volume_filter(candles)
        if not ok:
            logger.info(f"  {ticker}: низкий объём ({avg_vol:,.0f})"); return None

        price = get_current_price(ticker)
        if not price or not price["last"]:
            logger.info(f"  {ticker}: нет цены"); return None

        # [TF] Недельные свечи (52 недели)
        candles_w = get_candles_weekly(ticker, weeks=52)

        # [DG] Дивидендная зона (±5 дней)
        div_info = is_near_dividend_gap(ticker, window_days=5)

        news_score = get_sentiment_addon(ticker)
        analysis   = analyze_ticker(
            candles, price["last"],
            news_sentiment_score=news_score,
            df_weekly=candles_w,          # [TF]
            div_info=div_info,            # [DG]
        )

        # Лог
        tf_str = f", таймфрейм: {analysis.get('weekly_trend','?')[:25]}" if analysis.get('weekly_trend') else ""
        dg_str = f", ⚠️ дивотсечка" if div_info.get("near") else ""
        extra  = f", новости: {analysis['score_news']:+d}" if NEWS_AVAILABLE else ""
        logger.info(f"  {ticker}: {analysis['signal']} "
                    f"(RSI {analysis['rsi']}, тех: {analysis['score_tech']:+d}"
                    f"{extra}{tf_str}{dg_str}, итого: {analysis['score']:+d})")

        return calculate_trade_idea(ticker, price, analysis, style="swing")
    except Exception as e:
        logger.error(f"  {ticker}: ошибка — {e}", exc_info=True)
        return None


def generate_all_ideas(max_workers: int = 8) -> dict:
    logger.info("=" * 60)
    logger.info(f"Генерация идей: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    logger.info(f"Тикеров: {len(TICKERS)}, потоков: {max_workers}")
    logger.info("=" * 60)

    index_data = get_index_value()
    if index_data:
        logger.info(f"IMOEX: {index_data['value']:.0f} ({index_data['change_pct']:+.2f}%)")

    long_ideas, short_ideas = [], []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_analyze_one, t): t for t in TICKERS}
        for future in as_completed(futures):
            idea = future.result()
            if idea:
                (long_ideas if idea["signal"] == "ЛОНГ" else short_ideas).append(idea)

    long_ideas  = sorted(long_ideas,  key=lambda x: -x["score"])[:5]
    short_ideas = sorted(short_ideas, key=lambda x:  x["score"])[:4]

    logger.info(f"Итого: {len(long_ideas)} лонгов, {len(short_ideas)} шортов")
    return {
        "timestamp":   datetime.now().strftime("%d.%m.%Y %H:%M"),
        "index":       index_data,
        "long_ideas":  long_ideas,
        "short_ideas": short_ideas,
        "news_active": NEWS_AVAILABLE,
    }
