# ideas_generator.py — Генерация торговых идей (v8)
# [A]  Параллельный анализ через ThreadPoolExecutor (8 потоков)
# [TF] Недельные свечи передаются в analyze_ticker()
# [DG] Дивидендная проверка передаётся в analyze_ticker()
# [M]  Фильтр рыночного режима (market_regime.py)
# [P]  Секторный балл (sector_analyzer.py)
# [O]  Флаг высокой волатильности (ATR_HIGH_VOL_PCT)
# [N]  Уровни из стакана (ob_support / ob_resistance)

from concurrent.futures import ThreadPoolExecutor, as_completed
from config import DEPOSIT, MAX_RISK_PERCENT, TICKERS, TICKER_SIGNAL_FILTER
from data_fetcher import (get_candles, get_current_price, get_index_value,
                          get_candles_weekly, is_near_dividend_gap,
                          get_orderbook)    # [TF][DG][N]
from analyzer import analyze_ticker
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    from config import MIN_AVG_VOLUME
except ImportError:
    MIN_AVG_VOLUME = 0

try:
    from config import ATR_HIGH_VOL_PCT   # [O]
except ImportError:
    ATR_HIGH_VOL_PCT = 3.0   # порог высокой волатильности: ATR/цена > 3%

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

try:
    from market_regime import detect_regime, passes_regime_filter   # [M]
    REGIME_AVAILABLE = True
except ImportError:
    REGIME_AVAILABLE = False
    def detect_regime(): return None
    def passes_regime_filter(idea, regime): return True

try:
    from sector_analyzer import compute_sector_scores, get_sector_adj   # [P]
    SECTOR_AVAILABLE = True
except ImportError:
    SECTOR_AVAILABLE = False
    def compute_sector_scores(scores): return {}
    def get_sector_adj(ticker, sector_scores): return {"score_adj": 0, "sector_key": None, "sector_name": "нет данных", "sector_avg": None, "label": ""}



def _passes_ticker_filter(ticker: str, signal: str) -> bool:
    """Разрешён ли сигнал для тикера по TICKER_SIGNAL_FILTER [BT]."""
    allowed = TICKER_SIGNAL_FILTER.get(ticker)
    if allowed is None:
        return True
    return signal in allowed


def _passes_volume_filter(candles):
    if MIN_AVG_VOLUME <= 0 or candles.empty or "volume" not in candles.columns:
        return True, 0.0
    avg = candles["volume"].tail(20).mean()
    return (avg >= MIN_AVG_VOLUME, avg)


def _min_stop_dist(entry: float, atr, style: str) -> float:
    """[FIX] Нижний порог дистанции стопа: max(k*ATR, floor% от entry).
    Страхует от ситуации, когда ATR крошечный и стоп становится на 0.1% от цены —
    тогда RR формально огромный, но реально стоп выносит шумом.
    """
    k = 1.0 if style == "intraday" else 1.5
    atr_part = (atr or 0) * k if (atr and atr > 0) else 0.0
    floor_pct = 0.003 if style == "intraday" else 0.004  # 0.3% / 0.4%
    return max(atr_part, entry * floor_pct)


def calculate_trade_idea(ticker, price_data, analysis, style="swing",
                         sector_adj: dict = None, ob: dict = None):
    """
    sector_adj — результат get_sector_adj() [P]
    ob         — результат get_orderbook()  [N]
    """
    current_price = price_data["last"]
    if not current_price or current_price == 0:
        return None
    signal = analysis["signal"]
    if signal == "НЕЙТРАЛЬНО":
        return None
    support    = analysis.get("support")
    resistance = analysis.get("resistance")
    atr        = analysis.get("atr")
    atr_pct    = analysis.get("atr_pct", 0)

    # [N] Подтягиваем уровни из стакана, если есть
    ob_support    = ob.get("support")    if ob else None
    ob_resistance = ob.get("resistance") if ob else None

    if signal == "ЛОНГ":
        entry = (round(support * 1.002, 2)
                 if support and abs(current_price - support) / current_price < 0.03
                 else current_price)
        # [FIX] Единая дистанция стопа с защитой от крошечного ATR
        stop_dist = _min_stop_dist(entry, atr, style)
        atr_stop  = entry - stop_dist
        if support:
            stop_loss = round(min(support * 0.995, atr_stop), 2)
        else:
            stop_loss = round(atr_stop, 2)

        # TP: предпочитаем уровень стакана, если он выше entry [N]
        res_level = (ob_resistance
                     if ob_resistance and ob_resistance > entry
                     else resistance)
        if res_level:
            tp1 = round(res_level * 0.998, 2); tp2 = round(res_level * 1.02, 2)
        elif atr and atr > 0:
            tp1 = round(entry + 2*atr, 2);      tp2 = round(entry + 3*atr, 2)
        else:
            tp1 = round(entry * 1.03, 2);        tp2 = round(entry * 1.05, 2)
    else:
        entry = (round(resistance * 0.998, 2)
                 if resistance and abs(current_price - resistance) / current_price < 0.03
                 else current_price)
        # [FIX] Единая дистанция стопа для шорта
        stop_dist = _min_stop_dist(entry, atr, style)
        atr_stop  = entry + stop_dist
        if resistance:
            stop_loss = round(max(resistance * 1.005, atr_stop), 2)
        else:
            stop_loss = round(atr_stop, 2)

        # TP: предпочитаем уровень стакана, если он ниже entry [N]
        sup_level = (ob_support
                     if ob_support and ob_support < entry
                     else support)
        if sup_level:
            tp1 = round(sup_level * 1.002, 2); tp2 = round(sup_level * 0.98, 2)
        elif atr and atr > 0:
            tp1 = round(entry - 2*atr, 2);     tp2 = round(entry - 3*atr, 2)
        else:
            tp1 = round(entry * 0.97, 2);       tp2 = round(entry * 0.95, 2)

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
    # [FIX] Явный cap по риску — страховка от мелкого ATR/шумного стопа
    if risk_pct > MAX_RISK_PERCENT + 0.01:
        logger.info(f"  {ticker}: risk_pct {risk_pct}% > MAX {MAX_RISK_PERCENT}%, skip")
        return None

    # [P] Секторный балл
    s_adj       = sector_adj or {}
    score_sector = s_adj.get("score_adj", 0)
    sector_info  = {
        "sector_key":  s_adj.get("sector_key"),
        "sector_name": s_adj.get("sector_name", ""),
        "sector_avg":  s_adj.get("sector_avg"),
        "label":       s_adj.get("label", ""),
    }
    total_score = analysis["score"] + score_sector

    # [O] Высокая волатильность
    high_vol = (atr_pct >= ATR_HIGH_VOL_PCT) if atr_pct else False

    return {
        "ticker": ticker, "signal": signal, "style": style,
        "current_price": current_price, "entry": entry,
        "stop_loss": stop_loss, "take_profit_1": tp1, "take_profit_2": tp2,
        "risk_reward": f"1:{rr}", "num_shares": n_shares,
        "risk_rubles": int(risk_rub), "risk_pct": risk_pct,
        "potential_profit": int(pot_profit),
        "rsi": analysis["rsi"], "score": total_score,
        "score_tech": analysis["score_tech"], "score_news": analysis["score_news"],
        "score_sector": score_sector,           # [P]
        "sector_info":  sector_info,            # [P]
        "reasons": analysis["reasons"], "change_pct": price_data.get("change_pct", 0),
        "news_headlines": _get_news_headlines(ticker),
        "weekly_trend": analysis.get("weekly_trend", "нет данных"),
        "div_warning":  analysis.get("div_warning"),
        "ema50": analysis.get("ema50"), "ema200": analysis.get("ema200"),
        "obv_trend": analysis.get("obv_trend"), "candle_pats": analysis.get("candle_pats", []),
        "ob_support":    ob_support,            # [N]
        "ob_resistance": ob_resistance,         # [N]
        "high_vol":      high_vol,              # [O]
        "atr_pct":       atr_pct,
    }


# Общий кэш сырых баллов для секторного анализа [P]
_raw_scores: dict = {}


def _analyze_one(ticker):
    global _raw_scores
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

        # [N] Стакан
        ob = get_orderbook(ticker)

        news_score = get_sentiment_addon(ticker)
        analysis   = analyze_ticker(
            candles, price["last"],
            news_sentiment_score=news_score,
            df_weekly=candles_w,   # [TF]
            div_info=div_info,     # [DG]
        )

        # Сохраняем сырой балл для секторного расчёта [P]
        _raw_scores[ticker] = analysis["score"]

        # Лог
        tf_str = (f", таймфрейм: {analysis.get('weekly_trend','?')[:25]}"
                  if analysis.get('weekly_trend') else "")
        dg_str = f", ⚠️ дивотсечка" if div_info.get("near") else ""
        extra  = f", новости: {analysis['score_news']:+d}" if NEWS_AVAILABLE else ""
        logger.info(f"  {ticker}: {analysis['signal']} "
                    f"(RSI {analysis['rsi']}, тех: {analysis['score_tech']:+d}"
                    f"{extra}{tf_str}{dg_str}, итого: {analysis['score']:+d})")

        # sector_adj пока None — добавим после сбора всех баллов [P]
        return (ticker, price, analysis, ob)
    except Exception as e:
        logger.error(f"  {ticker}: ошибка — {e}", exc_info=True)
        return None


def generate_all_ideas(max_workers: int = 8) -> dict:
    global _raw_scores
    _raw_scores = {}

    logger.info("=" * 60)
    logger.info(f"Генерация идей: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    logger.info(f"Тикеров: {len(TICKERS)}, потоков: {max_workers}")
    logger.info("=" * 60)

    # [M] Режим рынка — определяем один раз до анализа тикеров
    regime = None
    if REGIME_AVAILABLE:
        try:
            regime = detect_regime()
            logger.info(f"[M] Рыночный режим: {regime.get('emoji','')} {regime.get('name','')}")
        except Exception as e:
            logger.warning(f"[M] Режим рынка недоступен: {e}")

    index_data = get_index_value()
    if index_data:
        logger.info(f"IMOEX: {index_data['value']:.0f} ({index_data['change_pct']:+.2f}%)")

    # ── Шаг 1: параллельный анализ всех тикеров [A] ──────────────────────────
    raw_results: list = []   # [(ticker, price, analysis, ob), ...]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_analyze_one, t): t for t in TICKERS}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                raw_results.append(result)

    # ── Шаг 2: секторные баллы (после сбора всех сырых) [P] ──────────────────
    sector_scores = {}
    if SECTOR_AVAILABLE and _raw_scores:
        try:
            sector_scores = compute_sector_scores(_raw_scores)
            for sk, sv in sector_scores.items():
                logger.info(f"[P] Сектор {sk}: avg={sv['avg']:+.2f}, тикеров: {sv['count']}")
        except Exception as e:
            logger.warning(f"[P] Секторный анализ: {e}")

    # ── Шаг 3: формируем идеи с учётом сектора и фильтра режима ──────────────
    long_ideas, short_ideas = [], []

    for ticker, price, analysis, ob in raw_results:
        # Секторная корректировка [P]
        s_adj = {}
        if SECTOR_AVAILABLE and sector_scores:
            try:
                s_adj = get_sector_adj(ticker, sector_scores)
            except Exception as e:
                logger.warning(f"[P] {ticker} сектор: {e}")

        idea = calculate_trade_idea(ticker, price, analysis,
                                    style="swing",
                                    sector_adj=s_adj, ob=ob)
        if idea is None:
            continue

        # [BT] Тикер-специфичный фильтр
        if not _passes_ticker_filter(ticker, idea['signal']):
            logger.info(f"  [BT] {ticker} {idea['signal']} отброшен (TICKER_SIGNAL_FILTER)")
            continue

        # [M] Фильтр рыночного режима
        if regime and not passes_regime_filter(idea, regime):
            logger.info(f"  [M] {ticker} отфильтрован режимом {regime.get('name','')}")
            continue

        # [O] Лог высокой волатильности
        if idea.get("high_vol"):
            logger.info(f"  [O] {ticker}: высокая волатильность ATR={idea['atr_pct']:.1f}%")

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
        "regime":      regime,          # [M] передаётся в notifier
    }
