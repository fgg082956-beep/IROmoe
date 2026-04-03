"""
news_sentiment.py — сбор заголовков и сентимент-анализ (v3)

Изменения v3:
- [I]  Фильтр стоп-слов — убирает спорт, букмекеры, рекламу
- [K]  Параллельный fetch новостей через ThreadPoolExecutor
- [R]  Вес источника: авторитетные домены ×1.5, сомнительные ×0.5
       Функция _get_source_weight() + поле source_weight в каждом заголовке
"""

import feedparser
import schedule
import time
import logging
import json
from datetime import datetime, timedelta
from urllib.parse import quote, urlparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)

# ── Импорт конфига ─────────────────────────────────────────────────────────
try:
    from config import TICKER_CONTEXT, NEWS_CATEGORY_WEIGHTS
    USE_CONTEXT = True
except ImportError:
    USE_CONTEXT = False

try:
    from config import TICKER_TO_COMPANY
except ImportError:
    TICKER_TO_COMPANY = {}

try:
    from config import (TICKERS, NEWS_MAX_HEADLINES, NEWS_MAX_AGE_HOURS,
                        NEWS_SCORE_CAP, NEWS_SCORE_WEIGHT)
except ImportError:
    from config import TICKERS
    NEWS_MAX_HEADLINES, NEWS_MAX_AGE_HOURS, NEWS_SCORE_CAP, NEWS_SCORE_WEIGHT = 10, 24, 3, 1.0

# [R] Trusted sources — импортируем из config если есть, иначе встроенный список
try:
    from config import TRUSTED_SOURCES
except ImportError:
    TRUSTED_SOURCES = {
        "high": [
            "tass.ru", "interfax.ru", "ria.ru", "rbc.ru",
            "cbr.ru", "moex.com", "kommersant.ru", "vedomosti.ru",
            "iz.rg.ru", "finmarket.ru", "1prime.ru",
        ],
        "low": [
            "zen.yandex.ru", "pikabu.ru", "fishki.net",
            "yaplakal.com", "woman.ru", "cosmo.ru",
        ],
    }

DATA_FILE = Path("news_sentiment_data.json")

# ── [I] Стоп-слова ──────────────────────────────────────────────────────────
_STOPWORDS = [
    "ставки", "букмекер", "прогноз матч", "футбол", "хоккей", "ставка на",
    "казино", "лотерея", "микрозайм", "микрокредит", "мфо", "займ",
    "крипто", "биткоин", "bitcoin", "криптовалют", "NFT", "форекс",
    "esports", "киберспорт", "betting",
]

def _is_stopword(title: str) -> bool:
    t = title.lower()
    return any(sw in t for sw in _STOPWORDS)


# ── [R] Вес источника ───────────────────────────────────────────────────────
def _get_source_weight(url: str) -> float:
    """
    Возвращает множитель доверия к источнику новости. [R]
      авторитетный домен → ×1.5
      сомнительный домен → ×0.5
      всё остальное      → ×1.0
    """
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return 1.0
    for trusted in TRUSTED_SOURCES.get("high", []):
        if domain.endswith(trusted):
            return 1.5
    for low in TRUSTED_SOURCES.get("low", []):
        if domain.endswith(low):
            return 0.5
    return 1.0


# ── Модель сентимент-анализа ────────────────────────────────────────────────
_sentiment_pipeline = None

def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:
            from transformers import pipeline
            logger.info("Загружаю модель (~29 МБ, первый раз)...")
            _sentiment_pipeline = pipeline(
                model="mxlcw/rubert-tiny2-russian-financial-sentiment",
                top_k=None)
            logger.info("Модель загружена.")
        except Exception as e:
            logger.error(f"Не удалось загрузить модель: {e}")
    return _sentiment_pipeline


# ── Получение новостей ──────────────────────────────────────────────────────
def fetch_google_news(query: str, max_items: int = NEWS_MAX_HEADLINES) -> list:
    """Получает заголовки из Google News RSS по поисковому запросу."""
    url = (f"https://news.google.com/rss/search"
           f"?q={quote(query)}&hl=ru&gl=RU&ceid=RU:ru")
    headlines = []
    cutoff = datetime.utcnow() - timedelta(hours=NEWS_MAX_AGE_HOURS)
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:max_items]:
            pub_dt = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub_dt = datetime(*entry.published_parsed[:6])
            if pub_dt and pub_dt < cutoff:
                continue
            title = entry.get("title", "").strip()
            if _is_stopword(title):          # [I]
                continue
            link = entry.get("link", "")
            headlines.append({
                "title":     title,
                "link":      link,
                "published": pub_dt.isoformat() if pub_dt else None,
                "source":    "google_news",
            })
    except Exception as e:
        logger.warning(f"Google News [{query}]: {e}")
    return headlines


def fetch_moex_news(max_items: int = 30) -> list:
    """Получает официальные новости Московской Биржи (RSS)."""
    url = "https://www.moex.com/export/news.aspx?cat=200"
    headlines = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:max_items]:
            pub_dt = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub_dt = datetime(*entry.published_parsed[:6])
            title = entry.get("title", "").strip()
            if _is_stopword(title):          # [I]
                continue
            headlines.append({
                "title":     title,
                "link":      entry.get("link", ""),
                "published": pub_dt.isoformat() if pub_dt else None,
                "source":    "moex_official",
            })
    except Exception as e:
        logger.warning(f"MOEX RSS: {e}")
    return headlines


# ── Сентимент-анализ одного заголовка ──────────────────────────────────────
def analyze_sentiment(text: str) -> dict:
    """Возвращает label, score (уверенность), sentiment_score (+1/0/-1)."""
    model = get_sentiment_pipeline()
    if model is None:
        return {"label": "neutral", "score": 0.0, "sentiment_score": 0}
    try:
        results = model(text[:512])
        best  = max(results[0], key=lambda x: x["score"])
        label = best["label"].lower()
        score = best["score"]
        return {
            "label": label,
            "score": round(score, 4),
            "sentiment_score": {"positive": 1, "negative": -1}.get(label, 0),
        }
    except Exception as e:
        logger.warning(f"Сентимент-анализ: {e}")
        return {"label": "neutral", "score": 0.0, "sentiment_score": 0}


# ── Сбор новостей по тикеру ─────────────────────────────────────────────────
def _fetch_one_category(category: str, queries: list, weight: float) -> list:
    """Вспомогательная функция для параллельного fetch. [K]"""
    items = []
    for query in queries:
        fetched = fetch_google_news(query, max_items=5)
        for h in fetched:
            h["category"]      = category
            h["weight"]        = weight
            h["source_weight"] = _get_source_weight(h.get("link", ""))  # [R]
        items.extend(fetched)
    return items


def _collect_headlines_for_ticker(ticker: str) -> list:
    """
    Собирает заголовки по всем категориям запросов для тикера.
    Каждому заголовку проставляется поле 'weight' и 'source_weight'. [K][R]
    """
    all_headlines = []

    if USE_CONTEXT and ticker in TICKER_CONTEXT:
        ctx     = TICKER_CONTEXT[ticker]
        weights = NEWS_CATEGORY_WEIGHTS
        # [K] Параллельный fetch по категориям
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {
                ex.submit(_fetch_one_category, cat, queries, weights.get(cat, 1.0)): cat
                for cat, queries in ctx.items()
            }
            for future in as_completed(futures):
                try:
                    all_headlines.extend(future.result())
                except Exception as e:
                    logger.warning(f"[K] Ошибка fetch категории: {e}")
    else:
        company = TICKER_TO_COMPANY.get(ticker, ticker)
        items   = fetch_google_news(company, max_items=NEWS_MAX_HEADLINES)
        for h in items:
            h["category"]      = "company"
            h["weight"]        = 1.0
            h["source_weight"] = _get_source_weight(h.get("link", ""))  # [R]
        all_headlines.extend(items)

    # Дополняем MOEX-новостями с упоминанием тикера
    moex_news    = fetch_moex_news()
    company_name = (TICKER_CONTEXT[ticker]["company"][0]
                    if USE_CONTEXT and ticker in TICKER_CONTEXT
                    else TICKER_TO_COMPANY.get(ticker, ""))
    for item in moex_news:
        t = item["title"].lower()
        if ticker.lower() in t or company_name.lower() in t:
            item["category"]      = "company"
            item["weight"]        = 1.0
            item["source_weight"] = _get_source_weight(item.get("link", ""))  # [R]
            all_headlines.append(item)

    # Дедупликация по URL и заголовку
    seen_urls, seen_titles, unique = set(), set(), []
    for h in all_headlines:
        key   = h.get("link") or h.get("title")
        title = h.get("title", "")
        if key not in seen_urls and title not in seen_titles:
            seen_urls.add(key)
            seen_titles.add(title)
            unique.append(h)

    return unique


# ── Взвешенный подсчёт сентимента ──────────────────────────────────────────
def _compute_weighted_sentiment(headlines: list) -> dict:
    """
    Анализирует тональность каждого заголовка и суммирует с учётом весов.
    Итоговый вес = category_weight × source_weight. [R]
    Финальный балл ограничивается ±NEWS_SCORE_CAP.
    """
    total_weighted = 0.0
    counts  = {"positive": 0, "negative": 0, "neutral": 0}
    analyzed = []

    for h in headlines:
        sentiment = analyze_sentiment(h["title"])
        h.update(sentiment)
        analyzed.append(h)

        cat_weight = h.get("weight", 1.0)
        src_weight = h.get("source_weight", 1.0)    # [R]
        eff_weight = cat_weight * src_weight         # [R] итоговый вес
        h["eff_weight"] = round(eff_weight, 3)       # сохраняем для отладки

        total_weighted += sentiment["sentiment_score"] * eff_weight
        counts[sentiment["label"]] = counts.get(sentiment["label"], 0) + 1

    raw_score = round(total_weighted * NEWS_SCORE_WEIGHT, 2)
    capped    = max(-NEWS_SCORE_CAP, min(NEWS_SCORE_CAP, int(round(raw_score))))

    return {
        "sentiment_score": capped,
        "raw_score":       raw_score,
        **counts,
        "headlines":       analyzed,
    }


# ── Основной цикл ───────────────────────────────────────────────────────────
def collect_and_analyze() -> dict:
    logger.info("=" * 50)
    logger.info(f"Сбор новостей: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if USE_CONTEXT:
        logger.info("Режим: расширенные словари (TICKER_CONTEXT)")
    else:
        logger.info("Режим: упрощённый (TICKER_TO_COMPANY)")

    all_data = load_data()
    results  = {}

    for ticker in TICKERS:
        logger.info(f"[{ticker}] Сбор заголовков...")
        headlines = _collect_headlines_for_ticker(ticker)
        logger.info(f"[{ticker}] Уникальных заголовков: {len(headlines)}")

        result = _compute_weighted_sentiment(headlines)

        cat_stats = {}
        for h in headlines:
            cat = h.get("category", "?")
            cat_stats[cat] = cat_stats.get(cat, 0) + 1

        score = result["sentiment_score"]
        sign  = "+" if score > 0 else ""
        logger.info(
            f"[{ticker}] Балл={sign}{score} (сырой={result['raw_score']:+.1f}) | "
            f"+{result['positive']} -{result['negative']} ~{result['neutral']} | "
            f"категории: {cat_stats}"
        )

        ticker_result = {
            "timestamp":       datetime.now().isoformat(),
            "sentiment_score": result["sentiment_score"],
            "raw_score":       result["raw_score"],
            "positive":        result["positive"],
            "negative":        result["negative"],
            "neutral":         result["neutral"],
            "total_headlines": len(headlines),
            "headlines":       result["headlines"],
        }
        results[ticker] = ticker_result

        if ticker not in all_data:
            all_data[ticker] = []
        all_data[ticker].append(ticker_result)

    save_data(all_data)
    logger.info("Данные сохранены → news_sentiment_data.json")
    return results


# ── Хранение данных ─────────────────────────────────────────────────────────
def load_data() -> dict:
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_data(data: dict) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_latest_scores() -> dict:
    data   = load_data()
    scores = {}
    for ticker, history in data.items():
        if history:
            scores[ticker] = history[-1]["sentiment_score"]
    return scores

def get_sentiment_addon(ticker: str) -> int:
    """Интеграционная функция — возвращает новостной балл тикера."""
    return get_latest_scores().get(ticker, 0)


# ── Планировщик ─────────────────────────────────────────────────────────────
def start_scheduler(interval_minutes: int = 60) -> None:
    logger.info(f"Планировщик запущен. Интервал: {interval_minutes} мин.")
    collect_and_analyze()
    schedule.every(interval_minutes).minutes.do(collect_and_analyze)
    while True:
        schedule.run_pending()
        time.sleep(30)


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Новостной сентимент для акций MOEX (v3)")
    parser.add_argument("--once",     action="store_true", help="Один цикл сбора и выход")
    parser.add_argument("--interval", type=int, default=60, help="Интервал в минутах")
    parser.add_argument("--scores",   action="store_true", help="Показать последние баллы из кэша")
    parser.add_argument("--ticker",   type=str, default=None, help="Детали по тикеру")
    args = parser.parse_args()

    if args.scores:
        scores = get_latest_scores()
        if scores:
            print("\n=== Последние сентимент-баллы ===")
            for t, s in sorted(scores.items(), key=lambda x: -x[1]):
                bar  = "█" * min(abs(s), 10) + "░" * (10 - min(abs(s), 10))
                sign = "+" if s > 0 else ""
                print(f"  {t:6s} {bar} {sign}{s}")
        else:
            print("Данных нет. Запусти: python news_sentiment.py --once")

    elif args.ticker:
        data = load_data()
        t    = args.ticker.upper()
        if t in data and data[t]:
            last = data[t][-1]
            print(f"\n=== {t} — последний анализ ===")
            print(f"  Время: {last['timestamp']}")
            print(f"  Балл: {last['sentiment_score']:+d} (сырой: {last.get('raw_score', '?'):+.1f})")
            print(f"  +{last['positive']} / -{last['negative']} / ~{last['neutral']}")
            print("\n  Заголовки:")
            for h in last.get("headlines", [])[:15]:
                cat   = h.get("category", "?")
                label = h.get("label", "?")
                w     = h.get("eff_weight", h.get("weight", 1.0))
                emoji = {"positive": "✅", "negative": "❌", "neutral": "➖"}.get(label, "?")
                sw    = h.get("source_weight", 1.0)
                src_tag = f" src×{sw:.1f}" if sw != 1.0 else ""
                print(f"    {emoji} [{cat:9s} ×{w:.2f}{src_tag}] {h['title'][:75]}")
        else:
            print(f"Нет данных по тикеру {t}")

    elif args.once:
        results = collect_and_analyze()
        print("\n=== Итог ===")
        for ticker, d in sorted(results.items(), key=lambda x: -x[1]["sentiment_score"]):
            s    = d["sentiment_score"]
            sign = "+" if s > 0 else ""
            print(f"  {ticker:6s} {sign}{s:2d} ({d['total_headlines']} заголовков)")

    else:
        start_scheduler(interval_minutes=args.interval)
