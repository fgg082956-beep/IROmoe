# ============================================================
# bot.py — Telegram-бот (v2)
# ============================================================
# [J] /history TICKER — история сентимент-баллов (последние 7)
# Полный файл (сокращён для записи, логика сохранена)

import time
import logging
import requests
from datetime import datetime
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BOT] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
_offset  = 0


def _get_updates(offset=0, timeout=30):
    try:
        r = requests.get(f"{BASE_URL}/getUpdates",
                         params={"offset": offset, "timeout": timeout,
                                 "allowed_updates": ["message","callback_query"]},
                         timeout=timeout+5)
        if r.status_code == 200:
            return r.json().get("result", [])
    except Exception as e:
        logger.warning(f"getUpdates: {e}")
    return []


def _send(chat_id, text, reply_markup=None):
    payload = {"chat_id": str(chat_id), "text": text[:4096],
               "parse_mode": "Markdown", "disable_web_page_preview": True}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    try:
        r = requests.post(f"{BASE_URL}/sendMessage", json=payload, timeout=10)
        if r.status_code == 200:
            return r.json()["result"]["message_id"]
    except Exception as e:
        logger.warning(f"sendMessage: {e}")
    return None


def _edit(chat_id, msg_id, text, reply_markup=None):
    payload = {"chat_id": str(chat_id), "message_id": msg_id,
               "text": text[:4096], "parse_mode": "Markdown",
               "disable_web_page_preview": True}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    try:
        r = requests.post(f"{BASE_URL}/editMessageText", json=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def _answer_callback(cq_id, text="", show_alert=False):
    try:
        requests.post(f"{BASE_URL}/answerCallbackQuery",
                      json={"callback_query_id": cq_id, "text": text,
                            "show_alert": show_alert}, timeout=5)
    except Exception:
        pass


def _typing(chat_id):
    try:
        requests.post(f"{BASE_URL}/sendChatAction",
                      json={"chat_id": str(chat_id), "action": "typing"}, timeout=5)
    except Exception:
        pass


def _try_import():
    m = {}
    try:
        from ideas_generator import generate_all_ideas
        m["generate_all_ideas"] = generate_all_ideas
    except ImportError as e:
        logger.error(f"ideas_generator: {e}")
    try:
        from news_sentiment import load_data, get_latest_scores
        m["ns_load"] = load_data
        m["get_latest_scores"] = get_latest_scores
    except ImportError as e:
        logger.warning(f"news_sentiment: {e}")
    try:
        from notifier import (send_ideas_one_by_one, format_single_idea_message,
                               _make_idea_keyboard)
        m["send_ideas_one_by_one"] = send_ideas_one_by_one
        m["format_single_idea_message"] = format_single_idea_message
        m["_make_idea_keyboard"] = _make_idea_keyboard
    except ImportError as e:
        logger.error(f"notifier: {e}")
    return m


_M = _try_import()


def _handle_start(chat_id):
    text = (
        "👋 *MOEX Trade Ideas Bot v2*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "📊 /ideas — полный анализ прямо сейчас\n"
        "📰 /news `SBER` — сентимент по тикеру\n"
        "📈 /history `SBER` — история баллов (7 замеров)\n"
        "📋 /scores — баллы по всем тикерам\n"
        "❓ /help — эта справка\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "⚠️ _Только авторизованный чат._"
    )
    _send(chat_id, text)


def _handle_ideas(chat_id):
    gen  = _M.get("generate_all_ideas")
    send = _M.get("send_ideas_one_by_one")
    if not gen or not send:
        _send(chat_id, "❌ Модуль анализа недоступен.")
        return
    _send(chat_id, "⏳ Запускаю анализ... _(1–2 минуты)_")
    _typing(chat_id)
    try:
        data = gen()
        send(data)
    except Exception as e:
        logger.error(f"/ideas: {e}")
        _send(chat_id, f"❌ Ошибка: `{e}`")


def _handle_news(chat_id, ticker):
    ns_load = _M.get("ns_load")
    if not ns_load:
        _send(chat_id, "❌ Модуль news_sentiment недоступен.")
        return
    ticker = ticker.upper().strip()
    if not ticker:
        _send(chat_id, "⚠️ Укажи тикер: `/news SBER`")
        return
    try:
        data = ns_load()
    except Exception as e:
        _send(chat_id, f"❌ Ошибка: `{e}`")
        return
    if ticker not in data or not data[ticker]:
        _send(chat_id, f"❓ Нет данных по *{ticker}*. Запусти /ideas.")
        return

    last  = data[ticker][-1]
    score = last["sentiment_score"]
    sign  = "+" if score > 0 else ""
    bar   = "█" * min(abs(score), 10) + "░" * (10 - min(abs(score), 10))
    ts    = last.get("timestamp", "")[:16].replace("T", " ")
    em    = "🟢" if score > 0 else ("🔴" if score < 0 else "⚪")

    emoji_map = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
    score_map = {"positive": "+1", "negative": "−1", "neutral": " 0"}

    lines = [
        f"📰 *Новостной сентимент: {ticker}*",
        f"🕐 {ts}", "━━━━━━━━━━━━━━━━━━━━━━━━",
        f"{em} Балл: `{bar}` {sign}{score}",
        f"✅ {last['positive']}  ❌ {last['negative']}  ➖ {last['neutral']}",
        "", "*Последние заголовки:*",
    ]
    headlines = sorted(last.get("headlines", []),
                       key=lambda h: (0 if h.get("label") != "neutral" else 1,
                                      -h.get("score", 0)))
    for h in headlines[:10]:
        lbl   = h.get("label", "neutral")
        conf  = int(h.get("score", 0) * 100)
        w     = h.get("weight", 1.0)
        title = h.get("title", "")[:70]
        lines.append(
            f"  {emoji_map.get(lbl,'⚪')} "
            f"`[{score_map.get(lbl,' 0')}, {conf}%, x{w:.2f}]` {title}"
        )
    tv_url = f"https://ru.tradingview.com/chart/?symbol=MOEX%3A{ticker}"
    markup = {"inline_keyboard": [[
        {"text": f"📈 График {ticker}", "url": tv_url},
        {"text": "🔄 Обновить", "callback_data": f"news_{ticker}"},
    ]]}
    _send(chat_id, "\n".join(lines), reply_markup=markup)


# ── [J] /history TICKER ──────────────────────────────────────
def _handle_history(chat_id, ticker):
    """История сентимент-баллов: последние 7 замеров со стрелками тренда."""
    ns_load = _M.get("ns_load")
    if not ns_load:
        _send(chat_id, "❌ Модуль news_sentiment недоступен.")
        return
    ticker = ticker.upper().strip()
    if not ticker:
        _send(chat_id, "⚠️ Укажи тикер: `/history SBER`")
        return
    try:
        data = ns_load()
    except Exception as e:
        _send(chat_id, f"❌ Ошибка: `{e}`")
        return
    if ticker not in data or not data[ticker]:
        _send(chat_id, f"❓ Нет истории по *{ticker}*.")
        return

    history = data[ticker]
    last_n  = history[-7:]
    lines   = [
        f"📈 *История сентимента: {ticker}*",
        f"_(последние {len(last_n)} замеров)_",
        "━━━━━━━━━━━━━━━━━━━━━━━━",
    ]
    prev_score = None
    for rec in last_n:
        s     = rec["sentiment_score"]
        ts    = rec.get("timestamp", "")[:16].replace("T", " ")
        bar   = "█" * min(abs(s), 10) + "░" * (10 - min(abs(s), 10))
        sign  = "+" if s > 0 else ""
        em    = "🟢" if s > 0 else ("🔴" if s < 0 else "⚪")
        arrow = ("↑" if s > prev_score else ("↓" if s < prev_score else "→")) \
                if prev_score is not None else "  "
        lines.append(f"{em} `{bar}` {sign}{s} {arrow}  _{ts}_")
        prev_score = s

    if len(last_n) >= 2:
        delta = last_n[-1]["sentiment_score"] - last_n[0]["sentiment_score"]
        trend = (f"↑ Улучшилось на +{delta}" if delta > 0
                 else (f"↓ Ухудшилось на {delta}" if delta < 0
                       else "→ Стабильно"))
        lines += ["", f"_{trend} за период_"]

    tv_url = f"https://ru.tradingview.com/chart/?symbol=MOEX%3A{ticker}"
    markup = {"inline_keyboard": [[
        {"text": f"📈 График {ticker}", "url": tv_url},
        {"text": "📰 Текущие новости", "callback_data": f"news_{ticker}"},
    ]]}
    _send(chat_id, "\n".join(lines), reply_markup=markup)


def _handle_scores(chat_id):
    get_scores = _M.get("get_latest_scores")
    if not get_scores:
        _send(chat_id, "❌ Модуль news_sentiment недоступен.")
        return
    try:
        scores = get_scores()
    except Exception as e:
        _send(chat_id, f"❌ Ошибка: `{e}`")
        return
    if not scores:
        _send(chat_id, "❓ Нет данных. Запусти /ideas.")
        return
    lines = ["📋 *Сентимент-баллы по тикерам*", "━━━━━━━━━━━━━━━━━━━━━━━━"]
    for t, s in sorted(scores.items(), key=lambda x: -x[1]):
        bar  = "█" * min(abs(s), 10) + "░" * (10 - min(abs(s), 10))
        sign = "+" if s > 0 else ""
        em   = "🟢" if s > 0 else ("🔴" if s < 0 else "⚪")
        lines.append(f"{em} `{t:5s}` `{bar}` {sign}{s}")
    _send(chat_id, "\n".join(lines))


def _handle_callback(update):
    cq      = update["callback_query"]
    cq_id   = cq["id"]
    chat_id = str(cq["message"]["chat"]["id"])
    msg_id  = cq["message"]["message_id"]
    cb_data = cq.get("data", "")
    if not _is_authorized(chat_id):
        _answer_callback(cq_id, "❌ Нет доступа", show_alert=True)
        return
    if cb_data.startswith("refresh_"):
        ticker = cb_data.removeprefix("refresh_")
        _answer_callback(cq_id, f"⏳ Обновляю {ticker}...")
        gen = _M.get("generate_all_ideas")
        fmt = _M.get("format_single_idea_message")
        mkb = _M.get("_make_idea_keyboard")
        if not gen or not fmt or not mkb:
            return
        try:
            d     = gen()
            all_i = ([(i, True) for i in d["long_ideas"]] +
                     [(i, False) for i in d["short_ideas"]])
            match = next(((i, il) for i, il in all_i if i["ticker"] == ticker), None)
            if match:
                idea, is_long = match
                new_text = fmt(idea, is_long, d["timestamp"],
                               d.get("index"), d.get("news_active", False))
                if not _edit(chat_id, msg_id, new_text, mkb(ticker)):
                    _send(chat_id, new_text, reply_markup=mkb(ticker))
            else:
                _edit(chat_id, msg_id,
                      f"ℹ️ *{ticker}* — сигнал исчез.\n_{d['timestamp']}_")
        except Exception as e:
            _answer_callback(cq_id, f"❌ {e}", show_alert=True)
    elif cb_data.startswith("news_"):
        ticker = cb_data.removeprefix("news_")
        _answer_callback(cq_id, f"⏳ Новости {ticker}...")
        _handle_news(chat_id, ticker)
    else:
        _answer_callback(cq_id, "?")


def _is_authorized(chat_id):
    return str(chat_id) == str(TELEGRAM_CHAT_ID)


def _process_update(update):
    global _offset
    _offset = max(_offset, update["update_id"] + 1)
    if "callback_query" in update:
        cq_chat = str(update["callback_query"]["message"]["chat"]["id"])
        if _is_authorized(cq_chat):
            _handle_callback(update)
        else:
            _answer_callback(update["callback_query"]["id"], "❌ Нет доступа", show_alert=True)
        return
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return
    chat_id = str(msg["chat"]["id"])
    text    = msg.get("text", "").strip()
    if not text.startswith("/"):
        return
    if not _is_authorized(chat_id):
        _send(chat_id, "❌ Нет доступа.")
        return
    parts   = text.split(maxsplit=1)
    command = parts[0].lower().split("@")[0]
    args    = parts[1].strip() if len(parts) > 1 else ""
    logger.info(f"Команда: {command!r} args={args!r}")
    dispatch = {
        "/start":   lambda: _handle_start(chat_id),
        "/help":    lambda: _handle_start(chat_id),
        "/ideas":   lambda: _handle_ideas(chat_id),
        "/news":    lambda: _handle_news(chat_id, args),
        "/scores":  lambda: _handle_scores(chat_id),
        "/history": lambda: _handle_history(chat_id, args),
    }
    handler = dispatch.get(command)
    if handler:
        handler()
    else:
        _send(chat_id, f"❓ Неизвестная команда: `{command}`\n/help — список.")


def run_polling():
    global _offset
    logger.info("=" * 50)
    logger.info("MOEX Trade Ideas Bot v2 запущен")
    logger.info(f"Chat ID: {TELEGRAM_CHAT_ID}")
    logger.info("=" * 50)
    while True:
        try:
            updates = _get_updates(offset=_offset, timeout=30)
            for upd in updates:
                try:
                    _process_update(upd)
                except Exception as e:
                    logger.error(f"Update {upd.get('update_id')}: {e}")
        except KeyboardInterrupt:
            logger.info("Бот остановлен.")
            break
        except Exception as e:
            logger.error(f"Polling: {e} — жду 5с...")
            time.sleep(5)


if __name__ == "__main__":
    if "СЮДА_ВСТАВЬ" in str(TELEGRAM_BOT_TOKEN):
        print("❌ Заполни TELEGRAM_BOT_TOKEN в config.py")
    else:
        run_polling()
