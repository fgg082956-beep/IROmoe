# notifier.py — Telegram уведомления (v8)
# [FIX] HTML-экранирование
# [M]  Строка режима рынка в шапке
# [P]  Строка состояния сектора в каждой идее
# [N]  Строка уровней из стакана в каждой идее
# [O]  Метка высокой волатильности

import requests, logging, json
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger       = logging.getLogger(__name__)
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
MAX_MSG_LEN  = 4096


def _escape(text) -> str:
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def send_telegram_message(text: str, reply_markup=None) -> bool:
    try:
        payload = {"chat_id": TELEGRAM_CHAT_ID,
                   "text": text, "parse_mode": "HTML"}
        if reply_markup:
            payload["reply_markup"] = json.dumps(reply_markup)
        r = requests.post(f"{TELEGRAM_API}/sendMessage", json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Telegram: {e}")
        return False


def _score_bar(score: int, max_score: int = 14) -> str:
    filled = min(abs(score), max_score)
    return "█" * filled + "░" * (max_score - filled)


def _news_block(headlines: list) -> str:
    if not headlines:
        return ""
    em = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
    sm = {"positive": "+1",  "negative": "−1",  "neutral": " 0"}
    top = sorted(
        headlines,
        key=lambda h: (0 if h.get("label", "neutral") == "neutral" else 1,
                       h.get("score", 0)),
        reverse=True,
    )[:5]
    lines = ["\n📰 <b>Новости:</b>"]
    for h in top:
        lbl   = h.get("label", "neutral")
        sc    = h.get("score", 0)
        w     = h.get("weight", 1.0)
        src_w = h.get("source_weight", 1.0)    # [R] вес источника
        title = _escape(h.get("title", ""))[:75]
        # Показываем вес источника только если он не 1.0
        src_tag = f" src×{src_w:.1f}" if src_w != 1.0 else ""
        lines.append(
            f"  {em.get(lbl, '?')} [{sm.get(lbl, ' 0')}, "
            f"{int(sc * 100)}%{src_tag}] {title}"
        )
    return "\n".join(lines)


def _make_idea_keyboard(ticker: str) -> dict:
    tv_url    = f"https://ru.tradingview.com/chart/?symbol=MOEX%3A{ticker}"
    finam_url = f"https://www.finam.ru/quote/moex/{ticker.lower()}/"
    return {
        "inline_keyboard": [
            [{"text": "📈 TradingView", "url": tv_url},
             {"text": "📊 Финам",       "url": finam_url}],
            [{"text": "🔄 Обновить анализ",
              "callback_data": f"refresh_{ticker}"}],
        ]
    }


def format_single_idea_message(idea: dict, index: int, total: int,
                                timestamp: str, index_data: dict = None,
                                regime: dict = None) -> str:
    sig    = idea["signal"]
    emoji  = "🟢" if sig == "ЛОНГ" else "🔴"
    ticker = _escape(idea["ticker"])
    score  = idea["score"]
    bar    = _score_bar(score)

    score_t = idea.get("score_tech", 0)
    score_n = idea.get("score_news", 0)
    score_s = idea.get("score_sector", 0)
    score_parts = f"тех: {score_t:+d}"
    if score_n: score_parts += f", новости: {score_n:+d}"
    if score_s: score_parts += f", сектор: {score_s:+d}"

    # Режим рынка [M]
    regime_str = ""
    if regime:
        regime_str = (f"{regime.get('emoji','?')} {_escape(regime.get('name',''))} | ")

    idx_str = ""
    if index_data and index_data.get("value"):
        idx_str = (f"IMOEX: <b>{index_data['value']:.0f}</b> "
                   f"({index_data['change_pct']:+.2f}%) | ")

    # Волатильность [O]
    vol_tag = " ⚠️ высокая волат." if idea.get("high_vol") else ""

    lines = [
        f"{emoji} <b>{ticker}</b>  •  {sig}  •  Идея {index}/{total}",
        f"{regime_str}{idx_str}{_escape(timestamp)}",
        "",
        f"💰 Цена: <b>{_escape(idea['current_price'])}</b>  |  "
        f"Изм.: {idea['change_pct']:+.1f}%{vol_tag}",
        f"📊 Балл: <b>{score:+d}</b> ({bar})",
        f"     [{score_parts}]",
        "",
        f"📈 Вход: <b>{_escape(idea['entry'])}</b>",
        f"🛑 Стоп: <b>{_escape(idea['stop_loss'])}</b>",
        f"🎯 TP1: <b>{_escape(idea['take_profit_1'])}</b>  |  "
        f"TP2: <b>{_escape(idea['take_profit_2'])}</b>",
        f"⚖️ RR: {_escape(idea['risk_reward'])}  |  Лотов: {idea['num_shares']}",
        f"💸 Риск: {idea['risk_rubles']}₽ ({idea['risk_pct']}%)",
        "",
    ]

    # Индикаторы
    ind = []
    if idea.get("rsi"):    ind.append(f"RSI {idea['rsi']:.0f}")
    if idea.get("ema50"):  ind.append(f"EMA50 {_escape(idea['ema50'])}")
    if idea.get("ema200"): ind.append(f"EMA200 {_escape(idea['ema200'])}")
    if ind:
        lines.append(f"📉 {'  |  '.join(ind)}")

    # [N] Уровни из стакана
    ob_s = idea.get("ob_support")
    ob_r = idea.get("ob_resistance")
    if ob_s or ob_r:
        parts = []
        if ob_s: parts.append(f"поддержка: <b>{ob_s}</b>")
        if ob_r: parts.append(f"сопротивление: <b>{ob_r}</b>")
        lines.append(f"📖 Стакан: {',  '.join(parts)}")

    # [P] Состояние сектора
    sector_info = idea.get("sector_info")
    if sector_info and sector_info.get("sector_key"):
        lines.append(f"🏭 {_escape(sector_info['label'])}")

    # Недельный тренд [TF]
    wt = idea.get("weekly_trend")
    if wt and wt != "нет данных":
        prefix = "✅" if ("растёт" in wt or "подтверждает" in wt) else "⚠️"
        lines.append(f"{prefix} Неделя: {_escape(wt)}")

    # Дивидендное предупреждение [DG]
    dw = idea.get("div_warning")
    if dw:
        lines.append(_escape(dw))

    lines.append("")

    # Причины сигнала
    reasons_clean = [
        r for r in idea.get("reasons", [])
        if not any(x in r for x in
                   ["⚠️ Недель", "✅ Недель", "отсечка", "Гэп", "Таймфрейм"])
        or r.startswith("⚠️ Высокая")
    ]
    if reasons_clean:
        lines.append("📋 <b>Сигналы:</b>")
        for r in reasons_clean[:8]:
            lines.append(f"  • {_escape(r)}")

    lines.append(_news_block(idea.get("news_headlines", [])))
    return "\n".join(str(l) for l in lines)


def send_ideas_one_by_one(data: dict):
    longs     = data.get("long_ideas",  [])
    shorts    = data.get("short_ideas", [])
    all_ideas = longs + shorts
    total     = len(all_ideas)
    regime    = data.get("regime")

    if total == 0:
        send_telegram_message("✅ Анализ завершён: перспективных идей не найдено.")
        return

    ts  = data.get("timestamp", "")
    idx = data.get("index")

    # [M] Режим рынка в шапке
    regime_header = ""
    if regime:
        regime_header = (
            f"\n{regime['emoji']} <b>{_escape(regime['name'])}</b>"
            f" | IMOEX {regime.get('imoex_last', '?')} "
            f"(EMA{20}: {regime.get('ema', '?')})"
        )

    bull    = len([i for i in all_ideas if i["signal"] == "ЛОНГ"])
    bear    = len([i for i in all_ideas if i["signal"] == "ШОРТ"])
    idx_str = ""
    if idx and idx.get("value"):
        idx_str = f"  |  IMOEX {idx['value']:.0f} ({idx['change_pct']:+.2f}%)"

    header = (
        f"📊 <b>Анализ MOEX</b> — {_escape(ts)}{idx_str}{regime_header}\n"
        f"Найдено: <b>{total}</b> (🟢 {bull} лонг, 🔴 {bear} шорт)"
    )
    send_telegram_message(header)

    for i, idea in enumerate(all_ideas, 1):
        msg = format_single_idea_message(
            idea, index=i, total=total, timestamp=ts,
            index_data=idx, regime=regime
        )
        send_telegram_message(msg, reply_markup=_make_idea_keyboard(idea["ticker"]))

    logger.info(f"Telegram: отправлено {total} сообщений")
