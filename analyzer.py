# analyzer.py — Технический анализ (v5)
# [C]  RSI: avg_loss.replace(0, 1e-10) — защита от деления на ноль
# [E]  EMA50/200: золотой/мёртвый крест (+/-2)
# [F]  OBV: дивергенция (+/-1)
# [G]  Дивергенция RSI (+/-2)
# [H]  Свечные паттерны (+/-1)
# [TF] Таймфреймовое подтверждение — недельный тренд (-2…+1)
# [DG] Дивидендный гэп — подавление/коррекция сигнала

import pandas as pd
import numpy as np
from config import RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL
from config import NEWS_SCORE_WEIGHT, NEWS_SCORE_CAP
from config import SCORE_LONG_THRESHOLD, SCORE_SHORT_THRESHOLD


# ── RSI ───────────────────────────────────────────────────────
def calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta    = series.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    for i in range(period, len(series)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period
    avg_loss = avg_loss.replace(0, 1e-10)   # [C]
    return 100 - (100 / (1 + avg_gain / avg_loss))


def calculate_macd(series: pd.Series) -> dict:
    ef = series.ewm(span=MACD_FAST, adjust=False).mean()
    es = series.ewm(span=MACD_SLOW, adjust=False).mean()
    ml = ef - es
    sg = ml.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return {"macd": ml, "signal": sg, "histogram": ml - sg}


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ── EMA тренд [E] ─────────────────────────────────────────────
def analyze_ema_trend(df: pd.DataFrame) -> dict:
    close  = df["close"]
    ema50  = calculate_ema(close, 50)
    ema200 = calculate_ema(close, 200)
    c50, c200 = ema50.iloc[-1], ema200.iloc[-1]
    p50, p200 = (ema50.iloc[-2], ema200.iloc[-2]) if len(df) > 1 else (c50, c200)
    golden = (p50 < p200) and (c50 >= c200)
    death  = (p50 > p200) and (c50 <= c200)
    if golden:
        return {"score":  2, "label": f"EMA50/200: золотой крест",
                "ema50": round(c50,2), "ema200": round(c200,2)}
    if death:
        return {"score": -2, "label": f"EMA50/200: мёртвый крест",
                "ema50": round(c50,2), "ema200": round(c200,2)}
    if c50 > c200:
        return {"score":  1, "label": f"EMA50 > EMA200 — тренд вверх",
                "ema50": round(c50,2), "ema200": round(c200,2)}
    return {"score": -1, "label": f"EMA50 < EMA200 — тренд вниз",
            "ema50": round(c50,2), "ema200": round(c200,2)}


# ── OBV [F] ───────────────────────────────────────────────────
def calculate_obv(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(dtype=float)
    close, vol = df["close"], df["volume"]
    obv = [0.0]
    for i in range(1, len(close)):
        if   close.iloc[i] > close.iloc[i-1]: obv.append(obv[-1] + vol.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]: obv.append(obv[-1] - vol.iloc[i])
        else:                                  obv.append(obv[-1])
    return pd.Series(obv, index=df.index)


def analyze_obv(df: pd.DataFrame) -> dict:
    obv = calculate_obv(df)
    if obv.empty or len(obv) < 10:
        return {"score": 0, "label": "", "obv_trend": "нет данных"}
    r_obv = obv.iloc[-10:].mean()
    p_obv = obv.iloc[-20:-10].mean() if len(obv) >= 20 else obv.iloc[0]
    r_cl  = df["close"].iloc[-10:].mean()
    p_cl  = df["close"].iloc[-20:-10].mean() if len(df) >= 20 else df["close"].iloc[0]
    if not (r_cl > p_cl) and (r_obv > p_obv):
        return {"score":  1, "label": "OBV: бычья дивергенция",
                "obv_trend": "растёт"}
    if (r_cl > p_cl) and not (r_obv > p_obv):
        return {"score": -1, "label": "OBV: медвежья дивергенция",
                "obv_trend": "падает"}
    return {"score": 0, "label": "", "obv_trend": "растёт" if r_obv > p_obv else "падает"}


# ── Дивергенция RSI [G] ───────────────────────────────────────
def analyze_rsi_divergence(df: pd.DataFrame, rsi: pd.Series, lookback: int = 14) -> dict:
    if len(df) < lookback * 2:
        return {"score": 0, "label": ""}
    close = df["close"]
    cmnp, pmnp = close.iloc[-lookback:].min(), close.iloc[-lookback*2:-lookback].min()
    cmnr, pmnr = rsi.iloc[-lookback:].min(),   rsi.iloc[-lookback*2:-lookback].min()
    cmxp, pmxp = close.iloc[-lookback:].max(), close.iloc[-lookback*2:-lookback].max()
    cmxr, pmxr = rsi.iloc[-lookback:].max(),   rsi.iloc[-lookback*2:-lookback].max()
    if any(pd.isna(v) for v in [cmnp, pmnp, cmnr, pmnr, cmxp, pmxp, cmxr, pmxr]):
        return {"score": 0, "label": ""}
    if cmnp < pmnp and cmnr > pmnr:
        return {"score":  2, "label": f"RSI дивергенция бычья"}
    if cmxp > pmxp and cmxr < pmxr:
        return {"score": -2, "label": f"RSI дивергенция медвежья"}
    return {"score": 0, "label": ""}


# ── Свечные паттерны [H] ──────────────────────────────────────
def analyze_candle_patterns(df: pd.DataFrame) -> dict:
    if len(df) < 2:
        return {"score": 0, "patterns": []}
    c, p = df.iloc[-1], df.iloc[-2]
    body = abs(c["close"] - c["open"])
    rng  = c["high"] - c["low"]
    if rng == 0:
        return {"score": 0, "patterns": []}
    upper_sh = c["high"] - max(c["close"], c["open"])
    lower_sh = min(c["close"], c["open"]) - c["low"]
    score, patterns = 0, []
    if lower_sh >= 2*body and upper_sh <= 0.1*rng and body/rng < 0.35:
        is_down = (df["close"].iloc[-5:].is_monotonic_decreasing
                   if len(df) >= 5 else df["close"].iloc[-1] < df["close"].iloc[0])
        if is_down:  score += 1; patterns.append("Молот — бычий разворот")
        else:        score -= 1; patterns.append("Висельник — медвежий разворот")
    if body/rng < 0.1:
        patterns.append("Доджи — нерешительность рынка")
    if (c["close"] > c["open"] and p["close"] < p["open"] and
            c["close"] > p["open"] and c["open"] < p["close"]):
        score += 1; patterns.append("Бычье поглощение")
    elif (c["close"] < c["open"] and p["close"] > p["open"] and
            c["close"] < p["open"] and c["open"] > p["close"]):
        score -= 1; patterns.append("Медвежье поглощение")
    return {"score": score, "patterns": patterns}


# ── [TF] Недельный тренд ─────────────────────────────────────
def analyze_weekly_trend(df_weekly: pd.DataFrame) -> dict:
    """
    Проверяет направление EMA20 на недельном таймфрейме.
    Возвращает score: +1 (совпадает с дневным ростом),
                      -1 (недельный медвежий),
                       0 (данных нет / нейтрально).
    Вызывается из analyze_ticker() и сравнивается с дневным сигналом.
    [TF]
    """
    if df_weekly is None or len(df_weekly) < 20:
        return {"score": 0, "label": ""}
    close  = df_weekly["close"]
    ema20w = close.ewm(span=20, adjust=False).mean()
    last   = ema20w.iloc[-1]
    prev   = ema20w.iloc[-2] if len(ema20w) > 1 else last
    slope  = last - prev
    if   slope > 0: return {"score":  1, "label": f"Недельный EMA20 растёт ({last:.1f})"}
    elif slope < 0: return {"score": -1, "label": f"Недельный EMA20 падает ({last:.1f})"}
    return {"score": 0, "label": ""}


# ── [DG] Дивидендный гэп ─────────────────────────────────────
def analyze_dividend_gap(div_info: dict, signal: str) -> dict:
    """
    Проверяет близость дивидендной отсечки и корректирует логику.

    Логика:
      • За 1–5 дней ДО отсечки (direction=before):
        - Лонг: предупреждение, балл −1 (после отсечки будет гэп вниз)
        - Шорт: предупреждение, балл −1 (лонгисты могут держать до отсечки)
      • После отсечки (direction=after, 1–5 дней):
        - Лонг: потенциальная точка входа, балл +1 (цена упала на дивиденд)
        - Шорт: балл −1 (гэп уже случился, откат вероятен)

    Возвращает {"score_adj": int, "warning": str | None}
    [DG]
    """
    if not div_info or not div_info.get("near"):
        return {"score_adj": 0, "warning": None}

    direction = div_info.get("direction")
    days      = div_info.get("days_to_ex", 0)
    value     = div_info.get("value", 0)
    ex_date   = div_info.get("ex_date", "?")

    if direction == "before":
        adj = -1
        warn = (f"⚠️ Отсечка через {days} дн. ({ex_date}, {value:.2f}₽) — "
                f"{'ожидай гэп вниз после' if signal=='ЛОНГ' else 'лонгисты держат до отсечки'}")
    elif direction == "after":
        if signal == "ЛОНГ":
            adj  = +1
            warn = (f"💡 Гэп уже был {abs(days)} дн. назад ({ex_date}, {value:.2f}₽) — "
                    f"возможна точка входа после дивидендного снижения")
        else:
            adj  = -1
            warn = (f"⚠️ Гэп был {abs(days)} дн. назад ({ex_date}, {value:.2f}₽) — "
                    f"вероятен отскок, шорт рискован")
    else:
        return {"score_adj": 0, "warning": None}

    return {"score_adj": adj, "warning": warn}


# ── Вспомогательные ───────────────────────────────────────────
def calculate_bollinger_bands(series, period=20, num_std=2.0):
    m = series.rolling(window=period).mean()
    s = series.rolling(window=period).std()
    u, l = m + s*num_std, m - s*num_std
    return {"upper": u, "middle": m, "lower": l,
            "percent_b": (series-l)/(u-l), "bandwidth": (u-l)/m}


def calculate_atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def analyze_volume(df, period=20):
    if "volume" not in df.columns:
        return {"signal": "нет данных", "ratio": 1.0}
    avg = df["volume"].rolling(period).mean().iloc[-1]
    if not avg or avg == 0 or pd.isna(avg):
        return {"signal": "нет данных", "ratio": 1.0}
    ratio = df["volume"].iloc[-1] / avg
    chg   = df["close"].iloc[-1] - df["close"].iloc[-2] if len(df) > 1 else 0
    sig   = ("бычий объём"    if ratio > 1.5 and chg > 0 else
             "медвежий объём" if ratio > 1.5 and chg < 0 else
             "низкий объём"   if ratio < 0.5 else "норма")
    return {"signal": sig, "ratio": round(ratio, 2)}


def calculate_stochastic(df, k_period=14, d_period=3):
    lo = df["low"].rolling(k_period).min()
    hi = df["high"].rolling(k_period).max()
    k  = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return {"k": k, "d": k.rolling(d_period).mean()}


def calculate_moving_averages(df):
    return {f"ma{p}": df["close"].rolling(p).mean() for p in [5, 10, 20]}


def find_support_resistance(df, window=5):
    if len(df) < window*2:
        return {"support": None, "resistance": None}
    cp   = df["close"].iloc[-1]
    mins = [df["low"].iloc[i] for i in range(window, len(df)-window)
            if df["low"].iloc[i] == df["low"].iloc[i-window:i+window+1].min()]
    maxs = [df["high"].iloc[i] for i in range(window, len(df)-window)
            if df["high"].iloc[i] == df["high"].iloc[i-window:i+window+1].max()]
    sb = [s for s in mins if s < cp]
    ra = [r for r in maxs if r > cp]
    return {"support":    round(max(sb), 2) if sb else None,
            "resistance": round(min(ra), 2) if ra else None}


# ── Главная функция анализа ───────────────────────────────────
def analyze_ticker(df: pd.DataFrame, current_price: float,
                   news_sentiment_score: int = 0,
                   df_weekly: pd.DataFrame = None,
                   div_info: dict = None) -> dict:
    """
    df          — дневные свечи
    df_weekly   — недельные свечи (опционально) [TF]
    div_info    — результат is_near_dividend_gap() (опционально) [DG]
    """
    if df.empty or len(df) < 26:
        return {"signal": "нет данных", "score": 0, "score_tech": 0,
                "score_news": 0, "reasons": []}
    close = df["close"]
    rsi   = calculate_rsi(close)
    macd  = calculate_macd(close)
    ma    = calculate_moving_averages(df)
    bb    = calculate_bollinger_bands(close)
    atr_s = calculate_atr(df)
    vol   = analyze_volume(df)
    stoch = calculate_stochastic(df)
    lvl   = find_support_resistance(df)
    ema_r = analyze_ema_trend(df)
    obv_r = analyze_obv(df)
    div_r = analyze_rsi_divergence(df, rsi)
    cnd_r = analyze_candle_patterns(df)
    wtf_r = analyze_weekly_trend(df_weekly)   # [TF]

    cr   = rsi.iloc[-1]
    cm   = macd["macd"].iloc[-1];  cs  = macd["signal"].iloc[-1]
    ch   = macd["histogram"].iloc[-1]; ph = macd["histogram"].iloc[-2] if len(df)>1 else 0
    cm5  = ma["ma5"].iloc[-1];  cm10 = ma["ma10"].iloc[-1]; cm20 = ma["ma20"].iloc[-1]
    cbp  = bb["percent_b"].iloc[-1]; cbu = bb["upper"].iloc[-1]; cbl = bb["lower"].iloc[-1]
    catr = atr_s.iloc[-1]
    ck   = stoch["k"].iloc[-1]; cd_ = stoch["d"].iloc[-1]
    pk   = stoch["k"].iloc[-2] if len(df)>1 else 50
    pd__ = stoch["d"].iloc[-2] if len(df)>1 else 50
    last = close.iloc[-1]
    atr_pct = (catr / last * 100) if last > 0 else 0

    ts = 0; reasons = []

    # RSI
    if not pd.isna(cr):
        if   cr < 30: ts += 2; reasons.append(f"RSI {cr:.0f} — перепроданность")
        elif cr < 40: ts += 1; reasons.append(f"RSI {cr:.0f} — зона покупки")
        elif cr > 70: ts -= 2; reasons.append(f"RSI {cr:.0f} — перекупленность")
        elif cr > 60: ts -= 1; reasons.append(f"RSI {cr:.0f} — зона продажи")

    # MACD
    if   cm > cs and ph < 0 and ch > 0: ts += 2; reasons.append("MACD: бычье пересечение")
    elif cm > cs:                        ts += 1; reasons.append("MACD выше сигнальной")
    elif cm < cs and ph > 0 and ch < 0: ts -= 2; reasons.append("MACD: медвежье пересечение")
    elif cm < cs:                        ts -= 1; reasons.append("MACD ниже сигнальной")

    if ema_r["score"] != 0: ts += ema_r["score"]; reasons.append(ema_r["label"])
    if obv_r["score"] != 0: ts += obv_r["score"]; reasons.append(obv_r["label"])
    if div_r["score"] != 0: ts += div_r["score"]; reasons.append(div_r["label"])
    if cnd_r["score"] != 0: ts += cnd_r["score"]
    reasons.extend(cnd_r["patterns"])

    # Боллинджер
    if not pd.isna(cbp):
        if   cbp < 0:   ts += 2; reasons.append(f"Боллинджер: ниже нижней полосы")
        elif cbp < 0.2: ts += 1; reasons.append(f"Боллинджер: у нижней полосы ({cbl:.1f})")
        elif cbp > 1:   ts -= 2; reasons.append(f"Боллинджер: выше верхней полосы")
        elif cbp > 0.8: ts -= 1; reasons.append(f"Боллинджер: у верхней полосы ({cbu:.1f})")

    # Объём
    if   vol["signal"] == "бычий объём":   ts += 1; reasons.append(f"Аномальный объём при росте (x{vol['ratio']})")
    elif vol["signal"] == "медвежий объём": ts -= 1; reasons.append(f"Аномальный объём при падении (x{vol['ratio']})")

    # MA выравнивание
    if   last > cm5 > cm10 > cm20: ts += 1; reasons.append("Цена выше всех MA — тренд вверх")
    elif last < cm5 < cm10 < cm20: ts -= 1; reasons.append("Цена ниже всех MA — тренд вниз")

    # Стохастик
    if not pd.isna(ck) and not pd.isna(cd_):
        k_up   = pk < pd__ and ck > cd_
        k_down = pk > pd__ and ck < cd_
        if   ck < 20 and k_up:   ts += 2; reasons.append(f"Стохастик: бычье пересечение ({ck:.0f})")
        elif ck < 20:             ts += 1; reasons.append(f"Стохастик {ck:.0f} — перепроданность")
        elif ck > 80 and k_down: ts -= 2; reasons.append(f"Стохастик: медвежье пересечение ({ck:.0f})")
        elif ck > 80:             ts -= 1; reasons.append(f"Стохастик {ck:.0f} — перекупленность")

    # [TF] Недельное подтверждение — применяем ПОСЛЕ подсчёта дневного балла
    tf_adj = 0
    if wtf_r["score"] != 0:
        daily_bull = ts > 0
        daily_bear = ts < 0
        # Совпадает с дневным — ничего не меняем (нейтрально)
        # Противоречит дневному — штраф
        if daily_bull and wtf_r["score"] < 0:
            tf_adj = -2
            reasons.append(f"⚠️ Таймфрейм: {wtf_r['label']} — против дневного сигнала")
        elif daily_bear and wtf_r["score"] > 0:
            tf_adj = -2
            reasons.append(f"⚠️ Таймфрейм: {wtf_r['label']} — против дневного сигнала")
        elif (daily_bull and wtf_r["score"] > 0) or (daily_bear and wtf_r["score"] < 0):
            tf_adj = +1
            reasons.append(f"✅ Таймфрейм: {wtf_r['label']} — подтверждает сигнал")

    ts += tf_adj

    # Новости
    raw_ns = int(round(news_sentiment_score * NEWS_SCORE_WEIGHT))
    ns     = max(-NEWS_SCORE_CAP, min(NEWS_SCORE_CAP, raw_ns))
    if ns > 0: reasons.append(f"Новости: позитивный фон (+{ns})")
    elif ns < 0: reasons.append(f"Новости: негативный фон ({ns})")

    total = ts + ns
    sig   = ("ЛОНГ"  if total >= SCORE_LONG_THRESHOLD else
             "ШОРТ"  if total <= SCORE_SHORT_THRESHOLD else "НЕЙТРАЛЬНО")

    # [DG] Дивидендный гэп — применяем поверх сигнала
    dg_result = analyze_dividend_gap(div_info, sig)
    dg_warn   = dg_result["warning"]
    total    += dg_result["score_adj"]
    if dg_result["score_adj"] != 0:
        sig = ("ЛОНГ"  if total >= SCORE_LONG_THRESHOLD else
               "ШОРТ"  if total <= SCORE_SHORT_THRESHOLD else "НЕЙТРАЛЬНО")
    if dg_warn:
        reasons.append(dg_warn)

    return {
        "signal": sig, "score": total, "score_tech": ts, "score_news": ns,
        "rsi":    round(cr,1) if not pd.isna(cr) else None,
        "macd": round(cm,4), "macd_signal": round(cs,4),
        "ema50": ema_r["ema50"], "ema200": ema_r["ema200"],
        "weekly_trend": wtf_r["label"] or "нет данных",
        "div_warning": dg_warn,
        "obv_trend": obv_r["obv_trend"], "candle_pats": cnd_r["patterns"],
        "ma5": round(cm5,2), "ma10": round(cm10,2), "ma20": round(cm20,2),
        "bb_upper": round(cbu,2) if not pd.isna(cbu) else None,
        "bb_lower": round(cbl,2) if not pd.isna(cbl) else None,
        "bb_percent": round(cbp*100,1) if not pd.isna(cbp) else None,
        "atr": round(catr,2) if not pd.isna(catr) else None,
        "atr_pct": round(atr_pct,2),
        "volume_signal": vol["signal"], "volume_ratio": vol["ratio"],
        "stoch_k": round(ck,1) if not pd.isna(ck) else None,
        "stoch_d": round(cd_,1) if not pd.isna(cd_) else None,
        "support": lvl["support"], "resistance": lvl["resistance"],
        "reasons": reasons,
    }
