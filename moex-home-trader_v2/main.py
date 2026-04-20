# main.py — точка входа (v5)
# [B] logging с ротацией файлов: logs/moex_YYYY-MM.log

import logging
import logging.handlers
import schedule
import time
from datetime import datetime
from pathlib import Path
from config import (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                    SCORE_LONG_THRESHOLD, SCORE_SHORT_THRESHOLD)

# ── [B] Логирование ───────────────────────────────────────────
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

def _setup_logging():
    fmt  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    dfmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=dfmt)

    console_h = logging.StreamHandler()
    console_h.setLevel(logging.INFO)
    console_h.setFormatter(formatter)

    log_file = LOGS_DIR / f"moex_{datetime.now().strftime('%Y-%m')}.log"
    file_h   = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5*1024*1024, backupCount=6, encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        root.addHandler(console_h)
        root.addHandler(file_h)
    return logging.getLogger("main")


logger = _setup_logging()
logging.getLogger("t_tech.invest.logging").setLevel(logging.WARNING)

try:
    from notifier import send_ideas_one_by_one
except ImportError as e:
    logger.error(f"notifier: {e}")
    def send_ideas_one_by_one(d): pass

try:
    from ideas_generator import generate_all_ideas
except ImportError as e:
    logger.critical(f"ideas_generator: {e}"); raise


def run_analysis_cycle():
    start = datetime.now()
    logger.info("=" * 60)
    logger.info(f"Запуск: {start.strftime('%d.%m.%Y %H:%M:%S')}")
    logger.info(f"Порог ЛОНГ >= {SCORE_LONG_THRESHOLD}, ШОРТ <= {SCORE_SHORT_THRESHOLD}")
    logger.info("=" * 60)
    try:
        data = generate_all_ideas()
    except Exception as e:
        logger.error(f"generate_all_ideas: {e}", exc_info=True); return

    longs, shorts = data.get("long_ideas", []), data.get("short_ideas", [])
    total = len(longs) + len(shorts)
    logger.info(f"Результат: {len(longs)} лонгов, {len(shorts)} шортов")

    if total == 0:
        logger.info("Нет перспективных идей.")
    else:
        try:
            send_ideas_one_by_one(data)
            logger.info("Telegram: отправлено")
        except Exception as e:
            logger.error(f"Telegram: {e}", exc_info=True)

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Цикл завершён за {elapsed:.0f}с")


def _is_trading_day(): return datetime.now().weekday() < 5

def _scheduled_run():
    if _is_trading_day(): run_analysis_cycle()
    else:
        day = ["пн","вт","ср","чт","пт","сб","вс"][datetime.now().weekday()]
        logger.info(f"Сегодня {day} — биржа закрыта.")


def start_scheduler(interval_minutes=60):
    logger.info(f"Планировщик: каждые {interval_minutes} мин. в торговые дни.")
    run_analysis_cycle()
    schedule.every(interval_minutes).minutes.do(_scheduled_run)
    while True:
        try:
            schedule.run_pending(); time.sleep(30)
        except KeyboardInterrupt:
            logger.info("Остановлен."); break
        except Exception as e:
            logger.error(f"Планировщик: {e}", exc_info=True); time.sleep(60)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--once",     action="store_true")
    p.add_argument("--interval", type=int, default=60)
    args = p.parse_args()
    if "СЮДА_ВСТАВЬ" in str(TELEGRAM_BOT_TOKEN):
        logger.critical("Заполни TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID в config.py!")
    elif args.once:
        run_analysis_cycle()
    else:
        start_scheduler(interval_minutes=args.interval)

# [FIX] duration log — оборачиваем любую функцию job()/run_once() логом длительности.
import logging as _lg_fix
import time as _t_fix
_main_log = _lg_fix.getLogger(__name__)

def _with_duration(fn):
    def _wrap(*a, **kw):
        started = _t_fix.time()
        try:
            return fn(*a, **kw)
        except Exception as e:
            _main_log.exception("job failed: %s", e)
        finally:
            _main_log.info("job finished in %.1fs", _t_fix.time() - started)
    _wrap.__name__ = getattr(fn, "__name__", "job")
    return _wrap

try:
    job = _with_duration(job)  # type: ignore
except NameError:
    pass
try:
    run_once = _with_duration(run_once)  # type: ignore
except NameError:
    pass
