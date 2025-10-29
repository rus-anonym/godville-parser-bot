import asyncio
import os
import re
import json
import random
import logging
from pathlib import Path
from typing import Optional, Tuple
from contextlib import suppress
from datetime import datetime, timedelta

from dotenv import load_dotenv
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# try zoneinfo (Python 3.9+)
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# ===================== Конфигурация =====================
load_dotenv()


def _env_flag(name: str, default: str = '0') -> bool:
    return os.getenv(name, default).strip().lower() in ('1', 'true', 'yes', 'y', 'on')


LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

GODVILLE_LOGIN = os.getenv('GODVILLE_LOGIN')
GODVILLE_PASSWORD = os.getenv('GODVILLE_PASSWORD')

# Экономия ресурсов по умолчанию
HEADLESS = _env_flag('HEADLESS', '1')
BLOCK_TRACKERS = _env_flag('BLOCK_TRACKERS', '1')
BLOCK_MEDIA = _env_flag('BLOCK_MEDIA', '1')  # режем image/font/media
SAVE_STATE = _env_flag('SAVE_STATE', '1')  # хранить state.json (куки и пр.)
AUTO_RESURRECT = _env_flag('AUTO_RESURRECT', '1')  # автоматически жать «Воскресить», если герой мёртв

STATE_PATH = Path(os.getenv('STATE_PATH', 'state.json'))

USER_AGENT = os.getenv('USER_AGENT',
                       'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36')
LOCALE = os.getenv('LOCALE', 'ru-RU')
VIEWPORT_W = int(os.getenv('VIEWPORT_W', '960'))
VIEWPORT_H = int(os.getenv('VIEWPORT_H', '600'))

LOGIN_URL = 'https://godville.net/login'
HERO_URL = 'https://godville.net/superhero'

# Режим действий: random | good | bad
ACTION_MODE_RAW = os.getenv('ACTION_MODE', 'random').strip().lower()
ALIASES = {
    'rand': 'random', 'rnd': 'random', 'случайно': 'random',
    'good-only': 'good', 'enc': 'good', 'encour': 'good', 'хорошо': 'good',
    'bad-only': 'bad', 'pun': 'bad', 'punish': 'bad', 'плохо': 'bad'
}
ACTION_MODE = ALIASES.get(ACTION_MODE_RAW, ACTION_MODE_RAW)
if ACTION_MODE not in ('random', 'good', 'bad'):
    ACTION_MODE = 'random'
ACTION_FALLBACK = _env_flag('ACTION_FALLBACK', '0')

# Интервалы между попытками действий
MIN_ACTION_INTERVAL_SEC = int(os.getenv('MIN_ACTION_INTERVAL_SEC', '5'))
MAX_ACTION_INTERVAL_SEC = int(os.getenv('MAX_ACTION_INTERVAL_SEC', '20'))

# Когда кнопок нет N раз подряд — «спим»
NO_BUTTONS_GRACE_CHECKS = int(os.getenv('NO_BUTTONS_GRACE_CHECKS', '3'))
SHORT_RETRY_DELAY_SEC = float(os.getenv('SHORT_RETRY_DELAY_SEC', '1.5'))

SLEEP_MIN_SEC = int(os.getenv('SLEEP_MIN_SEC', '3600'))
SLEEP_MAX_SEC = int(os.getenv('SLEEP_MAX_SEC', '7200'))

# Если не видим кнопку несколько раз — делаем мягкое обновление/переход
RELOAD_ON_MISS = int(os.getenv('RELOAD_ON_MISS', '2'))  # после скольких промахов делать page.reload
NAVIGATE_ON_MISS = int(os.getenv('NAVIGATE_ON_MISS', '4'))  # после скольких промахов делать goto(HERO_URL)

# Тайминги и клики
CLICK_TIMEOUT_MS = int(os.getenv('CLICK_TIMEOUT_MS', '2500'))
DETECT_TIMEOUT_MS = int(os.getenv('DETECT_TIMEOUT_MS', '7000'))
CLICK_RETRIES = int(os.getenv('CLICK_RETRIES', '3'))
POST_CLICK_WAIT_MS = int(os.getenv('POST_CLICK_WAIT_MS', '800'))

# Хосты-трекеры (отрежем для экономии трафика и шума)
TRACKER_HOST_SUBSTR = (
    'googletagmanager.com', 'google-analytics.com', 'doubleclick.net',
    'g.doubleclick.net', 'www.google.com/ccm'
)

# Селекторы кнопок
GOOD_SELECTORS = [
    '#cntrl1 a.enc_link', '#cntrl a.enc_link', 'a.enc_link',
    'a:has-text("Сделать хорошо")', 'button:has-text("Сделать хорошо")',
    '[onclick*="encour"]', 'a[href*="encour"]',
]
BAD_SELECTORS = [
    '#cntrl1 a.pun_link', '#cntrl a.pun_link', 'a.pun_link',
    'a:has-text("Сделать плохо")', 'button:has-text("Сделать плохо")',
    '[onclick*="punish"]', 'a[href*="punish"]',
]
RESURRECT_SELECTORS = [
    'a:has-text("Воскресить")', 'button:has-text("Воскресить")',
    'text=/Воскр/i',
    '[onclick*="resur"]', '[href*="resur"]',
    '[onclick*="reviv"]', '[href*="reviv"]',
    'a:has-text("Resurrect")', 'button:has-text("Resurrect")',
    'a:has-text("Revive")', 'button:has-text("Revive")',
]

# ===================== Газета / ежедневки =====================
ENABLE_GAZETTE = _env_flag('ENABLE_GAZETTE', '1')
COUPON_ENABLED = _env_flag('COUPON_ENABLED', '1')
WORKER_HEARTBEAT_SEC = int(os.getenv('WORKER_HEARTBEAT_SEC', '120'))
GPC_ENABLED = _env_flag('GPC_ENABLED', '1')
GPC_TARGET_PCT = int(os.getenv('GPC_TARGET_PCT', '25'))
GPC_LAST_CHANCE_MIN = int(os.getenv('GPC_LAST_CHANCE_MIN', '30'))
GPC_LAST_CHANCE_MIN_PCT = int(os.getenv('GPC_LAST_CHANCE_MIN_PCT', '5'))
GPC_POLL_MIN_MS = int(os.getenv('GPC_POLL_MIN_MS', '400'))
GPC_POLL_MAX_MS = int(os.getenv('GPC_POLL_MAX_MS', '1200'))
GPC_DISABLED_RECHECK_SEC = int(os.getenv('GPC_DISABLED_RECHECK_SEC', '10'))
GPC_DISABLED_RETRY_COUNT = int(os.getenv('GPC_DISABLED_RETRY_COUNT', '3'))
GPC_AUTO_REFRESH = _env_flag('GPC_AUTO_REFRESH', '0')
GPC_FAST_WINDOW_BELOW = int(os.getenv('GPC_FAST_WINDOW_BELOW', '3'))
GPC_LOG_DELTA = int(os.getenv('GPC_LOG_DELTA', '5'))

GAZETTE_URL = os.getenv('GAZETTE_URL', 'https://godville.net/news')
GODVILLE_TZ = os.getenv('GODVILLE_TZ', 'Europe/Moscow')
DAILY_RESET_TIME = os.getenv('DAILY_RESET_TIME', '00:00')  # HH:MM
BOT_STATE_PATH = Path(os.getenv('BOT_STATE_PATH', 'bot_state.json'))


# ===================== Маршрутизация запросов =====================
async def setup_routing(context):
    if not (BLOCK_TRACKERS or BLOCK_MEDIA):
        return

    async def route_all(route):
        try:
            req = route.request
            url = req.url
            rtype = req.resource_type

            if BLOCK_MEDIA and rtype in ('image', 'media', 'font'):
                return await route.abort()

            if BLOCK_TRACKERS and any(h in url for h in TRACKER_HOST_SUBSTR):
                return await route.abort()

            return await route.continue_()
        except Exception:
            with suppress(Exception):
                await route.continue_()

    await context.route("**/*", route_all)


# ===================== Утилиты =====================
async def dismiss_cookie_banners(page):
    candidates = (
        'button:has-text("Принять")', 'button:has-text("Соглас")',
        'button:has-text("OK")', 'button:has-text("ОК")',
        'button:has-text("Accept")', 'button:has-text("I agree")',
        'text=Принять', 'text=Соглас', 'text=Accept', 'text=I agree',
    )
    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if await loc.count() and await loc.is_visible():
                await loc.click()
                await asyncio.sleep(0.2)
        except Exception:
            pass


async def save_debug(page, prefix="debug"):
    try:
        await page.screenshot(path=f"{prefix}.png", full_page=True)
        with open(f"{prefix}.html", "w", encoding="utf-8") as f:
            f.write(await page.content())
        logging.info(f"Сохранил {prefix}.png / {prefix}.html")
    except Exception as e:
        logging.debug(f"Не удалось сохранить отладку: {e}")


async def _first_visible(page, selectors) -> Tuple[Optional[object], Optional[str]]:
    """Первый видимый локатор из набора селекторов."""
    for sel in selectors:
        loc = page.locator(sel).first
        try:
            if await loc.count():
                with suppress(Exception):
                    await loc.scroll_into_view_if_needed(timeout=300)
                if await loc.is_visible():
                    return loc, sel
        except Exception:
            continue
    return None, None


async def wait_prana_controls(page, which='any', timeout_ms=DETECT_TIMEOUT_MS) -> bool:
    """Ждём появления кнопок 'Сделать хорошо/плохо'. which: any|good|bad"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + (timeout_ms / 1000.0)
    while loop.time() < deadline:
        good_loc, _ = await _first_visible(page, GOOD_SELECTORS)
        bad_loc, _ = await _first_visible(page, BAD_SELECTORS)
        if which == 'good' and good_loc:
            return True
        if which == 'bad' and bad_loc:
            return True
        if which == 'any' and (good_loc or bad_loc):
            return True
        await asyncio.sleep(0.25)
    return False


async def find_action_buttons(page):
    good_loc, good_sel = await _first_visible(page, GOOD_SELECTORS)
    bad_loc, bad_sel = await _first_visible(page, BAD_SELECTORS)
    return good_loc, bad_loc, {"good": good_sel, "bad": bad_sel}


# ===================== Усиленные клики/верификации =====================
async def _element_enabled(loc) -> bool:
    try:
        if not await loc.count():
            return False
        with suppress(Exception):
            if await loc.is_disabled():
                return False
        h = await loc.element_handle()
        if not h:
            return False
        disabled_attr = await h.get_attribute("disabled")
        aria_dis = await h.get_attribute("aria-disabled")
        if disabled_attr is not None or (aria_dis and aria_dis.lower() in ("true", "1")):
            return False
        return True
    except Exception:
        return False


async def strong_click(page, loc, what="button", timeout=1500) -> bool:
    try:
        await loc.scroll_into_view_if_needed(timeout=300)
    except Exception:
        pass

    # Попытка 1: обычный click
    with suppress(Exception):
        await loc.click(timeout=timeout)
        return True

    # Попытка 2: force=True
    with suppress(Exception):
        await loc.click(timeout=timeout, force=True)
        return True

    # Попытка 3: клик мышью по bbox
    with suppress(Exception):
        box = await loc.bounding_box()
        if box and box.get("width", 0) > 0 and box.get("height", 0) > 0:
            x = box["x"] + box["width"] / 2
            y = box["y"] + box["height"] / 2
            await page.mouse.move(x, y)
            await page.mouse.click(x, y)
            return True

    # Попытка 4: dispatch_event
    with suppress(Exception):
        await loc.dispatch_event("click")
        return True

    # Попытка 5: JS el.click()
    with suppress(Exception):
        h = await loc.element_handle()
        if h:
            await page.evaluate("(el) => el.click()", h)
            return True

    logging.debug(f"strong_click: не смог кликнуть {what}")
    return False


async def wait_after_resurrect(page, timeout_ms=5000) -> bool:
    # Успех: исчезла кнопка «Воскресить» и/или появились кнопки prana
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_ms / 1000.0
    while loop.time() < deadline:
        loc, _ = await _first_visible(page, RESURRECT_SELECTORS)
        if not loc:
            if await wait_prana_controls(page, which='any', timeout_ms=800):
                return True
        await asyncio.sleep(0.2)
    return False


async def wait_after_prana_click(page, timeout_ms=3000) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_ms / 1000.0
    while loop.time() < deadline:
        good_loc, bad_loc, _ = await find_action_buttons(page)
        if not good_loc and not bad_loc:
            return True
        if good_loc and not await _element_enabled(good_loc):
            return True
        if bad_loc and not await _element_enabled(bad_loc):
            return True
        await asyncio.sleep(0.2)
    with suppress(Exception):
        await page.wait_for_load_state("networkidle", timeout=800)
    return True


# ===================== Логин/сессия =====================
async def perform_login(page, login: str, password: str) -> bool:
    logging.info("Открываю страницу логина...")
    await page.goto(LOGIN_URL, wait_until="domcontentloaded")

    await dismiss_cookie_banners(page)
    await page.wait_for_selector('form[action="/login"], input[name], button[type="submit"]', timeout=20000)

    user_sel = 'input[name="username"], input[name="login"], #username, form[action="/login"] input[type="text"]'
    pass_sel = 'input[name="password"], #password, form[action="/login"] input[type="password"]'
    submit_sel = 'button:has-text("Войти"), input[type="submit"], button[type="submit"]'

    await page.locator(user_sel).first.fill(login)
    await page.locator(pass_sel).first.fill(password)

    try:
        async with page.expect_navigation(wait_until="domcontentloaded", timeout=15000):
            await page.locator(submit_sel).first.click()
    except PlaywrightTimeoutError:
        logging.warning("Навигации после сабмита не было — проверяю вручную...")

    await page.goto(HERO_URL, wait_until="domcontentloaded")

    if "login" in page.url:
        logging.error("Логин не удался — всё ещё на /login.")
        await save_debug(page, "login_failed")
        return False

    try:
        await page.wait_for_selector('#cntrl1, #cntrl, #god_name', timeout=20000)
    except PlaywrightTimeoutError:
        logging.error("Не дождался признаков страницы героя.")
        await save_debug(page, "hero_wait_failed")
        return False

    logging.info("Авторизация прошла успешно.")
    return True


async def ensure_logged_in(context, page, login, password) -> bool:
    await page.goto(HERO_URL, wait_until="domcontentloaded")
    if "login" in page.url:
        logging.info("Сессии нет — логинюсь.")
        ok = await perform_login(page, login, password)
        if ok and SAVE_STATE:
            with suppress(Exception):
                await context.storage_state(path=str(STATE_PATH))
                logging.info(f"Session saved to {STATE_PATH}")
        return ok
    return True


# ===================== Время/день/состояние для ежедневок =====================
def _now_tz():
    try:
        if ZoneInfo:
            return datetime.now(ZoneInfo(GODVILLE_TZ))
        return datetime.now()
    except Exception:
        return datetime.now()


def _next_daily_reset_dt():
    now = _now_tz()
    try:
        hh, mm = map(int, DAILY_RESET_TIME.split(':'))
    except Exception:
        hh, mm = 0, 0
    reset = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if now >= reset:
        reset += timedelta(days=1)
    return reset


def _today_key(prefix: str) -> str:
    return f"{prefix}:{_now_tz().strftime('%Y-%m-%d')}"


def _load_bot_state() -> dict:
    try:
        if BOT_STATE_PATH.exists():
            with open(BOT_STATE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_bot_state(data: dict):
    with suppress(Exception):
        with open(BOT_STATE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _mark_done_today(prefix: str):
    st = _load_bot_state()
    st[_today_key(prefix)] = True
    _save_bot_state(st)


def _is_done_today(prefix: str) -> bool:
    st = _load_bot_state()
    return bool(st.get(_today_key(prefix)))


# ===================== Газета =====================
class GazettePage:
    def __init__(self, page):
        self.page = page

    async def open(self):
        await self.page.goto(GAZETTE_URL, wait_until="domcontentloaded")
        await dismiss_cookie_banners(self.page)
        logging.debug(f"Газета: открыта страница {self.page.url}")

    # ---------- Купон ----------
    async def claim_coupon(self) -> bool:
        try:
            btn = self.page.locator('#coupon_b')
            if not (await btn.count()):
                logging.info("Газета: купон — кнопка не найдена (возможно, уже забран).")
                return False
            if await btn.is_disabled():
                logging.info("Газета: купон — кнопка задизейблена (уже забран).")
                return False
            logging.info("Газета: купон — пытаюсь забрать...")
            ok = await strong_click(self.page, btn, what="Купон", timeout=1500)
            await asyncio.sleep(0.4)
            if ok:
                # Успех: кнопка исчезла/задизейблилась или появилось сообщение
                with suppress(Exception):
                    if await btn.is_disabled() or not await btn.is_visible():
                        logging.info("Газета: купон — успешно забран.")
                        return True
                msg = self.page.locator('#cpn_msg')
                if await msg.count():
                    txt = (await msg.inner_text()).strip()
                    logging.info(f"Газета: купон — сообщение: {txt or '(пусто)'}")
                    return True
            logging.info("Газета: купон — клик не подтвердился.")
        except Exception as e:
            logging.warning(f"Газета: купон — ошибка клика: {e}")
        return False

    # ---------- Праноконденсатор ----------
    async def _gpc_percent(self) -> Optional[int]:
        try:
            loc = self.page.locator('#gpc_val')
            if not (await loc.count()):
                return None
            text = (await loc.inner_text()).strip()
            m = re.search(r'(\d+)', text)
            return int(m.group(1)) if m else None
        except Exception:
            return None

    async def _gpc_can_discharge(self) -> bool:
        btn = self.page.locator('#gp_cap_use')
        if not (await btn.count()):
            return False
        with suppress(Exception):
            if await btn.is_disabled():
                return False
        h = await btn.element_handle()
        if not h:
            return False
        disabled_attr = await h.get_attribute("disabled")
        if disabled_attr is not None:
            return False
        return True

    async def _gpc_click_refresh_if_visible(self):
        if not GPC_AUTO_REFRESH:
            return
        try:
            r = self.page.locator('#gp_cap_r')
            if await r.count() and await r.is_visible():
                logging.debug("Газета: GPC — жму «Обновить».")
                await r.click(timeout=800)
                await asyncio.sleep(0.2)
        except Exception as e:
            logging.debug(f"Газета: GPC — не удалось нажать «Обновить»: {e}")

    async def _gpc_wait_disabled_confirmation(self, retries: int, delay_sec: int) -> bool:
        """
        True — кнопка стабильно disabled во всех повторах.
        False — где-то по пути снова стала доступной.
        """
        for i in range(1, retries + 1):
            await asyncio.sleep(delay_sec)
            await self._gpc_click_refresh_if_visible()
            try:
                can = await self._gpc_can_discharge()
                logging.debug(f"Газета: GPC — перепроверка disabled #{i}: can={can}")
                if can:
                    return False
            except Exception:
                pass
        return True

    async def _gpc_discharge(self) -> Tuple[bool, str]:
        """Пытается разрядить. Возвращает (ok, reason)."""
        try:
            btn = self.page.locator('#gp_cap_use')
            if not (await btn.count()):
                return False, "no_button"
            if await btn.is_disabled():
                return False, "disabled"

            ok = await strong_click(self.page, btn, what="Разрядить", timeout=1200)
            await asyncio.sleep(0.25)
            if not ok:
                return False, "click_failed"

            err = self.page.locator('#gpc_err')
            if await err.count():
                txt_raw = (await err.inner_text()).strip()
                txt = txt_raw.lower()
                if ("уже" in txt and "сегодня" in txt) or ("already" in txt and "today" in txt):
                    return False, "already_used_today"
                if txt_raw:
                    return False, f"server_msg: {txt_raw}"

            # Верифицируем падение процента или дизейбл кнопки
            for _ in range(6):
                val = await self._gpc_percent()
                if val is not None and val <= 1:
                    return True, "ok_drop_to_zero"
                with suppress(Exception):
                    if await btn.is_disabled():
                        return True, "ok_button_disabled"
                await asyncio.sleep(0.2)

            await self._gpc_click_refresh_if_visible()
            return False, "no_drop"
        except Exception as e:
            return False, f"exception: {e}"

    async def discharge_when_ready(
            self,
            target_pct: int = GPC_TARGET_PCT,
            last_chance_minutes: int = GPC_LAST_CHANCE_MIN,
            last_chance_min_pct: int = GPC_LAST_CHANCE_MIN_PCT,
            poll_min_ms: int = GPC_POLL_MIN_MS,
            poll_max_ms: int = GPC_POLL_MAX_MS,
            max_hours: float = 20.0,
    ) -> bool | None:
        """
        Ждём, пока проценты >= target_pct и нажимаем «Разрядить».
        Если до daily reset остаётся < last_chance_minutes — жмём при >= last_chance_min_pct.
        Возвращает True, если получилось разрядить (или сервер сказал «уже сегодня»).
        """
        start = _now_tz()
        early_deadline = start + timedelta(hours=max_hours)
        reset_dt = _next_daily_reset_dt()
        last_chance_dt = reset_dt - timedelta(minutes=last_chance_minutes)

        last_logged_pct = None
        logging.info(
            f"Газета: GPC — мониторинг запущен. Цель={target_pct}%, последний шанс за {last_chance_minutes} мин (>= {last_chance_min_pct}%).")

        while _now_tz() < early_deadline:
            pct = await self._gpc_percent()
            if pct is None:
                logging.debug("Газета: GPC — элемент #gpc_val не найден, обновляю страницу.")
                with suppress(Exception):
                    await self.open()
                await asyncio.sleep(poll_max_ms / 1000.0)
                continue

            # Логируем каждые GPC_LOG_DELTA% или при первом чтении
            if last_logged_pct is None or abs(pct - last_logged_pct) >= GPC_LOG_DELTA:
                logging.info(f"Газета: GPC — {pct}%.")
                last_logged_pct = pct

            now = _now_tz()
            near_goal = pct >= max(0, target_pct - GPC_FAST_WINDOW_BELOW)
            want_try = (pct >= target_pct) or (now >= last_chance_dt and pct >= last_chance_min_pct)

            if want_try:
                for attempt in range(1, CLICK_RETRIES + 1):
                    ok, why = await self._gpc_discharge()
                    logging.info(f"Газета: GPC — попытка #{attempt} при {pct}% -> {ok} ({why})")

                    if ok:
                        logging.info("Газета: GPC — разрядка успешна.")
                        return True

                    if why == "already_used_today":
                        logging.info("Газета: GPC — уже разряжали сегодня (по ответу сервера).")
                        return True

                    if why in ("disabled", "no_button", "no_drop", "click_failed"):
                        stable = await self._gpc_wait_disabled_confirmation(
                            retries=GPC_DISABLED_RETRY_COUNT,
                            delay_sec=GPC_DISABLED_RECHECK_SEC
                        )
                        if not stable:
                            logging.debug("Газета: GPC — кнопка снова доступна, повторяю попытку.")
                            continue
                        logging.debug("Газета: GPC — кнопка стабильно недоступна, ждём набора процентов.")
                        break

                    await asyncio.sleep(0.3 + 0.2 * attempt)

            # Финальная попытка прямо перед ресетом — «что есть»
            if (_next_daily_reset_dt() - now) <= timedelta(seconds=max(poll_max_ms / 1000.0, 5)):
                for attempt in range(1, CLICK_RETRIES + 1):
                    ok, why = await self._gpc_discharge()
                    logging.info(f"Газета: GPC — финал перед ресетом, попытка #{attempt} -> {ok} ({why})")
                    if ok or why == "already_used_today":
                        return True
                    stable = await self._gpc_wait_disabled_confirmation(
                        retries=GPC_DISABLED_RETRY_COUNT,
                        delay_sec=GPC_DISABLED_RECHECK_SEC
                    )
                    if not stable:
                        continue
                    break
                return False

            # Динамическая частота опроса: ближе к цели — быстрее
            if near_goal:
                dt = random.uniform(0.12, 0.25)  # ускоряемся в окне цели
            else:
                dt = random.uniform(poll_min_ms / 1000.0, poll_max_ms / 1000.0)
            await asyncio.sleep(dt)


# ===================== Воскресить и клик по действию =====================
async def click_resurrect_if_needed(page) -> bool:
    """Если видна кнопка 'Воскресить' — нажимаем усиленным кликом и ждём восстановления контролов."""
    loc, sel = await _first_visible(page, RESURRECT_SELECTORS)
    if not loc:
        return False

    logging.info("Герой мёртв — жму «Воскресить».")
    for attempt in range(1, CLICK_RETRIES + 1):
        try:
            await dismiss_cookie_banners(page)
            loc, sel = await _first_visible(page, RESURRECT_SELECTORS)
            if not loc:
                ok = await wait_after_resurrect(page, timeout_ms=2500)
                return ok

            ok = await strong_click(page, loc, what="Воскресить", timeout=CLICK_TIMEOUT_MS)
            await asyncio.sleep(POST_CLICK_WAIT_MS / 1000.0)
            if ok and await wait_after_resurrect(page, timeout_ms=3500):
                logging.info("Воскрешение выполнено.")
                return True
        except Exception as e:
            logging.debug(f"Воскресить: попытка #{attempt} ошибка: {e}")

        if attempt == 2:
            with suppress(Exception):
                await page.reload(wait_until="domcontentloaded")
        await asyncio.sleep(0.4 + 0.2 * attempt)

    logging.warning("Не удалось нажать «Воскресить» после нескольких попыток.")
    return False


async def click_prana_action(page) -> bool:
    """Кликает 'Воскресить' при необходимости, затем good/bad по режиму (с усиленными кликами и верификацией)."""
    if "superhero" not in page.url:
        await page.goto(HERO_URL, wait_until="domcontentloaded")

    await dismiss_cookie_banners(page)

    if AUTO_RESURRECT:
        resurrected = await click_resurrect_if_needed(page)
        if resurrected:
            return True

    which = 'any' if ACTION_MODE == 'random' else ACTION_MODE
    if not await wait_prana_controls(page, which=which, timeout_ms=DETECT_TIMEOUT_MS):
        return False

    good_loc, bad_loc, _ = await find_action_buttons(page)

    candidates = []
    if ACTION_MODE == 'random':
        if random.choice([True, False]):
            if good_loc: candidates.append(("Сделать хорошо", "good"))
            if bad_loc:  candidates.append(("Сделать плохо", "bad"))
        else:
            if bad_loc:  candidates.append(("Сделать плохо", "bad"))
            if good_loc: candidates.append(("Сделать хорошо", "good"))
    elif ACTION_MODE == 'good':
        if good_loc:
            candidates.append(("Сделать хорошо", "good"))
        elif ACTION_FALLBACK and bad_loc:
            candidates.append(("Сделать плохо", "bad"))
    else:  # ACTION_MODE == 'bad'
        if bad_loc:
            candidates.append(("Сделать плохо", "bad"))
        elif ACTION_FALLBACK and good_loc:
            candidates.append(("Сделать хорошо", "good"))

    for title, kind in candidates:
        for attempt in range(1, CLICK_RETRIES + 1):
            try:
                await dismiss_cookie_banners(page)
                gl, bl, _ = await find_action_buttons(page)
                loc_cur = gl if kind == "good" else bl
                if not loc_cur:
                    break

                ok = await strong_click(page, loc_cur, what=title, timeout=CLICK_TIMEOUT_MS)
                await asyncio.sleep(POST_CLICK_WAIT_MS / 1000.0)
                if ok:
                    if await wait_after_prana_click(page, timeout_ms=2500):
                        logging.info(f"Нажал: {title}")
                        return True
            except Exception as e:
                logging.debug(f"{title}: попытка #{attempt} ошибка: {e}")

            if attempt == 2:
                with suppress(Exception):
                    await page.reload(wait_until="domcontentloaded")
            await asyncio.sleep(0.3 + 0.2 * attempt)

    return False


# ===================== Фоновый воркер газеты =====================
async def gazette_worker(context, login, password):
    if not ENABLE_GAZETTE:
        return
    page = await context.new_page()
    page.set_default_timeout(15000)
    page.on("dialog", lambda d: asyncio.create_task(d.accept()))
    gaz = GazettePage(page)

    logging.info("Газета: воркер запущен (ENABLE_GAZETTE=1).")
    last_heartbeat = 0.0

    while True:
        try:
            # Логинимся (на своей вкладке) и открываем газету
            await ensure_logged_in(context, page, login, password)
            await gaz.open()

            # Heartbeat
            now_ts = asyncio.get_running_loop().time()
            if now_ts - last_heartbeat >= WORKER_HEARTBEAT_SEC:
                logging.info("Газета: heartbeat — воркер активен.")
                last_heartbeat = now_ts

            # Купон — один раз в день
            if COUPON_ENABLED and not _is_done_today("coupon"):
                logging.info("Газета: проверяю купон...")
                claimed = await gaz.claim_coupon()
                logging.info(f"Газета: купон — claimed={claimed}")
                if claimed:
                    _mark_done_today("coupon")

            # Праноконденсатор — следим до успешного разряда или до «последнего шанса»
            if GPC_ENABLED and not _is_done_today("gpc"):
                logging.info("Газета: GPC — начинаю мониторинг.")
                ok = await gaz.discharge_when_ready()
                logging.info(f"Газета: GPC — завершено со статусом ok={ok}.")
                if ok:
                    _mark_done_today("gpc")

        except Exception as e:
            logging.warning(f"Газета: ошибка в воркере: {e}")
            with suppress(Exception):
                await save_debug(page, "gazette_worker_debug")

        # Если всё сделано — ждём до ресета, иначе — короткий сон и продолжаем пасти GPC
        all_done = (_is_done_today("coupon") or not COUPON_ENABLED) and (_is_done_today("gpc") or not GPC_ENABLED)
        if all_done:
            to_reset = (_next_daily_reset_dt() - _now_tz()).total_seconds()
            nap = max(60.0, to_reset + random.uniform(15, 60))
            logging.info(f"Газета: всё сделано — сплю до ресета ~{nap / 60:.1f} мин.")
        else:
            nap = random.uniform(20, 45)  # почаще крутимся возле GPC
        await asyncio.sleep(nap)


# ===================== Основной цикл =====================
async def run_bot():
    if not GODVILLE_LOGIN or not GODVILLE_PASSWORD:
        logging.error("Не найдены GODVILLE_LOGIN / GODVILLE_PASSWORD в .env")
        return

    launch_args = [
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--disable-gpu",
        "--mute-audio",
        "--js-flags=--max-old-space-size=128",
    ]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS, args=launch_args)

        context_kwargs = dict(
            user_agent=USER_AGENT,
            locale=LOCALE,
            viewport={"width": VIEWPORT_W, "height": VIEWPORT_H},
            extra_http_headers={"Accept-Language": f"{LOCALE},ru;q=0.9,en;q=0.8"},
        )

        # Загружаем сохранённое состояние
        if SAVE_STATE and STATE_PATH.exists() and STATE_PATH.stat().st_size > 2:
            try:
                with open(STATE_PATH, "r", encoding="utf-8") as f:
                    state_obj = json.load(f)
                context_kwargs["storage_state"] = state_obj
                logging.info(f"Loaded storage state from {STATE_PATH}")
            except Exception as e:
                logging.warning(f"Failed to load storage state from {STATE_PATH}: {e}. Will login fresh.")

        context = await browser.new_context(**context_kwargs)
        await setup_routing(context)

        # Главная вкладка героя
        page = await context.new_page()
        page.set_default_timeout(20000)
        page.on("dialog", lambda d: asyncio.create_task(d.accept()))

        gaz_task = None
        try:
            if not await ensure_logged_in(context, page, GODVILLE_LOGIN, GODVILLE_PASSWORD):
                logging.error("Не удалось авторизоваться. Останавливаюсь.")
                return

            # Сохраняем state после логина
            if SAVE_STATE:
                with suppress(Exception):
                    await context.storage_state(path=str(STATE_PATH))
                    logging.info(f"Session saved to {STATE_PATH}")

            # Запускаем фонового работника газеты
            if ENABLE_GAZETTE:
                gaz_task = asyncio.create_task(gazette_worker(context, GODVILLE_LOGIN, GODVILLE_PASSWORD))
                logging.info("Газетный воркер запущен.")

            logging.info(
                f"Режим действий: {ACTION_MODE}{' + fallback' if ACTION_FALLBACK else ''}. Headless={HEADLESS}. AutoResurrect={AUTO_RESURRECT}."
            )
            miss_streak = 0
            ticks = 0

            while True:
                ticks += 1
                await asyncio.sleep(random.uniform(MIN_ACTION_INTERVAL_SEC, MAX_ACTION_INTERVAL_SEC))

                # Иногда прогоняем скрытые баннеры
                if ticks % 10 == 0:
                    await dismiss_cookie_banners(page)

                # Если разлогинило — перелогин
                if "login" in page.url:
                    if not await ensure_logged_in(context, page, GODVILLE_LOGIN, GODVILLE_PASSWORD):
                        logging.error("Перелогин не удался. Завершаю.")
                        return

                # Периодические подстраховки
                if miss_streak == RELOAD_ON_MISS:
                    with suppress(Exception):
                        await page.reload(wait_until="domcontentloaded")
                elif miss_streak >= NAVIGATE_ON_MISS:
                    with suppress(Exception):
                        await page.goto(HERO_URL, wait_until="domcontentloaded")

                clicked = await click_prana_action(page)
                if clicked:
                    miss_streak = 0
                    continue

                # Кнопок нет — короткий ретрай
                miss_streak += 1
                logging.info(f"Кнопок нет (#{miss_streak}). Повторная проверка через {SHORT_RETRY_DELAY_SEC:.1f} сек.")
                await asyncio.sleep(SHORT_RETRY_DELAY_SEC)

                clicked_retry = await click_prana_action(page)
                if clicked_retry:
                    miss_streak = 0
                    continue

                if miss_streak >= NO_BUTTONS_GRACE_CHECKS:
                    # Вероятно, прану потратили/действий нет — сон
                    nap = random.uniform(SLEEP_MIN_SEC, SLEEP_MAX_SEC)
                    logging.info(f"Кнопок нет {miss_streak} раз подряд — сон на {nap / 60:.0f} мин.")
                    miss_streak = 0
                    await asyncio.sleep(nap)

        except PlaywrightTimeoutError as te:
            logging.error(f"Таймаут: {te}")
            await save_debug(page, "timeout_debug")
        except Exception as e:
            logging.error(f"Необработанная ошибка: {e}")
            await save_debug(page, "crash_debug")
        finally:
            if gaz_task:
                gaz_task.cancel()
                with suppress(Exception):
                    await gaz_task
            await context.close()
            await browser.close()


if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logging.info("Остановлено пользователем.")
