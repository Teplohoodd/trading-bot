"""Telegram command and callback handlers."""

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from telegram import Update
from telegram.ext import ContextTypes

from telegram_bot.keyboards import (
    main_menu_keyboard,
    position_keyboard,
    confirmation_keyboard,
    mode_keyboard,
    profile_keyboard,
    pnl_period_keyboard,
    emergency_stop_keyboard,
    screener_keyboard,
    retrain_keyboard,
    back_keyboard,
    tickers_keyboard,
    pending_signals_keyboard,
)
from telegram_bot.formatters import (
    format_portfolio,
    format_status,
    format_pnl_summary,
    format_risk_status,
    format_watchlist,
    format_watchlist_with_direction,
    format_trade_opened,
    format_profile,
    format_custom_tickers,
    format_analysis,
)

logger = logging.getLogger(__name__)


async def _safe_edit(query, text: str, **kwargs):
    """Edit message text, silently ignoring 'not modified' errors."""
    try:
        await query.edit_message_text(text, **kwargs)
    except Exception as e:
        if "message is not modified" in str(e).lower():
            return
        raise


def get_engine(context: ContextTypes.DEFAULT_TYPE):
    return context.bot_data.get("engine")


def get_db(context: ContextTypes.DEFAULT_TYPE):
    return context.bot_data.get("db")


def get_settings(context: ContextTypes.DEFAULT_TYPE):
    return context.bot_data.get("settings")


def auth_check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Check if user is authorized."""
    settings = get_settings(context)
    chat_id = update.effective_chat.id
    # Auto-set chat_id on first interaction
    if settings.TELEGRAM_CHAT_ID == 0:
        settings.TELEGRAM_CHAT_ID = chat_id
        # Also update notification service
        notif = context.bot_data.get("notification_service")
        if notif:
            notif.chat_id = chat_id
        logger.info(f"Chat ID set to {chat_id}")
        return True
    if chat_id != settings.TELEGRAM_CHAT_ID:
        # Don't fail silently — the #1 cause of "bot ignores me" in the wild
        # is an auth mismatch we never log.
        user = update.effective_user
        logger.warning(
            f"Auth rejected: chat_id={chat_id} "
            f"(user=@{getattr(user, 'username', '?')}, id={getattr(user, 'id', '?')}) "
            f"≠ authorised={settings.TELEGRAM_CHAT_ID}"
        )
        return False
    return True


# ==================== Command Handlers ====================


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not auth_check(update, context):
        return
    await update.message.reply_text(
        "<b>Trading Bot</b>\n\n"
        "Autonomous trading with ML signals & risk management.\n"
        "Use the menu below or type /help for commands.",
        parse_mode="HTML",
        reply_markup=main_menu_keyboard(),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not auth_check(update, context):
        return
    await update.message.reply_text(
        "<b>Commands</b>\n\n"
        "/start — Main menu\n"
        "/status — Engine status\n"
        "/portfolio — Current positions\n"
        "/pnl — P&L summary\n"
        "/positions — Open positions details\n"
        "/buy TICKER LOTS — Buy (e.g. /buy SBER 5)\n"
        "/sell TICKER LOTS — Sell\n"
        "/mode — Switch autonomous / advisory / interactive\n"
        "/signals — Pending advisory signals\n"
        "/profile — Trading profile (conservative/moderate/aggressive)\n"
        "/addticker TICKER — Add ticker to custom watchlist\n"
        "/removeticker TICKER — Remove ticker from watchlist\n"
        "/risk — Risk dashboard\n"
        "/watchlist [long|short] — Screener results (default: long)\n"
        "/retrain — Retrain ML model\n"
        "/sync — Force sync with broker (detect external closes)\n"
        "/analyze TICKER — Full analysis: technicals + ML + fundamentals\n"
        "/stop — Emergency stop\n"
        "/help — This message",
        parse_mode="HTML",
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not auth_check(update, context):
        return
    engine = get_engine(context)
    if not engine:
        await update.message.reply_text("Engine not initialized")
        return

    status = await engine.get_status()
    await update.message.reply_text(
        format_status(status),
        parse_mode="HTML",
        reply_markup=main_menu_keyboard(),
    )


async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not auth_check(update, context):
        return
    engine = get_engine(context)
    if not engine:
        return

    positions = await engine.get_portfolio_display()
    text = format_portfolio(positions)

    # Build keyboards for each position
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    buttons = []
    for p in positions:
        buttons.append(
            [
                InlineKeyboardButton(f"Close {p['ticker']}", callback_data=f"close:{p['figi']}"),
                InlineKeyboardButton(f"Details", callback_data=f"details:{p['figi']}"),
            ]
        )
    buttons.append([InlineKeyboardButton("Refresh", callback_data="cmd:portfolio")])
    keyboard = InlineKeyboardMarkup(buttons) if buttons else back_keyboard()

    await update.message.reply_text(text, parse_mode="HTML", reply_markup=keyboard)


async def pnl_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not auth_check(update, context):
        return
    await update.message.reply_text(
        "Select P&L period:",
        reply_markup=pnl_period_keyboard(),
    )


async def positions_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not auth_check(update, context):
        return
    db = get_db(context)
    trades = await db.get_open_trades()

    if not trades:
        await update.message.reply_text("No open positions.", reply_markup=back_keyboard())
        return

    lines = ["<b>Open Positions</b>\n"]
    for t in trades:
        lot_size = t.get("lot_size", 1) or 1
        lots = t["lots"]
        shares = lots * lot_size
        entry = t["entry_price"]
        position_value = shares * entry
        short_tag = " [SHORT]" if t["direction"] == "sell" else ""
        size_label = (
            f"{shares} акц" if lot_size == 1 else f"{lots} лот × {lot_size} акц = {shares} шт"
        )
        lines.append(
            f"<b>{t['ticker']}</b>{short_tag} | {t['direction'].upper()}\n"
            f"  {size_label} / {position_value:,.0f} ₽\n"
            f"  Entry: {entry:.2f} ₽ | SL: {t.get('stop_loss', 'N/A')} | TP: {t.get('take_profit', 'N/A')}\n"
            f"  Strategy: {t['strategy']}"
        )
    await update.message.reply_text(
        "\n".join(lines), parse_mode="HTML", reply_markup=back_keyboard()
    )


async def buy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not auth_check(update, context):
        return
    await _trade_command(update, context, "buy")


async def sell_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not auth_check(update, context):
        return
    await _trade_command(update, context, "sell")


async def _trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE, direction: str):
    """Handle /buy and /sell commands."""
    args = context.args
    if not args or len(args) < 2:
        await update.message.reply_text(
            f"Usage: /{direction} TICKER LOTS\nExample: /{direction} SBER 5"
        )
        return

    ticker = args[0].upper()
    try:
        lots = int(args[1])
    except ValueError:
        await update.message.reply_text("Lots must be a number")
        return

    engine = get_engine(context)
    if not engine:
        return

    # Resolve FIGI — try exact kind first, then fallback to unfiltered
    instruments = await engine.broker.find_instrument(ticker, "INSTRUMENT_KIND_SHARE")
    if not instruments:
        instruments = await engine.broker.find_instrument(ticker)
    if not instruments:
        await update.message.reply_text(f"Instrument {ticker} not found")
        return

    # Prefer exact ticker match
    exact = [i for i in instruments if getattr(i, "ticker", "").upper() == ticker]
    figi = (exact or instruments)[0].figi
    price = await engine.broker.get_last_price(figi)

    await update.message.reply_text(
        f"<b>{direction.upper()} {lots} lots {ticker}</b>\n"
        f"Price: ~{float(price):.2f} RUB\n\n"
        f"Confirm?",
        parse_mode="HTML",
        reply_markup=confirmation_keyboard(direction, f"{figi}:{lots}:{ticker}"),
    )


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not auth_check(update, context):
        return
    settings = get_settings(context)
    await update.message.reply_text(
        f"Current mode: <b>{settings.MODE.upper()}</b>",
        parse_mode="HTML",
        reply_markup=mode_keyboard(settings.MODE),
    )


async def risk_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not auth_check(update, context):
        return
    engine = get_engine(context)
    if not engine:
        return
    risk = await engine.risk_manager.get_risk_status()
    await update.message.reply_text(
        format_risk_status(risk), parse_mode="HTML", reply_markup=back_keyboard()
    )


async def watchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not auth_check(update, context):
        return
    engine = get_engine(context)
    if not engine:
        return

    # Support /watchlist long or /watchlist short
    args = context.args or []
    direction = "short" if args and args[0].lower() == "short" else "long"

    await update.message.reply_text(f"Scanning market ({direction})...", parse_mode="HTML")
    custom = await get_db(context).get_custom_tickers()
    custom_figis = [c["figi"] for c in custom if c.get("figi")]
    candidates = await engine.screener.scan_universe(custom_figis=custom_figis, direction=direction)
    engine._watchlist = candidates
    await update.message.reply_text(
        format_watchlist_with_direction(candidates, direction),
        parse_mode="HTML",
        reply_markup=screener_keyboard(direction),
    )


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Full analysis of a single ticker: technicals + ML + fundamentals."""
    if not auth_check(update, context):
        return
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /analyze TICKER\nExample: /analyze SBER")
        return

    ticker = args[0].upper()
    engine = get_engine(context)
    if not engine:
        return

    msg = await update.message.reply_text(f"⏳ Analysing {ticker}...")

    # Resolve FIGI
    try:
        instruments = await engine.broker.find_instrument(ticker, kind="INSTRUMENT_KIND_SHARE")
        if not instruments:
            instruments = await engine.broker.find_instrument(ticker)
        exact = [i for i in instruments if i.ticker.upper() == ticker]
        instrument = exact[0] if exact else (instruments[0] if instruments else None)
        if not instrument:
            await msg.edit_text(f"❌ Ticker {ticker} not found.")
            return
        figi = instrument.figi
    except Exception as e:
        await msg.edit_text(f"❌ Instrument lookup error: {e}")
        return

    # Get strategies from engine
    ml_strategy = engine.strategies.get("ml_lightgbm")
    tech_strategy = engine.strategies.get("technical")

    try:
        result = await engine.screener.analyze_ticker(
            figi=figi,
            ticker=ticker,
            ml_strategy=ml_strategy,
            tech_strategy=tech_strategy,
        )
        text = format_analysis(result)
        # Telegram message limit is 4096 chars — truncate gracefully
        if len(text) > 4000:
            text = text[:3990] + "\n<i>…(truncated)</i>"
        await msg.edit_text(text, parse_mode="HTML", reply_markup=back_keyboard())
    except Exception as e:
        await msg.edit_text(f"❌ Analysis error: {e}")


async def retrain_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not auth_check(update, context):
        return
    await update.message.reply_text(
        "Retrain ML model?",
        reply_markup=retrain_keyboard(),
    )


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not auth_check(update, context):
        return
    await update.message.reply_text(
        "<b>EMERGENCY STOP</b>\n\n"
        "This will:\n"
        "- Cancel all pending orders\n"
        "- Pause autonomous trading\n\n"
        "Are you sure?",
        parse_mode="HTML",
        reply_markup=emergency_stop_keyboard(),
    )


async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not auth_check(update, context):
        return
    settings = get_settings(context)
    from config.instruments import TRADING_PROFILES

    current = getattr(settings, "ACTIVE_PROFILE", "moderate")
    cfg = TRADING_PROFILES.get(current, TRADING_PROFILES["moderate"])
    await update.message.reply_text(
        format_profile(current, cfg) + "\n\n<i>Select a profile:</i>",
        parse_mode="HTML",
        reply_markup=profile_keyboard(current),
    )


async def addticker_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not auth_check(update, context):
        return
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /addticker TICKER\nExample: /addticker SBER")
        return

    ticker = args[0].upper()
    engine = get_engine(context)
    db = get_db(context)

    # Resolve FIGI
    figi, name = "", ""
    try:
        instruments = await engine.broker.find_instrument(ticker, "INSTRUMENT_KIND_SHARE")
        if not instruments:
            instruments = await engine.broker.find_instrument(ticker)
        exact = [i for i in instruments if getattr(i, "ticker", "").upper() == ticker]
        inst = (exact or instruments)[0] if instruments else None
        if inst:
            figi = inst.figi
            name = getattr(inst, "name", "")
    except Exception as e:
        logger.warning(f"Could not resolve FIGI for {ticker}: {e}")

    await db.add_custom_ticker(ticker, figi, name)
    # Invalidate watchlist cache so next scan includes this ticker
    if engine:
        engine._watchlist = []

    await update.message.reply_text(
        f"Added <b>{ticker}</b> to custom watchlist.\n"
        f"{'FIGI: ' + figi if figi else 'FIGI not resolved — will be included in next scan'}",
        parse_mode="HTML",
        reply_markup=back_keyboard(),
    )


async def removeticker_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not auth_check(update, context):
        return
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /removeticker TICKER")
        return

    ticker = args[0].upper()
    db = get_db(context)
    engine = get_engine(context)
    await db.remove_custom_ticker(ticker)
    if engine:
        engine._watchlist = []
    await update.message.reply_text(
        f"Removed <b>{ticker}</b> from custom watchlist.", parse_mode="HTML"
    )


async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show pending advisory signals."""
    if not auth_check(update, context):
        return
    engine = get_engine(context)
    if not engine:
        return
    pending = engine.get_pending_signals()
    if not pending:
        await update.message.reply_text("No pending signals.", reply_markup=back_keyboard())
        return
    await update.message.reply_text(
        f"<b>Pending Signals ({len(pending)})</b>",
        parse_mode="HTML",
        reply_markup=pending_signals_keyboard(pending),
    )


async def sync_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Force immediate sync with broker to detect externally closed positions."""
    if not auth_check(update, context):
        return
    engine = get_engine(context)
    if not engine:
        await update.message.reply_text("Engine not running.")
        return
    msg = await update.message.reply_text("⏳ Syncing with broker...")
    try:
        await engine._sync_portfolio()
        open_trades = await get_db(context).get_open_trades()
        await msg.edit_text(
            f"✅ Sync complete. Open positions: <b>{len(open_trades)}</b>",
            parse_mode="HTML",
            reply_markup=back_keyboard(),
        )
    except Exception as e:
        await msg.edit_text(f"❌ Sync error: {e}", reply_markup=back_keyboard())


# ==================== Callback Query Handler ====================


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all inline button presses."""
    query = update.callback_query
    await query.answer()

    if not auth_check(update, context):
        return

    data = query.data
    engine = get_engine(context)
    db = get_db(context)
    settings = get_settings(context)

    # --- Navigation ---
    if data == "cmd:menu":
        await _safe_edit(query, "Main menu:", parse_mode="HTML", reply_markup=main_menu_keyboard())
        return

    if data == "cmd:status":
        status = await engine.get_status()
        await _safe_edit(
            query, format_status(status), parse_mode="HTML", reply_markup=main_menu_keyboard()
        )
        return

    if data == "cmd:portfolio":
        positions = await engine.get_portfolio_display()
        await _safe_edit(
            query, format_portfolio(positions), parse_mode="HTML", reply_markup=back_keyboard()
        )
        return

    if data == "cmd:positions":
        trades = await db.get_open_trades()
        if not trades:
            await _safe_edit(query, "No open positions.", reply_markup=back_keyboard())
            return
        lines = ["<b>Open Positions</b>\n"]
        for t in trades:
            lot_size = t.get("lot_size", 1) or 1
            lots = t["lots"]
            shares = lots * lot_size
            entry = t["entry_price"]
            position_value = shares * entry
            short_tag = " [SHORT]" if t["direction"] == "sell" else ""
            size_label = (
                f"{shares} акц" if lot_size == 1 else f"{lots} лот × {lot_size} акц = {shares} шт"
            )
            lines.append(
                f"<b>{t['ticker']}</b>{short_tag} | {t['direction'].upper()}\n"
                f"  {size_label} / {position_value:,.0f} ₽\n"
                f"  Entry: {entry:.2f} ₽"
            )
        await _safe_edit(query, "\n".join(lines), parse_mode="HTML", reply_markup=back_keyboard())
        return

    if data == "cmd:risk":
        risk = await engine.risk_manager.get_risk_status()
        await _safe_edit(
            query, format_risk_status(risk), parse_mode="HTML", reply_markup=back_keyboard()
        )
        return

    if data == "cmd:mode":
        await _safe_edit(
            query,
            f"Current mode: <b>{settings.MODE.upper()}</b>",
            parse_mode="HTML",
            reply_markup=mode_keyboard(settings.MODE),
        )
        return

    if data == "cmd:watchlist":
        await _safe_edit(
            query,
            "Выберите направление скрининга:",
            parse_mode="HTML",
            reply_markup=screener_keyboard(),
        )
        return

    if data == "cmd:pnl":
        await _safe_edit(query, "Select period:", reply_markup=pnl_period_keyboard())
        return

    if data == "cmd:profile":
        from config.instruments import TRADING_PROFILES

        current = getattr(settings, "ACTIVE_PROFILE", "moderate")
        cfg = TRADING_PROFILES.get(current, TRADING_PROFILES["moderate"])
        await _safe_edit(
            query,
            format_profile(current, cfg) + "\n\n<i>Select a profile:</i>",
            parse_mode="HTML",
            reply_markup=profile_keyboard(current),
        )
        return

    if data == "cmd:tickers":
        tickers = await db.get_custom_tickers()
        await _safe_edit(
            query,
            format_custom_tickers(tickers),
            parse_mode="HTML",
            reply_markup=tickers_keyboard(tickers),
        )
        return

    if data == "cmd:settings":
        current_profile = getattr(settings, "ACTIVE_PROFILE", "moderate")
        order_mode = getattr(settings, "ORDER_EXECUTION_MODE", "market")
        shorts = getattr(settings, "ALLOW_SHORTS", False)
        await _safe_edit(
            query,
            f"<b>Settings</b>\n\n"
            f"Mode: {settings.MODE}\n"
            f"Profile: {current_profile}\n"
            f"Order execution: {order_mode}\n"
            f"Shorts: {'enabled' if shorts else 'disabled'}\n"
            f"Max positions: {settings.MAX_POSITIONS}\n"
            f"Max risk/trade: {settings.MAX_PORTFOLIO_RISK_PCT:.0%}\n"
            f"Max daily loss: {settings.MAX_DAILY_LOSS_PCT:.0%}\n"
            f"Max drawdown: {settings.MAX_DRAWDOWN_PCT:.0%}\n"
            f"Leverage: {'ON' if settings.USE_LEVERAGE else 'OFF'}\n"
            f"Signal threshold: {settings.SIGNAL_THRESHOLD}\n"
            f"Scan interval: {settings.SCAN_INTERVAL_MINUTES}min",
            parse_mode="HTML",
            reply_markup=back_keyboard(),
        )
        return

    if data == "cancel":
        await _safe_edit(query, "Cancelled.", reply_markup=back_keyboard())
        return

    # --- Mode switch ---
    if data.startswith("mode:"):
        new_mode = data.split(":")[1]
        settings.MODE = new_mode
        if engine:
            engine._mode = new_mode
        # Persist to .env so restarts retain the chosen mode
        try:
            env_path = Path(".env")
            if env_path.exists():
                text = env_path.read_text(encoding="utf-8")
                text = re.sub(r"^MODE=.*$", f"MODE={new_mode}", text, flags=re.MULTILINE)
                env_path.write_text(text, encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Could not persist MODE to .env: {exc}")
        await _safe_edit(
            query,
            f"Mode switched to <b>{new_mode.upper()}</b>",
            parse_mode="HTML",
            reply_markup=main_menu_keyboard(),
        )
        return

    # --- Profile switch ---
    if data.startswith("profile:"):
        from config.instruments import TRADING_PROFILES

        profile_key = data.split(":")[1]
        cfg = TRADING_PROFILES.get(profile_key)
        if not cfg:
            await _safe_edit(query, "Unknown profile.", reply_markup=back_keyboard())
            return
        # Apply settings
        for k, v in cfg.items():
            if k not in ("label", "description") and hasattr(settings, k):
                setattr(settings, k, v)
        settings.ACTIVE_PROFILE = profile_key
        # Persist to .env
        try:
            env_path = Path(".env")
            if env_path.exists():
                text_env = env_path.read_text(encoding="utf-8")
                for k, v in cfg.items():
                    if k in ("label", "description"):
                        continue
                    pattern = rf"^{k}=.*$"
                    replacement = f"{k}={v}"
                    text_env = re.sub(pattern, replacement, text_env, flags=re.MULTILINE)
                    if not re.search(rf"^{k}=", text_env, re.MULTILINE):
                        text_env += f"\n{k}={v}"
                text_env = re.sub(
                    r"^ACTIVE_PROFILE=.*$",
                    f"ACTIVE_PROFILE={profile_key}",
                    text_env,
                    flags=re.MULTILINE,
                )
                if not re.search(r"^ACTIVE_PROFILE=", text_env, re.MULTILINE):
                    text_env += f"\nACTIVE_PROFILE={profile_key}"
                env_path.write_text(text_env, encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Could not persist profile to .env: {exc}")
        await _safe_edit(
            query,
            f"Profile set to <b>{cfg['label']}</b>\n\n" + format_profile(profile_key, cfg),
            parse_mode="HTML",
            reply_markup=main_menu_keyboard(),
        )
        return

    # --- Custom tickers ---
    if data.startswith("ticker:remove:"):
        ticker = data.split(":")[2]
        await db.remove_custom_ticker(ticker)
        if engine:
            engine._watchlist = []
        tickers = await db.get_custom_tickers()
        await _safe_edit(
            query,
            format_custom_tickers(tickers),
            parse_mode="HTML",
            reply_markup=tickers_keyboard(tickers),
        )
        return

    if data == "ticker:add_prompt":
        await _safe_edit(
            query,
            "Send the ticker with command:\n<code>/addticker TICKER</code>",
            parse_mode="HTML",
            reply_markup=back_keyboard(),
        )
        return

    # --- Advisory signals ---
    if data.startswith("advisory:execute:"):
        signal_id = data.split("advisory:execute:")[1]
        await _safe_edit(query, f"Executing signal {signal_id}...")
        try:
            result = await engine.execute_advisory_signal(signal_id)
            if result.get("success"):
                from telegram_bot.formatters import format_trade_opened

                await _safe_edit(
                    query,
                    format_trade_opened(result),
                    parse_mode="HTML",
                    reply_markup=back_keyboard(),
                )
            else:
                await _safe_edit(
                    query,
                    f"Trade failed: {result.get('reason', 'unknown')}",
                    reply_markup=back_keyboard(),
                )
        except Exception as e:
            await _safe_edit(query, f"Error: {e}", reply_markup=back_keyboard())
        return

    if data.startswith("advisory:skip:"):
        signal_id = data.split("advisory:skip:")[1]
        engine._pending_signals.pop(signal_id, None)
        await _safe_edit(query, "Signal skipped.", reply_markup=back_keyboard())
        return

    if data.startswith("advisory:detail:"):
        signal_id = data.split("advisory:detail:")[1]
        pending = engine._pending_signals.get(signal_id)
        if pending:
            from telegram_bot.formatters import format_advisory_signal
            from telegram_bot.keyboards import advisory_signal_keyboard

            await _safe_edit(
                query,
                format_advisory_signal(pending),
                parse_mode="HTML",
                reply_markup=advisory_signal_keyboard(signal_id),
            )
        else:
            await _safe_edit(query, "Signal expired.", reply_markup=back_keyboard())
        return

    # --- P&L periods ---
    if data.startswith("pnl:"):
        period = data.split(":")[1]
        now = datetime.now(timezone.utc)
        period_map = {
            "daily": (now - timedelta(days=1), "Today"),
            "weekly": (now - timedelta(weeks=1), "This Week"),
            "monthly": (now - timedelta(days=30), "This Month"),
            "all": (now - timedelta(days=365), "All Time"),
        }
        from_dt, label = period_map.get(period, (now - timedelta(days=1), "Today"))
        trades = await db.get_trades(from_dt.isoformat(), now.isoformat())
        closed = [t for t in trades if t["status"] == "closed"]
        text = format_pnl_summary(closed, label)
        from telegram import InlineKeyboardMarkup, InlineKeyboardButton

        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Today", callback_data="pnl:daily"),
                    InlineKeyboardButton("Week", callback_data="pnl:weekly"),
                ],
                [
                    InlineKeyboardButton("Month", callback_data="pnl:monthly"),
                    InlineKeyboardButton("All time", callback_data="pnl:all"),
                ],
                [InlineKeyboardButton("🏠 Main Menu", callback_data="cmd:menu")],
            ]
        )
        await _safe_edit(query, text, parse_mode="HTML", reply_markup=kb)
        return

    # --- Trade confirmation ---
    if data.startswith("buy_confirm:") or data.startswith("sell_confirm:"):
        parts = data.split(":")
        direction = parts[0].replace("_confirm", "")
        params = parts[1].split(":")
        # params format: figi:lots:ticker (but : is also delimiter)
        # Re-parse: everything after "buy_confirm:" or "sell_confirm:"
        param_str = data.split(":", 1)[1]
        param_parts = param_str.split(":")
        figi = param_parts[0]
        lots = int(param_parts[1])
        ticker = param_parts[2] if len(param_parts) > 2 else "???"

        await _safe_edit(query, f"Executing {direction} {lots} lots {ticker}...")

        try:
            result = await engine.manual_trade(figi, ticker, lots, direction)
            if result.get("success"):
                await _safe_edit(
                    query,
                    format_trade_opened(result),
                    parse_mode="HTML",
                    reply_markup=back_keyboard(),
                )
            else:
                await _safe_edit(
                    query,
                    f"Trade rejected: {result.get('reason', 'unknown')}",
                    reply_markup=back_keyboard(),
                )
        except Exception as e:
            await _safe_edit(query, f"Error: {e}", reply_markup=back_keyboard())
        return

    # --- Close position ---
    if data.startswith("close:"):
        figi = data.split(":")[1]
        await _safe_edit(
            query,
            f"Close position {figi}?",
            reply_markup=confirmation_keyboard("close", figi),
        )
        return

    if data.startswith("close_confirm:"):
        figi = data.split(":")[1]
        await _safe_edit(query, "Closing position...")
        try:
            result = await engine.close_position(figi, "manual")
            await _safe_edit(
                query,
                f"Position closed. P&L: {result.get('pnl', 0):.0f} RUB",
                parse_mode="HTML",
                reply_markup=back_keyboard(),
            )
        except Exception as e:
            await _safe_edit(query, f"Error closing: {e}", reply_markup=back_keyboard())
        return

    # --- Emergency stop ---
    if data == "stop_confirm:all":
        await _safe_edit(query, "Executing emergency stop...")
        try:
            await engine.emergency_stop()
            await _safe_edit(
                query,
                "<b>EMERGENCY STOP EXECUTED</b>\n\nAll orders cancelled. Autonomous trading paused.",
                parse_mode="HTML",
                reply_markup=main_menu_keyboard(),
            )
        except Exception as e:
            await _safe_edit(query, f"Error: {e}", reply_markup=back_keyboard())
        return

    # --- Screener ---
    if data in ("screener:long", "screener:short", "screener:run"):
        direction = "short" if data == "screener:short" else "long"
        await _safe_edit(query, f"Scanning market ({direction})...")
        custom = await db.get_custom_tickers()
        custom_figis = [c["figi"] for c in custom if c.get("figi")]
        candidates = await engine.screener.scan_universe(
            custom_figis=custom_figis, direction=direction
        )
        engine._watchlist = candidates
        await _safe_edit(
            query,
            format_watchlist_with_direction(candidates, direction),
            parse_mode="HTML",
            reply_markup=screener_keyboard(direction),
        )
        return

    # --- Retrain ---
    if data == "retrain:all":
        await _safe_edit(query, "Retraining model... This may take a few minutes.")
        try:
            await engine.retrain_models()
            await _safe_edit(query, "Model retrained successfully.", reply_markup=back_keyboard())
        except Exception as e:
            await _safe_edit(query, f"Retrain error: {e}", reply_markup=back_keyboard())
        return

    # --- Details ---
    if data.startswith("details:"):
        figi = data.split(":")[1]
        trades = await db.get_open_trades()
        trade = next((t for t in trades if t["figi"] == figi), None)
        if trade:
            price = await engine.broker.get_last_price(figi)
            entry = trade["entry_price"]
            pnl_pct = (float(price) - entry) / entry * 100 if entry > 0 else 0
            await _safe_edit(
                query,
                f"<b>{trade['ticker']} Details</b>\n\n"
                f"Direction: {trade['direction'].upper()}\n"
                f"Lots: {trade['lots']}\n"
                f"Entry: {entry:.2f}\n"
                f"Current: {float(price):.2f}\n"
                f"P&L: {pnl_pct:+.1f}%\n"
                f"Stop: {trade.get('stop_loss', 'N/A')}\n"
                f"Target: {trade.get('take_profit', 'N/A')}\n"
                f"Strategy: {trade['strategy']}\n"
                f"Opened: {trade['entry_time']}",
                parse_mode="HTML",
                reply_markup=position_keyboard(figi, trade["ticker"]),
            )
        else:
            await _safe_edit(query, "Position not found.", reply_markup=back_keyboard())
        return
