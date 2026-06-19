"""InlineKeyboardMarkup builders for Telegram bot."""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Status", callback_data="cmd:status"),
                InlineKeyboardButton("Portfolio", callback_data="cmd:portfolio"),
            ],
            [
                InlineKeyboardButton("Positions", callback_data="cmd:positions"),
                InlineKeyboardButton("P&L", callback_data="cmd:pnl"),
            ],
            [
                InlineKeyboardButton("Watchlist", callback_data="cmd:watchlist"),
                InlineKeyboardButton("Risk", callback_data="cmd:risk"),
            ],
            [
                InlineKeyboardButton("Mode", callback_data="cmd:mode"),
                InlineKeyboardButton("Profile", callback_data="cmd:profile"),
            ],
            [
                InlineKeyboardButton("Tickers", callback_data="cmd:tickers"),
                InlineKeyboardButton("Settings", callback_data="cmd:settings"),
            ],
        ]
    )


def position_keyboard(figi: str, ticker: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(f"Close {ticker}", callback_data=f"close:{figi}"),
                InlineKeyboardButton("Details", callback_data=f"details:{figi}"),
            ],
        ]
    )


def confirmation_keyboard(action: str, params: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("CONFIRM", callback_data=f"{action}_confirm:{params}"),
                InlineKeyboardButton("Cancel", callback_data="cancel"),
            ],
        ]
    )


def mode_keyboard(current_mode: str) -> InlineKeyboardMarkup:
    modes = ["autonomous", "advisory", "interactive"]
    buttons = []
    for m in modes:
        label = f"[{m.upper()}]" if m == current_mode else m.capitalize()
        buttons.append(InlineKeyboardButton(label, callback_data=f"mode:{m}"))
    return InlineKeyboardMarkup([buttons, [InlineKeyboardButton("Cancel", callback_data="cancel")]])


def profile_keyboard(current_profile: str) -> InlineKeyboardMarkup:
    from config.instruments import TRADING_PROFILES

    buttons = []
    for key, cfg in TRADING_PROFILES.items():
        label = f"[{cfg['label']}]" if key == current_profile else cfg["label"]
        buttons.append([InlineKeyboardButton(label, callback_data=f"profile:{key}")])
    buttons.append([InlineKeyboardButton("Cancel", callback_data="cancel")])
    return InlineKeyboardMarkup(buttons)


def pnl_period_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Today", callback_data="pnl:daily"),
                InlineKeyboardButton("Week", callback_data="pnl:weekly"),
            ],
            [
                InlineKeyboardButton("Month", callback_data="pnl:monthly"),
                InlineKeyboardButton("All time", callback_data="pnl:all"),
            ],
        ]
    )


def emergency_stop_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("CONFIRM EMERGENCY STOP", callback_data="stop_confirm:all")],
            [InlineKeyboardButton("Cancel", callback_data="cancel")],
        ]
    )


def screener_keyboard(direction: str = "long") -> InlineKeyboardMarkup:
    long_label = "[📈 Long]" if direction == "long" else "📈 Long"
    short_label = "[📉 Short]" if direction == "short" else "📉 Short"
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(long_label, callback_data="screener:long"),
                InlineKeyboardButton(short_label, callback_data="screener:short"),
            ],
            [InlineKeyboardButton("Manage Tickers", callback_data="cmd:tickers")],
            [InlineKeyboardButton("Back", callback_data="cmd:status")],
        ]
    )


def retrain_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Retrain All", callback_data="retrain:all")],
            [InlineKeyboardButton("Cancel", callback_data="cancel")],
        ]
    )


def back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Back to Menu", callback_data="cmd:menu")],
        ]
    )


def tickers_keyboard(custom_tickers: list[dict]) -> InlineKeyboardMarkup:
    """Show custom watchlist with remove buttons."""
    buttons = []
    for t in custom_tickers:
        buttons.append(
            [
                InlineKeyboardButton(
                    f"❌ Remove {t['ticker']}", callback_data=f"ticker:remove:{t['ticker']}"
                )
            ]
        )
    buttons.append([InlineKeyboardButton("+ Add Ticker", callback_data="ticker:add_prompt")])
    buttons.append([InlineKeyboardButton("Back", callback_data="cmd:menu")])
    return InlineKeyboardMarkup(buttons)


def advisory_signal_keyboard(signal_id: str) -> InlineKeyboardMarkup:
    """Approve / skip keyboard for advisory signals."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ Execute", callback_data=f"advisory:execute:{signal_id}"),
                InlineKeyboardButton("❌ Skip", callback_data=f"advisory:skip:{signal_id}"),
            ]
        ]
    )


def pending_signals_keyboard(signals: list[dict]) -> InlineKeyboardMarkup:
    """List all pending advisory signals with execute/skip buttons."""
    buttons = []
    for s in signals[:10]:  # max 10
        sid = s["signal_id"]
        label = f"{s['direction'].upper()} {s['ticker']} ({s['confidence']:.0%})"
        buttons.append(
            [
                InlineKeyboardButton(label, callback_data=f"advisory:detail:{sid}"),
                InlineKeyboardButton("✅", callback_data=f"advisory:execute:{sid}"),
                InlineKeyboardButton("❌", callback_data=f"advisory:skip:{sid}"),
            ]
        )
    buttons.append([InlineKeyboardButton("Back", callback_data="cmd:menu")])
    return InlineKeyboardMarkup(buttons)
