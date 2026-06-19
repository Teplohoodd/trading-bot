"""Message formatters for Telegram bot: portfolio, P&L, alerts, signals."""


def _fmt_position_size(lots: int, lot_size: int, price: float) -> str:
    """'1 лот × 100 акц = 1 486 ₽'"""
    shares = lots * lot_size
    value = shares * price
    if lot_size == 1:
        return f"{shares} акц × {price:.2f} ₽ = {value:,.0f} ₽"
    return f"{lots} лот × {lot_size} акц = {shares} шт / {value:,.0f} ₽"


def format_portfolio(positions: list[dict]) -> str:
    """Format portfolio positions as HTML table."""
    if not positions:
        return "<b>Portfolio is empty</b>"

    lines = ["<b>Портфель</b>\n"]
    total_pnl = 0

    for p in positions:
        ticker = p.get("ticker", "???")
        lots = p.get("lots", 0)
        lot_size = p.get("lot_size", 1)
        entry = p.get("entry_price", 0)
        current = p.get("current_price", entry)
        direction = p.get("direction", "buy")
        shares = lots * lot_size

        if direction == "buy":
            pnl_pct = (current - entry) / entry * 100 if entry > 0 else 0
            pnl_abs = (current - entry) * shares
        else:
            pnl_pct = (entry - current) / entry * 100 if entry > 0 else 0
            pnl_abs = (entry - current) * shares

        total_pnl += pnl_abs
        sign = "+" if pnl_pct >= 0 else ""
        short_tag = " [SHORT]" if direction == "sell" else ""

        lot_word = "лот" + ("а" if 2 <= lots <= 4 else "ов" if lots >= 5 else "")
        position_value = shares * current
        if lot_size == 1:
            size_line = f"{shares} акц"
        else:
            size_line = f"{lots} {lot_word} × {lot_size} акц = {shares} шт"
        lines.append(
            f"<b>{ticker}</b>{short_tag}  {size_line}\n"
            f"  Вход: {entry:.2f} → Сейчас: {current:.2f} ₽ | Объём: {position_value:,.0f} ₽\n"
            f"  P&L: <b>{sign}{pnl_abs:,.0f} ₽ ({sign}{pnl_pct:.1f}%)</b>"
        )

    sign = "+" if total_pnl >= 0 else ""
    lines.append(f"\n<b>Итого P&L: {sign}{total_pnl:,.0f} ₽</b>")
    return "\n".join(lines)


def format_status(status: dict) -> str:
    """Format engine status."""
    mode = status.get("mode", "interactive").upper()
    running = "Running" if status.get("is_running") else "Stopped"
    positions = status.get("open_positions", 0)
    daily_pnl = status.get("daily_pnl", 0)
    emoji = "+" if daily_pnl >= 0 else ""

    return (
        f"<b>Trading Bot Status</b>\n\n"
        f"Mode: <b>{mode}</b>\n"
        f"Engine: <b>{running}</b>\n"
        f"Open positions: <b>{positions}</b>\n"
        f"Daily P&L: <b>{emoji}{daily_pnl:.0f} RUB</b>"
    )


def format_pnl_summary(trades: list[dict], period: str) -> str:
    """Format P&L summary for a period."""
    if not trades:
        return f"<b>P&L ({period})</b>\n\nNo trades in this period."

    total_pnl = sum(t.get("pnl", 0) or 0 for t in trades)
    wins = [t for t in trades if (t.get("pnl") or 0) > 0]
    losses = [t for t in trades if (t.get("pnl") or 0) <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(abs(t["pnl"]) for t in losses) / len(losses) if losses else 0

    profit_factor = (
        sum(t["pnl"] for t in wins) / sum(abs(t["pnl"]) for t in losses)
        if losses and sum(abs(t["pnl"]) for t in losses) > 0
        else 0
    )

    emoji = "+" if total_pnl >= 0 else ""
    return (
        f"<b>P&L Summary ({period})</b>\n\n"
        f"Total P&L: <b>{emoji}{total_pnl:.0f} RUB</b>\n"
        f"Trades: <b>{len(trades)}</b>\n"
        f"Win rate: <b>{win_rate:.0f}%</b>\n"
        f"Avg win: <b>+{avg_win:.0f} RUB</b>\n"
        f"Avg loss: <b>-{avg_loss:.0f} RUB</b>\n"
        f"Profit factor: <b>{profit_factor:.2f}</b>"
    )


def format_risk_status(risk: dict) -> str:
    """Format risk status."""
    realized = risk.get("daily_pnl_realized", 0)
    unrealized = risk.get("daily_pnl_unrealized", 0)
    total = risk.get("daily_pnl", realized + unrealized)
    return (
        f"<b>Risk Dashboard</b>\n\n"
        f"Daily P&L: <b>{total:+,.0f} RUB</b>\n"
        f"  realized:   {realized:+,.0f} RUB\n"
        f"  unrealized: {unrealized:+,.0f} RUB\n"
        f"Daily loss limit: {risk.get('daily_loss_limit', 'N/A')}\n"
        f"Drawdown: <b>{risk.get('drawdown', '0%')}</b>\n"
        f"Drawdown limit: {risk.get('drawdown_limit', 'N/A')}\n"
        f"Portfolio: <b>{risk.get('current_portfolio', 0):,.0f} RUB</b>\n"
        f"Peak: {risk.get('peak_portfolio', 0):,.0f} RUB\n"
        f"Trades today: {risk.get('trades_today', 0)}"
    )


def format_signal(signal_data: dict) -> str:
    """Format a trading signal notification."""
    direction = signal_data.get("direction", "hold").upper()
    ticker = signal_data.get("ticker", "???")
    confidence = signal_data.get("confidence", 0)
    strategy = signal_data.get("strategy", "unknown")

    arrow = {"BUY": "BUY", "SELL": "SELL"}.get(direction, "HOLD")

    return (
        f"<b>Signal: {arrow} {ticker}</b>\n\n"
        f"Confidence: <b>{confidence:.0%}</b>\n"
        f"Strategy: {strategy}"
    )


def format_trade_opened(trade: dict) -> str:
    """Format trade opened notification."""
    direction = trade.get("direction", "buy").upper()
    ticker = trade.get("ticker", "???")
    lots = trade.get("lots", 0)
    lot_size = trade.get("lot_size", 1)
    price = trade.get("entry_price", 0)
    stop = trade.get("stop_loss", 0)
    target = trade.get("take_profit", 0)
    strategy = trade.get("strategy", "")
    confidence = trade.get("signal_confidence", 0)
    order_type = trade.get("order_type", "market")

    shares = lots * lot_size
    position_value = shares * price

    stop_pct = abs(stop - price) / price * 100 if price > 0 else 0
    short_label = " [SHORT]" if direction == "SELL" else ""

    # For shorts: target is below entry, stop above
    if direction == "SELL":
        stop_pct_label = f"+{stop_pct:.1f}%"
        target_pct = abs(target - price) / price * 100 if price > 0 else 0
        target_pct_label = f"-{target_pct:.1f}%"
    else:
        stop_pct_label = f"-{stop_pct:.1f}%"
        target_pct = abs(target - price) / price * 100 if price > 0 else 0
        target_pct_label = f"+{target_pct:.1f}%"

    size_line = _fmt_position_size(lots, lot_size, price)

    return (
        f"<b>OPEN {direction}{short_label} | {ticker}</b>\n\n"
        f"{size_line} [{order_type}]\n"
        f"Цена: {price:.2f} ₽ | Объём: {position_value:,.0f} ₽\n"
        f"Stop: {stop:.2f} ({stop_pct_label})\n"
        f"Target: {target:.2f} ({target_pct_label})\n"
        f"Strategy: {strategy} (conf: {confidence:.0%})"
    )


def format_trade_closed(trade: dict) -> str:
    """Format trade closed notification."""
    ticker = trade.get("ticker", "???")
    direction = trade.get("direction", "buy").upper()
    lots = trade.get("lots", 0)
    lot_size = trade.get("lot_size", 1)
    entry = trade.get("entry_price", 0)
    exit_p = trade.get("exit_price", 0)
    pnl = trade.get("pnl", 0)
    pnl_pct = trade.get("pnl_pct", 0)
    reason = trade.get("exit_reason", "unknown")

    shares = lots * lot_size
    emoji = "+" if pnl >= 0 else ""
    result = "PROFIT" if pnl >= 0 else "LOSS"
    size_label = f"{shares} акц" if lot_size == 1 else f"{lots} лот × {lot_size} акц = {shares} шт"

    return (
        f"<b>CLOSED {direction} | {ticker} | {result}</b>\n\n"
        f"{size_label}: {entry:.2f} → {exit_p:.2f} ₽\n"
        f"P&L: <b>{emoji}{pnl:,.0f} ₽ ({emoji}{pnl_pct:.1f}%)</b>\n"
        f"Reason: {reason}"
    )


def format_watchlist(candidates: list[dict]) -> str:
    """Format screener results (legacy, delegates to direction-aware formatter)."""
    direction = candidates[0].get("direction", "long") if candidates else "long"
    return format_watchlist_with_direction(candidates, direction)


def format_advisory_signal(data: dict) -> str:
    """Format advisory signal notification (requires user approval)."""
    direction = data.get("direction", "hold").upper()
    ticker = data.get("ticker", "???")
    confidence = data.get("confidence", 0)
    strategy = data.get("strategy", "unknown")
    lots = data.get("lots", 0)
    lot_size = data.get("lot_size", 1) or data.get("risk_metrics", {}).get("lot_size", 1)
    stop = data.get("stop_loss", 0)
    target = data.get("take_profit", 0)
    metrics = data.get("risk_metrics", {})
    price = metrics.get("price", 0) or 0

    shares = lots * lot_size
    position_value = shares * price if price else 0
    size_line = f"{shares} акц" if lot_size == 1 else f"{lots} лот × {lot_size} акц = {shares} шт"
    value_line = f" / {position_value:,.0f} ₽" if position_value else ""

    return (
        f"<b>SIGNAL: {direction} {ticker}</b>\n\n"
        f"Confidence: <b>{confidence:.0%}</b>\n"
        f"Strategy: {strategy}\n"
        f"Size: <b>{size_line}</b>{value_line}\n"
        f"Stop: {stop:.2f} ₽\n"
        f"Target: {target:.2f} ₽\n"
        f"Impact: {metrics.get('impact', 0):.2f}%\n\n"
        f"<i>Approve or skip this trade:</i>"
    )


def format_profile(profile_key: str, cfg: dict) -> str:
    """Format trading profile details."""
    return (
        f"<b>Profile: {cfg['label']}</b>\n"
        f"<i>{cfg['description']}</i>\n\n"
        f"Max positions: {cfg['MAX_POSITIONS']}\n"
        f"Max position: {cfg['MAX_POSITION_PCT']:.0%}\n"
        f"Risk/trade: {cfg['MAX_PORTFOLIO_RISK_PCT']:.0%}\n"
        f"Daily loss limit: {cfg['MAX_DAILY_LOSS_PCT']:.0%}\n"
        f"Max drawdown: {cfg['MAX_DRAWDOWN_PCT']:.0%}\n"
        f"Signal threshold: {cfg['SIGNAL_THRESHOLD']:.0%}\n"
        f"Kelly fraction: {cfg['KELLY_FRACTION']:.0%}\n"
        f"Scan interval: {cfg['SCAN_INTERVAL_MINUTES']}min"
    )


def _signal_arrow(direction: str, confidence: float) -> str:
    """Return a visual arrow with confidence bar."""
    if direction == "buy":
        return f"🟢 BUY  {confidence:.0%}"
    if direction == "sell":
        return f"🔴 SELL {confidence:.0%}"
    return f"⚪ HOLD {confidence:.0%}"


def format_analysis(data: dict) -> str:
    """Format full ticker analysis report."""
    ticker = data.get("ticker", "???")
    name = data.get("name", ticker)
    sector = data.get("sector", "")
    price = data.get("price", 0)
    lot_size = data.get("lot_size", 1)

    if data.get("error"):
        return f"<b>{ticker}</b>\n\n❌ Error: {data['error']}"

    lines = [f"<b>📊 Анализ: {ticker}</b>"]
    if name and name != ticker:
        lines.append(f"<i>{name}</i>")
    if sector:
        lines.append(f"Сектор: {sector}")
    lines.append(f"Цена: <b>{price:.2f} ₽</b>  |  Лот: {lot_size} акц")

    # Order book
    ob = data.get("order_book")
    if ob:
        imb = ob.get("imbalance", 0)
        imb_str = f"{'📈' if imb > 0.1 else '📉' if imb < -0.1 else '⚖️'} {imb:+.2f}"
        lines.append(
            f"Bid: {ob['bid']:.2f} / Ask: {ob['ask']:.2f}  "
            f"Спред: {ob['spread_bps']:.1f}bp  Дисбаланс: {imb_str}"
        )

    # Volume & volatility
    avg_vol_rub = data.get("avg_volume_rub", 0)
    m5 = data.get("momentum_5d", 0)
    m20 = data.get("momentum_20d", 0)
    if avg_vol_rub:
        lines.append(
            f"\nОбъём ср/ч: <b>{avg_vol_rub/1e6:.1f}M ₽</b> (≈ {avg_vol_rub*7/1e6:.0f}M ₽/д)"
        )
    lines.append(f"Моментум 5д: <b>{m5:+.1f}%</b>  |  20д: <b>{m20:+.1f}%</b>")

    # Technical indicators
    ind = data.get("indicators", {})
    if ind:
        lines.append("\n<b>─ Технические индикаторы ─</b>")
        rsi14 = ind.get("rsi_14", 0)
        rsi_label = "🔥 перекуплен" if rsi14 > 70 else "❄️ перепродан" if rsi14 < 30 else ""
        lines.append(f"RSI(14): <b>{rsi14:.1f}</b> {rsi_label}  RSI(7): {ind.get('rsi_7', 0):.1f}")

        macd_h = ind.get("macd_hist", 0)
        macd_label = "↑" if macd_h > 0 else "↓"
        lines.append(
            f"MACD: {ind.get('macd', 0):.4f} / Sig: {ind.get('macd_signal', 0):.4f}  "
            f"Hist: {macd_h:+.4f} {macd_label}"
        )

        bb = ind.get("bb_pct_b", 0)
        bb_label = "выше верхней полосы" if bb > 1 else "ниже нижней полосы" if bb < 0 else ""
        lines.append(f"BB %B: <b>{bb:.2f}</b> {bb_label}  Ширина: {ind.get('bb_width', 0):.3f}")

        atr_pct = ind.get("atr_pct", 0)
        lines.append(
            f"ATR(14): {ind.get('atr_14', 0):.4f} ({atr_pct:.2f}%)  ADX: {ind.get('adx_14', 0):.1f}"
        )

        stk = ind.get("stoch_k", 0)
        std = ind.get("stoch_d", 0)
        lines.append(f"Stoch %K: {stk:.1f}  %D: {std:.1f}")

        ema9 = ind.get("ema_9", 0)
        ema21 = ind.get("ema_21", 0)
        ema50 = ind.get("ema_50", 0)
        sma200 = ind.get("sma_200", 0)
        if price > 0:

            def _vs(ma):
                return (
                    f"+{(price-ma)/ma*100:.1f}%"
                    if ma and price > ma
                    else (f"{(price-ma)/ma*100:.1f}%" if ma else "N/A")
                )

            lines.append(
                f"EMA9: {ema9:.2f} ({_vs(ema9)})  EMA21: {ema21:.2f}  "
                f"EMA50: {ema50:.2f}  SMA200: {sma200:.2f} ({_vs(sma200)})"
            )

    # Signals
    tech = data.get("tech_signal")
    ml = data.get("ml_signal")
    if tech or ml:
        lines.append("\n<b>─ Сигналы стратегий ─</b>")
    if tech:
        lines.append(
            f"Технический: {_signal_arrow(tech['direction'], tech['confidence'])}  "
            f"(SL: -{tech['stop_pct']:.1f}%  TP: +{tech['target_pct']:.1f}%)"
        )
    if ml:
        ml_conf = ml.get("confidence", 0)
        ml_dir = ml.get("direction", "hold")
        if ml_conf == 0 and ml_dir == "hold":
            lines.append("ML (LightGBM): <i>Модель не обучена — запустите /retrain</i>")
        else:
            lines.append(f"ML (LightGBM): {_signal_arrow(ml_dir, ml_conf)}")
            if ml.get("features"):
                top_feats = sorted(ml["features"].items(), key=lambda x: abs(x[1]), reverse=True)[
                    :3
                ]
                feat_str = ", ".join(f"{k}={v:.3f}" for k, v in top_feats)
                lines.append(f"<i>Top features: {feat_str}</i>")
    elif ml is None:
        lines.append("ML (LightGBM): <i>Модель не обучена — запустите /retrain</i>")
    if not tech and not ml:
        lines.append("\n<i>Стратегии недоступны (модель не обучена)</i>")

    # Fundamentals
    fund = data.get("fundamentals", {})
    if fund:
        lines.append("\n<b>─ Фундаментал (TTM) ─</b>")
        if fund.get("market_cap"):
            mc = fund["market_cap"]
            mc_str = f"{mc/1e9:.1f}B ₽" if mc >= 1e9 else f"{mc/1e6:.0f}M ₽"
            lines.append(f"Капитализация: <b>{mc_str}</b>")
        if fund.get("pe"):
            lines.append(f"P/E: <b>{fund['pe']:.1f}</b>")
        if fund.get("pb"):
            lines.append(f"P/B: <b>{fund['pb']:.2f}</b>")
        if fund.get("ev_ebitda"):
            lines.append(f"EV/EBITDA: <b>{fund['ev_ebitda']:.1f}</b>")
        if fund.get("dividend_yield"):
            lines.append(f"Дивдоходность: <b>{fund['dividend_yield']:.1%}</b>")
        if fund.get("net_margin"):
            lines.append(f"Net margin: <b>{fund['net_margin']:.1%}</b>")
        if not any(fund.values()):
            lines.append("<i>Фундаментальные данные недоступны для этого тикера</i>")

    return "\n".join(lines)


def format_watchlist_with_direction(candidates: list[dict], direction: str) -> str:
    """Format screener results with direction label and RSI quality tags."""
    dir_label = "📈 LONG" if direction == "long" else "📉 SHORT"
    if not candidates:
        return f"<b>Watchlist ({dir_label})</b>\n\nКандидатов не найдено."

    lines = [
        f"<b>Top Candidates — {dir_label}</b>",
        f"<i>Score учитывает объём × ATR × моментум × RSI-таймирование × ADX</i>\n",
    ]
    for i, c in enumerate(candidates[:15], 1):
        custom_tag = " [custom]" if c.get("custom") else ""
        mom = c.get("momentum", 0)
        rsi = c.get("rsi_14", 0)
        adx = c.get("adx_14", 0)
        rsi_tag = c.get("rsi_tag", "")

        # Visual RSI quality marker
        if rsi_tag == "OB":
            rsi_warn = f" ⚠️RSI:{rsi:.0f}(OB)"
        elif rsi_tag == "OS":
            rsi_warn = f" ⚠️RSI:{rsi:.0f}(OS)"
        else:
            rsi_warn = f" RSI:{rsi:.0f}"

        adx_str = f" ADX:{adx:.0f}" if adx else ""

        kind_tag = " 🎯F" if c.get("kind") == "future" else ""
        lines.append(
            f"{i}. <b>{c['ticker']}</b>{kind_tag}{custom_tag} | Score: {c['score']:.1f}\n"
            f"   Vol/ч: {c['avg_volume_rub']/1e6:.1f}M ₽ | ATR: {c['atr_pct']:.1f}% | "
            f"Mom: {mom:+.1f}%{rsi_warn}{adx_str}"
        )
    return "\n".join(lines)


def format_custom_tickers(tickers: list[dict]) -> str:
    """Format custom watchlist."""
    if not tickers:
        return (
            "<b>Custom Watchlist</b>\n\n"
            "No custom tickers added yet.\n"
            "Use /addticker TICKER to add one."
        )
    lines = ["<b>Custom Watchlist</b>\n"]
    for t in tickers:
        name = f" — {t['name']}" if t.get("name") else ""
        lines.append(f"• <b>{t['ticker']}</b>{name}")
    return "\n".join(lines)
