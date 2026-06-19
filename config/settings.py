from typing import Literal
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # --- Secrets ---
    T_INVEST_TOKEN: str
    T_INVEST_ACCOUNT_ID: str
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: int = 0  # Set on first /start

    # --- Trading mode ---
    # autonomous  = bot trades by itself
    # advisory    = bot signals, user approves/rejects each trade via Telegram
    # interactive = manual only, no autonomous scanning
    MODE: Literal["autonomous", "advisory", "interactive"] = "interactive"

    # --- Active profile ---
    ACTIVE_PROFILE: str = "moderate"

    # --- Position limits ---
    MAX_POSITIONS: int = 5
    MAX_POSITION_PCT: float = 0.20  # Max 20% of portfolio per position
    # 3 % per-trade risk (was 2 %).  With a 45 % accuracy model, R/R ≈ 1.5
    # (stop 2 ATR, target 3 ATR), expected-value-positive portfolio lives
    # at ~2-4 % risk per bet.  Bumped so the sizer isn't gating out big
    # signals on thinly-priced tickers (the stop distance × lots × price
    # check was binding to 1 lot when portfolio × 2 % < 2 ATR × lot_price).
    MAX_PORTFOLIO_RISK_PCT: float = 0.03

    # --- Volatility targeting ---
    # Frazzini & Pedersen 2014 ("Betting Against Beta"), Moskowitz et al.
    # 2012 ("Time Series Momentum") — the canonical risk-parity / vol-target
    # pattern: scale lots so each position contributes the same expected
    # volatility, regardless of instrument vol regime.  We cap Kelly-implied
    # lots by vol-target lots; whichever is smaller wins.  Target is the
    # daily P&L volatility we're willing to take per position, in % of
    # portfolio.  0.5 % per position × 5 positions × √252 ≈ 18 % annual
    # portfolio vol — moderate.
    VOL_TARGET_ENABLED: bool = True
    VOL_TARGET_DAILY_PCT: float = 0.5  # target 0.5% daily P&L vol per position
    VOL_TARGET_MIN_DAILY_VOL: float = 0.005  # floor to avoid div-by-zero on quiet names

    # --- Kelly criterion ---
    # Half-Kelly (0.5).  Third-Kelly (0.33) still left most of the account
    # idle on small portfolios (observed live: 10 k RUB untouched) because
    # Kelly × confidence × regime_scale ≈ 0.33 × 0.65 × 1.0 = 2.1 % and a
    # single 1 500-RUB lot on a 50k portfolio was already ~3 % — so the
    # sizer rounded to 1 lot.  Half-Kelly is still well below the Thorp
    # optimal (full Kelly) but gives enough room for 2-3 lots on liquid
    # tickers when the signal is strong.
    # postmortem 2026-04-25: realized f*=-0.23 (negative edge).  Cut to 0.15
    # (third of prior 0.5) to limit sizing while strategy is recalibrated.
    # Restore to 0.25 once win_rate >= 45% and f* > 0 over 50+ trades.
    # 2026-05-14: bumped back to 0.25 — sells now have win_rate=49% f*≈+0.15
    # (n=55 closed), and 0.15 was rounding 90%+ of intended sizes to a single
    # lot (vol-target cap + concentration cap dominated).  Combined with the
    # wider ATR stops below this restores a meaningful position size on
    # high-conviction trades without unleashing leverage.
    KELLY_FRACTION: float = 0.25
    # Floor applied inside PositionSizer.kelly_fraction() when Kelly math
    # returns 0 (cold-start / no win-rate data).  Reduced proportionally.
    KELLY_FALLBACK_PCT: float = 0.03

    # --- Risk circuit breakers ---
    MAX_DAILY_LOSS_PCT: float = 0.03  # -3% daily loss -> stop trading
    MAX_DRAWDOWN_PCT: float = 0.10  # -10% from peak -> stop trading

    # --- Glosten-Milgrom spread threshold ---
    SPREAD_THRESHOLD: float = 2.0  # Skip if spread > 2x median

    # --- ML parameters ---
    ML_LOOKBACK_DAYS: int = 180
    ML_MIN_SAMPLES: int = 500
    ML_RETRAIN_HOURS: int = 24
    # postmortem 2026-04-25: confidence Q4 (0.83-0.92) had WORST mean pnl
    # (-0.55%) while Q1 (0.60-0.68) = -0.16%.  Spearman(conf,pnl)=-0.08.
    # Raised to 0.72 to cut the saturated-confidence noise band.
    # 2026-05-14 strategy_bakeoff (10 MOEX × 6mo): live confidence bucket
    # analysis showed conf 0.7 (n=50, sum −142 ₽) is the worst, conf 0.8+
    # is +112 ₽.  But the 0.72 cutoff also gates ML through TechnicalStrategy
    # below 0.72 — leaving ≤45 trades over 6 months across 10 tickers.
    # Lowering 0.72 → 0.65 increases activity ~3× while still excluding
    # the noisiest 0.5–0.65 band.  Combined with widened ATR stops it gives
    # mean-reversion exits more room to work.
    SIGNAL_THRESHOLD: float = 0.65
    # Meta-labelling (LdP ch.3) — secondary binary classifier predicts whether
    # the primary's directional pick is correct.  When the bundled meta model
    # is loaded, MLStrategy gates trades on meta_conf >= META_LABEL_THRESHOLD
    # AND scales reported confidence to (primary_conf × meta_conf)**0.5.
    # primary_min_conf is the floor BELOW which the primary is treated as
    # "hold" before reaching meta — keeps the meta dataset clean of rows the
    # primary was barely confident on.
    META_LABEL_THRESHOLD: float = 0.55
    META_PRIMARY_MIN_CONF: float = 0.5
    # ATR filter: skip signals when current ATR% is below this fraction of the
    # instrument's median ATR% over the lookback window.  atr_pct is the #1
    # permutation-importance feature (0.158) — in low-vol regimes the model's
    # directional signal is noise.  1.0 = require at least median ATR (any
    # above-average volatility day).  0.0 = disabled.
    ATR_FILTER_MEDIAN_MULT: float = 0.8  # require ≥ 80 % of median ATR%
    # Day-of-week filter: skip NEW entries on these weekdays (0=Mon, 4=Fri).
    # day_of_week is the #2 permutation-importance feature (0.052) — Mondays
    # carry gap risk from weekend news; Fridays have pre-weekend risk-off.
    # Empty list = disabled.
    # 2026-04-27 fix: was [0, 4] — combined with weekend closures + hour filter
    # this blocked ALL trading from Fri-Mon, producing a 3-day signal gap with
    # zero entries.  Postmortem evidence only supported HOUR filter (not weekday);
    # weekday filter was speculative.  Disabled until confirmed harmful with
    # ≥150 trades of evidence per weekday.
    SKIP_ENTRY_WEEKDAYS: list = []  # was [0, 4] — disabled, see note
    # Hour-of-day filter (MSK = UTC+3): skip new entries during these hours.
    # postmortem 2026-04-25: hours 7-9 MSK average -0.46 to -0.81% (pre-market
    # illiquidity), 11-12 MSK average -0.68 to -1.08% (mid-morning reversal).
    # Best hours: 10 MSK (+0.34%), 19-23 MSK (+0.66 to +0.99%).
    # 2026-04-27 fix: trimmed from [7,8,9,11,12] to [7,8,12] — keep only
    # the worst-performing hours (mean ≤ -0.46% with n≥4 trades).
    # 2026-04-30 postmortem (n=57, last 7 days): worst hours shifted to
    # [17, 10, 12, 18] (means -0.77/-0.71/-1.47/-0.45, n=7/7/2/3).  Hours
    # 7 and 8 had n=6/n=1 with insufficient evidence to keep them blocked.
    # New filter [10, 12, 17, 18]: covers the four worst on-evidence hours.
    # Empty list = disabled (no filter).
    # 2026-05-04: disabled by user — small per-hour samples (n=1..8 over a
    # week) don't justify hard blocks; previous "bad hours" likely correlation
    # not causation.  Re-enable only with ≥30 trades/hour confirmation.
    SKIP_ENTRY_HOURS_MSK: list = []

    # --- Scheduling ---
    SCAN_INTERVAL_MINUTES: int = 30
    PORTFOLIO_CHECK_MINUTES: int = 5

    # --- Order execution ---
    # market           = instant market order (worst slippage, guaranteed fill)
    # limit_aggressive = limit at best ask/bid (crosses spread, fast fill, less slippage)
    # limit_passive    = limit at best bid/ask (joins queue, saves spread, may not fill)
    ORDER_EXECUTION_MODE: Literal["market", "limit_aggressive", "limit_passive"] = (
        "limit_aggressive"
    )
    LIMIT_ORDER_TIMEOUT: int = 45  # seconds to wait for limit fill before cancelling
    LIMIT_ORDER_RETRY_MARKET: bool = True  # fallback to market if limit not filled in time

    # --- Broker commission (Tinkoff "Trader" tariff) ---
    # All values are fractions: 0.0005 = 0.05 %
    COMMISSION_SHARES_PCT: float = 0.0005  # shares, bonds, funds
    COMMISSION_FUTURES_PCT: float = 0.0004  # futures
    COMMISSION_CURRENCY_PCT: float = 0.005  # currency instruments
    COMMISSION_METALS_PCT: float = 0.015  # precious metals

    # --- Anti-whipsaw / premature-exit protection ---
    # When a position is closed, the same ticker cannot be re-entered for
    # this many minutes.  Prevents the bot from flipping direction right
    # after eating a round-trip commission on a losing exit.
    SAME_TICKER_COOLDOWN_MINUTES: int = 60
    # A strategy's signal_reversal exit is only honoured when the position's
    # unrealised P&L (as a fraction) is at least this multiple of the
    # round-trip commission.  Lowered 2.0 → 1.0 (postmortem 2026-04-25):
    # signal_reversal mean +0.24% (best category) — let the model exit sooner
    # once commission is cleared, rather than waiting for 2× coverage.
    # Stop-loss, take-profit, trailing-stop and time-exit are unaffected.
    MIN_EXIT_PROFIT_MULT_COMMISSION: float = 1.0
    # Minimum hold time before signal_reversal may fire.
    # 2026-05-16 post-mortem on TTM6 (long opened 12:07, reversed 12:12,
    # −318 RUB on a futures contract): the engine ran `should_exit` every
    # PORTFOLIO_CHECK_MINUTES tick and immediately re-scored the SAME
    # hourly bar that triggered entry, producing rapid whipsaws.
    # Hold-time bucket analysis of 86 historical signal_reversal exits:
    #   <5 min     n=3  sum=−10.7
    #   5−30 min   n=14 sum=−299.8  ← WHIPSAW BUCKET (TTM6 was here)
    #   30−120 min n=13 sum=+20.6
    #   2−4 h      n=11 sum=+87.5  win=91%
    #   4−12 h     n=23 sum=+98.8
    #   12−48 h    n=30 sum=−44.0
    # Setting this to 60 min (one full hourly bar) makes the reversal
    # signal honour at least one fresh feature update before flipping.
    MIN_HOLD_MINUTES_BEFORE_REVERSAL: int = 60
    # Winner-progress gate.  Reversal signals ON WINNERS are only accepted
    # once the position has covered this much of entry→target.  Losers
    # bypass this gate entirely (cut early on reversal confirmation).
    # 2026-05-14: disabled — strategy_bakeoff over 10 MOEX × 6mo shows that
    # the ONLY profitable strategy (S3e RSI mean-rev no-stops, +3.60 %
    # across 1123 trades, 60.2 % win-rate) exits as soon as the signal
    # flips, with NO progress requirement.  Live data echoes this:
    # signal_reversal exits make +150 ₽ across 86 trades (the only positive
    # exit category), while target_distance is reached on only 2/140 trades.
    # Keeping the gate at 0.35 was forcing winners to hold past the model's
    # reversal call, often giving back the gains before any exit fired.
    # MIN_EXIT_PROFIT_MULT_COMMISSION still enforces that |P&L| > 1 ×
    # round-trip commission, so we never close on pure noise.
    MIN_TARGET_PROGRESS_FRAC: float = 0.0
    # Time-exit floor: don't treat a position as "stalled" unless |P&L| <
    # this fraction AND it has held > 5 days.  Raised from old 0.3 % so we
    # don't close stalled positions for a loss on commission.
    TIME_EXIT_MAX_PNL_PCT: float = 0.3  # absolute %, read directly

    # --- Partial take-profit (MFE-aware exit) ---
    # postmortem 2026-04-30: 14 of 30 losers (47%) touched ≥+0.5% MFE before
    # turning back into losers.  A partial scale-out at +1.5×ATR locks in
    # profit on those trades while leaving runners intact.  Industry pattern
    # (Brunnermeier/Pedersen 2009 "Funding Liquidity"; Clenow ch.15) — half
    # off at first decisive move, trail the rest.
    PARTIAL_TP_ENABLED: bool = True
    PARTIAL_TP_TRIGGER_ATR: float = 1.5  # close PARTIAL_TP_FRAC at MFE >= this × ATR
    PARTIAL_TP_FRAC: float = 0.5  # fraction of lots to close at trigger
    PARTIAL_TP_MIN_LOTS: int = 2  # don't try to scale out positions of 1 lot

    # --- ATR-based stop / target multipliers ---
    # Previously hardcoded as 2.0 × ATR (stop) and 3.0 × ATR (target) in 4
    # places (position_sizer, ml_strategy, technical_strategy, risk.manager).
    # 2026-05-14 strategy_bakeoff finding: tight ATR stops are the single
    # biggest profitability killer for mean-reversion entries.
    #   S3e (no stops)        n=1123  total=+3.60 %  Sharpe=+0.26
    #   S3  (2 ATR / 3 ATR)   n=1870  total=−89.08 % Sharpe=−0.72
    #   S3f (1 ATR / 1 ATR)   n=3853  total=−249 %  Sharpe=−2.89
    # Removing stops entirely is operationally risky (engine death = no
    # safety net), so we widen to 4 × ATR as a catastrophe-only floor and
    # shorten target to 2 × ATR so signal_reversal can book wins before
    # giving them back.  Combined with MIN_TARGET_PROGRESS_FRAC=0 above,
    # this lets the model's reversal call drive exits 95 % of the time.
    STOP_ATR_MULT: float = 4.0  # was hardcoded 2.0
    TARGET_ATR_MULT: float = 2.0  # was hardcoded 3.0

    # --- Trailing stop ---
    # Activates once the position has moved this fraction of the way to
    # the take-profit target.
    #
    # IMPORTANT — previous 0.50 / 1.5 × ATR combination closed winners at
    # break-even (entry=100, target=106, ATR=2 → activation at 103, trail =
    # peak − 3 = 100 = entry).  Root cause of the asymmetric P&L the user
    # observed: losers rode to full SL (no progress gate blocks them),
    # winners got stopped on break-even.
    #
    # New config: activate at 70 % (peak safely above entry), trail by
    # 1.0 × ATR.  With the same numbers: activation 104.2, trail =
    # peak − 2 = 102.2 → still captures +2.2 % even on the tightest retrace.
    #
    # Literature: Clenow "Stocks on the Move" (2015) ch.12 recommends
    # trailing ≥ 70 % of target with 0.75-1.0 × ATR stop for swing trades.
    TRAILING_STOP_ENABLED: bool = True
    TRAILING_STOP_ACTIVATION_FRAC: float = 0.70  # activate at 70 % of target
    TRAILING_STOP_ATR_MULT: float = 1.0  # trail distance = 1.0 × ATR

    # --- StoplossGuard (freqtrade-pattern) ---
    # After STOP_GUARD_COUNT closed losers (stop_loss OR signal_reversal with
    # net loss) within STOP_GUARD_LOOKBACK_HOURS on the SAME side (buy/sell),
    # pause that side for STOP_GUARD_PAUSE_HOURS.  Mirrors freqtrade's
    # `StoplossGuard` with `only_per_side=True`.  Cuts the cluster-of-stops
    # losing pattern that postmortem 2026-04-30 shows: WUSH 3× losses,
    # GTRK 2× losses on consecutive entries.
    STOP_GUARD_ENABLED: bool = True
    STOP_GUARD_COUNT: int = 3
    STOP_GUARD_LOOKBACK_HOURS: int = 4
    STOP_GUARD_PAUSE_HOURS: int = 4

    # --- Long-side circuit breaker ---
    # postmortem 2026-04-30 (n=29 buy trades): win_rate 31%, mean P&L -0.83%,
    # realized Kelly f* = -0.62 on the long side while shorts had f* = +0.57.
    # When per-direction realized Kelly is negative AND we have ≥20 same-
    # direction closed trades, refuse new long entries unless confidence is
    # exceptionally high.  Mirrors the StoplossGuard pattern from freqtrade
    # but specialised for direction asymmetry rather than recent stop count.
    # 2026-05-14: disabled — strategy_bakeoff over 10 MOEX × 6mo hourly bars
    # shows the long/short asymmetry is regime-specific, not structural
    # (S3c long-only −4.23%, S3d short-only −4.68% — within 0.5σ).
    # The 50-buy live Kelly was dominated by 2 bad April days.  Re-enable
    # only with ≥100 closed long trades AND f* < −0.20 sustained 30+ days.
    LONG_AUTO_PAUSE: bool = False  # set True to gate longs by realised long-Kelly
    LONG_MIN_CONFIDENCE_WHEN_BAD_EDGE: float = 0.85  # only override pause if confidence ≥ this
    LONG_PAUSE_MIN_HISTORY: int = 20  # need ≥20 long closed trades before applying

    # --- Short selling ---
    # postmortem 2026-04-30: shorts had Kelly f* = +0.57 (real edge) while
    # longs had f* = -0.62.  ALLOW_SHORTS flipped to True so the screener's
    # short pass actually places trades on shares (not just futures).  The
    # tighter SHORT_MIN_CONFIDENCE + SHORT_POSITION_SCALE + carry-free cap
    # already gate this side conservatively.  Requires margin account with
    # shorting enabled at the broker — when the instrument's
    # short_enabled_flag is False the risk manager rejects the trade.
    ALLOW_SHORTS: bool = True  # requires margin account with shorting enabled
    # When ALLOW_SHORTS is on, the autonomous screener adds a second pass
    # ranked by NEGATIVE momentum + short_enabled_flag.  Shorts carry more
    # tail risk than longs (unlimited loss, forced closes around corp events,
    # broker can recall borrow), so we gate them tighter than longs:
    SHORT_MIN_CONFIDENCE: float = 0.70  # higher than general SIGNAL_THRESHOLD
    # Shorts have been outperforming longs in live logs (SMLT, TGKN closed
    # profitable while longs chopped), so the 0.5 handicap was leaving money
    # on the table.  0.75 keeps *some* extra caution around short-specific
    # tail risks (forced buy-in, corp events) without over-penalising the
    # direction that actually worked.
    SHORT_POSITION_SCALE: float = 0.75  # shorts sized at 75 % of Kelly-recommended
    # Tinkoff charges overnight carry on short positions whose notional value
    # crosses a tiered boundary (first tier ≈ 5000 RUB).  We cap short lots so
    # the position stays under this threshold, making carry = 0.  Set to 0 to
    # disable the cap (then shorts are sized purely by Kelly × scale).
    SHORT_CARRY_FREE_THRESHOLD_RUB: float = 5000.0
    # Don't open NEW shorts after this hour MSK — overnight carry kicks in at
    # end-of-day clearing, so opening a short at 22:30 means paying carry on
    # a position you barely held.  Existing shorts are unaffected and close
    # on their normal exit conditions (stop/TP/trailing/signal).
    SHORT_ENTRY_CUTOFF_HOUR_MSK: int = 20

    # --- Leverage ---
    USE_LEVERAGE: bool = False
    MAX_LEVERAGE: float = 2.0

    # --- Futures ---
    # Futures have higher intraday volatility, no overnight short carry, and
    # built-in leverage via ГО (initial margin).  We gate them behind a flag
    # because they require a different position-sizer path (contract size +
    # minPriceIncrement in currency, not % of notional).
    INCLUDE_FUTURES: bool = True  # screener scans futures alongside shares
    ALLOW_FUTURES_SHORT: bool = True  # futures shorts are free (no carry), default on
    FUTURES_MIN_DAYS_TO_EXPIRY: int = 14  # skip near-expiry (roll-over risk)
    FUTURES_MAX_POSITION_PCT: float = (
        0.20  # ГО per lot can reach 12-15% of small portfolio; 20% allows 1 lot
    )
    # Per-trade risk limit for futures (separate from shares).  FORTS contracts
    # have large minimum lot notional, so 1% (share default) blocks all trades
    # on a small account.  2% allows 1 lot of BRM6/GZM6 at 15k RUB portfolio.
    FUTURES_MAX_PORTFOLIO_RISK_PCT: float = 0.02
    # Separate confidence floor for futures.  Futures have higher intrabar
    # volatility and a shorter triple-barrier horizon (10 bars vs 20), so
    # signal-to-noise is systematically lower than on shares.  Raise the
    # floor to filter out the noisier tail; can be lowered once we have
    # asset_class-aware calibration + enough futures history.
    # Futures: kept lower than shares threshold because (a) ML model has no
    # futures training data yet and produces ~0.33 confidence per class, so
    # TechnicalStrategy drives futures signals alone; (b) the postmortem 0.75
    # raise was based on share data — no futures data to justify it.
    # Applied at the voting level (engine.py _evaluate_instrument) so that
    # TechnicalStrategy signals can clear the bar for futures instruments.
    SIGNAL_THRESHOLD_FUTURES: float = 0.65
    # Roll-window half-width (days around expiry to tag as "in roll").  Used
    # by features.py → in_roll_window flag.  Stocks don't roll, so the
    # feature is always 0 for shares.
    FUTURES_ROLL_WINDOW_DAYS: int = 7

    # --- Triple-barrier label horizon ---
    # Selected by IC (Spearman rank correlation) grid search in
    # scripts/tune_model.py --tune-horizon, NOT by F1_weighted (which is
    # inflated at short horizons by hold-class dominance: at h=1, hold%=93%,
    # F1_weighted=0.91 but IC=0.07 — nearly zero directional signal).
    #
    # Grid results (IC ranking, same dataset):
    #   h=30: IC=0.160  h=20: IC=0.158  h=14: IC=0.154
    #   h=10: IC=0.150  h= 7: IC=0.151  h= 5: IC=0.138  h=1: IC=0.068
    #
    # h=20 chosen: IC ≈ max (0.158 vs 0.160 at h=30), hold%=24% (balanced
    # three-class problem), ≈2 trading days — matches Fischer & Krauss (2018)
    # "5-10 day on daily" window converted to hourly.
    # Futures = 10 bars (= 20 × 0.5): mean-reverts faster than shares.
    TB_MAX_HOLD: int = 20
    TB_MAX_HOLD_FUTURES: int = 10

    # --- Paths ---
    DB_PATH: Path = Path("data/trade_bot.db")
    MODELS_DIR: Path = Path("data/models")
    LOGS_DIR: Path = Path("data/logs")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
