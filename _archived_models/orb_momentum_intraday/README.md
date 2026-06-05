# orb_momentum_intraday — Morning ORB on Momentum Leaders

**The only intraday / day-trading model in the book.** Every other model is a
multi-day swing rotation; this one is in-and-out the same day, **flat every
night (0 overnight risk)**.

## How it works

Each trading day:

1. **Select (at the open).** Rank the **Nifty 500** (point-in-time membership)
   by trailing **20-day return** and take the **top 3 momentum leaders**.
   *Why:* momentum stocks trend, so their intraday breakouts continue. Raw ORB
   on random names just whipsaws (−13% in tests) — the momentum filter is the edge.

2. **Opening range.** For each leader, the first **15 minutes** (3 × 5-min bars,
   09:15–09:30) set the range: `ORH` = high, `ORL` = low, `width = ORH − ORL`.

3. **Entry (long only).** Buy when price breaks **above ORH** — but **only if the
   breakout fires before 10:00**. Late breakouts are skipped (less room to run;
   morning-only nearly doubled the return). We never short — on up-momentum names
   the downside break mostly fails.

4. **Stop / target.** Stop at `ORL`; target at `ORH + 2 × width`.

5. **Exit.** Whichever first: stop, target, or **forced square-off at 15:25**.
   ~58% of trades ride the trend to the EOD-flat close, ~21% stop, ~20% target.

Equal-weight across the leaders that actually break out (~32% never cross ORH
before the cutoff, so they don't trade that day).

## Timing in one line
Scan the opening range 09:15–09:30 → buy momentum-leader breakouts **~09:30–10:00**
→ hold ~3.5h → **flat at 15:25**. ~1.6 trades/day.

## Performance (2025-03 → 2026-05, 231 days, realistic 0.15% slippage + 0.15% cost)
| metric | value |
|---|---|
| Total return | **+216%** |
| Annualized | **+251%** |
| Max drawdown | 17.2% |
| Sharpe | ~3.44 |
| Trades | 377 (~1.6/day), WR 53% |
| Per-trade avg | +0.54% |
| Months green | 13 / 15 (worst Feb-26 −11.9%) |

## Slippage sensitivity (THE caveat)
| fill assumption | annualized |
|---|---|
| optimistic (slip 0) | +500%+ (fantasy — ignore) |
| **realistic (slip 0.15%)** | **+251%** ← headline |
| conservative (slip 0.25%) | ~+46–90% |

The real number is decided by live fills. **Paper-trade before trusting the
magnitude.** Defensible claim: *positive-expectancy intraday momentum-breakout
edge*; exact CAGR pending live measurement.

## Honest limits
- Validated on **one bull regime** (15 months). Feb-26 shows it bleeds in chop;
  **no 2022-type bear tested intraday**.
- Bar-level fill model (5-min); tick reality differs.
- Assumes full capital rotates into the top-3 daily (liquidity caps size).

## Files
- `strategy.py` — shared core (params + `rank_momentum` + `orb_trade`). Single
  source of truth; backtest and live import it so they can't drift.
- `data.py` — 5-min bar layer (Fyers `resolution="5"`, cached to `cache5min/`).
- `backtest.py` — `python -m tools.models.orb_momentum_intraday.backtest --from … --to … --out …`
- `live_signal.py` — intraday signal generator (selection + breakout detection).

## Data
5-min bars are **not** in Postgres (DB is daily-only) — pulled from Fyers and
cached as pickles under `cache5min/` (gitignored). Backtest reads the cache;
`data.get_5min()` fetches+caches on demand. Override cache dir with `ORB5MIN_CACHE`.

## Live wiring (NOT yet scheduled)
Going live needs an **intraday scheduler** firing every 5 min from 09:30–10:00 to
detect breakouts on that day's top-3, an executor placing the long + bracket
(stop/target), and a 15:25 force-flat. `live_signal.py` provides the signal
logic; the cron + executor wiring is a separate deploy step. **Currently
backtest-only.**
