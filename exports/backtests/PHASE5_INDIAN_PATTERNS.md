# Phase 5 — Indian Market Patterns Added

_Date: 2026-05-12_

## New Winner: Sector RS + Calendar = +29.35% ROI

```
ema_200_400 + selector top-10 N500 + sector RS bottom-block + calendar filter + max=2 + ₹10L
```

- **ROI: +29.35%** (₹12.94L from ₹10L)
- **MaxDD: 9.58%** (~₹96K)
- **Trades: 24** (5 fewer than Phase 4 baseline)
- **Profit: ₹2,93,474**

vs Phase 4 baseline (no sector filter): +21.85% ROI / 9.58% DD / 28 trades

**+7.5pp improvement** with no DD penalty.

## Pattern Sweep Results

| Config | ROI% | DD% | Trades |
|--------|-----:|----:|-------:|
| **Selector top-10 + sector + calendar + max=2** | **+29.35** | **9.58** | 24 |
| Selector top-10 + sector only + max=2 | +29.20 | 9.57 | 24 |
| Selector top-10 (Phase 4 baseline) + max=2 | +21.85 | 9.58 | 28 |
| Selector + sector + max=3 | +19.69 | 7.43 | 33 |
| Selector + sector + cal + max=3 | +23.83 | 7.43 | 33 |
| Calendar only + max=2 | (worse, 4 winning trades blocked) | - | - |
| Sector require-top + max=2 | +1.46 (too restrictive) | - | 5 |

## Sector RS Findings (May 2025 → May 2026)

Sector leadership over 249 trading days:

**% time in TOP-2:**
- METAL: **67%** (dominant leader)
- AUTO: 33%
- PSE: 18%, REALTY: 18%, BANK: 17%, IT: 17%, ENERGY: 17%
- PHARMA: 6%, INFRA: 1%, FMCG: 1%

**% time in BOTTOM-2:**
- IT: **58%** (frequent laggard — despite individual IT stocks doing well)
- REALTY: 54%
- FMCG: 38%
- PHARMA: 17%

Note: NSE NIFTYIT was a laggard 58% of the year despite HCLTECH being
a top contributor — large IT names lagged while specific stocks
outperformed. RS filter still adds value because it catches the few
times entries fired in genuinely weak sectors.

## What got blocked

- AEGISLOG (ENERGY): 4 entries blocked
- SAILIFE (PHARMA): 4
- IKS (PHARMA): 3

These were entries fired when their sector was in bottom-2. Block-bottom
gate dropped exactly 11 (sector) + 2 (calendar) = 13 entries net.

## All Indian Market Patterns from Research

Patterns we tested:
1. ✅ **Sector RS** (rolling 60d return vs Nifty 50) — IMPLEMENTED, +7.5pp boost
2. ✅ **Calendar (expiry + budget)** — IMPLEMENTED, marginal gain combined
3. ❌ Regime gate (VIX / Nifty trend) — TESTED, HURTS (Phase 3)

Patterns we can't backtest (need historical archives), but WILL build
as live scanners for paper-trade phase:
4. **Delivery % surge** (NSE bhavcopy MTO column) — single CSV/day, free, expected 3-6% edge
5. **Bulk/block deal following** (NSE API + smart-money whitelist) — expected 8-15% edge per signal
6. **F&O ban + MWPL squeeze** (NSE derivatives snapshot) — expected 4-7% per trade, mean-reversion strategy
7. **FII flow regime** (NSE fiidiiTradeReact API) — size throttle overlay
8. **Earnings drift (PEAD India)** — screener.in scrape — expected 6-12% per holding

## Compounding scenario

If all live patterns deliver mid-range edge:
- Sector RS: ✅ +7.5pp confirmed
- Delivery %: +3pp (live overlay)
- Bulk deals: +5pp (alpha when smart money signal aligns)
- F&O ban: 0pp (separate mean-reversion strategy, not stacked)
- FII regime: +2pp via DD reduction → enables higher size
- Earnings drift: +3pp (selective use during Q1/Q4 windows)
- Calendar: ~0pp (already in)

Stacked theoretical ceiling: **~40-45%/yr at 10-15% DD**

Real-world claw-back factors:
- Slippage at scale: -2pp
- Tax + brokerage: -3pp (LTCG 12.5% > 1yr, STCG 20%, STT, etc.)
- Behavioral drift (skipping signals, override): -3pp
- Pattern decay over time: -3pp

**Honest realistic forward: 25-35%/yr at 10-15% DD.** Still well above
Nifty 50 (~12%/yr) and matches top-decile mutual fund / PMS performance.

The 5-10%/mo (60-120%/yr) headline target STILL not reachable on
cash equity. But 25-30%/yr is documented-achievable territory.

## Updated Production Config

```yaml
strategy:        ema_200_400
universe:        selector_top10        # monthly refresh
sector_filter:   block_bottom_2        # NEW
calendar_filter: expiry+budget         # NEW
capital_inr:     1_000_000             # ₹10L
max_concurrent:  2
slot_alloc:      500_000
min_price:       50
min_adv_lakh:    100
kill_switch:     -5%
```

## Live-mode pattern scanners to build next

1. `tools/live/sector_rs_live.py` — daily RS computation, updates today's allowed sectors
2. `tools/live/delivery_pct_scanner.py` — fetches NSE bhavcopy MTO, flags >60% delivery
3. `tools/live/bulk_deals_scanner.py` — fetches bulk-deal API, matches against smart-money whitelist
4. `tools/live/fno_ban_scanner.py` — fetches MWPL ban list, flags ban-entry/exit candidates
5. `tools/live/fii_flow_regime.py` — daily FII net buy/sell, sets size multiplier
6. `tools/live/earnings_calendar.py` — scrapes screener.in for results dates

These are live-only (no historical archives in cache). Build for paper-trade phase.

## Sources

- Research: Zerodha Varsity Modules 6-9, Capitalmind, Marcellus, Marketsmith India
- Data: NSE bhavcopy archives, Fyers Indian sectoral indices, screener.in (scrape)
