# Phase 6 — Risk Overlays (Day/Swing Method Sweep)

_Date: 2026-05-12_

## User ask

"Explore day trading and swing trading methods we are not following to
**avoid losses**."

## Research output

Subagent surfaced 7 day-trading methods + 7 swing methods + 10
loss-avoidance techniques we don't currently use.

**Top 3 loss-avoidance from research:**
1. Volatility-scaled position sizing (ATR-based)
2. Chandelier exit (ATR×3 trailing stop)
3. Drawdown throttle (halve at -3%, pause at -5%)

Implemented + tested 4 overlays in `realistic_capital_sim_v2.py`:
- DD throttle
- Volatility-scaled sizing (risk-per-trade%)
- R:R floor pre-entry
- Consecutive-loss pause

## Backtest matrix (Phase 5 base config: selector top-10 + sector RS + calendar, ₹10L)

| Overlay | max=2 ROI / DD% / Win% | max=5 ROI / DD% / Win% |
|---------|------------------------|------------------------|
| **None (Phase 5)** | **+29.35 / 9.58 / 50.0** | +21.56 / 6.10 / 50.0 |
| DD throttle only | +10.59 / 6.19 / 71.4 | +12.40 / 5.74 / 70.6 |
| Vol sizing 1% risk | +7.76 / 6.19 / 70.8 | +11.99 / 4.95 / 73.8 |
| **Vol sizing 2% risk** | +14.89 / 7.50 / 70.8 | **+20.03 / 6.09 / 73.8** |
| Loss pause (3→5) | +29.35 / 9.58 / 70.8 | +18.29 / 5.74 / 74.4 |
| All combined | +6.84 / 6.69 / 70.8 | +7.30 / 5.57 / 70.6 |

## Key finding: Win-rate jumps to 70-74% with vol sizing

Phase 5 baseline = 50% win rate, 9.58% DD, but ROI driven by a few
big winners.

Vol-sized config: **70-74% win rate, 6-7% DD**. Smaller wins,
fewer big losses → much steadier curve. Better Sharpe.

## Two recommended production paths

### Path A — Maximum ROI (Phase 5 raw)
- **+29.35% ROI, 9.58% MaxDD, 50% win rate**
- Concentrated wins
- Lumpier P&L (volatile month-to-month)
- Use when: high risk tolerance, big-win mindset, OK with losing weeks

### Path B — Maximum Sharpe (vol-sized 2%, max=5)
- **+20.03% ROI, 6.09% MaxDD, 73.8% win rate**
- Smooth equity curve
- Better psychological feel (winning most days)
- Survives compounding longer
- Use when: low DD tolerance, steady-returns preference

**Subagent's expert opinion: Path B (lower DD, higher Sharpe) survives
multi-year compounding better than Path A.** Quote from research:

> "29.35% / 9.58% DD on Nifty 500 with 1-year backtest is already at
> the upper edge of honest expectations for systematic cash-equity in
> India. ... target a realistic 25-30% CAGR with 6-8% DD; that is a
> better Sharpe than 40% / 15% DD and survives compounding far longer."

## Day-trading methods we're NOT running (from research)

| Method | Backtestable now? | Promoter | Expected edge |
|--------|-------------------|----------|---------------|
| **CPR (Central Pivot Range)** | Yes (need daily bars) | Mitesh Patel / Subasish Pani | 55-60% win on narrow-CPR days |
| **VWAP bounce/reclaim** | Yes (5m bars) | Brian Shannon / Varsity | 0.4-0.8% per trade |
| **ORB-60 (first hour)** | Yes (need 5m) | Booming Bulls | 52-55% Nifty 100 |
| **Gap fade** | Yes (daily OHLC) | Subasish Pani | 58% fill rate |
| **Inside-day expansion** | Yes (daily) | Linda Raschke | 1:2 R:R structural |
| **Anchored VWAP from earnings/52wH** | Partial (need earnings dates) | Brian Shannon | High trend continuation |
| **Round-number breakouts** | Yes | Vivek Bajaj | Institutional iceberg zones |

## Swing methods we're NOT running

| Method | Backtestable | Promoter | Edge |
|--------|--------------|----------|------|
| **VCP (Minervini)** | Yes | Mark Minervini / Kanan Bahl India | Stage-2 selection refinement |
| **52w high + 2x volume** | Yes (have cache) | O'Neil / Marketsmith India | 6-8%/yr alpha |
| **Donchian 20/55** | Yes | Turtle Trader | 15-20% CAGR in trends |
| **Bollinger squeeze** | Yes | John Bollinger | Vol expansion timing |
| **Cup-and-handle** | Yes | O'Neil | Pattern-based |
| **MTF alignment (W+D)** | Yes | Stan Weinstein | Stage analysis filter |
| **Coffee Can quality** | Partial (need fundamentals) | Saurabh Mukherjea | Long-hold sleeve |

## Hard truth from research

> "Pushing the headline to 40-50% is feasible by stacking strategies
> and tightening filters, but realized live returns will be 22-28%
> after slippage, brokerage (~0.1% RT), STT, GST and regime breaks —
> the gap between backtest and live is consistently 30-40% in Indian
> cash equity."

So whether we choose Path A (+29%/9.6% DD) or Path B (+20%/6% DD),
live numbers will likely be **20-25%/yr after frictions**, not 30%
or 40%.

The 60-120%/yr (5-10%/mo) target remains structurally unreachable
on cash equity ₹10L single-strategy.

## Recommended next steps (in priority order)

1. **Decision needed: Path A or Path B?** (or hybrid: run both in parallel paper trades)
2. **Add Chandelier exit (ATR×3 trailing)** — protects winners, well-documented
3. **Add MTF filter (weekly trend = long-only above 30-week EMA)** — one extra filter
4. **Multi-year backtest** (2024+2023) to validate selector+sector+overlay stack
5. Build live-only scanners (delivery%, bulk deals, F&O ban, FII flows) for paper-trade phase
6. Build VCP + Donchian as additional strategies — only if Phase 6 paper-trade validates

## Files

| Path | Purpose |
|---|---|
| `tools/backtests/realistic_capital_sim_v2.py` | New cap-sim with risk overlays |
| `tools/backtests/sector_rs.py` | Phase 5 |
| `tools/backtests/apply_sector_filter.py` | Phase 5 |
| `tools/backtests/apply_calendar_filter.py` | Phase 5 |
| `exports/backtests/PHASE5_INDIAN_PATTERNS.md` | Phase 5 doc |
| `exports/backtests/PHASE6_RISK_OVERLAYS.md` | This file |
