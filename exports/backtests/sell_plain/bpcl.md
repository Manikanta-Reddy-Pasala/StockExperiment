# BPCL (BPCL)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2025-11-10 09:15:00 → 2026-05-07 15:15:00 (847 bars)
- **Last close:** 308.20
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 3 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 1 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 0 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 0 / 0
- **Target hits / Stop hits / Partials:** 0 / 0 / 0
- **Avg / median % per leg:** 0.00% / 0.00%
- **Sum % (uncompounded):** 0.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 14:15:00 | 330.75 | 368.04 | 368.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-10 09:15:00 | 328.00 | 367.27 | 367.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 14:15:00 | 310.40 | 307.35 | 326.77 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-04-24 09:15:00 | 301.90 | 309.29 | 324.00 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-24 11:15:00 | 307.15 | 309.23 | 323.83 | ENTRY1 sustain failed after 120m |
| Cross detected — sustain check pending | 2026-04-29 14:15:00 | 303.75 | 309.20 | 322.17 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 09:15:00 | 297.60 | 309.04 | 321.96 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 1140m) |
| Cross detected — sustain check pending | 2026-05-07 11:15:00 | 305.65 | 307.49 | 319.31 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-05-07 12:15:00 | 308.45 | 307.50 | 319.25 | ENTRY1 sustain failed after 60m |

