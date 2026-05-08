# TMPV (TMPV)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 355.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 2 |
| ALERT3 | 2 |
| PENDING | 4 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 0 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 0 (incl. partial bookings)
- **Trades open at end:** 0
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

### Cycle 1 — BUY (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 13:15:00 | 433.79 | 417.61 | 417.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 434.64 | 419.05 | 418.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 422.03 | 422.51 | 420.30 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-09-23 09:15:00 | 425.45 | 422.53 | 420.33 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-23 10:15:00 | 423.03 | 422.54 | 420.34 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-23 15:15:00 | 425.58 | 422.64 | 420.45 | ENTRY1 cross detected — sustain check pending (15m) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 419.73 | 422.61 | 420.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 419.73 | 422.61 | 420.45 | EMA400 retest candle locked (from upside) |
| CROSSOVER_SKIP | 2025-09-30 15:15:00 | 413.67 | 418.62 | 418.63 | min_gap filter: gap=0.002% < 0.010% |
| TREND_RESET | 2025-09-30 15:15:00 | 413.67 | 418.62 | 418.63 | EMA inversion without crossover edge (EMA200=418.62 EMA400=418.63) — end cycle |
| CROSSOVER_SKIP | 2025-10-01 09:15:00 | 420.21 | 418.64 | 418.64 | min_gap filter: gap=0.000% < 0.010% |
| CROSSOVER_SKIP | 2025-10-13 13:15:00 | 402.36 | 419.02 | 419.03 | min_gap filter: gap=0.003% < 0.010% |
| CROSSOVER_SKIP | 2026-02-19 09:15:00 | 384.50 | 367.53 | 367.51 | min_gap filter: gap=0.006% < 0.010% |

### Cycle 2 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 331.95 | 368.39 | 368.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 12:15:00 | 328.90 | 367.64 | 368.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 330.90 | 326.70 | 341.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 13:15:00 | 341.60 | 328.29 | 341.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 341.60 | 328.29 | 341.24 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 339.25 | 328.68 | 341.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-13 10:15:00 | 344.25 | 328.84 | 341.25 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-30 12:15:00 | 337.85 | 343.36 | 345.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-30 13:15:00 | 340.05 | 343.33 | 345.77 | ENTRY2 sustain failed after 60m |

