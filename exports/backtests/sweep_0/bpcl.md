# BPCL (BPCL)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 303.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 2 |
| PENDING | 11 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| PARTIAL | 2 |
| TARGET_HIT | 3 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 5 / 0
- **Target hits / Stop hits / Partials:** 3 / 0 / 2
- **Avg / median % per leg:** 8.00% / 9.15%
- **Sum % (uncompounded):** 40.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 5 | 100.0% | 3 | 0 | 2 | 8.00% | 40.0% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| BUY @ 3rd Alert (retest2) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| retest2 (combined) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 10:15:00 | 310.30 | 322.28 | 322.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 12:15:00 | 307.80 | 322.02 | 322.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 322.85 | 318.98 | 320.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 322.85 | 318.98 | 320.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 322.85 | 318.98 | 320.36 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-09-15 11:15:00 | 317.45 | 319.11 | 320.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-15 12:15:00 | 317.95 | 319.10 | 320.31 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-15 13:15:00 | 315.85 | 319.07 | 320.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-15 14:15:00 | 317.90 | 319.06 | 320.28 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-16 11:15:00 | 317.45 | 319.06 | 320.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-16 12:15:00 | 318.15 | 319.05 | 320.24 | ENTRY2 sustain failed after 60m |

### Cycle 2 — BUY (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 12:15:00 | 334.05 | 321.26 | 321.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 09:15:00 | 335.30 | 322.87 | 322.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 12:15:00 | 332.15 | 332.18 | 327.83 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-10-15 10:15:00 | 338.95 | 332.25 | 327.97 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 11:15:00 | 338.40 | 332.31 | 328.02 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-17 13:15:00 | 335.55 | 332.84 | 328.63 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 14:15:00 | 335.80 | 332.87 | 328.67 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 334.00 | 333.33 | 329.27 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-10-27 09:15:00 | 338.20 | 333.29 | 329.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 10:15:00 | 337.85 | 333.33 | 329.43 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 09:15:00 | 352.59 | 335.26 | 330.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 10:15:00 | 355.32 | 335.48 | 330.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-11-03 12:15:00 | 369.38 | 339.08 | 333.15 | Target hit (10%) qty=0.50 alert=retest1 |
| Target hit | 2025-11-04 09:15:00 | 372.24 | 340.24 | 333.86 | Target hit (10%) qty=0.50 alert=retest1 |
| Target hit | 2025-11-04 09:15:00 | 371.64 | 340.24 | 333.86 | Target hit (10%) qty=1.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-09 11:15:00 | 334.55 | 369.07 | 366.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-09 12:15:00 | 334.25 | 368.72 | 366.68 | ENTRY2 sustain failed after 60m |

### Cycle 3 — SELL (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 09:15:00 | 325.15 | 364.39 | 364.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 323.85 | 363.99 | 364.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 14:15:00 | 310.40 | 307.34 | 325.99 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-24 09:15:00 | 301.90 | 309.28 | 323.38 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:15:00 | 305.65 | 309.25 | 323.29 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-29 14:15:00 | 303.75 | 309.19 | 321.62 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 15:15:00 | 304.60 | 309.15 | 321.53 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-05-07 11:15:00 | 305.65 | 307.49 | 318.84 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-05-07 12:15:00 | 308.45 | 307.50 | 318.79 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-05-08 09:15:00 | 303.00 | 307.47 | 318.55 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:15:00 | 305.20 | 307.45 | 318.48 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-10-15 11:15:00 | 338.40 | 2025-10-30 09:15:00 | 352.59 | PARTIAL | 0.50 | 4.19% |
| BUY | retest1 | 2025-10-17 14:15:00 | 335.80 | 2025-10-30 10:15:00 | 355.32 | PARTIAL | 0.50 | 5.81% |
| BUY | retest1 | 2025-10-15 11:15:00 | 338.40 | 2025-11-03 12:15:00 | 369.38 | TARGET_HIT | 0.50 | 9.15% |
| BUY | retest1 | 2025-10-17 14:15:00 | 335.80 | 2025-11-04 09:15:00 | 372.24 | TARGET_HIT | 0.50 | 10.85% |
| BUY | retest2 | 2025-10-27 10:15:00 | 337.85 | 2025-11-04 09:15:00 | 371.64 | TARGET_HIT | 1.00 | 10.00% |
