# Jupiter Wagons Ltd. (JWL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 298.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 4 |
| ALERT3 | 28 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 26 |
| PARTIAL | 13 |
| TARGET_HIT | 11 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 39 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 24 / 15
- **Target hits / Stop hits / Partials:** 11 / 15 / 13
- **Avg / median % per leg:** 3.68% / 5.00%
- **Sum % (uncompounded):** 143.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.61% | -15.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.61% | -15.7% |
| SELL (all) | 33 | 24 | 72.7% | 11 | 9 | 13 | 4.82% | 159.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 24 | 72.7% | 11 | 9 | 13 | 4.82% | 159.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 39 | 24 | 61.5% | 11 | 15 | 13 | 3.68% | 143.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 11:15:00 | 554.80 | 588.25 | 588.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 15:15:00 | 554.00 | 586.98 | 587.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 12:15:00 | 558.00 | 557.81 | 569.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-13 13:00:00 | 558.00 | 557.81 | 569.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 526.00 | 503.44 | 523.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 526.00 | 503.44 | 523.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 523.00 | 503.64 | 523.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:15:00 | 513.85 | 503.64 | 523.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 15:15:00 | 488.16 | 504.31 | 520.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-12 13:15:00 | 462.47 | 500.53 | 517.47 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2024-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 12:15:00 | 524.35 | 507.25 | 507.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 14:15:00 | 535.75 | 507.69 | 507.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 14:15:00 | 510.65 | 511.66 | 509.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-26 15:00:00 | 510.65 | 511.66 | 509.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 508.70 | 511.63 | 509.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 09:15:00 | 515.15 | 511.63 | 509.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 10:15:00 | 507.90 | 511.56 | 509.58 | SL hit (close<static) qty=1.00 sl=508.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 477.05 | 507.81 | 507.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 12:15:00 | 476.50 | 505.17 | 506.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 11:15:00 | 487.45 | 486.33 | 496.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 11:15:00 | 487.45 | 486.33 | 496.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 11:15:00 | 487.45 | 486.33 | 496.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 11:45:00 | 494.65 | 486.33 | 496.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 13:15:00 | 489.50 | 486.32 | 495.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 13:30:00 | 487.50 | 486.32 | 495.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 14:15:00 | 495.10 | 486.41 | 495.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 14:30:00 | 506.20 | 486.41 | 495.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 491.65 | 486.46 | 495.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:15:00 | 492.80 | 486.46 | 495.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 490.45 | 486.50 | 495.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:15:00 | 491.00 | 486.50 | 495.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 483.60 | 486.47 | 495.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 09:15:00 | 470.95 | 487.51 | 495.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:30:00 | 479.00 | 486.03 | 494.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 12:15:00 | 479.55 | 486.03 | 494.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 13:15:00 | 478.20 | 485.97 | 494.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 12:15:00 | 447.40 | 484.56 | 493.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 12:15:00 | 455.05 | 484.56 | 493.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 12:15:00 | 455.57 | 484.56 | 493.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 12:15:00 | 454.29 | 484.56 | 493.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-24 13:15:00 | 423.86 | 484.02 | 492.97 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 12:15:00 | 445.95 | 369.53 | 369.34 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 14:15:00 | 369.30 | 382.23 | 382.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 09:15:00 | 368.30 | 381.96 | 382.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 368.95 | 345.02 | 357.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 368.95 | 345.02 | 357.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 368.95 | 345.02 | 357.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 370.25 | 345.02 | 357.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 361.70 | 345.19 | 357.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:15:00 | 358.70 | 345.35 | 357.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 13:30:00 | 359.65 | 345.64 | 357.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 15:15:00 | 340.76 | 345.59 | 357.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 15:15:00 | 341.67 | 345.59 | 357.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-28 09:15:00 | 322.83 | 343.50 | 355.39 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 325.40 | 309.06 | 308.98 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 14:15:00 | 293.00 | 308.94 | 308.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 290.30 | 308.62 | 308.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 11:15:00 | 326.20 | 306.87 | 307.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 11:15:00 | 326.20 | 306.87 | 307.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 326.20 | 306.87 | 307.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:00:00 | 326.20 | 306.87 | 307.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 325.35 | 307.06 | 307.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:45:00 | 322.20 | 308.25 | 308.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:45:00 | 322.45 | 308.39 | 308.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 14:00:00 | 322.80 | 308.53 | 308.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 15:00:00 | 322.95 | 308.67 | 308.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 326.40 | 308.96 | 308.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 326.40 | 308.96 | 308.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 334.45 | 312.50 | 310.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 13:15:00 | 310.30 | 312.77 | 311.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 13:15:00 | 310.30 | 312.77 | 311.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 310.30 | 312.77 | 311.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 310.30 | 312.77 | 311.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 301.50 | 312.66 | 310.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 301.50 | 312.66 | 310.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 302.10 | 312.55 | 310.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 308.10 | 312.55 | 310.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:30:00 | 304.70 | 312.05 | 310.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 296.55 | 310.87 | 310.26 | SL hit (close<static) qty=1.00 sl=298.35 alert=retest2 |

### Cycle 9 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 299.80 | 309.67 | 309.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 293.80 | 309.02 | 309.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 10:15:00 | 287.95 | 287.74 | 296.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 10:30:00 | 287.80 | 287.74 | 296.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 292.80 | 287.82 | 296.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:45:00 | 294.70 | 287.82 | 296.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 298.30 | 287.98 | 296.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 298.30 | 287.98 | 296.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 304.10 | 288.14 | 296.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 285.55 | 288.14 | 296.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 295.60 | 288.21 | 296.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 09:15:00 | 280.00 | 288.19 | 296.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 11:45:00 | 283.30 | 288.02 | 296.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:30:00 | 282.80 | 287.98 | 296.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 09:45:00 | 282.60 | 287.85 | 295.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 269.13 | 287.13 | 295.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 268.66 | 287.13 | 295.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 268.47 | 287.13 | 295.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:15:00 | 266.00 | 285.96 | 294.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-23 09:15:00 | 254.97 | 278.83 | 288.89 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 10 — BUY (started 2026-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 15:15:00 | 298.90 | 283.07 | 283.01 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-04 09:15:00 | 513.85 | 2024-11-08 15:15:00 | 488.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-04 09:15:00 | 513.85 | 2024-11-12 13:15:00 | 462.47 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-12-27 09:15:00 | 515.15 | 2024-12-27 10:15:00 | 507.90 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-01-03 09:15:00 | 516.45 | 2025-01-06 09:15:00 | 499.00 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2025-01-03 14:45:00 | 511.35 | 2025-01-06 09:15:00 | 499.00 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-01-22 09:15:00 | 470.95 | 2025-01-24 12:15:00 | 447.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 11:30:00 | 479.00 | 2025-01-24 12:15:00 | 455.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 12:15:00 | 479.55 | 2025-01-24 12:15:00 | 455.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 13:15:00 | 478.20 | 2025-01-24 12:15:00 | 454.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-22 09:15:00 | 470.95 | 2025-01-24 13:15:00 | 423.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-23 11:30:00 | 479.00 | 2025-01-24 13:15:00 | 431.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-23 12:15:00 | 479.55 | 2025-01-24 13:15:00 | 431.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-23 13:15:00 | 478.20 | 2025-01-24 13:15:00 | 430.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-24 11:30:00 | 357.55 | 2025-03-26 09:15:00 | 371.50 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2025-04-07 09:15:00 | 342.95 | 2025-04-07 09:15:00 | 325.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 342.95 | 2025-04-07 10:15:00 | 348.45 | STOP_HIT | 0.50 | -1.60% |
| SELL | retest2 | 2025-04-07 14:15:00 | 358.15 | 2025-04-07 14:15:00 | 371.35 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2025-04-30 14:00:00 | 358.00 | 2025-05-06 15:15:00 | 340.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 14:00:00 | 358.00 | 2025-05-12 09:15:00 | 361.55 | STOP_HIT | 0.50 | -0.99% |
| SELL | retest2 | 2025-05-13 13:00:00 | 369.05 | 2025-05-13 14:15:00 | 373.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-08-21 12:15:00 | 358.70 | 2025-08-21 15:15:00 | 340.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 13:30:00 | 359.65 | 2025-08-21 15:15:00 | 341.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 12:15:00 | 358.70 | 2025-08-28 09:15:00 | 322.83 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-21 13:30:00 | 359.65 | 2025-08-28 09:15:00 | 323.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-16 11:45:00 | 322.20 | 2026-01-19 09:15:00 | 326.40 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-01-16 12:45:00 | 322.45 | 2026-01-19 09:15:00 | 326.40 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-01-16 14:00:00 | 322.80 | 2026-01-19 09:15:00 | 326.40 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-01-16 15:00:00 | 322.95 | 2026-01-19 09:15:00 | 326.40 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-02-02 09:15:00 | 308.10 | 2026-02-06 09:15:00 | 296.55 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest2 | 2026-02-02 14:30:00 | 304.70 | 2026-02-06 09:15:00 | 296.55 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2026-02-09 13:45:00 | 306.10 | 2026-02-12 10:15:00 | 299.80 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2026-03-10 09:15:00 | 280.00 | 2026-03-12 09:15:00 | 269.13 | PARTIAL | 0.50 | 3.88% |
| SELL | retest2 | 2026-03-10 11:45:00 | 283.30 | 2026-03-12 09:15:00 | 268.66 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2026-03-10 12:30:00 | 282.80 | 2026-03-12 09:15:00 | 268.47 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2026-03-11 09:45:00 | 282.60 | 2026-03-13 10:15:00 | 266.00 | PARTIAL | 0.50 | 5.87% |
| SELL | retest2 | 2026-03-10 09:15:00 | 280.00 | 2026-03-23 09:15:00 | 254.97 | TARGET_HIT | 0.50 | 8.94% |
| SELL | retest2 | 2026-03-10 11:45:00 | 283.30 | 2026-03-23 09:15:00 | 254.52 | TARGET_HIT | 0.50 | 10.16% |
| SELL | retest2 | 2026-03-10 12:30:00 | 282.80 | 2026-03-23 09:15:00 | 254.34 | TARGET_HIT | 0.50 | 10.06% |
| SELL | retest2 | 2026-03-11 09:45:00 | 282.60 | 2026-03-23 10:15:00 | 252.00 | TARGET_HIT | 0.50 | 10.83% |
