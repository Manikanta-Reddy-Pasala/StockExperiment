# Exide Industries Ltd. (EXIDEIND)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 361.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 1 |
| ALERT3 | 61 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 23 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 22
- **Target hits / Stop hits / Partials:** 1 / 22 / 1
- **Avg / median % per leg:** -0.77% / -1.24%
- **Sum % (uncompounded):** -18.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 0 | 0.0% | 0 | 22 | 0 | -1.52% | -33.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 0 | 0.0% | 0 | 22 | 0 | -1.52% | -33.5% |
| SELL (all) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 24 | 2 | 8.3% | 1 | 22 | 1 | -0.77% | -18.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 10:15:00 | 305.00 | 315.60 | 315.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 09:15:00 | 304.20 | 314.97 | 315.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 14:15:00 | 314.30 | 314.12 | 314.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-02 15:00:00 | 314.30 | 314.12 | 314.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 314.00 | 314.13 | 314.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 09:45:00 | 314.85 | 314.13 | 314.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 12:15:00 | 315.25 | 314.14 | 314.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 12:45:00 | 315.75 | 314.14 | 314.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 13:15:00 | 314.80 | 314.15 | 314.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 13:30:00 | 315.75 | 314.15 | 314.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 14:15:00 | 313.70 | 314.15 | 314.83 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 11:15:00 | 360.15 | 315.66 | 315.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 12:15:00 | 371.90 | 316.22 | 315.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 461.65 | 463.84 | 426.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-04 11:00:00 | 461.65 | 463.84 | 426.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 521.25 | 547.65 | 521.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:00:00 | 521.25 | 547.65 | 521.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 522.90 | 547.40 | 521.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 12:15:00 | 524.35 | 547.40 | 521.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 15:15:00 | 524.20 | 546.73 | 521.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 10:15:00 | 519.35 | 546.02 | 521.46 | SL hit (close<static) qty=1.00 sl=521.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 11:15:00 | 492.70 | 509.22 | 509.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 09:15:00 | 491.65 | 508.47 | 508.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 10:15:00 | 487.60 | 484.62 | 493.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-27 11:00:00 | 487.60 | 484.62 | 493.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 495.25 | 484.72 | 493.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 12:00:00 | 495.25 | 484.72 | 493.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 495.50 | 484.83 | 493.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 12:30:00 | 496.85 | 484.83 | 493.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 496.55 | 484.95 | 493.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 14:00:00 | 496.55 | 484.95 | 493.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 496.95 | 485.07 | 493.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 496.95 | 485.07 | 493.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 494.05 | 485.39 | 493.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:00:00 | 494.05 | 485.39 | 493.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 498.80 | 485.52 | 493.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:00:00 | 498.80 | 485.52 | 493.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 497.20 | 488.52 | 494.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:30:00 | 495.20 | 488.52 | 494.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 495.00 | 488.58 | 494.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:15:00 | 497.45 | 488.58 | 494.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 507.20 | 488.77 | 494.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:00:00 | 507.20 | 488.77 | 494.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 504.10 | 488.92 | 494.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:45:00 | 505.30 | 488.92 | 494.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 490.20 | 489.21 | 494.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:30:00 | 495.25 | 489.21 | 494.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 494.00 | 489.36 | 494.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 501.00 | 489.36 | 494.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 500.65 | 489.48 | 494.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:15:00 | 504.45 | 489.48 | 494.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 518.80 | 489.77 | 494.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 11:00:00 | 518.80 | 489.77 | 494.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 13:15:00 | 521.95 | 498.50 | 498.44 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-10-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 11:15:00 | 471.20 | 498.29 | 498.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 12:15:00 | 469.25 | 498.00 | 498.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 455.20 | 448.02 | 464.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-28 10:00:00 | 455.20 | 448.02 | 464.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 11:15:00 | 463.60 | 450.03 | 462.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 12:00:00 | 463.60 | 450.03 | 462.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 12:15:00 | 463.10 | 450.16 | 462.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 12:30:00 | 463.55 | 450.16 | 462.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 13:15:00 | 463.90 | 450.30 | 462.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 13:30:00 | 463.80 | 450.30 | 462.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 462.20 | 450.54 | 462.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:15:00 | 465.70 | 450.54 | 462.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 465.75 | 450.69 | 462.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:30:00 | 465.85 | 450.69 | 462.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 464.50 | 452.51 | 462.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 15:00:00 | 464.50 | 452.51 | 462.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 462.50 | 454.10 | 463.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:45:00 | 462.80 | 454.10 | 463.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 13:15:00 | 462.50 | 454.18 | 463.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 13:30:00 | 463.20 | 454.18 | 463.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 462.20 | 454.35 | 462.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:15:00 | 463.75 | 454.35 | 462.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 462.20 | 454.43 | 462.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 11:30:00 | 458.55 | 454.51 | 462.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 435.62 | 453.85 | 461.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-24 09:15:00 | 412.69 | 449.38 | 458.58 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 12:15:00 | 394.50 | 372.82 | 372.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 11:15:00 | 395.85 | 382.56 | 378.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 386.55 | 386.87 | 381.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 11:00:00 | 386.55 | 386.87 | 381.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 380.10 | 386.73 | 381.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 379.00 | 386.73 | 381.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 385.20 | 386.72 | 381.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:30:00 | 386.20 | 386.70 | 381.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 10:15:00 | 385.55 | 386.59 | 381.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 12:15:00 | 380.00 | 386.23 | 381.82 | SL hit (close<static) qty=1.00 sl=380.05 alert=retest2 |

### Cycle 7 — SELL (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 15:15:00 | 376.00 | 383.51 | 383.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 373.85 | 383.41 | 383.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 13:15:00 | 383.50 | 382.33 | 382.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 13:15:00 | 383.50 | 382.33 | 382.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 383.50 | 382.33 | 382.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:00:00 | 383.50 | 382.33 | 382.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 393.75 | 382.44 | 382.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 393.75 | 382.44 | 382.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 396.55 | 383.60 | 383.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 11:15:00 | 397.35 | 383.74 | 383.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 12:15:00 | 408.25 | 408.50 | 399.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 13:00:00 | 408.25 | 408.50 | 399.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 399.20 | 408.00 | 399.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 399.20 | 408.00 | 399.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 398.85 | 407.91 | 399.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 398.85 | 407.91 | 399.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 400.10 | 407.76 | 399.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:45:00 | 399.90 | 407.76 | 399.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 399.60 | 407.68 | 399.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 14:30:00 | 399.00 | 407.68 | 399.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 399.30 | 407.59 | 399.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 396.60 | 407.59 | 399.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 397.15 | 406.52 | 399.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:45:00 | 396.95 | 406.52 | 399.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 398.20 | 402.46 | 398.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 398.45 | 402.46 | 398.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 399.30 | 402.43 | 398.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:45:00 | 396.95 | 402.43 | 398.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 405.00 | 402.37 | 398.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 10:30:00 | 407.70 | 402.41 | 398.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 15:15:00 | 397.00 | 402.30 | 398.88 | SL hit (close<static) qty=1.00 sl=398.10 alert=retest2 |

### Cycle 9 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 379.60 | 397.05 | 397.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 378.20 | 392.44 | 394.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 380.25 | 379.94 | 385.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 15:00:00 | 380.25 | 379.94 | 385.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 322.75 | 309.74 | 321.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:45:00 | 324.00 | 309.74 | 321.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 323.25 | 309.88 | 321.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:00:00 | 323.25 | 309.88 | 321.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 12:15:00 | 324.10 | 310.02 | 321.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:45:00 | 324.30 | 310.02 | 321.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 327.70 | 310.77 | 321.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:00:00 | 327.70 | 310.77 | 321.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 13:15:00 | 365.35 | 328.88 | 328.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 374.30 | 332.31 | 330.49 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-07-31 12:15:00 | 524.35 | 2024-08-01 10:15:00 | 519.35 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-07-31 15:15:00 | 524.20 | 2024-08-01 10:15:00 | 519.35 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-12-17 11:30:00 | 458.55 | 2024-12-19 09:15:00 | 435.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 11:30:00 | 458.55 | 2024-12-24 09:15:00 | 412.69 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-06-16 11:30:00 | 386.20 | 2025-06-18 12:15:00 | 380.00 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-06-17 10:15:00 | 385.55 | 2025-06-18 12:15:00 | 380.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-06-24 09:30:00 | 385.65 | 2025-07-03 10:15:00 | 381.90 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-06-24 14:45:00 | 385.25 | 2025-07-11 14:15:00 | 381.35 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-07-02 11:30:00 | 384.60 | 2025-07-11 14:15:00 | 381.35 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-07-04 09:15:00 | 387.40 | 2025-07-11 14:15:00 | 381.35 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-07-04 15:15:00 | 384.50 | 2025-07-11 15:15:00 | 379.75 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-07-07 10:00:00 | 384.45 | 2025-07-11 15:15:00 | 379.75 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-07-15 09:15:00 | 386.30 | 2025-07-16 09:15:00 | 380.45 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-07-16 12:30:00 | 384.35 | 2025-07-17 09:15:00 | 382.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-07-17 11:15:00 | 384.05 | 2025-07-18 09:15:00 | 380.80 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-07-18 09:15:00 | 384.30 | 2025-07-18 09:15:00 | 380.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-07-21 10:00:00 | 388.20 | 2025-07-25 14:15:00 | 381.30 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-07-29 09:30:00 | 386.45 | 2025-08-01 09:15:00 | 379.65 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-07-31 10:45:00 | 387.40 | 2025-08-01 09:15:00 | 379.65 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-07-31 11:15:00 | 387.40 | 2025-08-01 09:15:00 | 379.65 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-08-04 14:45:00 | 389.95 | 2025-08-05 13:15:00 | 379.80 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-08-05 10:45:00 | 389.75 | 2025-08-05 13:15:00 | 379.80 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-08-05 11:30:00 | 389.80 | 2025-08-05 13:15:00 | 379.80 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-10-07 10:30:00 | 407.70 | 2025-10-08 15:15:00 | 397.00 | STOP_HIT | 1.00 | -2.62% |
