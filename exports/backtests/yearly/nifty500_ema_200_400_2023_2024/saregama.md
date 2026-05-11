# Saregama India Ltd (SAREGAMA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 360.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 15 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT2_SKIP | 8 |
| ALERT3 | 41 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 45 |
| PARTIAL | 14 |
| TARGET_HIT | 11 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 59 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 26
- **Target hits / Stop hits / Partials:** 11 / 34 / 14
- **Avg / median % per leg:** 2.64% / 3.31%
- **Sum % (uncompounded):** 155.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 5 | 4 | 0 | 4.30% | 38.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 5 | 55.6% | 5 | 4 | 0 | 4.30% | 38.7% |
| SELL (all) | 50 | 28 | 56.0% | 6 | 30 | 14 | 2.34% | 117.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 50 | 28 | 56.0% | 6 | 30 | 14 | 2.34% | 117.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 59 | 33 | 55.9% | 11 | 34 | 14 | 2.64% | 155.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 11:15:00 | 367.55 | 386.04 | 386.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-29 12:15:00 | 366.50 | 385.84 | 385.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-17 09:15:00 | 361.50 | 346.77 | 357.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-17 09:15:00 | 361.50 | 346.77 | 357.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 361.50 | 346.77 | 357.89 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 14:15:00 | 384.25 | 364.03 | 363.99 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 12:15:00 | 352.00 | 365.41 | 365.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 347.55 | 364.85 | 365.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-01 09:15:00 | 355.35 | 354.94 | 359.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 10:15:00 | 359.95 | 354.99 | 359.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 10:15:00 | 359.95 | 354.99 | 359.37 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2024-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 10:15:00 | 415.20 | 361.38 | 361.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 09:15:00 | 425.35 | 377.67 | 370.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 13:15:00 | 386.40 | 387.63 | 377.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 11:15:00 | 376.10 | 387.21 | 377.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 11:15:00 | 376.10 | 387.21 | 377.98 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2024-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 11:15:00 | 346.40 | 371.58 | 371.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 13:15:00 | 344.80 | 371.07 | 371.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 11:15:00 | 370.65 | 369.54 | 370.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-02 11:15:00 | 370.65 | 369.54 | 370.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 11:15:00 | 370.65 | 369.54 | 370.62 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 12:15:00 | 386.15 | 371.66 | 371.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-04 14:15:00 | 387.30 | 371.95 | 371.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 13:15:00 | 413.25 | 419.62 | 405.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-22 14:00:00 | 413.25 | 419.62 | 405.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 406.50 | 419.31 | 405.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:15:00 | 405.95 | 419.31 | 405.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 406.30 | 419.18 | 405.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 11:15:00 | 407.45 | 419.18 | 405.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-23 12:15:00 | 448.20 | 419.70 | 406.38 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 511.25 | 534.65 | 534.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 12:15:00 | 508.55 | 533.69 | 534.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 504.65 | 499.03 | 512.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-28 09:30:00 | 505.80 | 499.03 | 512.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 508.00 | 498.88 | 511.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:45:00 | 510.00 | 498.88 | 511.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 516.45 | 499.23 | 511.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:30:00 | 518.00 | 499.23 | 511.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 515.50 | 499.39 | 511.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:30:00 | 514.95 | 499.39 | 511.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 512.30 | 505.48 | 513.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 11:45:00 | 510.15 | 505.51 | 513.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 09:30:00 | 507.80 | 505.52 | 512.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 14:15:00 | 516.05 | 505.89 | 512.75 | SL hit (close>static) qty=1.00 sl=514.95 alert=retest2 |

### Cycle 8 — BUY (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 09:15:00 | 531.70 | 509.51 | 509.40 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 15:15:00 | 489.00 | 511.17 | 511.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 09:15:00 | 484.45 | 510.90 | 511.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 10:15:00 | 522.60 | 505.07 | 508.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 10:15:00 | 522.60 | 505.07 | 508.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 522.60 | 505.07 | 508.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:45:00 | 520.55 | 505.07 | 508.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 11:15:00 | 528.00 | 505.30 | 508.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 11:30:00 | 527.50 | 505.30 | 508.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 15:15:00 | 527.15 | 510.60 | 510.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-10 13:15:00 | 532.35 | 511.97 | 511.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 09:15:00 | 509.50 | 512.34 | 511.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-11 09:15:00 | 509.50 | 512.34 | 511.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 509.50 | 512.34 | 511.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:45:00 | 508.70 | 512.34 | 511.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 10:15:00 | 511.80 | 512.34 | 511.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 10:30:00 | 507.30 | 512.34 | 511.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 11:15:00 | 512.50 | 512.34 | 511.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 11:30:00 | 512.15 | 512.34 | 511.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 12:15:00 | 505.40 | 512.27 | 511.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 13:00:00 | 505.40 | 512.27 | 511.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 13:15:00 | 510.20 | 512.25 | 511.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-11 14:45:00 | 517.05 | 512.31 | 511.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-12 09:45:00 | 512.85 | 512.35 | 511.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-14 12:15:00 | 499.55 | 513.46 | 512.19 | SL hit (close<static) qty=1.00 sl=505.40 alert=retest2 |

### Cycle 11 — SELL (started 2025-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 13:15:00 | 474.95 | 510.92 | 510.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 09:15:00 | 470.20 | 509.84 | 510.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 12:15:00 | 500.60 | 490.96 | 499.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 12:15:00 | 500.60 | 490.96 | 499.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 500.60 | 490.96 | 499.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:00:00 | 500.60 | 490.96 | 499.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 498.10 | 491.03 | 499.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-07 14:45:00 | 494.40 | 491.03 | 499.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 09:45:00 | 487.70 | 490.97 | 498.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 14:15:00 | 497.65 | 490.13 | 497.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 14:45:00 | 491.60 | 490.10 | 497.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 500.00 | 490.17 | 497.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 500.00 | 490.17 | 497.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 500.15 | 490.27 | 497.22 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-19 10:15:00 | 504.00 | 490.93 | 497.31 | SL hit (close>static) qty=1.00 sl=501.00 alert=retest2 |

### Cycle 12 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 516.55 | 501.80 | 501.79 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 10:15:00 | 475.50 | 501.61 | 501.72 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 525.50 | 501.77 | 501.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 531.00 | 502.06 | 501.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 14:15:00 | 524.15 | 528.33 | 517.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 15:00:00 | 524.15 | 528.33 | 517.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 526.50 | 528.25 | 517.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 15:00:00 | 532.30 | 528.20 | 517.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 12:45:00 | 535.50 | 528.38 | 518.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 15:15:00 | 534.25 | 528.65 | 518.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 10:30:00 | 534.00 | 528.77 | 519.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-04 09:15:00 | 585.53 | 540.78 | 531.48 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2025-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 12:15:00 | 503.85 | 531.83 | 531.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 14:15:00 | 502.20 | 531.26 | 531.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 09:15:00 | 518.20 | 506.28 | 516.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 09:15:00 | 518.20 | 506.28 | 516.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 518.20 | 506.28 | 516.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 518.20 | 506.28 | 516.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 515.50 | 506.37 | 516.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 12:30:00 | 510.50 | 506.51 | 516.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 13:15:00 | 511.00 | 506.51 | 516.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 15:00:00 | 510.50 | 506.62 | 516.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 10:00:00 | 511.00 | 506.69 | 516.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 485.45 | 504.60 | 513.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 485.45 | 504.60 | 513.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 484.97 | 504.40 | 513.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 484.97 | 504.40 | 513.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 492.95 | 494.05 | 505.53 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 494.10 | 494.05 | 505.47 | SL hit (close>ema200) qty=0.50 sl=494.05 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-23 11:15:00 | 407.45 | 2024-05-23 12:15:00 | 448.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-09 11:45:00 | 510.15 | 2024-12-11 14:15:00 | 516.05 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-12-11 09:30:00 | 507.80 | 2024-12-11 14:15:00 | 516.05 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-12-12 09:15:00 | 508.20 | 2024-12-23 09:15:00 | 482.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 09:15:00 | 508.20 | 2024-12-31 10:15:00 | 457.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-03 11:30:00 | 510.05 | 2025-01-03 13:15:00 | 544.85 | STOP_HIT | 1.00 | -6.82% |
| BUY | retest2 | 2025-02-11 14:45:00 | 517.05 | 2025-02-14 12:15:00 | 499.55 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2025-02-12 09:45:00 | 512.85 | 2025-02-14 12:15:00 | 499.55 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-03-07 14:45:00 | 494.40 | 2025-03-19 10:15:00 | 504.00 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-03-10 09:45:00 | 487.70 | 2025-03-19 10:15:00 | 504.00 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2025-03-17 14:15:00 | 497.65 | 2025-03-19 10:15:00 | 504.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-03-17 14:45:00 | 491.60 | 2025-03-19 10:15:00 | 504.00 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-05-02 15:00:00 | 532.30 | 2025-06-04 09:15:00 | 585.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-05 12:45:00 | 535.50 | 2025-06-04 09:15:00 | 589.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-06 15:15:00 | 534.25 | 2025-06-04 09:15:00 | 587.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-07 10:30:00 | 534.00 | 2025-06-04 09:15:00 | 587.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-13 15:15:00 | 539.95 | 2025-06-18 13:15:00 | 526.50 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-06-16 10:45:00 | 541.85 | 2025-06-18 13:15:00 | 526.50 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-07-18 12:30:00 | 510.50 | 2025-07-25 10:15:00 | 485.45 | PARTIAL | 0.50 | 4.91% |
| SELL | retest2 | 2025-07-18 13:15:00 | 511.00 | 2025-07-25 10:15:00 | 485.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 15:00:00 | 510.50 | 2025-07-25 11:15:00 | 484.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 10:00:00 | 511.00 | 2025-07-25 11:15:00 | 484.97 | PARTIAL | 0.50 | 5.09% |
| SELL | retest2 | 2025-07-18 12:30:00 | 510.50 | 2025-08-06 10:15:00 | 494.10 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2025-07-18 13:15:00 | 511.00 | 2025-08-06 10:15:00 | 494.10 | STOP_HIT | 0.50 | 3.31% |
| SELL | retest2 | 2025-07-18 15:00:00 | 510.50 | 2025-08-06 10:15:00 | 494.10 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2025-07-21 10:00:00 | 511.00 | 2025-08-06 10:15:00 | 494.10 | STOP_HIT | 0.50 | 3.31% |
| SELL | retest2 | 2025-08-07 09:30:00 | 481.70 | 2025-09-08 09:15:00 | 506.55 | STOP_HIT | 1.00 | -5.16% |
| SELL | retest2 | 2025-08-07 11:15:00 | 481.95 | 2025-09-08 09:15:00 | 506.55 | STOP_HIT | 1.00 | -5.10% |
| SELL | retest2 | 2025-08-12 10:45:00 | 482.05 | 2025-09-08 09:15:00 | 506.55 | STOP_HIT | 1.00 | -5.08% |
| SELL | retest2 | 2025-08-13 09:30:00 | 479.55 | 2025-09-08 09:15:00 | 506.55 | STOP_HIT | 1.00 | -5.63% |
| SELL | retest2 | 2025-08-22 15:00:00 | 489.15 | 2025-09-08 09:15:00 | 506.55 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2025-08-25 09:15:00 | 491.55 | 2025-09-08 09:15:00 | 506.55 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-08-25 13:00:00 | 490.80 | 2025-09-08 09:15:00 | 506.55 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2025-09-02 15:15:00 | 490.55 | 2025-09-12 10:15:00 | 499.45 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-09-03 12:15:00 | 495.15 | 2025-09-18 09:15:00 | 500.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-09-03 12:45:00 | 494.65 | 2025-09-18 09:15:00 | 500.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-09-04 11:00:00 | 493.15 | 2025-09-18 09:15:00 | 500.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-09-11 13:30:00 | 495.00 | 2025-09-18 09:15:00 | 500.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-09-12 13:15:00 | 493.35 | 2025-09-18 09:15:00 | 500.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-09-12 14:45:00 | 493.45 | 2025-09-18 09:15:00 | 500.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-09-15 09:45:00 | 491.80 | 2025-09-18 09:15:00 | 500.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-09-15 12:30:00 | 493.95 | 2025-09-26 09:15:00 | 468.40 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2025-09-17 13:00:00 | 493.05 | 2025-10-09 09:15:00 | 457.61 | PARTIAL | 0.50 | 7.19% |
| SELL | retest2 | 2025-09-17 14:00:00 | 493.05 | 2025-10-09 09:15:00 | 457.85 | PARTIAL | 0.50 | 7.14% |
| SELL | retest2 | 2025-09-17 15:15:00 | 491.00 | 2025-10-09 09:15:00 | 457.95 | PARTIAL | 0.50 | 6.73% |
| SELL | retest2 | 2025-09-18 15:00:00 | 493.05 | 2025-10-09 09:15:00 | 455.57 | PARTIAL | 0.50 | 7.60% |
| SELL | retest2 | 2025-09-15 12:30:00 | 493.95 | 2025-11-06 09:15:00 | 433.53 | TARGET_HIT | 0.50 | 12.23% |
| SELL | retest2 | 2025-09-17 13:00:00 | 493.05 | 2025-11-06 09:15:00 | 433.75 | TARGET_HIT | 0.50 | 12.03% |
| SELL | retest2 | 2025-09-17 14:00:00 | 493.05 | 2025-11-06 09:15:00 | 433.85 | TARGET_HIT | 0.50 | 12.01% |
| SELL | retest2 | 2025-09-17 15:15:00 | 491.00 | 2025-11-06 09:15:00 | 431.60 | TARGET_HIT | 0.50 | 12.10% |
| SELL | retest2 | 2025-09-18 15:00:00 | 493.05 | 2025-11-06 09:15:00 | 443.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 357.75 | 2026-02-13 15:15:00 | 341.33 | PARTIAL | 0.50 | 4.59% |
| SELL | retest2 | 2026-02-12 10:45:00 | 359.30 | 2026-02-16 11:15:00 | 339.86 | PARTIAL | 0.50 | 5.41% |
| SELL | retest2 | 2026-02-12 09:15:00 | 357.75 | 2026-02-17 13:15:00 | 345.60 | STOP_HIT | 0.50 | 3.40% |
| SELL | retest2 | 2026-02-12 10:45:00 | 359.30 | 2026-02-17 13:15:00 | 345.60 | STOP_HIT | 0.50 | 3.81% |
| SELL | retest2 | 2026-03-16 14:45:00 | 358.95 | 2026-03-19 10:15:00 | 341.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-16 14:45:00 | 358.95 | 2026-03-19 10:15:00 | 341.30 | STOP_HIT | 0.50 | 4.92% |
| SELL | retest2 | 2026-04-27 14:45:00 | 359.65 | 2026-04-30 10:15:00 | 341.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 14:45:00 | 359.65 | 2026-04-30 10:15:00 | 342.70 | STOP_HIT | 0.50 | 4.71% |
