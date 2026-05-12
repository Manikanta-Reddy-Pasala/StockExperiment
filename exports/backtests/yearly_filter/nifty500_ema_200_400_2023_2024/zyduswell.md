# Zydus Wellness Ltd. (ZYDUSWELL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 517.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 2 |
| ALERT3 | 90 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 78 |
| PARTIAL | 13 |
| TARGET_HIT | 11 |
| STOP_HIT | 71 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 95 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 67
- **Target hits / Stop hits / Partials:** 11 / 71 / 13
- **Avg / median % per leg:** 0.43% / -1.24%
- **Sum % (uncompounded):** 41.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 3 | 9.7% | 3 | 28 | 0 | -0.44% | -13.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 31 | 3 | 9.7% | 3 | 28 | 0 | -0.44% | -13.6% |
| SELL (all) | 64 | 25 | 39.1% | 8 | 43 | 13 | 0.86% | 54.8% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -7.99% | -32.0% |
| SELL @ 3rd Alert (retest2) | 60 | 25 | 41.7% | 8 | 39 | 13 | 1.45% | 86.8% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -7.99% | -32.0% |
| retest2 (combined) | 91 | 28 | 30.8% | 11 | 67 | 13 | 0.80% | 73.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 15:15:00 | 307.00 | 298.90 | 298.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 09:15:00 | 310.62 | 299.02 | 298.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 09:15:00 | 314.79 | 316.16 | 309.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-12 09:45:00 | 313.60 | 316.16 | 309.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 313.71 | 317.96 | 312.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-28 09:15:00 | 316.00 | 317.38 | 312.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-28 09:45:00 | 315.59 | 317.37 | 312.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 14:15:00 | 317.12 | 316.75 | 312.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-04 13:15:00 | 311.68 | 316.51 | 312.80 | SL hit (close<static) qty=1.00 sl=312.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-11-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 09:15:00 | 306.95 | 311.36 | 311.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 10:15:00 | 305.36 | 311.30 | 311.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 13:15:00 | 311.34 | 310.91 | 311.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 13:15:00 | 311.34 | 310.91 | 311.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 13:15:00 | 311.34 | 310.91 | 311.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 13:45:00 | 311.51 | 310.91 | 311.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 14:15:00 | 310.24 | 310.90 | 311.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 14:30:00 | 311.53 | 310.90 | 311.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 15:15:00 | 307.20 | 310.86 | 311.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 09:15:00 | 310.80 | 310.86 | 311.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 09:15:00 | 311.17 | 310.87 | 311.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-03 11:15:00 | 309.41 | 310.87 | 311.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-03 13:00:00 | 309.60 | 310.86 | 311.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-06 10:15:00 | 312.67 | 310.81 | 311.08 | SL hit (close>static) qty=1.00 sl=312.30 alert=retest2 |

### Cycle 3 — BUY (started 2023-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 11:15:00 | 321.16 | 310.36 | 310.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-19 14:15:00 | 322.80 | 311.35 | 310.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 12:15:00 | 323.30 | 324.97 | 320.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-17 13:00:00 | 323.30 | 324.97 | 320.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 320.57 | 324.87 | 320.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 09:45:00 | 317.58 | 324.87 | 320.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 10:15:00 | 322.71 | 324.85 | 320.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 13:45:00 | 323.17 | 324.78 | 320.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 15:15:00 | 324.00 | 324.75 | 320.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-23 09:15:00 | 323.39 | 324.73 | 320.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-23 09:15:00 | 319.31 | 324.68 | 320.43 | SL hit (close<static) qty=1.00 sl=319.85 alert=retest2 |

### Cycle 4 — SELL (started 2024-02-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 13:15:00 | 312.82 | 319.48 | 319.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-01 10:15:00 | 312.21 | 319.24 | 319.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 09:15:00 | 318.94 | 318.61 | 319.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-05 09:30:00 | 318.77 | 318.61 | 319.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 11:15:00 | 319.12 | 318.62 | 319.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 11:45:00 | 318.76 | 318.62 | 319.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 12:15:00 | 318.17 | 318.61 | 319.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-06 10:15:00 | 316.22 | 318.57 | 318.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 09:30:00 | 315.99 | 318.32 | 318.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 10:30:00 | 315.27 | 318.29 | 318.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 12:45:00 | 316.20 | 318.26 | 318.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 13:15:00 | 316.95 | 318.24 | 318.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 13:45:00 | 318.09 | 318.24 | 318.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 14:15:00 | 315.51 | 318.22 | 318.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 15:00:00 | 315.51 | 318.22 | 318.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 12:15:00 | 322.39 | 318.22 | 318.77 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-03-11 12:15:00 | 322.39 | 318.22 | 318.77 | SL hit (close>static) qty=1.00 sl=319.57 alert=retest2 |

### Cycle 5 — BUY (started 2024-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 12:15:00 | 332.28 | 314.09 | 314.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 14:15:00 | 334.55 | 317.11 | 315.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 337.00 | 340.72 | 332.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-04 11:00:00 | 337.00 | 340.72 | 332.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 435.96 | 448.21 | 435.09 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 15:15:00 | 389.70 | 426.76 | 426.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 387.98 | 424.98 | 425.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 10:15:00 | 391.18 | 391.03 | 403.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-31 11:00:00 | 391.18 | 391.03 | 403.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 401.98 | 391.17 | 401.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 397.01 | 391.17 | 401.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 397.43 | 391.23 | 401.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 14:00:00 | 394.14 | 391.41 | 401.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 12:15:00 | 379.75 | 391.42 | 401.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 374.43 | 390.49 | 400.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 12:15:00 | 389.51 | 389.40 | 399.22 | SL hit (close>ema200) qty=0.50 sl=389.40 alert=retest2 |

### Cycle 7 — BUY (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 12:15:00 | 359.22 | 347.19 | 347.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 362.60 | 347.73 | 347.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 13:15:00 | 405.34 | 406.24 | 394.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 14:00:00 | 405.34 | 406.24 | 394.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 402.80 | 405.89 | 395.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 12:45:00 | 394.84 | 405.89 | 395.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 398.14 | 406.63 | 396.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:00:00 | 398.14 | 406.63 | 396.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 397.14 | 406.44 | 396.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:30:00 | 397.30 | 406.44 | 396.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 396.52 | 406.34 | 396.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 15:00:00 | 396.52 | 406.34 | 396.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 396.20 | 406.24 | 396.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:30:00 | 393.64 | 406.14 | 396.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 393.98 | 406.02 | 396.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:45:00 | 393.60 | 406.02 | 396.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 394.28 | 405.90 | 396.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:45:00 | 395.48 | 399.56 | 395.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 11:30:00 | 394.98 | 399.41 | 395.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 12:45:00 | 397.34 | 399.39 | 395.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-01 10:15:00 | 435.03 | 400.82 | 397.01 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 14:15:00 | 430.25 | 454.54 | 454.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 10:15:00 | 429.30 | 453.82 | 454.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 15:15:00 | 434.45 | 433.76 | 441.16 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 10:15:00 | 426.45 | 433.71 | 441.10 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 13:00:00 | 426.85 | 433.08 | 440.42 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 15:00:00 | 426.20 | 432.95 | 440.28 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 09:15:00 | 424.70 | 432.90 | 440.22 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 460.10 | 431.65 | 438.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 460.10 | 431.65 | 438.79 | SL hit (close>ema400) qty=1.00 sl=438.79 alert=retest1 |

### Cycle 9 — BUY (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 14:15:00 | 475.30 | 444.75 | 444.63 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 437.30 | 444.62 | 444.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 432.95 | 444.37 | 444.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 10:15:00 | 437.00 | 435.68 | 439.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 10:15:00 | 437.00 | 435.68 | 439.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 437.00 | 435.68 | 439.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:00:00 | 437.00 | 435.68 | 439.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 441.80 | 435.74 | 439.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 441.80 | 435.74 | 439.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 440.20 | 435.79 | 439.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 440.20 | 435.79 | 439.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 442.35 | 435.85 | 439.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:45:00 | 442.95 | 435.85 | 439.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 447.50 | 435.97 | 439.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 15:00:00 | 447.50 | 435.97 | 439.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 439.50 | 436.47 | 439.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 428.45 | 436.54 | 439.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 10:15:00 | 407.03 | 434.86 | 438.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-26 11:15:00 | 385.61 | 412.23 | 422.68 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 11 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 522.80 | 419.18 | 418.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 09:15:00 | 531.55 | 439.94 | 430.02 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-09-28 09:15:00 | 316.00 | 2023-10-04 13:15:00 | 311.68 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2023-09-28 09:45:00 | 315.59 | 2023-10-04 13:15:00 | 311.68 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2023-10-03 14:15:00 | 317.12 | 2023-10-04 13:15:00 | 311.68 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2023-10-05 09:15:00 | 315.80 | 2023-10-09 09:15:00 | 308.06 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2023-10-11 09:45:00 | 312.20 | 2023-10-19 11:15:00 | 312.00 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2023-10-11 10:45:00 | 312.63 | 2023-10-19 11:15:00 | 312.00 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2023-10-11 15:00:00 | 312.03 | 2023-10-23 09:15:00 | 309.65 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2023-10-12 10:00:00 | 311.95 | 2023-10-23 09:15:00 | 309.65 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2023-10-12 15:15:00 | 314.20 | 2023-10-23 11:15:00 | 309.16 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2023-10-13 10:15:00 | 312.86 | 2023-10-23 11:15:00 | 309.16 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2023-10-13 13:45:00 | 312.78 | 2023-10-23 11:15:00 | 309.16 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2023-10-13 15:15:00 | 313.00 | 2023-10-23 11:15:00 | 309.16 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2023-10-17 09:15:00 | 313.70 | 2023-10-23 13:15:00 | 304.04 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2023-10-17 09:45:00 | 313.22 | 2023-10-23 13:15:00 | 304.04 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2023-10-20 09:15:00 | 314.80 | 2023-10-23 13:15:00 | 304.04 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2023-10-20 10:00:00 | 313.00 | 2023-10-23 13:15:00 | 304.04 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2023-11-03 11:15:00 | 309.41 | 2023-11-06 10:15:00 | 312.67 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2023-11-03 13:00:00 | 309.60 | 2023-11-06 10:15:00 | 312.67 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2023-11-06 11:45:00 | 307.80 | 2023-11-06 12:15:00 | 312.90 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2023-11-06 15:15:00 | 309.60 | 2023-11-21 15:15:00 | 314.00 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2023-11-07 09:15:00 | 308.08 | 2023-11-21 15:15:00 | 314.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2023-11-12 18:30:00 | 308.40 | 2023-11-21 15:15:00 | 314.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2023-11-17 10:15:00 | 308.11 | 2023-11-21 15:15:00 | 314.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2023-11-20 09:30:00 | 308.87 | 2023-11-21 15:15:00 | 314.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2023-11-28 10:15:00 | 309.38 | 2023-11-28 11:15:00 | 313.02 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2023-11-28 14:15:00 | 309.03 | 2023-12-04 10:15:00 | 312.37 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2023-11-28 15:15:00 | 309.20 | 2023-12-04 10:15:00 | 312.37 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2023-11-29 14:45:00 | 309.40 | 2023-12-04 10:15:00 | 312.37 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2023-12-01 14:15:00 | 305.84 | 2023-12-04 10:15:00 | 312.37 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2023-12-01 15:00:00 | 308.70 | 2023-12-04 10:15:00 | 312.37 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2023-12-04 09:45:00 | 308.27 | 2023-12-04 10:15:00 | 312.37 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2023-12-06 09:15:00 | 308.62 | 2023-12-07 10:15:00 | 311.47 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2023-12-12 13:15:00 | 307.59 | 2023-12-14 13:15:00 | 313.41 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2023-12-13 09:15:00 | 307.33 | 2023-12-14 13:15:00 | 313.41 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2023-12-14 10:00:00 | 307.46 | 2023-12-14 13:15:00 | 313.41 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2023-12-14 12:15:00 | 307.21 | 2023-12-14 13:15:00 | 313.41 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-01-18 13:45:00 | 323.17 | 2024-01-23 09:15:00 | 319.31 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-01-18 15:15:00 | 324.00 | 2024-01-23 09:15:00 | 319.31 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-01-23 09:15:00 | 323.39 | 2024-01-23 09:15:00 | 319.31 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-01-24 13:30:00 | 323.62 | 2024-01-25 10:15:00 | 319.62 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-01-25 09:15:00 | 322.19 | 2024-01-25 10:15:00 | 319.62 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-01-30 10:00:00 | 324.00 | 2024-02-01 15:15:00 | 317.20 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-01-30 15:15:00 | 321.40 | 2024-02-01 15:15:00 | 317.20 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-02-01 10:00:00 | 321.24 | 2024-02-01 15:15:00 | 317.20 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-02-06 09:15:00 | 322.47 | 2024-02-12 09:15:00 | 315.80 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-02-06 13:00:00 | 320.80 | 2024-02-12 09:15:00 | 315.80 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-02-06 15:00:00 | 321.04 | 2024-02-12 09:15:00 | 315.80 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-02-07 09:15:00 | 321.41 | 2024-02-12 09:15:00 | 315.80 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-03-06 10:15:00 | 316.22 | 2024-03-11 12:15:00 | 322.39 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-03-07 09:30:00 | 315.99 | 2024-03-11 12:15:00 | 322.39 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-03-07 10:30:00 | 315.27 | 2024-03-11 12:15:00 | 322.39 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-03-07 12:45:00 | 316.20 | 2024-03-11 12:15:00 | 322.39 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-03-11 14:15:00 | 319.01 | 2024-03-13 11:15:00 | 303.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 15:00:00 | 317.32 | 2024-03-13 11:15:00 | 301.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 14:15:00 | 319.01 | 2024-04-03 09:15:00 | 309.57 | STOP_HIT | 0.50 | 2.96% |
| SELL | retest2 | 2024-03-11 15:00:00 | 317.32 | 2024-04-03 09:15:00 | 309.57 | STOP_HIT | 0.50 | 2.44% |
| SELL | retest2 | 2024-04-15 09:15:00 | 314.26 | 2024-04-16 09:15:00 | 324.00 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2024-11-08 14:00:00 | 394.14 | 2024-11-13 09:15:00 | 374.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 14:00:00 | 394.14 | 2024-11-14 12:15:00 | 389.51 | STOP_HIT | 0.50 | 1.17% |
| SELL | retest2 | 2024-11-11 12:15:00 | 379.75 | 2024-11-28 11:15:00 | 406.18 | STOP_HIT | 1.00 | -6.96% |
| SELL | retest2 | 2024-11-27 11:15:00 | 394.36 | 2024-11-28 11:15:00 | 406.18 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2024-12-19 09:45:00 | 394.20 | 2024-12-30 14:15:00 | 374.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-19 09:45:00 | 394.20 | 2024-12-31 14:15:00 | 396.80 | STOP_HIT | 0.50 | -0.66% |
| SELL | retest2 | 2025-01-02 09:45:00 | 394.08 | 2025-01-02 14:15:00 | 402.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-01-06 09:15:00 | 391.83 | 2025-01-13 09:15:00 | 374.38 | PARTIAL | 0.50 | 4.45% |
| SELL | retest2 | 2025-01-07 14:30:00 | 394.08 | 2025-01-13 11:15:00 | 372.24 | PARTIAL | 0.50 | 5.54% |
| SELL | retest2 | 2025-01-07 15:00:00 | 393.54 | 2025-01-13 11:15:00 | 373.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 09:15:00 | 391.83 | 2025-01-27 09:15:00 | 352.65 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-07 14:30:00 | 394.08 | 2025-01-27 09:15:00 | 354.67 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-07 15:00:00 | 393.54 | 2025-01-27 09:15:00 | 354.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 348.70 | 2025-02-14 10:15:00 | 331.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 11:45:00 | 348.40 | 2025-02-14 10:15:00 | 330.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 13:00:00 | 349.01 | 2025-02-14 10:15:00 | 331.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 14:00:00 | 349.95 | 2025-02-14 10:15:00 | 332.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 348.70 | 2025-02-27 12:15:00 | 314.95 | TARGET_HIT | 0.50 | 9.68% |
| SELL | retest2 | 2025-02-11 11:45:00 | 348.40 | 2025-02-28 09:15:00 | 313.83 | TARGET_HIT | 0.50 | 9.92% |
| SELL | retest2 | 2025-02-11 13:00:00 | 349.01 | 2025-02-28 09:15:00 | 313.56 | TARGET_HIT | 0.50 | 10.16% |
| SELL | retest2 | 2025-02-11 14:00:00 | 349.95 | 2025-02-28 09:15:00 | 314.11 | TARGET_HIT | 0.50 | 10.24% |
| SELL | retest2 | 2025-03-28 11:15:00 | 339.45 | 2025-04-08 13:15:00 | 350.00 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-04-02 13:00:00 | 340.75 | 2025-04-08 13:15:00 | 350.00 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-04-03 11:00:00 | 340.30 | 2025-04-08 13:15:00 | 350.00 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-05-06 13:00:00 | 340.60 | 2025-05-12 09:15:00 | 354.60 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2025-08-18 09:45:00 | 395.48 | 2025-09-01 10:15:00 | 435.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-26 11:30:00 | 394.98 | 2025-09-01 10:15:00 | 434.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-26 12:45:00 | 397.34 | 2025-09-01 10:15:00 | 437.07 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2025-12-23 10:15:00 | 426.45 | 2025-12-31 09:15:00 | 460.10 | STOP_HIT | 1.00 | -7.89% |
| SELL | retest1 | 2025-12-24 13:00:00 | 426.85 | 2025-12-31 09:15:00 | 460.10 | STOP_HIT | 1.00 | -7.79% |
| SELL | retest1 | 2025-12-24 15:00:00 | 426.20 | 2025-12-31 09:15:00 | 460.10 | STOP_HIT | 1.00 | -7.95% |
| SELL | retest1 | 2025-12-26 09:15:00 | 424.70 | 2025-12-31 09:15:00 | 460.10 | STOP_HIT | 1.00 | -8.34% |
| SELL | retest2 | 2026-02-02 09:15:00 | 428.45 | 2026-02-04 10:15:00 | 407.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-02 09:15:00 | 428.45 | 2026-02-26 11:15:00 | 385.61 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-13 13:30:00 | 431.50 | 2026-03-13 14:15:00 | 409.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 13:30:00 | 431.50 | 2026-03-13 14:15:00 | 401.20 | STOP_HIT | 0.50 | 7.02% |
| SELL | retest2 | 2026-03-24 13:30:00 | 436.35 | 2026-03-25 09:15:00 | 446.10 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2026-03-27 14:15:00 | 435.80 | 2026-03-27 14:15:00 | 448.05 | STOP_HIT | 1.00 | -2.81% |
