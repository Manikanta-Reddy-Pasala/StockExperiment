# REC Ltd. (RECLTD)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 359.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 5 |
| TARGET_HIT | 7 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 8
- **Target hits / Stop hits / Partials:** 7 / 8 / 5
- **Avg / median % per leg:** 3.66% / 5.00%
- **Sum % (uncompounded):** 73.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.21% | -12.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.21% | -12.8% |
| SELL (all) | 16 | 12 | 75.0% | 7 | 4 | 5 | 5.38% | 86.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 12 | 75.0% | 7 | 4 | 5 | 5.38% | 86.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 20 | 12 | 60.0% | 7 | 8 | 5 | 3.66% | 73.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 11:15:00 | 546.15 | 576.54 | 576.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 543.00 | 575.14 | 575.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 13:15:00 | 553.95 | 552.45 | 561.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-16 14:00:00 | 553.95 | 552.45 | 561.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 530.95 | 523.27 | 535.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:30:00 | 532.75 | 523.27 | 535.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 533.90 | 523.89 | 535.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:00:00 | 533.90 | 523.89 | 535.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 535.20 | 524.17 | 535.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:30:00 | 536.55 | 524.17 | 535.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 534.35 | 524.27 | 535.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:45:00 | 535.65 | 524.27 | 535.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 12:15:00 | 532.35 | 524.44 | 535.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 12:30:00 | 534.25 | 524.44 | 535.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 535.60 | 524.77 | 535.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:45:00 | 537.65 | 524.77 | 535.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 534.35 | 524.86 | 535.02 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 12:15:00 | 555.90 | 541.88 | 541.83 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 15:15:00 | 537.00 | 541.76 | 541.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 526.10 | 541.60 | 541.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 532.45 | 526.32 | 532.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 10:15:00 | 532.45 | 526.32 | 532.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 532.45 | 526.32 | 532.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 11:00:00 | 532.45 | 526.32 | 532.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 534.30 | 526.40 | 532.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 12:00:00 | 534.30 | 526.40 | 532.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 540.15 | 526.54 | 532.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:00:00 | 540.15 | 526.54 | 532.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 425.60 | 411.56 | 434.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:00:00 | 424.40 | 420.28 | 434.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 15:00:00 | 424.10 | 420.19 | 433.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 09:45:00 | 421.45 | 420.29 | 433.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:15:00 | 417.90 | 420.64 | 433.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 10:15:00 | 403.18 | 420.41 | 433.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 14:15:00 | 402.89 | 419.89 | 432.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 381.96 | 419.24 | 432.16 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 11:15:00 | 374.70 | 363.20 | 363.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 375.30 | 363.70 | 363.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 361.25 | 365.05 | 364.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 12:15:00 | 361.25 | 365.05 | 364.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 361.25 | 365.05 | 364.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 361.25 | 365.05 | 364.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 361.40 | 365.01 | 364.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:15:00 | 360.40 | 365.01 | 364.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 364.30 | 365.01 | 364.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 365.45 | 365.01 | 364.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 366.15 | 365.02 | 364.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 11:15:00 | 376.65 | 365.02 | 364.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:30:00 | 369.15 | 365.10 | 364.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 359.80 | 365.04 | 364.22 | SL hit (close<static) qty=1.00 sl=361.10 alert=retest2 |

### Cycle 5 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 343.55 | 364.04 | 364.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 338.80 | 358.06 | 360.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 10:15:00 | 346.35 | 345.71 | 352.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-18 10:45:00 | 346.20 | 345.71 | 352.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 338.55 | 333.64 | 342.97 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2026-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 13:15:00 | 384.00 | 348.81 | 348.74 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-04-01 11:00:00 | 424.40 | 2025-04-04 10:15:00 | 403.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-02 15:00:00 | 424.10 | 2025-04-04 14:15:00 | 402.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 11:00:00 | 424.40 | 2025-04-07 09:15:00 | 381.96 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-02 15:00:00 | 424.10 | 2025-04-07 09:15:00 | 381.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-03 09:45:00 | 421.45 | 2025-04-07 09:15:00 | 379.31 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-04 09:15:00 | 417.90 | 2025-04-07 09:15:00 | 376.11 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-25 11:45:00 | 421.10 | 2025-04-28 11:15:00 | 429.15 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-04-25 15:15:00 | 421.00 | 2025-04-28 11:15:00 | 429.15 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-04-30 14:00:00 | 421.15 | 2025-05-02 09:15:00 | 431.65 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-04-30 14:30:00 | 420.65 | 2025-05-02 09:15:00 | 431.65 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-05-02 11:15:00 | 423.40 | 2025-05-07 09:15:00 | 402.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 11:15:00 | 423.40 | 2025-05-09 09:15:00 | 381.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-06-09 12:45:00 | 425.80 | 2025-06-12 14:15:00 | 404.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 15:15:00 | 426.00 | 2025-06-12 14:15:00 | 404.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 12:45:00 | 425.80 | 2025-06-19 12:15:00 | 383.22 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-06-09 15:15:00 | 426.00 | 2025-06-19 12:15:00 | 383.40 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-01 11:15:00 | 376.65 | 2026-02-01 14:15:00 | 359.80 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest2 | 2026-02-01 12:30:00 | 369.15 | 2026-02-01 14:15:00 | 359.80 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2026-02-02 09:15:00 | 367.00 | 2026-02-02 10:15:00 | 358.35 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2026-02-03 09:15:00 | 369.90 | 2026-02-09 09:15:00 | 357.10 | STOP_HIT | 1.00 | -3.46% |
