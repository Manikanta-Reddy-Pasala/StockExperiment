# Laurus Labs Ltd. (LAURUSLABS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1225.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 3 |
| ALERT3 | 31 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 41 |
| PARTIAL | 0 |
| TARGET_HIT | 10 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 44 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 31
- **Target hits / Stop hits / Partials:** 10 / 34 / 0
- **Avg / median % per leg:** -0.08% / -2.20%
- **Sum % (uncompounded):** -3.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 13 | 32.5% | 10 | 30 | 0 | -0.02% | -0.7% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.25% | -9.7% |
| BUY @ 3rd Alert (retest2) | 37 | 13 | 35.1% | 10 | 27 | 0 | 0.24% | 9.0% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.75% | -3.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.75% | -3.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.25% | -9.7% |
| retest2 (combined) | 41 | 13 | 31.7% | 10 | 31 | 0 | 0.15% | 6.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 15:15:00 | 359.55 | 382.04 | 382.05 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 11:15:00 | 395.05 | 378.69 | 378.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-18 09:15:00 | 399.45 | 380.22 | 379.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 11:15:00 | 410.60 | 412.03 | 401.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 09:15:00 | 407.55 | 411.98 | 401.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 407.55 | 411.98 | 401.44 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 14:15:00 | 392.10 | 399.41 | 399.43 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-02 10:15:00 | 410.55 | 399.51 | 399.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 11:15:00 | 415.05 | 399.67 | 399.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 14:15:00 | 423.60 | 425.00 | 415.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 15:00:00 | 423.60 | 425.00 | 415.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 418.10 | 424.93 | 415.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 11:15:00 | 429.60 | 424.91 | 415.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-07 13:15:00 | 427.75 | 431.15 | 421.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 11:00:00 | 427.55 | 431.34 | 422.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 13:45:00 | 426.00 | 436.02 | 429.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 424.20 | 435.74 | 429.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 418.35 | 435.74 | 429.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 409.90 | 435.48 | 429.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 409.90 | 435.48 | 429.12 | SL hit (close<static) qty=1.00 sl=414.60 alert=retest2 |

### Cycle 5 — SELL (started 2024-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 13:15:00 | 433.90 | 439.41 | 439.41 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 15:15:00 | 446.55 | 439.45 | 439.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 448.80 | 439.54 | 439.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 09:15:00 | 473.80 | 474.76 | 461.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-19 10:00:00 | 473.80 | 474.76 | 461.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 470.50 | 474.33 | 461.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 09:15:00 | 476.10 | 473.74 | 461.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 14:30:00 | 473.60 | 473.58 | 461.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 11:15:00 | 460.30 | 473.04 | 462.22 | SL hit (close<static) qty=1.00 sl=461.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-05 09:15:00 | 546.80 | 561.36 | 561.42 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 15:15:00 | 567.20 | 561.49 | 561.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 586.30 | 561.73 | 561.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 14:15:00 | 558.50 | 563.69 | 562.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 14:15:00 | 558.50 | 563.69 | 562.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 558.50 | 563.69 | 562.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:30:00 | 559.30 | 563.69 | 562.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 560.95 | 563.67 | 562.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 565.25 | 563.67 | 562.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 568.45 | 563.87 | 562.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:45:00 | 566.35 | 563.87 | 562.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 571.05 | 590.84 | 579.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 571.05 | 590.84 | 579.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 577.50 | 590.71 | 579.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 09:15:00 | 588.20 | 585.14 | 578.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 10:15:00 | 586.05 | 585.13 | 578.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-21 15:15:00 | 644.65 | 596.41 | 585.22 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 12:15:00 | 976.95 | 1006.22 | 1006.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 13:15:00 | 966.30 | 1005.83 | 1006.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 14:15:00 | 1013.25 | 1004.63 | 1005.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 14:15:00 | 1013.25 | 1004.63 | 1005.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 1013.25 | 1004.63 | 1005.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 15:00:00 | 1013.25 | 1004.63 | 1005.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 1014.20 | 1004.73 | 1005.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 1001.00 | 1004.73 | 1005.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 1012.90 | 1004.99 | 1005.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 1005.50 | 1005.57 | 1005.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 11:00:00 | 1009.60 | 1005.60 | 1005.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:00:00 | 1011.10 | 1005.96 | 1006.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:45:00 | 1011.60 | 1006.09 | 1006.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 10:15:00 | 1017.00 | 1006.20 | 1006.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 1017.00 | 1006.20 | 1006.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 1024.10 | 1007.29 | 1006.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 11:15:00 | 1026.40 | 1027.67 | 1018.26 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 13:00:00 | 1030.60 | 1027.70 | 1018.32 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 13:45:00 | 1029.90 | 1027.82 | 1018.43 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 15:15:00 | 1031.20 | 1027.83 | 1018.48 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 997.10 | 1028.98 | 1019.81 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 997.10 | 1028.98 | 1019.81 | SL hit (close<ema400) qty=1.00 sl=1019.81 alert=retest1 |

### Cycle 11 — SELL (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 09:15:00 | 971.40 | 1014.23 | 1014.44 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2026-04-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 15:15:00 | 1059.00 | 1013.43 | 1013.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 10:15:00 | 1061.05 | 1014.34 | 1013.82 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-26 11:15:00 | 429.60 | 2024-06-04 10:15:00 | 409.90 | STOP_HIT | 1.00 | -4.59% |
| BUY | retest2 | 2024-05-07 13:15:00 | 427.75 | 2024-06-04 10:15:00 | 409.90 | STOP_HIT | 1.00 | -4.17% |
| BUY | retest2 | 2024-05-13 11:00:00 | 427.55 | 2024-06-04 10:15:00 | 409.90 | STOP_HIT | 1.00 | -4.13% |
| BUY | retest2 | 2024-06-03 13:45:00 | 426.00 | 2024-06-04 10:15:00 | 409.90 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2024-06-06 09:15:00 | 427.80 | 2024-06-25 10:15:00 | 427.90 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2024-06-06 10:00:00 | 432.45 | 2024-06-25 10:15:00 | 427.90 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-06-06 14:45:00 | 427.15 | 2024-06-27 12:15:00 | 423.30 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-06-18 14:45:00 | 427.05 | 2024-06-27 12:15:00 | 423.30 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-06-19 11:15:00 | 432.80 | 2024-06-27 12:15:00 | 423.30 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-06-21 09:15:00 | 434.85 | 2024-06-27 12:15:00 | 423.30 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2024-07-02 11:00:00 | 438.70 | 2024-07-08 09:15:00 | 482.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-23 12:30:00 | 433.35 | 2024-08-06 13:15:00 | 423.70 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2024-09-23 09:15:00 | 476.10 | 2024-09-25 11:15:00 | 460.30 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2024-09-23 14:30:00 | 473.60 | 2024-09-25 11:15:00 | 460.30 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2024-10-11 10:15:00 | 473.25 | 2024-10-22 09:15:00 | 459.80 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2024-10-11 11:45:00 | 474.60 | 2024-10-22 09:15:00 | 459.80 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2024-10-18 10:15:00 | 471.75 | 2024-10-22 09:15:00 | 459.80 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2024-10-18 10:45:00 | 475.90 | 2024-10-22 09:15:00 | 459.80 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2024-10-25 11:30:00 | 475.30 | 2024-11-25 09:15:00 | 522.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-28 10:15:00 | 473.85 | 2024-11-25 09:15:00 | 521.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-15 12:30:00 | 564.15 | 2025-01-27 09:15:00 | 523.80 | STOP_HIT | 1.00 | -7.15% |
| BUY | retest2 | 2025-01-16 09:30:00 | 568.40 | 2025-01-27 09:15:00 | 523.80 | STOP_HIT | 1.00 | -7.85% |
| BUY | retest2 | 2025-01-17 12:15:00 | 563.20 | 2025-01-27 09:15:00 | 523.80 | STOP_HIT | 1.00 | -7.00% |
| BUY | retest2 | 2025-01-22 14:15:00 | 563.15 | 2025-01-27 09:15:00 | 523.80 | STOP_HIT | 1.00 | -6.99% |
| BUY | retest2 | 2025-04-11 09:15:00 | 588.20 | 2025-04-21 15:15:00 | 644.65 | TARGET_HIT | 1.00 | 9.60% |
| BUY | retest2 | 2025-04-11 10:15:00 | 586.05 | 2025-04-22 09:15:00 | 647.02 | TARGET_HIT | 1.00 | 10.40% |
| BUY | retest2 | 2025-05-09 12:00:00 | 585.35 | 2025-05-22 09:15:00 | 589.30 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2025-05-09 14:00:00 | 584.85 | 2025-05-22 09:15:00 | 589.30 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2025-05-13 09:15:00 | 605.50 | 2025-05-22 09:15:00 | 589.30 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-05-13 11:45:00 | 599.50 | 2025-05-22 09:15:00 | 589.30 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-05-14 09:15:00 | 599.30 | 2025-06-05 10:15:00 | 643.89 | TARGET_HIT | 1.00 | 7.44% |
| BUY | retest2 | 2025-05-21 09:15:00 | 600.00 | 2025-06-05 10:15:00 | 643.34 | TARGET_HIT | 1.00 | 7.22% |
| BUY | retest2 | 2025-05-26 10:15:00 | 598.55 | 2025-06-09 10:15:00 | 658.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-27 10:45:00 | 598.85 | 2025-06-09 10:15:00 | 658.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-27 11:45:00 | 597.70 | 2025-06-09 10:15:00 | 657.47 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 1005.50 | 2026-02-16 10:15:00 | 1017.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-02-13 11:00:00 | 1009.60 | 2026-02-16 10:15:00 | 1017.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-02-13 15:00:00 | 1011.10 | 2026-02-16 10:15:00 | 1017.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-02-16 09:45:00 | 1011.60 | 2026-02-16 10:15:00 | 1017.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2026-03-04 13:00:00 | 1030.60 | 2026-03-09 09:15:00 | 997.10 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest1 | 2026-03-04 13:45:00 | 1029.90 | 2026-03-09 09:15:00 | 997.10 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest1 | 2026-03-04 15:15:00 | 1031.20 | 2026-03-09 09:15:00 | 997.10 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2026-03-09 12:30:00 | 1011.10 | 2026-03-16 12:15:00 | 966.00 | STOP_HIT | 1.00 | -4.46% |
| BUY | retest2 | 2026-03-09 13:30:00 | 1013.90 | 2026-03-16 12:15:00 | 966.00 | STOP_HIT | 1.00 | -4.72% |
