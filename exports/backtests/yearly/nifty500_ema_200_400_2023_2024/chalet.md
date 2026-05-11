# Chalet Hotels Ltd. (CHALET)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 787.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 3 |
| ALERT3 | 32 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 38 |
| PARTIAL | 1 |
| TARGET_HIT | 7 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 41 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 32
- **Target hits / Stop hits / Partials:** 7 / 33 / 1
- **Avg / median % per leg:** -0.56% / -1.94%
- **Sum % (uncompounded):** -22.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 7 | 22.6% | 7 | 24 | 0 | 0.06% | 1.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 31 | 7 | 22.6% | 7 | 24 | 0 | 0.06% | 1.8% |
| SELL (all) | 10 | 2 | 20.0% | 0 | 9 | 1 | -2.46% | -24.6% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -9.64% | -19.3% |
| SELL @ 3rd Alert (retest2) | 8 | 2 | 25.0% | 0 | 7 | 1 | -0.67% | -5.3% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -9.64% | -19.3% |
| retest2 (combined) | 39 | 9 | 23.1% | 7 | 31 | 1 | -0.09% | -3.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 10:15:00 | 760.15 | 812.45 | 812.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 724.95 | 802.72 | 807.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-07 09:15:00 | 801.75 | 796.47 | 803.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-07 09:15:00 | 801.75 | 796.47 | 803.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 801.75 | 796.47 | 803.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:30:00 | 799.20 | 796.47 | 803.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 800.80 | 796.51 | 803.62 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 12:15:00 | 834.90 | 809.48 | 809.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 14:15:00 | 836.05 | 811.43 | 810.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 14:15:00 | 813.30 | 815.57 | 812.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 14:15:00 | 813.30 | 815.57 | 812.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 813.30 | 815.57 | 812.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 15:00:00 | 813.30 | 815.57 | 812.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 820.00 | 815.62 | 812.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 09:15:00 | 823.15 | 815.62 | 812.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 12:15:00 | 820.65 | 815.78 | 812.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 13:15:00 | 804.00 | 815.68 | 812.94 | SL hit (close<static) qty=1.00 sl=809.20 alert=retest2 |

### Cycle 3 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 801.70 | 816.01 | 816.04 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 15:15:00 | 858.00 | 816.35 | 816.20 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 09:15:00 | 795.40 | 817.56 | 817.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 13:15:00 | 788.80 | 816.54 | 817.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 801.75 | 799.15 | 806.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-26 10:15:00 | 808.00 | 799.15 | 806.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 810.35 | 799.26 | 806.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:45:00 | 811.60 | 799.26 | 806.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 811.40 | 799.38 | 806.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 13:15:00 | 806.90 | 800.91 | 807.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 14:15:00 | 813.45 | 801.14 | 807.31 | SL hit (close>static) qty=1.00 sl=813.10 alert=retest2 |

### Cycle 6 — BUY (started 2024-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 15:15:00 | 875.00 | 812.60 | 812.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 881.45 | 813.28 | 812.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 09:15:00 | 849.45 | 852.03 | 836.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-19 10:00:00 | 849.45 | 852.03 | 836.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 848.45 | 867.79 | 850.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:00:00 | 848.45 | 867.79 | 850.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 11:15:00 | 856.55 | 867.68 | 850.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 10:45:00 | 862.30 | 866.19 | 850.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 13:30:00 | 869.50 | 866.12 | 850.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 15:00:00 | 863.75 | 874.33 | 858.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 10:30:00 | 863.60 | 873.83 | 858.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 870.85 | 873.80 | 858.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-22 09:15:00 | 847.00 | 872.97 | 859.39 | SL hit (close<static) qty=1.00 sl=847.60 alert=retest2 |

### Cycle 7 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 11:15:00 | 784.10 | 904.43 | 904.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 09:15:00 | 760.75 | 883.13 | 893.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 15:15:00 | 754.25 | 746.20 | 792.63 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-03 09:15:00 | 710.45 | 746.20 | 792.63 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-03 13:45:00 | 731.00 | 745.21 | 790.99 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 790.05 | 750.17 | 787.38 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-07 14:15:00 | 790.05 | 750.17 | 787.38 | SL hit (close>ema400) qty=1.00 sl=787.38 alert=retest1 |

### Cycle 8 — BUY (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 13:15:00 | 863.00 | 802.44 | 802.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 873.95 | 803.15 | 802.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 788.75 | 805.71 | 803.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 788.75 | 805.71 | 803.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 788.75 | 805.71 | 803.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 15:00:00 | 815.00 | 805.27 | 803.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 09:15:00 | 825.60 | 805.32 | 803.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 09:45:00 | 811.85 | 806.57 | 804.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 10:00:00 | 813.00 | 806.42 | 804.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 810.00 | 810.52 | 806.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 822.00 | 810.52 | 806.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 12:00:00 | 819.45 | 814.07 | 809.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 10:00:00 | 819.95 | 814.50 | 809.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 10:45:00 | 819.25 | 814.56 | 809.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 815.00 | 814.57 | 809.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-28 15:00:00 | 815.00 | 814.57 | 809.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 801.55 | 814.56 | 810.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 801.55 | 814.56 | 810.07 | SL hit (close<static) qty=1.00 sl=803.30 alert=retest2 |

### Cycle 9 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 924.10 | 960.31 | 960.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 909.05 | 958.86 | 959.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 908.75 | 908.42 | 926.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 15:00:00 | 908.75 | 908.42 | 926.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 896.05 | 884.78 | 901.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:15:00 | 905.30 | 884.78 | 901.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 910.00 | 885.04 | 901.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:45:00 | 913.30 | 885.04 | 901.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 910.05 | 885.28 | 902.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 11:30:00 | 906.45 | 885.39 | 901.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 09:15:00 | 861.13 | 884.43 | 900.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 888.75 | 884.48 | 899.94 | SL hit (close>ema200) qty=0.50 sl=884.48 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-27 09:15:00 | 823.15 | 2024-06-27 13:15:00 | 804.00 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2024-06-27 12:15:00 | 820.65 | 2024-06-27 13:15:00 | 804.00 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-07-01 12:30:00 | 826.80 | 2024-07-02 09:15:00 | 909.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-22 15:15:00 | 820.50 | 2024-07-23 09:15:00 | 801.70 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-08-27 13:15:00 | 806.90 | 2024-08-27 14:15:00 | 813.45 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-08-28 15:00:00 | 806.10 | 2024-08-28 15:15:00 | 818.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-08-29 14:30:00 | 806.55 | 2024-08-30 10:15:00 | 818.25 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-08-30 10:00:00 | 808.95 | 2024-08-30 10:15:00 | 818.25 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-10-08 10:45:00 | 862.30 | 2024-10-22 09:15:00 | 847.00 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-10-08 13:30:00 | 869.50 | 2024-10-22 09:15:00 | 847.00 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2024-10-17 15:00:00 | 863.75 | 2024-10-22 09:15:00 | 847.00 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2024-10-18 10:30:00 | 863.60 | 2024-10-22 09:15:00 | 847.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-10-25 13:00:00 | 880.60 | 2024-10-25 14:15:00 | 844.00 | STOP_HIT | 1.00 | -4.16% |
| BUY | retest2 | 2024-10-31 10:15:00 | 887.40 | 2024-11-05 11:15:00 | 855.10 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest2 | 2024-10-31 14:15:00 | 873.65 | 2024-11-05 11:15:00 | 855.10 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-11-01 18:00:00 | 891.90 | 2024-11-05 11:15:00 | 855.10 | STOP_HIT | 1.00 | -4.13% |
| BUY | retest2 | 2024-11-05 15:15:00 | 863.40 | 2024-11-13 09:15:00 | 832.65 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2024-11-25 09:15:00 | 865.05 | 2024-12-09 12:15:00 | 951.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-13 13:15:00 | 859.10 | 2025-01-14 14:15:00 | 847.10 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-01-13 13:45:00 | 858.35 | 2025-01-14 14:15:00 | 847.10 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest1 | 2025-03-03 09:15:00 | 710.45 | 2025-03-07 14:15:00 | 790.05 | STOP_HIT | 1.00 | -11.20% |
| SELL | retest1 | 2025-03-03 13:45:00 | 731.00 | 2025-03-07 14:15:00 | 790.05 | STOP_HIT | 1.00 | -8.08% |
| SELL | retest2 | 2025-03-13 09:15:00 | 765.50 | 2025-03-19 09:15:00 | 799.15 | STOP_HIT | 1.00 | -4.40% |
| SELL | retest2 | 2025-03-18 10:30:00 | 775.95 | 2025-03-19 09:15:00 | 799.15 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2025-04-07 15:00:00 | 815.00 | 2025-04-30 09:15:00 | 801.55 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-04-08 09:15:00 | 825.60 | 2025-04-30 09:15:00 | 801.55 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest2 | 2025-04-09 09:45:00 | 811.85 | 2025-04-30 09:15:00 | 801.55 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-04-11 10:00:00 | 813.00 | 2025-04-30 09:15:00 | 801.55 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-04-21 09:15:00 | 822.00 | 2025-05-08 14:15:00 | 801.80 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-04-25 12:00:00 | 819.45 | 2025-05-08 15:15:00 | 797.60 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-04-28 10:00:00 | 819.95 | 2025-05-09 09:15:00 | 781.15 | STOP_HIT | 1.00 | -4.73% |
| BUY | retest2 | 2025-04-28 10:45:00 | 819.25 | 2025-05-09 09:15:00 | 781.15 | STOP_HIT | 1.00 | -4.65% |
| BUY | retest2 | 2025-05-05 09:45:00 | 825.20 | 2025-05-09 09:15:00 | 781.15 | STOP_HIT | 1.00 | -5.34% |
| BUY | retest2 | 2025-05-05 15:00:00 | 821.10 | 2025-05-09 09:15:00 | 781.15 | STOP_HIT | 1.00 | -4.87% |
| BUY | retest2 | 2025-05-06 10:00:00 | 819.05 | 2025-05-13 09:15:00 | 896.50 | TARGET_HIT | 1.00 | 9.46% |
| BUY | retest2 | 2025-05-07 11:00:00 | 819.90 | 2025-05-13 09:15:00 | 893.04 | TARGET_HIT | 1.00 | 8.92% |
| BUY | retest2 | 2025-05-08 09:15:00 | 818.10 | 2025-05-13 09:15:00 | 894.30 | TARGET_HIT | 1.00 | 9.31% |
| BUY | retest2 | 2025-05-08 14:30:00 | 816.40 | 2025-05-14 09:15:00 | 908.16 | TARGET_HIT | 1.00 | 11.24% |
| BUY | retest2 | 2025-05-12 09:15:00 | 829.00 | 2025-05-14 11:15:00 | 911.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-02 11:30:00 | 906.45 | 2026-01-07 09:15:00 | 861.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-02 11:30:00 | 906.45 | 2026-01-07 10:15:00 | 888.75 | STOP_HIT | 0.50 | 1.95% |
