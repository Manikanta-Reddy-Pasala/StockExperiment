# Zydus Lifesciences Ltd. (ZYDUSLIFE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 939.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 25 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 48 |
| PARTIAL | 22 |
| TARGET_HIT | 9 |
| STOP_HIT | 38 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 69 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 46 / 23
- **Target hits / Stop hits / Partials:** 9 / 38 / 22
- **Avg / median % per leg:** 2.63% / 2.04%
- **Sum % (uncompounded):** 181.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 5 | 33.3% | 5 | 10 | 0 | 2.37% | 35.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 5 | 33.3% | 5 | 10 | 0 | 2.37% | 35.5% |
| SELL (all) | 54 | 41 | 75.9% | 4 | 28 | 22 | 2.71% | 146.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 54 | 41 | 75.9% | 4 | 28 | 22 | 2.71% | 146.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 69 | 46 | 66.7% | 9 | 38 | 22 | 2.63% | 181.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 15:15:00 | 589.00 | 609.85 | 609.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 12:15:00 | 586.95 | 609.14 | 609.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-07 09:15:00 | 595.50 | 591.38 | 598.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 11:15:00 | 596.40 | 591.47 | 598.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 11:15:00 | 596.40 | 591.47 | 598.51 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 13:15:00 | 637.80 | 603.95 | 603.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-20 10:15:00 | 642.30 | 605.30 | 604.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 14:15:00 | 962.50 | 967.28 | 907.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-12 15:00:00 | 962.50 | 967.28 | 907.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 10:15:00 | 931.25 | 958.71 | 914.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 11:45:00 | 932.10 | 958.47 | 914.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 10:30:00 | 931.80 | 957.01 | 915.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 11:30:00 | 933.60 | 956.77 | 915.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 12:45:00 | 933.05 | 956.54 | 915.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-06 15:15:00 | 1025.31 | 962.69 | 927.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 12:15:00 | 1044.95 | 1134.38 | 1134.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 1041.10 | 1116.34 | 1124.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 983.30 | 981.10 | 1015.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-03 09:45:00 | 984.45 | 981.10 | 1015.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 986.05 | 976.51 | 994.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 11:30:00 | 979.20 | 976.62 | 994.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 09:45:00 | 983.10 | 975.63 | 992.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-07 10:15:00 | 1009.60 | 975.96 | 992.87 | SL hit (close>static) qty=1.00 sl=997.40 alert=retest2 |

### Cycle 4 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 931.80 | 901.81 | 901.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 938.50 | 903.68 | 902.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 965.55 | 966.52 | 946.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 11:00:00 | 965.55 | 966.52 | 946.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 957.10 | 969.53 | 953.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 11:15:00 | 960.70 | 969.53 | 953.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 10:15:00 | 960.90 | 968.95 | 953.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 11:30:00 | 960.80 | 968.77 | 953.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 951.40 | 972.52 | 958.69 | SL hit (close<static) qty=1.00 sl=951.55 alert=retest2 |

### Cycle 5 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 930.00 | 991.70 | 991.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 928.20 | 973.13 | 981.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 925.00 | 924.65 | 940.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-06 09:45:00 | 925.65 | 924.65 | 940.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 939.35 | 924.98 | 939.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 939.35 | 924.98 | 939.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 933.05 | 925.06 | 939.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 11:30:00 | 928.25 | 925.09 | 939.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 881.84 | 922.53 | 937.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 901.10 | 897.38 | 915.35 | SL hit (close>ema200) qty=0.50 sl=897.38 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 931.50 | 906.06 | 906.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 957.45 | 909.17 | 907.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 913.15 | 914.88 | 910.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 09:15:00 | 913.15 | 914.88 | 910.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 913.15 | 914.88 | 910.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 10:15:00 | 925.05 | 912.61 | 910.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-24 11:45:00 | 932.10 | 2024-05-06 15:15:00 | 1025.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-25 10:30:00 | 931.80 | 2024-05-06 15:15:00 | 1024.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-25 11:30:00 | 933.60 | 2024-05-06 15:15:00 | 1026.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-25 12:45:00 | 933.05 | 2024-05-06 15:15:00 | 1026.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 13:15:00 | 1004.90 | 2024-06-14 10:15:00 | 1105.39 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-03 11:30:00 | 979.20 | 2025-01-07 10:15:00 | 1009.60 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-01-07 09:45:00 | 983.10 | 2025-01-07 10:15:00 | 1009.60 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-01-08 12:15:00 | 983.35 | 2025-01-09 09:15:00 | 1012.95 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2025-01-08 12:45:00 | 982.35 | 2025-01-09 09:15:00 | 1012.95 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-01-13 09:15:00 | 988.50 | 2025-01-17 11:15:00 | 995.10 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-01-13 09:45:00 | 988.65 | 2025-01-27 09:15:00 | 939.07 | PARTIAL | 0.50 | 5.01% |
| SELL | retest2 | 2025-01-15 09:15:00 | 988.25 | 2025-01-27 09:15:00 | 939.22 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2025-01-15 11:00:00 | 990.95 | 2025-01-27 09:15:00 | 938.84 | PARTIAL | 0.50 | 5.26% |
| SELL | retest2 | 2025-01-16 10:30:00 | 980.40 | 2025-01-27 09:15:00 | 941.40 | PARTIAL | 0.50 | 3.98% |
| SELL | retest2 | 2025-01-22 09:15:00 | 978.15 | 2025-01-27 13:15:00 | 929.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-13 09:45:00 | 988.65 | 2025-01-28 09:15:00 | 891.86 | TARGET_HIT | 0.50 | 9.79% |
| SELL | retest2 | 2025-01-24 09:15:00 | 961.00 | 2025-01-28 09:15:00 | 912.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-15 09:15:00 | 988.25 | 2025-02-01 14:15:00 | 970.75 | STOP_HIT | 0.50 | 1.77% |
| SELL | retest2 | 2025-01-15 11:00:00 | 990.95 | 2025-02-01 14:15:00 | 970.75 | STOP_HIT | 0.50 | 2.04% |
| SELL | retest2 | 2025-01-16 10:30:00 | 980.40 | 2025-02-01 14:15:00 | 970.75 | STOP_HIT | 0.50 | 0.98% |
| SELL | retest2 | 2025-01-22 09:15:00 | 978.15 | 2025-02-01 14:15:00 | 970.75 | STOP_HIT | 0.50 | 0.76% |
| SELL | retest2 | 2025-01-24 09:15:00 | 961.00 | 2025-02-01 14:15:00 | 970.75 | STOP_HIT | 0.50 | -1.01% |
| SELL | retest2 | 2025-02-10 09:15:00 | 981.75 | 2025-02-12 09:15:00 | 932.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 981.75 | 2025-02-19 09:15:00 | 883.58 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-25 09:15:00 | 907.00 | 2025-04-04 09:15:00 | 861.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 09:45:00 | 913.20 | 2025-04-04 09:15:00 | 867.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 09:15:00 | 907.00 | 2025-04-07 09:15:00 | 816.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-03 09:45:00 | 913.20 | 2025-04-07 09:15:00 | 821.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-19 10:45:00 | 917.95 | 2025-05-21 09:15:00 | 872.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-19 10:45:00 | 917.95 | 2025-05-21 09:15:00 | 893.70 | STOP_HIT | 0.50 | 2.64% |
| SELL | retest2 | 2025-05-26 10:15:00 | 918.25 | 2025-06-03 10:15:00 | 931.80 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-07-22 11:15:00 | 960.70 | 2025-08-01 09:15:00 | 951.40 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-07-23 10:15:00 | 960.90 | 2025-08-01 09:15:00 | 951.40 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-23 11:30:00 | 960.80 | 2025-08-01 09:15:00 | 951.40 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-08-04 13:30:00 | 961.55 | 2025-08-05 09:15:00 | 950.40 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-08-05 09:15:00 | 960.60 | 2025-08-05 09:15:00 | 950.40 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-08-05 12:00:00 | 960.25 | 2025-08-05 12:15:00 | 954.50 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-08-05 15:00:00 | 960.80 | 2025-08-06 09:15:00 | 936.60 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-08-12 12:30:00 | 960.85 | 2025-08-12 13:15:00 | 954.55 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-08-13 11:00:00 | 966.30 | 2025-11-06 13:15:00 | 938.00 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-11-06 13:00:00 | 963.25 | 2025-11-06 13:15:00 | 938.00 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2026-01-07 11:30:00 | 928.25 | 2026-01-12 09:15:00 | 881.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 11:30:00 | 928.25 | 2026-02-03 09:15:00 | 901.10 | STOP_HIT | 0.50 | 2.92% |
| SELL | retest2 | 2026-02-26 11:30:00 | 931.50 | 2026-03-10 09:15:00 | 917.55 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest2 | 2026-02-26 12:00:00 | 931.40 | 2026-03-16 10:15:00 | 884.92 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2026-02-27 09:30:00 | 928.45 | 2026-03-16 10:15:00 | 884.83 | PARTIAL | 0.50 | 4.70% |
| SELL | retest2 | 2026-03-09 09:15:00 | 892.45 | 2026-03-16 10:15:00 | 882.03 | PARTIAL | 0.50 | 1.17% |
| SELL | retest2 | 2026-03-13 10:15:00 | 904.95 | 2026-03-23 14:15:00 | 859.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 11:00:00 | 905.20 | 2026-03-23 14:15:00 | 859.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 12:15:00 | 903.50 | 2026-03-23 14:15:00 | 858.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 12:00:00 | 931.40 | 2026-03-25 09:15:00 | 901.65 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2026-02-27 09:30:00 | 928.45 | 2026-03-25 09:15:00 | 901.65 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2026-03-09 09:15:00 | 892.45 | 2026-03-25 09:15:00 | 901.65 | STOP_HIT | 0.50 | -1.03% |
| SELL | retest2 | 2026-03-13 10:15:00 | 904.95 | 2026-03-25 09:15:00 | 901.65 | STOP_HIT | 0.50 | 0.36% |
| SELL | retest2 | 2026-03-13 11:00:00 | 905.20 | 2026-03-25 09:15:00 | 901.65 | STOP_HIT | 0.50 | 0.39% |
| SELL | retest2 | 2026-03-13 12:15:00 | 903.50 | 2026-03-25 09:15:00 | 901.65 | STOP_HIT | 0.50 | 0.20% |
| SELL | retest2 | 2026-03-16 10:15:00 | 887.45 | 2026-04-01 14:15:00 | 858.32 | PARTIAL | 0.50 | 3.28% |
| SELL | retest2 | 2026-03-17 11:30:00 | 892.65 | 2026-04-02 09:15:00 | 843.08 | PARTIAL | 0.50 | 5.55% |
| SELL | retest2 | 2026-03-17 15:00:00 | 891.95 | 2026-04-02 09:15:00 | 848.02 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2026-03-19 09:15:00 | 883.80 | 2026-04-02 09:15:00 | 847.35 | PARTIAL | 0.50 | 4.12% |
| SELL | retest2 | 2026-03-27 12:15:00 | 903.50 | 2026-04-02 09:15:00 | 839.61 | PARTIAL | 0.50 | 7.07% |
| SELL | retest2 | 2026-03-16 10:15:00 | 887.45 | 2026-04-08 14:15:00 | 891.80 | STOP_HIT | 0.50 | -0.49% |
| SELL | retest2 | 2026-03-17 11:30:00 | 892.65 | 2026-04-08 14:15:00 | 891.80 | STOP_HIT | 0.50 | 0.10% |
| SELL | retest2 | 2026-03-17 15:00:00 | 891.95 | 2026-04-08 14:15:00 | 891.80 | STOP_HIT | 0.50 | 0.02% |
| SELL | retest2 | 2026-03-19 09:15:00 | 883.80 | 2026-04-08 14:15:00 | 891.80 | STOP_HIT | 0.50 | -0.91% |
| SELL | retest2 | 2026-03-27 12:15:00 | 903.50 | 2026-04-08 14:15:00 | 891.80 | STOP_HIT | 0.50 | 1.29% |
| SELL | retest2 | 2026-04-09 11:45:00 | 903.75 | 2026-04-10 09:15:00 | 910.85 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-04-09 13:15:00 | 902.90 | 2026-04-10 09:15:00 | 910.85 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-04-13 09:15:00 | 905.00 | 2026-04-13 09:15:00 | 910.65 | STOP_HIT | 1.00 | -0.62% |
