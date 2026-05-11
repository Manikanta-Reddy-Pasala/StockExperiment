# Adani Energy Solutions Ltd. (ADANIENSOL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1351.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 17 |
| PARTIAL | 0 |
| TARGET_HIT | 5 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 14
- **Target hits / Stop hits / Partials:** 3 / 14 / 0
- **Avg / median % per leg:** -0.51% / -1.85%
- **Sum % (uncompounded):** -8.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 3 | 18.8% | 3 | 13 | 0 | -0.50% | -8.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 3 | 18.8% | 3 | 13 | 0 | -0.50% | -8.0% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.71% | -0.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.71% | -0.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 17 | 3 | 17.6% | 3 | 14 | 0 | -0.51% | -8.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 09:15:00 | 807.50 | 865.25 | 865.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 10:15:00 | 803.90 | 864.64 | 864.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 10:15:00 | 825.50 | 823.01 | 839.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 11:00:00 | 825.50 | 823.01 | 839.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 813.40 | 795.67 | 816.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:30:00 | 812.55 | 795.67 | 816.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 824.55 | 796.40 | 816.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 824.55 | 796.40 | 816.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 822.35 | 796.66 | 816.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 834.40 | 796.66 | 816.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 813.60 | 797.43 | 816.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:30:00 | 817.45 | 797.43 | 816.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 816.20 | 797.78 | 816.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:45:00 | 813.25 | 797.93 | 816.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 819.00 | 798.14 | 816.09 | SL hit (close>static) qty=1.00 sl=817.75 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 09:15:00 | 893.35 | 828.53 | 828.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 14:15:00 | 926.20 | 855.34 | 843.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 15:15:00 | 968.00 | 969.13 | 932.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-25 09:30:00 | 968.30 | 969.18 | 932.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 956.75 | 974.31 | 946.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 949.55 | 974.31 | 946.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 987.40 | 1008.26 | 982.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 984.60 | 1008.26 | 982.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 983.50 | 1008.01 | 982.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:00:00 | 983.50 | 1008.01 | 982.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 972.80 | 1007.66 | 982.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:00:00 | 972.80 | 1007.66 | 982.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 969.80 | 1007.28 | 982.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:30:00 | 968.00 | 1007.28 | 982.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 891.60 | 965.36 | 965.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 866.80 | 964.38 | 964.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 944.40 | 930.80 | 945.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 944.40 | 930.80 | 945.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 944.40 | 930.80 | 945.89 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 10:15:00 | 1029.55 | 957.75 | 957.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 12:15:00 | 1031.85 | 959.20 | 958.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 971.80 | 990.61 | 977.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 971.80 | 990.61 | 977.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 971.80 | 990.61 | 977.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 15:15:00 | 989.70 | 986.69 | 976.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 11:00:00 | 991.00 | 986.88 | 977.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 09:15:00 | 992.50 | 986.29 | 977.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 10:00:00 | 989.90 | 986.33 | 977.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 991.80 | 988.74 | 979.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 997.50 | 988.74 | 979.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 11:15:00 | 998.60 | 989.92 | 980.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 12:45:00 | 998.40 | 990.12 | 980.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 14:00:00 | 997.00 | 990.19 | 980.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 974.10 | 990.07 | 980.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 974.10 | 990.07 | 980.66 | SL hit (close<static) qty=1.00 sl=976.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-12 10:45:00 | 813.25 | 2025-09-12 11:15:00 | 819.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2026-03-05 15:15:00 | 989.70 | 2026-03-16 10:15:00 | 974.10 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-03-06 11:00:00 | 991.00 | 2026-03-16 10:15:00 | 974.10 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2026-03-10 09:15:00 | 992.50 | 2026-03-16 10:15:00 | 974.10 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2026-03-10 10:00:00 | 989.90 | 2026-03-16 10:15:00 | 974.10 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-03-12 10:15:00 | 997.50 | 2026-03-23 09:15:00 | 966.20 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2026-03-13 11:15:00 | 998.60 | 2026-03-23 09:15:00 | 966.20 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2026-03-13 12:45:00 | 998.40 | 2026-03-23 09:15:00 | 966.20 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2026-03-13 14:00:00 | 997.00 | 2026-03-23 09:15:00 | 966.20 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2026-03-16 13:15:00 | 984.20 | 2026-03-23 09:15:00 | 966.20 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-03-17 10:00:00 | 985.60 | 2026-03-23 10:15:00 | 947.80 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2026-03-17 10:45:00 | 984.30 | 2026-03-23 10:15:00 | 947.80 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2026-03-17 13:15:00 | 985.70 | 2026-03-23 10:15:00 | 947.80 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2026-03-17 14:45:00 | 1000.90 | 2026-03-23 10:15:00 | 947.80 | STOP_HIT | 1.00 | -5.31% |
| BUY | retest2 | 2026-04-06 14:30:00 | 995.10 | 2026-04-10 09:15:00 | 1094.61 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 10:15:00 | 995.90 | 2026-04-10 09:15:00 | 1095.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1037.60 | 2026-04-10 12:15:00 | 1141.36 | TARGET_HIT | 1.00 | 10.00% |
