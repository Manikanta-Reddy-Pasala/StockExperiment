# BAJFINANCE (BAJFINANCE)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 954.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 18 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 19
- **Target hits / Stop hits / Partials:** 0 / 19 / 0
- **Avg / median % per leg:** -1.66% / -1.39%
- **Sum % (uncompounded):** -31.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.31% | -6.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.31% | -6.5% |
| SELL (all) | 14 | 0 | 0.0% | 0 | 14 | 0 | -1.78% | -24.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 0 | 0.0% | 0 | 14 | 0 | -1.78% | -24.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 0 | 0.0% | 0 | 19 | 0 | -1.66% | -31.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 15:15:00 | 975.00 | 1008.84 | 1008.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 968.45 | 1003.44 | 1006.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 955.95 | 954.92 | 974.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 12:00:00 | 955.95 | 954.92 | 974.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 969.95 | 955.52 | 974.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 969.95 | 955.52 | 974.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 972.00 | 956.66 | 973.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:45:00 | 977.90 | 956.66 | 973.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 972.05 | 956.81 | 973.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 10:00:00 | 964.50 | 959.49 | 973.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 11:00:00 | 964.70 | 959.54 | 973.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 12:00:00 | 965.30 | 959.60 | 973.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 15:00:00 | 965.50 | 959.84 | 973.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 970.50 | 960.03 | 973.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:45:00 | 972.00 | 960.03 | 973.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 977.95 | 960.63 | 973.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 977.95 | 960.63 | 973.41 | SL hit (close>static) qty=1.00 sl=975.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 977.95 | 960.63 | 973.41 | SL hit (close>static) qty=1.00 sl=975.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 977.95 | 960.63 | 973.41 | SL hit (close>static) qty=1.00 sl=975.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 977.95 | 960.63 | 973.41 | SL hit (close>static) qty=1.00 sl=975.90 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-02-12 10:00:00 | 977.95 | 960.63 | 973.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 987.80 | 960.90 | 973.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 987.80 | 960.90 | 973.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 1037.60 | 983.25 | 983.13 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 951.70 | 984.31 | 984.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 925.45 | 983.72 | 984.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 906.60 | 885.10 | 920.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 10:15:00 | 921.70 | 885.46 | 920.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 921.70 | 885.46 | 920.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 921.70 | 885.46 | 920.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 918.10 | 885.79 | 920.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:15:00 | 908.80 | 886.96 | 920.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 898.85 | 890.44 | 920.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 13:15:00 | 912.50 | 892.21 | 919.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 15:15:00 | 912.50 | 892.64 | 919.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 931.20 | 893.22 | 919.34 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 931.20 | 893.22 | 919.34 | SL hit (close>static) qty=1.00 sl=926.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 931.20 | 893.22 | 919.34 | SL hit (close>static) qty=1.00 sl=926.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 931.20 | 893.22 | 919.34 | SL hit (close>static) qty=1.00 sl=926.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 931.20 | 893.22 | 919.34 | SL hit (close>static) qty=1.00 sl=926.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-04-16 10:00:00 | 931.20 | 893.22 | 919.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 922.40 | 893.51 | 919.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:15:00 | 920.50 | 893.51 | 919.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 11:15:00 | 939.75 | 897.60 | 918.80 | SL hit (close>static) qty=1.00 sl=932.60 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:30:00 | 920.05 | 901.92 | 919.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:00:00 | 919.85 | 901.92 | 919.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 14:00:00 | 921.70 | 902.48 | 919.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 918.00 | 902.64 | 919.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 15:15:00 | 916.50 | 902.64 | 919.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 15:15:00 | 923.00 | 903.75 | 919.61 | SL hit (close>static) qty=1.00 sl=922.75 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 10:00:00 | 913.95 | 903.85 | 919.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 13:15:00 | 926.90 | 904.61 | 919.65 | SL hit (close>static) qty=1.00 sl=922.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-29 12:15:00 | 936.40 | 907.14 | 920.01 | SL hit (close>static) qty=1.00 sl=932.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-29 12:15:00 | 936.40 | 907.14 | 920.01 | SL hit (close>static) qty=1.00 sl=932.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-29 12:15:00 | 936.40 | 907.14 | 920.01 | SL hit (close>static) qty=1.00 sl=932.60 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-11 15:15:00 | 1009.40 | 2025-11-24 13:15:00 | 994.80 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-11-13 15:15:00 | 1006.70 | 2025-11-24 13:15:00 | 994.80 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-11-14 10:45:00 | 1008.40 | 2025-11-24 13:15:00 | 994.80 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-11-24 10:30:00 | 1007.20 | 2025-11-24 13:15:00 | 994.80 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-24 09:15:00 | 1022.90 | 2025-12-24 15:15:00 | 1009.40 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-02-10 10:00:00 | 964.50 | 2026-02-12 09:15:00 | 977.95 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-02-10 11:00:00 | 964.70 | 2026-02-12 09:15:00 | 977.95 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-02-10 12:00:00 | 965.30 | 2026-02-12 09:15:00 | 977.95 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-02-10 15:00:00 | 965.50 | 2026-02-12 09:15:00 | 977.95 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-04-09 09:15:00 | 908.80 | 2026-04-16 09:15:00 | 931.20 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-04-13 09:15:00 | 898.85 | 2026-04-16 09:15:00 | 931.20 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2026-04-15 13:15:00 | 912.50 | 2026-04-16 09:15:00 | 931.20 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-04-15 15:15:00 | 912.50 | 2026-04-16 09:15:00 | 931.20 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-04-16 11:15:00 | 920.50 | 2026-04-21 11:15:00 | 939.75 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2026-04-23 09:30:00 | 920.05 | 2026-04-24 15:15:00 | 923.00 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2026-04-23 10:00:00 | 919.85 | 2026-04-27 13:15:00 | 926.90 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-04-23 14:00:00 | 921.70 | 2026-04-29 12:15:00 | 936.40 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-04-23 15:15:00 | 916.50 | 2026-04-29 12:15:00 | 936.40 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2026-04-27 10:00:00 | 913.95 | 2026-04-29 12:15:00 | 936.40 | STOP_HIT | 1.00 | -2.46% |
