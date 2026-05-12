# JSWSTEEL (JSWSTEEL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1272.00
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
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 35 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 56 |
| PARTIAL | 0 |
| TARGET_HIT | 8 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 56 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 44
- **Target hits / Stop hits / Partials:** 8 / 48 / 0
- **Avg / median % per leg:** -0.15% / -1.72%
- **Sum % (uncompounded):** -8.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 53 | 12 | 22.6% | 8 | 45 | 0 | -0.06% | -3.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 53 | 12 | 22.6% | 8 | 45 | 0 | -0.06% | -3.3% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.64% | -4.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.64% | -4.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 56 | 12 | 21.4% | 8 | 48 | 0 | -0.15% | -8.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 15:15:00 | 923.80 | 968.01 | 968.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 11:15:00 | 919.75 | 966.67 | 967.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-21 09:15:00 | 924.80 | 923.64 | 939.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-21 09:30:00 | 931.85 | 923.64 | 939.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 932.45 | 923.99 | 937.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:30:00 | 938.75 | 923.99 | 937.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 946.40 | 924.21 | 937.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:00:00 | 946.40 | 924.21 | 937.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 946.65 | 924.43 | 937.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:45:00 | 948.50 | 924.43 | 937.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 939.05 | 923.75 | 936.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:00:00 | 939.05 | 923.75 | 936.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 933.70 | 923.85 | 936.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 12:30:00 | 924.35 | 928.03 | 937.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 930.50 | 928.08 | 937.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 09:15:00 | 925.00 | 928.18 | 937.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 11:15:00 | 941.85 | 928.54 | 936.80 | SL hit (close>static) qty=1.00 sl=940.90 alert=retest2 |

### Cycle 2 — BUY (started 2025-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-14 15:15:00 | 962.85 | 942.69 | 942.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-17 13:15:00 | 966.85 | 943.67 | 943.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 10:15:00 | 955.00 | 955.45 | 950.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-28 11:00:00 | 955.00 | 955.45 | 950.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 12:15:00 | 945.55 | 955.32 | 950.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 13:00:00 | 945.55 | 955.32 | 950.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 13:15:00 | 944.95 | 955.22 | 950.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 14:00:00 | 944.95 | 955.22 | 950.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 954.00 | 955.16 | 950.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 961.00 | 955.16 | 950.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-21 09:15:00 | 1057.10 | 989.95 | 972.45 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 1076.90 | 1134.81 | 1135.00 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 15:15:00 | 1187.00 | 1131.48 | 1131.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 09:15:00 | 1188.50 | 1132.05 | 1131.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 1225.60 | 1231.96 | 1204.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-04 10:00:00 | 1225.60 | 1231.96 | 1204.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 1202.80 | 1231.53 | 1204.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:00:00 | 1202.80 | 1231.53 | 1204.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 1212.80 | 1231.34 | 1204.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 13:15:00 | 1215.70 | 1231.34 | 1204.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 14:00:00 | 1217.90 | 1231.21 | 1204.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:15:00 | 1220.10 | 1230.79 | 1204.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 1186.60 | 1231.31 | 1206.86 | SL hit (close<static) qty=1.00 sl=1200.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 10:15:00 | 1120.50 | 1191.41 | 1191.63 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 09:15:00 | 1250.30 | 1186.69 | 1186.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 10:15:00 | 1253.30 | 1187.35 | 1186.77 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-13 12:45:00 | 852.35 | 2024-06-04 11:15:00 | 834.00 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2024-05-13 13:30:00 | 853.40 | 2024-06-04 11:15:00 | 834.00 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2024-06-04 12:45:00 | 856.10 | 2024-06-04 14:15:00 | 847.05 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-06-06 09:15:00 | 887.00 | 2024-07-29 12:15:00 | 898.95 | STOP_HIT | 1.00 | 1.35% |
| BUY | retest2 | 2024-06-06 13:15:00 | 884.80 | 2024-08-02 14:15:00 | 899.60 | STOP_HIT | 1.00 | 1.67% |
| BUY | retest2 | 2024-07-22 09:30:00 | 882.60 | 2024-08-02 14:15:00 | 899.60 | STOP_HIT | 1.00 | 1.93% |
| BUY | retest2 | 2024-07-23 12:30:00 | 883.25 | 2024-08-02 14:15:00 | 899.60 | STOP_HIT | 1.00 | 1.85% |
| BUY | retest2 | 2024-07-29 11:15:00 | 904.50 | 2024-08-02 14:15:00 | 899.60 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-07-30 12:15:00 | 903.90 | 2024-08-05 10:15:00 | 864.75 | STOP_HIT | 1.00 | -4.33% |
| BUY | retest2 | 2024-07-30 14:30:00 | 904.35 | 2024-08-05 10:15:00 | 864.75 | STOP_HIT | 1.00 | -4.38% |
| BUY | retest2 | 2024-07-31 09:15:00 | 907.20 | 2024-08-05 10:15:00 | 864.75 | STOP_HIT | 1.00 | -4.68% |
| BUY | retest2 | 2024-07-31 10:30:00 | 919.10 | 2024-08-05 10:15:00 | 864.75 | STOP_HIT | 1.00 | -5.91% |
| BUY | retest2 | 2024-08-12 09:30:00 | 915.80 | 2024-08-14 10:15:00 | 893.40 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2024-08-12 10:00:00 | 917.85 | 2024-08-14 10:15:00 | 893.40 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2024-08-19 13:00:00 | 915.00 | 2024-09-24 09:15:00 | 1006.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-04 11:30:00 | 931.60 | 2024-09-30 09:15:00 | 1021.52 | TARGET_HIT | 1.00 | 9.65% |
| BUY | retest2 | 2024-09-04 13:15:00 | 928.65 | 2024-09-30 09:15:00 | 1023.99 | TARGET_HIT | 1.00 | 10.27% |
| BUY | retest2 | 2024-09-06 11:45:00 | 930.90 | 2024-09-30 09:15:00 | 1023.17 | TARGET_HIT | 1.00 | 9.91% |
| BUY | retest2 | 2024-09-09 12:30:00 | 930.15 | 2024-09-30 10:15:00 | 1024.76 | TARGET_HIT | 1.00 | 10.17% |
| BUY | retest2 | 2024-10-28 15:15:00 | 972.15 | 2024-10-29 09:15:00 | 958.95 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-11-01 18:15:00 | 971.40 | 2024-11-04 09:15:00 | 954.70 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-11-05 09:15:00 | 971.85 | 2024-11-12 12:15:00 | 964.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-11-05 09:45:00 | 973.00 | 2024-11-12 12:15:00 | 964.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-11-26 14:30:00 | 964.65 | 2024-12-18 09:15:00 | 956.10 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-11-27 09:45:00 | 965.65 | 2024-12-18 09:15:00 | 956.10 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-11-27 12:45:00 | 963.15 | 2024-12-18 09:15:00 | 956.10 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-11-29 09:45:00 | 962.20 | 2024-12-18 11:15:00 | 947.25 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-12-02 10:30:00 | 977.00 | 2024-12-18 11:15:00 | 947.25 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2024-12-02 11:15:00 | 977.55 | 2024-12-18 11:15:00 | 947.25 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2024-12-13 11:15:00 | 977.10 | 2024-12-18 11:15:00 | 947.25 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-02-01 12:30:00 | 924.35 | 2025-02-04 11:15:00 | 941.85 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-02-01 14:15:00 | 930.50 | 2025-02-04 11:15:00 | 941.85 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-02-03 09:15:00 | 925.00 | 2025-02-04 11:15:00 | 941.85 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-03-03 09:15:00 | 961.00 | 2025-03-21 09:15:00 | 1057.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-08 09:15:00 | 960.95 | 2025-04-08 09:15:00 | 943.95 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-04-08 12:45:00 | 955.35 | 2025-04-09 09:15:00 | 942.15 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-04-08 15:15:00 | 956.20 | 2025-04-09 09:15:00 | 942.15 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-04-11 11:15:00 | 990.25 | 2025-05-02 11:15:00 | 968.20 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-04-11 14:30:00 | 989.40 | 2025-05-02 11:15:00 | 968.20 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-05-12 12:30:00 | 994.50 | 2025-06-02 09:15:00 | 976.40 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-05-13 15:15:00 | 989.70 | 2025-06-02 09:15:00 | 976.40 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-05-15 09:15:00 | 1010.50 | 2025-06-02 09:15:00 | 976.40 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2025-05-22 09:30:00 | 993.20 | 2025-06-02 09:15:00 | 976.40 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-05-22 10:30:00 | 996.10 | 2025-06-02 09:15:00 | 976.40 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-05-30 15:15:00 | 994.00 | 2025-06-02 09:15:00 | 976.40 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-06-09 11:15:00 | 1012.75 | 2025-06-13 09:15:00 | 989.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-06-09 11:45:00 | 1011.90 | 2025-06-13 09:15:00 | 989.00 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-06-10 09:30:00 | 1013.00 | 2025-06-13 09:15:00 | 989.00 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-06-11 09:15:00 | 1013.50 | 2025-06-13 09:15:00 | 989.00 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-06-20 10:15:00 | 1007.70 | 2025-06-23 09:15:00 | 987.90 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-06-24 09:15:00 | 1010.60 | 2025-09-08 09:15:00 | 1111.66 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-24 10:15:00 | 1014.55 | 2025-09-16 12:15:00 | 1116.01 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-04 13:15:00 | 1215.70 | 2026-03-09 09:15:00 | 1186.60 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2026-03-04 14:00:00 | 1217.90 | 2026-03-09 09:15:00 | 1186.60 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2026-03-05 09:15:00 | 1220.10 | 2026-03-09 09:15:00 | 1186.60 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2026-03-10 12:45:00 | 1213.00 | 2026-03-11 09:15:00 | 1195.30 | STOP_HIT | 1.00 | -1.46% |
