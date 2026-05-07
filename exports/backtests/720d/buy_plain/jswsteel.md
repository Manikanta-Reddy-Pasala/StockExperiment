# JSWSTEEL (JSWSTEEL)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1280.40
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 6 |
| PENDING | 29 |
| PENDING_CANCEL | 8 |
| ENTRY1 | 0 |
| ENTRY2 | 21 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 13
- **Target hits / Stop hits / Partials:** 0 / 21 / 4
- **Avg / median % per leg:** 4.00% / -0.21%
- **Sum % (uncompounded):** 100.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 12 | 48.0% | 0 | 21 | 4 | 4.00% | 100.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 25 | 12 | 48.0% | 0 | 21 | 4 | 4.00% | 100.1% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 12 | 48.0% | 0 | 21 | 4 | 4.00% | 100.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 13:15:00 | 932.10 | 910.06 | 909.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 14:15:00 | 933.30 | 910.29 | 910.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 09:15:00 | 921.00 | 924.41 | 918.27 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 921.00 | 924.41 | 918.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 921.00 | 924.41 | 918.27 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-09-04 13:15:00 | 931.70 | 924.45 | 918.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 14:15:00 | 933.65 | 924.54 | 918.49 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-06 11:15:00 | 931.60 | 924.94 | 919.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 12:15:00 | 931.80 | 925.01 | 919.09 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-12 09:15:00 | 934.05 | 926.94 | 920.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 10:15:00 | 932.25 | 926.99 | 920.86 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-19 11:15:00 | 930.90 | 978.92 | 973.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-19 12:15:00 | 925.60 | 978.39 | 973.32 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-20 10:15:00 | 934.10 | 975.78 | 972.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 11:15:00 | 936.25 | 975.39 | 971.95 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 924.35 | 974.88 | 971.71 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-23 09:15:00 | 937.70 | 973.03 | 970.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 10:15:00 | 946.80 | 972.76 | 970.72 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-23 14:15:00 | 938.20 | 971.39 | 970.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 15:15:00 | 937.50 | 971.05 | 969.90 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-26 11:15:00 | 919.75 | 966.67 | 967.71 | SL hit (close<static) qty=1.00 sl=921.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-26 11:15:00 | 919.75 | 966.67 | 967.71 | SL hit (close<static) qty=1.00 sl=921.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-30 09:15:00 | 908.50 | 960.91 | 964.69 | SL hit (close<static) qty=1.00 sl=912.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-30 09:15:00 | 908.50 | 960.91 | 964.69 | SL hit (close<static) qty=1.00 sl=912.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-30 09:15:00 | 908.50 | 960.91 | 964.69 | SL hit (close<static) qty=1.00 sl=912.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-30 09:15:00 | 908.50 | 960.91 | 964.69 | SL hit (close<static) qty=1.00 sl=912.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-23 13:15:00 | 936.65 | 923.76 | 937.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-23 14:15:00 | 929.40 | 923.82 | 937.90 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-24 10:15:00 | 946.40 | 924.21 | 937.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 11:15:00 | 946.65 | 924.43 | 937.93 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-27 09:15:00 | 918.05 | 924.79 | 937.78 | SL hit (close<static) qty=1.00 sl=921.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-29 11:15:00 | 939.05 | 923.75 | 936.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-29 12:15:00 | 933.70 | 923.85 | 936.22 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-29 13:15:00 | 937.40 | 923.99 | 936.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-29 14:15:00 | 939.45 | 924.14 | 936.25 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 954.30 | 924.57 | 936.34 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-02-01 12:15:00 | 919.00 | 928.03 | 937.19 | SL hit (close<static) qty=1.00 sl=921.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-07 09:15:00 | 961.70 | 932.08 | 937.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:15:00 | 973.75 | 932.49 | 938.15 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-10 11:15:00 | 957.50 | 935.23 | 939.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 12:15:00 | 960.55 | 935.49 | 939.43 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-12 11:15:00 | 959.20 | 937.90 | 940.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-12 12:15:00 | 957.20 | 938.10 | 940.52 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-12 14:15:00 | 957.40 | 938.43 | 940.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-12 15:15:00 | 956.20 | 938.61 | 940.74 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-17 09:15:00 | 958.55 | 942.84 | 942.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 09:15:00 | 958.55 | 942.84 | 942.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 09:15:00 | 958.55 | 942.84 | 942.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 09:15:00 | 958.55 | 942.84 | 942.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 09:15:00 | 958.55 | 942.84 | 942.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-17 14:15:00 | 974.20 | 943.98 | 943.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 10:15:00 | 955.00 | 955.45 | 950.07 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 12:15:00 | 945.55 | 955.32 | 950.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 12:15:00 | 945.55 | 955.32 | 950.06 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-03-03 09:15:00 | 958.10 | 955.19 | 950.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-03 10:15:00 | 956.75 | 955.20 | 950.13 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 930.65 | 1019.50 | 995.40 | SL hit (close<static) qty=1.00 sl=945.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-08 13:15:00 | 957.95 | 1010.79 | 992.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-08 14:15:00 | 953.80 | 1010.22 | 992.02 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-08 15:15:00 | 956.20 | 1009.68 | 991.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-09 09:15:00 | 942.15 | 1009.01 | 991.59 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-04-11 09:15:00 | 983.35 | 1005.11 | 990.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 10:15:00 | 986.90 | 1004.93 | 990.18 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-06 09:15:00 | 963.00 | 1012.30 | 1000.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 10:15:00 | 964.10 | 1011.82 | 1000.59 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-09 11:15:00 | 958.30 | 1001.92 | 996.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 12:15:00 | 957.40 | 1001.48 | 996.37 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 957.30 | 1001.04 | 996.17 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 982.40 | 999.96 | 995.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 987.50 | 999.84 | 995.66 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-09-08 09:15:00 | 1108.71 | 1052.16 | 1041.40 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-09-08 09:15:00 | 1101.01 | 1052.16 | 1041.40 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-09-23 12:15:00 | 1134.93 | 1084.84 | 1064.22 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-09-23 12:15:00 | 1135.62 | 1084.84 | 1064.22 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 1136.40 | 1137.31 | 1108.93 | SL hit (close<ema200) qty=0.50 sl=1137.31 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 1136.40 | 1137.31 | 1108.93 | SL hit (close<ema200) qty=0.50 sl=1137.31 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 1136.40 | 1137.31 | 1108.93 | SL hit (close<ema200) qty=0.50 sl=1137.31 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 1136.40 | 1137.31 | 1108.93 | SL hit (close<ema200) qty=0.50 sl=1137.31 alert=retest2 |

### Cycle 3 — BUY (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 15:15:00 | 1187.00 | 1131.48 | 1131.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 09:15:00 | 1188.50 | 1132.05 | 1131.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 1225.60 | 1231.96 | 1204.48 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 11:15:00 | 1202.80 | 1231.53 | 1204.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 1202.80 | 1231.53 | 1204.53 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-04 13:15:00 | 1217.90 | 1231.21 | 1204.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-04 14:15:00 | 1212.20 | 1231.02 | 1204.68 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-05 09:15:00 | 1230.10 | 1230.78 | 1204.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 10:15:00 | 1239.40 | 1230.87 | 1205.00 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 1186.60 | 1231.31 | 1206.86 | SL hit (close<static) qty=1.00 sl=1202.10 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-10 15:15:00 | 1219.60 | 1227.65 | 1206.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-11 09:15:00 | 1195.30 | 1227.33 | 1206.44 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 1222.20 | 1172.96 | 1180.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 10:15:00 | 1221.10 | 1173.44 | 1180.41 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 1193.50 | 1175.54 | 1181.27 | SL hit (close<static) qty=1.00 sl=1202.10 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-15 09:15:00 | 1222.40 | 1177.75 | 1182.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-15 10:15:00 | 1217.30 | 1178.14 | 1182.37 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-15 12:15:00 | 1220.50 | 1178.91 | 1182.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 13:15:00 | 1220.80 | 1179.32 | 1182.90 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-17 10:15:00 | 1219.00 | 1183.52 | 1184.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 11:15:00 | 1225.40 | 1183.93 | 1185.07 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-20 09:15:00 | 1250.30 | 1186.69 | 1186.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-20 09:15:00 | 1250.30 | 1186.69 | 1186.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 09:15:00 | 1250.30 | 1186.69 | 1186.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 10:15:00 | 1253.30 | 1187.35 | 1186.77 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-09-04 14:15:00 | 933.65 | 2024-12-26 11:15:00 | 919.75 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-09-06 12:15:00 | 931.80 | 2024-12-26 11:15:00 | 919.75 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-09-12 10:15:00 | 932.25 | 2024-12-30 09:15:00 | 908.50 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2024-12-20 11:15:00 | 936.25 | 2024-12-30 09:15:00 | 908.50 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2024-12-23 10:15:00 | 946.80 | 2024-12-30 09:15:00 | 908.50 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2024-12-23 15:15:00 | 937.50 | 2024-12-30 09:15:00 | 908.50 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2025-01-24 11:15:00 | 946.65 | 2025-01-27 09:15:00 | 918.05 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2025-01-29 14:15:00 | 939.45 | 2025-02-01 12:15:00 | 919.00 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-02-07 10:15:00 | 973.75 | 2025-02-17 09:15:00 | 958.55 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-02-10 12:15:00 | 960.55 | 2025-02-17 09:15:00 | 958.55 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-02-12 12:15:00 | 957.20 | 2025-02-17 09:15:00 | 958.55 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-02-12 15:15:00 | 956.20 | 2025-02-17 09:15:00 | 958.55 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2025-03-03 10:15:00 | 956.75 | 2025-04-07 09:15:00 | 930.65 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-04-11 10:15:00 | 986.90 | 2025-09-08 09:15:00 | 1108.71 | PARTIAL | 0.50 | 12.34% |
| BUY | retest2 | 2025-05-06 10:15:00 | 964.10 | 2025-09-08 09:15:00 | 1101.01 | PARTIAL | 0.50 | 14.20% |
| BUY | retest2 | 2025-05-09 12:15:00 | 957.40 | 2025-09-23 12:15:00 | 1134.93 | PARTIAL | 0.50 | 18.54% |
| BUY | retest2 | 2025-05-12 10:15:00 | 987.50 | 2025-09-23 12:15:00 | 1135.62 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-11 10:15:00 | 986.90 | 2025-10-23 14:15:00 | 1136.40 | STOP_HIT | 0.50 | 15.15% |
| BUY | retest2 | 2025-05-06 10:15:00 | 964.10 | 2025-10-23 14:15:00 | 1136.40 | STOP_HIT | 0.50 | 17.87% |
| BUY | retest2 | 2025-05-09 12:15:00 | 957.40 | 2025-10-23 14:15:00 | 1136.40 | STOP_HIT | 0.50 | 18.70% |
| BUY | retest2 | 2025-05-12 10:15:00 | 987.50 | 2025-10-23 14:15:00 | 1136.40 | STOP_HIT | 0.50 | 15.08% |
| BUY | retest2 | 2026-03-05 10:15:00 | 1239.40 | 2026-03-09 09:15:00 | 1186.60 | STOP_HIT | 1.00 | -4.26% |
| BUY | retest2 | 2026-04-10 10:15:00 | 1221.10 | 2026-04-13 09:15:00 | 1193.50 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2026-04-15 13:15:00 | 1220.80 | 2026-04-20 09:15:00 | 1250.30 | STOP_HIT | 1.00 | 2.42% |
| BUY | retest2 | 2026-04-17 11:15:00 | 1225.40 | 2026-04-20 09:15:00 | 1250.30 | STOP_HIT | 1.00 | 2.03% |
