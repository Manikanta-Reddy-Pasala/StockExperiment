# ICICIBANK (ICICIBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 1279.50
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
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 4 |
| PENDING | 16 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 1 |
| ENTRY2 | 9 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 6
- **Target hits / Stop hits / Partials:** 2 / 8 / 4
- **Avg / median % per leg:** 9.41% / 9.29%
- **Sum % (uncompounded):** 131.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 8 | 57.1% | 2 | 8 | 4 | 9.41% | 131.7% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.76% | -1.8% |
| BUY @ 3rd Alert (retest2) | 13 | 8 | 61.5% | 2 | 7 | 4 | 10.26% | 133.4% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.76% | -1.8% |
| retest2 (combined) | 13 | 8 | 61.5% | 2 | 7 | 4 | 10.26% | 133.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 15:15:00 | 1002.70 | 945.74 | 945.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 09:15:00 | 1004.05 | 949.75 | 947.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 10:15:00 | 986.30 | 988.42 | 973.82 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-01-05 09:15:00 | 996.00 | 987.82 | 974.88 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-05 10:15:00 | 988.80 | 987.82 | 974.95 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-05 12:15:00 | 994.80 | 987.92 | 975.13 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-05 13:15:00 | 986.60 | 987.91 | 975.18 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-05 14:15:00 | 994.10 | 987.97 | 975.28 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 15:15:00 | 993.70 | 988.03 | 975.37 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-08 10:15:00 | 993.65 | 988.13 | 975.55 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-08 11:15:00 | 990.75 | 988.16 | 975.62 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 15:15:00 | 977.75 | 988.04 | 976.24 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-01-09 15:15:00 | 976.24 | 988.04 | 976.24 | SL hit qty=1.00 sl=976.24 alert=retest1 |
| Cross detected — sustain check pending | 2024-01-10 09:15:00 | 987.55 | 988.04 | 976.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 10:15:00 | 990.10 | 988.06 | 976.36 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-17 13:15:00 | 986.10 | 991.67 | 980.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-17 14:15:00 | 981.40 | 991.56 | 980.34 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-18 09:15:00 | 983.70 | 991.37 | 980.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 975.55 | 991.37 | 980.35 | SL hit qty=1.00 sl=975.55 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 10:15:00 | 989.30 | 991.35 | 980.40 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-18 13:15:00 | 984.30 | 991.19 | 980.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 14:15:00 | 986.15 | 991.14 | 980.51 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-04-29 10:15:00 | 1134.07 | 1084.72 | 1067.06 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-04-29 11:15:00 | 1137.69 | 1085.29 | 1067.43 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2024-09-18 11:15:00 | 1282.00 | 1224.19 | 1205.14 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit — 30% from entry | 2024-09-18 12:15:00 | 1286.09 | 1224.85 | 1205.57 | Target hit (30%) qty=0.50 alert=retest2 |
| CROSSOVER_SKIP | 2025-01-15 09:15:00 | 1237.50 | 1279.88 | 1279.94 | HTF filter: close above htf_sma |

### Cycle 2 — BUY (started 2025-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 11:15:00 | 1341.20 | 1254.86 | 1254.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 09:15:00 | 1358.20 | 1259.24 | 1256.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1287.85 | 1295.44 | 1278.62 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 10:15:00 | 1276.10 | 1295.25 | 1278.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 10:15:00 | 1276.10 | 1295.25 | 1278.61 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-08 09:15:00 | 1295.80 | 1294.52 | 1278.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 10:15:00 | 1296.40 | 1294.54 | 1278.82 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-09 11:15:00 | 1291.10 | 1294.69 | 1279.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 12:15:00 | 1296.75 | 1294.71 | 1279.60 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-07-25 09:15:00 | 1490.86 | 1439.83 | 1423.74 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-07-25 09:15:00 | 1491.26 | 1439.83 | 1423.74 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2025-09-03 12:15:00 | 1391.20 | 1429.48 | 1429.51 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2026-01-12 13:15:00 | 1416.90 | 1377.35 | 1377.27 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 13:15:00 | 1416.90 | 1377.35 | 1377.27 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 1416.90 | 1377.35 | 1377.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 1422.70 | 1378.48 | 1377.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 1362.30 | 1386.10 | 1381.95 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 1362.30 | 1386.10 | 1381.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1362.30 | 1386.10 | 1381.95 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2026-01-23 13:15:00 | 1347.60 | 1378.54 | 1378.57 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2026-02-04 09:15:00 | 1402.90 | 1374.32 | 1376.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 10:15:00 | 1406.20 | 1374.63 | 1376.25 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-05 14:15:00 | 1398.30 | 1377.89 | 1377.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 14:15:00 | 1398.30 | 1377.89 | 1377.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 09:15:00 | 1403.60 | 1378.29 | 1378.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 1387.30 | 1391.98 | 1386.14 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 1387.30 | 1391.98 | 1386.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 1387.30 | 1391.98 | 1386.14 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-20 09:15:00 | 1398.20 | 1392.01 | 1386.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:15:00 | 1398.70 | 1392.08 | 1386.27 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-20 13:15:00 | 1397.50 | 1392.20 | 1386.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-20 14:15:00 | 1394.20 | 1392.22 | 1386.46 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-23 09:15:00 | 1406.70 | 1392.39 | 1386.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 10:15:00 | 1401.70 | 1392.48 | 1386.67 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-24 12:15:00 | 1385.70 | 1392.80 | 1387.10 | SL hit qty=1.00 sl=1385.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 12:15:00 | 1385.70 | 1392.80 | 1387.10 | SL hit qty=1.00 sl=1385.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-25 09:15:00 | 1399.30 | 1392.70 | 1387.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-25 10:15:00 | 1394.90 | 1392.72 | 1387.20 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-25 11:15:00 | 1399.60 | 1392.79 | 1387.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 12:15:00 | 1400.00 | 1392.86 | 1387.32 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-27 10:15:00 | 1385.70 | 1393.75 | 1388.10 | SL hit qty=1.00 sl=1385.70 alert=retest2 |
| CROSSOVER_SKIP | 2026-03-06 14:15:00 | 1312.00 | 1383.42 | 1383.48 | slope filter: EMA200 not falling 0.50% over 350 bars |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-05 15:15:00 | 993.70 | 2024-01-09 15:15:00 | 976.24 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-01-10 10:15:00 | 990.10 | 2024-01-18 09:15:00 | 975.55 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-01-18 10:15:00 | 989.30 | 2024-04-29 10:15:00 | 1134.07 | PARTIAL | 0.50 | 14.63% |
| BUY | retest2 | 2024-01-18 14:15:00 | 986.15 | 2024-04-29 11:15:00 | 1137.69 | PARTIAL | 0.50 | 15.37% |
| BUY | retest2 | 2024-01-18 10:15:00 | 989.30 | 2024-09-18 11:15:00 | 1282.00 | TARGET_HIT | 0.50 | 29.59% |
| BUY | retest2 | 2024-01-18 14:15:00 | 986.15 | 2024-09-18 12:15:00 | 1286.09 | TARGET_HIT | 0.50 | 30.42% |
| BUY | retest2 | 2025-04-08 10:15:00 | 1296.40 | 2025-07-25 09:15:00 | 1490.86 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-09 12:15:00 | 1296.75 | 2025-07-25 09:15:00 | 1491.26 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-08 10:15:00 | 1296.40 | 2026-01-12 13:15:00 | 1416.90 | STOP_HIT | 0.50 | 9.29% |
| BUY | retest2 | 2025-04-09 12:15:00 | 1296.75 | 2026-01-12 13:15:00 | 1416.90 | STOP_HIT | 0.50 | 9.27% |
| BUY | retest2 | 2026-02-04 10:15:00 | 1406.20 | 2026-02-05 14:15:00 | 1398.30 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2026-02-20 10:15:00 | 1398.70 | 2026-02-24 12:15:00 | 1385.70 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-02-23 10:15:00 | 1401.70 | 2026-02-24 12:15:00 | 1385.70 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-02-25 12:15:00 | 1400.00 | 2026-02-27 10:15:00 | 1385.70 | STOP_HIT | 1.00 | -1.02% |
