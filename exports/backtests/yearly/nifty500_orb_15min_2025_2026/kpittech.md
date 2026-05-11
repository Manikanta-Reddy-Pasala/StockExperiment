# KPIT Technologies Ltd. (KPITTECH)

## Backtest Summary

- **Window:** 2025-12-08 09:15:00 → 2026-05-08 15:25:00 (6375 bars)
- **Last close:** 725.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 19 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 15
- **Target hits / Stop hits / Partials:** 4 / 15 / 7
- **Avg / median % per leg:** 0.23% / 0.00%
- **Sum % (uncompounded):** 5.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | -0.08% | -0.4% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | -0.08% | -0.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 21 | 9 | 42.9% | 3 | 12 | 6 | 0.30% | 6.4% |
| SELL @ 2nd Alert (retest1) | 21 | 9 | 42.9% | 3 | 12 | 6 | 0.30% | 6.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 26 | 11 | 42.3% | 4 | 15 | 7 | 0.23% | 6.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 11:00:00 | 1193.00 | 1210.03 | 0.00 | ORB-short ORB[1205.00,1218.10] vol=1.9x ATR=3.63 |
| Stop hit — per-position SL triggered | 2025-12-10 11:05:00 | 1196.63 | 1209.06 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-12-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 11:00:00 | 1227.30 | 1227.60 | 0.00 | ORB-short ORB[1228.60,1238.70] vol=2.1x ATR=3.02 |
| Stop hit — per-position SL triggered | 2025-12-12 11:05:00 | 1230.32 | 1228.12 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-12-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 10:35:00 | 1222.20 | 1225.10 | 0.00 | ORB-short ORB[1226.10,1237.70] vol=2.2x ATR=2.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 10:45:00 | 1218.03 | 1223.96 | 0.00 | T1 1.5R @ 1218.03 |
| Stop hit — per-position SL triggered | 2025-12-15 10:55:00 | 1222.20 | 1223.82 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-12-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 10:25:00 | 1235.70 | 1233.35 | 0.00 | ORB-long ORB[1229.00,1235.50] vol=1.8x ATR=2.82 |
| Stop hit — per-position SL triggered | 2025-12-24 10:50:00 | 1232.88 | 1233.55 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-12-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:10:00 | 1199.30 | 1207.82 | 0.00 | ORB-short ORB[1202.10,1211.90] vol=1.6x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 11:50:00 | 1194.78 | 1204.54 | 0.00 | T1 1.5R @ 1194.78 |
| Target hit | 2025-12-29 15:20:00 | 1188.20 | 1197.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-01-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 11:00:00 | 1166.00 | 1171.23 | 0.00 | ORB-short ORB[1175.10,1184.00] vol=2.2x ATR=2.46 |
| Stop hit — per-position SL triggered | 2026-01-01 11:25:00 | 1168.46 | 1169.96 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-01-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 09:40:00 | 1154.90 | 1158.02 | 0.00 | ORB-short ORB[1156.50,1167.20] vol=3.4x ATR=3.03 |
| Stop hit — per-position SL triggered | 2026-01-02 09:55:00 | 1157.93 | 1157.84 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-01-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 09:30:00 | 1156.20 | 1163.27 | 0.00 | ORB-short ORB[1157.40,1174.20] vol=2.1x ATR=3.18 |
| Stop hit — per-position SL triggered | 2026-01-05 09:40:00 | 1159.38 | 1161.68 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-01-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 10:20:00 | 1137.90 | 1144.85 | 0.00 | ORB-short ORB[1144.50,1154.20] vol=2.7x ATR=3.48 |
| Stop hit — per-position SL triggered | 2026-01-06 10:45:00 | 1141.38 | 1144.03 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-01-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 10:10:00 | 1099.20 | 1107.34 | 0.00 | ORB-short ORB[1104.50,1117.40] vol=1.5x ATR=3.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 10:40:00 | 1093.28 | 1104.05 | 0.00 | T1 1.5R @ 1093.28 |
| Stop hit — per-position SL triggered | 2026-01-28 12:30:00 | 1099.20 | 1100.13 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-01-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 10:05:00 | 1085.90 | 1093.94 | 0.00 | ORB-short ORB[1095.80,1110.10] vol=3.3x ATR=4.47 |
| Stop hit — per-position SL triggered | 2026-01-29 10:40:00 | 1090.37 | 1091.42 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-02-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 09:30:00 | 971.30 | 981.04 | 0.00 | ORB-short ORB[976.80,988.00] vol=1.5x ATR=3.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-05 09:40:00 | 965.53 | 976.34 | 0.00 | T1 1.5R @ 965.53 |
| Stop hit — per-position SL triggered | 2026-02-05 10:10:00 | 971.30 | 973.50 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-02-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 10:45:00 | 945.80 | 950.28 | 0.00 | ORB-short ORB[946.60,960.00] vol=1.7x ATR=2.47 |
| Stop hit — per-position SL triggered | 2026-02-06 11:05:00 | 948.27 | 949.78 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 974.10 | 970.57 | 0.00 | ORB-long ORB[964.10,973.00] vol=2.7x ATR=2.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:35:00 | 978.10 | 971.55 | 0.00 | T1 1.5R @ 978.10 |
| Target hit | 2026-02-10 10:10:00 | 978.30 | 979.55 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 856.40 | 861.84 | 0.00 | ORB-short ORB[860.00,869.10] vol=2.7x ATR=2.61 |
| Stop hit — per-position SL triggered | 2026-02-19 09:45:00 | 859.01 | 860.70 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-02-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:10:00 | 796.20 | 797.68 | 0.00 | ORB-short ORB[797.50,808.40] vol=1.7x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:35:00 | 791.91 | 797.46 | 0.00 | T1 1.5R @ 791.91 |
| Target hit | 2026-02-27 15:20:00 | 770.50 | 787.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 721.85 | 727.74 | 0.00 | ORB-short ORB[723.15,733.95] vol=1.8x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:50:00 | 717.69 | 724.41 | 0.00 | T1 1.5R @ 717.69 |
| Target hit | 2026-04-24 14:35:00 | 708.40 | 706.79 | 0.00 | Trail-exit close>VWAP |

### Cycle 18 — BUY (started 2026-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:30:00 | 746.65 | 735.31 | 0.00 | ORB-long ORB[727.65,738.05] vol=2.2x ATR=3.71 |
| Stop hit — per-position SL triggered | 2026-04-30 09:40:00 | 742.94 | 738.77 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-05-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:45:00 | 773.00 | 765.09 | 0.00 | ORB-long ORB[756.55,767.50] vol=1.6x ATR=4.08 |
| Stop hit — per-position SL triggered | 2026-05-04 11:40:00 | 768.92 | 769.30 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-12-10 11:00:00 | 1193.00 | 2025-12-10 11:05:00 | 1196.63 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-12 11:00:00 | 1227.30 | 2025-12-12 11:05:00 | 1230.32 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-15 10:35:00 | 1222.20 | 2025-12-15 10:45:00 | 1218.03 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-12-15 10:35:00 | 1222.20 | 2025-12-15 10:55:00 | 1222.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-24 10:25:00 | 1235.70 | 2025-12-24 10:50:00 | 1232.88 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-29 11:10:00 | 1199.30 | 2025-12-29 11:50:00 | 1194.78 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-12-29 11:10:00 | 1199.30 | 2025-12-29 15:20:00 | 1188.20 | TARGET_HIT | 0.50 | 0.93% |
| SELL | retest1 | 2026-01-01 11:00:00 | 1166.00 | 2026-01-01 11:25:00 | 1168.46 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-01-02 09:40:00 | 1154.90 | 2026-01-02 09:55:00 | 1157.93 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-01-05 09:30:00 | 1156.20 | 2026-01-05 09:40:00 | 1159.38 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-01-06 10:20:00 | 1137.90 | 2026-01-06 10:45:00 | 1141.38 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-01-28 10:10:00 | 1099.20 | 2026-01-28 10:40:00 | 1093.28 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-01-28 10:10:00 | 1099.20 | 2026-01-28 12:30:00 | 1099.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-29 10:05:00 | 1085.90 | 2026-01-29 10:40:00 | 1090.37 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-02-05 09:30:00 | 971.30 | 2026-02-05 09:40:00 | 965.53 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-02-05 09:30:00 | 971.30 | 2026-02-05 10:10:00 | 971.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-06 10:45:00 | 945.80 | 2026-02-06 11:05:00 | 948.27 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-10 09:30:00 | 974.10 | 2026-02-10 09:35:00 | 978.10 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-02-10 09:30:00 | 974.10 | 2026-02-10 10:10:00 | 978.30 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-19 09:30:00 | 856.40 | 2026-02-19 09:45:00 | 859.01 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-27 11:10:00 | 796.20 | 2026-02-27 11:35:00 | 791.91 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-02-27 11:10:00 | 796.20 | 2026-02-27 15:20:00 | 770.50 | TARGET_HIT | 0.50 | 3.23% |
| SELL | retest1 | 2026-04-24 09:30:00 | 721.85 | 2026-04-24 09:50:00 | 717.69 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-04-24 09:30:00 | 721.85 | 2026-04-24 14:35:00 | 708.40 | TARGET_HIT | 0.50 | 1.86% |
| BUY | retest1 | 2026-04-30 09:30:00 | 746.65 | 2026-04-30 09:40:00 | 742.94 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-05-04 09:45:00 | 773.00 | 2026-05-04 11:40:00 | 768.92 | STOP_HIT | 1.00 | -0.53% |
