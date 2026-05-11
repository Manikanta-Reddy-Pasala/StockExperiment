# Poly Medicure Ltd. (POLYMED)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1649.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 230 |
| ALERT1 | 155 |
| ALERT2 | 152 |
| ALERT2_SKIP | 101 |
| ALERT3 | 292 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 114 |
| PARTIAL | 26 |
| TARGET_HIT | 14 |
| STOP_HIT | 100 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 140 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 62 / 78
- **Target hits / Stop hits / Partials:** 14 / 100 / 26
- **Avg / median % per leg:** 1.61% / -0.46%
- **Sum % (uncompounded):** 225.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 7 | 13.7% | 7 | 44 | 0 | -0.05% | -2.5% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.68% | -1.7% |
| BUY @ 3rd Alert (retest2) | 50 | 7 | 14.0% | 7 | 43 | 0 | -0.02% | -0.8% |
| SELL (all) | 89 | 55 | 61.8% | 7 | 56 | 26 | 2.57% | 228.4% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.93% | -14.8% |
| SELL @ 3rd Alert (retest2) | 86 | 55 | 64.0% | 7 | 53 | 26 | 2.83% | 243.2% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.12% | -16.5% |
| retest2 (combined) | 136 | 62 | 45.6% | 14 | 96 | 26 | 1.78% | 242.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 11:15:00 | 979.50 | 982.76 | 983.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 12:15:00 | 976.35 | 981.48 | 982.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 14:15:00 | 969.55 | 966.50 | 972.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 14:15:00 | 969.55 | 966.50 | 972.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 14:15:00 | 969.55 | 966.50 | 972.52 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 10:15:00 | 979.40 | 974.21 | 973.83 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 12:15:00 | 963.85 | 973.32 | 973.97 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 13:15:00 | 975.00 | 973.01 | 972.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-29 09:15:00 | 976.10 | 974.27 | 973.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 15:15:00 | 973.45 | 977.26 | 975.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-29 15:15:00 | 973.45 | 977.26 | 975.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 15:15:00 | 973.45 | 977.26 | 975.85 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 11:15:00 | 971.35 | 974.65 | 974.92 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 12:15:00 | 975.45 | 974.35 | 974.29 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-01 15:15:00 | 971.05 | 974.37 | 974.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-02 12:15:00 | 967.00 | 972.49 | 973.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-05 13:15:00 | 973.30 | 970.09 | 971.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 13:15:00 | 973.30 | 970.09 | 971.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 13:15:00 | 973.30 | 970.09 | 971.51 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-06-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 11:15:00 | 974.80 | 970.11 | 969.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 12:15:00 | 976.55 | 971.39 | 970.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-09 13:15:00 | 991.55 | 993.17 | 985.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-09 15:15:00 | 980.00 | 990.50 | 985.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 15:15:00 | 980.00 | 990.50 | 985.39 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 10:15:00 | 983.25 | 983.86 | 983.91 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 09:15:00 | 995.35 | 985.80 | 984.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-15 09:15:00 | 1077.00 | 1009.66 | 997.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 09:15:00 | 1136.10 | 1138.84 | 1117.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 15:15:00 | 1130.00 | 1151.86 | 1142.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 15:15:00 | 1130.00 | 1151.86 | 1142.02 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-26 13:15:00 | 1140.80 | 1144.71 | 1145.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-26 15:15:00 | 1130.75 | 1141.85 | 1143.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 09:15:00 | 1147.60 | 1143.00 | 1144.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 1147.60 | 1143.00 | 1144.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 1147.60 | 1143.00 | 1144.16 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-06-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 12:15:00 | 1150.00 | 1144.23 | 1143.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 15:15:00 | 1154.00 | 1147.85 | 1145.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-30 10:15:00 | 1138.40 | 1146.89 | 1145.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 10:15:00 | 1138.40 | 1146.89 | 1145.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 10:15:00 | 1138.40 | 1146.89 | 1145.78 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-07-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-06 13:15:00 | 1148.05 | 1151.08 | 1151.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-06 14:15:00 | 1138.70 | 1148.60 | 1150.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-07 11:15:00 | 1148.30 | 1141.69 | 1145.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 11:15:00 | 1148.30 | 1141.69 | 1145.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 11:15:00 | 1148.30 | 1141.69 | 1145.71 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 09:15:00 | 1161.00 | 1148.41 | 1147.66 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-11 11:15:00 | 1146.35 | 1149.17 | 1149.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-11 12:15:00 | 1139.10 | 1147.15 | 1148.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 15:15:00 | 1149.00 | 1145.98 | 1147.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 15:15:00 | 1149.00 | 1145.98 | 1147.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 15:15:00 | 1149.00 | 1145.98 | 1147.29 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 09:15:00 | 1248.55 | 1166.50 | 1156.49 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 15:15:00 | 1109.00 | 1159.21 | 1160.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-17 14:15:00 | 1088.00 | 1108.59 | 1121.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-18 09:15:00 | 1146.90 | 1112.38 | 1120.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 09:15:00 | 1146.90 | 1112.38 | 1120.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 1146.90 | 1112.38 | 1120.46 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 12:15:00 | 1141.05 | 1126.71 | 1125.78 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 13:15:00 | 1104.25 | 1122.22 | 1123.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-18 15:15:00 | 1103.00 | 1115.54 | 1120.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-19 09:15:00 | 1117.75 | 1115.98 | 1120.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-19 09:15:00 | 1117.75 | 1115.98 | 1120.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 09:15:00 | 1117.75 | 1115.98 | 1120.10 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 12:15:00 | 1129.25 | 1114.72 | 1113.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-24 09:15:00 | 1145.50 | 1126.05 | 1119.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 15:15:00 | 1142.00 | 1143.12 | 1132.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 14:15:00 | 1150.00 | 1154.50 | 1149.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 14:15:00 | 1150.00 | 1154.50 | 1149.02 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-16 10:15:00 | 1405.10 | 1445.43 | 1449.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-17 10:15:00 | 1377.50 | 1401.95 | 1421.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 09:15:00 | 1435.50 | 1376.25 | 1386.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 09:15:00 | 1435.50 | 1376.25 | 1386.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 1435.50 | 1376.25 | 1386.96 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 11:15:00 | 1435.80 | 1397.21 | 1395.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 09:15:00 | 1460.50 | 1424.84 | 1410.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 13:15:00 | 1436.50 | 1441.83 | 1424.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 15:15:00 | 1427.00 | 1438.37 | 1426.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 15:15:00 | 1427.00 | 1438.37 | 1426.17 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-08-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 10:15:00 | 1418.20 | 1435.62 | 1436.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 09:15:00 | 1400.70 | 1420.56 | 1428.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-25 15:15:00 | 1415.00 | 1413.94 | 1420.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 09:15:00 | 1434.05 | 1417.96 | 1421.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 1434.05 | 1417.96 | 1421.86 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 12:15:00 | 1436.75 | 1426.18 | 1425.03 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 12:15:00 | 1417.35 | 1425.28 | 1426.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 12:15:00 | 1415.00 | 1421.69 | 1423.47 | Break + close below crossover candle low |

### Cycle 26 — BUY (started 2023-08-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 14:15:00 | 1466.70 | 1429.67 | 1426.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 15:15:00 | 1468.00 | 1437.33 | 1430.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 09:15:00 | 1431.90 | 1436.25 | 1430.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 09:15:00 | 1431.90 | 1436.25 | 1430.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 09:15:00 | 1431.90 | 1436.25 | 1430.61 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-01 11:15:00 | 1395.40 | 1422.12 | 1424.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-04 09:15:00 | 1387.95 | 1402.83 | 1413.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-05 10:15:00 | 1400.50 | 1397.61 | 1404.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 13:15:00 | 1394.10 | 1397.96 | 1402.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 13:15:00 | 1394.10 | 1397.96 | 1402.80 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 11:15:00 | 1385.85 | 1382.46 | 1382.27 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-11 13:15:00 | 1371.10 | 1380.39 | 1381.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 09:15:00 | 1361.90 | 1375.59 | 1378.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 10:15:00 | 1370.20 | 1354.75 | 1363.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 10:15:00 | 1370.20 | 1354.75 | 1363.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 10:15:00 | 1370.20 | 1354.75 | 1363.07 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-09-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 12:15:00 | 1412.45 | 1370.60 | 1369.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-13 15:15:00 | 1425.00 | 1393.14 | 1380.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 10:15:00 | 1444.15 | 1448.85 | 1425.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 14:15:00 | 1453.05 | 1466.61 | 1454.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 14:15:00 | 1453.05 | 1466.61 | 1454.10 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 13:15:00 | 1428.40 | 1446.63 | 1448.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 15:15:00 | 1423.40 | 1439.86 | 1444.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 11:15:00 | 1397.90 | 1390.03 | 1402.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 12:15:00 | 1392.15 | 1390.46 | 1401.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 12:15:00 | 1392.15 | 1390.46 | 1401.69 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-09-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 09:15:00 | 1401.85 | 1392.77 | 1392.35 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 1381.55 | 1392.71 | 1392.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-29 09:15:00 | 1376.80 | 1388.13 | 1390.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-03 09:15:00 | 1399.00 | 1380.52 | 1384.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 1399.00 | 1380.52 | 1384.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 1399.00 | 1380.52 | 1384.10 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 11:15:00 | 1403.70 | 1387.79 | 1386.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 12:15:00 | 1420.30 | 1394.29 | 1389.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 09:15:00 | 1402.10 | 1406.59 | 1398.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 09:15:00 | 1402.10 | 1406.59 | 1398.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 1402.10 | 1406.59 | 1398.12 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 10:15:00 | 1386.50 | 1403.05 | 1403.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 12:15:00 | 1370.55 | 1393.57 | 1399.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 1403.55 | 1386.40 | 1393.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 1403.55 | 1386.40 | 1393.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 1403.55 | 1386.40 | 1393.05 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 14:15:00 | 1402.35 | 1396.41 | 1396.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 15:15:00 | 1410.00 | 1399.13 | 1397.47 | Break + close above crossover candle high |

### Cycle 37 — SELL (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-11 09:15:00 | 1379.10 | 1395.12 | 1395.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-11 13:15:00 | 1368.60 | 1385.00 | 1390.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-12 13:15:00 | 1373.55 | 1365.60 | 1375.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 14:15:00 | 1381.45 | 1368.77 | 1376.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 14:15:00 | 1381.45 | 1368.77 | 1376.32 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 10:15:00 | 1382.85 | 1377.04 | 1376.89 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 13:15:00 | 1369.00 | 1376.24 | 1376.66 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 14:15:00 | 1384.80 | 1375.80 | 1375.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-18 09:15:00 | 1387.20 | 1378.12 | 1376.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 12:15:00 | 1379.10 | 1380.01 | 1378.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 12:15:00 | 1379.10 | 1380.01 | 1378.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 12:15:00 | 1379.10 | 1380.01 | 1378.07 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 09:15:00 | 1334.25 | 1374.66 | 1379.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 10:15:00 | 1332.35 | 1366.20 | 1374.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 13:15:00 | 1336.25 | 1326.75 | 1341.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 14:15:00 | 1344.10 | 1330.22 | 1342.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 14:15:00 | 1344.10 | 1330.22 | 1342.13 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 09:15:00 | 1357.95 | 1342.97 | 1341.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 14:15:00 | 1373.05 | 1357.00 | 1349.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 10:15:00 | 1393.95 | 1398.76 | 1387.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 12:15:00 | 1385.10 | 1394.79 | 1387.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 12:15:00 | 1385.10 | 1394.79 | 1387.48 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2023-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 11:15:00 | 1389.10 | 1425.81 | 1426.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-06 13:15:00 | 1384.20 | 1411.94 | 1419.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-07 11:15:00 | 1397.90 | 1393.24 | 1405.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 12:15:00 | 1409.30 | 1396.45 | 1405.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 12:15:00 | 1409.30 | 1396.45 | 1405.95 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 09:15:00 | 1436.20 | 1409.71 | 1409.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-09 09:15:00 | 1533.85 | 1447.90 | 1429.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 14:15:00 | 1473.00 | 1474.17 | 1452.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 09:15:00 | 1447.00 | 1468.16 | 1453.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 1447.00 | 1468.16 | 1453.35 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2023-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 10:15:00 | 1429.65 | 1446.38 | 1448.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 14:15:00 | 1420.30 | 1433.59 | 1440.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 09:15:00 | 1452.65 | 1435.55 | 1440.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 09:15:00 | 1452.65 | 1435.55 | 1440.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 1452.65 | 1435.55 | 1440.33 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2023-11-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 11:15:00 | 1462.75 | 1444.25 | 1443.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 12:15:00 | 1483.75 | 1452.15 | 1447.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 10:15:00 | 1453.00 | 1467.25 | 1458.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 10:15:00 | 1453.00 | 1467.25 | 1458.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 10:15:00 | 1453.00 | 1467.25 | 1458.31 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2023-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 12:15:00 | 1519.00 | 1522.77 | 1523.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 13:15:00 | 1513.55 | 1520.93 | 1522.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-22 15:15:00 | 1520.00 | 1519.58 | 1521.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 09:15:00 | 1520.65 | 1519.79 | 1521.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 1520.65 | 1519.79 | 1521.26 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2023-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 11:15:00 | 1533.90 | 1521.50 | 1520.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 09:15:00 | 1549.45 | 1530.50 | 1525.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-28 13:15:00 | 1536.00 | 1541.67 | 1533.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 13:15:00 | 1536.00 | 1541.67 | 1533.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 13:15:00 | 1536.00 | 1541.67 | 1533.31 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2023-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 15:15:00 | 1487.00 | 1525.04 | 1526.85 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2023-11-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 12:15:00 | 1533.40 | 1524.17 | 1523.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 09:15:00 | 1548.75 | 1529.29 | 1526.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 09:15:00 | 1634.35 | 1653.82 | 1642.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 09:15:00 | 1634.35 | 1653.82 | 1642.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 09:15:00 | 1634.35 | 1653.82 | 1642.58 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 12:15:00 | 1589.25 | 1631.35 | 1634.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 13:15:00 | 1565.50 | 1593.51 | 1609.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 10:15:00 | 1587.55 | 1585.29 | 1599.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 14:15:00 | 1593.65 | 1585.65 | 1595.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 14:15:00 | 1593.65 | 1585.65 | 1595.17 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2023-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 14:15:00 | 1604.10 | 1598.24 | 1597.77 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2023-12-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-14 11:15:00 | 1587.55 | 1597.08 | 1597.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-14 14:15:00 | 1581.00 | 1591.03 | 1594.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-15 14:15:00 | 1564.40 | 1554.00 | 1570.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 09:15:00 | 1451.60 | 1449.04 | 1459.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 09:15:00 | 1451.60 | 1449.04 | 1459.46 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2023-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 11:15:00 | 1542.60 | 1469.54 | 1467.08 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 09:15:00 | 1475.30 | 1486.09 | 1486.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-01 14:15:00 | 1471.00 | 1483.53 | 1485.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-03 15:15:00 | 1450.00 | 1445.79 | 1454.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 09:15:00 | 1442.55 | 1445.14 | 1453.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 09:15:00 | 1442.55 | 1445.14 | 1453.29 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2024-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 11:15:00 | 1469.55 | 1450.69 | 1450.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 14:15:00 | 1484.95 | 1460.67 | 1455.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 12:15:00 | 1505.95 | 1508.50 | 1493.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 09:15:00 | 1490.35 | 1504.45 | 1496.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 1490.35 | 1504.45 | 1496.14 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 12:15:00 | 1468.80 | 1488.82 | 1490.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 13:15:00 | 1464.65 | 1483.98 | 1488.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 10:15:00 | 1481.90 | 1478.55 | 1483.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 11:15:00 | 1480.05 | 1478.85 | 1483.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 11:15:00 | 1480.05 | 1478.85 | 1483.24 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 13:15:00 | 1436.15 | 1408.94 | 1407.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 14:15:00 | 1452.40 | 1417.63 | 1411.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 09:15:00 | 1581.80 | 1595.32 | 1571.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 09:15:00 | 1581.80 | 1595.32 | 1571.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 1581.80 | 1595.32 | 1571.02 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 1533.50 | 1562.33 | 1564.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 12:15:00 | 1513.90 | 1531.84 | 1544.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 11:15:00 | 1521.50 | 1517.43 | 1530.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 12:15:00 | 1542.10 | 1522.37 | 1531.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 12:15:00 | 1542.10 | 1522.37 | 1531.56 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 12:15:00 | 1540.10 | 1535.94 | 1535.70 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-02-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 13:15:00 | 1529.60 | 1534.67 | 1535.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-14 15:15:00 | 1526.15 | 1533.18 | 1534.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-15 09:15:00 | 1534.20 | 1533.39 | 1534.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-15 09:15:00 | 1534.20 | 1533.39 | 1534.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 09:15:00 | 1534.20 | 1533.39 | 1534.38 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-02-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 11:15:00 | 1543.45 | 1535.70 | 1535.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 12:15:00 | 1551.35 | 1538.83 | 1536.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 13:15:00 | 1535.45 | 1538.15 | 1536.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-15 13:15:00 | 1535.45 | 1538.15 | 1536.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 13:15:00 | 1535.45 | 1538.15 | 1536.62 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-02-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 15:15:00 | 1529.00 | 1535.66 | 1535.71 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 09:15:00 | 1564.85 | 1541.50 | 1538.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 10:15:00 | 1575.90 | 1548.38 | 1541.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 14:15:00 | 1581.55 | 1583.98 | 1570.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 15:15:00 | 1565.00 | 1580.19 | 1570.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 15:15:00 | 1565.00 | 1580.19 | 1570.42 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 09:15:00 | 1583.85 | 1595.72 | 1596.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-23 10:15:00 | 1570.45 | 1590.67 | 1594.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 14:15:00 | 1610.00 | 1587.92 | 1591.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 14:15:00 | 1610.00 | 1587.92 | 1591.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 14:15:00 | 1610.00 | 1587.92 | 1591.15 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 14:15:00 | 1629.95 | 1590.28 | 1586.71 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-02-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 09:15:00 | 1575.65 | 1590.22 | 1591.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-29 10:15:00 | 1556.35 | 1583.44 | 1587.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-01 09:15:00 | 1580.00 | 1575.78 | 1581.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 09:15:00 | 1580.00 | 1575.78 | 1581.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 1580.00 | 1575.78 | 1581.21 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-03-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 13:15:00 | 1592.60 | 1585.36 | 1584.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 14:15:00 | 1623.35 | 1592.96 | 1588.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 11:15:00 | 1610.40 | 1612.50 | 1602.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 13:15:00 | 1597.15 | 1609.35 | 1602.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 13:15:00 | 1597.15 | 1609.35 | 1602.77 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 10:15:00 | 1581.95 | 1598.63 | 1599.47 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-05 12:15:00 | 1604.15 | 1600.59 | 1600.27 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 1559.35 | 1592.16 | 1596.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 10:15:00 | 1542.00 | 1569.51 | 1579.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 11:15:00 | 1413.70 | 1412.70 | 1449.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 09:15:00 | 1424.05 | 1415.11 | 1436.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 1424.05 | 1415.11 | 1436.55 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 11:15:00 | 1491.35 | 1444.33 | 1439.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 14:15:00 | 1502.35 | 1469.02 | 1452.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 10:15:00 | 1474.10 | 1475.98 | 1460.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 1519.30 | 1497.40 | 1484.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 1519.30 | 1497.40 | 1484.62 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2024-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-02 15:15:00 | 1560.00 | 1573.23 | 1574.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-03 09:15:00 | 1559.00 | 1570.39 | 1572.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-03 12:15:00 | 1585.00 | 1568.69 | 1571.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-03 12:15:00 | 1585.00 | 1568.69 | 1571.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 12:15:00 | 1585.00 | 1568.69 | 1571.05 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2024-04-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 09:15:00 | 1584.05 | 1567.38 | 1567.08 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 10:15:00 | 1553.10 | 1566.15 | 1567.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 12:15:00 | 1552.95 | 1562.16 | 1565.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-09 09:15:00 | 1570.70 | 1555.13 | 1560.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 09:15:00 | 1570.70 | 1555.13 | 1560.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 09:15:00 | 1570.70 | 1555.13 | 1560.19 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 14:15:00 | 1573.30 | 1563.52 | 1563.09 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 13:15:00 | 1554.00 | 1562.92 | 1563.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 09:15:00 | 1544.45 | 1558.76 | 1561.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-12 11:15:00 | 1559.70 | 1558.35 | 1560.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-12 12:00:00 | 1559.70 | 1558.35 | 1560.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 14:15:00 | 1544.15 | 1554.55 | 1558.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 14:30:00 | 1556.05 | 1554.55 | 1558.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 15:15:00 | 1546.00 | 1552.84 | 1557.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 09:15:00 | 1517.30 | 1552.84 | 1557.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 11:15:00 | 1539.60 | 1545.90 | 1552.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 14:15:00 | 1532.30 | 1545.52 | 1551.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 15:15:00 | 1538.25 | 1546.18 | 1550.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 15:15:00 | 1538.25 | 1544.60 | 1549.68 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-04-16 13:15:00 | 1570.15 | 1552.56 | 1551.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 13:15:00 | 1570.15 | 1552.56 | 1551.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-16 14:15:00 | 1572.25 | 1556.50 | 1553.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 15:15:00 | 1553.20 | 1555.84 | 1553.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 15:15:00 | 1553.20 | 1555.84 | 1553.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 15:15:00 | 1553.20 | 1555.84 | 1553.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:15:00 | 1548.80 | 1555.84 | 1553.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 1551.55 | 1554.98 | 1553.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 10:00:00 | 1551.55 | 1554.98 | 1553.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 1580.40 | 1560.07 | 1555.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 10:30:00 | 1608.85 | 1575.23 | 1566.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 10:15:00 | 1601.20 | 1575.34 | 1570.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-23 09:15:00 | 1543.00 | 1564.96 | 1567.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 09:15:00 | 1543.00 | 1564.96 | 1567.95 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 11:15:00 | 1570.45 | 1559.43 | 1558.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 14:15:00 | 1581.75 | 1567.84 | 1562.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 10:15:00 | 1631.15 | 1634.62 | 1619.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-30 11:00:00 | 1631.15 | 1634.62 | 1619.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 1625.30 | 1634.88 | 1626.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 10:00:00 | 1625.30 | 1634.88 | 1626.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 10:15:00 | 1612.40 | 1630.39 | 1625.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 11:00:00 | 1612.40 | 1630.39 | 1625.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 11:15:00 | 1629.20 | 1630.15 | 1625.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 12:45:00 | 1637.95 | 1631.73 | 1626.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 09:15:00 | 1607.20 | 1645.49 | 1649.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 09:15:00 | 1607.20 | 1645.49 | 1649.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 10:15:00 | 1585.80 | 1633.55 | 1643.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 12:15:00 | 1599.75 | 1598.09 | 1613.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 13:00:00 | 1599.75 | 1598.09 | 1613.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 14:15:00 | 1610.10 | 1601.60 | 1612.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 14:45:00 | 1608.00 | 1601.60 | 1612.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 15:15:00 | 1604.95 | 1602.27 | 1611.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 09:15:00 | 1587.25 | 1602.73 | 1608.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 13:15:00 | 1590.05 | 1598.62 | 1603.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 11:15:00 | 1589.90 | 1601.22 | 1603.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 11:15:00 | 1620.75 | 1605.12 | 1605.29 | SL hit (close>static) qty=1.00 sl=1613.55 alert=retest2 |

### Cycle 82 — BUY (started 2024-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 12:15:00 | 1621.35 | 1608.37 | 1606.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 14:15:00 | 1649.30 | 1618.54 | 1611.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 14:15:00 | 1675.00 | 1677.91 | 1661.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 15:00:00 | 1675.00 | 1677.91 | 1661.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 1675.00 | 1677.33 | 1662.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:30:00 | 1665.35 | 1670.86 | 1660.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 1653.35 | 1667.36 | 1660.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 11:45:00 | 1653.90 | 1665.49 | 1659.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 12:45:00 | 1657.00 | 1663.18 | 1659.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 14:45:00 | 1656.00 | 1659.65 | 1658.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:45:00 | 1658.95 | 1661.62 | 1659.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 1661.50 | 1661.60 | 1659.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 10:30:00 | 1659.40 | 1661.60 | 1659.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 11:15:00 | 1664.25 | 1662.13 | 1659.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 12:45:00 | 1676.60 | 1664.70 | 1661.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-17 13:15:00 | 1651.60 | 1662.08 | 1660.46 | SL hit (close<static) qty=1.00 sl=1656.10 alert=retest2 |

### Cycle 83 — SELL (started 2024-05-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 14:15:00 | 1642.95 | 1658.26 | 1658.87 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 09:15:00 | 1790.00 | 1679.85 | 1667.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 10:15:00 | 1828.90 | 1709.66 | 1682.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 14:15:00 | 1748.85 | 1750.81 | 1714.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 15:00:00 | 1748.85 | 1750.81 | 1714.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 1799.10 | 1801.06 | 1792.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:45:00 | 1798.00 | 1801.06 | 1792.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 12:15:00 | 1803.50 | 1804.56 | 1796.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 12:45:00 | 1805.35 | 1804.56 | 1796.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 13:15:00 | 1802.95 | 1804.24 | 1797.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 14:00:00 | 1802.95 | 1804.24 | 1797.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 15:15:00 | 1799.60 | 1803.23 | 1798.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 09:15:00 | 1827.95 | 1803.23 | 1798.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 09:15:00 | 1791.05 | 1800.79 | 1797.37 | SL hit (close<static) qty=1.00 sl=1797.75 alert=retest2 |

### Cycle 85 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 1771.55 | 1792.40 | 1794.22 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 14:15:00 | 1810.20 | 1795.99 | 1795.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 11:15:00 | 1822.75 | 1807.91 | 1801.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 12:15:00 | 1799.70 | 1820.50 | 1814.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 12:15:00 | 1799.70 | 1820.50 | 1814.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 12:15:00 | 1799.70 | 1820.50 | 1814.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:00:00 | 1799.70 | 1820.50 | 1814.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 1795.85 | 1815.57 | 1812.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:45:00 | 1788.35 | 1815.57 | 1812.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2024-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 14:15:00 | 1778.00 | 1808.05 | 1809.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 15:15:00 | 1755.25 | 1790.22 | 1798.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 1813.40 | 1794.86 | 1799.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 1813.40 | 1794.86 | 1799.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1813.40 | 1794.86 | 1799.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:45:00 | 1795.20 | 1794.26 | 1798.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 13:30:00 | 1799.75 | 1799.66 | 1800.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 14:15:00 | 1799.10 | 1799.66 | 1800.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 1709.76 | 1792.80 | 1797.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 1709.14 | 1792.80 | 1797.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 11:15:00 | 1705.44 | 1760.78 | 1781.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 12:15:00 | 1615.68 | 1751.99 | 1775.28 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 88 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 1848.25 | 1773.42 | 1766.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 1858.85 | 1827.18 | 1811.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 10:15:00 | 1838.40 | 1843.80 | 1830.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-11 10:15:00 | 1838.40 | 1843.80 | 1830.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 1838.40 | 1843.80 | 1830.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 10:30:00 | 1839.10 | 1843.80 | 1830.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 11:15:00 | 1831.05 | 1841.25 | 1830.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 11:45:00 | 1835.95 | 1841.25 | 1830.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 12:15:00 | 1820.35 | 1837.07 | 1829.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 13:00:00 | 1820.35 | 1837.07 | 1829.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 13:15:00 | 1826.95 | 1835.05 | 1829.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 1840.95 | 1833.01 | 1829.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 11:15:00 | 1844.10 | 1834.44 | 1830.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 09:15:00 | 2025.05 | 1943.91 | 1906.19 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 14:15:00 | 1992.95 | 2014.34 | 2014.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 10:15:00 | 1977.35 | 2000.00 | 2005.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 1994.05 | 1978.44 | 1990.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 09:15:00 | 1994.05 | 1978.44 | 1990.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 1994.05 | 1978.44 | 1990.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 12:30:00 | 1947.95 | 1973.65 | 1985.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 10:15:00 | 1986.00 | 1977.10 | 1976.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 10:15:00 | 1986.00 | 1977.10 | 1976.30 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 1968.55 | 1975.46 | 1975.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 12:15:00 | 1957.70 | 1967.52 | 1971.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 09:15:00 | 1975.55 | 1964.42 | 1968.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 09:15:00 | 1975.55 | 1964.42 | 1968.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1975.55 | 1964.42 | 1968.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:30:00 | 1971.90 | 1964.42 | 1968.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 1977.70 | 1967.07 | 1968.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:45:00 | 1976.25 | 1967.07 | 1968.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2024-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 11:15:00 | 2022.00 | 1978.06 | 1973.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 12:15:00 | 2059.25 | 1994.30 | 1981.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 12:15:00 | 2137.65 | 2140.83 | 2118.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-10 13:00:00 | 2137.65 | 2140.83 | 2118.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 2131.00 | 2142.62 | 2133.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:30:00 | 2121.30 | 2142.62 | 2133.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 2123.15 | 2138.73 | 2132.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:00:00 | 2123.15 | 2138.73 | 2132.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 2103.45 | 2131.67 | 2130.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:00:00 | 2103.45 | 2131.67 | 2130.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 12:15:00 | 2114.85 | 2128.31 | 2128.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 09:15:00 | 2081.60 | 2113.72 | 2121.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 12:15:00 | 2101.50 | 2101.20 | 2112.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-15 13:00:00 | 2101.50 | 2101.20 | 2112.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 2116.85 | 2104.33 | 2113.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 13:30:00 | 2117.30 | 2104.33 | 2113.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 2116.10 | 2106.68 | 2113.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 15:00:00 | 2116.10 | 2106.68 | 2113.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 2120.95 | 2109.54 | 2114.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:15:00 | 2126.20 | 2109.54 | 2114.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 2147.20 | 2117.07 | 2117.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:45:00 | 2143.10 | 2117.07 | 2117.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 10:15:00 | 2142.10 | 2122.07 | 2119.37 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 2086.85 | 2117.34 | 2119.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 2077.85 | 2102.47 | 2110.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 2124.65 | 2092.12 | 2098.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 2124.65 | 2092.12 | 2098.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 2124.65 | 2092.12 | 2098.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 2124.65 | 2092.12 | 2098.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 2126.90 | 2099.08 | 2101.21 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2024-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 11:15:00 | 2130.25 | 2105.31 | 2103.85 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 13:15:00 | 2030.40 | 2094.44 | 2099.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 12:15:00 | 2013.85 | 2050.62 | 2072.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 15:15:00 | 2031.05 | 2028.01 | 2043.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-25 09:15:00 | 2024.05 | 2028.01 | 2043.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 2005.10 | 2023.43 | 2039.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 11:15:00 | 2001.00 | 2021.96 | 2037.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 1994.00 | 2010.69 | 2024.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 13:00:00 | 1997.00 | 2005.82 | 2017.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 14:45:00 | 1999.05 | 2003.51 | 2014.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 1971.35 | 1996.99 | 2009.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 10:15:00 | 1969.45 | 1996.99 | 2009.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 11:45:00 | 1966.65 | 1971.76 | 1985.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 14:00:00 | 1965.10 | 1969.35 | 1981.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 11:15:00 | 1900.95 | 1943.16 | 1964.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 11:15:00 | 1894.30 | 1943.16 | 1964.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 11:15:00 | 1897.15 | 1943.16 | 1964.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 11:15:00 | 1899.10 | 1943.16 | 1964.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 15:15:00 | 1870.98 | 1904.84 | 1937.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 15:15:00 | 1868.32 | 1904.84 | 1937.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 15:15:00 | 1866.84 | 1904.84 | 1937.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-02 09:15:00 | 1860.90 | 1855.80 | 1887.95 | SL hit (close>ema200) qty=0.50 sl=1855.80 alert=retest2 |

### Cycle 98 — BUY (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 14:15:00 | 1853.10 | 1843.77 | 1843.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 15:15:00 | 1859.95 | 1847.00 | 1845.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 1866.35 | 1871.40 | 1861.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 14:15:00 | 1866.35 | 1871.40 | 1861.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 1866.35 | 1871.40 | 1861.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 1866.35 | 1871.40 | 1861.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 1875.00 | 1872.12 | 1862.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 1887.65 | 1872.12 | 1862.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 10:00:00 | 1875.85 | 1872.87 | 1863.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 13:00:00 | 1875.90 | 1872.05 | 1865.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 1893.25 | 1875.26 | 1868.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 1933.50 | 1921.81 | 1905.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-14 12:15:00 | 1867.20 | 1896.72 | 1900.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 12:15:00 | 1867.20 | 1896.72 | 1900.45 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 1958.30 | 1907.44 | 1902.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 1991.25 | 1945.39 | 1925.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 13:15:00 | 2160.40 | 2163.97 | 2116.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 14:00:00 | 2160.40 | 2163.97 | 2116.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 2148.20 | 2168.68 | 2143.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:00:00 | 2148.20 | 2168.68 | 2143.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 2153.95 | 2165.73 | 2144.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 15:15:00 | 2154.00 | 2165.73 | 2144.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 2154.00 | 2163.39 | 2145.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:15:00 | 2180.10 | 2163.39 | 2145.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-26 09:15:00 | 2398.11 | 2220.57 | 2185.74 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 12:15:00 | 2497.25 | 2524.06 | 2526.31 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 2544.05 | 2522.16 | 2521.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 15:15:00 | 2547.00 | 2527.13 | 2524.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 11:15:00 | 2526.45 | 2530.12 | 2526.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 11:15:00 | 2526.45 | 2530.12 | 2526.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 2526.45 | 2530.12 | 2526.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 15:00:00 | 2547.70 | 2532.84 | 2528.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:45:00 | 2540.30 | 2532.90 | 2529.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 10:15:00 | 2517.75 | 2529.87 | 2528.35 | SL hit (close<static) qty=1.00 sl=2521.25 alert=retest2 |

### Cycle 103 — SELL (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 12:15:00 | 2525.15 | 2527.35 | 2527.37 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 2530.00 | 2527.68 | 2527.44 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 09:15:00 | 2517.40 | 2525.62 | 2526.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-13 12:15:00 | 2504.95 | 2517.68 | 2522.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 14:15:00 | 2540.05 | 2520.30 | 2522.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 14:15:00 | 2540.05 | 2520.30 | 2522.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 2540.05 | 2520.30 | 2522.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 15:00:00 | 2540.05 | 2520.30 | 2522.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-09-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 15:15:00 | 2558.80 | 2528.00 | 2525.91 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 11:15:00 | 2475.60 | 2515.55 | 2520.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 14:15:00 | 2448.60 | 2497.68 | 2510.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 15:15:00 | 2450.00 | 2442.67 | 2468.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-18 09:15:00 | 2496.15 | 2442.67 | 2468.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 2499.40 | 2454.02 | 2470.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:45:00 | 2512.60 | 2454.02 | 2470.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 2546.00 | 2472.41 | 2477.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:45:00 | 2563.35 | 2472.41 | 2477.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 12:15:00 | 2502.90 | 2484.82 | 2482.81 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 10:15:00 | 2470.70 | 2482.39 | 2483.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 13:15:00 | 2451.15 | 2470.55 | 2477.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 15:15:00 | 2476.50 | 2469.88 | 2475.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 15:15:00 | 2476.50 | 2469.88 | 2475.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 2476.50 | 2469.88 | 2475.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 2453.30 | 2469.88 | 2475.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 2451.50 | 2466.21 | 2473.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 10:15:00 | 2440.75 | 2466.21 | 2473.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 09:15:00 | 2318.71 | 2392.31 | 2426.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-24 11:15:00 | 2363.75 | 2360.04 | 2384.84 | SL hit (close>ema200) qty=0.50 sl=2360.04 alert=retest2 |

### Cycle 110 — BUY (started 2024-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 09:15:00 | 2386.80 | 2322.76 | 2320.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 10:15:00 | 2392.85 | 2336.78 | 2327.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 14:15:00 | 2353.35 | 2355.47 | 2341.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 15:00:00 | 2353.35 | 2355.47 | 2341.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 2294.20 | 2345.40 | 2339.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 10:00:00 | 2294.20 | 2345.40 | 2339.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 2302.95 | 2336.91 | 2335.80 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 2302.75 | 2330.08 | 2332.80 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 10:15:00 | 2357.85 | 2329.98 | 2329.67 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 11:15:00 | 2319.45 | 2327.87 | 2328.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 12:15:00 | 2315.80 | 2325.46 | 2327.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 14:15:00 | 2353.80 | 2328.30 | 2328.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 14:15:00 | 2353.80 | 2328.30 | 2328.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 2353.80 | 2328.30 | 2328.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 15:00:00 | 2353.80 | 2328.30 | 2328.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2024-10-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 15:15:00 | 2349.00 | 2332.44 | 2330.19 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 2300.95 | 2326.14 | 2327.54 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 2344.10 | 2323.75 | 2321.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 15:15:00 | 2358.00 | 2330.60 | 2324.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 14:15:00 | 2340.00 | 2352.12 | 2341.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 14:15:00 | 2340.00 | 2352.12 | 2341.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 2340.00 | 2352.12 | 2341.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 15:00:00 | 2340.00 | 2352.12 | 2341.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 15:15:00 | 2336.75 | 2349.04 | 2340.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 09:15:00 | 2364.40 | 2349.04 | 2340.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 11:45:00 | 2355.40 | 2353.53 | 2345.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 12:15:00 | 2320.10 | 2346.20 | 2348.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2024-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 12:15:00 | 2320.10 | 2346.20 | 2348.10 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 09:15:00 | 2484.10 | 2369.37 | 2357.46 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 09:15:00 | 2483.55 | 2548.10 | 2552.28 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 2635.65 | 2564.50 | 2558.27 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 12:15:00 | 2541.40 | 2562.43 | 2563.52 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 15:15:00 | 2578.00 | 2563.89 | 2563.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 2664.00 | 2583.91 | 2572.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 2905.35 | 2926.76 | 2828.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 14:15:00 | 2863.95 | 2916.90 | 2861.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 2863.95 | 2916.90 | 2861.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 14:30:00 | 2847.40 | 2916.90 | 2861.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 2879.00 | 2909.32 | 2862.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:15:00 | 2894.00 | 2909.32 | 2862.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 2889.15 | 2905.29 | 2865.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 10:15:00 | 2906.15 | 2905.29 | 2865.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 13:15:00 | 2839.25 | 2874.63 | 2862.04 | SL hit (close<static) qty=1.00 sl=2843.20 alert=retest2 |

### Cycle 123 — SELL (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 15:15:00 | 2815.00 | 2852.75 | 2853.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 09:15:00 | 2796.95 | 2834.70 | 2843.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 12:15:00 | 2835.55 | 2831.78 | 2839.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 12:15:00 | 2835.55 | 2831.78 | 2839.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 2835.55 | 2831.78 | 2839.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:00:00 | 2835.55 | 2831.78 | 2839.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 2838.40 | 2833.10 | 2839.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:15:00 | 2840.55 | 2833.10 | 2839.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 2845.90 | 2835.66 | 2839.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 2845.90 | 2835.66 | 2839.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 2855.00 | 2839.53 | 2841.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 2829.85 | 2839.53 | 2841.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 2818.70 | 2835.36 | 2839.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 14:00:00 | 2791.30 | 2811.45 | 2825.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 15:00:00 | 2790.45 | 2807.25 | 2822.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-13 09:15:00 | 2512.17 | 2691.45 | 2736.02 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 2797.45 | 2641.43 | 2626.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 11:15:00 | 2855.60 | 2684.26 | 2646.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 2779.00 | 2818.38 | 2737.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 09:45:00 | 2757.15 | 2818.38 | 2737.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 2772.00 | 2786.85 | 2754.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:15:00 | 2682.15 | 2786.85 | 2754.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 2682.35 | 2765.95 | 2748.04 | EMA400 retest candle locked (from upside) |

### Cycle 125 — SELL (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 12:15:00 | 2696.40 | 2731.86 | 2735.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 14:15:00 | 2686.80 | 2716.49 | 2727.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 2740.75 | 2719.03 | 2726.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 09:15:00 | 2740.75 | 2719.03 | 2726.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 2740.75 | 2719.03 | 2726.30 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2024-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 14:15:00 | 2737.00 | 2729.90 | 2729.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 15:15:00 | 2745.90 | 2733.10 | 2731.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 13:15:00 | 2749.15 | 2750.58 | 2742.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 13:15:00 | 2749.15 | 2750.58 | 2742.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 2749.15 | 2750.58 | 2742.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:45:00 | 2748.35 | 2750.58 | 2742.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 2778.00 | 2756.06 | 2745.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:30:00 | 2753.10 | 2756.06 | 2745.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 2713.75 | 2754.63 | 2747.09 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 11:15:00 | 2713.95 | 2738.47 | 2740.56 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 2795.70 | 2749.38 | 2744.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 10:15:00 | 2802.65 | 2760.03 | 2749.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 10:15:00 | 2869.35 | 2870.17 | 2835.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 10:45:00 | 2862.45 | 2870.17 | 2835.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 2827.05 | 2884.88 | 2860.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 2827.05 | 2884.88 | 2860.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 2809.55 | 2869.81 | 2855.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:00:00 | 2809.55 | 2869.81 | 2855.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2024-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 13:15:00 | 2813.80 | 2843.15 | 2845.59 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 11:15:00 | 2858.20 | 2845.05 | 2844.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 12:15:00 | 2875.00 | 2851.04 | 2847.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 13:15:00 | 2922.05 | 2930.39 | 2906.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-11 13:30:00 | 2926.70 | 2930.39 | 2906.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 2947.80 | 2933.87 | 2909.88 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 2834.25 | 2898.21 | 2903.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 2825.75 | 2841.36 | 2855.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 10:15:00 | 2665.75 | 2659.66 | 2702.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 10:15:00 | 2665.75 | 2659.66 | 2702.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 2665.75 | 2659.66 | 2702.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:45:00 | 2709.15 | 2659.66 | 2702.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 2548.40 | 2521.35 | 2537.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:45:00 | 2571.00 | 2521.35 | 2537.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 2522.30 | 2521.54 | 2535.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:30:00 | 2559.90 | 2521.54 | 2535.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 2532.90 | 2523.86 | 2533.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:30:00 | 2530.25 | 2523.86 | 2533.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 2526.65 | 2524.42 | 2532.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:45:00 | 2528.75 | 2524.42 | 2532.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 2550.55 | 2529.64 | 2534.25 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 09:15:00 | 2568.65 | 2537.45 | 2537.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 14:15:00 | 2622.50 | 2559.91 | 2548.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 11:15:00 | 2583.60 | 2585.46 | 2566.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-31 11:45:00 | 2582.55 | 2585.46 | 2566.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 2780.00 | 2805.75 | 2769.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:00:00 | 2780.00 | 2805.75 | 2769.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 12:15:00 | 2758.10 | 2812.28 | 2798.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 13:00:00 | 2758.10 | 2812.28 | 2798.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 2752.30 | 2800.28 | 2794.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 13:45:00 | 2746.80 | 2800.28 | 2794.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2025-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 14:15:00 | 2735.85 | 2787.40 | 2789.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 09:15:00 | 2710.05 | 2768.66 | 2780.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 15:15:00 | 2490.00 | 2488.26 | 2517.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-17 09:15:00 | 2487.25 | 2488.26 | 2517.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 2522.80 | 2496.51 | 2516.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:15:00 | 2534.35 | 2496.51 | 2516.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 2517.70 | 2500.75 | 2516.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 12:15:00 | 2512.75 | 2500.75 | 2516.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 13:00:00 | 2513.00 | 2503.20 | 2516.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 13:30:00 | 2511.50 | 2504.96 | 2515.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 14:00:00 | 2512.00 | 2504.96 | 2515.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 2511.80 | 2506.33 | 2515.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:45:00 | 2517.10 | 2506.33 | 2515.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 2515.00 | 2508.06 | 2515.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 2521.55 | 2508.06 | 2515.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 2525.95 | 2511.64 | 2516.25 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-20 14:15:00 | 2521.45 | 2518.84 | 2518.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 14:15:00 | 2521.45 | 2518.84 | 2518.48 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 2463.60 | 2509.48 | 2515.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 2377.55 | 2476.05 | 2498.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 10:15:00 | 2426.00 | 2411.35 | 2449.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 10:15:00 | 2426.00 | 2411.35 | 2449.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 2426.00 | 2411.35 | 2449.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 2426.00 | 2411.35 | 2449.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 2417.30 | 2400.15 | 2420.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 12:00:00 | 2417.30 | 2400.15 | 2420.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 12:15:00 | 2420.85 | 2404.29 | 2420.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:00:00 | 2404.35 | 2404.30 | 2419.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 15:15:00 | 2462.00 | 2420.60 | 2424.53 | SL hit (close>static) qty=1.00 sl=2445.00 alert=retest2 |

### Cycle 136 — BUY (started 2025-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 13:15:00 | 2315.45 | 2309.08 | 2308.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 14:15:00 | 2331.30 | 2313.52 | 2310.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 10:15:00 | 2316.90 | 2318.62 | 2313.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 11:00:00 | 2316.90 | 2318.62 | 2313.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 2314.20 | 2317.74 | 2313.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 2299.80 | 2317.74 | 2313.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 2286.05 | 2311.40 | 2311.41 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 13:15:00 | 2343.90 | 2317.90 | 2314.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 14:15:00 | 2456.10 | 2345.54 | 2327.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 2360.15 | 2364.99 | 2340.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-03 09:45:00 | 2358.20 | 2364.99 | 2340.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 2336.60 | 2359.31 | 2339.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:30:00 | 2336.90 | 2359.31 | 2339.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 11:15:00 | 2305.85 | 2348.62 | 2336.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 11:45:00 | 2303.35 | 2348.62 | 2336.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 2288.55 | 2336.61 | 2332.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 12:30:00 | 2281.00 | 2336.61 | 2332.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2025-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 14:15:00 | 2292.85 | 2323.43 | 2326.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 15:15:00 | 2281.00 | 2314.94 | 2322.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 2374.60 | 2326.88 | 2327.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 2374.60 | 2326.88 | 2327.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 2374.60 | 2326.88 | 2327.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 2374.60 | 2326.88 | 2327.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — BUY (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 10:15:00 | 2413.65 | 2344.23 | 2335.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 11:15:00 | 2530.00 | 2381.38 | 2353.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 2440.30 | 2458.86 | 2427.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 2440.30 | 2458.86 | 2427.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 2440.30 | 2458.86 | 2427.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 2428.55 | 2458.86 | 2427.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 2407.95 | 2448.68 | 2425.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:00:00 | 2407.95 | 2448.68 | 2425.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 2392.00 | 2437.34 | 2422.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:30:00 | 2394.05 | 2437.34 | 2422.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 2407.55 | 2431.38 | 2421.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 15:15:00 | 2435.00 | 2424.65 | 2419.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:00:00 | 2435.00 | 2426.57 | 2421.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:45:00 | 2434.70 | 2427.60 | 2422.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 14:15:00 | 2410.80 | 2418.58 | 2419.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 14:15:00 | 2410.80 | 2418.58 | 2419.31 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 15:15:00 | 2432.00 | 2421.27 | 2420.47 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 2282.85 | 2393.58 | 2407.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 12:15:00 | 2244.10 | 2334.37 | 2374.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 2329.85 | 2326.98 | 2363.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-10 15:00:00 | 2329.85 | 2326.98 | 2363.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 2233.85 | 2225.71 | 2251.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 12:15:00 | 2227.25 | 2225.71 | 2251.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:00:00 | 2222.15 | 2223.73 | 2243.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:30:00 | 2212.35 | 2213.08 | 2235.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 2115.89 | 2195.65 | 2225.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 12:15:00 | 2111.04 | 2168.64 | 2207.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 2101.73 | 2127.33 | 2174.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 2114.10 | 2096.50 | 2137.85 | SL hit (close>ema200) qty=0.50 sl=2096.50 alert=retest2 |

### Cycle 144 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 2184.40 | 2134.32 | 2129.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 12:15:00 | 2210.30 | 2149.52 | 2136.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 09:15:00 | 2161.85 | 2176.36 | 2156.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 10:00:00 | 2161.85 | 2176.36 | 2156.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 2188.65 | 2178.81 | 2159.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 11:30:00 | 2193.40 | 2183.67 | 2163.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 12:30:00 | 2192.55 | 2185.89 | 2176.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 13:30:00 | 2196.65 | 2192.39 | 2180.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 13:15:00 | 2159.75 | 2176.69 | 2178.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 2159.75 | 2176.69 | 2178.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 15:15:00 | 2149.05 | 2167.15 | 2173.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 10:15:00 | 2175.00 | 2166.20 | 2171.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 10:15:00 | 2175.00 | 2166.20 | 2171.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 2175.00 | 2166.20 | 2171.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 11:00:00 | 2175.00 | 2166.20 | 2171.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 2197.15 | 2172.39 | 2174.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 11:45:00 | 2200.00 | 2172.39 | 2174.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 12:15:00 | 2200.05 | 2177.92 | 2176.48 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 2150.00 | 2176.66 | 2177.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 2135.30 | 2168.39 | 2173.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 15:15:00 | 2061.10 | 2050.09 | 2088.04 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-03 09:15:00 | 2017.25 | 2050.09 | 2088.04 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-03 11:00:00 | 2026.55 | 2035.61 | 2074.27 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 2100.75 | 2050.02 | 2063.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-04 09:15:00 | 2100.75 | 2050.02 | 2063.49 | SL hit (close>ema400) qty=1.00 sl=2063.49 alert=retest1 |

### Cycle 148 — BUY (started 2025-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 11:15:00 | 2124.35 | 2075.97 | 2073.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 14:15:00 | 2151.55 | 2104.66 | 2088.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 2292.10 | 2300.96 | 2246.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 2292.10 | 2300.96 | 2246.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 2273.65 | 2290.10 | 2262.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 11:00:00 | 2301.90 | 2292.46 | 2265.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 13:45:00 | 2302.10 | 2296.09 | 2274.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 2224.35 | 2283.80 | 2274.22 | SL hit (close<static) qty=1.00 sl=2254.80 alert=retest2 |

### Cycle 149 — SELL (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 11:15:00 | 2210.30 | 2260.14 | 2264.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 14:15:00 | 2190.85 | 2229.70 | 2248.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 11:15:00 | 2203.90 | 2203.71 | 2227.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 12:15:00 | 2209.00 | 2203.71 | 2227.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 2221.60 | 2207.04 | 2223.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 2221.60 | 2207.04 | 2223.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 2205.00 | 2206.63 | 2221.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 2190.10 | 2206.63 | 2221.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 2214.50 | 2208.21 | 2220.78 | EMA400 retest candle locked (from downside) |

### Cycle 150 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 2225.65 | 2216.86 | 2216.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 2235.70 | 2223.20 | 2219.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 09:15:00 | 2224.75 | 2225.39 | 2221.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 10:00:00 | 2224.75 | 2225.39 | 2221.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 10:15:00 | 2209.20 | 2222.16 | 2220.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 10:30:00 | 2215.30 | 2222.16 | 2220.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 11:15:00 | 2225.45 | 2222.81 | 2220.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 12:45:00 | 2240.10 | 2225.46 | 2222.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:30:00 | 2244.35 | 2241.84 | 2236.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 11:15:00 | 2234.80 | 2267.87 | 2270.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 2234.80 | 2267.87 | 2270.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 10:15:00 | 2191.75 | 2223.98 | 2244.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 2213.00 | 2205.73 | 2218.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 13:15:00 | 2213.00 | 2205.73 | 2218.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 2213.00 | 2205.73 | 2218.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:00:00 | 2213.00 | 2205.73 | 2218.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 2216.20 | 2207.83 | 2218.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 2216.20 | 2207.83 | 2218.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 2209.25 | 2208.11 | 2217.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 2205.00 | 2208.11 | 2217.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 2232.05 | 2212.90 | 2218.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:00:00 | 2232.05 | 2212.90 | 2218.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 2235.75 | 2217.47 | 2220.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 11:00:00 | 2235.75 | 2217.47 | 2220.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 2254.00 | 2224.78 | 2223.53 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 2204.90 | 2227.39 | 2227.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 11:15:00 | 2172.20 | 2216.35 | 2222.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 2231.45 | 2191.78 | 2204.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 2231.45 | 2191.78 | 2204.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 2231.45 | 2191.78 | 2204.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 10:00:00 | 2231.45 | 2191.78 | 2204.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 2237.80 | 2200.98 | 2207.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:15:00 | 2247.90 | 2200.98 | 2207.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 11:15:00 | 2263.90 | 2213.57 | 2213.04 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 12:15:00 | 2207.85 | 2212.42 | 2212.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 13:15:00 | 2184.95 | 2206.93 | 2210.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 15:15:00 | 2212.00 | 2204.27 | 2208.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 15:15:00 | 2212.00 | 2204.27 | 2208.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 2212.00 | 2204.27 | 2208.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 2255.30 | 2204.27 | 2208.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 2205.45 | 2204.51 | 2207.84 | EMA400 retest candle locked (from downside) |

### Cycle 156 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 2225.25 | 2212.26 | 2211.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 12:15:00 | 2234.95 | 2216.79 | 2213.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 2175.00 | 2220.88 | 2217.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 2175.00 | 2220.88 | 2217.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 2175.00 | 2220.88 | 2217.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:45:00 | 2174.40 | 2220.88 | 2217.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 2174.60 | 2211.62 | 2213.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 2124.30 | 2185.20 | 2200.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 2016.00 | 2015.98 | 2078.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 2022.30 | 2015.98 | 2078.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 2019.85 | 2024.20 | 2049.48 | EMA400 retest candle locked (from downside) |

### Cycle 158 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 2123.85 | 2050.10 | 2049.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 10:15:00 | 2134.15 | 2066.91 | 2057.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 15:15:00 | 2442.00 | 2446.43 | 2409.66 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:15:00 | 2463.50 | 2446.43 | 2409.66 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 15:15:00 | 2422.00 | 2452.82 | 2433.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-23 15:15:00 | 2422.00 | 2452.82 | 2433.51 | SL hit (close<ema400) qty=1.00 sl=2433.51 alert=retest1 |

### Cycle 159 — SELL (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 09:15:00 | 2471.80 | 2555.05 | 2556.20 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 12:15:00 | 2581.00 | 2542.11 | 2539.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 15:15:00 | 2602.20 | 2564.39 | 2550.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 14:15:00 | 2818.90 | 2822.75 | 2749.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-07 14:45:00 | 2775.60 | 2822.75 | 2749.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 2781.00 | 2820.36 | 2761.10 | EMA400 retest candle locked (from upside) |

### Cycle 161 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 2656.00 | 2739.18 | 2743.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 2545.50 | 2700.44 | 2725.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 14:15:00 | 2362.10 | 2360.06 | 2399.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-15 15:00:00 | 2362.10 | 2360.06 | 2399.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 2400.00 | 2368.73 | 2390.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 12:00:00 | 2400.00 | 2368.73 | 2390.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 12:15:00 | 2411.80 | 2377.35 | 2392.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 12:45:00 | 2421.90 | 2377.35 | 2392.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 2366.00 | 2386.29 | 2393.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:45:00 | 2392.90 | 2386.29 | 2393.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 2390.00 | 2385.38 | 2391.63 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 14:15:00 | 2442.00 | 2402.95 | 2398.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 2472.30 | 2423.23 | 2409.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 13:15:00 | 2440.10 | 2444.45 | 2425.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 14:00:00 | 2440.10 | 2444.45 | 2425.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 2467.10 | 2477.10 | 2462.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 2467.10 | 2477.10 | 2462.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 2465.60 | 2474.80 | 2462.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:30:00 | 2461.80 | 2474.80 | 2462.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 2445.60 | 2468.96 | 2460.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:00:00 | 2445.60 | 2468.96 | 2460.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 2444.50 | 2464.07 | 2459.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:00:00 | 2444.50 | 2464.07 | 2459.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 2427.70 | 2451.82 | 2454.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 13:15:00 | 2408.90 | 2436.63 | 2446.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 14:15:00 | 2435.60 | 2418.22 | 2427.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 14:15:00 | 2435.60 | 2418.22 | 2427.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 2435.60 | 2418.22 | 2427.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 15:00:00 | 2435.60 | 2418.22 | 2427.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 2432.00 | 2420.97 | 2428.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 2391.60 | 2420.97 | 2428.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 09:15:00 | 2272.02 | 2300.87 | 2330.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-03 13:15:00 | 2239.10 | 2235.03 | 2256.28 | SL hit (close>ema200) qty=0.50 sl=2235.03 alert=retest2 |

### Cycle 164 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 2274.60 | 2243.13 | 2241.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 2293.90 | 2269.83 | 2258.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 12:15:00 | 2264.10 | 2271.04 | 2261.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 12:15:00 | 2264.10 | 2271.04 | 2261.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 2264.10 | 2271.04 | 2261.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:00:00 | 2264.10 | 2271.04 | 2261.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 2270.40 | 2270.91 | 2262.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:30:00 | 2270.30 | 2270.91 | 2262.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 2301.10 | 2276.86 | 2267.34 | EMA400 retest candle locked (from upside) |

### Cycle 165 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 2250.30 | 2263.57 | 2264.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 2240.40 | 2256.45 | 2261.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 09:15:00 | 2248.00 | 2244.44 | 2252.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 2248.00 | 2244.44 | 2252.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2248.00 | 2244.44 | 2252.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:00:00 | 2248.00 | 2244.44 | 2252.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 2255.90 | 2246.73 | 2252.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 11:00:00 | 2255.90 | 2246.73 | 2252.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 2214.50 | 2240.28 | 2249.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 14:45:00 | 2209.00 | 2230.08 | 2241.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 2208.60 | 2229.46 | 2240.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:45:00 | 2209.10 | 2224.49 | 2237.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:15:00 | 2098.55 | 2127.18 | 2154.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:15:00 | 2098.17 | 2127.18 | 2154.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:15:00 | 2098.64 | 2127.18 | 2154.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 2088.90 | 2082.25 | 2115.81 | SL hit (close>ema200) qty=0.50 sl=2082.25 alert=retest2 |

### Cycle 166 — BUY (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 15:15:00 | 2135.00 | 2100.59 | 2097.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 10:15:00 | 2189.60 | 2136.58 | 2121.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 14:15:00 | 2187.40 | 2194.59 | 2172.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 14:15:00 | 2187.40 | 2194.59 | 2172.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 2187.40 | 2194.59 | 2172.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:45:00 | 2183.30 | 2194.59 | 2172.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 2162.70 | 2187.48 | 2172.84 | EMA400 retest candle locked (from upside) |

### Cycle 167 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 2160.10 | 2167.29 | 2167.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 10:15:00 | 2147.20 | 2163.00 | 2165.49 | Break + close below crossover candle low |

### Cycle 168 — BUY (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 13:15:00 | 2197.40 | 2167.38 | 2166.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 14:15:00 | 2233.10 | 2180.53 | 2172.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 09:15:00 | 2237.40 | 2256.41 | 2239.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 2237.40 | 2256.41 | 2239.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 2237.40 | 2256.41 | 2239.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 2237.40 | 2256.41 | 2239.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 2233.00 | 2251.73 | 2238.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:00:00 | 2233.00 | 2251.73 | 2238.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 2249.00 | 2250.92 | 2240.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:30:00 | 2244.50 | 2250.92 | 2240.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 2252.60 | 2251.26 | 2241.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 11:15:00 | 2260.70 | 2245.93 | 2241.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 12:30:00 | 2262.20 | 2246.16 | 2242.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 13:15:00 | 2236.30 | 2244.19 | 2241.88 | SL hit (close<static) qty=1.00 sl=2240.90 alert=retest2 |

### Cycle 169 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 2229.90 | 2239.82 | 2240.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 12:15:00 | 2220.00 | 2235.85 | 2238.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 2213.00 | 2210.61 | 2220.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 2210.00 | 2209.82 | 2218.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 2210.00 | 2209.82 | 2218.48 | EMA400 retest candle locked (from downside) |

### Cycle 170 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 2230.30 | 2221.57 | 2220.91 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 13:15:00 | 2220.00 | 2220.61 | 2220.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 2217.00 | 2219.78 | 2220.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 11:15:00 | 2219.80 | 2219.79 | 2220.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 11:15:00 | 2219.80 | 2219.79 | 2220.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 2219.80 | 2219.79 | 2220.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:45:00 | 2220.70 | 2219.79 | 2220.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 2217.10 | 2219.25 | 2219.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:30:00 | 2218.80 | 2219.25 | 2219.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 2217.30 | 2218.86 | 2219.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:30:00 | 2217.70 | 2218.86 | 2219.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 2221.00 | 2219.29 | 2219.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:00:00 | 2221.00 | 2219.29 | 2219.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 2215.30 | 2218.49 | 2219.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 2189.70 | 2218.49 | 2219.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 12:45:00 | 2210.00 | 2211.06 | 2214.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 13:15:00 | 2222.00 | 2213.25 | 2215.52 | SL hit (close>static) qty=1.00 sl=2221.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 11:15:00 | 2021.70 | 1959.94 | 1959.13 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 1958.20 | 1971.78 | 1972.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 11:15:00 | 1935.90 | 1964.60 | 1969.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 1938.20 | 1936.53 | 1947.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 09:15:00 | 1945.20 | 1936.53 | 1947.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1936.00 | 1936.43 | 1946.49 | EMA400 retest candle locked (from downside) |

### Cycle 174 — BUY (started 2025-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 13:15:00 | 1987.50 | 1947.85 | 1942.46 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 1925.30 | 1942.87 | 1944.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 15:15:00 | 1918.00 | 1937.90 | 1942.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 09:15:00 | 1903.70 | 1873.31 | 1896.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 1903.70 | 1873.31 | 1896.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1903.70 | 1873.31 | 1896.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:45:00 | 1897.60 | 1873.31 | 1896.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 1903.50 | 1879.35 | 1897.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:45:00 | 1903.00 | 1879.35 | 1897.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1905.70 | 1897.48 | 1900.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 09:15:00 | 1883.90 | 1899.11 | 1900.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 11:15:00 | 1914.90 | 1900.17 | 1899.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 11:15:00 | 1914.90 | 1900.17 | 1899.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 10:15:00 | 1944.00 | 1911.05 | 1905.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 14:15:00 | 1965.00 | 1966.83 | 1947.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 15:00:00 | 1965.00 | 1966.83 | 1947.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 2100.00 | 2096.40 | 2073.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 12:00:00 | 2101.00 | 2096.59 | 2079.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 13:15:00 | 2102.30 | 2096.15 | 2080.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 2071.50 | 2081.22 | 2081.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 10:15:00 | 2071.50 | 2081.22 | 2081.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-01 12:15:00 | 2068.00 | 2077.04 | 2079.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 13:15:00 | 2077.40 | 2077.11 | 2079.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 13:15:00 | 2077.40 | 2077.11 | 2079.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 2077.40 | 2077.11 | 2079.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:45:00 | 2075.90 | 2077.11 | 2079.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 2071.80 | 2076.05 | 2078.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 2079.60 | 2076.05 | 2078.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 2065.10 | 2072.57 | 2076.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 12:00:00 | 2059.10 | 2068.35 | 2073.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 14:15:00 | 2053.80 | 2065.70 | 2071.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 15:15:00 | 2082.00 | 2071.02 | 2073.12 | SL hit (close>static) qty=1.00 sl=2079.90 alert=retest2 |

### Cycle 178 — BUY (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 11:15:00 | 2012.00 | 1997.09 | 1995.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 12:15:00 | 2014.20 | 2000.51 | 1997.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 09:15:00 | 1997.50 | 2008.54 | 2002.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 1997.50 | 2008.54 | 2002.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1997.50 | 2008.54 | 2002.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:45:00 | 1999.60 | 2008.54 | 2002.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 1995.80 | 2005.99 | 2002.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:45:00 | 1996.90 | 2005.99 | 2002.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — SELL (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 12:15:00 | 1985.50 | 1999.73 | 1999.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 13:15:00 | 1980.30 | 1995.85 | 1998.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 11:15:00 | 1998.10 | 1992.33 | 1995.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 11:15:00 | 1998.10 | 1992.33 | 1995.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 1998.10 | 1992.33 | 1995.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:30:00 | 1992.40 | 1992.33 | 1995.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 1988.70 | 1991.60 | 1994.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 12:30:00 | 1988.00 | 1992.11 | 1993.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 14:00:00 | 1987.40 | 1991.17 | 1992.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 15:15:00 | 2000.00 | 1993.92 | 1993.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 15:15:00 | 2000.00 | 1993.92 | 1993.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 2059.20 | 2006.97 | 1999.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 2052.40 | 2055.68 | 2034.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 12:15:00 | 2040.50 | 2049.92 | 2037.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 2040.50 | 2049.92 | 2037.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:45:00 | 2038.40 | 2049.92 | 2037.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 2032.30 | 2046.40 | 2036.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:00:00 | 2032.30 | 2046.40 | 2036.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 2000.00 | 2037.12 | 2033.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 2000.00 | 2037.12 | 2033.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 15:15:00 | 1989.00 | 2027.49 | 2029.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 09:15:00 | 1972.00 | 2016.39 | 2024.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 13:15:00 | 1959.60 | 1959.39 | 1979.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:00:00 | 1959.60 | 1959.39 | 1979.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1974.90 | 1960.20 | 1974.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 1974.90 | 1960.20 | 1974.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1961.80 | 1960.52 | 1973.53 | EMA400 retest candle locked (from downside) |

### Cycle 182 — BUY (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 09:15:00 | 2035.00 | 1988.52 | 1982.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 12:15:00 | 2050.20 | 2016.46 | 1998.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 1956.30 | 2012.08 | 2003.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 1956.30 | 2012.08 | 2003.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1956.30 | 2012.08 | 2003.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:00:00 | 1956.30 | 2012.08 | 2003.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 1940.00 | 1997.66 | 1997.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:30:00 | 1938.50 | 1997.66 | 1997.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 1930.30 | 1984.19 | 1991.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 09:15:00 | 1897.10 | 1939.59 | 1948.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 1886.30 | 1874.86 | 1891.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:15:00 | 1886.90 | 1874.86 | 1891.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1910.20 | 1881.93 | 1892.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 1910.20 | 1881.93 | 1892.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1912.50 | 1888.04 | 1894.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:30:00 | 1911.00 | 1888.04 | 1894.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 14:15:00 | 1911.10 | 1899.90 | 1899.14 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 11:15:00 | 1897.40 | 1900.53 | 1900.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 14:15:00 | 1894.00 | 1899.01 | 1899.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 15:15:00 | 1904.00 | 1900.01 | 1900.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 15:15:00 | 1904.00 | 1900.01 | 1900.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 1904.00 | 1900.01 | 1900.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 1883.50 | 1900.01 | 1900.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 1925.00 | 1867.14 | 1865.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 1925.00 | 1867.14 | 1865.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 1928.70 | 1879.45 | 1871.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 1892.10 | 1900.77 | 1887.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 10:00:00 | 1892.10 | 1900.77 | 1887.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 1901.70 | 1900.96 | 1888.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 11:15:00 | 1906.90 | 1900.96 | 1888.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:00:00 | 1903.50 | 1901.18 | 1891.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:30:00 | 1905.50 | 1901.50 | 1892.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1874.70 | 1896.06 | 1891.86 | SL hit (close<static) qty=1.00 sl=1888.00 alert=retest2 |

### Cycle 187 — SELL (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 14:15:00 | 1881.00 | 1888.96 | 1889.59 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 1958.00 | 1901.32 | 1895.01 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 1893.20 | 1906.42 | 1907.47 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 1913.50 | 1907.67 | 1907.48 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 1905.20 | 1907.18 | 1907.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 1901.90 | 1906.12 | 1906.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 13:15:00 | 1907.30 | 1906.36 | 1906.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 13:15:00 | 1907.30 | 1906.36 | 1906.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 1907.30 | 1906.36 | 1906.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:45:00 | 1920.50 | 1906.36 | 1906.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 1904.90 | 1906.07 | 1906.66 | EMA400 retest candle locked (from downside) |

### Cycle 192 — BUY (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 15:15:00 | 1913.00 | 1907.45 | 1907.23 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 1894.40 | 1904.84 | 1906.07 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 1914.00 | 1907.98 | 1907.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 12:15:00 | 1944.20 | 1915.22 | 1910.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 12:15:00 | 1967.40 | 1967.55 | 1952.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 13:00:00 | 1967.40 | 1967.55 | 1952.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 1955.20 | 1975.95 | 1965.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:30:00 | 1959.40 | 1975.95 | 1965.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 1949.60 | 1970.68 | 1964.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:45:00 | 1949.00 | 1970.68 | 1964.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — SELL (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 14:15:00 | 1915.00 | 1955.90 | 1958.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 1905.90 | 1931.80 | 1940.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 15:15:00 | 1898.00 | 1893.58 | 1907.03 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 09:15:00 | 1860.10 | 1893.58 | 1907.03 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1990.00 | 1889.37 | 1891.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1990.00 | 1889.37 | 1891.92 | SL hit (close>ema400) qty=1.00 sl=1891.92 alert=retest1 |

### Cycle 196 — BUY (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 10:15:00 | 2004.40 | 1912.38 | 1902.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 11:15:00 | 2034.00 | 1936.70 | 1914.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 12:15:00 | 2007.80 | 2011.21 | 1976.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 12:45:00 | 2008.80 | 2011.21 | 1976.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1990.10 | 2012.89 | 1988.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:45:00 | 1987.20 | 2012.89 | 1988.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1978.80 | 2006.07 | 1988.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 1978.80 | 2006.07 | 1988.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 1978.20 | 2000.50 | 1987.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:30:00 | 1975.40 | 2000.50 | 1987.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 1980.90 | 1996.58 | 1986.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:30:00 | 1977.50 | 1996.58 | 1986.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 1947.80 | 1974.94 | 1978.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 1943.50 | 1959.89 | 1969.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 14:15:00 | 1944.80 | 1943.18 | 1953.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 15:00:00 | 1944.80 | 1943.18 | 1953.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1906.20 | 1936.07 | 1948.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 10:15:00 | 1905.20 | 1936.07 | 1948.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 11:00:00 | 1893.10 | 1927.48 | 1943.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 13:15:00 | 1903.20 | 1918.84 | 1936.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 1994.00 | 1901.28 | 1896.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — BUY (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 09:15:00 | 1994.00 | 1901.28 | 1896.68 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2025-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 15:15:00 | 1907.00 | 1918.83 | 1920.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-26 09:15:00 | 1898.50 | 1914.77 | 1918.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 15:15:00 | 1907.80 | 1902.14 | 1908.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-27 09:15:00 | 1891.00 | 1902.14 | 1908.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1891.00 | 1899.91 | 1907.14 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 11:15:00 | 1951.10 | 1915.59 | 1913.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 12:15:00 | 1974.60 | 1927.39 | 1918.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 1936.50 | 1941.25 | 1929.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 1936.50 | 1941.25 | 1929.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1936.50 | 1941.25 | 1929.48 | EMA400 retest candle locked (from upside) |

### Cycle 201 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 1918.10 | 1931.98 | 1932.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 1898.30 | 1916.61 | 1923.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 13:15:00 | 1916.60 | 1915.51 | 1921.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 14:00:00 | 1916.60 | 1915.51 | 1921.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 1917.00 | 1915.80 | 1921.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 1917.00 | 1915.80 | 1921.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1908.90 | 1914.13 | 1919.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:15:00 | 1915.40 | 1914.13 | 1919.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1916.00 | 1914.51 | 1919.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 12:30:00 | 1903.20 | 1913.91 | 1918.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:30:00 | 1906.60 | 1912.81 | 1917.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 11:15:00 | 1901.10 | 1906.47 | 1912.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 1918.80 | 1891.10 | 1888.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 1918.80 | 1891.10 | 1888.83 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 1887.10 | 1888.15 | 1888.15 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 14:15:00 | 1891.50 | 1888.82 | 1888.46 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 1884.00 | 1887.86 | 1888.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 1877.60 | 1885.81 | 1887.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 13:15:00 | 1886.40 | 1881.67 | 1884.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 13:15:00 | 1886.40 | 1881.67 | 1884.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 1886.40 | 1881.67 | 1884.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:00:00 | 1886.40 | 1881.67 | 1884.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 1882.80 | 1881.89 | 1884.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:30:00 | 1878.80 | 1881.75 | 1883.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 1893.30 | 1884.42 | 1884.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 1893.30 | 1884.42 | 1884.34 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 1871.20 | 1882.49 | 1883.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 1861.50 | 1873.37 | 1877.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 15:15:00 | 1852.00 | 1851.79 | 1859.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 09:15:00 | 1836.50 | 1851.79 | 1859.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 1827.00 | 1846.83 | 1856.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 13:00:00 | 1818.20 | 1835.19 | 1847.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 09:15:00 | 1727.29 | 1741.80 | 1758.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 14:15:00 | 1758.70 | 1710.44 | 1721.41 | SL hit (close>ema200) qty=0.50 sl=1710.44 alert=retest2 |

### Cycle 208 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 1762.10 | 1733.60 | 1730.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 1770.10 | 1753.59 | 1744.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 10:15:00 | 1748.10 | 1756.11 | 1748.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 10:15:00 | 1748.10 | 1756.11 | 1748.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1748.10 | 1756.11 | 1748.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 1748.10 | 1756.11 | 1748.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 1734.40 | 1751.77 | 1747.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:45:00 | 1733.20 | 1751.77 | 1747.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 1724.50 | 1746.32 | 1745.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:30:00 | 1726.90 | 1746.32 | 1745.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 1724.30 | 1741.91 | 1743.35 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 1810.00 | 1750.60 | 1746.47 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2026-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 14:15:00 | 1775.80 | 1780.90 | 1781.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 10:15:00 | 1756.60 | 1774.36 | 1777.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 12:15:00 | 1648.00 | 1643.41 | 1666.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 13:00:00 | 1648.00 | 1643.41 | 1666.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 1655.00 | 1647.30 | 1662.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 1649.30 | 1647.30 | 1662.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1646.10 | 1647.06 | 1660.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:30:00 | 1638.40 | 1644.50 | 1658.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 13:00:00 | 1635.00 | 1640.06 | 1653.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 11:15:00 | 1556.48 | 1609.45 | 1631.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 1591.00 | 1587.03 | 1610.44 | SL hit (close>ema200) qty=0.50 sl=1587.03 alert=retest2 |

### Cycle 212 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 1562.50 | 1501.20 | 1499.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 14:15:00 | 1587.50 | 1546.59 | 1523.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 1551.30 | 1556.65 | 1535.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 10:45:00 | 1552.50 | 1556.65 | 1535.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 1527.10 | 1547.11 | 1535.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:45:00 | 1530.10 | 1547.11 | 1535.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 1522.90 | 1542.27 | 1534.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 15:00:00 | 1522.90 | 1542.27 | 1534.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 1480.00 | 1525.69 | 1528.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 1433.50 | 1495.68 | 1510.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 1440.10 | 1421.97 | 1454.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:45:00 | 1441.10 | 1421.97 | 1454.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 1454.30 | 1434.53 | 1452.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:45:00 | 1452.80 | 1434.53 | 1452.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 1464.40 | 1440.50 | 1453.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 14:45:00 | 1463.80 | 1440.50 | 1453.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 1461.00 | 1444.60 | 1454.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:15:00 | 1480.80 | 1444.60 | 1454.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 214 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 1515.00 | 1468.03 | 1463.71 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 1425.00 | 1465.34 | 1469.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 11:15:00 | 1398.70 | 1452.01 | 1462.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 10:15:00 | 1302.50 | 1299.82 | 1329.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 10:30:00 | 1300.70 | 1299.82 | 1329.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 1319.20 | 1304.01 | 1313.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 1319.20 | 1304.01 | 1313.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 1322.70 | 1307.75 | 1314.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 1322.70 | 1307.75 | 1314.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 1321.00 | 1310.40 | 1314.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 1337.50 | 1310.40 | 1314.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 1317.00 | 1309.93 | 1312.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:00:00 | 1317.00 | 1309.93 | 1312.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 1307.60 | 1309.47 | 1312.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 1299.60 | 1309.77 | 1312.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 1284.10 | 1274.57 | 1273.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 216 — BUY (started 2026-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 14:15:00 | 1284.10 | 1274.57 | 1273.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 10:15:00 | 1295.00 | 1281.82 | 1277.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 15:15:00 | 1287.00 | 1289.53 | 1283.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 15:15:00 | 1287.00 | 1289.53 | 1283.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 1287.00 | 1289.53 | 1283.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:00:00 | 1280.70 | 1287.77 | 1283.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 1282.90 | 1286.79 | 1283.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:45:00 | 1285.00 | 1286.79 | 1283.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 1272.30 | 1283.89 | 1282.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:30:00 | 1271.70 | 1283.89 | 1282.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — SELL (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 12:15:00 | 1264.40 | 1280.00 | 1280.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 14:15:00 | 1261.00 | 1273.89 | 1277.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 13:15:00 | 1252.90 | 1251.82 | 1262.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-27 13:30:00 | 1254.10 | 1251.82 | 1262.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 1251.10 | 1251.67 | 1261.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:15:00 | 1266.50 | 1251.67 | 1261.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 1266.50 | 1254.64 | 1262.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:30:00 | 1286.60 | 1263.01 | 1265.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 218 — BUY (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 10:15:00 | 1291.00 | 1268.61 | 1267.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-02 14:15:00 | 1305.00 | 1283.36 | 1275.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-05 10:15:00 | 1362.00 | 1368.06 | 1337.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-05 11:00:00 | 1362.00 | 1368.06 | 1337.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 1341.00 | 1359.51 | 1340.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:45:00 | 1345.10 | 1359.51 | 1340.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1356.30 | 1358.87 | 1342.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 15:15:00 | 1370.00 | 1358.87 | 1342.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 13:15:00 | 1367.00 | 1363.93 | 1351.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 1335.30 | 1355.51 | 1351.51 | SL hit (close<static) qty=1.00 sl=1341.10 alert=retest2 |

### Cycle 219 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 1325.00 | 1346.41 | 1347.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 14:15:00 | 1319.70 | 1336.06 | 1342.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-11 09:15:00 | 1321.90 | 1304.55 | 1317.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 09:15:00 | 1321.90 | 1304.55 | 1317.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 1321.90 | 1304.55 | 1317.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:00:00 | 1321.90 | 1304.55 | 1317.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 1333.00 | 1310.24 | 1319.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:30:00 | 1325.40 | 1310.24 | 1319.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 1334.60 | 1315.11 | 1320.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:30:00 | 1348.00 | 1315.11 | 1320.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1333.80 | 1321.70 | 1322.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1333.80 | 1321.70 | 1322.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1321.40 | 1321.64 | 1322.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 09:15:00 | 1295.80 | 1321.64 | 1322.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 1335.30 | 1318.92 | 1320.26 | SL hit (close>static) qty=1.00 sl=1333.60 alert=retest2 |

### Cycle 220 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 1333.80 | 1321.90 | 1321.49 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 14:15:00 | 1309.20 | 1319.26 | 1320.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 15:15:00 | 1306.20 | 1316.65 | 1319.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 15:15:00 | 1256.80 | 1256.59 | 1273.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 09:15:00 | 1252.60 | 1256.59 | 1273.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1261.90 | 1248.68 | 1258.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 1275.10 | 1248.68 | 1258.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 1281.00 | 1255.14 | 1260.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 1281.00 | 1255.14 | 1260.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 1281.30 | 1260.37 | 1262.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:30:00 | 1281.30 | 1260.37 | 1262.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 1286.00 | 1265.50 | 1264.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 1288.10 | 1270.02 | 1266.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 15:15:00 | 1272.20 | 1273.06 | 1269.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 09:15:00 | 1257.30 | 1273.06 | 1269.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1257.90 | 1270.03 | 1268.04 | EMA400 retest candle locked (from upside) |

### Cycle 223 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1247.30 | 1265.48 | 1266.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1241.10 | 1257.28 | 1261.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 1267.00 | 1252.85 | 1257.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 10:15:00 | 1267.00 | 1252.85 | 1257.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1267.00 | 1252.85 | 1257.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 1267.00 | 1252.85 | 1257.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1276.90 | 1257.66 | 1259.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:00:00 | 1276.90 | 1257.66 | 1259.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 1241.00 | 1255.60 | 1258.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 1213.00 | 1253.12 | 1256.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:15:00 | 1232.80 | 1235.23 | 1236.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1296.00 | 1247.00 | 1241.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 224 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1296.00 | 1247.00 | 1241.36 | EMA200 above EMA400 |

### Cycle 225 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 1233.00 | 1256.01 | 1257.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1202.10 | 1241.10 | 1250.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1251.00 | 1218.48 | 1230.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1251.00 | 1218.48 | 1230.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1251.00 | 1218.48 | 1230.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1251.00 | 1218.48 | 1230.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1252.80 | 1225.34 | 1232.65 | EMA400 retest candle locked (from downside) |

### Cycle 226 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 1264.90 | 1240.10 | 1238.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 13:15:00 | 1270.20 | 1246.12 | 1241.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 1256.30 | 1257.80 | 1248.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 1256.30 | 1257.80 | 1248.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 1256.30 | 1257.80 | 1248.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 1280.10 | 1260.52 | 1250.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-07 09:15:00 | 1408.11 | 1363.45 | 1328.51 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 227 — SELL (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 09:15:00 | 1448.00 | 1471.67 | 1471.75 | EMA200 below EMA400 |

### Cycle 228 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 1487.40 | 1467.88 | 1467.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 1513.70 | 1481.02 | 1476.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 12:15:00 | 1486.90 | 1487.52 | 1480.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-23 13:00:00 | 1486.90 | 1487.52 | 1480.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1483.80 | 1487.32 | 1482.89 | EMA400 retest candle locked (from upside) |

### Cycle 229 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 1440.00 | 1472.57 | 1476.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 1425.80 | 1463.22 | 1472.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 10:15:00 | 1440.00 | 1438.60 | 1454.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 11:00:00 | 1440.00 | 1438.60 | 1454.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1489.00 | 1449.61 | 1456.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:45:00 | 1501.80 | 1449.61 | 1456.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 1491.20 | 1457.93 | 1459.69 | EMA400 retest candle locked (from downside) |

### Cycle 230 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 1492.70 | 1464.88 | 1462.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 1508.00 | 1473.51 | 1466.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 1510.40 | 1514.84 | 1504.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 1510.40 | 1514.84 | 1504.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1510.40 | 1514.84 | 1504.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 1545.70 | 1513.38 | 1507.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 11:30:00 | 1538.10 | 1525.51 | 1515.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 13:00:00 | 1530.10 | 1526.43 | 1516.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 15:15:00 | 1532.00 | 1524.87 | 1517.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 1532.00 | 1526.29 | 1518.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 09:30:00 | 1551.00 | 1534.25 | 1523.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 12:00:00 | 1558.10 | 1540.97 | 1528.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 13:30:00 | 1547.30 | 1544.28 | 1532.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-07 10:15:00 | 1683.11 | 1644.37 | 1605.73 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-15 09:15:00 | 1517.30 | 2024-04-16 13:15:00 | 1570.15 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2024-04-15 11:15:00 | 1539.60 | 2024-04-16 13:15:00 | 1570.15 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-04-15 14:15:00 | 1532.30 | 2024-04-16 13:15:00 | 1570.15 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2024-04-15 15:15:00 | 1538.25 | 2024-04-16 13:15:00 | 1570.15 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-04-19 10:30:00 | 1608.85 | 2024-04-23 09:15:00 | 1543.00 | STOP_HIT | 1.00 | -4.09% |
| BUY | retest2 | 2024-04-22 10:15:00 | 1601.20 | 2024-04-23 09:15:00 | 1543.00 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest2 | 2024-05-02 12:45:00 | 1637.95 | 2024-05-07 09:15:00 | 1607.20 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-05-10 09:15:00 | 1587.25 | 2024-05-13 11:15:00 | 1620.75 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-05-10 13:15:00 | 1590.05 | 2024-05-13 11:15:00 | 1620.75 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-05-13 11:15:00 | 1589.90 | 2024-05-13 11:15:00 | 1620.75 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2024-05-16 11:45:00 | 1653.90 | 2024-05-17 13:15:00 | 1651.60 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2024-05-16 12:45:00 | 1657.00 | 2024-05-17 14:15:00 | 1642.95 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-05-16 14:45:00 | 1656.00 | 2024-05-17 14:15:00 | 1642.95 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-05-17 09:45:00 | 1658.95 | 2024-05-17 14:15:00 | 1642.95 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-05-17 12:45:00 | 1676.60 | 2024-05-17 14:15:00 | 1642.95 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-05-17 13:45:00 | 1679.35 | 2024-05-17 14:15:00 | 1642.95 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2024-05-28 09:15:00 | 1827.95 | 2024-05-28 09:15:00 | 1791.05 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-06-03 10:45:00 | 1795.20 | 2024-06-04 09:15:00 | 1709.76 | PARTIAL | 0.50 | 4.76% |
| SELL | retest2 | 2024-06-03 13:30:00 | 1799.75 | 2024-06-04 09:15:00 | 1709.14 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2024-06-03 14:15:00 | 1799.10 | 2024-06-04 11:15:00 | 1705.44 | PARTIAL | 0.50 | 5.21% |
| SELL | retest2 | 2024-06-03 10:45:00 | 1795.20 | 2024-06-04 12:15:00 | 1615.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 13:30:00 | 1799.75 | 2024-06-04 12:15:00 | 1619.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 14:15:00 | 1799.10 | 2024-06-04 12:15:00 | 1619.19 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-06-12 09:15:00 | 1840.95 | 2024-06-18 09:15:00 | 2025.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-12 11:15:00 | 1844.10 | 2024-06-18 09:15:00 | 2028.51 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-28 12:30:00 | 1947.95 | 2024-07-02 10:15:00 | 1986.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-07-25 11:15:00 | 2001.00 | 2024-07-31 11:15:00 | 1900.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-26 09:15:00 | 1994.00 | 2024-07-31 11:15:00 | 1894.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-26 13:00:00 | 1997.00 | 2024-07-31 11:15:00 | 1897.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-26 14:45:00 | 1999.05 | 2024-07-31 11:15:00 | 1899.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-29 10:15:00 | 1969.45 | 2024-07-31 15:15:00 | 1870.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-30 11:45:00 | 1966.65 | 2024-07-31 15:15:00 | 1868.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-30 14:00:00 | 1965.10 | 2024-07-31 15:15:00 | 1866.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-25 11:15:00 | 2001.00 | 2024-08-02 09:15:00 | 1860.90 | STOP_HIT | 0.50 | 7.00% |
| SELL | retest2 | 2024-07-26 09:15:00 | 1994.00 | 2024-08-02 09:15:00 | 1860.90 | STOP_HIT | 0.50 | 6.68% |
| SELL | retest2 | 2024-07-26 13:00:00 | 1997.00 | 2024-08-02 09:15:00 | 1860.90 | STOP_HIT | 0.50 | 6.82% |
| SELL | retest2 | 2024-07-26 14:45:00 | 1999.05 | 2024-08-02 09:15:00 | 1860.90 | STOP_HIT | 0.50 | 6.91% |
| SELL | retest2 | 2024-07-29 10:15:00 | 1969.45 | 2024-08-02 09:15:00 | 1860.90 | STOP_HIT | 0.50 | 5.51% |
| SELL | retest2 | 2024-07-30 11:45:00 | 1966.65 | 2024-08-02 09:15:00 | 1860.90 | STOP_HIT | 0.50 | 5.38% |
| SELL | retest2 | 2024-07-30 14:00:00 | 1965.10 | 2024-08-02 09:15:00 | 1860.90 | STOP_HIT | 0.50 | 5.30% |
| BUY | retest2 | 2024-08-09 09:15:00 | 1887.65 | 2024-08-14 12:15:00 | 1867.20 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-08-09 10:00:00 | 1875.85 | 2024-08-14 12:15:00 | 1867.20 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-08-09 13:00:00 | 1875.90 | 2024-08-14 12:15:00 | 1867.20 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-08-12 09:15:00 | 1893.25 | 2024-08-14 12:15:00 | 1867.20 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-08-23 09:15:00 | 2180.10 | 2024-08-26 09:15:00 | 2398.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-11 15:00:00 | 2547.70 | 2024-09-12 10:15:00 | 2517.75 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-09-12 09:45:00 | 2540.30 | 2024-09-12 10:15:00 | 2517.75 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-09-20 10:15:00 | 2440.75 | 2024-09-23 09:15:00 | 2318.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-20 10:15:00 | 2440.75 | 2024-09-24 11:15:00 | 2363.75 | STOP_HIT | 0.50 | 3.15% |
| BUY | retest2 | 2024-10-10 09:15:00 | 2364.40 | 2024-10-11 12:15:00 | 2320.10 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-10-10 11:45:00 | 2355.40 | 2024-10-11 12:15:00 | 2320.10 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-11-05 10:15:00 | 2906.15 | 2024-11-05 13:15:00 | 2839.25 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2024-11-08 14:00:00 | 2791.30 | 2024-11-13 09:15:00 | 2512.17 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-08 15:00:00 | 2790.45 | 2024-11-13 09:15:00 | 2511.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-17 12:15:00 | 2512.75 | 2025-01-20 14:15:00 | 2521.45 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-01-17 13:00:00 | 2513.00 | 2025-01-20 14:15:00 | 2521.45 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-01-17 13:30:00 | 2511.50 | 2025-01-20 14:15:00 | 2521.45 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-01-17 14:00:00 | 2512.00 | 2025-01-20 14:15:00 | 2521.45 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-01-24 14:00:00 | 2404.35 | 2025-01-24 15:15:00 | 2462.00 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-01-27 10:00:00 | 2395.90 | 2025-01-28 09:15:00 | 2276.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-28 09:15:00 | 2302.45 | 2025-01-28 09:15:00 | 2187.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-27 10:00:00 | 2395.90 | 2025-01-28 10:15:00 | 2156.31 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-28 09:15:00 | 2302.45 | 2025-01-28 10:15:00 | 2072.20 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-06 15:15:00 | 2435.00 | 2025-02-07 14:15:00 | 2410.80 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-02-07 11:00:00 | 2435.00 | 2025-02-07 14:15:00 | 2410.80 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-02-07 11:45:00 | 2434.70 | 2025-02-07 14:15:00 | 2410.80 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-02-13 12:15:00 | 2227.25 | 2025-02-14 10:15:00 | 2115.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 15:00:00 | 2222.15 | 2025-02-14 12:15:00 | 2111.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 09:30:00 | 2212.35 | 2025-02-17 09:15:00 | 2101.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 12:15:00 | 2227.25 | 2025-02-17 14:15:00 | 2114.10 | STOP_HIT | 0.50 | 5.08% |
| SELL | retest2 | 2025-02-13 15:00:00 | 2222.15 | 2025-02-17 14:15:00 | 2114.10 | STOP_HIT | 0.50 | 4.86% |
| SELL | retest2 | 2025-02-14 09:30:00 | 2212.35 | 2025-02-17 14:15:00 | 2114.10 | STOP_HIT | 0.50 | 4.44% |
| BUY | retest2 | 2025-02-20 11:30:00 | 2193.40 | 2025-02-24 13:15:00 | 2159.75 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-02-21 12:30:00 | 2192.55 | 2025-02-24 13:15:00 | 2159.75 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-02-21 13:30:00 | 2196.65 | 2025-02-24 13:15:00 | 2159.75 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest1 | 2025-03-03 09:15:00 | 2017.25 | 2025-03-04 09:15:00 | 2100.75 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest1 | 2025-03-03 11:00:00 | 2026.55 | 2025-03-04 09:15:00 | 2100.75 | STOP_HIT | 1.00 | -3.66% |
| BUY | retest2 | 2025-03-10 11:00:00 | 2301.90 | 2025-03-11 09:15:00 | 2224.35 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2025-03-10 13:45:00 | 2302.10 | 2025-03-11 09:15:00 | 2224.35 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2025-03-19 12:45:00 | 2240.10 | 2025-03-25 11:15:00 | 2234.80 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-03-21 09:30:00 | 2244.35 | 2025-03-25 11:15:00 | 2234.80 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-04-23 09:15:00 | 2463.50 | 2025-04-23 15:15:00 | 2422.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-04-24 10:30:00 | 2562.20 | 2025-05-02 09:15:00 | 2471.80 | STOP_HIT | 1.00 | -3.53% |
| BUY | retest2 | 2025-04-24 15:15:00 | 2519.90 | 2025-05-02 09:15:00 | 2471.80 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-04-25 15:00:00 | 2524.70 | 2025-05-02 09:15:00 | 2471.80 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-05-27 09:15:00 | 2391.60 | 2025-05-30 09:15:00 | 2272.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-27 09:15:00 | 2391.60 | 2025-06-03 13:15:00 | 2239.10 | STOP_HIT | 0.50 | 6.38% |
| SELL | retest2 | 2025-06-13 14:45:00 | 2209.00 | 2025-06-19 10:15:00 | 2098.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-16 09:15:00 | 2208.60 | 2025-06-19 10:15:00 | 2098.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-16 09:45:00 | 2209.10 | 2025-06-19 10:15:00 | 2098.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-13 14:45:00 | 2209.00 | 2025-06-20 09:15:00 | 2088.90 | STOP_HIT | 0.50 | 5.44% |
| SELL | retest2 | 2025-06-16 09:15:00 | 2208.60 | 2025-06-20 09:15:00 | 2088.90 | STOP_HIT | 0.50 | 5.42% |
| SELL | retest2 | 2025-06-16 09:45:00 | 2209.10 | 2025-06-20 09:15:00 | 2088.90 | STOP_HIT | 0.50 | 5.44% |
| BUY | retest2 | 2025-07-04 11:15:00 | 2260.70 | 2025-07-04 13:15:00 | 2236.30 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-07-04 12:30:00 | 2262.20 | 2025-07-04 13:15:00 | 2236.30 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-07-07 09:15:00 | 2267.90 | 2025-07-07 10:15:00 | 2237.30 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-07-14 09:15:00 | 2189.70 | 2025-07-14 13:15:00 | 2222.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-07-14 12:45:00 | 2210.00 | 2025-07-14 13:15:00 | 2222.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-07-15 09:30:00 | 2205.10 | 2025-07-17 09:15:00 | 2094.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-15 09:30:00 | 2205.10 | 2025-07-17 12:15:00 | 2131.70 | STOP_HIT | 0.50 | 3.33% |
| SELL | retest2 | 2025-08-19 09:15:00 | 1883.90 | 2025-08-19 11:15:00 | 1914.90 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-08-28 12:00:00 | 2101.00 | 2025-09-01 10:15:00 | 2071.50 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-08-28 13:15:00 | 2102.30 | 2025-09-01 10:15:00 | 2071.50 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-09-02 12:00:00 | 2059.10 | 2025-09-02 15:15:00 | 2082.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-09-02 14:15:00 | 2053.80 | 2025-09-02 15:15:00 | 2082.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-09-04 13:00:00 | 2060.00 | 2025-09-10 12:15:00 | 1957.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-04 14:45:00 | 2053.80 | 2025-09-10 12:15:00 | 1951.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-04 13:00:00 | 2060.00 | 2025-09-11 09:15:00 | 2000.40 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2025-09-04 14:45:00 | 2053.80 | 2025-09-11 09:15:00 | 2000.40 | STOP_HIT | 0.50 | 2.60% |
| SELL | retest2 | 2025-09-17 12:30:00 | 1988.00 | 2025-09-17 15:15:00 | 2000.00 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-09-17 14:00:00 | 1987.40 | 2025-09-17 15:15:00 | 2000.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-10-13 09:15:00 | 1883.50 | 2025-10-16 10:15:00 | 1925.00 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-10-17 11:15:00 | 1906.90 | 2025-10-20 09:15:00 | 1874.70 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-10-17 14:00:00 | 1903.50 | 2025-10-20 09:15:00 | 1874.70 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-10-17 14:30:00 | 1905.50 | 2025-10-20 09:15:00 | 1874.70 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-10-20 12:30:00 | 1906.70 | 2025-10-20 13:15:00 | 1879.20 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest1 | 2025-11-10 09:15:00 | 1860.10 | 2025-11-11 09:15:00 | 1990.00 | STOP_HIT | 1.00 | -6.98% |
| SELL | retest2 | 2025-11-18 10:15:00 | 1905.20 | 2025-11-24 09:15:00 | 1994.00 | STOP_HIT | 1.00 | -4.66% |
| SELL | retest2 | 2025-11-18 11:00:00 | 1893.10 | 2025-11-24 09:15:00 | 1994.00 | STOP_HIT | 1.00 | -5.33% |
| SELL | retest2 | 2025-11-18 13:15:00 | 1903.20 | 2025-11-24 09:15:00 | 1994.00 | STOP_HIT | 1.00 | -4.77% |
| SELL | retest2 | 2025-12-04 12:30:00 | 1903.20 | 2025-12-09 14:15:00 | 1918.80 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-12-04 13:30:00 | 1906.60 | 2025-12-09 14:15:00 | 1918.80 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-12-05 11:15:00 | 1901.10 | 2025-12-09 14:15:00 | 1918.80 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-12-12 10:30:00 | 1878.80 | 2025-12-12 13:15:00 | 1893.30 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-18 13:00:00 | 1818.20 | 2025-12-30 09:15:00 | 1727.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-18 13:00:00 | 1818.20 | 2025-12-31 14:15:00 | 1758.70 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2026-01-20 10:30:00 | 1638.40 | 2026-01-21 11:15:00 | 1556.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 10:30:00 | 1638.40 | 2026-01-22 09:15:00 | 1591.00 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2026-01-20 13:00:00 | 1635.00 | 2026-01-27 09:15:00 | 1553.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 13:00:00 | 1635.00 | 2026-01-28 14:15:00 | 1511.40 | STOP_HIT | 0.50 | 7.56% |
| SELL | retest2 | 2026-02-19 09:15:00 | 1299.60 | 2026-02-24 14:15:00 | 1284.10 | STOP_HIT | 1.00 | 1.19% |
| BUY | retest2 | 2026-03-05 15:15:00 | 1370.00 | 2026-03-09 09:15:00 | 1335.30 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2026-03-06 13:15:00 | 1367.00 | 2026-03-09 09:15:00 | 1335.30 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2026-03-12 09:15:00 | 1295.80 | 2026-03-12 11:15:00 | 1335.30 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1213.00 | 2026-03-25 09:15:00 | 1296.00 | STOP_HIT | 1.00 | -6.84% |
| SELL | retest2 | 2026-03-24 15:15:00 | 1232.80 | 2026-03-25 09:15:00 | 1296.00 | STOP_HIT | 1.00 | -5.13% |
| BUY | retest2 | 2026-04-02 11:15:00 | 1280.10 | 2026-04-07 09:15:00 | 1408.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-04 09:15:00 | 1545.70 | 2026-05-07 10:15:00 | 1683.11 | TARGET_HIT | 1.00 | 8.89% |
| BUY | retest2 | 2026-05-04 11:30:00 | 1538.10 | 2026-05-07 10:15:00 | 1685.20 | TARGET_HIT | 1.00 | 9.56% |
| BUY | retest2 | 2026-05-04 13:00:00 | 1530.10 | 2026-05-07 11:15:00 | 1691.91 | TARGET_HIT | 1.00 | 10.58% |
