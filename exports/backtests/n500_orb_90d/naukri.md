# Info Edge (India) Ltd. (NAUKRI)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 978.40
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 9
- **Target hits / Stop hits / Partials:** 3 / 9 / 5
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 1.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 6 | 60.0% | 2 | 4 | 4 | 0.20% | 2.0% |
| BUY @ 2nd Alert (retest1) | 10 | 6 | 60.0% | 2 | 4 | 4 | 0.20% | 2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.01% | -0.1% |
| SELL @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.01% | -0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 8 | 47.1% | 3 | 9 | 5 | 0.11% | 1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:45:00 | 1187.00 | 1178.29 | 0.00 | ORB-long ORB[1170.60,1183.20] vol=2.1x ATR=3.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:50:00 | 1192.68 | 1180.72 | 0.00 | T1 1.5R @ 1192.68 |
| Target hit | 2026-02-10 11:15:00 | 1192.90 | 1193.22 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:15:00 | 1140.80 | 1132.32 | 0.00 | ORB-long ORB[1115.50,1132.20] vol=1.7x ATR=3.95 |
| Stop hit — per-position SL triggered | 2026-02-17 10:20:00 | 1136.85 | 1132.69 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:45:00 | 1122.40 | 1124.02 | 0.00 | ORB-short ORB[1123.10,1135.00] vol=3.2x ATR=3.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:50:00 | 1116.53 | 1119.76 | 0.00 | T1 1.5R @ 1116.53 |
| Target hit | 2026-02-18 13:15:00 | 1110.80 | 1108.43 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — SELL (started 2026-03-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 10:50:00 | 971.50 | 973.56 | 0.00 | ORB-short ORB[974.30,988.00] vol=4.0x ATR=3.02 |
| Stop hit — per-position SL triggered | 2026-03-10 11:05:00 | 974.52 | 973.27 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:40:00 | 967.50 | 973.55 | 0.00 | ORB-short ORB[971.60,978.90] vol=1.6x ATR=3.11 |
| Stop hit — per-position SL triggered | 2026-03-11 09:45:00 | 970.61 | 973.28 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 10:10:00 | 940.00 | 944.74 | 0.00 | ORB-short ORB[943.00,952.00] vol=1.6x ATR=2.83 |
| Stop hit — per-position SL triggered | 2026-03-12 10:35:00 | 942.83 | 941.62 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 10:50:00 | 959.90 | 951.50 | 0.00 | ORB-long ORB[947.30,956.10] vol=1.5x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:55:00 | 964.06 | 952.25 | 0.00 | T1 1.5R @ 964.06 |
| Stop hit — per-position SL triggered | 2026-03-13 12:20:00 | 959.90 | 957.62 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 10:30:00 | 1064.35 | 1056.93 | 0.00 | ORB-long ORB[1050.05,1064.00] vol=1.5x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 10:40:00 | 1069.08 | 1058.28 | 0.00 | T1 1.5R @ 1069.08 |
| Stop hit — per-position SL triggered | 2026-04-20 11:10:00 | 1064.35 | 1060.89 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:10:00 | 1076.90 | 1072.55 | 0.00 | ORB-long ORB[1067.00,1076.50] vol=2.3x ATR=2.47 |
| Stop hit — per-position SL triggered | 2026-04-21 11:20:00 | 1074.43 | 1072.68 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:50:00 | 1003.45 | 1005.82 | 0.00 | ORB-short ORB[1003.65,1016.30] vol=1.8x ATR=3.51 |
| Stop hit — per-position SL triggered | 2026-04-24 10:10:00 | 1006.96 | 1005.81 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:10:00 | 1003.15 | 999.28 | 0.00 | ORB-long ORB[991.45,1002.55] vol=1.7x ATR=3.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:20:00 | 1008.13 | 1001.42 | 0.00 | T1 1.5R @ 1008.13 |
| Target hit | 2026-04-28 12:30:00 | 1005.00 | 1006.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2026-05-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:50:00 | 977.20 | 980.52 | 0.00 | ORB-short ORB[978.75,988.70] vol=2.2x ATR=3.33 |
| Stop hit — per-position SL triggered | 2026-05-08 10:20:00 | 980.53 | 978.41 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:45:00 | 1187.00 | 2026-02-10 09:50:00 | 1192.68 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-02-10 09:45:00 | 1187.00 | 2026-02-10 11:15:00 | 1192.90 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2026-02-17 10:15:00 | 1140.80 | 2026-02-17 10:20:00 | 1136.85 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-18 09:45:00 | 1122.40 | 2026-02-18 09:50:00 | 1116.53 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-02-18 09:45:00 | 1122.40 | 2026-02-18 13:15:00 | 1110.80 | TARGET_HIT | 0.50 | 1.03% |
| SELL | retest1 | 2026-03-10 10:50:00 | 971.50 | 2026-03-10 11:05:00 | 974.52 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-11 09:40:00 | 967.50 | 2026-03-11 09:45:00 | 970.61 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-12 10:10:00 | 940.00 | 2026-03-12 10:35:00 | 942.83 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-13 10:50:00 | 959.90 | 2026-03-13 10:55:00 | 964.06 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-03-13 10:50:00 | 959.90 | 2026-03-13 12:20:00 | 959.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-20 10:30:00 | 1064.35 | 2026-04-20 10:40:00 | 1069.08 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-04-20 10:30:00 | 1064.35 | 2026-04-20 11:10:00 | 1064.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 11:10:00 | 1076.90 | 2026-04-21 11:20:00 | 1074.43 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-04-24 09:50:00 | 1003.45 | 2026-04-24 10:10:00 | 1006.96 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-28 10:10:00 | 1003.15 | 2026-04-28 10:20:00 | 1008.13 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-28 10:10:00 | 1003.15 | 2026-04-28 12:30:00 | 1005.00 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2026-05-08 09:50:00 | 977.20 | 2026-05-08 10:20:00 | 980.53 | STOP_HIT | 1.00 | -0.34% |
