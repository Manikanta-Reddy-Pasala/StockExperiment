# Abbott India Ltd. (ABBOTINDIA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 26850.00
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
| ENTRY1 | 26 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 24
- **Target hits / Stop hits / Partials:** 2 / 24 / 6
- **Avg / median % per leg:** -0.00% / -0.18%
- **Sum % (uncompounded):** -0.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 5 | 31.2% | 1 | 11 | 4 | 0.10% | 1.6% |
| BUY @ 2nd Alert (retest1) | 16 | 5 | 31.2% | 1 | 11 | 4 | 0.10% | 1.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 3 | 18.8% | 1 | 13 | 2 | -0.11% | -1.7% |
| SELL @ 2nd Alert (retest1) | 16 | 3 | 18.8% | 1 | 13 | 2 | -0.11% | -1.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 32 | 8 | 25.0% | 2 | 24 | 6 | -0.00% | -0.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:40:00 | 27140.00 | 27243.59 | 0.00 | ORB-short ORB[27150.00,27495.00] vol=1.6x ATR=63.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:45:00 | 27044.29 | 27173.28 | 0.00 | T1 1.5R @ 27044.29 |
| Stop hit — per-position SL triggered | 2026-02-11 10:45:00 | 27140.00 | 27136.05 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 09:55:00 | 26740.00 | 26872.35 | 0.00 | ORB-short ORB[26745.00,27135.00] vol=1.7x ATR=81.67 |
| Stop hit — per-position SL triggered | 2026-02-12 10:00:00 | 26821.67 | 26861.81 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:40:00 | 26590.00 | 26502.70 | 0.00 | ORB-long ORB[26365.00,26565.00] vol=1.5x ATR=48.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:00:00 | 26662.77 | 26517.20 | 0.00 | T1 1.5R @ 26662.77 |
| Stop hit — per-position SL triggered | 2026-02-16 11:15:00 | 26590.00 | 26522.17 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:40:00 | 26735.00 | 26624.86 | 0.00 | ORB-long ORB[26440.00,26595.00] vol=5.4x ATR=58.31 |
| Stop hit — per-position SL triggered | 2026-02-20 09:45:00 | 26676.69 | 26643.83 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 10:15:00 | 26510.00 | 26446.82 | 0.00 | ORB-long ORB[26240.00,26495.00] vol=2.6x ATR=53.69 |
| Stop hit — per-position SL triggered | 2026-02-23 10:20:00 | 26456.31 | 26448.11 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:30:00 | 26245.00 | 26305.81 | 0.00 | ORB-short ORB[26300.00,26520.00] vol=1.6x ATR=47.14 |
| Stop hit — per-position SL triggered | 2026-02-24 10:40:00 | 26292.14 | 26305.04 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-02 09:30:00 | 26400.00 | 26307.08 | 0.00 | ORB-long ORB[26000.00,26365.00] vol=2.9x ATR=92.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:35:00 | 26539.03 | 26377.53 | 0.00 | T1 1.5R @ 26539.03 |
| Target hit | 2026-03-02 15:20:00 | 26885.00 | 26669.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-03-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:35:00 | 27420.00 | 27486.81 | 0.00 | ORB-short ORB[27445.00,27630.00] vol=1.8x ATR=82.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 09:45:00 | 27296.08 | 27455.17 | 0.00 | T1 1.5R @ 27296.08 |
| Stop hit — per-position SL triggered | 2026-03-06 10:15:00 | 27420.00 | 27430.75 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 10:25:00 | 27140.00 | 26937.93 | 0.00 | ORB-long ORB[26735.00,27095.00] vol=2.9x ATR=96.13 |
| Stop hit — per-position SL triggered | 2026-03-09 10:30:00 | 27043.87 | 26942.87 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:35:00 | 26830.00 | 26960.45 | 0.00 | ORB-short ORB[26915.00,27095.00] vol=1.7x ATR=69.13 |
| Stop hit — per-position SL triggered | 2026-03-11 12:25:00 | 26899.13 | 26914.23 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 10:15:00 | 26465.00 | 26401.26 | 0.00 | ORB-long ORB[26190.00,26370.00] vol=2.3x ATR=80.18 |
| Stop hit — per-position SL triggered | 2026-03-19 11:00:00 | 26384.82 | 26406.21 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 10:55:00 | 26225.00 | 26326.32 | 0.00 | ORB-short ORB[26325.00,26555.00] vol=2.1x ATR=79.99 |
| Stop hit — per-position SL triggered | 2026-03-27 11:00:00 | 26304.99 | 26322.80 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-06 09:55:00 | 25995.00 | 26116.29 | 0.00 | ORB-short ORB[26150.00,26440.00] vol=5.9x ATR=85.13 |
| Stop hit — per-position SL triggered | 2026-04-06 10:35:00 | 26080.13 | 26077.25 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 11:15:00 | 25860.00 | 26014.53 | 0.00 | ORB-short ORB[25935.00,26110.00] vol=1.8x ATR=68.51 |
| Target hit | 2026-04-08 15:20:00 | 25810.00 | 25919.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2026-04-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:50:00 | 25845.00 | 25607.09 | 0.00 | ORB-long ORB[25310.00,25625.00] vol=2.4x ATR=66.96 |
| Stop hit — per-position SL triggered | 2026-04-13 10:55:00 | 25778.04 | 25610.58 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 25920.00 | 25832.76 | 0.00 | ORB-long ORB[25760.00,25910.00] vol=1.7x ATR=70.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:50:00 | 26025.08 | 25889.34 | 0.00 | T1 1.5R @ 26025.08 |
| Stop hit — per-position SL triggered | 2026-04-15 13:40:00 | 25920.00 | 25932.37 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-04-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 09:35:00 | 25730.00 | 25801.95 | 0.00 | ORB-short ORB[25735.00,25990.00] vol=3.7x ATR=71.66 |
| Stop hit — per-position SL triggered | 2026-04-17 10:10:00 | 25801.66 | 25756.40 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-20 11:10:00 | 25530.00 | 25698.22 | 0.00 | ORB-short ORB[25665.00,25890.00] vol=4.8x ATR=56.33 |
| Stop hit — per-position SL triggered | 2026-04-20 11:30:00 | 25586.33 | 25674.38 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-04-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 11:05:00 | 25475.00 | 25529.17 | 0.00 | ORB-short ORB[25500.00,25690.00] vol=4.0x ATR=39.45 |
| Stop hit — per-position SL triggered | 2026-04-21 11:40:00 | 25514.45 | 25520.45 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 11:15:00 | 25600.00 | 25559.26 | 0.00 | ORB-long ORB[25400.00,25545.00] vol=7.8x ATR=65.84 |
| Stop hit — per-position SL triggered | 2026-04-22 11:20:00 | 25534.16 | 25559.24 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2026-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:10:00 | 25405.00 | 25504.25 | 0.00 | ORB-short ORB[25440.00,25625.00] vol=2.5x ATR=46.60 |
| Stop hit — per-position SL triggered | 2026-04-23 11:25:00 | 25451.60 | 25492.52 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2026-04-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:55:00 | 25370.00 | 25286.96 | 0.00 | ORB-long ORB[25180.00,25350.00] vol=1.6x ATR=61.22 |
| Stop hit — per-position SL triggered | 2026-04-27 10:30:00 | 25308.78 | 25305.92 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2026-04-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:50:00 | 25205.00 | 25339.58 | 0.00 | ORB-short ORB[25290.00,25475.00] vol=2.5x ATR=67.60 |
| Stop hit — per-position SL triggered | 2026-04-28 11:30:00 | 25272.60 | 25320.26 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2026-04-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:00:00 | 25740.00 | 25614.13 | 0.00 | ORB-long ORB[25495.00,25645.00] vol=3.7x ATR=57.22 |
| Stop hit — per-position SL triggered | 2026-04-29 11:45:00 | 25682.78 | 25633.07 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2026-05-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 09:50:00 | 25310.00 | 25327.82 | 0.00 | ORB-short ORB[25350.00,25575.00] vol=2.7x ATR=65.88 |
| Stop hit — per-position SL triggered | 2026-05-05 10:05:00 | 25375.88 | 25334.09 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2026-05-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:45:00 | 26725.00 | 26600.82 | 0.00 | ORB-long ORB[26430.00,26665.00] vol=1.7x ATR=106.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 09:50:00 | 26885.05 | 26660.95 | 0.00 | T1 1.5R @ 26885.05 |
| Stop hit — per-position SL triggered | 2026-05-07 09:55:00 | 26725.00 | 26668.47 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:40:00 | 27140.00 | 2026-02-11 09:45:00 | 27044.29 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-11 09:40:00 | 27140.00 | 2026-02-11 10:45:00 | 27140.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-12 09:55:00 | 26740.00 | 2026-02-12 10:00:00 | 26821.67 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-16 10:40:00 | 26590.00 | 2026-02-16 11:00:00 | 26662.77 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2026-02-16 10:40:00 | 26590.00 | 2026-02-16 11:15:00 | 26590.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 09:40:00 | 26735.00 | 2026-02-20 09:45:00 | 26676.69 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-23 10:15:00 | 26510.00 | 2026-02-23 10:20:00 | 26456.31 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-24 10:30:00 | 26245.00 | 2026-02-24 10:40:00 | 26292.14 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-03-02 09:30:00 | 26400.00 | 2026-03-02 09:35:00 | 26539.03 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-03-02 09:30:00 | 26400.00 | 2026-03-02 15:20:00 | 26885.00 | TARGET_HIT | 0.50 | 1.84% |
| SELL | retest1 | 2026-03-06 09:35:00 | 27420.00 | 2026-03-06 09:45:00 | 27296.08 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-03-06 09:35:00 | 27420.00 | 2026-03-06 10:15:00 | 27420.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-09 10:25:00 | 27140.00 | 2026-03-09 10:30:00 | 27043.87 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-11 10:35:00 | 26830.00 | 2026-03-11 12:25:00 | 26899.13 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-19 10:15:00 | 26465.00 | 2026-03-19 11:00:00 | 26384.82 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-27 10:55:00 | 26225.00 | 2026-03-27 11:00:00 | 26304.99 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-06 09:55:00 | 25995.00 | 2026-04-06 10:35:00 | 26080.13 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-08 11:15:00 | 25860.00 | 2026-04-08 15:20:00 | 25810.00 | TARGET_HIT | 1.00 | 0.19% |
| BUY | retest1 | 2026-04-13 10:50:00 | 25845.00 | 2026-04-13 10:55:00 | 25778.04 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-15 09:40:00 | 25920.00 | 2026-04-15 11:50:00 | 26025.08 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-04-15 09:40:00 | 25920.00 | 2026-04-15 13:40:00 | 25920.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-17 09:35:00 | 25730.00 | 2026-04-17 10:10:00 | 25801.66 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-20 11:10:00 | 25530.00 | 2026-04-20 11:30:00 | 25586.33 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-21 11:05:00 | 25475.00 | 2026-04-21 11:40:00 | 25514.45 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2026-04-22 11:15:00 | 25600.00 | 2026-04-22 11:20:00 | 25534.16 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-23 11:10:00 | 25405.00 | 2026-04-23 11:25:00 | 25451.60 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-04-27 09:55:00 | 25370.00 | 2026-04-27 10:30:00 | 25308.78 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-04-28 10:50:00 | 25205.00 | 2026-04-28 11:30:00 | 25272.60 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-29 11:00:00 | 25740.00 | 2026-04-29 11:45:00 | 25682.78 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-05-05 09:50:00 | 25310.00 | 2026-05-05 10:05:00 | 25375.88 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-05-07 09:45:00 | 26725.00 | 2026-05-07 09:50:00 | 26885.05 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-05-07 09:45:00 | 26725.00 | 2026-05-07 09:55:00 | 26725.00 | STOP_HIT | 0.50 | 0.00% |
