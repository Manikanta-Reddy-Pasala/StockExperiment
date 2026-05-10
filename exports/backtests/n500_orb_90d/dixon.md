# Dixon Technologies (India) Ltd. (DIXON)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 10825.00
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
| ENTRY1 | 16 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 5 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 11
- **Target hits / Stop hits / Partials:** 5 / 11 / 6
- **Avg / median % per leg:** 0.19% / 0.06%
- **Sum % (uncompounded):** 4.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 5 | 35.7% | 2 | 9 | 3 | 0.07% | 0.9% |
| BUY @ 2nd Alert (retest1) | 14 | 5 | 35.7% | 2 | 9 | 3 | 0.07% | 0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 6 | 75.0% | 3 | 2 | 3 | 0.40% | 3.2% |
| SELL @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 3 | 2 | 3 | 0.40% | 3.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 11 | 50.0% | 5 | 11 | 6 | 0.19% | 4.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:30:00 | 11726.00 | 11642.24 | 0.00 | ORB-long ORB[11550.00,11699.00] vol=3.1x ATR=78.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 12:20:00 | 11844.14 | 11708.69 | 0.00 | T1 1.5R @ 11844.14 |
| Target hit | 2026-02-09 15:20:00 | 11733.00 | 11734.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:40:00 | 11545.00 | 11589.95 | 0.00 | ORB-short ORB[11570.00,11666.00] vol=2.2x ATR=35.04 |
| Stop hit — per-position SL triggered | 2026-02-11 09:55:00 | 11580.04 | 11584.45 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:15:00 | 11802.00 | 11727.27 | 0.00 | ORB-long ORB[11610.00,11754.00] vol=1.6x ATR=35.45 |
| Stop hit — per-position SL triggered | 2026-02-17 10:40:00 | 11766.55 | 11737.03 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 11555.00 | 11613.13 | 0.00 | ORB-short ORB[11586.00,11700.00] vol=3.7x ATR=37.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:45:00 | 11499.27 | 11570.33 | 0.00 | T1 1.5R @ 11499.27 |
| Target hit | 2026-02-18 11:50:00 | 11427.00 | 11419.75 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — SELL (started 2026-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:35:00 | 11330.00 | 11383.56 | 0.00 | ORB-short ORB[11351.00,11464.00] vol=2.0x ATR=37.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:25:00 | 11273.59 | 11339.17 | 0.00 | T1 1.5R @ 11273.59 |
| Target hit | 2026-02-19 12:30:00 | 11263.00 | 11261.35 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — BUY (started 2026-03-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 10:05:00 | 9969.00 | 9847.86 | 0.00 | ORB-long ORB[9770.00,9900.00] vol=1.7x ATR=58.24 |
| Stop hit — per-position SL triggered | 2026-03-04 10:15:00 | 9910.76 | 9856.05 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:35:00 | 10443.00 | 10323.53 | 0.00 | ORB-long ORB[10253.00,10379.00] vol=2.4x ATR=59.65 |
| Stop hit — per-position SL triggered | 2026-03-17 09:40:00 | 10383.35 | 10344.53 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:10:00 | 9995.00 | 10090.49 | 0.00 | ORB-short ORB[10070.00,10200.00] vol=1.5x ATR=43.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:25:00 | 9930.28 | 10065.54 | 0.00 | T1 1.5R @ 9930.28 |
| Target hit | 2026-03-23 13:55:00 | 9942.00 | 9935.84 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — BUY (started 2026-04-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 09:30:00 | 10462.50 | 10367.21 | 0.00 | ORB-long ORB[10288.00,10420.00] vol=1.7x ATR=55.36 |
| Stop hit — per-position SL triggered | 2026-04-13 09:35:00 | 10407.14 | 10372.10 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:15:00 | 10906.50 | 10819.21 | 0.00 | ORB-long ORB[10714.50,10849.00] vol=1.9x ATR=43.82 |
| Stop hit — per-position SL triggered | 2026-04-15 10:35:00 | 10862.68 | 10832.12 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:10:00 | 11435.00 | 11355.78 | 0.00 | ORB-long ORB[11211.00,11374.00] vol=2.4x ATR=32.77 |
| Stop hit — per-position SL triggered | 2026-04-21 10:15:00 | 11402.23 | 11359.33 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 10:15:00 | 11053.00 | 10902.42 | 0.00 | ORB-long ORB[10801.00,10935.00] vol=1.9x ATR=48.18 |
| Stop hit — per-position SL triggered | 2026-04-24 10:30:00 | 11004.82 | 10913.81 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:55:00 | 11073.00 | 10971.94 | 0.00 | ORB-long ORB[10855.00,10975.00] vol=1.6x ATR=40.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 11:25:00 | 11133.05 | 11002.74 | 0.00 | T1 1.5R @ 11133.05 |
| Target hit | 2026-04-27 15:20:00 | 11329.00 | 11115.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2026-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:35:00 | 11504.50 | 11429.56 | 0.00 | ORB-long ORB[11360.00,11485.00] vol=1.8x ATR=45.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 09:40:00 | 11572.88 | 11446.91 | 0.00 | T1 1.5R @ 11572.88 |
| Stop hit — per-position SL triggered | 2026-04-29 09:45:00 | 11504.50 | 11448.61 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 11:00:00 | 11070.00 | 11136.87 | 0.00 | ORB-short ORB[11152.50,11300.00] vol=1.8x ATR=36.23 |
| Stop hit — per-position SL triggered | 2026-04-30 11:05:00 | 11106.23 | 11135.49 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 11650.00 | 11514.85 | 0.00 | ORB-long ORB[11368.00,11469.00] vol=4.2x ATR=53.53 |
| Stop hit — per-position SL triggered | 2026-05-05 09:35:00 | 11596.47 | 11543.93 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:30:00 | 11726.00 | 2026-02-09 12:20:00 | 11844.14 | PARTIAL | 0.50 | 1.01% |
| BUY | retest1 | 2026-02-09 10:30:00 | 11726.00 | 2026-02-09 15:20:00 | 11733.00 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2026-02-11 09:40:00 | 11545.00 | 2026-02-11 09:55:00 | 11580.04 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-17 10:15:00 | 11802.00 | 2026-02-17 10:40:00 | 11766.55 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-18 09:30:00 | 11555.00 | 2026-02-18 09:45:00 | 11499.27 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-02-18 09:30:00 | 11555.00 | 2026-02-18 11:50:00 | 11427.00 | TARGET_HIT | 0.50 | 1.11% |
| SELL | retest1 | 2026-02-19 09:35:00 | 11330.00 | 2026-02-19 10:25:00 | 11273.59 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-02-19 09:35:00 | 11330.00 | 2026-02-19 12:30:00 | 11263.00 | TARGET_HIT | 0.50 | 0.59% |
| BUY | retest1 | 2026-03-04 10:05:00 | 9969.00 | 2026-03-04 10:15:00 | 9910.76 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2026-03-17 09:35:00 | 10443.00 | 2026-03-17 09:40:00 | 10383.35 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2026-03-23 10:10:00 | 9995.00 | 2026-03-23 10:25:00 | 9930.28 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-03-23 10:10:00 | 9995.00 | 2026-03-23 13:55:00 | 9942.00 | TARGET_HIT | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-13 09:30:00 | 10462.50 | 2026-04-13 09:35:00 | 10407.14 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2026-04-15 10:15:00 | 10906.50 | 2026-04-15 10:35:00 | 10862.68 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-21 10:10:00 | 11435.00 | 2026-04-21 10:15:00 | 11402.23 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-24 10:15:00 | 11053.00 | 2026-04-24 10:30:00 | 11004.82 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-27 10:55:00 | 11073.00 | 2026-04-27 11:25:00 | 11133.05 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-04-27 10:55:00 | 11073.00 | 2026-04-27 15:20:00 | 11329.00 | TARGET_HIT | 0.50 | 2.31% |
| BUY | retest1 | 2026-04-29 09:35:00 | 11504.50 | 2026-04-29 09:40:00 | 11572.88 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-04-29 09:35:00 | 11504.50 | 2026-04-29 09:45:00 | 11504.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-30 11:00:00 | 11070.00 | 2026-04-30 11:05:00 | 11106.23 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-05-05 09:30:00 | 11650.00 | 2026-05-05 09:35:00 | 11596.47 | STOP_HIT | 1.00 | -0.46% |
