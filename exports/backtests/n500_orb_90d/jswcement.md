# JSW Cement Ltd. (JSWCEMENT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 124.32
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
| ENTRY1 | 22 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 19
- **Target hits / Stop hits / Partials:** 3 / 19 / 8
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 2.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 5 | 35.7% | 1 | 9 | 4 | 0.04% | 0.5% |
| BUY @ 2nd Alert (retest1) | 14 | 5 | 35.7% | 1 | 9 | 4 | 0.04% | 0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 6 | 37.5% | 2 | 10 | 4 | 0.14% | 2.3% |
| SELL @ 2nd Alert (retest1) | 16 | 6 | 37.5% | 2 | 10 | 4 | 0.14% | 2.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 30 | 11 | 36.7% | 3 | 19 | 8 | 0.09% | 2.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:40:00 | 122.20 | 122.92 | 0.00 | ORB-short ORB[123.20,124.38] vol=1.6x ATR=0.41 |
| Stop hit — per-position SL triggered | 2026-02-10 11:40:00 | 122.61 | 122.84 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 120.62 | 121.13 | 0.00 | ORB-short ORB[120.85,122.50] vol=1.8x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:10:00 | 120.09 | 120.73 | 0.00 | T1 1.5R @ 120.09 |
| Target hit | 2026-02-11 15:20:00 | 118.52 | 119.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:10:00 | 117.32 | 117.76 | 0.00 | ORB-short ORB[117.60,119.00] vol=1.8x ATR=0.44 |
| Stop hit — per-position SL triggered | 2026-02-13 10:25:00 | 117.76 | 117.76 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 10:35:00 | 117.57 | 117.97 | 0.00 | ORB-short ORB[117.81,119.01] vol=2.4x ATR=0.41 |
| Stop hit — per-position SL triggered | 2026-02-16 10:40:00 | 117.98 | 117.89 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:50:00 | 121.79 | 120.79 | 0.00 | ORB-long ORB[119.10,120.80] vol=3.1x ATR=0.47 |
| Stop hit — per-position SL triggered | 2026-02-17 10:30:00 | 121.32 | 121.12 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:40:00 | 123.79 | 122.97 | 0.00 | ORB-long ORB[122.00,123.33] vol=2.0x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:45:00 | 124.33 | 123.05 | 0.00 | T1 1.5R @ 124.33 |
| Stop hit — per-position SL triggered | 2026-02-18 12:45:00 | 123.79 | 123.55 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 11:10:00 | 126.08 | 125.16 | 0.00 | ORB-long ORB[124.30,125.62] vol=5.5x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:25:00 | 126.73 | 125.77 | 0.00 | T1 1.5R @ 126.73 |
| Target hit | 2026-02-26 14:50:00 | 127.02 | 127.06 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2026-03-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:05:00 | 119.17 | 120.03 | 0.00 | ORB-short ORB[119.61,120.68] vol=2.5x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:20:00 | 118.45 | 119.66 | 0.00 | T1 1.5R @ 118.45 |
| Stop hit — per-position SL triggered | 2026-03-05 11:00:00 | 119.17 | 119.41 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:15:00 | 120.59 | 119.79 | 0.00 | ORB-long ORB[118.52,120.03] vol=1.6x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 11:40:00 | 121.13 | 119.92 | 0.00 | T1 1.5R @ 121.13 |
| Stop hit — per-position SL triggered | 2026-03-12 12:40:00 | 120.59 | 120.26 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:35:00 | 117.62 | 118.35 | 0.00 | ORB-short ORB[118.20,119.66] vol=7.1x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:20:00 | 116.84 | 118.04 | 0.00 | T1 1.5R @ 116.84 |
| Stop hit — per-position SL triggered | 2026-03-13 11:35:00 | 117.62 | 117.90 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:25:00 | 115.00 | 115.63 | 0.00 | ORB-short ORB[115.09,116.50] vol=2.9x ATR=0.43 |
| Stop hit — per-position SL triggered | 2026-03-16 10:35:00 | 115.43 | 115.62 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 10:55:00 | 115.99 | 116.50 | 0.00 | ORB-short ORB[116.46,117.80] vol=1.6x ATR=0.43 |
| Stop hit — per-position SL triggered | 2026-03-17 11:15:00 | 116.42 | 116.42 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-03-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 11:05:00 | 116.30 | 116.82 | 0.00 | ORB-short ORB[116.32,117.36] vol=2.3x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-03-19 11:10:00 | 116.60 | 116.82 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:45:00 | 123.48 | 122.46 | 0.00 | ORB-long ORB[121.45,123.00] vol=2.4x ATR=0.64 |
| Stop hit — per-position SL triggered | 2026-04-08 09:55:00 | 122.84 | 122.55 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:25:00 | 128.00 | 127.51 | 0.00 | ORB-long ORB[126.50,127.87] vol=6.6x ATR=0.44 |
| Stop hit — per-position SL triggered | 2026-04-10 10:45:00 | 127.56 | 127.55 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:35:00 | 127.64 | 128.43 | 0.00 | ORB-short ORB[128.10,129.60] vol=2.1x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:00:00 | 127.14 | 128.13 | 0.00 | T1 1.5R @ 127.14 |
| Target hit | 2026-04-16 15:20:00 | 126.10 | 126.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2026-04-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:45:00 | 128.19 | 127.65 | 0.00 | ORB-long ORB[127.00,128.07] vol=1.8x ATR=0.32 |
| Stop hit — per-position SL triggered | 2026-04-23 10:30:00 | 127.87 | 127.81 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:55:00 | 124.50 | 125.15 | 0.00 | ORB-short ORB[124.91,126.55] vol=1.7x ATR=0.44 |
| Stop hit — per-position SL triggered | 2026-04-24 10:00:00 | 124.94 | 125.13 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-04-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:45:00 | 125.64 | 124.79 | 0.00 | ORB-long ORB[123.70,125.48] vol=1.9x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 11:00:00 | 126.26 | 124.91 | 0.00 | T1 1.5R @ 126.26 |
| Stop hit — per-position SL triggered | 2026-04-27 11:10:00 | 125.64 | 124.93 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2026-04-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:40:00 | 124.10 | 124.70 | 0.00 | ORB-short ORB[124.31,125.48] vol=2.4x ATR=0.46 |
| Stop hit — per-position SL triggered | 2026-04-28 10:25:00 | 124.56 | 124.48 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2026-05-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:45:00 | 123.92 | 122.91 | 0.00 | ORB-long ORB[122.31,123.14] vol=2.4x ATR=0.38 |
| Stop hit — per-position SL triggered | 2026-05-05 10:50:00 | 123.54 | 123.40 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2026-05-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:30:00 | 126.27 | 125.54 | 0.00 | ORB-long ORB[124.80,125.88] vol=2.5x ATR=0.40 |
| Stop hit — per-position SL triggered | 2026-05-07 09:35:00 | 125.87 | 125.59 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:40:00 | 122.20 | 2026-02-10 11:40:00 | 122.61 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-11 09:30:00 | 120.62 | 2026-02-11 10:10:00 | 120.09 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-02-11 09:30:00 | 120.62 | 2026-02-11 15:20:00 | 118.52 | TARGET_HIT | 0.50 | 1.74% |
| SELL | retest1 | 2026-02-13 10:10:00 | 117.32 | 2026-02-13 10:25:00 | 117.76 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-16 10:35:00 | 117.57 | 2026-02-16 10:40:00 | 117.98 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-17 09:50:00 | 121.79 | 2026-02-17 10:30:00 | 121.32 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-18 10:40:00 | 123.79 | 2026-02-18 10:45:00 | 124.33 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-18 10:40:00 | 123.79 | 2026-02-18 12:45:00 | 123.79 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 11:10:00 | 126.08 | 2026-02-26 11:25:00 | 126.73 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-02-26 11:10:00 | 126.08 | 2026-02-26 14:50:00 | 127.02 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2026-03-05 10:05:00 | 119.17 | 2026-03-05 10:20:00 | 118.45 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-03-05 10:05:00 | 119.17 | 2026-03-05 11:00:00 | 119.17 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-12 11:15:00 | 120.59 | 2026-03-12 11:40:00 | 121.13 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-03-12 11:15:00 | 120.59 | 2026-03-12 12:40:00 | 120.59 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 10:35:00 | 117.62 | 2026-03-13 11:20:00 | 116.84 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-03-13 10:35:00 | 117.62 | 2026-03-13 11:35:00 | 117.62 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-16 10:25:00 | 115.00 | 2026-03-16 10:35:00 | 115.43 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-17 10:55:00 | 115.99 | 2026-03-17 11:15:00 | 116.42 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-19 11:05:00 | 116.30 | 2026-03-19 11:10:00 | 116.60 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-08 09:45:00 | 123.48 | 2026-04-08 09:55:00 | 122.84 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2026-04-10 10:25:00 | 128.00 | 2026-04-10 10:45:00 | 127.56 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-16 09:35:00 | 127.64 | 2026-04-16 10:00:00 | 127.14 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-04-16 09:35:00 | 127.64 | 2026-04-16 15:20:00 | 126.10 | TARGET_HIT | 0.50 | 1.21% |
| BUY | retest1 | 2026-04-23 09:45:00 | 128.19 | 2026-04-23 10:30:00 | 127.87 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-04-24 09:55:00 | 124.50 | 2026-04-24 10:00:00 | 124.94 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-27 10:45:00 | 125.64 | 2026-04-27 11:00:00 | 126.26 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-27 10:45:00 | 125.64 | 2026-04-27 11:10:00 | 125.64 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-28 09:40:00 | 124.10 | 2026-04-28 10:25:00 | 124.56 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-05-05 09:45:00 | 123.92 | 2026-05-05 10:50:00 | 123.54 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-05-07 09:30:00 | 126.27 | 2026-05-07 09:35:00 | 125.87 | STOP_HIT | 1.00 | -0.32% |
