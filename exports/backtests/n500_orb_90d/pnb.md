# Punjab National Bank (PNB)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 107.20
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
| ENTRY1 | 15 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 4 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 11
- **Target hits / Stop hits / Partials:** 4 / 11 / 8
- **Avg / median % per leg:** 0.33% / 0.30%
- **Sum % (uncompounded):** 7.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 7 | 53.8% | 3 | 6 | 4 | 0.40% | 5.2% |
| BUY @ 2nd Alert (retest1) | 13 | 7 | 53.8% | 3 | 6 | 4 | 0.40% | 5.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 5 | 50.0% | 1 | 5 | 4 | 0.24% | 2.4% |
| SELL @ 2nd Alert (retest1) | 10 | 5 | 50.0% | 1 | 5 | 4 | 0.24% | 2.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 12 | 52.2% | 4 | 11 | 8 | 0.33% | 7.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 121.41 | 122.03 | 0.00 | ORB-short ORB[121.65,122.91] vol=1.8x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 12:00:00 | 121.04 | 121.87 | 0.00 | T1 1.5R @ 121.04 |
| Stop hit — per-position SL triggered | 2026-02-12 12:30:00 | 121.41 | 121.80 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:15:00 | 122.13 | 121.26 | 0.00 | ORB-long ORB[119.81,121.58] vol=1.5x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:20:00 | 122.66 | 121.57 | 0.00 | T1 1.5R @ 122.66 |
| Target hit | 2026-02-17 15:20:00 | 124.91 | 124.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 126.45 | 125.88 | 0.00 | ORB-long ORB[124.83,126.35] vol=2.4x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:40:00 | 127.01 | 126.14 | 0.00 | T1 1.5R @ 127.01 |
| Target hit | 2026-02-18 15:20:00 | 128.15 | 127.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-02-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:45:00 | 130.53 | 130.91 | 0.00 | ORB-short ORB[130.56,131.66] vol=1.7x ATR=0.35 |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 130.88 | 130.86 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:40:00 | 117.38 | 117.82 | 0.00 | ORB-short ORB[117.50,118.44] vol=1.6x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:30:00 | 116.98 | 117.67 | 0.00 | T1 1.5R @ 116.98 |
| Target hit | 2026-03-11 15:20:00 | 115.81 | 117.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2026-03-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:50:00 | 115.55 | 114.69 | 0.00 | ORB-long ORB[113.66,114.94] vol=1.6x ATR=0.47 |
| Stop hit — per-position SL triggered | 2026-03-12 10:15:00 | 115.08 | 114.90 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:40:00 | 114.34 | 115.13 | 0.00 | ORB-short ORB[114.55,116.27] vol=1.7x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:55:00 | 113.72 | 114.85 | 0.00 | T1 1.5R @ 113.72 |
| Stop hit — per-position SL triggered | 2026-03-13 11:30:00 | 114.34 | 114.32 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:45:00 | 109.98 | 109.24 | 0.00 | ORB-long ORB[108.55,109.67] vol=1.5x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 14:30:00 | 110.80 | 109.83 | 0.00 | T1 1.5R @ 110.80 |
| Target hit | 2026-04-08 15:20:00 | 111.13 | 110.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 112.86 | 113.46 | 0.00 | ORB-short ORB[113.06,114.05] vol=1.6x ATR=0.41 |
| Stop hit — per-position SL triggered | 2026-04-15 09:40:00 | 113.27 | 113.44 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:55:00 | 114.62 | 114.04 | 0.00 | ORB-long ORB[113.05,114.20] vol=2.0x ATR=0.25 |
| Stop hit — per-position SL triggered | 2026-04-16 11:25:00 | 114.37 | 114.13 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:05:00 | 114.32 | 113.87 | 0.00 | ORB-long ORB[112.85,114.10] vol=1.5x ATR=0.32 |
| Stop hit — per-position SL triggered | 2026-04-17 10:30:00 | 114.00 | 113.93 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 114.69 | 114.23 | 0.00 | ORB-long ORB[113.60,114.49] vol=1.6x ATR=0.26 |
| Stop hit — per-position SL triggered | 2026-04-21 09:40:00 | 114.43 | 114.31 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:50:00 | 115.08 | 114.49 | 0.00 | ORB-long ORB[113.35,114.55] vol=3.3x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:20:00 | 115.49 | 114.68 | 0.00 | T1 1.5R @ 115.49 |
| Stop hit — per-position SL triggered | 2026-04-22 11:45:00 | 115.08 | 114.74 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:55:00 | 112.68 | 112.13 | 0.00 | ORB-long ORB[111.24,112.60] vol=1.5x ATR=0.33 |
| Stop hit — per-position SL triggered | 2026-04-28 11:05:00 | 112.35 | 112.31 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:10:00 | 108.85 | 109.48 | 0.00 | ORB-short ORB[109.17,110.80] vol=1.8x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:45:00 | 108.32 | 109.18 | 0.00 | T1 1.5R @ 108.32 |
| Stop hit — per-position SL triggered | 2026-04-30 12:10:00 | 108.85 | 109.03 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 11:15:00 | 121.41 | 2026-02-12 12:00:00 | 121.04 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-02-12 11:15:00 | 121.41 | 2026-02-12 12:30:00 | 121.41 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:15:00 | 122.13 | 2026-02-17 10:20:00 | 122.66 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-17 10:15:00 | 122.13 | 2026-02-17 15:20:00 | 124.91 | TARGET_HIT | 0.50 | 2.28% |
| BUY | retest1 | 2026-02-18 09:30:00 | 126.45 | 2026-02-18 09:40:00 | 127.01 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-18 09:30:00 | 126.45 | 2026-02-18 15:20:00 | 128.15 | TARGET_HIT | 0.50 | 1.34% |
| SELL | retest1 | 2026-02-25 09:45:00 | 130.53 | 2026-02-25 10:15:00 | 130.88 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-03-11 10:40:00 | 117.38 | 2026-03-11 11:30:00 | 116.98 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-03-11 10:40:00 | 117.38 | 2026-03-11 15:20:00 | 115.81 | TARGET_HIT | 0.50 | 1.34% |
| BUY | retest1 | 2026-03-12 09:50:00 | 115.55 | 2026-03-12 10:15:00 | 115.08 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-03-13 09:40:00 | 114.34 | 2026-03-13 09:55:00 | 113.72 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-03-13 09:40:00 | 114.34 | 2026-03-13 11:30:00 | 114.34 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-08 09:45:00 | 109.98 | 2026-04-08 14:30:00 | 110.80 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2026-04-08 09:45:00 | 109.98 | 2026-04-08 15:20:00 | 111.13 | TARGET_HIT | 0.50 | 1.05% |
| SELL | retest1 | 2026-04-15 09:35:00 | 112.86 | 2026-04-15 09:40:00 | 113.27 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-16 10:55:00 | 114.62 | 2026-04-16 11:25:00 | 114.37 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-04-17 10:05:00 | 114.32 | 2026-04-17 10:30:00 | 114.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-21 09:30:00 | 114.69 | 2026-04-21 09:40:00 | 114.43 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-04-22 10:50:00 | 115.08 | 2026-04-22 11:20:00 | 115.49 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-04-22 10:50:00 | 115.08 | 2026-04-22 11:45:00 | 115.08 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-28 09:55:00 | 112.68 | 2026-04-28 11:05:00 | 112.35 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-30 10:10:00 | 108.85 | 2026-04-30 10:45:00 | 108.32 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-04-30 10:10:00 | 108.85 | 2026-04-30 12:10:00 | 108.85 | STOP_HIT | 0.50 | 0.00% |
