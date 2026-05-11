# Karur Vysya Bank Ltd. (KARURVYSYA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2025-07-09 15:25:00 (3225 bars)
- **Last close:** 226.71
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 5 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 13
- **Target hits / Stop hits / Partials:** 5 / 13 / 7
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 4.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.13% | 1.9% |
| BUY @ 2nd Alert (retest1) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.13% | 1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 6 | 54.5% | 3 | 5 | 3 | 0.23% | 2.5% |
| SELL @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 3 | 5 | 3 | 0.23% | 2.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 12 | 48.0% | 5 | 13 | 7 | 0.18% | 4.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-12 10:50:00 | 176.67 | 177.92 | 0.00 | ORB-short ORB[177.35,179.93] vol=2.1x ATR=1.11 |
| Target hit | 2025-05-12 15:20:00 | 175.83 | 176.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2025-05-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-15 11:10:00 | 183.18 | 183.60 | 0.00 | ORB-short ORB[183.54,185.34] vol=3.0x ATR=0.41 |
| Stop hit — per-position SL triggered | 2025-05-15 12:05:00 | 183.59 | 183.57 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 10:55:00 | 186.83 | 185.76 | 0.00 | ORB-long ORB[184.76,186.58] vol=2.7x ATR=0.53 |
| Stop hit — per-position SL triggered | 2025-05-16 12:30:00 | 186.30 | 186.25 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 10:45:00 | 186.50 | 187.19 | 0.00 | ORB-short ORB[186.67,188.58] vol=2.1x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-22 11:10:00 | 185.78 | 187.06 | 0.00 | T1 1.5R @ 185.78 |
| Target hit | 2025-05-22 15:20:00 | 184.66 | 185.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2025-05-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 10:30:00 | 187.04 | 185.70 | 0.00 | ORB-long ORB[183.79,185.65] vol=3.1x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 11:10:00 | 187.85 | 186.35 | 0.00 | T1 1.5R @ 187.85 |
| Target hit | 2025-05-23 15:20:00 | 188.58 | 187.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2025-05-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 09:30:00 | 187.72 | 188.32 | 0.00 | ORB-short ORB[187.76,190.00] vol=1.9x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 09:45:00 | 187.00 | 188.07 | 0.00 | T1 1.5R @ 187.00 |
| Target hit | 2025-05-28 12:40:00 | 186.08 | 186.00 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — SELL (started 2025-05-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 11:00:00 | 184.38 | 185.20 | 0.00 | ORB-short ORB[185.16,186.54] vol=2.3x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-05-29 11:15:00 | 184.74 | 185.18 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 09:55:00 | 192.54 | 193.39 | 0.00 | ORB-short ORB[192.92,194.25] vol=3.2x ATR=0.50 |
| Stop hit — per-position SL triggered | 2025-06-06 10:05:00 | 193.04 | 193.17 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 10:55:00 | 201.30 | 198.26 | 0.00 | ORB-long ORB[196.54,198.75] vol=3.6x ATR=0.59 |
| Stop hit — per-position SL triggered | 2025-06-11 11:00:00 | 200.71 | 198.36 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 10:45:00 | 203.00 | 201.66 | 0.00 | ORB-long ORB[200.05,202.72] vol=4.8x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 10:50:00 | 203.93 | 202.04 | 0.00 | T1 1.5R @ 203.93 |
| Stop hit — per-position SL triggered | 2025-06-12 10:55:00 | 203.00 | 202.19 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 10:15:00 | 207.41 | 204.88 | 0.00 | ORB-long ORB[203.33,205.60] vol=1.9x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-06-23 10:20:00 | 206.62 | 205.05 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:35:00 | 208.03 | 206.53 | 0.00 | ORB-long ORB[204.67,207.66] vol=4.2x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-06-24 10:40:00 | 207.34 | 206.55 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-06-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-25 11:00:00 | 208.32 | 209.29 | 0.00 | ORB-short ORB[208.38,210.33] vol=3.0x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 11:45:00 | 207.62 | 209.17 | 0.00 | T1 1.5R @ 207.62 |
| Stop hit — per-position SL triggered | 2025-06-25 12:20:00 | 208.32 | 208.69 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-06-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 10:55:00 | 205.75 | 206.41 | 0.00 | ORB-short ORB[206.00,208.34] vol=4.1x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-06-26 11:25:00 | 206.21 | 206.34 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-06-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 10:30:00 | 210.17 | 209.29 | 0.00 | ORB-long ORB[207.17,209.58] vol=7.3x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-06-27 10:40:00 | 209.55 | 209.32 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 11:15:00 | 226.33 | 224.17 | 0.00 | ORB-long ORB[223.00,226.25] vol=2.7x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 11:20:00 | 227.57 | 224.85 | 0.00 | T1 1.5R @ 227.57 |
| Target hit | 2025-07-01 15:20:00 | 228.63 | 227.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2025-07-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 10:25:00 | 226.88 | 225.77 | 0.00 | ORB-long ORB[223.33,226.58] vol=2.8x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 10:30:00 | 228.00 | 225.91 | 0.00 | T1 1.5R @ 228.00 |
| Stop hit — per-position SL triggered | 2025-07-08 10:45:00 | 226.88 | 226.06 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 10:30:00 | 227.67 | 226.68 | 0.00 | ORB-long ORB[224.17,227.21] vol=1.7x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-07-09 10:55:00 | 226.98 | 227.04 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-12 10:50:00 | 176.67 | 2025-05-12 15:20:00 | 175.83 | TARGET_HIT | 1.00 | 0.48% |
| SELL | retest1 | 2025-05-15 11:10:00 | 183.18 | 2025-05-15 12:05:00 | 183.59 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-05-16 10:55:00 | 186.83 | 2025-05-16 12:30:00 | 186.30 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-05-22 10:45:00 | 186.50 | 2025-05-22 11:10:00 | 185.78 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-05-22 10:45:00 | 186.50 | 2025-05-22 15:20:00 | 184.66 | TARGET_HIT | 0.50 | 0.99% |
| BUY | retest1 | 2025-05-23 10:30:00 | 187.04 | 2025-05-23 11:10:00 | 187.85 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-05-23 10:30:00 | 187.04 | 2025-05-23 15:20:00 | 188.58 | TARGET_HIT | 0.50 | 0.82% |
| SELL | retest1 | 2025-05-28 09:30:00 | 187.72 | 2025-05-28 09:45:00 | 187.00 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-05-28 09:30:00 | 187.72 | 2025-05-28 12:40:00 | 186.08 | TARGET_HIT | 0.50 | 0.87% |
| SELL | retest1 | 2025-05-29 11:00:00 | 184.38 | 2025-05-29 11:15:00 | 184.74 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-06-06 09:55:00 | 192.54 | 2025-06-06 10:05:00 | 193.04 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-06-11 10:55:00 | 201.30 | 2025-06-11 11:00:00 | 200.71 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-06-12 10:45:00 | 203.00 | 2025-06-12 10:50:00 | 203.93 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-06-12 10:45:00 | 203.00 | 2025-06-12 10:55:00 | 203.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-23 10:15:00 | 207.41 | 2025-06-23 10:20:00 | 206.62 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-06-24 10:35:00 | 208.03 | 2025-06-24 10:40:00 | 207.34 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-06-25 11:00:00 | 208.32 | 2025-06-25 11:45:00 | 207.62 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-06-25 11:00:00 | 208.32 | 2025-06-25 12:20:00 | 208.32 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-26 10:55:00 | 205.75 | 2025-06-26 11:25:00 | 206.21 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-06-27 10:30:00 | 210.17 | 2025-06-27 10:40:00 | 209.55 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-07-01 11:15:00 | 226.33 | 2025-07-01 11:20:00 | 227.57 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-07-01 11:15:00 | 226.33 | 2025-07-01 15:20:00 | 228.63 | TARGET_HIT | 0.50 | 1.02% |
| BUY | retest1 | 2025-07-08 10:25:00 | 226.88 | 2025-07-08 10:30:00 | 228.00 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-07-08 10:25:00 | 226.88 | 2025-07-08 10:45:00 | 226.88 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-09 10:30:00 | 227.67 | 2025-07-09 10:55:00 | 226.98 | STOP_HIT | 1.00 | -0.30% |
