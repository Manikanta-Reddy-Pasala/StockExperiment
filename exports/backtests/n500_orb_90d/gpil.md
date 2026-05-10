# Godawari Power & Ispat Ltd. (GPIL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 295.00
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
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 13
- **Target hits / Stop hits / Partials:** 3 / 13 / 5
- **Avg / median % per leg:** 0.08% / -0.30%
- **Sum % (uncompounded):** 1.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 5 | 35.7% | 2 | 9 | 3 | -0.04% | -0.5% |
| BUY @ 2nd Alert (retest1) | 14 | 5 | 35.7% | 2 | 9 | 3 | -0.04% | -0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.31% | 2.2% |
| SELL @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.31% | 2.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 8 | 38.1% | 3 | 13 | 5 | 0.08% | 1.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 260.63 | 259.60 | 0.00 | ORB-long ORB[257.43,260.40] vol=3.9x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:45:00 | 262.68 | 260.79 | 0.00 | T1 1.5R @ 262.68 |
| Target hit | 2026-02-10 10:05:00 | 262.00 | 262.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 257.51 | 259.12 | 0.00 | ORB-short ORB[259.00,261.36] vol=2.5x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:40:00 | 256.02 | 258.38 | 0.00 | T1 1.5R @ 256.02 |
| Stop hit — per-position SL triggered | 2026-02-11 09:45:00 | 257.51 | 258.42 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:30:00 | 261.65 | 258.81 | 0.00 | ORB-long ORB[256.00,259.40] vol=2.6x ATR=1.20 |
| Stop hit — per-position SL triggered | 2026-02-12 09:40:00 | 260.45 | 259.57 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 275.21 | 274.14 | 0.00 | ORB-long ORB[272.07,274.80] vol=1.8x ATR=0.84 |
| Stop hit — per-position SL triggered | 2026-02-18 09:45:00 | 274.37 | 274.21 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:55:00 | 268.03 | 269.21 | 0.00 | ORB-short ORB[268.66,272.51] vol=3.8x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:10:00 | 266.50 | 268.85 | 0.00 | T1 1.5R @ 266.50 |
| Target hit | 2026-02-19 15:20:00 | 261.25 | 265.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:15:00 | 263.51 | 261.97 | 0.00 | ORB-long ORB[259.10,261.89] vol=1.7x ATR=0.93 |
| Stop hit — per-position SL triggered | 2026-02-20 15:05:00 | 262.58 | 262.78 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:05:00 | 275.21 | 273.43 | 0.00 | ORB-long ORB[270.63,273.90] vol=3.9x ATR=1.08 |
| Stop hit — per-position SL triggered | 2026-02-26 10:20:00 | 274.13 | 273.71 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-02-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:35:00 | 274.34 | 273.15 | 0.00 | ORB-long ORB[270.11,273.89] vol=2.0x ATR=1.00 |
| Stop hit — per-position SL triggered | 2026-02-27 09:40:00 | 273.34 | 273.21 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:35:00 | 246.50 | 248.28 | 0.00 | ORB-short ORB[248.00,250.50] vol=1.7x ATR=1.25 |
| Stop hit — per-position SL triggered | 2026-03-05 09:45:00 | 247.75 | 248.04 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:15:00 | 254.40 | 251.86 | 0.00 | ORB-long ORB[250.00,252.25] vol=3.0x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:20:00 | 256.10 | 252.77 | 0.00 | T1 1.5R @ 256.10 |
| Target hit | 2026-03-17 12:20:00 | 254.55 | 254.73 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 262.35 | 261.11 | 0.00 | ORB-long ORB[258.55,262.00] vol=1.5x ATR=1.08 |
| Stop hit — per-position SL triggered | 2026-03-18 10:05:00 | 261.27 | 261.58 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-02 09:45:00 | 269.85 | 272.24 | 0.00 | ORB-short ORB[271.20,275.00] vol=1.5x ATR=1.61 |
| Stop hit — per-position SL triggered | 2026-04-02 11:35:00 | 271.46 | 271.00 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:45:00 | 290.25 | 287.14 | 0.00 | ORB-long ORB[283.00,287.20] vol=1.6x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 11:05:00 | 292.22 | 288.44 | 0.00 | T1 1.5R @ 292.22 |
| Stop hit — per-position SL triggered | 2026-04-13 11:35:00 | 290.25 | 289.35 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 298.80 | 295.69 | 0.00 | ORB-long ORB[292.20,295.95] vol=5.1x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-04-15 09:45:00 | 297.24 | 296.03 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 310.40 | 308.20 | 0.00 | ORB-long ORB[304.95,309.00] vol=3.0x ATR=1.28 |
| Stop hit — per-position SL triggered | 2026-04-16 09:35:00 | 309.12 | 308.23 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:05:00 | 298.30 | 300.53 | 0.00 | ORB-short ORB[300.20,304.00] vol=2.2x ATR=1.25 |
| Stop hit — per-position SL triggered | 2026-04-17 10:15:00 | 299.55 | 300.41 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:40:00 | 260.63 | 2026-02-10 09:45:00 | 262.68 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2026-02-10 09:40:00 | 260.63 | 2026-02-10 10:05:00 | 262.00 | TARGET_HIT | 0.50 | 0.53% |
| SELL | retest1 | 2026-02-11 09:30:00 | 257.51 | 2026-02-11 09:40:00 | 256.02 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-02-11 09:30:00 | 257.51 | 2026-02-11 09:45:00 | 257.51 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-12 09:30:00 | 261.65 | 2026-02-12 09:40:00 | 260.45 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-02-18 09:35:00 | 275.21 | 2026-02-18 09:45:00 | 274.37 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-19 09:55:00 | 268.03 | 2026-02-19 10:10:00 | 266.50 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-02-19 09:55:00 | 268.03 | 2026-02-19 15:20:00 | 261.25 | TARGET_HIT | 0.50 | 2.53% |
| BUY | retest1 | 2026-02-20 11:15:00 | 263.51 | 2026-02-20 15:05:00 | 262.58 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-26 10:05:00 | 275.21 | 2026-02-26 10:20:00 | 274.13 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-02-27 09:35:00 | 274.34 | 2026-02-27 09:40:00 | 273.34 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-03-05 09:35:00 | 246.50 | 2026-03-05 09:45:00 | 247.75 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2026-03-17 10:15:00 | 254.40 | 2026-03-17 10:20:00 | 256.10 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-03-17 10:15:00 | 254.40 | 2026-03-17 12:20:00 | 254.55 | TARGET_HIT | 0.50 | 0.06% |
| BUY | retest1 | 2026-03-18 09:30:00 | 262.35 | 2026-03-18 10:05:00 | 261.27 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-02 09:45:00 | 269.85 | 2026-04-02 11:35:00 | 271.46 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2026-04-13 10:45:00 | 290.25 | 2026-04-13 11:05:00 | 292.22 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-04-13 10:45:00 | 290.25 | 2026-04-13 11:35:00 | 290.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-15 09:40:00 | 298.80 | 2026-04-15 09:45:00 | 297.24 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-04-16 09:30:00 | 310.40 | 2026-04-16 09:35:00 | 309.12 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-17 10:05:00 | 298.30 | 2026-04-17 10:15:00 | 299.55 | STOP_HIT | 1.00 | -0.42% |
