# Bharat Heavy Electricals Ltd. (BHEL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 403.20
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
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 10
- **Target hits / Stop hits / Partials:** 2 / 10 / 3
- **Avg / median % per leg:** 0.42% / -0.26%
- **Sum % (uncompounded):** 6.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 5 | 35.7% | 2 | 9 | 3 | 0.46% | 6.5% |
| BUY @ 2nd Alert (retest1) | 14 | 5 | 35.7% | 2 | 9 | 3 | 0.46% | 6.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.24% | -0.2% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.24% | -0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 5 | 33.3% | 2 | 10 | 3 | 0.42% | 6.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:05:00 | 273.90 | 271.99 | 0.00 | ORB-long ORB[267.60,270.95] vol=5.0x ATR=1.08 |
| Stop hit — per-position SL triggered | 2026-02-09 11:20:00 | 272.82 | 272.23 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:00:00 | 258.30 | 255.51 | 0.00 | ORB-long ORB[253.00,256.35] vol=2.0x ATR=0.86 |
| Stop hit — per-position SL triggered | 2026-02-16 11:15:00 | 257.44 | 255.60 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:40:00 | 261.25 | 262.62 | 0.00 | ORB-short ORB[262.25,264.40] vol=1.5x ATR=0.63 |
| Stop hit — per-position SL triggered | 2026-02-26 11:00:00 | 261.88 | 262.51 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:35:00 | 266.25 | 265.19 | 0.00 | ORB-long ORB[262.80,265.80] vol=1.8x ATR=0.92 |
| Stop hit — per-position SL triggered | 2026-02-27 09:50:00 | 265.33 | 265.31 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:15:00 | 252.90 | 251.06 | 0.00 | ORB-long ORB[249.50,251.90] vol=2.0x ATR=1.07 |
| Stop hit — per-position SL triggered | 2026-03-05 11:35:00 | 251.83 | 251.52 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:35:00 | 259.95 | 254.93 | 0.00 | ORB-long ORB[251.85,255.10] vol=1.6x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 10:55:00 | 261.75 | 256.82 | 0.00 | T1 1.5R @ 261.75 |
| Target hit | 2026-03-12 15:20:00 | 267.65 | 265.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-03-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:40:00 | 263.95 | 262.35 | 0.00 | ORB-long ORB[259.35,263.25] vol=1.5x ATR=1.11 |
| Stop hit — per-position SL triggered | 2026-03-25 09:50:00 | 262.84 | 262.46 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-30 09:30:00 | 252.50 | 250.87 | 0.00 | ORB-long ORB[249.20,252.10] vol=2.0x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 10:00:00 | 254.20 | 252.03 | 0.00 | T1 1.5R @ 254.20 |
| Stop hit — per-position SL triggered | 2026-03-30 10:10:00 | 252.50 | 252.06 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 11:10:00 | 265.14 | 263.73 | 0.00 | ORB-long ORB[261.08,264.80] vol=2.1x ATR=1.02 |
| Stop hit — per-position SL triggered | 2026-04-08 13:25:00 | 264.12 | 264.25 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 11:15:00 | 297.04 | 295.36 | 0.00 | ORB-long ORB[293.41,296.39] vol=6.2x ATR=0.77 |
| Stop hit — per-position SL triggered | 2026-04-16 11:20:00 | 296.27 | 295.55 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 11:05:00 | 347.80 | 345.45 | 0.00 | ORB-long ORB[342.04,345.90] vol=1.6x ATR=1.30 |
| Stop hit — per-position SL triggered | 2026-04-30 11:30:00 | 346.50 | 345.68 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:30:00 | 389.45 | 386.81 | 0.00 | ORB-long ORB[383.45,387.00] vol=2.1x ATR=1.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 09:40:00 | 391.96 | 388.93 | 0.00 | T1 1.5R @ 391.96 |
| Target hit | 2026-05-07 15:20:00 | 406.80 | 400.88 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:05:00 | 273.90 | 2026-02-09 11:20:00 | 272.82 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-02-16 11:00:00 | 258.30 | 2026-02-16 11:15:00 | 257.44 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-26 10:40:00 | 261.25 | 2026-02-26 11:00:00 | 261.88 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-27 09:35:00 | 266.25 | 2026-02-27 09:50:00 | 265.33 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-05 10:15:00 | 252.90 | 2026-03-05 11:35:00 | 251.83 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-12 10:35:00 | 259.95 | 2026-03-12 10:55:00 | 261.75 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-03-12 10:35:00 | 259.95 | 2026-03-12 15:20:00 | 267.65 | TARGET_HIT | 0.50 | 2.96% |
| BUY | retest1 | 2026-03-25 09:40:00 | 263.95 | 2026-03-25 09:50:00 | 262.84 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-30 09:30:00 | 252.50 | 2026-03-30 10:00:00 | 254.20 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-03-30 09:30:00 | 252.50 | 2026-03-30 10:10:00 | 252.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-08 11:10:00 | 265.14 | 2026-04-08 13:25:00 | 264.12 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-16 11:15:00 | 297.04 | 2026-04-16 11:20:00 | 296.27 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-30 11:05:00 | 347.80 | 2026-04-30 11:30:00 | 346.50 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-05-07 09:30:00 | 389.45 | 2026-05-07 09:40:00 | 391.96 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-05-07 09:30:00 | 389.45 | 2026-05-07 15:20:00 | 406.80 | TARGET_HIT | 0.50 | 4.46% |
