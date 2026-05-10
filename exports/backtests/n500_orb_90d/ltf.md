# L&T Finance Ltd. (LTF)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 302.85
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
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 10
- **Target hits / Stop hits / Partials:** 2 / 10 / 4
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 1.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.39% | -1.5% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.39% | -1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.28% | 3.3% |
| SELL @ 2nd Alert (retest1) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.28% | 3.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 6 | 37.5% | 2 | 10 | 4 | 0.11% | 1.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:10:00 | 296.35 | 292.56 | 0.00 | ORB-long ORB[287.50,291.60] vol=1.9x ATR=1.12 |
| Stop hit — per-position SL triggered | 2026-02-09 11:30:00 | 295.23 | 292.91 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:50:00 | 290.70 | 287.70 | 0.00 | ORB-long ORB[285.15,288.50] vol=1.5x ATR=1.08 |
| Stop hit — per-position SL triggered | 2026-02-12 09:55:00 | 289.62 | 287.98 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:50:00 | 285.70 | 287.96 | 0.00 | ORB-short ORB[288.10,291.20] vol=1.7x ATR=1.09 |
| Stop hit — per-position SL triggered | 2026-02-13 10:30:00 | 286.79 | 286.65 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 11:10:00 | 291.10 | 294.49 | 0.00 | ORB-short ORB[292.75,296.05] vol=1.6x ATR=0.91 |
| Stop hit — per-position SL triggered | 2026-02-17 11:25:00 | 292.01 | 294.35 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:00:00 | 296.40 | 297.39 | 0.00 | ORB-short ORB[297.05,299.65] vol=3.0x ATR=0.99 |
| Stop hit — per-position SL triggered | 2026-02-24 10:35:00 | 297.39 | 297.21 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:10:00 | 272.10 | 273.31 | 0.00 | ORB-short ORB[272.40,275.85] vol=1.6x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:55:00 | 270.96 | 272.96 | 0.00 | T1 1.5R @ 270.96 |
| Target hit | 2026-03-11 15:20:00 | 266.10 | 270.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-03-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:40:00 | 259.50 | 261.13 | 0.00 | ORB-short ORB[261.10,264.55] vol=1.7x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:55:00 | 258.05 | 260.44 | 0.00 | T1 1.5R @ 258.05 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 259.50 | 259.30 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:55:00 | 259.90 | 259.26 | 0.00 | ORB-long ORB[256.10,259.40] vol=1.8x ATR=1.13 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 258.77 | 259.49 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 10:25:00 | 241.10 | 244.31 | 0.00 | ORB-short ORB[244.05,247.50] vol=1.7x ATR=1.31 |
| Stop hit — per-position SL triggered | 2026-03-30 11:55:00 | 242.41 | 242.87 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:00:00 | 282.91 | 284.36 | 0.00 | ORB-short ORB[283.99,286.99] vol=1.8x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:55:00 | 281.64 | 284.07 | 0.00 | T1 1.5R @ 281.64 |
| Target hit | 2026-04-16 15:20:00 | 280.58 | 281.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:15:00 | 285.52 | 282.13 | 0.00 | ORB-long ORB[277.73,281.70] vol=2.1x ATR=1.02 |
| Stop hit — per-position SL triggered | 2026-04-17 10:45:00 | 284.50 | 283.03 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:10:00 | 291.50 | 293.09 | 0.00 | ORB-short ORB[292.00,293.99] vol=1.8x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:00:00 | 290.21 | 292.22 | 0.00 | T1 1.5R @ 290.21 |
| Stop hit — per-position SL triggered | 2026-04-21 11:05:00 | 291.50 | 292.09 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:10:00 | 296.35 | 2026-02-09 11:30:00 | 295.23 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-12 09:50:00 | 290.70 | 2026-02-12 09:55:00 | 289.62 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-13 09:50:00 | 285.70 | 2026-02-13 10:30:00 | 286.79 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-02-17 11:10:00 | 291.10 | 2026-02-17 11:25:00 | 292.01 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-24 10:00:00 | 296.40 | 2026-02-24 10:35:00 | 297.39 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-11 11:10:00 | 272.10 | 2026-03-11 11:55:00 | 270.96 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-03-11 11:10:00 | 272.10 | 2026-03-11 15:20:00 | 266.10 | TARGET_HIT | 0.50 | 2.21% |
| SELL | retest1 | 2026-03-13 09:40:00 | 259.50 | 2026-03-13 09:55:00 | 258.05 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-03-13 09:40:00 | 259.50 | 2026-03-13 10:50:00 | 259.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-16 09:55:00 | 259.90 | 2026-03-16 10:15:00 | 258.77 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-03-30 10:25:00 | 241.10 | 2026-03-30 11:55:00 | 242.41 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2026-04-16 11:00:00 | 282.91 | 2026-04-16 11:55:00 | 281.64 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-04-16 11:00:00 | 282.91 | 2026-04-16 15:20:00 | 280.58 | TARGET_HIT | 0.50 | 0.82% |
| BUY | retest1 | 2026-04-17 10:15:00 | 285.52 | 2026-04-17 10:45:00 | 284.50 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-04-21 10:10:00 | 291.50 | 2026-04-21 11:00:00 | 290.21 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-04-21 10:10:00 | 291.50 | 2026-04-21 11:05:00 | 291.50 | STOP_HIT | 0.50 | 0.00% |
