# ONGC (ONGC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 279.00
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
| ENTRY1 | 11 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 8
- **Target hits / Stop hits / Partials:** 3 / 8 / 6
- **Avg / median % per leg:** 0.30% / 0.35%
- **Sum % (uncompounded):** 5.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 3 | 37.5% | 0 | 5 | 3 | 0.11% | 0.8% |
| BUY @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 0 | 5 | 3 | 0.11% | 0.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 6 | 66.7% | 3 | 3 | 3 | 0.47% | 4.3% |
| SELL @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 3 | 3 | 3 | 0.47% | 4.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 9 | 52.9% | 3 | 8 | 6 | 0.30% | 5.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:45:00 | 268.75 | 270.04 | 0.00 | ORB-short ORB[269.50,273.00] vol=1.7x ATR=0.60 |
| Stop hit — per-position SL triggered | 2026-02-17 11:15:00 | 269.35 | 269.70 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 11:10:00 | 278.50 | 277.31 | 0.00 | ORB-long ORB[276.10,278.40] vol=2.8x ATR=0.62 |
| Stop hit — per-position SL triggered | 2026-02-26 11:25:00 | 277.88 | 277.39 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 10:45:00 | 280.10 | 278.33 | 0.00 | ORB-long ORB[277.20,279.55] vol=2.1x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:05:00 | 281.15 | 279.24 | 0.00 | T1 1.5R @ 281.15 |
| Stop hit — per-position SL triggered | 2026-02-27 11:50:00 | 280.10 | 280.14 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:25:00 | 267.00 | 268.47 | 0.00 | ORB-short ORB[269.00,271.30] vol=2.1x ATR=0.75 |
| Stop hit — per-position SL triggered | 2026-03-13 10:30:00 | 267.75 | 268.42 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:15:00 | 262.05 | 260.90 | 0.00 | ORB-long ORB[259.15,261.80] vol=3.0x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:20:00 | 263.25 | 261.01 | 0.00 | T1 1.5R @ 263.25 |
| Stop hit — per-position SL triggered | 2026-03-17 10:25:00 | 262.05 | 261.04 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 11:05:00 | 270.65 | 268.83 | 0.00 | ORB-long ORB[266.40,268.80] vol=1.5x ATR=0.70 |
| Stop hit — per-position SL triggered | 2026-03-25 11:25:00 | 269.95 | 268.98 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:35:00 | 284.85 | 286.30 | 0.00 | ORB-short ORB[285.35,288.10] vol=2.0x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:10:00 | 283.84 | 285.48 | 0.00 | T1 1.5R @ 283.84 |
| Target hit | 2026-04-16 15:20:00 | 282.95 | 284.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-05-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:10:00 | 288.60 | 290.74 | 0.00 | ORB-short ORB[291.70,295.90] vol=2.4x ATR=0.81 |
| Stop hit — per-position SL triggered | 2026-05-05 11:25:00 | 289.41 | 290.65 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-05-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:30:00 | 287.15 | 288.31 | 0.00 | ORB-short ORB[287.80,290.25] vol=1.6x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:05:00 | 285.93 | 287.62 | 0.00 | T1 1.5R @ 285.93 |
| Target hit | 2026-05-06 15:20:00 | 280.30 | 283.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:15:00 | 283.65 | 281.95 | 0.00 | ORB-long ORB[279.70,282.75] vol=1.6x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 10:35:00 | 285.06 | 282.42 | 0.00 | T1 1.5R @ 285.06 |
| Stop hit — per-position SL triggered | 2026-05-07 12:40:00 | 283.65 | 283.56 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:50:00 | 281.60 | 281.84 | 0.00 | ORB-short ORB[281.80,284.00] vol=1.8x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:25:00 | 280.62 | 281.74 | 0.00 | T1 1.5R @ 280.62 |
| Target hit | 2026-05-08 15:20:00 | 279.15 | 280.35 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-17 10:45:00 | 268.75 | 2026-02-17 11:15:00 | 269.35 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-26 11:10:00 | 278.50 | 2026-02-26 11:25:00 | 277.88 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-27 10:45:00 | 280.10 | 2026-02-27 11:05:00 | 281.15 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-02-27 10:45:00 | 280.10 | 2026-02-27 11:50:00 | 280.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 10:25:00 | 267.00 | 2026-03-13 10:30:00 | 267.75 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-03-17 10:15:00 | 262.05 | 2026-03-17 10:20:00 | 263.25 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-03-17 10:15:00 | 262.05 | 2026-03-17 10:25:00 | 262.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-25 11:05:00 | 270.65 | 2026-03-25 11:25:00 | 269.95 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-16 10:35:00 | 284.85 | 2026-04-16 11:10:00 | 283.84 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-04-16 10:35:00 | 284.85 | 2026-04-16 15:20:00 | 282.95 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2026-05-05 11:10:00 | 288.60 | 2026-05-05 11:25:00 | 289.41 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-05-06 10:30:00 | 287.15 | 2026-05-06 11:05:00 | 285.93 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-05-06 10:30:00 | 287.15 | 2026-05-06 15:20:00 | 280.30 | TARGET_HIT | 0.50 | 2.39% |
| BUY | retest1 | 2026-05-07 10:15:00 | 283.65 | 2026-05-07 10:35:00 | 285.06 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-05-07 10:15:00 | 283.65 | 2026-05-07 12:40:00 | 283.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 10:50:00 | 281.60 | 2026-05-08 11:25:00 | 280.62 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-05-08 10:50:00 | 281.60 | 2026-05-08 15:20:00 | 279.15 | TARGET_HIT | 0.50 | 0.87% |
