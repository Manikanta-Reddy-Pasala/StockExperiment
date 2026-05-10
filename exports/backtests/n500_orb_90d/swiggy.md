# Swiggy Ltd. (SWIGGY)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 282.80
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
| ENTRY1 | 13 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 11
- **Target hits / Stop hits / Partials:** 2 / 11 / 4
- **Avg / median % per leg:** 0.09% / -0.27%
- **Sum % (uncompounded):** 1.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 3 | 30.0% | 1 | 7 | 2 | 0.12% | 1.2% |
| BUY @ 2nd Alert (retest1) | 10 | 3 | 30.0% | 1 | 7 | 2 | 0.12% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.04% | 0.3% |
| SELL @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.04% | 0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 6 | 35.3% | 2 | 11 | 4 | 0.09% | 1.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:40:00 | 342.20 | 338.50 | 0.00 | ORB-long ORB[333.65,337.65] vol=4.5x ATR=1.63 |
| Stop hit — per-position SL triggered | 2026-02-16 10:55:00 | 340.57 | 339.29 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:40:00 | 321.05 | 322.61 | 0.00 | ORB-short ORB[321.35,324.90] vol=1.6x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:35:00 | 319.31 | 321.46 | 0.00 | T1 1.5R @ 319.31 |
| Target hit | 2026-02-23 14:15:00 | 320.45 | 320.44 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2026-02-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:35:00 | 311.80 | 314.30 | 0.00 | ORB-short ORB[314.20,317.60] vol=1.6x ATR=1.12 |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 312.92 | 313.24 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:15:00 | 305.20 | 306.17 | 0.00 | ORB-short ORB[305.50,308.00] vol=1.5x ATR=0.83 |
| Stop hit — per-position SL triggered | 2026-02-27 10:20:00 | 306.03 | 306.13 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:15:00 | 298.60 | 296.12 | 0.00 | ORB-long ORB[292.80,297.20] vol=1.6x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:55:00 | 300.57 | 297.13 | 0.00 | T1 1.5R @ 300.57 |
| Stop hit — per-position SL triggered | 2026-03-18 11:45:00 | 298.60 | 297.64 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:40:00 | 290.25 | 286.46 | 0.00 | ORB-long ORB[284.05,287.30] vol=2.3x ATR=1.43 |
| Stop hit — per-position SL triggered | 2026-03-20 10:55:00 | 288.82 | 287.12 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 11:00:00 | 278.35 | 276.10 | 0.00 | ORB-long ORB[273.40,277.40] vol=1.8x ATR=0.90 |
| Stop hit — per-position SL triggered | 2026-04-10 11:05:00 | 277.45 | 276.15 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:10:00 | 273.35 | 271.77 | 0.00 | ORB-long ORB[268.80,272.70] vol=3.1x ATR=1.34 |
| Stop hit — per-position SL triggered | 2026-04-15 10:40:00 | 272.01 | 271.98 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 09:35:00 | 279.40 | 281.28 | 0.00 | ORB-short ORB[280.00,283.35] vol=2.4x ATR=1.20 |
| Stop hit — per-position SL triggered | 2026-04-17 09:45:00 | 280.60 | 281.17 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 280.40 | 279.52 | 0.00 | ORB-long ORB[277.45,280.25] vol=2.5x ATR=0.85 |
| Stop hit — per-position SL triggered | 2026-04-21 09:40:00 | 279.55 | 279.76 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 11:15:00 | 287.40 | 284.68 | 0.00 | ORB-long ORB[282.45,286.25] vol=6.7x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:30:00 | 288.79 | 285.76 | 0.00 | T1 1.5R @ 288.79 |
| Target hit | 2026-04-22 15:20:00 | 294.85 | 290.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2026-04-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:50:00 | 282.00 | 284.28 | 0.00 | ORB-short ORB[284.00,287.55] vol=2.0x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:30:00 | 280.30 | 282.97 | 0.00 | T1 1.5R @ 280.30 |
| Stop hit — per-position SL triggered | 2026-04-28 11:05:00 | 282.00 | 282.20 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-05-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:45:00 | 276.40 | 273.49 | 0.00 | ORB-long ORB[270.70,273.75] vol=4.2x ATR=1.21 |
| Stop hit — per-position SL triggered | 2026-05-04 12:10:00 | 275.19 | 274.82 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-16 10:40:00 | 342.20 | 2026-02-16 10:55:00 | 340.57 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-02-23 09:40:00 | 321.05 | 2026-02-23 10:35:00 | 319.31 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-02-23 09:40:00 | 321.05 | 2026-02-23 14:15:00 | 320.45 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2026-02-25 09:35:00 | 311.80 | 2026-02-25 10:15:00 | 312.92 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-27 10:15:00 | 305.20 | 2026-02-27 10:20:00 | 306.03 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-18 10:15:00 | 298.60 | 2026-03-18 10:55:00 | 300.57 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-03-18 10:15:00 | 298.60 | 2026-03-18 11:45:00 | 298.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-20 10:40:00 | 290.25 | 2026-03-20 10:55:00 | 288.82 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-04-10 11:00:00 | 278.35 | 2026-04-10 11:05:00 | 277.45 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-15 10:10:00 | 273.35 | 2026-04-15 10:40:00 | 272.01 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-04-17 09:35:00 | 279.40 | 2026-04-17 09:45:00 | 280.60 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-21 09:35:00 | 280.40 | 2026-04-21 09:40:00 | 279.55 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-22 11:15:00 | 287.40 | 2026-04-22 11:30:00 | 288.79 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-04-22 11:15:00 | 287.40 | 2026-04-22 15:20:00 | 294.85 | TARGET_HIT | 0.50 | 2.59% |
| SELL | retest1 | 2026-04-28 09:50:00 | 282.00 | 2026-04-28 10:30:00 | 280.30 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-04-28 09:50:00 | 282.00 | 2026-04-28 11:05:00 | 282.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 10:45:00 | 276.40 | 2026-05-04 12:10:00 | 275.19 | STOP_HIT | 1.00 | -0.44% |
