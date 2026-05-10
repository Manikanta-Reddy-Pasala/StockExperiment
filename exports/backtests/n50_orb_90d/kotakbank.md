# KOTAKBANK (KOTAKBANK)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 381.00
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
| TARGET_HIT | 3 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 13
- **Target hits / Stop hits / Partials:** 3 / 13 / 6
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 1.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 7 | 53.8% | 3 | 6 | 4 | 0.15% | 2.0% |
| BUY @ 2nd Alert (retest1) | 13 | 7 | 53.8% | 3 | 6 | 4 | 0.15% | 2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 2 | 22.2% | 0 | 7 | 2 | -0.07% | -0.7% |
| SELL @ 2nd Alert (retest1) | 9 | 2 | 22.2% | 0 | 7 | 2 | -0.07% | -0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 9 | 40.9% | 3 | 13 | 6 | 0.06% | 1.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:55:00 | 432.40 | 430.60 | 0.00 | ORB-long ORB[429.10,431.50] vol=1.6x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 11:45:00 | 433.82 | 431.48 | 0.00 | T1 1.5R @ 433.82 |
| Stop hit — per-position SL triggered | 2026-02-10 12:55:00 | 432.40 | 431.96 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:50:00 | 426.75 | 428.58 | 0.00 | ORB-short ORB[427.45,431.05] vol=1.7x ATR=0.90 |
| Stop hit — per-position SL triggered | 2026-02-12 10:55:00 | 427.65 | 428.46 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:50:00 | 421.65 | 422.39 | 0.00 | ORB-short ORB[421.70,425.50] vol=2.1x ATR=0.82 |
| Stop hit — per-position SL triggered | 2026-02-17 11:15:00 | 422.47 | 422.30 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:05:00 | 418.40 | 417.32 | 0.00 | ORB-long ORB[415.00,417.75] vol=1.9x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:15:00 | 419.80 | 417.60 | 0.00 | T1 1.5R @ 419.80 |
| Target hit | 2026-02-20 15:20:00 | 421.30 | 420.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-02-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 10:45:00 | 426.45 | 426.18 | 0.00 | ORB-long ORB[420.35,426.40] vol=2.5x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:00:00 | 427.81 | 426.25 | 0.00 | T1 1.5R @ 427.81 |
| Target hit | 2026-02-23 15:20:00 | 430.70 | 429.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-03-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:50:00 | 369.05 | 370.87 | 0.00 | ORB-short ORB[370.40,373.60] vol=2.1x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:20:00 | 367.51 | 370.39 | 0.00 | T1 1.5R @ 367.51 |
| Stop hit — per-position SL triggered | 2026-03-13 11:45:00 | 369.05 | 369.96 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:55:00 | 373.55 | 371.89 | 0.00 | ORB-long ORB[368.60,372.30] vol=2.8x ATR=1.08 |
| Stop hit — per-position SL triggered | 2026-03-17 11:05:00 | 372.47 | 372.01 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 10:50:00 | 373.00 | 373.22 | 0.00 | ORB-short ORB[373.10,376.00] vol=2.6x ATR=1.01 |
| Stop hit — per-position SL triggered | 2026-03-18 11:20:00 | 374.01 | 373.14 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:05:00 | 363.90 | 364.56 | 0.00 | ORB-short ORB[364.60,368.85] vol=3.9x ATR=1.14 |
| Stop hit — per-position SL triggered | 2026-03-27 11:45:00 | 365.04 | 364.32 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 10:55:00 | 357.15 | 360.04 | 0.00 | ORB-short ORB[358.90,364.05] vol=2.6x ATR=1.24 |
| Stop hit — per-position SL triggered | 2026-04-01 11:20:00 | 358.39 | 359.73 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:05:00 | 376.50 | 375.78 | 0.00 | ORB-long ORB[372.60,375.35] vol=1.8x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 10:25:00 | 378.49 | 376.22 | 0.00 | T1 1.5R @ 378.49 |
| Target hit | 2026-04-10 12:35:00 | 377.15 | 377.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — BUY (started 2026-04-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:45:00 | 383.90 | 382.00 | 0.00 | ORB-long ORB[377.55,380.60] vol=1.7x ATR=0.95 |
| Stop hit — per-position SL triggered | 2026-04-29 11:10:00 | 382.95 | 382.36 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 10:35:00 | 380.85 | 378.48 | 0.00 | ORB-long ORB[375.30,379.15] vol=1.6x ATR=1.15 |
| Stop hit — per-position SL triggered | 2026-04-30 11:10:00 | 379.70 | 379.08 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 374.70 | 372.18 | 0.00 | ORB-long ORB[369.20,371.65] vol=6.3x ATR=1.09 |
| Stop hit — per-position SL triggered | 2026-05-05 11:40:00 | 373.61 | 372.60 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-05-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:55:00 | 372.30 | 373.14 | 0.00 | ORB-short ORB[372.60,375.10] vol=1.5x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:00:00 | 371.35 | 373.06 | 0.00 | T1 1.5R @ 371.35 |
| Stop hit — per-position SL triggered | 2026-05-06 11:05:00 | 372.30 | 372.93 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 11:10:00 | 379.40 | 377.39 | 0.00 | ORB-long ORB[374.30,377.60] vol=2.1x ATR=1.09 |
| Stop hit — per-position SL triggered | 2026-05-07 11:40:00 | 378.31 | 377.61 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:55:00 | 432.40 | 2026-02-10 11:45:00 | 433.82 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-02-10 10:55:00 | 432.40 | 2026-02-10 12:55:00 | 432.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-12 10:50:00 | 426.75 | 2026-02-12 10:55:00 | 427.65 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-17 10:50:00 | 421.65 | 2026-02-17 11:15:00 | 422.47 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-02-20 10:05:00 | 418.40 | 2026-02-20 10:15:00 | 419.80 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-02-20 10:05:00 | 418.40 | 2026-02-20 15:20:00 | 421.30 | TARGET_HIT | 0.50 | 0.69% |
| BUY | retest1 | 2026-02-23 10:45:00 | 426.45 | 2026-02-23 11:00:00 | 427.81 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-23 10:45:00 | 426.45 | 2026-02-23 15:20:00 | 430.70 | TARGET_HIT | 0.50 | 1.00% |
| SELL | retest1 | 2026-03-13 10:50:00 | 369.05 | 2026-03-13 11:20:00 | 367.51 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-03-13 10:50:00 | 369.05 | 2026-03-13 11:45:00 | 369.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:55:00 | 373.55 | 2026-03-17 11:05:00 | 372.47 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-18 10:50:00 | 373.00 | 2026-03-18 11:20:00 | 374.01 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-03-27 11:05:00 | 363.90 | 2026-03-27 11:45:00 | 365.04 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-01 10:55:00 | 357.15 | 2026-04-01 11:20:00 | 358.39 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-10 10:05:00 | 376.50 | 2026-04-10 10:25:00 | 378.49 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-10 10:05:00 | 376.50 | 2026-04-10 12:35:00 | 377.15 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2026-04-29 10:45:00 | 383.90 | 2026-04-29 11:10:00 | 382.95 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-30 10:35:00 | 380.85 | 2026-04-30 11:10:00 | 379.70 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-05 11:00:00 | 374.70 | 2026-05-05 11:40:00 | 373.61 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-05-06 10:55:00 | 372.30 | 2026-05-06 11:00:00 | 371.35 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-05-06 10:55:00 | 372.30 | 2026-05-06 11:05:00 | 372.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-07 11:10:00 | 379.40 | 2026-05-07 11:40:00 | 378.31 | STOP_HIT | 1.00 | -0.29% |
