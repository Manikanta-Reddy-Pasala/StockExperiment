# Star Health and Allied Insurance Company Ltd. (STARHEALTH)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 519.05
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
| PARTIAL | 9 |
| TARGET_HIT | 5 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 13
- **Target hits / Stop hits / Partials:** 5 / 13 / 9
- **Avg / median % per leg:** 0.27% / 0.38%
- **Sum % (uncompounded):** 7.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 11 | 52.4% | 4 | 10 | 7 | 0.27% | 5.7% |
| BUY @ 2nd Alert (retest1) | 21 | 11 | 52.4% | 4 | 10 | 7 | 0.27% | 5.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.25% | 1.5% |
| SELL @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.25% | 1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 27 | 14 | 51.9% | 5 | 13 | 9 | 0.27% | 7.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:35:00 | 467.10 | 464.14 | 0.00 | ORB-long ORB[460.65,465.75] vol=5.6x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:40:00 | 469.15 | 465.61 | 0.00 | T1 1.5R @ 469.15 |
| Stop hit — per-position SL triggered | 2026-02-10 13:30:00 | 467.10 | 467.74 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:15:00 | 471.25 | 469.61 | 0.00 | ORB-long ORB[467.10,471.05] vol=2.0x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 12:10:00 | 473.25 | 470.74 | 0.00 | T1 1.5R @ 473.25 |
| Target hit | 2026-02-11 13:50:00 | 480.95 | 480.97 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2026-02-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 09:40:00 | 481.50 | 478.54 | 0.00 | ORB-long ORB[475.00,479.75] vol=1.7x ATR=1.79 |
| Stop hit — per-position SL triggered | 2026-02-13 09:45:00 | 479.71 | 478.60 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:45:00 | 469.45 | 468.48 | 0.00 | ORB-long ORB[462.20,469.10] vol=1.7x ATR=1.16 |
| Stop hit — per-position SL triggered | 2026-02-16 11:45:00 | 468.29 | 468.52 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:45:00 | 479.15 | 480.99 | 0.00 | ORB-short ORB[479.50,485.60] vol=1.6x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 14:00:00 | 476.95 | 480.07 | 0.00 | T1 1.5R @ 476.95 |
| Target hit | 2026-02-18 15:20:00 | 472.45 | 478.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-02-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 11:00:00 | 453.15 | 455.60 | 0.00 | ORB-short ORB[454.10,459.50] vol=6.6x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:20:00 | 451.38 | 455.13 | 0.00 | T1 1.5R @ 451.38 |
| Stop hit — per-position SL triggered | 2026-02-24 11:25:00 | 453.15 | 455.12 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:50:00 | 454.55 | 452.58 | 0.00 | ORB-long ORB[446.15,451.80] vol=3.8x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 13:05:00 | 456.62 | 453.82 | 0.00 | T1 1.5R @ 456.62 |
| Stop hit — per-position SL triggered | 2026-03-10 13:50:00 | 454.55 | 454.11 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:55:00 | 455.60 | 451.74 | 0.00 | ORB-long ORB[447.05,452.20] vol=4.2x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:00:00 | 457.35 | 452.55 | 0.00 | T1 1.5R @ 457.35 |
| Stop hit — per-position SL triggered | 2026-03-11 11:45:00 | 455.60 | 456.09 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 09:40:00 | 461.50 | 459.83 | 0.00 | ORB-long ORB[456.60,461.20] vol=1.5x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 09:45:00 | 463.46 | 460.34 | 0.00 | T1 1.5R @ 463.46 |
| Target hit | 2026-03-19 10:25:00 | 463.30 | 463.33 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2026-03-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 10:30:00 | 460.35 | 456.00 | 0.00 | ORB-long ORB[451.80,457.10] vol=2.3x ATR=1.82 |
| Stop hit — per-position SL triggered | 2026-03-27 10:40:00 | 458.53 | 457.00 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 10:45:00 | 451.35 | 453.18 | 0.00 | ORB-short ORB[452.10,458.05] vol=1.5x ATR=2.04 |
| Stop hit — per-position SL triggered | 2026-04-01 11:45:00 | 453.39 | 452.65 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:55:00 | 471.60 | 469.21 | 0.00 | ORB-long ORB[465.55,469.60] vol=4.6x ATR=1.89 |
| Stop hit — per-position SL triggered | 2026-04-08 10:40:00 | 469.71 | 470.72 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:45:00 | 466.25 | 470.50 | 0.00 | ORB-short ORB[468.00,475.00] vol=1.9x ATR=1.29 |
| Stop hit — per-position SL triggered | 2026-04-09 10:00:00 | 467.54 | 468.90 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 481.70 | 479.26 | 0.00 | ORB-long ORB[475.00,480.00] vol=1.8x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 09:40:00 | 484.53 | 482.48 | 0.00 | T1 1.5R @ 484.53 |
| Target hit | 2026-04-15 13:05:00 | 484.40 | 486.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — BUY (started 2026-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:55:00 | 487.85 | 485.48 | 0.00 | ORB-long ORB[480.05,486.35] vol=3.4x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:10:00 | 490.61 | 486.66 | 0.00 | T1 1.5R @ 490.61 |
| Target hit | 2026-04-16 15:20:00 | 496.50 | 492.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2026-04-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:50:00 | 517.00 | 512.99 | 0.00 | ORB-long ORB[505.55,511.70] vol=1.6x ATR=1.58 |
| Stop hit — per-position SL triggered | 2026-04-23 11:05:00 | 515.42 | 514.03 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:55:00 | 519.25 | 517.17 | 0.00 | ORB-long ORB[513.90,517.95] vol=1.6x ATR=1.51 |
| Stop hit — per-position SL triggered | 2026-04-28 11:25:00 | 517.74 | 518.22 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 523.90 | 521.94 | 0.00 | ORB-long ORB[519.00,523.20] vol=1.8x ATR=1.74 |
| Stop hit — per-position SL triggered | 2026-05-06 09:45:00 | 522.16 | 521.95 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:35:00 | 467.10 | 2026-02-10 10:40:00 | 469.15 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-02-10 10:35:00 | 467.10 | 2026-02-10 13:30:00 | 467.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-11 10:15:00 | 471.25 | 2026-02-11 12:10:00 | 473.25 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-02-11 10:15:00 | 471.25 | 2026-02-11 13:50:00 | 480.95 | TARGET_HIT | 0.50 | 2.06% |
| BUY | retest1 | 2026-02-13 09:40:00 | 481.50 | 2026-02-13 09:45:00 | 479.71 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-16 10:45:00 | 469.45 | 2026-02-16 11:45:00 | 468.29 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-18 10:45:00 | 479.15 | 2026-02-18 14:00:00 | 476.95 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-18 10:45:00 | 479.15 | 2026-02-18 15:20:00 | 472.45 | TARGET_HIT | 0.50 | 1.40% |
| SELL | retest1 | 2026-02-24 11:00:00 | 453.15 | 2026-02-24 11:20:00 | 451.38 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-24 11:00:00 | 453.15 | 2026-02-24 11:25:00 | 453.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-10 10:50:00 | 454.55 | 2026-03-10 13:05:00 | 456.62 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-03-10 10:50:00 | 454.55 | 2026-03-10 13:50:00 | 454.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-11 10:55:00 | 455.60 | 2026-03-11 11:00:00 | 457.35 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-03-11 10:55:00 | 455.60 | 2026-03-11 11:45:00 | 455.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-19 09:40:00 | 461.50 | 2026-03-19 09:45:00 | 463.46 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-03-19 09:40:00 | 461.50 | 2026-03-19 10:25:00 | 463.30 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2026-03-27 10:30:00 | 460.35 | 2026-03-27 10:40:00 | 458.53 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-01 10:45:00 | 451.35 | 2026-04-01 11:45:00 | 453.39 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-08 09:55:00 | 471.60 | 2026-04-08 10:40:00 | 469.71 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-09 09:45:00 | 466.25 | 2026-04-09 10:00:00 | 467.54 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-15 09:35:00 | 481.70 | 2026-04-15 09:40:00 | 484.53 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-04-15 09:35:00 | 481.70 | 2026-04-15 13:05:00 | 484.40 | TARGET_HIT | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-16 09:55:00 | 487.85 | 2026-04-16 10:10:00 | 490.61 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-04-16 09:55:00 | 487.85 | 2026-04-16 15:20:00 | 496.50 | TARGET_HIT | 0.50 | 1.77% |
| BUY | retest1 | 2026-04-23 10:50:00 | 517.00 | 2026-04-23 11:05:00 | 515.42 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-28 09:55:00 | 519.25 | 2026-04-28 11:25:00 | 517.74 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-05-06 09:40:00 | 523.90 | 2026-05-06 09:45:00 | 522.16 | STOP_HIT | 1.00 | -0.33% |
