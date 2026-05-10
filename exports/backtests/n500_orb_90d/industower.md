# Indus Towers Ltd. (INDUSTOWER)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 402.80
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
| ENTRY1 | 19 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 4 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 15
- **Target hits / Stop hits / Partials:** 4 / 15 / 10
- **Avg / median % per leg:** 0.23% / 0.00%
- **Sum % (uncompounded):** 6.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | -0.01% | -0.1% |
| BUY @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | -0.01% | -0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 18 | 10 | 55.6% | 3 | 8 | 7 | 0.38% | 6.8% |
| SELL @ 2nd Alert (retest1) | 18 | 10 | 55.6% | 3 | 8 | 7 | 0.38% | 6.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 29 | 14 | 48.3% | 4 | 15 | 10 | 0.23% | 6.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:50:00 | 460.05 | 458.49 | 0.00 | ORB-long ORB[455.05,459.65] vol=1.9x ATR=1.50 |
| Stop hit — per-position SL triggered | 2026-02-10 10:00:00 | 458.55 | 458.61 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:25:00 | 467.30 | 463.53 | 0.00 | ORB-long ORB[457.20,461.00] vol=1.8x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:30:00 | 469.88 | 464.20 | 0.00 | T1 1.5R @ 469.88 |
| Stop hit — per-position SL triggered | 2026-02-11 12:05:00 | 467.30 | 467.41 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 476.20 | 473.47 | 0.00 | ORB-long ORB[469.50,474.45] vol=1.6x ATR=1.49 |
| Stop hit — per-position SL triggered | 2026-02-18 09:35:00 | 474.71 | 473.79 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:45:00 | 475.70 | 478.33 | 0.00 | ORB-short ORB[477.20,481.50] vol=1.5x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:20:00 | 473.43 | 477.53 | 0.00 | T1 1.5R @ 473.43 |
| Stop hit — per-position SL triggered | 2026-02-19 12:00:00 | 475.70 | 476.66 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:00:00 | 470.45 | 472.19 | 0.00 | ORB-short ORB[472.35,476.75] vol=1.8x ATR=1.27 |
| Stop hit — per-position SL triggered | 2026-02-23 11:10:00 | 471.72 | 472.07 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:05:00 | 465.20 | 468.02 | 0.00 | ORB-short ORB[468.75,473.00] vol=1.5x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:55:00 | 463.33 | 467.30 | 0.00 | T1 1.5R @ 463.33 |
| Target hit | 2026-02-25 15:20:00 | 462.00 | 463.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-02-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:55:00 | 455.60 | 459.05 | 0.00 | ORB-short ORB[458.20,462.70] vol=4.4x ATR=1.18 |
| Stop hit — per-position SL triggered | 2026-02-26 11:05:00 | 456.78 | 458.83 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:55:00 | 434.65 | 435.55 | 0.00 | ORB-short ORB[437.30,440.80] vol=3.7x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:05:00 | 432.41 | 434.79 | 0.00 | T1 1.5R @ 432.41 |
| Target hit | 2026-03-13 15:20:00 | 423.05 | 428.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-03-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:30:00 | 434.70 | 432.02 | 0.00 | ORB-long ORB[427.30,433.50] vol=1.6x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-03-17 10:35:00 | 433.14 | 433.56 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 440.45 | 439.06 | 0.00 | ORB-long ORB[436.00,439.95] vol=3.0x ATR=1.48 |
| Stop hit — per-position SL triggered | 2026-03-18 09:40:00 | 438.97 | 439.00 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-25 09:30:00 | 430.35 | 432.32 | 0.00 | ORB-short ORB[430.70,434.80] vol=2.6x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 09:35:00 | 427.76 | 431.75 | 0.00 | T1 1.5R @ 427.76 |
| Stop hit — per-position SL triggered | 2026-03-25 09:45:00 | 430.35 | 430.64 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:55:00 | 418.95 | 419.85 | 0.00 | ORB-short ORB[420.65,424.70] vol=2.0x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:55:00 | 417.17 | 419.23 | 0.00 | T1 1.5R @ 417.17 |
| Target hit | 2026-04-16 15:20:00 | 413.50 | 416.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2026-04-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:35:00 | 402.10 | 405.59 | 0.00 | ORB-short ORB[403.05,408.30] vol=2.7x ATR=1.10 |
| Stop hit — per-position SL triggered | 2026-04-23 11:55:00 | 403.20 | 404.68 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:40:00 | 398.20 | 399.91 | 0.00 | ORB-short ORB[403.50,406.60] vol=1.8x ATR=1.21 |
| Stop hit — per-position SL triggered | 2026-04-24 11:30:00 | 399.41 | 398.97 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:50:00 | 402.90 | 404.06 | 0.00 | ORB-short ORB[403.40,407.50] vol=1.6x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 11:30:00 | 401.37 | 403.57 | 0.00 | T1 1.5R @ 401.37 |
| Stop hit — per-position SL triggered | 2026-04-27 12:35:00 | 402.90 | 402.90 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:45:00 | 419.60 | 417.76 | 0.00 | ORB-long ORB[415.00,419.00] vol=3.3x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:15:00 | 421.71 | 418.66 | 0.00 | T1 1.5R @ 421.71 |
| Stop hit — per-position SL triggered | 2026-04-29 10:20:00 | 419.60 | 418.75 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:40:00 | 404.85 | 400.62 | 0.00 | ORB-long ORB[396.45,401.00] vol=1.6x ATR=1.93 |
| Stop hit — per-position SL triggered | 2026-05-05 10:10:00 | 402.92 | 403.29 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-05-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:45:00 | 408.00 | 406.57 | 0.00 | ORB-long ORB[404.45,407.65] vol=1.8x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:50:00 | 410.06 | 406.98 | 0.00 | T1 1.5R @ 410.06 |
| Target hit | 2026-05-06 12:20:00 | 408.60 | 409.36 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — SELL (started 2026-05-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:40:00 | 405.25 | 407.24 | 0.00 | ORB-short ORB[406.65,411.80] vol=1.5x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 10:50:00 | 403.56 | 406.33 | 0.00 | T1 1.5R @ 403.56 |
| Stop hit — per-position SL triggered | 2026-05-07 11:20:00 | 405.25 | 406.14 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:50:00 | 460.05 | 2026-02-10 10:00:00 | 458.55 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-11 10:25:00 | 467.30 | 2026-02-11 10:30:00 | 469.88 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-02-11 10:25:00 | 467.30 | 2026-02-11 12:05:00 | 467.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 09:30:00 | 476.20 | 2026-02-18 09:35:00 | 474.71 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-19 09:45:00 | 475.70 | 2026-02-19 10:20:00 | 473.43 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-02-19 09:45:00 | 475.70 | 2026-02-19 12:00:00 | 475.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-23 11:00:00 | 470.45 | 2026-02-23 11:10:00 | 471.72 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-25 11:05:00 | 465.20 | 2026-02-25 11:55:00 | 463.33 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-25 11:05:00 | 465.20 | 2026-02-25 15:20:00 | 462.00 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2026-02-26 10:55:00 | 455.60 | 2026-02-26 11:05:00 | 456.78 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-03-13 10:55:00 | 434.65 | 2026-03-13 11:05:00 | 432.41 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-03-13 10:55:00 | 434.65 | 2026-03-13 15:20:00 | 423.05 | TARGET_HIT | 0.50 | 2.67% |
| BUY | retest1 | 2026-03-17 09:30:00 | 434.70 | 2026-03-17 10:35:00 | 433.14 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-18 09:30:00 | 440.45 | 2026-03-18 09:40:00 | 438.97 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-25 09:30:00 | 430.35 | 2026-03-25 09:35:00 | 427.76 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-03-25 09:30:00 | 430.35 | 2026-03-25 09:45:00 | 430.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 10:55:00 | 418.95 | 2026-04-16 11:55:00 | 417.17 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-16 10:55:00 | 418.95 | 2026-04-16 15:20:00 | 413.50 | TARGET_HIT | 0.50 | 1.30% |
| SELL | retest1 | 2026-04-23 10:35:00 | 402.10 | 2026-04-23 11:55:00 | 403.20 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-24 10:40:00 | 398.20 | 2026-04-24 11:30:00 | 399.41 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-04-27 10:50:00 | 402.90 | 2026-04-27 11:30:00 | 401.37 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-27 10:50:00 | 402.90 | 2026-04-27 12:35:00 | 402.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 09:45:00 | 419.60 | 2026-04-29 10:15:00 | 421.71 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-29 09:45:00 | 419.60 | 2026-04-29 10:20:00 | 419.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 09:40:00 | 404.85 | 2026-05-05 10:10:00 | 402.92 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-05-06 09:45:00 | 408.00 | 2026-05-06 09:50:00 | 410.06 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-05-06 09:45:00 | 408.00 | 2026-05-06 12:20:00 | 408.60 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2026-05-07 10:40:00 | 405.25 | 2026-05-07 10:50:00 | 403.56 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-05-07 10:40:00 | 405.25 | 2026-05-07 11:20:00 | 405.25 | STOP_HIT | 0.50 | 0.00% |
