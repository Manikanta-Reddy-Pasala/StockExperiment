# Power Finance Corporation Ltd. (PFC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 461.60
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
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 9
- **Target hits / Stop hits / Partials:** 3 / 9 / 8
- **Avg / median % per leg:** 0.34% / 0.36%
- **Sum % (uncompounded):** 6.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 8 | 61.5% | 3 | 5 | 5 | 0.45% | 5.9% |
| BUY @ 2nd Alert (retest1) | 13 | 8 | 61.5% | 3 | 5 | 5 | 0.45% | 5.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 3 | 42.9% | 0 | 4 | 3 | 0.14% | 1.0% |
| SELL @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 0 | 4 | 3 | 0.14% | 1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 11 | 55.0% | 3 | 9 | 8 | 0.34% | 6.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:05:00 | 414.25 | 412.20 | 0.00 | ORB-long ORB[409.30,412.55] vol=1.7x ATR=0.81 |
| Stop hit — per-position SL triggered | 2026-02-17 11:10:00 | 413.44 | 412.30 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 11:15:00 | 421.80 | 419.94 | 0.00 | ORB-long ORB[417.65,421.25] vol=2.2x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:25:00 | 423.20 | 420.35 | 0.00 | T1 1.5R @ 423.20 |
| Stop hit — per-position SL triggered | 2026-02-18 11:30:00 | 421.80 | 420.41 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 11:10:00 | 415.00 | 410.92 | 0.00 | ORB-long ORB[408.05,410.95] vol=1.7x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:30:00 | 416.63 | 411.61 | 0.00 | T1 1.5R @ 416.63 |
| Stop hit — per-position SL triggered | 2026-02-24 12:15:00 | 415.00 | 412.80 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:00:00 | 419.75 | 422.90 | 0.00 | ORB-short ORB[423.15,426.90] vol=1.7x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:40:00 | 418.23 | 422.18 | 0.00 | T1 1.5R @ 418.23 |
| Stop hit — per-position SL triggered | 2026-02-26 13:25:00 | 419.75 | 420.84 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:40:00 | 404.70 | 401.55 | 0.00 | ORB-long ORB[398.00,402.20] vol=3.1x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:45:00 | 407.21 | 402.23 | 0.00 | T1 1.5R @ 407.21 |
| Target hit | 2026-03-05 15:20:00 | 414.45 | 408.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-03-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:30:00 | 406.45 | 412.66 | 0.00 | ORB-short ORB[412.85,418.25] vol=1.7x ATR=1.87 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 408.32 | 411.87 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:25:00 | 399.50 | 404.03 | 0.00 | ORB-short ORB[401.60,407.45] vol=2.4x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:55:00 | 396.66 | 402.73 | 0.00 | T1 1.5R @ 396.66 |
| Stop hit — per-position SL triggered | 2026-03-16 11:00:00 | 399.50 | 402.66 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:15:00 | 408.75 | 406.85 | 0.00 | ORB-long ORB[404.30,408.70] vol=1.6x ATR=1.62 |
| Stop hit — per-position SL triggered | 2026-03-17 10:30:00 | 407.13 | 407.10 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:00:00 | 453.10 | 450.99 | 0.00 | ORB-long ORB[446.50,451.90] vol=1.5x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:25:00 | 456.12 | 452.33 | 0.00 | T1 1.5R @ 456.12 |
| Target hit | 2026-04-16 13:05:00 | 457.85 | 458.62 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — SELL (started 2026-04-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:45:00 | 464.80 | 467.56 | 0.00 | ORB-short ORB[466.25,471.15] vol=5.9x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:55:00 | 463.02 | 467.26 | 0.00 | T1 1.5R @ 463.02 |
| Stop hit — per-position SL triggered | 2026-04-24 11:30:00 | 464.80 | 466.84 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 479.00 | 476.10 | 0.00 | ORB-long ORB[470.20,476.70] vol=4.4x ATR=1.89 |
| Stop hit — per-position SL triggered | 2026-04-28 09:35:00 | 477.11 | 476.91 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 10:55:00 | 452.50 | 452.25 | 0.00 | ORB-long ORB[444.00,449.95] vol=1.7x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 12:15:00 | 454.67 | 452.51 | 0.00 | T1 1.5R @ 454.67 |
| Target hit | 2026-05-05 15:20:00 | 456.75 | 455.33 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 11:05:00 | 414.25 | 2026-02-17 11:10:00 | 413.44 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-02-18 11:15:00 | 421.80 | 2026-02-18 11:25:00 | 423.20 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-02-18 11:15:00 | 421.80 | 2026-02-18 11:30:00 | 421.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 11:10:00 | 415.00 | 2026-02-24 11:30:00 | 416.63 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-24 11:10:00 | 415.00 | 2026-02-24 12:15:00 | 415.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-26 11:00:00 | 419.75 | 2026-02-26 11:40:00 | 418.23 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-26 11:00:00 | 419.75 | 2026-02-26 13:25:00 | 419.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-05 10:40:00 | 404.70 | 2026-03-05 10:45:00 | 407.21 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-03-05 10:40:00 | 404.70 | 2026-03-05 15:20:00 | 414.45 | TARGET_HIT | 0.50 | 2.41% |
| SELL | retest1 | 2026-03-13 10:30:00 | 406.45 | 2026-03-13 10:50:00 | 408.32 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-03-16 10:25:00 | 399.50 | 2026-03-16 10:55:00 | 396.66 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2026-03-16 10:25:00 | 399.50 | 2026-03-16 11:00:00 | 399.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:15:00 | 408.75 | 2026-03-17 10:30:00 | 407.13 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-16 10:00:00 | 453.10 | 2026-04-16 10:25:00 | 456.12 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-04-16 10:00:00 | 453.10 | 2026-04-16 13:05:00 | 457.85 | TARGET_HIT | 0.50 | 1.05% |
| SELL | retest1 | 2026-04-24 10:45:00 | 464.80 | 2026-04-24 10:55:00 | 463.02 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-24 10:45:00 | 464.80 | 2026-04-24 11:30:00 | 464.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-28 09:30:00 | 479.00 | 2026-04-28 09:35:00 | 477.11 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-05-05 10:55:00 | 452.50 | 2026-05-05 12:15:00 | 454.67 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-05-05 10:55:00 | 452.50 | 2026-05-05 15:20:00 | 456.75 | TARGET_HIT | 0.50 | 0.94% |
