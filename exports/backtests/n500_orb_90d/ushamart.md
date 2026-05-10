# Usha Martin Ltd. (USHAMART)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 472.00
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
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 9
- **Target hits / Stop hits / Partials:** 2 / 9 / 5
- **Avg / median % per leg:** 0.21% / 0.00%
- **Sum % (uncompounded):** 3.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 7 | 53.8% | 2 | 6 | 5 | 0.35% | 4.6% |
| BUY @ 2nd Alert (retest1) | 13 | 7 | 53.8% | 2 | 6 | 5 | 0.35% | 4.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.43% | -1.3% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.43% | -1.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 7 | 43.8% | 2 | 9 | 5 | 0.21% | 3.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:45:00 | 422.75 | 419.47 | 0.00 | ORB-long ORB[414.40,420.00] vol=2.3x ATR=1.73 |
| Stop hit — per-position SL triggered | 2026-02-17 09:50:00 | 421.02 | 419.94 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-03-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 10:10:00 | 416.25 | 412.05 | 0.00 | ORB-long ORB[408.85,415.00] vol=3.1x ATR=2.23 |
| Stop hit — per-position SL triggered | 2026-03-04 10:40:00 | 414.02 | 412.71 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:15:00 | 420.30 | 417.78 | 0.00 | ORB-long ORB[414.35,420.00] vol=2.2x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:35:00 | 422.34 | 418.74 | 0.00 | T1 1.5R @ 422.34 |
| Stop hit — per-position SL triggered | 2026-03-05 11:00:00 | 420.30 | 419.58 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 417.35 | 422.56 | 0.00 | ORB-short ORB[420.50,425.45] vol=2.1x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-03-06 10:50:00 | 418.72 | 421.99 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:40:00 | 416.90 | 420.04 | 0.00 | ORB-short ORB[420.30,425.75] vol=7.1x ATR=2.29 |
| Stop hit — per-position SL triggered | 2026-03-11 09:50:00 | 419.19 | 419.81 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:50:00 | 405.10 | 403.97 | 0.00 | ORB-long ORB[399.50,404.90] vol=1.7x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:30:00 | 407.30 | 404.60 | 0.00 | T1 1.5R @ 407.30 |
| Target hit | 2026-03-18 15:20:00 | 412.10 | 409.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-03-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:00:00 | 400.15 | 403.66 | 0.00 | ORB-short ORB[402.80,407.20] vol=1.6x ATR=1.61 |
| Stop hit — per-position SL triggered | 2026-03-19 10:50:00 | 401.76 | 402.87 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 10:20:00 | 408.05 | 405.28 | 0.00 | ORB-long ORB[402.45,407.05] vol=3.0x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-06 10:30:00 | 411.01 | 405.99 | 0.00 | T1 1.5R @ 411.01 |
| Stop hit — per-position SL triggered | 2026-04-06 12:40:00 | 408.05 | 408.31 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 10:55:00 | 445.45 | 442.81 | 0.00 | ORB-long ORB[439.00,443.75] vol=6.0x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:00:00 | 447.50 | 443.69 | 0.00 | T1 1.5R @ 447.50 |
| Stop hit — per-position SL triggered | 2026-04-24 11:50:00 | 445.45 | 447.44 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:40:00 | 463.40 | 461.51 | 0.00 | ORB-long ORB[455.60,461.70] vol=4.1x ATR=2.53 |
| Stop hit — per-position SL triggered | 2026-04-29 14:20:00 | 460.87 | 463.09 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 11:00:00 | 462.70 | 461.39 | 0.00 | ORB-long ORB[456.45,462.10] vol=2.4x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:50:00 | 465.05 | 462.86 | 0.00 | T1 1.5R @ 465.05 |
| Target hit | 2026-05-06 15:20:00 | 470.35 | 467.95 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 09:45:00 | 422.75 | 2026-02-17 09:50:00 | 421.02 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-03-04 10:10:00 | 416.25 | 2026-03-04 10:40:00 | 414.02 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2026-03-05 10:15:00 | 420.30 | 2026-03-05 10:35:00 | 422.34 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-03-05 10:15:00 | 420.30 | 2026-03-05 11:00:00 | 420.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:45:00 | 417.35 | 2026-03-06 10:50:00 | 418.72 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-03-11 09:40:00 | 416.90 | 2026-03-11 09:50:00 | 419.19 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2026-03-18 09:50:00 | 405.10 | 2026-03-18 10:30:00 | 407.30 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-03-18 09:50:00 | 405.10 | 2026-03-18 15:20:00 | 412.10 | TARGET_HIT | 0.50 | 1.73% |
| SELL | retest1 | 2026-03-19 10:00:00 | 400.15 | 2026-03-19 10:50:00 | 401.76 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-06 10:20:00 | 408.05 | 2026-04-06 10:30:00 | 411.01 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2026-04-06 10:20:00 | 408.05 | 2026-04-06 12:40:00 | 408.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-24 10:55:00 | 445.45 | 2026-04-24 11:00:00 | 447.50 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-04-24 10:55:00 | 445.45 | 2026-04-24 11:50:00 | 445.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 09:40:00 | 463.40 | 2026-04-29 14:20:00 | 460.87 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2026-05-06 11:00:00 | 462.70 | 2026-05-06 11:50:00 | 465.05 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-05-06 11:00:00 | 462.70 | 2026-05-06 15:20:00 | 470.35 | TARGET_HIT | 0.50 | 1.65% |
