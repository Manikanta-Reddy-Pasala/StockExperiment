# RHI MAGNESITA INDIA LTD. (RHIM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 409.05
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
- **Avg / median % per leg:** 0.21% / 0.48%
- **Sum % (uncompounded):** 3.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 7 | 63.6% | 3 | 4 | 4 | 0.26% | 2.9% |
| BUY @ 2nd Alert (retest1) | 11 | 7 | 63.6% | 3 | 4 | 4 | 0.26% | 2.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.12% | 0.7% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.12% | 0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 9 | 52.9% | 3 | 8 | 6 | 0.21% | 3.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:05:00 | 444.95 | 443.24 | 0.00 | ORB-long ORB[435.15,440.75] vol=2.4x ATR=2.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:40:00 | 448.05 | 444.05 | 0.00 | T1 1.5R @ 448.05 |
| Stop hit — per-position SL triggered | 2026-02-09 12:20:00 | 444.95 | 444.18 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:35:00 | 437.40 | 435.04 | 0.00 | ORB-long ORB[431.50,435.90] vol=3.7x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:45:00 | 439.72 | 436.67 | 0.00 | T1 1.5R @ 439.72 |
| Target hit | 2026-02-26 10:15:00 | 440.50 | 440.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2026-02-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:55:00 | 431.30 | 437.10 | 0.00 | ORB-short ORB[436.80,442.80] vol=2.3x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:30:00 | 429.22 | 435.05 | 0.00 | T1 1.5R @ 429.22 |
| Stop hit — per-position SL triggered | 2026-02-27 11:45:00 | 431.30 | 434.75 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:30:00 | 401.60 | 398.98 | 0.00 | ORB-long ORB[393.05,398.10] vol=1.7x ATR=1.77 |
| Stop hit — per-position SL triggered | 2026-03-06 10:45:00 | 399.83 | 399.07 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 09:55:00 | 343.40 | 345.27 | 0.00 | ORB-short ORB[345.00,350.00] vol=1.9x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-24 10:25:00 | 340.14 | 344.04 | 0.00 | T1 1.5R @ 340.14 |
| Stop hit — per-position SL triggered | 2026-03-24 11:45:00 | 343.40 | 342.93 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:40:00 | 390.60 | 386.78 | 0.00 | ORB-long ORB[382.00,387.30] vol=1.9x ATR=1.90 |
| Stop hit — per-position SL triggered | 2026-04-10 09:50:00 | 388.70 | 388.25 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:45:00 | 396.30 | 399.31 | 0.00 | ORB-short ORB[398.25,402.50] vol=1.7x ATR=1.68 |
| Stop hit — per-position SL triggered | 2026-04-16 10:30:00 | 397.98 | 398.22 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:55:00 | 405.10 | 401.36 | 0.00 | ORB-long ORB[398.05,402.45] vol=2.5x ATR=1.67 |
| Stop hit — per-position SL triggered | 2026-04-17 11:00:00 | 403.43 | 401.48 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:00:00 | 415.75 | 409.86 | 0.00 | ORB-long ORB[407.35,411.95] vol=2.1x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:10:00 | 418.38 | 414.13 | 0.00 | T1 1.5R @ 418.38 |
| Target hit | 2026-04-22 10:35:00 | 418.00 | 419.23 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 410.50 | 407.15 | 0.00 | ORB-long ORB[403.55,408.15] vol=1.7x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:50:00 | 413.14 | 411.63 | 0.00 | T1 1.5R @ 413.14 |
| Target hit | 2026-04-27 11:35:00 | 412.55 | 412.71 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2026-04-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:35:00 | 408.75 | 410.83 | 0.00 | ORB-short ORB[409.30,415.00] vol=5.7x ATR=1.26 |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 410.01 | 410.69 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:05:00 | 444.95 | 2026-02-09 11:40:00 | 448.05 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-02-09 11:05:00 | 444.95 | 2026-02-09 12:20:00 | 444.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 09:35:00 | 437.40 | 2026-02-26 09:45:00 | 439.72 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-02-26 09:35:00 | 437.40 | 2026-02-26 10:15:00 | 440.50 | TARGET_HIT | 0.50 | 0.71% |
| SELL | retest1 | 2026-02-27 10:55:00 | 431.30 | 2026-02-27 11:30:00 | 429.22 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-02-27 10:55:00 | 431.30 | 2026-02-27 11:45:00 | 431.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-06 10:30:00 | 401.60 | 2026-03-06 10:45:00 | 399.83 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-03-24 09:55:00 | 343.40 | 2026-03-24 10:25:00 | 340.14 | PARTIAL | 0.50 | 0.95% |
| SELL | retest1 | 2026-03-24 09:55:00 | 343.40 | 2026-03-24 11:45:00 | 343.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 09:40:00 | 390.60 | 2026-04-10 09:50:00 | 388.70 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-04-16 09:45:00 | 396.30 | 2026-04-16 10:30:00 | 397.98 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-17 10:55:00 | 405.10 | 2026-04-17 11:00:00 | 403.43 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-22 10:00:00 | 415.75 | 2026-04-22 10:10:00 | 418.38 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-22 10:00:00 | 415.75 | 2026-04-22 10:35:00 | 418.00 | TARGET_HIT | 0.50 | 0.54% |
| BUY | retest1 | 2026-04-27 09:30:00 | 410.50 | 2026-04-27 09:50:00 | 413.14 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-04-27 09:30:00 | 410.50 | 2026-04-27 11:35:00 | 412.55 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2026-04-29 10:35:00 | 408.75 | 2026-04-29 11:15:00 | 410.01 | STOP_HIT | 1.00 | -0.31% |
