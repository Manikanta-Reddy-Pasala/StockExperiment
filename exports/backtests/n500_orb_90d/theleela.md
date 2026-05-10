# Leela Palaces Hotels & Resorts Ltd. (THELEELA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 421.30
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
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 7
- **Target hits / Stop hits / Partials:** 2 / 7 / 7
- **Avg / median % per leg:** 0.26% / 0.44%
- **Sum % (uncompounded):** 4.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 7 | 63.6% | 2 | 4 | 5 | 0.30% | 3.3% |
| BUY @ 2nd Alert (retest1) | 11 | 7 | 63.6% | 2 | 4 | 5 | 0.30% | 3.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 2 | 40.0% | 0 | 3 | 2 | 0.17% | 0.8% |
| SELL @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 3 | 2 | 0.17% | 0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 9 | 56.2% | 2 | 7 | 7 | 0.26% | 4.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 437.30 | 438.21 | 0.00 | ORB-short ORB[437.90,442.40] vol=2.2x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 09:50:00 | 434.70 | 436.91 | 0.00 | T1 1.5R @ 434.70 |
| Stop hit — per-position SL triggered | 2026-02-19 10:15:00 | 437.30 | 436.36 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:45:00 | 435.90 | 433.12 | 0.00 | ORB-long ORB[430.00,434.85] vol=2.1x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:30:00 | 438.21 | 435.15 | 0.00 | T1 1.5R @ 438.21 |
| Stop hit — per-position SL triggered | 2026-02-20 12:10:00 | 435.90 | 436.53 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 10:25:00 | 428.40 | 434.08 | 0.00 | ORB-short ORB[435.45,438.40] vol=5.4x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 10:30:00 | 425.68 | 433.72 | 0.00 | T1 1.5R @ 425.68 |
| Stop hit — per-position SL triggered | 2026-03-10 10:35:00 | 428.40 | 433.49 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:15:00 | 425.65 | 424.30 | 0.00 | ORB-long ORB[420.00,425.35] vol=1.6x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 10:25:00 | 428.19 | 424.67 | 0.00 | T1 1.5R @ 428.19 |
| Target hit | 2026-04-13 14:25:00 | 429.50 | 429.73 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2026-04-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:45:00 | 435.15 | 434.32 | 0.00 | ORB-long ORB[431.20,435.00] vol=1.5x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:45:00 | 437.49 | 435.64 | 0.00 | T1 1.5R @ 437.49 |
| Target hit | 2026-04-15 10:50:00 | 435.35 | 435.64 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2026-04-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:30:00 | 429.65 | 428.05 | 0.00 | ORB-long ORB[424.00,429.10] vol=1.7x ATR=0.88 |
| Stop hit — per-position SL triggered | 2026-04-17 09:40:00 | 428.77 | 428.63 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 11:10:00 | 428.30 | 424.46 | 0.00 | ORB-long ORB[421.45,427.40] vol=8.6x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 11:15:00 | 430.20 | 426.14 | 0.00 | T1 1.5R @ 430.20 |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 428.30 | 428.83 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:50:00 | 431.80 | 435.49 | 0.00 | ORB-short ORB[435.00,439.00] vol=1.5x ATR=1.67 |
| Stop hit — per-position SL triggered | 2026-04-24 10:00:00 | 433.47 | 435.38 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:50:00 | 420.15 | 419.41 | 0.00 | ORB-long ORB[415.00,420.00] vol=3.3x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 13:40:00 | 422.13 | 420.30 | 0.00 | T1 1.5R @ 422.13 |
| Stop hit — per-position SL triggered | 2026-05-06 13:45:00 | 420.15 | 420.30 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-19 09:30:00 | 437.30 | 2026-02-19 09:50:00 | 434.70 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-02-19 09:30:00 | 437.30 | 2026-02-19 10:15:00 | 437.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 09:45:00 | 435.90 | 2026-02-20 10:30:00 | 438.21 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-02-20 09:45:00 | 435.90 | 2026-02-20 12:10:00 | 435.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-10 10:25:00 | 428.40 | 2026-03-10 10:30:00 | 425.68 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2026-03-10 10:25:00 | 428.40 | 2026-03-10 10:35:00 | 428.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-13 10:15:00 | 425.65 | 2026-04-13 10:25:00 | 428.19 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-13 10:15:00 | 425.65 | 2026-04-13 14:25:00 | 429.50 | TARGET_HIT | 0.50 | 0.90% |
| BUY | retest1 | 2026-04-15 09:45:00 | 435.15 | 2026-04-15 10:45:00 | 437.49 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-04-15 09:45:00 | 435.15 | 2026-04-15 10:50:00 | 435.35 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2026-04-17 09:30:00 | 429.65 | 2026-04-17 09:40:00 | 428.77 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-04-20 11:10:00 | 428.30 | 2026-04-20 11:15:00 | 430.20 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-04-20 11:10:00 | 428.30 | 2026-04-20 15:15:00 | 428.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 09:50:00 | 431.80 | 2026-04-24 10:00:00 | 433.47 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-05-06 09:50:00 | 420.15 | 2026-05-06 13:40:00 | 422.13 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-05-06 09:50:00 | 420.15 | 2026-05-06 13:45:00 | 420.15 | STOP_HIT | 0.50 | 0.00% |
