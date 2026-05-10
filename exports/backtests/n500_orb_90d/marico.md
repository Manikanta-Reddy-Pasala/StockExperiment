# Marico Ltd. (MARICO)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 830.50
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
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 9
- **Target hits / Stop hits / Partials:** 2 / 9 / 3
- **Avg / median % per leg:** 0.05% / -0.19%
- **Sum % (uncompounded):** 0.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 1 | 6 | 1 | -0.10% | -0.8% |
| BUY @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 1 | 6 | 1 | -0.10% | -0.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.24% | 1.5% |
| SELL @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.24% | 1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 5 | 35.7% | 2 | 9 | 3 | 0.05% | 0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 758.00 | 754.22 | 0.00 | ORB-long ORB[749.40,757.90] vol=2.1x ATR=1.96 |
| Stop hit — per-position SL triggered | 2026-02-10 09:50:00 | 756.04 | 755.05 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:15:00 | 769.65 | 766.28 | 0.00 | ORB-long ORB[757.00,767.60] vol=1.6x ATR=1.45 |
| Stop hit — per-position SL triggered | 2026-02-16 11:55:00 | 768.20 | 766.67 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 784.50 | 781.71 | 0.00 | ORB-long ORB[773.60,783.90] vol=2.9x ATR=1.84 |
| Stop hit — per-position SL triggered | 2026-02-18 09:45:00 | 782.66 | 782.03 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:50:00 | 789.00 | 793.11 | 0.00 | ORB-short ORB[791.65,800.00] vol=2.1x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:05:00 | 786.30 | 791.84 | 0.00 | T1 1.5R @ 786.30 |
| Target hit | 2026-02-19 15:20:00 | 778.25 | 785.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-02-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:05:00 | 784.95 | 782.53 | 0.00 | ORB-long ORB[775.15,780.35] vol=1.5x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:15:00 | 788.15 | 782.81 | 0.00 | T1 1.5R @ 788.15 |
| Target hit | 2026-02-20 14:30:00 | 787.45 | 788.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2026-02-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:00:00 | 810.85 | 805.96 | 0.00 | ORB-long ORB[801.00,804.35] vol=2.0x ATR=2.03 |
| Stop hit — per-position SL triggered | 2026-02-24 10:25:00 | 808.82 | 808.61 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 09:40:00 | 726.15 | 730.03 | 0.00 | ORB-short ORB[730.25,736.80] vol=2.0x ATR=2.70 |
| Stop hit — per-position SL triggered | 2026-03-24 10:00:00 | 728.85 | 729.31 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 10:55:00 | 756.70 | 758.10 | 0.00 | ORB-short ORB[756.90,764.50] vol=3.3x ATR=1.94 |
| Stop hit — per-position SL triggered | 2026-04-15 11:25:00 | 758.64 | 758.06 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:05:00 | 749.45 | 750.99 | 0.00 | ORB-short ORB[751.25,760.20] vol=7.3x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 12:00:00 | 746.62 | 750.23 | 0.00 | T1 1.5R @ 746.62 |
| Stop hit — per-position SL triggered | 2026-04-16 12:10:00 | 749.45 | 750.08 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:45:00 | 757.95 | 750.96 | 0.00 | ORB-long ORB[741.35,749.10] vol=1.7x ATR=2.49 |
| Stop hit — per-position SL triggered | 2026-04-17 12:30:00 | 755.46 | 753.52 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:15:00 | 785.50 | 783.33 | 0.00 | ORB-long ORB[774.90,784.05] vol=9.7x ATR=2.01 |
| Stop hit — per-position SL triggered | 2026-05-04 11:20:00 | 783.49 | 783.39 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:40:00 | 758.00 | 2026-02-10 09:50:00 | 756.04 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-16 11:15:00 | 769.65 | 2026-02-16 11:55:00 | 768.20 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-02-18 09:30:00 | 784.50 | 2026-02-18 09:45:00 | 782.66 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-19 10:50:00 | 789.00 | 2026-02-19 11:05:00 | 786.30 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-19 10:50:00 | 789.00 | 2026-02-19 15:20:00 | 778.25 | TARGET_HIT | 0.50 | 1.36% |
| BUY | retest1 | 2026-02-20 11:05:00 | 784.95 | 2026-02-20 11:15:00 | 788.15 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-02-20 11:05:00 | 784.95 | 2026-02-20 14:30:00 | 787.45 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-24 10:00:00 | 810.85 | 2026-02-24 10:25:00 | 808.82 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-24 09:40:00 | 726.15 | 2026-03-24 10:00:00 | 728.85 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-04-15 10:55:00 | 756.70 | 2026-04-15 11:25:00 | 758.64 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-16 11:05:00 | 749.45 | 2026-04-16 12:00:00 | 746.62 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-16 11:05:00 | 749.45 | 2026-04-16 12:10:00 | 749.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 10:45:00 | 757.95 | 2026-04-17 12:30:00 | 755.46 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-05-04 11:15:00 | 785.50 | 2026-05-04 11:20:00 | 783.49 | STOP_HIT | 1.00 | -0.26% |
