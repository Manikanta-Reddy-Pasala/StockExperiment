# Billionbrains Garage Ventures Ltd. (GROWW)

## Backtest Summary

- **Window:** 2025-11-12 10:00:00 → 2026-05-08 15:25:00 (7491 bars)
- **Last close:** 204.45
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
| ENTRY1 | 10 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 6
- **Target hits / Stop hits / Partials:** 4 / 6 / 7
- **Avg / median % per leg:** 0.41% / 0.50%
- **Sum % (uncompounded):** 7.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 7 | 58.3% | 2 | 5 | 5 | 0.33% | 4.0% |
| BUY @ 2nd Alert (retest1) | 12 | 7 | 58.3% | 2 | 5 | 5 | 0.33% | 4.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 0.61% | 3.1% |
| SELL @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 2 | 1 | 2 | 0.61% | 3.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 11 | 64.7% | 4 | 6 | 7 | 0.41% | 7.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-01-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 10:05:00 | 158.80 | 156.26 | 0.00 | ORB-long ORB[154.50,155.35] vol=4.9x ATR=0.70 |
| Stop hit — per-position SL triggered | 2026-01-07 10:10:00 | 158.10 | 156.67 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-01-14 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 10:10:00 | 160.12 | 161.75 | 0.00 | ORB-short ORB[161.60,163.50] vol=1.9x ATR=0.67 |
| Stop hit — per-position SL triggered | 2026-01-14 10:20:00 | 160.79 | 161.55 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-01-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-23 09:50:00 | 172.10 | 169.12 | 0.00 | ORB-long ORB[165.26,167.80] vol=4.4x ATR=0.90 |
| Stop hit — per-position SL triggered | 2026-01-23 09:55:00 | 171.20 | 170.09 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-01-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:55:00 | 175.46 | 172.30 | 0.00 | ORB-long ORB[170.30,171.90] vol=2.0x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 10:10:00 | 176.83 | 173.97 | 0.00 | T1 1.5R @ 176.83 |
| Target hit | 2026-01-30 12:25:00 | 178.00 | 178.61 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — SELL (started 2026-02-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:30:00 | 169.30 | 170.14 | 0.00 | ORB-short ORB[169.67,171.50] vol=1.5x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:30:00 | 168.46 | 169.58 | 0.00 | T1 1.5R @ 168.46 |
| Target hit | 2026-02-23 15:20:00 | 167.35 | 168.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2026-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:30:00 | 158.08 | 157.62 | 0.00 | ORB-long ORB[156.05,157.99] vol=1.6x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 09:40:00 | 158.97 | 157.77 | 0.00 | T1 1.5R @ 158.97 |
| Stop hit — per-position SL triggered | 2026-03-06 10:25:00 | 158.08 | 158.17 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:15:00 | 164.50 | 162.59 | 0.00 | ORB-long ORB[160.76,163.20] vol=1.6x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 10:25:00 | 165.83 | 163.58 | 0.00 | T1 1.5R @ 165.83 |
| Stop hit — per-position SL triggered | 2026-03-25 12:00:00 | 164.50 | 164.32 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:05:00 | 175.70 | 173.55 | 0.00 | ORB-long ORB[171.13,173.75] vol=1.9x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 10:15:00 | 177.04 | 174.22 | 0.00 | T1 1.5R @ 177.04 |
| Stop hit — per-position SL triggered | 2026-04-08 10:30:00 | 175.70 | 174.60 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:15:00 | 219.30 | 217.30 | 0.00 | ORB-long ORB[215.53,218.17] vol=2.4x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:20:00 | 220.50 | 217.76 | 0.00 | T1 1.5R @ 220.50 |
| Target hit | 2026-05-04 13:25:00 | 219.35 | 219.49 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 11:15:00 | 207.08 | 207.87 | 0.00 | ORB-short ORB[207.30,210.20] vol=1.5x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 12:00:00 | 206.09 | 207.76 | 0.00 | T1 1.5R @ 206.09 |
| Target hit | 2026-05-08 15:20:00 | 204.28 | 206.69 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-01-07 10:05:00 | 158.80 | 2026-01-07 10:10:00 | 158.10 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-01-14 10:10:00 | 160.12 | 2026-01-14 10:20:00 | 160.79 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-01-23 09:50:00 | 172.10 | 2026-01-23 09:55:00 | 171.20 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-01-30 09:55:00 | 175.46 | 2026-01-30 10:10:00 | 176.83 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2026-01-30 09:55:00 | 175.46 | 2026-01-30 12:25:00 | 178.00 | TARGET_HIT | 0.50 | 1.45% |
| SELL | retest1 | 2026-02-23 09:30:00 | 169.30 | 2026-02-23 10:30:00 | 168.46 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-02-23 09:30:00 | 169.30 | 2026-02-23 15:20:00 | 167.35 | TARGET_HIT | 0.50 | 1.15% |
| BUY | retest1 | 2026-03-06 09:30:00 | 158.08 | 2026-03-06 09:40:00 | 158.97 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-03-06 09:30:00 | 158.08 | 2026-03-06 10:25:00 | 158.08 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-25 10:15:00 | 164.50 | 2026-03-25 10:25:00 | 165.83 | PARTIAL | 0.50 | 0.81% |
| BUY | retest1 | 2026-03-25 10:15:00 | 164.50 | 2026-03-25 12:00:00 | 164.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-08 10:05:00 | 175.70 | 2026-04-08 10:15:00 | 177.04 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2026-04-08 10:05:00 | 175.70 | 2026-04-08 10:30:00 | 175.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 11:15:00 | 219.30 | 2026-05-04 11:20:00 | 220.50 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-05-04 11:15:00 | 219.30 | 2026-05-04 13:25:00 | 219.35 | TARGET_HIT | 0.50 | 0.02% |
| SELL | retest1 | 2026-05-08 11:15:00 | 207.08 | 2026-05-08 12:00:00 | 206.09 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-05-08 11:15:00 | 207.08 | 2026-05-08 15:20:00 | 204.28 | TARGET_HIT | 0.50 | 1.35% |
