# Tata Steel Ltd. (TATASTEEL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2025-07-09 15:25:00 (3225 bars)
- **Last close:** 159.20
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
- **Avg / median % per leg:** -0.01% / -0.21%
- **Sum % (uncompounded):** -0.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.08% | 0.8% |
| BUY @ 2nd Alert (retest1) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.08% | 0.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.23% | -0.9% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.23% | -0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 5 | 35.7% | 2 | 9 | 3 | -0.01% | -0.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:35:00 | 157.36 | 155.74 | 0.00 | ORB-long ORB[154.30,156.29] vol=1.6x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-05-15 10:45:00 | 156.80 | 155.93 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-22 11:15:00 | 163.16 | 162.37 | 0.00 | ORB-long ORB[160.36,162.80] vol=1.9x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-05-22 11:20:00 | 162.79 | 162.38 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:30:00 | 165.04 | 164.33 | 0.00 | ORB-long ORB[163.10,164.68] vol=1.5x ATR=0.42 |
| Stop hit — per-position SL triggered | 2025-05-26 09:55:00 | 164.62 | 164.64 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 11:00:00 | 163.03 | 162.01 | 0.00 | ORB-long ORB[161.63,162.68] vol=1.9x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 11:25:00 | 163.60 | 162.27 | 0.00 | T1 1.5R @ 163.60 |
| Stop hit — per-position SL triggered | 2025-05-27 11:45:00 | 163.03 | 162.40 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 09:55:00 | 157.18 | 158.09 | 0.00 | ORB-short ORB[157.59,159.30] vol=2.0x ATR=0.40 |
| Stop hit — per-position SL triggered | 2025-06-10 10:10:00 | 157.58 | 157.98 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 09:40:00 | 154.94 | 155.43 | 0.00 | ORB-short ORB[155.04,156.26] vol=1.7x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-06-12 10:30:00 | 155.30 | 155.20 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 10:50:00 | 153.16 | 152.12 | 0.00 | ORB-long ORB[151.32,153.06] vol=3.7x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 12:30:00 | 153.80 | 152.55 | 0.00 | T1 1.5R @ 153.80 |
| Target hit | 2025-06-16 15:20:00 | 154.12 | 153.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2025-06-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 154.56 | 153.95 | 0.00 | ORB-long ORB[153.17,154.35] vol=1.7x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-06-17 09:45:00 | 154.21 | 154.13 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:55:00 | 155.00 | 154.35 | 0.00 | ORB-long ORB[153.64,154.80] vol=2.0x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 11:05:00 | 155.46 | 154.46 | 0.00 | T1 1.5R @ 155.46 |
| Target hit | 2025-06-24 13:20:00 | 155.26 | 155.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — SELL (started 2025-07-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:55:00 | 158.10 | 158.77 | 0.00 | ORB-short ORB[158.65,160.44] vol=1.5x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-07-01 11:40:00 | 158.45 | 158.61 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 09:45:00 | 160.19 | 160.47 | 0.00 | ORB-short ORB[160.35,161.95] vol=5.0x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-07-09 09:55:00 | 160.53 | 160.46 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-15 10:35:00 | 157.36 | 2025-05-15 10:45:00 | 156.80 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-05-22 11:15:00 | 163.16 | 2025-05-22 11:20:00 | 162.79 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-05-26 09:30:00 | 165.04 | 2025-05-26 09:55:00 | 164.62 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-05-27 11:00:00 | 163.03 | 2025-05-27 11:25:00 | 163.60 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-05-27 11:00:00 | 163.03 | 2025-05-27 11:45:00 | 163.03 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-10 09:55:00 | 157.18 | 2025-06-10 10:10:00 | 157.58 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-06-12 09:40:00 | 154.94 | 2025-06-12 10:30:00 | 155.30 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-06-16 10:50:00 | 153.16 | 2025-06-16 12:30:00 | 153.80 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-06-16 10:50:00 | 153.16 | 2025-06-16 15:20:00 | 154.12 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2025-06-17 09:30:00 | 154.56 | 2025-06-17 09:45:00 | 154.21 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-06-24 10:55:00 | 155.00 | 2025-06-24 11:05:00 | 155.46 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-06-24 10:55:00 | 155.00 | 2025-06-24 13:20:00 | 155.26 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2025-07-01 10:55:00 | 158.10 | 2025-07-01 11:40:00 | 158.45 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-09 09:45:00 | 160.19 | 2025-07-09 09:55:00 | 160.53 | STOP_HIT | 1.00 | -0.21% |
