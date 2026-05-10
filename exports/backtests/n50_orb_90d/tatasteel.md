# TATASTEEL (TATASTEEL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 214.60
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 10
- **Target hits / Stop hits / Partials:** 1 / 10 / 1
- **Avg / median % per leg:** -0.03% / -0.24%
- **Sum % (uncompounded):** -0.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.20% | 1.2% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.20% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.26% | -1.5% |
| SELL @ 2nd Alert (retest1) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.26% | -1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 2 | 16.7% | 1 | 10 | 1 | -0.03% | -0.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 204.07 | 202.87 | 0.00 | ORB-long ORB[201.08,202.99] vol=2.6x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:35:00 | 205.20 | 203.53 | 0.00 | T1 1.5R @ 205.20 |
| Target hit | 2026-02-10 11:40:00 | 207.76 | 207.78 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2026-02-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:05:00 | 205.85 | 208.26 | 0.00 | ORB-short ORB[207.68,209.90] vol=2.3x ATR=0.60 |
| Stop hit — per-position SL triggered | 2026-02-11 11:15:00 | 206.45 | 208.03 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:50:00 | 209.12 | 208.34 | 0.00 | ORB-long ORB[206.20,208.85] vol=1.9x ATR=0.62 |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 208.50 | 208.56 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 204.16 | 205.64 | 0.00 | ORB-short ORB[204.75,206.85] vol=1.8x ATR=0.71 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 204.87 | 205.22 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:05:00 | 202.95 | 203.67 | 0.00 | ORB-short ORB[203.06,205.80] vol=1.9x ATR=0.59 |
| Stop hit — per-position SL triggered | 2026-02-17 10:20:00 | 203.54 | 203.57 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 212.12 | 210.96 | 0.00 | ORB-long ORB[209.01,211.70] vol=1.8x ATR=0.67 |
| Stop hit — per-position SL triggered | 2026-04-16 09:55:00 | 211.45 | 211.37 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-20 11:00:00 | 209.96 | 210.99 | 0.00 | ORB-short ORB[210.00,212.43] vol=3.8x ATR=0.51 |
| Stop hit — per-position SL triggered | 2026-04-20 11:20:00 | 210.47 | 210.94 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:10:00 | 209.50 | 209.83 | 0.00 | ORB-short ORB[210.36,212.00] vol=1.6x ATR=0.43 |
| Stop hit — per-position SL triggered | 2026-04-24 11:15:00 | 209.93 | 209.83 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 11:10:00 | 214.54 | 213.79 | 0.00 | ORB-long ORB[210.93,214.00] vol=1.9x ATR=0.45 |
| Stop hit — per-position SL triggered | 2026-04-27 11:30:00 | 214.09 | 213.87 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:15:00 | 212.70 | 213.49 | 0.00 | ORB-short ORB[213.01,214.45] vol=1.8x ATR=0.34 |
| Stop hit — per-position SL triggered | 2026-05-06 11:50:00 | 213.04 | 213.40 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:40:00 | 218.58 | 217.17 | 0.00 | ORB-long ORB[215.75,217.13] vol=1.7x ATR=0.71 |
| Stop hit — per-position SL triggered | 2026-05-07 10:05:00 | 217.87 | 217.67 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:30:00 | 204.07 | 2026-02-10 09:35:00 | 205.20 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-02-10 09:30:00 | 204.07 | 2026-02-10 11:40:00 | 207.76 | TARGET_HIT | 0.50 | 1.81% |
| SELL | retest1 | 2026-02-11 11:05:00 | 205.85 | 2026-02-11 11:15:00 | 206.45 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-12 09:50:00 | 209.12 | 2026-02-12 10:15:00 | 208.50 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-13 09:30:00 | 204.16 | 2026-02-13 09:40:00 | 204.87 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-17 10:05:00 | 202.95 | 2026-02-17 10:20:00 | 203.54 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-16 09:30:00 | 212.12 | 2026-04-16 09:55:00 | 211.45 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-20 11:00:00 | 209.96 | 2026-04-20 11:20:00 | 210.47 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-04-24 11:10:00 | 209.50 | 2026-04-24 11:15:00 | 209.93 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-04-27 11:10:00 | 214.54 | 2026-04-27 11:30:00 | 214.09 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-05-06 11:15:00 | 212.70 | 2026-05-06 11:50:00 | 213.04 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2026-05-07 09:40:00 | 218.58 | 2026-05-07 10:05:00 | 217.87 | STOP_HIT | 1.00 | -0.32% |
