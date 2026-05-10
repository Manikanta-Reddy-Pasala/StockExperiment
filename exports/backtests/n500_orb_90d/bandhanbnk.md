# Bandhan Bank Ltd. (BANDHANBNK)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 206.25
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
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 8
- **Target hits / Stop hits / Partials:** 2 / 8 / 4
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 1.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.23% | 2.0% |
| BUY @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.23% | 2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.07% | -0.4% |
| SELL @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.07% | -0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.12% | 1.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:00:00 | 167.60 | 165.74 | 0.00 | ORB-long ORB[163.64,165.62] vol=1.6x ATR=0.49 |
| Stop hit — per-position SL triggered | 2026-02-16 11:10:00 | 167.11 | 165.78 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:20:00 | 168.80 | 167.83 | 0.00 | ORB-long ORB[166.81,168.24] vol=2.1x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:30:00 | 169.48 | 168.19 | 0.00 | T1 1.5R @ 169.48 |
| Stop hit — per-position SL triggered | 2026-02-17 10:40:00 | 168.80 | 168.27 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 169.88 | 168.91 | 0.00 | ORB-long ORB[167.59,169.22] vol=3.1x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:40:00 | 170.73 | 169.65 | 0.00 | T1 1.5R @ 170.73 |
| Target hit | 2026-02-18 15:20:00 | 171.41 | 170.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-02-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:45:00 | 172.02 | 171.37 | 0.00 | ORB-long ORB[170.56,171.75] vol=1.8x ATR=0.48 |
| Stop hit — per-position SL triggered | 2026-02-19 09:50:00 | 171.54 | 171.38 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:45:00 | 183.23 | 181.72 | 0.00 | ORB-long ORB[180.26,182.20] vol=1.6x ATR=0.82 |
| Stop hit — per-position SL triggered | 2026-03-11 09:55:00 | 182.41 | 181.82 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 10:45:00 | 143.40 | 145.30 | 0.00 | ORB-short ORB[145.22,146.83] vol=1.6x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 10:55:00 | 142.47 | 144.96 | 0.00 | T1 1.5R @ 142.47 |
| Stop hit — per-position SL triggered | 2026-03-30 11:35:00 | 143.40 | 144.47 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:25:00 | 172.13 | 170.57 | 0.00 | ORB-long ORB[169.10,171.49] vol=2.1x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:50:00 | 173.30 | 171.47 | 0.00 | T1 1.5R @ 173.30 |
| Target hit | 2026-04-15 13:50:00 | 173.10 | 173.16 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2026-04-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:45:00 | 169.15 | 171.26 | 0.00 | ORB-short ORB[172.10,174.33] vol=1.5x ATR=0.69 |
| Stop hit — per-position SL triggered | 2026-04-24 11:10:00 | 169.84 | 170.85 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-05-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:00:00 | 206.98 | 208.56 | 0.00 | ORB-short ORB[207.60,210.63] vol=1.8x ATR=0.65 |
| Stop hit — per-position SL triggered | 2026-05-06 11:05:00 | 207.63 | 208.54 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 11:10:00 | 204.05 | 205.45 | 0.00 | ORB-short ORB[205.06,207.95] vol=2.7x ATR=0.58 |
| Stop hit — per-position SL triggered | 2026-05-08 11:35:00 | 204.63 | 205.32 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-16 11:00:00 | 167.60 | 2026-02-16 11:10:00 | 167.11 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-17 10:20:00 | 168.80 | 2026-02-17 10:30:00 | 169.48 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-02-17 10:20:00 | 168.80 | 2026-02-17 10:40:00 | 168.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 09:30:00 | 169.88 | 2026-02-18 09:40:00 | 170.73 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-02-18 09:30:00 | 169.88 | 2026-02-18 15:20:00 | 171.41 | TARGET_HIT | 0.50 | 0.90% |
| BUY | retest1 | 2026-02-19 09:45:00 | 172.02 | 2026-02-19 09:50:00 | 171.54 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-03-11 09:45:00 | 183.23 | 2026-03-11 09:55:00 | 182.41 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-03-30 10:45:00 | 143.40 | 2026-03-30 10:55:00 | 142.47 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-03-30 10:45:00 | 143.40 | 2026-03-30 11:35:00 | 143.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-15 10:25:00 | 172.13 | 2026-04-15 11:50:00 | 173.30 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-04-15 10:25:00 | 172.13 | 2026-04-15 13:50:00 | 173.10 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2026-04-24 10:45:00 | 169.15 | 2026-04-24 11:10:00 | 169.84 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-05-06 11:00:00 | 206.98 | 2026-05-06 11:05:00 | 207.63 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-05-08 11:10:00 | 204.05 | 2026-05-08 11:35:00 | 204.63 | STOP_HIT | 1.00 | -0.28% |
