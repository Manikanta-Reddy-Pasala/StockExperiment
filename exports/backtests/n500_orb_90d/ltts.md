# L&T Technology Services Ltd. (LTTS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3801.60
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
- **Avg / median % per leg:** 0.51% / 0.00%
- **Sum % (uncompounded):** 7.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.75% | 3.7% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.75% | 3.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.38% | 3.5% |
| SELL @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.38% | 3.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.51% | 7.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:35:00 | 3485.00 | 3494.70 | 0.00 | ORB-short ORB[3486.20,3519.00] vol=1.7x ATR=10.54 |
| Stop hit — per-position SL triggered | 2026-02-19 09:45:00 | 3495.54 | 3494.10 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:35:00 | 3387.60 | 3406.39 | 0.00 | ORB-short ORB[3402.70,3441.10] vol=1.9x ATR=12.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:40:00 | 3369.07 | 3390.67 | 0.00 | T1 1.5R @ 3369.07 |
| Stop hit — per-position SL triggered | 2026-02-23 12:20:00 | 3387.60 | 3386.93 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 10:10:00 | 3361.00 | 3344.14 | 0.00 | ORB-long ORB[3312.70,3360.00] vol=1.6x ATR=13.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 12:10:00 | 3381.61 | 3356.84 | 0.00 | T1 1.5R @ 3381.61 |
| Target hit | 2026-02-27 15:20:00 | 3504.90 | 3445.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:30:00 | 3279.00 | 3300.83 | 0.00 | ORB-short ORB[3297.50,3324.80] vol=3.9x ATR=12.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:15:00 | 3259.78 | 3285.91 | 0.00 | T1 1.5R @ 3259.78 |
| Target hit | 2026-03-06 15:20:00 | 3182.00 | 3240.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-04-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-20 09:40:00 | 3448.70 | 3478.92 | 0.00 | ORB-short ORB[3465.00,3511.10] vol=1.6x ATR=11.74 |
| Stop hit — per-position SL triggered | 2026-04-20 09:55:00 | 3460.44 | 3471.59 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:50:00 | 3635.50 | 3604.82 | 0.00 | ORB-long ORB[3583.50,3621.10] vol=2.4x ATR=15.11 |
| Stop hit — per-position SL triggered | 2026-04-21 11:20:00 | 3620.39 | 3607.92 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:35:00 | 3502.60 | 3483.06 | 0.00 | ORB-long ORB[3458.00,3499.90] vol=1.9x ATR=15.30 |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 3487.30 | 3488.32 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 11:00:00 | 3490.10 | 3529.16 | 0.00 | ORB-short ORB[3513.00,3559.60] vol=5.5x ATR=12.78 |
| Stop hit — per-position SL triggered | 2026-04-30 12:05:00 | 3502.88 | 3513.92 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 3735.40 | 3723.63 | 0.00 | ORB-long ORB[3687.40,3732.60] vol=2.0x ATR=11.56 |
| Stop hit — per-position SL triggered | 2026-05-05 09:35:00 | 3723.84 | 3724.14 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:05:00 | 3775.90 | 3797.77 | 0.00 | ORB-short ORB[3790.80,3843.20] vol=1.7x ATR=9.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 12:30:00 | 3761.53 | 3789.30 | 0.00 | T1 1.5R @ 3761.53 |
| Stop hit — per-position SL triggered | 2026-05-07 13:05:00 | 3775.90 | 3787.17 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-19 09:35:00 | 3485.00 | 2026-02-19 09:45:00 | 3495.54 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-23 09:35:00 | 3387.60 | 2026-02-23 10:40:00 | 3369.07 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-02-23 09:35:00 | 3387.60 | 2026-02-23 12:20:00 | 3387.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-27 10:10:00 | 3361.00 | 2026-02-27 12:10:00 | 3381.61 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-02-27 10:10:00 | 3361.00 | 2026-02-27 15:20:00 | 3504.90 | TARGET_HIT | 0.50 | 4.28% |
| SELL | retest1 | 2026-03-06 09:30:00 | 3279.00 | 2026-03-06 10:15:00 | 3259.78 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-03-06 09:30:00 | 3279.00 | 2026-03-06 15:20:00 | 3182.00 | TARGET_HIT | 0.50 | 2.96% |
| SELL | retest1 | 2026-04-20 09:40:00 | 3448.70 | 2026-04-20 09:55:00 | 3460.44 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-21 10:50:00 | 3635.50 | 2026-04-21 11:20:00 | 3620.39 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-23 09:35:00 | 3502.60 | 2026-04-23 10:15:00 | 3487.30 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-04-30 11:00:00 | 3490.10 | 2026-04-30 12:05:00 | 3502.88 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-05-05 09:30:00 | 3735.40 | 2026-05-05 09:35:00 | 3723.84 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-05-07 11:05:00 | 3775.90 | 2026-05-07 12:30:00 | 3761.53 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-05-07 11:05:00 | 3775.90 | 2026-05-07 13:05:00 | 3775.90 | STOP_HIT | 0.50 | 0.00% |
