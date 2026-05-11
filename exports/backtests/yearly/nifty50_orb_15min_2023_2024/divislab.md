# DIVISLAB (DIVISLAB)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:25:00 (36946 bars)
- **Last close:** 6705.00
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 5
- **Target hits / Stop hits / Partials:** 2 / 5 / 3
- **Avg / median % per leg:** 0.07% / 0.13%
- **Sum % (uncompounded):** 0.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 2 | 2 | 2 | 0.15% | 0.9% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 2 | 2 | 2 | 0.15% | 0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.05% | -0.2% |
| SELL @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.05% | -0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.07% | 0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-04-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-08 10:30:00 | 3776.70 | 3751.66 | 0.00 | ORB-long ORB[3720.55,3751.75] vol=2.3x ATR=16.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 10:45:00 | 3801.32 | 3767.89 | 0.00 | T1 1.5R @ 3801.32 |
| Target hit | 2024-04-08 13:00:00 | 3788.75 | 3792.71 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2024-04-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 09:45:00 | 3808.95 | 3790.06 | 0.00 | ORB-long ORB[3772.05,3806.00] vol=2.0x ATR=10.72 |
| Stop hit — per-position SL triggered | 2024-04-09 10:25:00 | 3798.23 | 3798.20 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-04-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-18 09:35:00 | 3735.20 | 3754.95 | 0.00 | ORB-short ORB[3746.80,3781.10] vol=1.7x ATR=10.18 |
| Stop hit — per-position SL triggered | 2024-04-18 10:05:00 | 3745.38 | 3746.79 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 09:35:00 | 3797.00 | 3783.81 | 0.00 | ORB-long ORB[3765.55,3788.00] vol=1.6x ATR=9.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 10:00:00 | 3811.44 | 3793.49 | 0.00 | T1 1.5R @ 3811.44 |
| Target hit | 2024-04-24 10:50:00 | 3802.05 | 3802.64 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — SELL (started 2024-05-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-02 10:30:00 | 3952.50 | 3961.89 | 0.00 | ORB-short ORB[3957.00,4009.00] vol=5.3x ATR=11.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-02 10:45:00 | 3935.62 | 3959.44 | 0.00 | T1 1.5R @ 3935.62 |
| Stop hit — per-position SL triggered | 2024-05-02 11:30:00 | 3952.50 | 3957.65 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-06 09:30:00 | 3983.30 | 3958.73 | 0.00 | ORB-long ORB[3930.75,3969.95] vol=1.5x ATR=12.76 |
| Stop hit — per-position SL triggered | 2024-05-06 09:35:00 | 3970.54 | 3961.46 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 09:35:00 | 3866.40 | 3900.14 | 0.00 | ORB-short ORB[3895.00,3949.05] vol=1.8x ATR=14.04 |
| Stop hit — per-position SL triggered | 2024-05-09 09:45:00 | 3880.44 | 3892.59 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-04-08 10:30:00 | 3776.70 | 2024-04-08 10:45:00 | 3801.32 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-04-08 10:30:00 | 3776.70 | 2024-04-08 13:00:00 | 3788.75 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2024-04-09 09:45:00 | 3808.95 | 2024-04-09 10:25:00 | 3798.23 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-04-18 09:35:00 | 3735.20 | 2024-04-18 10:05:00 | 3745.38 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-04-24 09:35:00 | 3797.00 | 2024-04-24 10:00:00 | 3811.44 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-04-24 09:35:00 | 3797.00 | 2024-04-24 10:50:00 | 3802.05 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2024-05-02 10:30:00 | 3952.50 | 2024-05-02 10:45:00 | 3935.62 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-05-02 10:30:00 | 3952.50 | 2024-05-02 11:30:00 | 3952.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-06 09:30:00 | 3983.30 | 2024-05-06 09:35:00 | 3970.54 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-05-09 09:35:00 | 3866.40 | 2024-05-09 09:45:00 | 3880.44 | STOP_HIT | 1.00 | -0.36% |
