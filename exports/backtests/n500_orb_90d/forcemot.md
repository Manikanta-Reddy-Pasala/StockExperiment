# Force Motors Ltd. (FORCEMOT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 20851.00
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
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 3
- **Target hits / Stop hits / Partials:** 4 / 3 / 5
- **Avg / median % per leg:** 0.62% / 0.51%
- **Sum % (uncompounded):** 7.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 5 | 83.3% | 2 | 1 | 3 | 1.06% | 6.4% |
| BUY @ 2nd Alert (retest1) | 6 | 5 | 83.3% | 2 | 1 | 3 | 1.06% | 6.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 4 | 66.7% | 2 | 2 | 2 | 0.17% | 1.0% |
| SELL @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 2 | 2 | 2 | 0.17% | 1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 9 | 75.0% | 4 | 3 | 5 | 0.62% | 7.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 22651.00 | 22515.09 | 0.00 | ORB-long ORB[22359.00,22600.00] vol=2.0x ATR=90.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:35:00 | 22787.35 | 22632.39 | 0.00 | T1 1.5R @ 22787.35 |
| Target hit | 2026-02-10 15:20:00 | 23618.00 | 23305.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:40:00 | 23450.00 | 23762.57 | 0.00 | ORB-short ORB[23623.00,23975.00] vol=2.1x ATR=122.38 |
| Stop hit — per-position SL triggered | 2026-02-11 09:45:00 | 23572.38 | 23738.46 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 24788.00 | 24588.22 | 0.00 | ORB-long ORB[24304.00,24673.00] vol=1.6x ATR=136.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:55:00 | 24992.58 | 24735.73 | 0.00 | T1 1.5R @ 24992.58 |
| Target hit | 2026-02-24 13:30:00 | 24824.00 | 24845.68 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2026-02-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:05:00 | 25623.00 | 25475.79 | 0.00 | ORB-long ORB[25267.00,25536.00] vol=3.0x ATR=87.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:30:00 | 25753.54 | 25540.23 | 0.00 | T1 1.5R @ 25753.54 |
| Stop hit — per-position SL triggered | 2026-02-26 10:50:00 | 25623.00 | 25555.43 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:30:00 | 25143.00 | 25436.03 | 0.00 | ORB-short ORB[25401.00,25764.00] vol=3.3x ATR=100.71 |
| Stop hit — per-position SL triggered | 2026-02-27 09:35:00 | 25243.71 | 25404.13 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:55:00 | 22564.00 | 22645.75 | 0.00 | ORB-short ORB[22616.00,22805.00] vol=1.7x ATR=116.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:50:00 | 22389.40 | 22602.13 | 0.00 | T1 1.5R @ 22389.40 |
| Target hit | 2026-04-16 14:30:00 | 22537.00 | 22440.31 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — SELL (started 2026-04-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:30:00 | 22499.00 | 22679.33 | 0.00 | ORB-short ORB[22517.00,22788.00] vol=1.5x ATR=87.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 12:05:00 | 22368.49 | 22603.02 | 0.00 | T1 1.5R @ 22368.49 |
| Target hit | 2026-04-17 15:20:00 | 22390.00 | 22513.55 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:30:00 | 22651.00 | 2026-02-10 09:35:00 | 22787.35 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-02-10 09:30:00 | 22651.00 | 2026-02-10 15:20:00 | 23618.00 | TARGET_HIT | 0.50 | 4.27% |
| SELL | retest1 | 2026-02-11 09:40:00 | 23450.00 | 2026-02-11 09:45:00 | 23572.38 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-02-24 09:35:00 | 24788.00 | 2026-02-24 09:55:00 | 24992.58 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2026-02-24 09:35:00 | 24788.00 | 2026-02-24 13:30:00 | 24824.00 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2026-02-26 10:05:00 | 25623.00 | 2026-02-26 10:30:00 | 25753.54 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-02-26 10:05:00 | 25623.00 | 2026-02-26 10:50:00 | 25623.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 09:30:00 | 25143.00 | 2026-02-27 09:35:00 | 25243.71 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-16 09:55:00 | 22564.00 | 2026-04-16 10:50:00 | 22389.40 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2026-04-16 09:55:00 | 22564.00 | 2026-04-16 14:30:00 | 22537.00 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2026-04-17 10:30:00 | 22499.00 | 2026-04-17 12:05:00 | 22368.49 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-04-17 10:30:00 | 22499.00 | 2026-04-17 15:20:00 | 22390.00 | TARGET_HIT | 0.50 | 0.48% |
