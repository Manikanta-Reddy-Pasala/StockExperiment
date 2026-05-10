# Hitachi Energy India Ltd. (POWERINDIA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 33960.00
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
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 8
- **Target hits / Stop hits / Partials:** 3 / 8 / 4
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 2.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 6 | 50.0% | 3 | 6 | 3 | 0.16% | 1.9% |
| BUY @ 2nd Alert (retest1) | 12 | 6 | 50.0% | 3 | 6 | 3 | 0.16% | 1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.13% | 0.4% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.13% | 0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 7 | 46.7% | 3 | 8 | 4 | 0.15% | 2.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:30:00 | 22908.00 | 22762.07 | 0.00 | ORB-long ORB[22600.00,22823.00] vol=1.5x ATR=75.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 09:40:00 | 23020.82 | 22867.18 | 0.00 | T1 1.5R @ 23020.82 |
| Target hit | 2026-02-12 10:05:00 | 22930.00 | 22938.15 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2026-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:30:00 | 22749.00 | 22598.08 | 0.00 | ORB-long ORB[22350.00,22670.00] vol=1.9x ATR=81.30 |
| Stop hit — per-position SL triggered | 2026-02-16 09:40:00 | 22667.70 | 22616.95 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:40:00 | 24259.00 | 24116.49 | 0.00 | ORB-long ORB[23827.00,24170.00] vol=1.8x ATR=96.29 |
| Stop hit — per-position SL triggered | 2026-02-23 10:00:00 | 24162.71 | 24170.72 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:30:00 | 24472.00 | 24222.93 | 0.00 | ORB-long ORB[24000.00,24245.00] vol=2.3x ATR=73.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:30:00 | 24582.08 | 24315.00 | 0.00 | T1 1.5R @ 24582.08 |
| Target hit | 2026-02-24 15:20:00 | 24951.00 | 24604.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-02-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:35:00 | 25240.00 | 25056.47 | 0.00 | ORB-long ORB[24881.00,25179.00] vol=1.6x ATR=101.31 |
| Stop hit — per-position SL triggered | 2026-02-25 09:50:00 | 25138.69 | 25088.71 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 09:35:00 | 25550.00 | 25723.27 | 0.00 | ORB-short ORB[25600.00,25940.00] vol=1.7x ATR=128.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 09:50:00 | 25356.98 | 25659.12 | 0.00 | T1 1.5R @ 25356.98 |
| Stop hit — per-position SL triggered | 2026-04-08 10:15:00 | 25550.00 | 25569.92 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:15:00 | 28710.00 | 28559.54 | 0.00 | ORB-long ORB[28180.00,28500.00] vol=1.6x ATR=81.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 14:30:00 | 28831.84 | 28613.62 | 0.00 | T1 1.5R @ 28831.84 |
| Target hit | 2026-04-17 15:20:00 | 28930.00 | 28683.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-04-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:40:00 | 31965.00 | 32243.67 | 0.00 | ORB-short ORB[32085.00,32550.00] vol=3.2x ATR=117.65 |
| Stop hit — per-position SL triggered | 2026-04-27 11:40:00 | 32082.65 | 32163.97 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:25:00 | 32990.00 | 32789.67 | 0.00 | ORB-long ORB[32655.00,32925.00] vol=3.4x ATR=117.25 |
| Stop hit — per-position SL triggered | 2026-04-29 10:30:00 | 32872.75 | 32805.11 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 10:00:00 | 33210.00 | 32908.86 | 0.00 | ORB-long ORB[32530.00,33025.00] vol=1.5x ATR=134.74 |
| Stop hit — per-position SL triggered | 2026-04-30 10:10:00 | 33075.26 | 32943.36 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 34170.00 | 34005.00 | 0.00 | ORB-long ORB[33720.00,34090.00] vol=1.9x ATR=125.23 |
| Stop hit — per-position SL triggered | 2026-05-05 09:50:00 | 34044.77 | 34047.62 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-12 09:30:00 | 22908.00 | 2026-02-12 09:40:00 | 23020.82 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-12 09:30:00 | 22908.00 | 2026-02-12 10:05:00 | 22930.00 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2026-02-16 09:30:00 | 22749.00 | 2026-02-16 09:40:00 | 22667.70 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-23 09:40:00 | 24259.00 | 2026-02-23 10:00:00 | 24162.71 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-02-24 10:30:00 | 24472.00 | 2026-02-24 11:30:00 | 24582.08 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-24 10:30:00 | 24472.00 | 2026-02-24 15:20:00 | 24951.00 | TARGET_HIT | 0.50 | 1.96% |
| BUY | retest1 | 2026-02-25 09:35:00 | 25240.00 | 2026-02-25 09:50:00 | 25138.69 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-08 09:35:00 | 25550.00 | 2026-04-08 09:50:00 | 25356.98 | PARTIAL | 0.50 | 0.76% |
| SELL | retest1 | 2026-04-08 09:35:00 | 25550.00 | 2026-04-08 10:15:00 | 25550.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 11:15:00 | 28710.00 | 2026-04-17 14:30:00 | 28831.84 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-04-17 11:15:00 | 28710.00 | 2026-04-17 15:20:00 | 28930.00 | TARGET_HIT | 0.50 | 0.77% |
| SELL | retest1 | 2026-04-27 10:40:00 | 31965.00 | 2026-04-27 11:40:00 | 32082.65 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-29 10:25:00 | 32990.00 | 2026-04-29 10:30:00 | 32872.75 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-30 10:00:00 | 33210.00 | 2026-04-30 10:10:00 | 33075.26 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-05-05 09:30:00 | 34170.00 | 2026-05-05 09:50:00 | 34044.77 | STOP_HIT | 1.00 | -0.37% |
