# Larsen & Toubro Ltd. (LT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3978.00
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
| ENTRY1 | 16 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 5 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 11
- **Target hits / Stop hits / Partials:** 5 / 11 / 6
- **Avg / median % per leg:** 0.17% / 0.13%
- **Sum % (uncompounded):** 3.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 7 | 58.3% | 4 | 5 | 3 | 0.21% | 2.6% |
| BUY @ 2nd Alert (retest1) | 12 | 7 | 58.3% | 4 | 5 | 3 | 0.21% | 2.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.11% | 1.1% |
| SELL @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.11% | 1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 11 | 50.0% | 5 | 11 | 6 | 0.17% | 3.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:55:00 | 4106.70 | 4104.94 | 0.00 | ORB-long ORB[4062.40,4099.80] vol=3.3x ATR=13.56 |
| Target hit | 2026-02-09 15:20:00 | 4112.10 | 4108.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:40:00 | 4227.60 | 4210.69 | 0.00 | ORB-long ORB[4190.10,4220.00] vol=1.6x ATR=8.38 |
| Stop hit — per-position SL triggered | 2026-02-17 09:55:00 | 4219.22 | 4213.35 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:50:00 | 4303.10 | 4283.18 | 0.00 | ORB-long ORB[4261.80,4299.90] vol=2.9x ATR=8.32 |
| Stop hit — per-position SL triggered | 2026-02-18 10:55:00 | 4294.78 | 4283.84 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:10:00 | 4384.30 | 4405.54 | 0.00 | ORB-short ORB[4401.00,4440.00] vol=1.7x ATR=9.63 |
| Stop hit — per-position SL triggered | 2026-02-24 10:20:00 | 4393.93 | 4404.26 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:55:00 | 4337.10 | 4310.15 | 0.00 | ORB-long ORB[4254.20,4306.00] vol=2.8x ATR=11.45 |
| Stop hit — per-position SL triggered | 2026-02-25 11:30:00 | 4325.65 | 4312.25 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 11:05:00 | 4298.80 | 4282.35 | 0.00 | ORB-long ORB[4263.60,4298.00] vol=3.2x ATR=7.44 |
| Stop hit — per-position SL triggered | 2026-02-27 11:25:00 | 4291.36 | 4284.21 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 11:00:00 | 3465.00 | 3491.48 | 0.00 | ORB-short ORB[3471.20,3516.90] vol=1.9x ATR=10.27 |
| Stop hit — per-position SL triggered | 2026-03-20 11:20:00 | 3475.27 | 3485.92 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:45:00 | 3602.00 | 3575.90 | 0.00 | ORB-long ORB[3546.00,3575.00] vol=1.6x ATR=14.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 09:50:00 | 3623.73 | 3580.90 | 0.00 | T1 1.5R @ 3623.73 |
| Target hit | 2026-03-25 15:20:00 | 3650.00 | 3620.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2026-04-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 10:55:00 | 3615.70 | 3625.30 | 0.00 | ORB-short ORB[3621.00,3655.00] vol=1.8x ATR=11.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 11:45:00 | 3597.99 | 3623.71 | 0.00 | T1 1.5R @ 3597.99 |
| Stop hit — per-position SL triggered | 2026-04-01 12:00:00 | 3615.70 | 3623.28 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:50:00 | 3705.00 | 3696.67 | 0.00 | ORB-long ORB[3658.00,3699.50] vol=2.0x ATR=10.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 11:55:00 | 3721.48 | 3700.57 | 0.00 | T1 1.5R @ 3721.48 |
| Target hit | 2026-04-07 15:20:00 | 3722.00 | 3710.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2026-04-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:50:00 | 4104.00 | 4119.56 | 0.00 | ORB-short ORB[4105.00,4142.90] vol=2.7x ATR=8.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 12:15:00 | 4091.65 | 4112.06 | 0.00 | T1 1.5R @ 4091.65 |
| Stop hit — per-position SL triggered | 2026-04-17 14:45:00 | 4104.00 | 4104.87 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-20 10:55:00 | 4077.20 | 4099.07 | 0.00 | ORB-short ORB[4078.00,4130.00] vol=2.1x ATR=9.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 12:15:00 | 4062.83 | 4088.59 | 0.00 | T1 1.5R @ 4062.83 |
| Target hit | 2026-04-20 15:20:00 | 4041.20 | 4066.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2026-04-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 11:05:00 | 4034.80 | 4024.88 | 0.00 | ORB-long ORB[3978.50,4019.40] vol=1.9x ATR=7.88 |
| Stop hit — per-position SL triggered | 2026-04-23 11:20:00 | 4026.92 | 4025.20 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:10:00 | 4003.90 | 4026.23 | 0.00 | ORB-short ORB[4037.20,4088.10] vol=2.6x ATR=8.69 |
| Stop hit — per-position SL triggered | 2026-04-24 12:15:00 | 4012.59 | 4016.59 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:20:00 | 4088.00 | 4076.24 | 0.00 | ORB-long ORB[4051.30,4078.30] vol=2.3x ATR=8.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:15:00 | 4100.72 | 4082.30 | 0.00 | T1 1.5R @ 4100.72 |
| Target hit | 2026-04-29 15:05:00 | 4100.20 | 4100.42 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — SELL (started 2026-05-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:50:00 | 3975.70 | 3989.55 | 0.00 | ORB-short ORB[3987.90,4014.00] vol=1.8x ATR=6.52 |
| Stop hit — per-position SL triggered | 2026-05-08 11:05:00 | 3982.22 | 3988.76 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:55:00 | 4106.70 | 2026-02-09 15:20:00 | 4112.10 | TARGET_HIT | 1.00 | 0.13% |
| BUY | retest1 | 2026-02-17 09:40:00 | 4227.60 | 2026-02-17 09:55:00 | 4219.22 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-18 10:50:00 | 4303.10 | 2026-02-18 10:55:00 | 4294.78 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-24 10:10:00 | 4384.30 | 2026-02-24 10:20:00 | 4393.93 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-25 10:55:00 | 4337.10 | 2026-02-25 11:30:00 | 4325.65 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-27 11:05:00 | 4298.80 | 2026-02-27 11:25:00 | 4291.36 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-03-20 11:00:00 | 3465.00 | 2026-03-20 11:20:00 | 3475.27 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-25 09:45:00 | 3602.00 | 2026-03-25 09:50:00 | 3623.73 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-03-25 09:45:00 | 3602.00 | 2026-03-25 15:20:00 | 3650.00 | TARGET_HIT | 0.50 | 1.33% |
| SELL | retest1 | 2026-04-01 10:55:00 | 3615.70 | 2026-04-01 11:45:00 | 3597.99 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-04-01 10:55:00 | 3615.70 | 2026-04-01 12:00:00 | 3615.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-07 10:50:00 | 3705.00 | 2026-04-07 11:55:00 | 3721.48 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-04-07 10:50:00 | 3705.00 | 2026-04-07 15:20:00 | 3722.00 | TARGET_HIT | 0.50 | 0.46% |
| SELL | retest1 | 2026-04-17 10:50:00 | 4104.00 | 2026-04-17 12:15:00 | 4091.65 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-04-17 10:50:00 | 4104.00 | 2026-04-17 14:45:00 | 4104.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-20 10:55:00 | 4077.20 | 2026-04-20 12:15:00 | 4062.83 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-04-20 10:55:00 | 4077.20 | 2026-04-20 15:20:00 | 4041.20 | TARGET_HIT | 0.50 | 0.88% |
| BUY | retest1 | 2026-04-23 11:05:00 | 4034.80 | 2026-04-23 11:20:00 | 4026.92 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-04-24 11:10:00 | 4003.90 | 2026-04-24 12:15:00 | 4012.59 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-04-29 10:20:00 | 4088.00 | 2026-04-29 11:15:00 | 4100.72 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2026-04-29 10:20:00 | 4088.00 | 2026-04-29 15:05:00 | 4100.20 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2026-05-08 10:50:00 | 3975.70 | 2026-05-08 11:05:00 | 3982.22 | STOP_HIT | 1.00 | -0.16% |
