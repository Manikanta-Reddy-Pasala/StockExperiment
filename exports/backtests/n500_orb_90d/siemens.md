# Siemens Ltd. (SIEMENS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3838.00
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
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 8
- **Target hits / Stop hits / Partials:** 1 / 8 / 3
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 1.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.25% | 1.5% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.25% | 1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.04% | 0.2% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.04% | 0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 4 | 33.3% | 1 | 8 | 3 | 0.14% | 1.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:40:00 | 3142.00 | 3126.65 | 0.00 | ORB-long ORB[3116.00,3138.20] vol=2.5x ATR=7.95 |
| Stop hit — per-position SL triggered | 2026-02-11 11:00:00 | 3134.05 | 3129.21 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:05:00 | 3213.80 | 3198.02 | 0.00 | ORB-long ORB[3173.10,3206.50] vol=2.0x ATR=9.70 |
| Stop hit — per-position SL triggered | 2026-02-19 10:10:00 | 3204.10 | 3198.73 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 3327.60 | 3308.96 | 0.00 | ORB-long ORB[3274.20,3314.30] vol=2.8x ATR=12.22 |
| Stop hit — per-position SL triggered | 2026-02-24 09:55:00 | 3315.38 | 3316.56 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:45:00 | 3278.10 | 3308.37 | 0.00 | ORB-short ORB[3302.40,3336.90] vol=1.8x ATR=10.69 |
| Stop hit — per-position SL triggered | 2026-02-27 10:10:00 | 3288.79 | 3296.60 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:45:00 | 3277.90 | 3251.08 | 0.00 | ORB-long ORB[3219.60,3259.00] vol=1.5x ATR=15.09 |
| Stop hit — per-position SL triggered | 2026-03-12 10:05:00 | 3262.81 | 3254.43 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:50:00 | 3625.00 | 3602.44 | 0.00 | ORB-long ORB[3565.00,3606.00] vol=2.0x ATR=15.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:00:00 | 3647.89 | 3621.20 | 0.00 | T1 1.5R @ 3647.89 |
| Target hit | 2026-04-17 15:20:00 | 3706.90 | 3667.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-04-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:10:00 | 3774.50 | 3833.33 | 0.00 | ORB-short ORB[3845.10,3893.60] vol=2.1x ATR=13.56 |
| Stop hit — per-position SL triggered | 2026-04-24 11:20:00 | 3788.06 | 3831.59 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-05-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:10:00 | 3826.70 | 3847.61 | 0.00 | ORB-short ORB[3842.00,3875.80] vol=1.7x ATR=11.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:15:00 | 3809.47 | 3843.13 | 0.00 | T1 1.5R @ 3809.47 |
| Stop hit — per-position SL triggered | 2026-05-07 11:45:00 | 3826.70 | 3838.43 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-05-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 11:05:00 | 3821.80 | 3849.77 | 0.00 | ORB-short ORB[3853.30,3906.90] vol=1.5x ATR=11.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 12:30:00 | 3804.43 | 3840.00 | 0.00 | T1 1.5R @ 3804.43 |
| Stop hit — per-position SL triggered | 2026-05-08 14:40:00 | 3821.80 | 3822.71 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 10:40:00 | 3142.00 | 2026-02-11 11:00:00 | 3134.05 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-19 10:05:00 | 3213.80 | 2026-02-19 10:10:00 | 3204.10 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-24 09:30:00 | 3327.60 | 2026-02-24 09:55:00 | 3315.38 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-27 09:45:00 | 3278.10 | 2026-02-27 10:10:00 | 3288.79 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-03-12 09:45:00 | 3277.90 | 2026-03-12 10:05:00 | 3262.81 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-04-17 09:50:00 | 3625.00 | 2026-04-17 10:00:00 | 3647.89 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-17 09:50:00 | 3625.00 | 2026-04-17 15:20:00 | 3706.90 | TARGET_HIT | 0.50 | 2.26% |
| SELL | retest1 | 2026-04-24 11:10:00 | 3774.50 | 2026-04-24 11:20:00 | 3788.06 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-05-07 11:10:00 | 3826.70 | 2026-05-07 11:15:00 | 3809.47 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-05-07 11:10:00 | 3826.70 | 2026-05-07 11:45:00 | 3826.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 11:05:00 | 3821.80 | 2026-05-08 12:30:00 | 3804.43 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-05-08 11:05:00 | 3821.80 | 2026-05-08 14:40:00 | 3821.80 | STOP_HIT | 0.50 | 0.00% |
