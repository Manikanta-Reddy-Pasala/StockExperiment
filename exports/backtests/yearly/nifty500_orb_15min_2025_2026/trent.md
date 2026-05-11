# Trent Ltd. (TRENT)

## Backtest Summary

- **Window:** 2026-04-06 09:15:00 → 2026-05-08 15:25:00 (1725 bars)
- **Last close:** 4249.10
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 8
- **Target hits / Stop hits / Partials:** 1 / 8 / 2
- **Avg / median % per leg:** -0.11% / -0.29%
- **Sum % (uncompounded):** -1.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 3 | 37.5% | 1 | 5 | 2 | -0.03% | -0.2% |
| BUY @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 1 | 5 | 2 | -0.03% | -0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.31% | -0.9% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.31% | -0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 3 | 27.3% | 1 | 8 | 2 | -0.11% | -1.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-04-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 09:35:00 | 3854.50 | 3812.76 | 0.00 | ORB-long ORB[3776.30,3828.60] vol=1.6x ATR=23.25 |
| Stop hit — per-position SL triggered | 2026-04-07 09:40:00 | 3831.25 | 3820.78 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-04-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:40:00 | 3926.30 | 3903.66 | 0.00 | ORB-long ORB[3871.30,3912.90] vol=3.4x ATR=14.56 |
| Stop hit — per-position SL triggered | 2026-04-10 10:45:00 | 3911.74 | 3903.94 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-04-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:40:00 | 4063.00 | 4002.02 | 0.00 | ORB-long ORB[4000.20,4034.80] vol=2.9x ATR=16.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:55:00 | 4088.41 | 4025.69 | 0.00 | T1 1.5R @ 4088.41 |
| Stop hit — per-position SL triggered | 2026-04-16 11:25:00 | 4063.00 | 4034.53 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:15:00 | 4162.90 | 4125.98 | 0.00 | ORB-long ORB[4090.00,4129.20] vol=1.6x ATR=16.33 |
| Stop hit — per-position SL triggered | 2026-04-17 10:50:00 | 4146.57 | 4131.88 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:10:00 | 4318.00 | 4278.06 | 0.00 | ORB-long ORB[4242.80,4278.70] vol=1.8x ATR=12.66 |
| Stop hit — per-position SL triggered | 2026-04-21 10:25:00 | 4305.34 | 4286.27 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:05:00 | 4294.60 | 4315.75 | 0.00 | ORB-short ORB[4297.30,4348.50] vol=1.5x ATR=17.30 |
| Stop hit — per-position SL triggered | 2026-04-27 10:30:00 | 4311.90 | 4313.83 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:40:00 | 4207.50 | 4234.70 | 0.00 | ORB-short ORB[4223.10,4269.90] vol=1.6x ATR=13.25 |
| Stop hit — per-position SL triggered | 2026-04-28 09:45:00 | 4220.75 | 4232.94 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 11:10:00 | 4260.00 | 4267.82 | 0.00 | ORB-short ORB[4262.40,4304.70] vol=2.3x ATR=9.40 |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 4269.40 | 4268.59 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 4223.00 | 4202.30 | 0.00 | ORB-long ORB[4156.50,4216.50] vol=1.8x ATR=11.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:45:00 | 4239.93 | 4218.05 | 0.00 | T1 1.5R @ 4239.93 |
| Target hit | 2026-05-06 11:55:00 | 4239.30 | 4239.82 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-04-07 09:35:00 | 3854.50 | 2026-04-07 09:40:00 | 3831.25 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2026-04-10 10:40:00 | 3926.30 | 2026-04-10 10:45:00 | 3911.74 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-16 10:40:00 | 4063.00 | 2026-04-16 10:55:00 | 4088.41 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-16 10:40:00 | 4063.00 | 2026-04-16 11:25:00 | 4063.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 10:15:00 | 4162.90 | 2026-04-17 10:50:00 | 4146.57 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-21 10:10:00 | 4318.00 | 2026-04-21 10:25:00 | 4305.34 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-27 10:05:00 | 4294.60 | 2026-04-27 10:30:00 | 4311.90 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-28 09:40:00 | 4207.50 | 2026-04-28 09:45:00 | 4220.75 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-29 11:10:00 | 4260.00 | 2026-04-29 11:15:00 | 4269.40 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-05-06 09:40:00 | 4223.00 | 2026-05-06 09:45:00 | 4239.93 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-05-06 09:40:00 | 4223.00 | 2026-05-06 11:55:00 | 4239.30 | TARGET_HIT | 0.50 | 0.39% |
