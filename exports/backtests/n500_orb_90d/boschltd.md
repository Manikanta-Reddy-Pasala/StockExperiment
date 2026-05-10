# Bosch Ltd. (BOSCHLTD)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 38050.00
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 15
- **Target hits / Stop hits / Partials:** 3 / 15 / 5
- **Avg / median % per leg:** 0.08% / -0.21%
- **Sum % (uncompounded):** 1.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 5 | 38.5% | 2 | 8 | 3 | 0.16% | 2.0% |
| BUY @ 2nd Alert (retest1) | 13 | 5 | 38.5% | 2 | 8 | 3 | 0.16% | 2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 3 | 30.0% | 1 | 7 | 2 | -0.02% | -0.2% |
| SELL @ 2nd Alert (retest1) | 10 | 3 | 30.0% | 1 | 7 | 2 | -0.02% | -0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 8 | 34.8% | 3 | 15 | 5 | 0.08% | 1.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:45:00 | 35740.00 | 35510.91 | 0.00 | ORB-long ORB[35200.00,35670.00] vol=2.4x ATR=124.04 |
| Stop hit — per-position SL triggered | 2026-02-10 13:15:00 | 35615.96 | 35673.21 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 35835.00 | 36042.30 | 0.00 | ORB-short ORB[35925.00,36420.00] vol=1.9x ATR=109.57 |
| Stop hit — per-position SL triggered | 2026-02-13 09:35:00 | 35944.57 | 36030.24 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:25:00 | 35075.00 | 34922.01 | 0.00 | ORB-long ORB[34800.00,35065.00] vol=4.0x ATR=110.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:50:00 | 35240.31 | 34957.64 | 0.00 | T1 1.5R @ 35240.31 |
| Target hit | 2026-02-20 14:25:00 | 35210.00 | 35210.58 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2026-02-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:40:00 | 35960.00 | 35782.11 | 0.00 | ORB-long ORB[35430.00,35695.00] vol=4.8x ATR=110.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:25:00 | 36125.19 | 35923.43 | 0.00 | T1 1.5R @ 36125.19 |
| Target hit | 2026-02-25 12:05:00 | 36830.00 | 36861.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:15:00 | 36350.00 | 36470.07 | 0.00 | ORB-short ORB[36435.00,36745.00] vol=1.5x ATR=79.46 |
| Stop hit — per-position SL triggered | 2026-02-26 11:20:00 | 36429.46 | 36470.67 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:10:00 | 36055.00 | 36283.55 | 0.00 | ORB-short ORB[36350.00,36835.00] vol=2.8x ATR=79.64 |
| Stop hit — per-position SL triggered | 2026-02-27 12:00:00 | 36134.64 | 36238.42 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:15:00 | 33300.00 | 33432.20 | 0.00 | ORB-short ORB[33450.00,33715.00] vol=1.5x ATR=71.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:35:00 | 33192.22 | 33414.37 | 0.00 | T1 1.5R @ 33192.22 |
| Stop hit — per-position SL triggered | 2026-03-05 14:40:00 | 33300.00 | 33295.43 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:10:00 | 31935.00 | 32145.42 | 0.00 | ORB-short ORB[32065.00,32430.00] vol=2.1x ATR=88.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:55:00 | 31802.37 | 32097.51 | 0.00 | T1 1.5R @ 31802.37 |
| Target hit | 2026-03-11 15:20:00 | 31695.00 | 31926.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-03-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:55:00 | 30825.00 | 30606.30 | 0.00 | ORB-long ORB[30390.00,30790.00] vol=2.4x ATR=113.09 |
| Stop hit — per-position SL triggered | 2026-03-17 10:30:00 | 30711.91 | 30681.08 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:10:00 | 30625.00 | 30511.25 | 0.00 | ORB-long ORB[30300.00,30595.00] vol=2.0x ATR=99.83 |
| Stop hit — per-position SL triggered | 2026-03-20 10:15:00 | 30525.17 | 30512.90 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 09:50:00 | 36915.00 | 37107.40 | 0.00 | ORB-short ORB[36960.00,37350.00] vol=2.1x ATR=172.02 |
| Stop hit — per-position SL triggered | 2026-04-10 10:00:00 | 37087.02 | 37107.27 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:45:00 | 38060.00 | 37919.77 | 0.00 | ORB-long ORB[37790.00,37980.00] vol=1.9x ATR=81.14 |
| Stop hit — per-position SL triggered | 2026-04-21 09:50:00 | 37978.86 | 37930.12 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 38295.00 | 38140.95 | 0.00 | ORB-long ORB[37975.00,38280.00] vol=1.6x ATR=91.96 |
| Stop hit — per-position SL triggered | 2026-04-22 10:00:00 | 38203.04 | 38214.76 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 11:05:00 | 37155.00 | 37072.49 | 0.00 | ORB-long ORB[36730.00,37050.00] vol=1.9x ATR=113.80 |
| Stop hit — per-position SL triggered | 2026-04-27 11:35:00 | 37041.20 | 37079.37 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:00:00 | 37145.00 | 37220.92 | 0.00 | ORB-short ORB[37165.00,37470.00] vol=1.8x ATR=109.67 |
| Stop hit — per-position SL triggered | 2026-04-29 10:30:00 | 37254.67 | 37203.11 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-05-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:35:00 | 35915.00 | 36023.23 | 0.00 | ORB-short ORB[35925.00,36150.00] vol=2.1x ATR=70.40 |
| Stop hit — per-position SL triggered | 2026-05-06 10:40:00 | 35985.40 | 36022.45 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:45:00 | 37600.00 | 37190.94 | 0.00 | ORB-long ORB[36645.00,37055.00] vol=1.5x ATR=146.80 |
| Stop hit — per-position SL triggered | 2026-05-07 09:50:00 | 37453.20 | 37206.07 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-05-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:45:00 | 38465.00 | 38004.30 | 0.00 | ORB-long ORB[37750.00,38090.00] vol=3.2x ATR=121.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:50:00 | 38646.97 | 38174.18 | 0.00 | T1 1.5R @ 38646.97 |
| Stop hit — per-position SL triggered | 2026-05-08 11:45:00 | 38465.00 | 38288.30 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:45:00 | 35740.00 | 2026-02-10 13:15:00 | 35615.96 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-13 09:30:00 | 35835.00 | 2026-02-13 09:35:00 | 35944.57 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-20 10:25:00 | 35075.00 | 2026-02-20 10:50:00 | 35240.31 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-02-20 10:25:00 | 35075.00 | 2026-02-20 14:25:00 | 35210.00 | TARGET_HIT | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-25 09:40:00 | 35960.00 | 2026-02-25 10:25:00 | 36125.19 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-25 09:40:00 | 35960.00 | 2026-02-25 12:05:00 | 36830.00 | TARGET_HIT | 0.50 | 2.42% |
| SELL | retest1 | 2026-02-26 11:15:00 | 36350.00 | 2026-02-26 11:20:00 | 36429.46 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-27 11:10:00 | 36055.00 | 2026-02-27 12:00:00 | 36134.64 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-03-05 11:15:00 | 33300.00 | 2026-03-05 11:35:00 | 33192.22 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-03-05 11:15:00 | 33300.00 | 2026-03-05 14:40:00 | 33300.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 11:10:00 | 31935.00 | 2026-03-11 11:55:00 | 31802.37 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-03-11 11:10:00 | 31935.00 | 2026-03-11 15:20:00 | 31695.00 | TARGET_HIT | 0.50 | 0.75% |
| BUY | retest1 | 2026-03-17 09:55:00 | 30825.00 | 2026-03-17 10:30:00 | 30711.91 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-20 10:10:00 | 30625.00 | 2026-03-20 10:15:00 | 30525.17 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-10 09:50:00 | 36915.00 | 2026-04-10 10:00:00 | 37087.02 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-04-21 09:45:00 | 38060.00 | 2026-04-21 09:50:00 | 37978.86 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-04-22 09:30:00 | 38295.00 | 2026-04-22 10:00:00 | 38203.04 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-04-27 11:05:00 | 37155.00 | 2026-04-27 11:35:00 | 37041.20 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-29 10:00:00 | 37145.00 | 2026-04-29 10:30:00 | 37254.67 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-05-06 10:35:00 | 35915.00 | 2026-05-06 10:40:00 | 35985.40 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-05-07 09:45:00 | 37600.00 | 2026-05-07 09:50:00 | 37453.20 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-05-08 10:45:00 | 38465.00 | 2026-05-08 10:50:00 | 38646.97 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-05-08 10:45:00 | 38465.00 | 2026-05-08 11:45:00 | 38465.00 | STOP_HIT | 0.50 | 0.00% |
