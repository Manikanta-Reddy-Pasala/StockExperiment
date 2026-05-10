# Alkem Laboratories Ltd. (ALKEM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 5560.00
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
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 16
- **Target hits / Stop hits / Partials:** 2 / 16 / 6
- **Avg / median % per leg:** 0.23% / 0.00%
- **Sum % (uncompounded):** 5.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 5 | 31.2% | 1 | 11 | 4 | 0.05% | 0.7% |
| BUY @ 2nd Alert (retest1) | 16 | 5 | 31.2% | 1 | 11 | 4 | 0.05% | 0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.60% | 4.8% |
| SELL @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.60% | 4.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 24 | 8 | 33.3% | 2 | 16 | 6 | 0.23% | 5.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:30:00 | 5766.50 | 5731.60 | 0.00 | ORB-long ORB[5670.00,5741.00] vol=2.0x ATR=24.34 |
| Stop hit — per-position SL triggered | 2026-02-09 11:20:00 | 5742.16 | 5743.26 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:55:00 | 5878.50 | 5811.10 | 0.00 | ORB-long ORB[5775.50,5840.00] vol=1.7x ATR=18.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:20:00 | 5906.68 | 5824.33 | 0.00 | T1 1.5R @ 5906.68 |
| Stop hit — per-position SL triggered | 2026-02-11 11:45:00 | 5878.50 | 5837.15 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 11:05:00 | 5767.00 | 5791.73 | 0.00 | ORB-short ORB[5783.00,5850.00] vol=1.6x ATR=17.23 |
| Stop hit — per-position SL triggered | 2026-02-13 11:50:00 | 5784.23 | 5784.90 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:15:00 | 5410.00 | 5441.68 | 0.00 | ORB-short ORB[5429.00,5483.50] vol=2.0x ATR=11.59 |
| Stop hit — per-position SL triggered | 2026-02-18 11:50:00 | 5421.59 | 5440.12 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:40:00 | 5530.50 | 5515.74 | 0.00 | ORB-long ORB[5468.00,5509.50] vol=2.0x ATR=13.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:20:00 | 5551.44 | 5529.41 | 0.00 | T1 1.5R @ 5551.44 |
| Stop hit — per-position SL triggered | 2026-02-25 12:00:00 | 5530.50 | 5534.48 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:45:00 | 5749.50 | 5689.09 | 0.00 | ORB-long ORB[5622.50,5650.00] vol=1.5x ATR=20.94 |
| Stop hit — per-position SL triggered | 2026-02-26 10:00:00 | 5728.56 | 5700.27 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 09:35:00 | 5525.50 | 5512.94 | 0.00 | ORB-long ORB[5476.50,5521.50] vol=1.5x ATR=16.46 |
| Stop hit — per-position SL triggered | 2026-03-05 10:10:00 | 5509.04 | 5518.21 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 5518.00 | 5544.14 | 0.00 | ORB-short ORB[5548.00,5583.00] vol=2.5x ATR=12.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 11:10:00 | 5499.02 | 5541.08 | 0.00 | T1 1.5R @ 5499.02 |
| Stop hit — per-position SL triggered | 2026-03-06 11:30:00 | 5518.00 | 5539.23 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 09:45:00 | 5595.50 | 5573.45 | 0.00 | ORB-long ORB[5500.00,5577.50] vol=3.9x ATR=18.68 |
| Stop hit — per-position SL triggered | 2026-03-10 09:55:00 | 5576.82 | 5578.34 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 10:25:00 | 5240.00 | 5247.12 | 0.00 | ORB-short ORB[5241.00,5298.50] vol=1.8x ATR=16.94 |
| Stop hit — per-position SL triggered | 2026-03-20 10:55:00 | 5256.94 | 5247.13 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 11:05:00 | 5421.00 | 5358.85 | 0.00 | ORB-long ORB[5311.50,5355.00] vol=2.1x ATR=12.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 11:40:00 | 5440.02 | 5380.73 | 0.00 | T1 1.5R @ 5440.02 |
| Stop hit — per-position SL triggered | 2026-03-25 14:30:00 | 5421.00 | 5403.61 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:15:00 | 5586.50 | 5550.08 | 0.00 | ORB-long ORB[5501.50,5570.00] vol=1.9x ATR=16.09 |
| Stop hit — per-position SL triggered | 2026-04-17 10:30:00 | 5570.41 | 5551.93 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:10:00 | 5704.00 | 5673.42 | 0.00 | ORB-long ORB[5627.00,5664.00] vol=3.0x ATR=14.24 |
| Stop hit — per-position SL triggered | 2026-04-21 10:40:00 | 5689.76 | 5693.07 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:50:00 | 5493.50 | 5531.84 | 0.00 | ORB-short ORB[5515.50,5578.50] vol=1.9x ATR=18.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:45:00 | 5465.87 | 5523.94 | 0.00 | T1 1.5R @ 5465.87 |
| Target hit | 2026-04-24 15:20:00 | 5218.00 | 5356.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2026-04-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 11:10:00 | 5404.00 | 5366.15 | 0.00 | ORB-long ORB[5315.00,5370.00] vol=2.3x ATR=11.59 |
| Stop hit — per-position SL triggered | 2026-04-28 11:20:00 | 5392.41 | 5368.92 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 11:10:00 | 5351.50 | 5399.11 | 0.00 | ORB-short ORB[5399.00,5453.50] vol=3.2x ATR=13.99 |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 5365.49 | 5397.39 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:55:00 | 5480.00 | 5451.74 | 0.00 | ORB-long ORB[5404.00,5467.50] vol=2.0x ATR=15.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:00:00 | 5503.22 | 5457.58 | 0.00 | T1 1.5R @ 5503.22 |
| Target hit | 2026-05-06 15:05:00 | 5568.00 | 5577.77 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — BUY (started 2026-05-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:50:00 | 5655.00 | 5610.11 | 0.00 | ORB-long ORB[5561.00,5645.00] vol=3.4x ATR=19.51 |
| Stop hit — per-position SL triggered | 2026-05-08 11:20:00 | 5635.49 | 5618.81 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:30:00 | 5766.50 | 2026-02-09 11:20:00 | 5742.16 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-02-11 10:55:00 | 5878.50 | 2026-02-11 11:20:00 | 5906.68 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-02-11 10:55:00 | 5878.50 | 2026-02-11 11:45:00 | 5878.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 11:05:00 | 5767.00 | 2026-02-13 11:50:00 | 5784.23 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-18 11:15:00 | 5410.00 | 2026-02-18 11:50:00 | 5421.59 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-02-25 10:40:00 | 5530.50 | 2026-02-25 11:20:00 | 5551.44 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-25 10:40:00 | 5530.50 | 2026-02-25 12:00:00 | 5530.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 09:45:00 | 5749.50 | 2026-02-26 10:00:00 | 5728.56 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-05 09:35:00 | 5525.50 | 2026-03-05 10:10:00 | 5509.04 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-06 10:45:00 | 5518.00 | 2026-03-06 11:10:00 | 5499.02 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-03-06 10:45:00 | 5518.00 | 2026-03-06 11:30:00 | 5518.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-10 09:45:00 | 5595.50 | 2026-03-10 09:55:00 | 5576.82 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-03-20 10:25:00 | 5240.00 | 2026-03-20 10:55:00 | 5256.94 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-25 11:05:00 | 5421.00 | 2026-03-25 11:40:00 | 5440.02 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-03-25 11:05:00 | 5421.00 | 2026-03-25 14:30:00 | 5421.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 10:15:00 | 5586.50 | 2026-04-17 10:30:00 | 5570.41 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-21 10:10:00 | 5704.00 | 2026-04-21 10:40:00 | 5689.76 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-04-24 10:50:00 | 5493.50 | 2026-04-24 11:45:00 | 5465.87 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-04-24 10:50:00 | 5493.50 | 2026-04-24 15:20:00 | 5218.00 | TARGET_HIT | 0.50 | 5.02% |
| BUY | retest1 | 2026-04-28 11:10:00 | 5404.00 | 2026-04-28 11:20:00 | 5392.41 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-04-29 11:10:00 | 5351.50 | 2026-04-29 11:15:00 | 5365.49 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-05-06 09:55:00 | 5480.00 | 2026-05-06 10:00:00 | 5503.22 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-05-06 09:55:00 | 5480.00 | 2026-05-06 15:05:00 | 5568.00 | TARGET_HIT | 0.50 | 1.61% |
| BUY | retest1 | 2026-05-08 10:50:00 | 5655.00 | 2026-05-08 11:20:00 | 5635.49 | STOP_HIT | 1.00 | -0.34% |
