# ITC Hotels Ltd. (ITCHOTELS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 164.58
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
| ENTRY1 | 17 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 5 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 12
- **Target hits / Stop hits / Partials:** 5 / 12 / 6
- **Avg / median % per leg:** 0.36% / 0.00%
- **Sum % (uncompounded):** 8.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 8 | 80.0% | 4 | 2 | 4 | 0.91% | 9.1% |
| BUY @ 2nd Alert (retest1) | 10 | 8 | 80.0% | 4 | 2 | 4 | 0.91% | 9.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 3 | 23.1% | 1 | 10 | 2 | -0.07% | -0.8% |
| SELL @ 2nd Alert (retest1) | 13 | 3 | 23.1% | 1 | 10 | 2 | -0.07% | -0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 11 | 47.8% | 5 | 12 | 6 | 0.36% | 8.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:15:00 | 186.30 | 187.10 | 0.00 | ORB-short ORB[186.67,188.70] vol=1.6x ATR=0.71 |
| Stop hit — per-position SL triggered | 2026-02-10 10:50:00 | 187.01 | 186.86 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:00:00 | 187.39 | 188.71 | 0.00 | ORB-short ORB[187.60,189.85] vol=2.3x ATR=0.67 |
| Stop hit — per-position SL triggered | 2026-02-11 10:20:00 | 188.06 | 188.49 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 09:50:00 | 177.76 | 178.60 | 0.00 | ORB-short ORB[178.25,180.60] vol=2.0x ATR=0.56 |
| Stop hit — per-position SL triggered | 2026-02-16 09:55:00 | 178.32 | 178.56 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:20:00 | 179.42 | 180.24 | 0.00 | ORB-short ORB[180.06,181.74] vol=2.2x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:35:00 | 178.75 | 179.98 | 0.00 | T1 1.5R @ 178.75 |
| Stop hit — per-position SL triggered | 2026-02-19 11:30:00 | 179.42 | 179.70 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:40:00 | 176.62 | 176.99 | 0.00 | ORB-short ORB[176.72,178.41] vol=2.2x ATR=0.44 |
| Stop hit — per-position SL triggered | 2026-02-27 10:30:00 | 177.06 | 176.87 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:15:00 | 164.00 | 164.64 | 0.00 | ORB-short ORB[164.53,165.90] vol=3.0x ATR=0.51 |
| Stop hit — per-position SL triggered | 2026-03-05 11:40:00 | 164.51 | 164.57 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:15:00 | 155.16 | 153.36 | 0.00 | ORB-long ORB[150.72,153.03] vol=2.2x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 11:35:00 | 155.86 | 153.67 | 0.00 | T1 1.5R @ 155.86 |
| Target hit | 2026-03-18 15:20:00 | 159.26 | 157.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:15:00 | 159.84 | 158.35 | 0.00 | ORB-long ORB[156.54,158.40] vol=2.6x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:30:00 | 160.72 | 158.79 | 0.00 | T1 1.5R @ 160.72 |
| Target hit | 2026-04-15 15:20:00 | 161.75 | 161.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2026-04-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:40:00 | 162.60 | 163.48 | 0.00 | ORB-short ORB[163.21,164.66] vol=2.8x ATR=0.65 |
| Stop hit — per-position SL triggered | 2026-04-16 10:45:00 | 163.25 | 163.22 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 163.83 | 163.01 | 0.00 | ORB-long ORB[161.49,163.40] vol=2.1x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:00:00 | 164.64 | 163.46 | 0.00 | T1 1.5R @ 164.64 |
| Target hit | 2026-04-21 14:40:00 | 165.59 | 165.61 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2026-04-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:45:00 | 161.30 | 162.13 | 0.00 | ORB-short ORB[161.58,163.52] vol=1.9x ATR=0.37 |
| Stop hit — per-position SL triggered | 2026-04-23 11:00:00 | 161.67 | 162.07 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 159.90 | 160.26 | 0.00 | ORB-short ORB[160.02,161.10] vol=1.7x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 09:35:00 | 159.36 | 160.07 | 0.00 | T1 1.5R @ 159.36 |
| Target hit | 2026-04-28 15:20:00 | 157.97 | 158.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2026-04-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:00:00 | 159.80 | 159.12 | 0.00 | ORB-long ORB[158.00,159.24] vol=3.8x ATR=0.42 |
| Stop hit — per-position SL triggered | 2026-04-29 10:10:00 | 159.38 | 159.22 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:55:00 | 162.90 | 161.63 | 0.00 | ORB-long ORB[160.10,161.44] vol=4.8x ATR=0.56 |
| Stop hit — per-position SL triggered | 2026-05-04 12:15:00 | 162.34 | 162.21 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:55:00 | 164.45 | 163.57 | 0.00 | ORB-long ORB[162.90,164.09] vol=3.9x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:10:00 | 165.26 | 164.09 | 0.00 | T1 1.5R @ 165.26 |
| Target hit | 2026-05-06 15:20:00 | 169.14 | 167.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2026-05-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:05:00 | 167.81 | 168.75 | 0.00 | ORB-short ORB[168.17,170.20] vol=2.1x ATR=0.46 |
| Stop hit — per-position SL triggered | 2026-05-07 11:25:00 | 168.27 | 168.68 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:50:00 | 164.90 | 165.57 | 0.00 | ORB-short ORB[164.95,166.50] vol=1.8x ATR=0.41 |
| Stop hit — per-position SL triggered | 2026-05-08 10:55:00 | 165.31 | 165.54 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:15:00 | 186.30 | 2026-02-10 10:50:00 | 187.01 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-02-11 10:00:00 | 187.39 | 2026-02-11 10:20:00 | 188.06 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-16 09:50:00 | 177.76 | 2026-02-16 09:55:00 | 178.32 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-19 10:20:00 | 179.42 | 2026-02-19 10:35:00 | 178.75 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-02-19 10:20:00 | 179.42 | 2026-02-19 11:30:00 | 179.42 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 09:40:00 | 176.62 | 2026-02-27 10:30:00 | 177.06 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-05 11:15:00 | 164.00 | 2026-03-05 11:40:00 | 164.51 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-03-18 11:15:00 | 155.16 | 2026-03-18 11:35:00 | 155.86 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-03-18 11:15:00 | 155.16 | 2026-03-18 15:20:00 | 159.26 | TARGET_HIT | 0.50 | 2.64% |
| BUY | retest1 | 2026-04-15 10:15:00 | 159.84 | 2026-04-15 10:30:00 | 160.72 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-15 10:15:00 | 159.84 | 2026-04-15 15:20:00 | 161.75 | TARGET_HIT | 0.50 | 1.19% |
| SELL | retest1 | 2026-04-16 09:40:00 | 162.60 | 2026-04-16 10:45:00 | 163.25 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-21 09:35:00 | 163.83 | 2026-04-21 10:00:00 | 164.64 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-21 09:35:00 | 163.83 | 2026-04-21 14:40:00 | 165.59 | TARGET_HIT | 0.50 | 1.07% |
| SELL | retest1 | 2026-04-23 10:45:00 | 161.30 | 2026-04-23 11:00:00 | 161.67 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-04-28 09:30:00 | 159.90 | 2026-04-28 09:35:00 | 159.36 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-04-28 09:30:00 | 159.90 | 2026-04-28 15:20:00 | 157.97 | TARGET_HIT | 0.50 | 1.21% |
| BUY | retest1 | 2026-04-29 10:00:00 | 159.80 | 2026-04-29 10:10:00 | 159.38 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-05-04 10:55:00 | 162.90 | 2026-05-04 12:15:00 | 162.34 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-05-06 09:55:00 | 164.45 | 2026-05-06 10:10:00 | 165.26 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-05-06 09:55:00 | 164.45 | 2026-05-06 15:20:00 | 169.14 | TARGET_HIT | 0.50 | 2.85% |
| SELL | retest1 | 2026-05-07 11:05:00 | 167.81 | 2026-05-07 11:25:00 | 168.27 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-05-08 10:50:00 | 164.90 | 2026-05-08 10:55:00 | 165.31 | STOP_HIT | 1.00 | -0.25% |
