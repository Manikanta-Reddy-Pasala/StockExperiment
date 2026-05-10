# Steel Authority of India Ltd. (SAIL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 184.80
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 4 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 10
- **Target hits / Stop hits / Partials:** 4 / 10 / 8
- **Avg / median % per leg:** 0.19% / 0.25%
- **Sum % (uncompounded):** 4.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.15% | 1.5% |
| BUY @ 2nd Alert (retest1) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.15% | 1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 7 | 58.3% | 2 | 5 | 5 | 0.23% | 2.8% |
| SELL @ 2nd Alert (retest1) | 12 | 7 | 58.3% | 2 | 5 | 5 | 0.23% | 2.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 12 | 54.5% | 4 | 10 | 8 | 0.19% | 4.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 162.50 | 161.97 | 0.00 | ORB-long ORB[160.37,162.40] vol=1.6x ATR=0.50 |
| Stop hit — per-position SL triggered | 2026-02-11 09:40:00 | 162.00 | 162.15 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 156.19 | 157.30 | 0.00 | ORB-short ORB[157.20,158.79] vol=4.1x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 09:40:00 | 155.03 | 156.77 | 0.00 | T1 1.5R @ 155.03 |
| Stop hit — per-position SL triggered | 2026-02-13 09:45:00 | 156.19 | 156.76 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:00:00 | 156.36 | 157.27 | 0.00 | ORB-short ORB[157.03,158.84] vol=2.4x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:05:00 | 155.60 | 156.94 | 0.00 | T1 1.5R @ 155.60 |
| Target hit | 2026-02-17 12:20:00 | 155.97 | 155.71 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — SELL (started 2026-02-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:05:00 | 158.81 | 159.29 | 0.00 | ORB-short ORB[159.54,161.74] vol=3.0x ATR=0.46 |
| Stop hit — per-position SL triggered | 2026-02-19 12:05:00 | 159.27 | 159.22 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:35:00 | 157.47 | 156.58 | 0.00 | ORB-long ORB[155.49,157.00] vol=2.5x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:45:00 | 158.23 | 156.79 | 0.00 | T1 1.5R @ 158.23 |
| Target hit | 2026-02-20 15:20:00 | 158.48 | 158.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-02-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:05:00 | 156.89 | 158.42 | 0.00 | ORB-short ORB[158.42,159.70] vol=2.3x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:20:00 | 156.28 | 158.19 | 0.00 | T1 1.5R @ 156.28 |
| Stop hit — per-position SL triggered | 2026-02-23 12:15:00 | 156.89 | 157.72 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:25:00 | 158.45 | 157.47 | 0.00 | ORB-long ORB[155.80,157.96] vol=1.7x ATR=0.60 |
| Stop hit — per-position SL triggered | 2026-02-24 10:50:00 | 157.85 | 157.68 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:40:00 | 163.90 | 164.92 | 0.00 | ORB-short ORB[164.53,165.79] vol=1.7x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:00:00 | 163.16 | 164.61 | 0.00 | T1 1.5R @ 163.16 |
| Stop hit — per-position SL triggered | 2026-02-26 12:00:00 | 163.90 | 164.24 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:00:00 | 149.58 | 147.01 | 0.00 | ORB-long ORB[145.09,147.18] vol=3.6x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:20:00 | 151.13 | 148.36 | 0.00 | T1 1.5R @ 151.13 |
| Stop hit — per-position SL triggered | 2026-03-17 10:55:00 | 149.58 | 148.86 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:15:00 | 155.29 | 154.08 | 0.00 | ORB-long ORB[151.65,153.11] vol=2.2x ATR=0.57 |
| Stop hit — per-position SL triggered | 2026-03-18 11:45:00 | 154.72 | 154.23 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 09:30:00 | 161.49 | 162.17 | 0.00 | ORB-short ORB[161.70,164.07] vol=3.2x ATR=0.78 |
| Stop hit — per-position SL triggered | 2026-04-13 09:35:00 | 162.27 | 162.13 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:00:00 | 167.66 | 169.37 | 0.00 | ORB-short ORB[169.25,171.10] vol=2.0x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:10:00 | 166.74 | 169.07 | 0.00 | T1 1.5R @ 166.74 |
| Target hit | 2026-04-15 15:20:00 | 166.51 | 167.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2026-05-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:55:00 | 188.35 | 186.68 | 0.00 | ORB-long ORB[185.15,187.09] vol=2.1x ATR=0.82 |
| Stop hit — per-position SL triggered | 2026-05-05 10:10:00 | 187.53 | 186.94 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:30:00 | 188.04 | 187.04 | 0.00 | ORB-long ORB[186.24,187.70] vol=2.2x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:55:00 | 189.22 | 187.71 | 0.00 | T1 1.5R @ 189.22 |
| Target hit | 2026-05-07 14:35:00 | 188.37 | 188.52 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 09:30:00 | 162.50 | 2026-02-11 09:40:00 | 162.00 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-13 09:30:00 | 156.19 | 2026-02-13 09:40:00 | 155.03 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2026-02-13 09:30:00 | 156.19 | 2026-02-13 09:45:00 | 156.19 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-17 10:00:00 | 156.36 | 2026-02-17 10:05:00 | 155.60 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-02-17 10:00:00 | 156.36 | 2026-02-17 12:20:00 | 155.97 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2026-02-19 11:05:00 | 158.81 | 2026-02-19 12:05:00 | 159.27 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-20 10:35:00 | 157.47 | 2026-02-20 10:45:00 | 158.23 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-20 10:35:00 | 157.47 | 2026-02-20 15:20:00 | 158.48 | TARGET_HIT | 0.50 | 0.64% |
| SELL | retest1 | 2026-02-23 11:05:00 | 156.89 | 2026-02-23 11:20:00 | 156.28 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-23 11:05:00 | 156.89 | 2026-02-23 12:15:00 | 156.89 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 10:25:00 | 158.45 | 2026-02-24 10:50:00 | 157.85 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-02-26 10:40:00 | 163.90 | 2026-02-26 11:00:00 | 163.16 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-02-26 10:40:00 | 163.90 | 2026-02-26 12:00:00 | 163.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:00:00 | 149.58 | 2026-03-17 10:20:00 | 151.13 | PARTIAL | 0.50 | 1.04% |
| BUY | retest1 | 2026-03-17 10:00:00 | 149.58 | 2026-03-17 10:55:00 | 149.58 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 11:15:00 | 155.29 | 2026-03-18 11:45:00 | 154.72 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-04-13 09:30:00 | 161.49 | 2026-04-13 09:35:00 | 162.27 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-04-15 11:00:00 | 167.66 | 2026-04-15 11:10:00 | 166.74 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-04-15 11:00:00 | 167.66 | 2026-04-15 15:20:00 | 166.51 | TARGET_HIT | 0.50 | 0.69% |
| BUY | retest1 | 2026-05-05 09:55:00 | 188.35 | 2026-05-05 10:10:00 | 187.53 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-05-07 10:30:00 | 188.04 | 2026-05-07 11:55:00 | 189.22 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-05-07 10:30:00 | 188.04 | 2026-05-07 14:35:00 | 188.37 | TARGET_HIT | 0.50 | 0.18% |
