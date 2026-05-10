# Castrol India Ltd. (CASTROLIND)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 185.00
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
| ENTRY1 | 22 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 18
- **Target hits / Stop hits / Partials:** 4 / 18 / 7
- **Avg / median % per leg:** 0.04% / 0.00%
- **Sum % (uncompounded):** 1.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 7 | 41.2% | 3 | 10 | 4 | 0.07% | 1.2% |
| BUY @ 2nd Alert (retest1) | 17 | 7 | 41.2% | 3 | 10 | 4 | 0.07% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 4 | 33.3% | 1 | 8 | 3 | -0.01% | -0.1% |
| SELL @ 2nd Alert (retest1) | 12 | 4 | 33.3% | 1 | 8 | 3 | -0.01% | -0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 29 | 11 | 37.9% | 4 | 18 | 7 | 0.04% | 1.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:35:00 | 188.82 | 187.81 | 0.00 | ORB-long ORB[185.55,186.70] vol=4.6x ATR=0.58 |
| Target hit | 2026-02-09 15:20:00 | 189.09 | 188.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:35:00 | 187.29 | 186.93 | 0.00 | ORB-long ORB[186.42,187.10] vol=1.9x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:10:00 | 187.78 | 187.21 | 0.00 | T1 1.5R @ 187.78 |
| Stop hit — per-position SL triggered | 2026-02-17 10:15:00 | 187.29 | 187.22 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:45:00 | 188.78 | 188.23 | 0.00 | ORB-long ORB[187.72,188.75] vol=1.7x ATR=0.31 |
| Stop hit — per-position SL triggered | 2026-02-18 11:00:00 | 188.47 | 188.25 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 189.88 | 189.47 | 0.00 | ORB-long ORB[187.60,189.85] vol=4.1x ATR=0.37 |
| Stop hit — per-position SL triggered | 2026-02-19 09:45:00 | 189.51 | 189.49 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:50:00 | 187.00 | 187.52 | 0.00 | ORB-short ORB[187.12,188.38] vol=2.1x ATR=0.35 |
| Stop hit — per-position SL triggered | 2026-02-23 10:05:00 | 187.35 | 187.40 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 186.34 | 186.67 | 0.00 | ORB-short ORB[186.50,187.40] vol=2.2x ATR=0.25 |
| Stop hit — per-position SL triggered | 2026-02-24 09:40:00 | 186.59 | 186.65 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:40:00 | 187.75 | 187.32 | 0.00 | ORB-long ORB[186.85,187.62] vol=2.2x ATR=0.32 |
| Stop hit — per-position SL triggered | 2026-02-26 09:55:00 | 187.43 | 187.40 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-02 11:00:00 | 186.71 | 185.70 | 0.00 | ORB-long ORB[184.10,186.50] vol=3.6x ATR=0.38 |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 186.33 | 185.75 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 11:05:00 | 187.01 | 186.74 | 0.00 | ORB-long ORB[186.00,186.95] vol=2.2x ATR=0.24 |
| Stop hit — per-position SL triggered | 2026-03-06 11:10:00 | 186.77 | 186.75 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 10:15:00 | 185.86 | 185.99 | 0.00 | ORB-short ORB[186.01,186.87] vol=2.7x ATR=0.31 |
| Stop hit — per-position SL triggered | 2026-03-10 15:20:00 | 186.00 | 185.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-03-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:45:00 | 176.62 | 176.06 | 0.00 | ORB-long ORB[173.78,176.36] vol=2.2x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 09:55:00 | 177.47 | 176.25 | 0.00 | T1 1.5R @ 177.47 |
| Target hit | 2026-03-25 15:20:00 | 178.10 | 177.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2026-04-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-02 11:10:00 | 175.05 | 174.36 | 0.00 | ORB-long ORB[173.52,174.97] vol=3.7x ATR=0.38 |
| Stop hit — per-position SL triggered | 2026-04-02 11:15:00 | 174.67 | 174.37 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-06 10:15:00 | 176.23 | 176.70 | 0.00 | ORB-short ORB[176.40,178.00] vol=1.6x ATR=0.51 |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 176.74 | 176.61 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 09:30:00 | 180.30 | 179.82 | 0.00 | ORB-long ORB[178.20,180.00] vol=1.7x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 09:35:00 | 180.89 | 179.94 | 0.00 | T1 1.5R @ 180.89 |
| Stop hit — per-position SL triggered | 2026-04-09 09:40:00 | 180.30 | 179.98 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:40:00 | 179.28 | 177.93 | 0.00 | ORB-long ORB[176.11,178.20] vol=1.8x ATR=0.46 |
| Stop hit — per-position SL triggered | 2026-04-13 13:00:00 | 178.82 | 178.31 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 09:30:00 | 179.77 | 180.36 | 0.00 | ORB-short ORB[179.95,181.00] vol=2.3x ATR=0.45 |
| Stop hit — per-position SL triggered | 2026-04-15 10:25:00 | 180.22 | 180.18 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:30:00 | 184.90 | 184.25 | 0.00 | ORB-long ORB[182.90,184.75] vol=3.0x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:40:00 | 185.70 | 184.70 | 0.00 | T1 1.5R @ 185.70 |
| Stop hit — per-position SL triggered | 2026-04-23 10:00:00 | 184.90 | 184.76 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:10:00 | 183.06 | 183.32 | 0.00 | ORB-short ORB[183.11,185.21] vol=3.3x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 15:05:00 | 182.62 | 183.16 | 0.00 | T1 1.5R @ 182.62 |
| Target hit | 2026-04-24 15:20:00 | 182.76 | 183.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2026-04-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:05:00 | 184.64 | 183.77 | 0.00 | ORB-long ORB[182.90,184.61] vol=4.2x ATR=0.53 |
| Target hit | 2026-04-29 15:20:00 | 184.67 | 184.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2026-05-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:50:00 | 184.41 | 185.26 | 0.00 | ORB-short ORB[184.47,186.55] vol=1.9x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:25:00 | 183.88 | 184.92 | 0.00 | T1 1.5R @ 183.88 |
| Stop hit — per-position SL triggered | 2026-05-05 11:35:00 | 184.41 | 184.91 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2026-05-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:05:00 | 185.04 | 185.34 | 0.00 | ORB-short ORB[185.15,186.40] vol=9.6x ATR=0.35 |
| Stop hit — per-position SL triggered | 2026-05-06 11:10:00 | 185.39 | 185.34 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2026-05-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:25:00 | 185.79 | 185.96 | 0.00 | ORB-short ORB[185.83,187.00] vol=1.6x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:15:00 | 185.24 | 185.74 | 0.00 | T1 1.5R @ 185.24 |
| Stop hit — per-position SL triggered | 2026-05-07 11:25:00 | 185.79 | 185.74 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:35:00 | 188.82 | 2026-02-09 15:20:00 | 189.09 | TARGET_HIT | 1.00 | 0.14% |
| BUY | retest1 | 2026-02-17 09:35:00 | 187.29 | 2026-02-17 10:10:00 | 187.78 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2026-02-17 09:35:00 | 187.29 | 2026-02-17 10:15:00 | 187.29 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 10:45:00 | 188.78 | 2026-02-18 11:00:00 | 188.47 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-02-19 09:30:00 | 189.88 | 2026-02-19 09:45:00 | 189.51 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-23 09:50:00 | 187.00 | 2026-02-23 10:05:00 | 187.35 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-24 09:35:00 | 186.34 | 2026-02-24 09:40:00 | 186.59 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2026-02-26 09:40:00 | 187.75 | 2026-02-26 09:55:00 | 187.43 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-03-02 11:00:00 | 186.71 | 2026-03-02 11:15:00 | 186.33 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-03-06 11:05:00 | 187.01 | 2026-03-06 11:10:00 | 186.77 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2026-03-10 10:15:00 | 185.86 | 2026-03-10 15:20:00 | 186.00 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest1 | 2026-03-25 09:45:00 | 176.62 | 2026-03-25 09:55:00 | 177.47 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-03-25 09:45:00 | 176.62 | 2026-03-25 15:20:00 | 178.10 | TARGET_HIT | 0.50 | 0.84% |
| BUY | retest1 | 2026-04-02 11:10:00 | 175.05 | 2026-04-02 11:15:00 | 174.67 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-06 10:15:00 | 176.23 | 2026-04-06 11:15:00 | 176.74 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-09 09:30:00 | 180.30 | 2026-04-09 09:35:00 | 180.89 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-04-09 09:30:00 | 180.30 | 2026-04-09 09:40:00 | 180.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-13 10:40:00 | 179.28 | 2026-04-13 13:00:00 | 178.82 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-15 09:30:00 | 179.77 | 2026-04-15 10:25:00 | 180.22 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-23 09:30:00 | 184.90 | 2026-04-23 09:40:00 | 185.70 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-04-23 09:30:00 | 184.90 | 2026-04-23 10:00:00 | 184.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 11:10:00 | 183.06 | 2026-04-24 15:05:00 | 182.62 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2026-04-24 11:10:00 | 183.06 | 2026-04-24 15:20:00 | 182.76 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2026-04-29 10:05:00 | 184.64 | 2026-04-29 15:20:00 | 184.67 | TARGET_HIT | 1.00 | 0.02% |
| SELL | retest1 | 2026-05-05 10:50:00 | 184.41 | 2026-05-05 11:25:00 | 183.88 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-05-05 10:50:00 | 184.41 | 2026-05-05 11:35:00 | 184.41 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-06 11:05:00 | 185.04 | 2026-05-06 11:10:00 | 185.39 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-05-07 10:25:00 | 185.79 | 2026-05-07 11:15:00 | 185.24 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-05-07 10:25:00 | 185.79 | 2026-05-07 11:25:00 | 185.79 | STOP_HIT | 0.50 | 0.00% |
