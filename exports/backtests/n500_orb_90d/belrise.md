# Belrise Industries Ltd. (BELRISE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 222.30
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
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 5
- **Target hits / Stop hits / Partials:** 4 / 5 / 6
- **Avg / median % per leg:** 0.82% / 0.56%
- **Sum % (uncompounded):** 12.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 6 | 60.0% | 2 | 4 | 4 | 0.75% | 7.5% |
| BUY @ 2nd Alert (retest1) | 10 | 6 | 60.0% | 2 | 4 | 4 | 0.75% | 7.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 0.96% | 4.8% |
| SELL @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 2 | 1 | 2 | 0.96% | 4.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 10 | 66.7% | 4 | 5 | 6 | 0.82% | 12.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 186.00 | 184.63 | 0.00 | ORB-long ORB[182.90,185.50] vol=1.9x ATR=0.82 |
| Stop hit — per-position SL triggered | 2026-02-10 09:50:00 | 185.18 | 184.74 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 09:45:00 | 182.89 | 181.48 | 0.00 | ORB-long ORB[180.51,182.25] vol=2.9x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:50:00 | 184.04 | 182.20 | 0.00 | T1 1.5R @ 184.04 |
| Target hit | 2026-02-11 15:20:00 | 191.95 | 190.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:40:00 | 190.39 | 189.68 | 0.00 | ORB-long ORB[187.73,190.00] vol=4.7x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:50:00 | 191.58 | 190.17 | 0.00 | T1 1.5R @ 191.58 |
| Stop hit — per-position SL triggered | 2026-02-18 10:15:00 | 190.39 | 190.41 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:45:00 | 187.78 | 186.25 | 0.00 | ORB-long ORB[184.59,187.14] vol=2.3x ATR=0.75 |
| Stop hit — per-position SL triggered | 2026-02-27 09:50:00 | 187.03 | 186.45 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:45:00 | 184.49 | 186.07 | 0.00 | ORB-short ORB[185.82,187.70] vol=2.2x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:00:00 | 183.29 | 185.17 | 0.00 | T1 1.5R @ 183.29 |
| Target hit | 2026-03-13 15:20:00 | 177.60 | 180.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2026-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:45:00 | 209.90 | 208.65 | 0.00 | ORB-long ORB[207.74,209.50] vol=1.7x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:50:00 | 211.08 | 209.98 | 0.00 | T1 1.5R @ 211.08 |
| Stop hit — per-position SL triggered | 2026-04-16 10:00:00 | 209.90 | 210.10 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:05:00 | 221.32 | 222.28 | 0.00 | ORB-short ORB[221.87,224.38] vol=1.8x ATR=1.11 |
| Stop hit — per-position SL triggered | 2026-04-21 10:45:00 | 222.43 | 222.15 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-05-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:40:00 | 214.17 | 215.05 | 0.00 | ORB-short ORB[214.70,217.30] vol=3.2x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:30:00 | 212.97 | 214.84 | 0.00 | T1 1.5R @ 212.97 |
| Target hit | 2026-05-05 15:20:00 | 213.39 | 214.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 216.73 | 215.63 | 0.00 | ORB-long ORB[214.54,216.20] vol=3.1x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:35:00 | 217.80 | 216.51 | 0.00 | T1 1.5R @ 217.80 |
| Target hit | 2026-05-06 10:20:00 | 219.01 | 219.30 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:40:00 | 186.00 | 2026-02-10 09:50:00 | 185.18 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-02-11 09:45:00 | 182.89 | 2026-02-11 09:50:00 | 184.04 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-02-11 09:45:00 | 182.89 | 2026-02-11 15:20:00 | 191.95 | TARGET_HIT | 0.50 | 4.95% |
| BUY | retest1 | 2026-02-18 09:40:00 | 190.39 | 2026-02-18 09:50:00 | 191.58 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-02-18 09:40:00 | 190.39 | 2026-02-18 10:15:00 | 190.39 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-27 09:45:00 | 187.78 | 2026-02-27 09:50:00 | 187.03 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-03-13 09:45:00 | 184.49 | 2026-03-13 10:00:00 | 183.29 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-03-13 09:45:00 | 184.49 | 2026-03-13 15:20:00 | 177.60 | TARGET_HIT | 0.50 | 3.73% |
| BUY | retest1 | 2026-04-16 09:45:00 | 209.90 | 2026-04-16 09:50:00 | 211.08 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-16 09:45:00 | 209.90 | 2026-04-16 10:00:00 | 209.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-21 10:05:00 | 221.32 | 2026-04-21 10:45:00 | 222.43 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-05-05 10:40:00 | 214.17 | 2026-05-05 11:30:00 | 212.97 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-05-05 10:40:00 | 214.17 | 2026-05-05 15:20:00 | 213.39 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2026-05-06 09:30:00 | 216.73 | 2026-05-06 09:35:00 | 217.80 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-05-06 09:30:00 | 216.73 | 2026-05-06 10:20:00 | 219.01 | TARGET_HIT | 0.50 | 1.05% |
