# Welspun Living Ltd. (WELSPUNLIV)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 134.05
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
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 9
- **Target hits / Stop hits / Partials:** 2 / 9 / 2
- **Avg / median % per leg:** -0.05% / -0.37%
- **Sum % (uncompounded):** -0.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 2 | 22.2% | 1 | 7 | 1 | -0.14% | -1.3% |
| BUY @ 2nd Alert (retest1) | 9 | 2 | 22.2% | 1 | 7 | 1 | -0.14% | -1.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.17% | 0.7% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.17% | 0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 4 | 30.8% | 2 | 9 | 2 | -0.05% | -0.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:35:00 | 139.55 | 138.51 | 0.00 | ORB-long ORB[137.71,139.39] vol=2.3x ATR=0.50 |
| Stop hit — per-position SL triggered | 2026-02-19 10:45:00 | 139.05 | 138.68 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:35:00 | 138.03 | 137.26 | 0.00 | ORB-long ORB[136.40,137.75] vol=1.5x ATR=0.60 |
| Stop hit — per-position SL triggered | 2026-02-20 10:00:00 | 137.43 | 137.43 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:55:00 | 117.00 | 117.54 | 0.00 | ORB-short ORB[117.01,118.60] vol=1.6x ATR=0.49 |
| Stop hit — per-position SL triggered | 2026-03-10 10:05:00 | 117.49 | 117.49 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:25:00 | 111.29 | 110.59 | 0.00 | ORB-long ORB[109.35,110.80] vol=1.6x ATR=0.48 |
| Stop hit — per-position SL triggered | 2026-03-17 10:35:00 | 110.81 | 110.61 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:45:00 | 123.90 | 123.52 | 0.00 | ORB-long ORB[122.60,123.83] vol=1.8x ATR=0.47 |
| Stop hit — per-position SL triggered | 2026-04-10 09:55:00 | 123.43 | 123.54 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:00:00 | 130.47 | 129.05 | 0.00 | ORB-long ORB[127.54,129.45] vol=3.3x ATR=0.57 |
| Stop hit — per-position SL triggered | 2026-04-21 10:05:00 | 129.90 | 129.33 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 132.40 | 131.81 | 0.00 | ORB-long ORB[130.81,132.37] vol=2.8x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:45:00 | 133.32 | 132.21 | 0.00 | T1 1.5R @ 133.32 |
| Target hit | 2026-04-27 14:00:00 | 133.46 | 133.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2026-04-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:35:00 | 131.53 | 132.54 | 0.00 | ORB-short ORB[132.01,133.10] vol=3.0x ATR=0.38 |
| Stop hit — per-position SL triggered | 2026-04-29 11:00:00 | 131.91 | 132.42 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:30:00 | 130.60 | 129.78 | 0.00 | ORB-long ORB[128.61,130.45] vol=2.4x ATR=0.50 |
| Stop hit — per-position SL triggered | 2026-05-04 09:50:00 | 130.10 | 129.89 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-05-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:45:00 | 129.51 | 129.22 | 0.00 | ORB-long ORB[128.11,129.10] vol=4.7x ATR=0.48 |
| Stop hit — per-position SL triggered | 2026-05-05 10:05:00 | 129.03 | 129.26 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:40:00 | 133.38 | 134.01 | 0.00 | ORB-short ORB[133.65,135.38] vol=2.3x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:20:00 | 131.97 | 133.62 | 0.00 | T1 1.5R @ 131.97 |
| Target hit | 2026-05-07 14:55:00 | 132.91 | 132.87 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-19 10:35:00 | 139.55 | 2026-02-19 10:45:00 | 139.05 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-20 09:35:00 | 138.03 | 2026-02-20 10:00:00 | 137.43 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-03-10 09:55:00 | 117.00 | 2026-03-10 10:05:00 | 117.49 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-17 10:25:00 | 111.29 | 2026-03-17 10:35:00 | 110.81 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-10 09:45:00 | 123.90 | 2026-04-10 09:55:00 | 123.43 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-21 10:00:00 | 130.47 | 2026-04-21 10:05:00 | 129.90 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-27 09:30:00 | 132.40 | 2026-04-27 09:45:00 | 133.32 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-04-27 09:30:00 | 132.40 | 2026-04-27 14:00:00 | 133.46 | TARGET_HIT | 0.50 | 0.80% |
| SELL | retest1 | 2026-04-29 10:35:00 | 131.53 | 2026-04-29 11:00:00 | 131.91 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-05-04 09:30:00 | 130.60 | 2026-05-04 09:50:00 | 130.10 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-05-05 09:45:00 | 129.51 | 2026-05-05 10:05:00 | 129.03 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-05-07 10:40:00 | 133.38 | 2026-05-07 11:20:00 | 131.97 | PARTIAL | 0.50 | 1.06% |
| SELL | retest1 | 2026-05-07 10:40:00 | 133.38 | 2026-05-07 14:55:00 | 132.91 | TARGET_HIT | 0.50 | 0.35% |
