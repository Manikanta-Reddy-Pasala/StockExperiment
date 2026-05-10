# Paradeep Phosphates Ltd. (PARADEEP)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 125.09
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
| ENTRY1 | 10 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 9
- **Target hits / Stop hits / Partials:** 1 / 9 / 3
- **Avg / median % per leg:** 0.01% / -0.35%
- **Sum % (uncompounded):** 0.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.51% | -1.5% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.51% | -1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.17% | 1.7% |
| SELL @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.17% | 1.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 4 | 30.8% | 1 | 9 | 3 | 0.01% | 0.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:40:00 | 122.82 | 122.25 | 0.00 | ORB-long ORB[121.25,122.79] vol=2.3x ATR=0.45 |
| Stop hit — per-position SL triggered | 2026-02-17 09:45:00 | 122.37 | 122.31 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 123.72 | 124.70 | 0.00 | ORB-short ORB[124.57,125.78] vol=4.1x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:40:00 | 122.92 | 124.21 | 0.00 | T1 1.5R @ 122.92 |
| Target hit | 2026-02-18 15:05:00 | 122.00 | 121.84 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2026-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:35:00 | 120.09 | 120.97 | 0.00 | ORB-short ORB[120.83,122.25] vol=1.6x ATR=0.48 |
| Stop hit — per-position SL triggered | 2026-02-19 11:25:00 | 120.57 | 120.44 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 09:30:00 | 117.18 | 117.89 | 0.00 | ORB-short ORB[117.80,119.19] vol=2.7x ATR=0.57 |
| Stop hit — per-position SL triggered | 2026-02-20 10:35:00 | 117.75 | 117.42 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 09:35:00 | 114.08 | 114.74 | 0.00 | ORB-short ORB[114.19,115.69] vol=2.4x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 09:40:00 | 113.26 | 114.57 | 0.00 | T1 1.5R @ 113.26 |
| Stop hit — per-position SL triggered | 2026-04-07 10:25:00 | 114.08 | 114.18 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:45:00 | 122.60 | 121.28 | 0.00 | ORB-long ORB[119.84,121.35] vol=2.1x ATR=0.62 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 121.98 | 121.50 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:00:00 | 136.83 | 134.91 | 0.00 | ORB-long ORB[133.90,135.37] vol=1.5x ATR=0.90 |
| Stop hit — per-position SL triggered | 2026-04-27 11:25:00 | 135.93 | 135.52 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 11:00:00 | 132.00 | 132.67 | 0.00 | ORB-short ORB[133.49,135.40] vol=1.9x ATR=0.46 |
| Stop hit — per-position SL triggered | 2026-04-29 12:40:00 | 132.46 | 132.48 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-05-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 09:35:00 | 128.83 | 130.10 | 0.00 | ORB-short ORB[129.58,131.49] vol=1.5x ATR=0.68 |
| Stop hit — per-position SL triggered | 2026-05-04 09:45:00 | 129.51 | 129.92 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 123.85 | 124.71 | 0.00 | ORB-short ORB[124.28,125.97] vol=1.7x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:50:00 | 122.97 | 124.11 | 0.00 | T1 1.5R @ 122.97 |
| Stop hit — per-position SL triggered | 2026-05-06 10:10:00 | 123.85 | 123.96 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 09:40:00 | 122.82 | 2026-02-17 09:45:00 | 122.37 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-18 09:30:00 | 123.72 | 2026-02-18 09:40:00 | 122.92 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-02-18 09:30:00 | 123.72 | 2026-02-18 15:05:00 | 122.00 | TARGET_HIT | 0.50 | 1.39% |
| SELL | retest1 | 2026-02-19 09:35:00 | 120.09 | 2026-02-19 11:25:00 | 120.57 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-02-20 09:30:00 | 117.18 | 2026-02-20 10:35:00 | 117.75 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-04-07 09:35:00 | 114.08 | 2026-04-07 09:40:00 | 113.26 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2026-04-07 09:35:00 | 114.08 | 2026-04-07 10:25:00 | 114.08 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 09:45:00 | 122.60 | 2026-04-10 10:05:00 | 121.98 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2026-04-27 10:00:00 | 136.83 | 2026-04-27 11:25:00 | 135.93 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest1 | 2026-04-29 11:00:00 | 132.00 | 2026-04-29 12:40:00 | 132.46 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-05-04 09:35:00 | 128.83 | 2026-05-04 09:45:00 | 129.51 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2026-05-06 09:30:00 | 123.85 | 2026-05-06 09:50:00 | 122.97 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2026-05-06 09:30:00 | 123.85 | 2026-05-06 10:10:00 | 123.85 | STOP_HIT | 0.50 | 0.00% |
