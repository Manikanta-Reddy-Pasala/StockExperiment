# Power Grid Corporation of India Ltd. (POWERGRID)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 313.90
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
| ENTRY1 | 13 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 10
- **Target hits / Stop hits / Partials:** 3 / 10 / 4
- **Avg / median % per leg:** 0.22% / -0.18%
- **Sum % (uncompounded):** 3.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.35% | 3.5% |
| BUY @ 2nd Alert (retest1) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.35% | 3.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 2 | 28.6% | 1 | 5 | 1 | 0.03% | 0.2% |
| SELL @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 1 | 5 | 1 | 0.03% | 0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 7 | 41.2% | 3 | 10 | 4 | 0.22% | 3.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:15:00 | 291.25 | 291.98 | 0.00 | ORB-short ORB[291.35,294.30] vol=1.6x ATR=0.68 |
| Stop hit — per-position SL triggered | 2026-02-13 10:35:00 | 291.93 | 291.89 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:55:00 | 298.15 | 298.96 | 0.00 | ORB-short ORB[299.25,301.60] vol=4.3x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:15:00 | 297.16 | 298.86 | 0.00 | T1 1.5R @ 297.16 |
| Target hit | 2026-02-19 15:20:00 | 294.50 | 296.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:00:00 | 304.70 | 305.98 | 0.00 | ORB-short ORB[306.50,307.70] vol=1.6x ATR=0.61 |
| Stop hit — per-position SL triggered | 2026-02-26 11:20:00 | 305.31 | 305.91 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 11:00:00 | 302.15 | 300.59 | 0.00 | ORB-long ORB[297.00,299.95] vol=1.7x ATR=0.81 |
| Stop hit — per-position SL triggered | 2026-03-06 11:25:00 | 301.34 | 300.77 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:25:00 | 301.80 | 298.39 | 0.00 | ORB-long ORB[295.10,298.25] vol=1.5x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 10:35:00 | 303.40 | 299.52 | 0.00 | T1 1.5R @ 303.40 |
| Target hit | 2026-03-12 15:20:00 | 303.85 | 302.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-04-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-06 10:45:00 | 288.60 | 291.76 | 0.00 | ORB-short ORB[291.80,294.95] vol=1.7x ATR=1.20 |
| Stop hit — per-position SL triggered | 2026-04-06 12:45:00 | 289.80 | 290.41 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 09:35:00 | 298.30 | 297.02 | 0.00 | ORB-long ORB[293.85,298.00] vol=1.7x ATR=0.78 |
| Stop hit — per-position SL triggered | 2026-04-09 09:40:00 | 297.52 | 297.11 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:30:00 | 305.25 | 303.02 | 0.00 | ORB-long ORB[301.85,304.25] vol=1.7x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:55:00 | 306.38 | 303.76 | 0.00 | T1 1.5R @ 306.38 |
| Target hit | 2026-04-15 15:20:00 | 313.00 | 309.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-04-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 11:05:00 | 321.80 | 319.12 | 0.00 | ORB-long ORB[315.55,319.40] vol=1.6x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 11:10:00 | 322.85 | 319.54 | 0.00 | T1 1.5R @ 322.85 |
| Stop hit — per-position SL triggered | 2026-04-20 11:15:00 | 321.80 | 319.65 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:55:00 | 322.10 | 319.09 | 0.00 | ORB-long ORB[317.75,320.35] vol=3.3x ATR=0.83 |
| Stop hit — per-position SL triggered | 2026-04-27 11:05:00 | 321.27 | 319.75 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:15:00 | 318.20 | 319.98 | 0.00 | ORB-short ORB[318.70,322.80] vol=1.7x ATR=0.67 |
| Stop hit — per-position SL triggered | 2026-04-28 11:20:00 | 318.87 | 319.95 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:50:00 | 321.75 | 320.51 | 0.00 | ORB-long ORB[318.25,320.55] vol=2.0x ATR=0.59 |
| Stop hit — per-position SL triggered | 2026-04-29 11:10:00 | 321.16 | 320.60 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:55:00 | 316.35 | 318.35 | 0.00 | ORB-short ORB[316.55,320.25] vol=1.5x ATR=0.82 |
| Stop hit — per-position SL triggered | 2026-04-30 11:25:00 | 317.17 | 318.19 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 10:15:00 | 291.25 | 2026-02-13 10:35:00 | 291.93 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-19 10:55:00 | 298.15 | 2026-02-19 11:15:00 | 297.16 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-19 10:55:00 | 298.15 | 2026-02-19 15:20:00 | 294.50 | TARGET_HIT | 0.50 | 1.22% |
| SELL | retest1 | 2026-02-26 11:00:00 | 304.70 | 2026-02-26 11:20:00 | 305.31 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-03-06 11:00:00 | 302.15 | 2026-03-06 11:25:00 | 301.34 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-12 10:25:00 | 301.80 | 2026-03-12 10:35:00 | 303.40 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-03-12 10:25:00 | 301.80 | 2026-03-12 15:20:00 | 303.85 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2026-04-06 10:45:00 | 288.60 | 2026-04-06 12:45:00 | 289.80 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-09 09:35:00 | 298.30 | 2026-04-09 09:40:00 | 297.52 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-15 10:30:00 | 305.25 | 2026-04-15 10:55:00 | 306.38 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-04-15 10:30:00 | 305.25 | 2026-04-15 15:20:00 | 313.00 | TARGET_HIT | 0.50 | 2.54% |
| BUY | retest1 | 2026-04-20 11:05:00 | 321.80 | 2026-04-20 11:10:00 | 322.85 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-04-20 11:05:00 | 321.80 | 2026-04-20 11:15:00 | 321.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 10:55:00 | 322.10 | 2026-04-27 11:05:00 | 321.27 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-28 11:15:00 | 318.20 | 2026-04-28 11:20:00 | 318.87 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-04-29 10:50:00 | 321.75 | 2026-04-29 11:10:00 | 321.16 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-04-30 10:55:00 | 316.35 | 2026-04-30 11:25:00 | 317.17 | STOP_HIT | 1.00 | -0.26% |
