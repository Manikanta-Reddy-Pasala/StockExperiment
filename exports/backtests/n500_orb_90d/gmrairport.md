# GMR Airports Ltd. (GMRAIRPORT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 101.30
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 9
- **Target hits / Stop hits / Partials:** 0 / 9 / 1
- **Avg / median % per leg:** -0.17% / -0.26%
- **Sum % (uncompounded):** -1.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.06% | -0.3% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.06% | -0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.28% | -1.4% |
| SELL @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.28% | -1.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 1 | 10.0% | 0 | 9 | 1 | -0.17% | -1.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:50:00 | 97.54 | 97.58 | 0.00 | ORB-short ORB[97.70,98.57] vol=1.7x ATR=0.22 |
| Stop hit — per-position SL triggered | 2026-02-10 11:00:00 | 97.76 | 97.58 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 96.04 | 96.50 | 0.00 | ORB-short ORB[96.22,97.33] vol=2.3x ATR=0.17 |
| Stop hit — per-position SL triggered | 2026-02-12 12:45:00 | 96.21 | 96.38 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:30:00 | 101.38 | 101.53 | 0.00 | ORB-short ORB[101.74,102.29] vol=2.0x ATR=0.27 |
| Stop hit — per-position SL triggered | 2026-02-27 11:55:00 | 101.65 | 101.49 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:15:00 | 94.40 | 93.09 | 0.00 | ORB-long ORB[92.08,93.11] vol=1.6x ATR=0.34 |
| Stop hit — per-position SL triggered | 2026-03-12 11:20:00 | 94.06 | 93.11 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:30:00 | 91.14 | 90.47 | 0.00 | ORB-long ORB[90.00,91.13] vol=1.8x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:40:00 | 91.78 | 90.84 | 0.00 | T1 1.5R @ 91.78 |
| Stop hit — per-position SL triggered | 2026-03-16 10:05:00 | 91.14 | 90.96 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:55:00 | 97.75 | 97.14 | 0.00 | ORB-long ORB[95.80,97.24] vol=1.8x ATR=0.25 |
| Stop hit — per-position SL triggered | 2026-04-17 11:00:00 | 97.50 | 97.15 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:35:00 | 95.11 | 95.56 | 0.00 | ORB-short ORB[95.29,96.56] vol=1.5x ATR=0.36 |
| Stop hit — per-position SL triggered | 2026-04-24 09:40:00 | 95.47 | 95.55 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-05-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:10:00 | 97.55 | 97.30 | 0.00 | ORB-long ORB[96.22,97.45] vol=2.9x ATR=0.36 |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 97.19 | 97.31 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-05-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:30:00 | 96.29 | 96.71 | 0.00 | ORB-short ORB[97.06,98.32] vol=1.6x ATR=0.36 |
| Stop hit — per-position SL triggered | 2026-05-05 10:55:00 | 96.65 | 96.63 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:50:00 | 97.54 | 2026-02-10 11:00:00 | 97.76 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-12 11:15:00 | 96.04 | 2026-02-12 12:45:00 | 96.21 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-02-27 10:30:00 | 101.38 | 2026-02-27 11:55:00 | 101.65 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-12 11:15:00 | 94.40 | 2026-03-12 11:20:00 | 94.06 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-16 09:30:00 | 91.14 | 2026-03-16 09:40:00 | 91.78 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-03-16 09:30:00 | 91.14 | 2026-03-16 10:05:00 | 91.14 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 10:55:00 | 97.75 | 2026-04-17 11:00:00 | 97.50 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-24 09:35:00 | 95.11 | 2026-04-24 09:40:00 | 95.47 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-05-04 10:10:00 | 97.55 | 2026-05-04 10:15:00 | 97.19 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-05-05 10:30:00 | 96.29 | 2026-05-05 10:55:00 | 96.65 | STOP_HIT | 1.00 | -0.37% |
