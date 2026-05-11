# GMR Airports Ltd. (GMRAIRPORT)

## Backtest Summary

- **Window:** 2026-01-06 09:15:00 → 2026-05-08 15:25:00 (3375 bars)
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 12
- **Target hits / Stop hits / Partials:** 0 / 12 / 2
- **Avg / median % per leg:** -0.14% / -0.27%
- **Sum % (uncompounded):** -1.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.09% | -0.5% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.09% | -0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 1 | 11.1% | 0 | 8 | 1 | -0.16% | -1.5% |
| SELL @ 2nd Alert (retest1) | 9 | 1 | 11.1% | 0 | 8 | 1 | -0.16% | -1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 2 | 14.3% | 0 | 12 | 2 | -0.14% | -2.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 11:05:00 | 104.40 | 104.53 | 0.00 | ORB-short ORB[104.42,105.77] vol=1.9x ATR=0.38 |
| Stop hit — per-position SL triggered | 2026-01-06 11:30:00 | 104.78 | 104.52 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-01-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 10:50:00 | 104.32 | 103.93 | 0.00 | ORB-long ORB[103.46,104.15] vol=3.5x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 10:55:00 | 104.73 | 104.13 | 0.00 | T1 1.5R @ 104.73 |
| Stop hit — per-position SL triggered | 2026-01-07 12:40:00 | 104.32 | 104.32 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-01-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:55:00 | 103.00 | 103.88 | 0.00 | ORB-short ORB[103.90,104.99] vol=1.9x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-01-08 11:35:00 | 103.30 | 103.65 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-01-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-12 11:05:00 | 98.22 | 98.82 | 0.00 | ORB-short ORB[98.35,99.79] vol=3.7x ATR=0.37 |
| Stop hit — per-position SL triggered | 2026-01-12 12:10:00 | 98.59 | 98.73 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-01-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:45:00 | 102.43 | 101.12 | 0.00 | ORB-long ORB[99.99,100.89] vol=3.0x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-01-16 10:50:00 | 102.13 | 101.26 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-01-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:30:00 | 98.97 | 99.67 | 0.00 | ORB-short ORB[99.33,100.80] vol=2.0x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:45:00 | 98.49 | 99.39 | 0.00 | T1 1.5R @ 98.49 |
| Stop hit — per-position SL triggered | 2026-01-20 10:05:00 | 98.97 | 99.28 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-01-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 10:20:00 | 93.22 | 93.35 | 0.00 | ORB-short ORB[93.37,93.92] vol=2.5x ATR=0.25 |
| Stop hit — per-position SL triggered | 2026-01-29 11:40:00 | 93.47 | 93.29 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-01-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 10:10:00 | 93.91 | 93.69 | 0.00 | ORB-long ORB[92.90,93.88] vol=2.4x ATR=0.27 |
| Stop hit — per-position SL triggered | 2026-01-30 11:10:00 | 93.64 | 93.74 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-02-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 09:40:00 | 98.30 | 98.20 | 0.00 | ORB-long ORB[97.59,98.29] vol=1.6x ATR=0.28 |
| Stop hit — per-position SL triggered | 2026-02-09 09:45:00 | 98.02 | 98.21 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-02-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:50:00 | 97.54 | 97.58 | 0.00 | ORB-short ORB[97.70,98.57] vol=1.7x ATR=0.22 |
| Stop hit — per-position SL triggered | 2026-02-10 11:00:00 | 97.76 | 97.58 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 96.04 | 96.50 | 0.00 | ORB-short ORB[96.22,97.33] vol=2.3x ATR=0.17 |
| Stop hit — per-position SL triggered | 2026-02-12 12:45:00 | 96.21 | 96.38 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-02-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:30:00 | 101.38 | 101.53 | 0.00 | ORB-short ORB[101.74,102.29] vol=2.0x ATR=0.27 |
| Stop hit — per-position SL triggered | 2026-02-27 11:55:00 | 101.65 | 101.49 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-01-06 11:05:00 | 104.40 | 2026-01-06 11:30:00 | 104.78 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-01-07 10:50:00 | 104.32 | 2026-01-07 10:55:00 | 104.73 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-01-07 10:50:00 | 104.32 | 2026-01-07 12:40:00 | 104.32 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-08 10:55:00 | 103.00 | 2026-01-08 11:35:00 | 103.30 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-01-12 11:05:00 | 98.22 | 2026-01-12 12:10:00 | 98.59 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-01-16 10:45:00 | 102.43 | 2026-01-16 10:50:00 | 102.13 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-01-20 09:30:00 | 98.97 | 2026-01-20 09:45:00 | 98.49 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-01-20 09:30:00 | 98.97 | 2026-01-20 10:05:00 | 98.97 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-29 10:20:00 | 93.22 | 2026-01-29 11:40:00 | 93.47 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-01-30 10:10:00 | 93.91 | 2026-01-30 11:10:00 | 93.64 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-09 09:40:00 | 98.30 | 2026-02-09 09:45:00 | 98.02 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-10 10:50:00 | 97.54 | 2026-02-10 11:00:00 | 97.76 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-12 11:15:00 | 96.04 | 2026-02-12 12:45:00 | 96.21 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-02-27 10:30:00 | 101.38 | 2026-02-27 11:55:00 | 101.65 | STOP_HIT | 1.00 | -0.27% |
