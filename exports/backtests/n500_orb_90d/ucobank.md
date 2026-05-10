# UCO Bank (UCOBANK)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 26.79
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
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 14
- **Target hits / Stop hits / Partials:** 3 / 14 / 7
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 2.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 3 | 33.3% | 1 | 6 | 2 | 0.12% | 1.1% |
| BUY @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 1 | 6 | 2 | 0.12% | 1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 7 | 46.7% | 2 | 8 | 5 | 0.08% | 1.3% |
| SELL @ 2nd Alert (retest1) | 15 | 7 | 46.7% | 2 | 8 | 5 | 0.08% | 1.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 24 | 10 | 41.7% | 3 | 14 | 7 | 0.10% | 2.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 28.82 | 28.90 | 0.00 | ORB-short ORB[28.87,29.14] vol=2.8x ATR=0.07 |
| Stop hit — per-position SL triggered | 2026-02-11 09:55:00 | 28.89 | 28.88 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 28.31 | 28.41 | 0.00 | ORB-short ORB[28.33,28.55] vol=2.1x ATR=0.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 10:15:00 | 28.18 | 28.34 | 0.00 | T1 1.5R @ 28.18 |
| Stop hit — per-position SL triggered | 2026-02-13 10:25:00 | 28.31 | 28.33 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:50:00 | 28.51 | 28.45 | 0.00 | ORB-long ORB[28.29,28.49] vol=1.7x ATR=0.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:10:00 | 28.63 | 28.49 | 0.00 | T1 1.5R @ 28.63 |
| Target hit | 2026-02-17 11:00:00 | 29.01 | 29.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2026-02-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:30:00 | 29.04 | 28.86 | 0.00 | ORB-long ORB[28.69,28.99] vol=2.1x ATR=0.15 |
| Stop hit — per-position SL triggered | 2026-02-20 15:20:00 | 28.92 | 28.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 28.68 | 28.75 | 0.00 | ORB-short ORB[28.71,28.88] vol=2.2x ATR=0.07 |
| Stop hit — per-position SL triggered | 2026-02-24 09:35:00 | 28.75 | 28.74 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:50:00 | 29.85 | 29.53 | 0.00 | ORB-long ORB[29.32,29.68] vol=3.0x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:00:00 | 30.08 | 29.67 | 0.00 | T1 1.5R @ 30.08 |
| Stop hit — per-position SL triggered | 2026-02-25 10:05:00 | 29.85 | 29.69 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:05:00 | 27.37 | 27.45 | 0.00 | ORB-short ORB[27.43,27.69] vol=1.6x ATR=0.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:35:00 | 27.25 | 27.42 | 0.00 | T1 1.5R @ 27.25 |
| Target hit | 2026-03-05 14:45:00 | 27.34 | 27.34 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — BUY (started 2026-03-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:05:00 | 27.51 | 27.44 | 0.00 | ORB-long ORB[27.25,27.46] vol=1.7x ATR=0.08 |
| Stop hit — per-position SL triggered | 2026-03-06 10:35:00 | 27.43 | 27.45 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:55:00 | 26.29 | 26.41 | 0.00 | ORB-short ORB[26.37,26.70] vol=1.6x ATR=0.10 |
| Stop hit — per-position SL triggered | 2026-03-10 10:05:00 | 26.39 | 26.40 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:00:00 | 26.07 | 26.21 | 0.00 | ORB-short ORB[26.26,26.43] vol=5.1x ATR=0.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:20:00 | 25.93 | 26.16 | 0.00 | T1 1.5R @ 25.93 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 26.07 | 26.13 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 26.33 | 26.17 | 0.00 | ORB-long ORB[25.90,26.27] vol=2.3x ATR=0.10 |
| Stop hit — per-position SL triggered | 2026-04-10 09:35:00 | 26.23 | 26.18 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:55:00 | 26.55 | 26.72 | 0.00 | ORB-short ORB[26.59,26.92] vol=1.6x ATR=0.08 |
| Stop hit — per-position SL triggered | 2026-04-16 10:15:00 | 26.63 | 26.70 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 26.87 | 26.74 | 0.00 | ORB-long ORB[26.51,26.82] vol=1.7x ATR=0.12 |
| Stop hit — per-position SL triggered | 2026-04-21 11:50:00 | 26.75 | 26.86 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 26.54 | 26.68 | 0.00 | ORB-short ORB[26.61,26.87] vol=1.7x ATR=0.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:50:00 | 26.41 | 26.63 | 0.00 | T1 1.5R @ 26.41 |
| Target hit | 2026-04-24 14:30:00 | 26.47 | 26.47 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — BUY (started 2026-04-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:20:00 | 26.73 | 26.54 | 0.00 | ORB-long ORB[26.40,26.59] vol=1.8x ATR=0.08 |
| Stop hit — per-position SL triggered | 2026-04-29 10:30:00 | 26.65 | 26.56 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:00:00 | 26.36 | 26.47 | 0.00 | ORB-short ORB[26.38,26.63] vol=1.8x ATR=0.09 |
| Stop hit — per-position SL triggered | 2026-04-30 10:40:00 | 26.45 | 26.44 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:35:00 | 27.03 | 27.10 | 0.00 | ORB-short ORB[27.04,27.34] vol=1.6x ATR=0.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:25:00 | 26.91 | 27.05 | 0.00 | T1 1.5R @ 26.91 |
| Stop hit — per-position SL triggered | 2026-05-08 14:45:00 | 27.03 | 26.99 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:30:00 | 28.82 | 2026-02-11 09:55:00 | 28.89 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-13 09:30:00 | 28.31 | 2026-02-13 10:15:00 | 28.18 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-13 09:30:00 | 28.31 | 2026-02-13 10:25:00 | 28.31 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 09:50:00 | 28.51 | 2026-02-17 10:10:00 | 28.63 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-02-17 09:50:00 | 28.51 | 2026-02-17 11:00:00 | 29.01 | TARGET_HIT | 0.50 | 1.75% |
| BUY | retest1 | 2026-02-20 09:30:00 | 29.04 | 2026-02-20 15:20:00 | 28.92 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-02-24 09:30:00 | 28.68 | 2026-02-24 09:35:00 | 28.75 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-25 09:50:00 | 29.85 | 2026-02-25 10:00:00 | 30.08 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2026-02-25 09:50:00 | 29.85 | 2026-02-25 10:05:00 | 29.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 11:05:00 | 27.37 | 2026-03-05 11:35:00 | 27.25 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-03-05 11:05:00 | 27.37 | 2026-03-05 14:45:00 | 27.34 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2026-03-06 10:05:00 | 27.51 | 2026-03-06 10:35:00 | 27.43 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-10 09:55:00 | 26.29 | 2026-03-10 10:05:00 | 26.39 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-13 10:00:00 | 26.07 | 2026-03-13 10:20:00 | 25.93 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-03-13 10:00:00 | 26.07 | 2026-03-13 10:50:00 | 26.07 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 09:30:00 | 26.33 | 2026-04-10 09:35:00 | 26.23 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-04-16 09:55:00 | 26.55 | 2026-04-16 10:15:00 | 26.63 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-21 09:30:00 | 26.87 | 2026-04-21 11:50:00 | 26.75 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-04-24 09:30:00 | 26.54 | 2026-04-24 09:50:00 | 26.41 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-04-24 09:30:00 | 26.54 | 2026-04-24 14:30:00 | 26.47 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2026-04-29 10:20:00 | 26.73 | 2026-04-29 10:30:00 | 26.65 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-30 10:00:00 | 26.36 | 2026-04-30 10:40:00 | 26.45 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-05-08 09:35:00 | 27.03 | 2026-05-08 11:25:00 | 26.91 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-05-08 09:35:00 | 27.03 | 2026-05-08 14:45:00 | 27.03 | STOP_HIT | 0.50 | 0.00% |
