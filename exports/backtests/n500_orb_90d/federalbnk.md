# Federal Bank Ltd. (FEDERALBNK)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 297.40
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
| ENTRY1 | 16 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 15
- **Target hits / Stop hits / Partials:** 1 / 15 / 5
- **Avg / median % per leg:** -0.04% / -0.18%
- **Sum % (uncompounded):** -0.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 4 | 28.6% | 0 | 10 | 4 | -0.01% | -0.1% |
| BUY @ 2nd Alert (retest1) | 14 | 4 | 28.6% | 0 | 10 | 4 | -0.01% | -0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.10% | -0.7% |
| SELL @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.10% | -0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 6 | 28.6% | 1 | 15 | 5 | -0.04% | -0.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:50:00 | 287.35 | 286.51 | 0.00 | ORB-long ORB[285.65,287.15] vol=2.8x ATR=0.52 |
| Stop hit — per-position SL triggered | 2026-02-10 10:55:00 | 286.83 | 286.54 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 11:10:00 | 290.20 | 285.02 | 0.00 | ORB-long ORB[279.65,283.05] vol=2.2x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:25:00 | 291.60 | 286.80 | 0.00 | T1 1.5R @ 291.60 |
| Stop hit — per-position SL triggered | 2026-02-11 14:00:00 | 290.20 | 289.96 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 09:40:00 | 289.00 | 287.71 | 0.00 | ORB-long ORB[285.10,288.20] vol=2.1x ATR=0.65 |
| Stop hit — per-position SL triggered | 2026-02-13 09:45:00 | 288.35 | 287.83 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:45:00 | 297.25 | 296.27 | 0.00 | ORB-long ORB[295.25,296.90] vol=1.8x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:55:00 | 298.21 | 296.55 | 0.00 | T1 1.5R @ 298.21 |
| Stop hit — per-position SL triggered | 2026-02-24 10:00:00 | 297.25 | 296.64 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:50:00 | 298.80 | 300.20 | 0.00 | ORB-short ORB[299.15,300.90] vol=4.3x ATR=0.62 |
| Stop hit — per-position SL triggered | 2026-02-26 11:10:00 | 299.42 | 300.01 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:45:00 | 287.10 | 290.09 | 0.00 | ORB-short ORB[289.60,293.75] vol=1.5x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 10:25:00 | 285.50 | 288.75 | 0.00 | T1 1.5R @ 285.50 |
| Target hit | 2026-03-04 14:10:00 | 286.75 | 286.59 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — BUY (started 2026-03-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:45:00 | 281.00 | 278.75 | 0.00 | ORB-long ORB[277.00,280.50] vol=1.9x ATR=1.09 |
| Stop hit — per-position SL triggered | 2026-03-10 10:55:00 | 279.91 | 278.83 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:35:00 | 263.30 | 262.40 | 0.00 | ORB-long ORB[260.45,263.00] vol=2.1x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:45:00 | 264.60 | 262.90 | 0.00 | T1 1.5R @ 264.60 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 263.30 | 263.08 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 11:10:00 | 257.75 | 259.66 | 0.00 | ORB-short ORB[259.00,262.25] vol=2.1x ATR=0.87 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 258.62 | 259.47 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:55:00 | 288.25 | 286.48 | 0.00 | ORB-long ORB[283.50,287.45] vol=3.8x ATR=0.97 |
| Stop hit — per-position SL triggered | 2026-04-13 11:05:00 | 287.28 | 286.55 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:15:00 | 289.10 | 291.94 | 0.00 | ORB-short ORB[292.25,294.90] vol=1.6x ATR=0.73 |
| Stop hit — per-position SL triggered | 2026-04-15 11:40:00 | 289.83 | 291.60 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:25:00 | 297.95 | 297.29 | 0.00 | ORB-long ORB[295.60,297.30] vol=2.0x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:45:00 | 298.93 | 297.57 | 0.00 | T1 1.5R @ 298.93 |
| Stop hit — per-position SL triggered | 2026-04-21 11:25:00 | 297.95 | 297.80 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:10:00 | 297.40 | 296.23 | 0.00 | ORB-long ORB[295.00,296.50] vol=1.7x ATR=0.74 |
| Stop hit — per-position SL triggered | 2026-04-22 10:30:00 | 296.66 | 296.40 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:45:00 | 292.70 | 293.91 | 0.00 | ORB-short ORB[292.85,295.85] vol=1.6x ATR=0.86 |
| Stop hit — per-position SL triggered | 2026-04-28 11:35:00 | 293.56 | 293.33 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 296.10 | 294.77 | 0.00 | ORB-long ORB[293.30,295.90] vol=1.5x ATR=1.16 |
| Stop hit — per-position SL triggered | 2026-05-06 09:40:00 | 294.94 | 294.88 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-05-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 11:00:00 | 293.00 | 294.05 | 0.00 | ORB-short ORB[293.95,297.00] vol=3.6x ATR=0.75 |
| Stop hit — per-position SL triggered | 2026-05-08 11:40:00 | 293.75 | 293.89 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:50:00 | 287.35 | 2026-02-10 10:55:00 | 286.83 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-02-11 11:10:00 | 290.20 | 2026-02-11 11:25:00 | 291.60 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-02-11 11:10:00 | 290.20 | 2026-02-11 14:00:00 | 290.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-13 09:40:00 | 289.00 | 2026-02-13 09:45:00 | 288.35 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-24 09:45:00 | 297.25 | 2026-02-24 09:55:00 | 298.21 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-24 09:45:00 | 297.25 | 2026-02-24 10:00:00 | 297.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-26 10:50:00 | 298.80 | 2026-02-26 11:10:00 | 299.42 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-03-04 09:45:00 | 287.10 | 2026-03-04 10:25:00 | 285.50 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-03-04 09:45:00 | 287.10 | 2026-03-04 14:10:00 | 286.75 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2026-03-10 10:45:00 | 281.00 | 2026-03-10 10:55:00 | 279.91 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-03-16 09:35:00 | 263.30 | 2026-03-16 09:45:00 | 264.60 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-03-16 09:35:00 | 263.30 | 2026-03-16 10:15:00 | 263.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-24 11:10:00 | 257.75 | 2026-03-24 11:15:00 | 258.62 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-13 10:55:00 | 288.25 | 2026-04-13 11:05:00 | 287.28 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-15 11:15:00 | 289.10 | 2026-04-15 11:40:00 | 289.83 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-21 10:25:00 | 297.95 | 2026-04-21 10:45:00 | 298.93 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-04-21 10:25:00 | 297.95 | 2026-04-21 11:25:00 | 297.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 10:10:00 | 297.40 | 2026-04-22 10:30:00 | 296.66 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-04-28 09:45:00 | 292.70 | 2026-04-28 11:35:00 | 293.56 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-06 09:30:00 | 296.10 | 2026-05-06 09:40:00 | 294.94 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-05-08 11:00:00 | 293.00 | 2026-05-08 11:40:00 | 293.75 | STOP_HIT | 1.00 | -0.26% |
