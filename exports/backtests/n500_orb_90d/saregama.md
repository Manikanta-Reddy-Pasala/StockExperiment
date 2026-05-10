# Saregama India Ltd (SAREGAMA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 360.00
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
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 9
- **Target hits / Stop hits / Partials:** 3 / 9 / 5
- **Avg / median % per leg:** 0.35% / 0.00%
- **Sum % (uncompounded):** 6.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.65% | 4.5% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.65% | 4.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.15% | 1.5% |
| SELL @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.15% | 1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 8 | 47.1% | 3 | 9 | 5 | 0.35% | 6.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:25:00 | 344.15 | 341.56 | 0.00 | ORB-long ORB[338.15,342.10] vol=2.1x ATR=1.22 |
| Stop hit — per-position SL triggered | 2026-02-17 10:30:00 | 342.93 | 341.64 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:50:00 | 324.55 | 327.55 | 0.00 | ORB-short ORB[327.60,331.20] vol=8.8x ATR=1.25 |
| Stop hit — per-position SL triggered | 2026-02-25 11:05:00 | 325.80 | 326.77 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-03-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 09:55:00 | 322.00 | 319.55 | 0.00 | ORB-long ORB[316.65,319.55] vol=2.4x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:00:00 | 323.94 | 320.38 | 0.00 | T1 1.5R @ 323.94 |
| Target hit | 2026-03-05 15:20:00 | 328.00 | 323.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-03-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:10:00 | 332.60 | 330.07 | 0.00 | ORB-long ORB[327.25,331.75] vol=3.5x ATR=1.30 |
| Stop hit — per-position SL triggered | 2026-03-11 10:15:00 | 331.30 | 330.34 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 10:55:00 | 314.30 | 315.08 | 0.00 | ORB-short ORB[316.05,320.00] vol=1.9x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 12:35:00 | 312.88 | 314.80 | 0.00 | T1 1.5R @ 312.88 |
| Target hit | 2026-04-07 15:20:00 | 310.05 | 313.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:35:00 | 342.00 | 343.66 | 0.00 | ORB-short ORB[343.10,346.75] vol=2.3x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:40:00 | 340.01 | 343.28 | 0.00 | T1 1.5R @ 340.01 |
| Stop hit — per-position SL triggered | 2026-04-16 10:50:00 | 342.00 | 341.80 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:45:00 | 339.05 | 340.67 | 0.00 | ORB-short ORB[339.35,343.95] vol=2.1x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:55:00 | 337.16 | 340.16 | 0.00 | T1 1.5R @ 337.16 |
| Stop hit — per-position SL triggered | 2026-04-22 10:00:00 | 339.05 | 340.02 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:45:00 | 351.50 | 349.33 | 0.00 | ORB-long ORB[345.55,349.50] vol=5.4x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:50:00 | 353.64 | 350.32 | 0.00 | T1 1.5R @ 353.64 |
| Target hit | 2026-04-27 15:20:00 | 360.35 | 359.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2026-04-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:35:00 | 359.20 | 360.80 | 0.00 | ORB-short ORB[359.50,364.65] vol=1.7x ATR=1.79 |
| Stop hit — per-position SL triggered | 2026-04-28 09:40:00 | 360.99 | 360.79 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-05-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:10:00 | 348.30 | 346.07 | 0.00 | ORB-long ORB[344.00,346.90] vol=7.9x ATR=1.07 |
| Stop hit — per-position SL triggered | 2026-05-04 12:10:00 | 347.23 | 346.90 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:55:00 | 343.30 | 344.94 | 0.00 | ORB-short ORB[344.30,347.10] vol=1.7x ATR=0.85 |
| Stop hit — per-position SL triggered | 2026-05-06 11:10:00 | 344.15 | 344.77 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:40:00 | 347.95 | 349.73 | 0.00 | ORB-short ORB[348.80,352.45] vol=1.6x ATR=1.16 |
| Stop hit — per-position SL triggered | 2026-05-08 09:45:00 | 349.11 | 349.68 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 10:25:00 | 344.15 | 2026-02-17 10:30:00 | 342.93 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-25 10:50:00 | 324.55 | 2026-02-25 11:05:00 | 325.80 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-03-05 09:55:00 | 322.00 | 2026-03-05 10:00:00 | 323.94 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-03-05 09:55:00 | 322.00 | 2026-03-05 15:20:00 | 328.00 | TARGET_HIT | 0.50 | 1.86% |
| BUY | retest1 | 2026-03-11 10:10:00 | 332.60 | 2026-03-11 10:15:00 | 331.30 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-04-07 10:55:00 | 314.30 | 2026-04-07 12:35:00 | 312.88 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-04-07 10:55:00 | 314.30 | 2026-04-07 15:20:00 | 310.05 | TARGET_HIT | 0.50 | 1.35% |
| SELL | retest1 | 2026-04-16 09:35:00 | 342.00 | 2026-04-16 09:40:00 | 340.01 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-04-16 09:35:00 | 342.00 | 2026-04-16 10:50:00 | 342.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-22 09:45:00 | 339.05 | 2026-04-22 09:55:00 | 337.16 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-04-22 09:45:00 | 339.05 | 2026-04-22 10:00:00 | 339.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 09:45:00 | 351.50 | 2026-04-27 09:50:00 | 353.64 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-04-27 09:45:00 | 351.50 | 2026-04-27 15:20:00 | 360.35 | TARGET_HIT | 0.50 | 2.52% |
| SELL | retest1 | 2026-04-28 09:35:00 | 359.20 | 2026-04-28 09:40:00 | 360.99 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-05-04 11:10:00 | 348.30 | 2026-05-04 12:10:00 | 347.23 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-05-06 10:55:00 | 343.30 | 2026-05-06 11:10:00 | 344.15 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-05-08 09:40:00 | 347.95 | 2026-05-08 09:45:00 | 349.11 | STOP_HIT | 1.00 | -0.33% |
