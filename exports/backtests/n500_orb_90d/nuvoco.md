# Nuvoco Vistas Corporation Ltd. (NUVOCO)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 328.90
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 9
- **Target hits / Stop hits / Partials:** 0 / 9 / 0
- **Avg / median % per leg:** -0.41% / -0.42%
- **Sum % (uncompounded):** -3.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.43% | -2.6% |
| BUY @ 2nd Alert (retest1) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.43% | -2.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.37% | -1.1% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.37% | -1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 0 | 0.0% | 0 | 9 | 0 | -0.41% | -3.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:55:00 | 345.00 | 345.65 | 0.00 | ORB-short ORB[345.10,350.00] vol=3.8x ATR=1.22 |
| Stop hit — per-position SL triggered | 2026-02-11 11:00:00 | 346.22 | 345.78 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 330.70 | 331.17 | 0.00 | ORB-short ORB[331.00,333.80] vol=5.6x ATR=1.02 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 331.72 | 331.15 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:35:00 | 334.35 | 333.07 | 0.00 | ORB-long ORB[331.10,333.20] vol=4.9x ATR=1.06 |
| Stop hit — per-position SL triggered | 2026-02-26 11:30:00 | 333.29 | 333.60 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 09:30:00 | 303.15 | 300.64 | 0.00 | ORB-long ORB[298.15,302.35] vol=1.7x ATR=1.78 |
| Stop hit — per-position SL triggered | 2026-03-27 09:35:00 | 301.37 | 300.69 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 09:30:00 | 305.35 | 309.58 | 0.00 | ORB-short ORB[309.20,313.75] vol=4.0x ATR=1.35 |
| Stop hit — per-position SL triggered | 2026-04-17 09:40:00 | 306.70 | 308.23 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 295.75 | 294.63 | 0.00 | ORB-long ORB[291.90,295.70] vol=2.7x ATR=0.83 |
| Stop hit — per-position SL triggered | 2026-04-27 09:35:00 | 294.92 | 294.67 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 10:05:00 | 289.20 | 286.13 | 0.00 | ORB-long ORB[283.40,287.60] vol=2.8x ATR=1.21 |
| Stop hit — per-position SL triggered | 2026-04-30 10:10:00 | 287.99 | 286.19 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-05-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:50:00 | 293.15 | 291.74 | 0.00 | ORB-long ORB[288.45,291.95] vol=2.7x ATR=1.32 |
| Stop hit — per-position SL triggered | 2026-05-05 10:10:00 | 291.83 | 291.85 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:00:00 | 306.85 | 301.51 | 0.00 | ORB-long ORB[297.15,301.00] vol=6.5x ATR=1.66 |
| Stop hit — per-position SL triggered | 2026-05-06 10:05:00 | 305.19 | 302.69 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 10:55:00 | 345.00 | 2026-02-11 11:00:00 | 346.22 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-24 09:35:00 | 330.70 | 2026-02-24 09:45:00 | 331.72 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-26 09:35:00 | 334.35 | 2026-02-26 11:30:00 | 333.29 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-27 09:30:00 | 303.15 | 2026-03-27 09:35:00 | 301.37 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest1 | 2026-04-17 09:30:00 | 305.35 | 2026-04-17 09:40:00 | 306.70 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-27 09:30:00 | 295.75 | 2026-04-27 09:35:00 | 294.92 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-30 10:05:00 | 289.20 | 2026-04-30 10:10:00 | 287.99 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-05-05 09:50:00 | 293.15 | 2026-05-05 10:10:00 | 291.83 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-05-06 10:00:00 | 306.85 | 2026-05-06 10:05:00 | 305.19 | STOP_HIT | 1.00 | -0.54% |
