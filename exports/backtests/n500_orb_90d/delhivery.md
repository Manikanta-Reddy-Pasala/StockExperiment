# Delhivery Ltd. (DELHIVERY)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 479.00
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
| ENTRY1 | 15 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 14
- **Target hits / Stop hits / Partials:** 1 / 14 / 5
- **Avg / median % per leg:** -0.03% / 0.00%
- **Sum % (uncompounded):** -0.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | -0.09% | -0.4% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | -0.09% | -0.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 4 | 26.7% | 0 | 11 | 4 | -0.01% | -0.2% |
| SELL @ 2nd Alert (retest1) | 15 | 4 | 26.7% | 0 | 11 | 4 | -0.01% | -0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 6 | 30.0% | 1 | 14 | 5 | -0.03% | -0.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:55:00 | 435.50 | 432.14 | 0.00 | ORB-long ORB[428.20,434.00] vol=8.9x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 11:00:00 | 437.89 | 432.49 | 0.00 | T1 1.5R @ 437.89 |
| Target hit | 2026-02-10 12:50:00 | 436.35 | 436.67 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 425.15 | 426.33 | 0.00 | ORB-short ORB[425.20,429.65] vol=1.5x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:30:00 | 423.67 | 426.09 | 0.00 | T1 1.5R @ 423.67 |
| Stop hit — per-position SL triggered | 2026-02-12 13:25:00 | 425.15 | 425.06 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:50:00 | 432.30 | 434.56 | 0.00 | ORB-short ORB[434.75,438.55] vol=1.7x ATR=1.01 |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 433.31 | 434.30 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 11:15:00 | 426.40 | 429.76 | 0.00 | ORB-short ORB[426.45,432.70] vol=1.8x ATR=1.28 |
| Stop hit — per-position SL triggered | 2026-02-20 11:25:00 | 427.68 | 429.67 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:15:00 | 440.40 | 439.42 | 0.00 | ORB-long ORB[436.15,440.00] vol=1.9x ATR=1.41 |
| Stop hit — per-position SL triggered | 2026-02-25 10:55:00 | 438.99 | 439.60 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:40:00 | 446.40 | 442.80 | 0.00 | ORB-long ORB[438.05,442.85] vol=2.9x ATR=1.36 |
| Stop hit — per-position SL triggered | 2026-02-26 11:00:00 | 445.04 | 443.07 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 09:30:00 | 423.90 | 420.68 | 0.00 | ORB-long ORB[417.70,422.25] vol=1.6x ATR=2.36 |
| Stop hit — per-position SL triggered | 2026-03-04 09:35:00 | 421.54 | 420.90 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:15:00 | 398.35 | 401.39 | 0.00 | ORB-short ORB[402.40,406.95] vol=3.4x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:35:00 | 396.13 | 400.46 | 0.00 | T1 1.5R @ 396.13 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 398.35 | 400.10 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:55:00 | 394.85 | 400.15 | 0.00 | ORB-short ORB[399.55,404.85] vol=3.3x ATR=1.61 |
| Stop hit — per-position SL triggered | 2026-03-16 11:05:00 | 396.46 | 399.83 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 11:00:00 | 413.00 | 414.21 | 0.00 | ORB-short ORB[413.80,418.80] vol=6.4x ATR=1.69 |
| Stop hit — per-position SL triggered | 2026-03-19 12:05:00 | 414.69 | 413.91 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 09:55:00 | 411.50 | 415.03 | 0.00 | ORB-short ORB[414.80,420.85] vol=1.6x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-24 10:20:00 | 408.24 | 413.60 | 0.00 | T1 1.5R @ 408.24 |
| Stop hit — per-position SL triggered | 2026-03-24 11:30:00 | 411.50 | 410.98 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 10:50:00 | 416.60 | 418.71 | 0.00 | ORB-short ORB[418.00,423.70] vol=2.6x ATR=1.67 |
| Stop hit — per-position SL triggered | 2026-03-30 10:55:00 | 418.27 | 418.65 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:10:00 | 453.55 | 456.27 | 0.00 | ORB-short ORB[455.10,460.00] vol=1.7x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 10:45:00 | 451.02 | 454.88 | 0.00 | T1 1.5R @ 451.02 |
| Stop hit — per-position SL triggered | 2026-04-23 10:50:00 | 453.55 | 454.83 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:15:00 | 442.90 | 446.99 | 0.00 | ORB-short ORB[443.00,449.00] vol=4.0x ATR=1.46 |
| Stop hit — per-position SL triggered | 2026-04-24 11:20:00 | 444.36 | 446.73 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-05-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:45:00 | 481.00 | 483.63 | 0.00 | ORB-short ORB[482.10,487.50] vol=1.6x ATR=1.79 |
| Stop hit — per-position SL triggered | 2026-05-08 09:55:00 | 482.79 | 483.54 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:55:00 | 435.50 | 2026-02-10 11:00:00 | 437.89 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-02-10 10:55:00 | 435.50 | 2026-02-10 12:50:00 | 436.35 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2026-02-12 11:15:00 | 425.15 | 2026-02-12 11:30:00 | 423.67 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-12 11:15:00 | 425.15 | 2026-02-12 13:25:00 | 425.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 10:50:00 | 432.30 | 2026-02-18 11:15:00 | 433.31 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-20 11:15:00 | 426.40 | 2026-02-20 11:25:00 | 427.68 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-25 10:15:00 | 440.40 | 2026-02-25 10:55:00 | 438.99 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-26 10:40:00 | 446.40 | 2026-02-26 11:00:00 | 445.04 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-04 09:30:00 | 423.90 | 2026-03-04 09:35:00 | 421.54 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2026-03-13 10:15:00 | 398.35 | 2026-03-13 10:35:00 | 396.13 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-03-13 10:15:00 | 398.35 | 2026-03-13 10:50:00 | 398.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-16 10:55:00 | 394.85 | 2026-03-16 11:05:00 | 396.46 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-03-19 11:00:00 | 413.00 | 2026-03-19 12:05:00 | 414.69 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-03-24 09:55:00 | 411.50 | 2026-03-24 10:20:00 | 408.24 | PARTIAL | 0.50 | 0.79% |
| SELL | retest1 | 2026-03-24 09:55:00 | 411.50 | 2026-03-24 11:30:00 | 411.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-30 10:50:00 | 416.60 | 2026-03-30 10:55:00 | 418.27 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-23 10:10:00 | 453.55 | 2026-04-23 10:45:00 | 451.02 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-04-23 10:10:00 | 453.55 | 2026-04-23 10:50:00 | 453.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 11:15:00 | 442.90 | 2026-04-24 11:20:00 | 444.36 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-05-08 09:45:00 | 481.00 | 2026-05-08 09:55:00 | 482.79 | STOP_HIT | 1.00 | -0.37% |
