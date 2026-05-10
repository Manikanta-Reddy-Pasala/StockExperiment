# Allied Blenders and Distillers Ltd. (ABDL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 594.00
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
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 10
- **Target hits / Stop hits / Partials:** 2 / 10 / 7
- **Avg / median % per leg:** 0.35% / 0.00%
- **Sum % (uncompounded):** 6.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | -0.02% | -0.1% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | -0.02% | -0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 8 | 50.0% | 2 | 8 | 6 | 0.42% | 6.7% |
| SELL @ 2nd Alert (retest1) | 16 | 8 | 50.0% | 2 | 8 | 6 | 0.42% | 6.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 9 | 47.4% | 2 | 10 | 7 | 0.35% | 6.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:10:00 | 520.10 | 525.59 | 0.00 | ORB-short ORB[523.30,529.60] vol=3.5x ATR=1.51 |
| Stop hit — per-position SL triggered | 2026-02-18 11:45:00 | 521.61 | 524.87 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:10:00 | 500.50 | 503.46 | 0.00 | ORB-short ORB[501.95,508.00] vol=1.9x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:25:00 | 498.42 | 503.02 | 0.00 | T1 1.5R @ 498.42 |
| Stop hit — per-position SL triggered | 2026-02-23 11:30:00 | 500.50 | 502.92 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:50:00 | 494.50 | 498.00 | 0.00 | ORB-short ORB[496.00,503.00] vol=1.7x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:10:00 | 492.01 | 497.11 | 0.00 | T1 1.5R @ 492.01 |
| Target hit | 2026-02-25 15:20:00 | 490.00 | 490.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-03-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:45:00 | 460.45 | 463.70 | 0.00 | ORB-short ORB[462.10,468.95] vol=3.3x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:15:00 | 456.91 | 462.40 | 0.00 | T1 1.5R @ 456.91 |
| Target hit | 2026-03-13 15:20:00 | 441.45 | 448.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-03-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:35:00 | 440.20 | 445.91 | 0.00 | ORB-short ORB[441.75,448.00] vol=1.9x ATR=2.14 |
| Stop hit — per-position SL triggered | 2026-03-16 10:55:00 | 442.34 | 445.35 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 11:00:00 | 413.25 | 416.02 | 0.00 | ORB-short ORB[414.30,418.55] vol=2.5x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 11:05:00 | 411.28 | 415.64 | 0.00 | T1 1.5R @ 411.28 |
| Stop hit — per-position SL triggered | 2026-03-20 11:10:00 | 413.25 | 415.62 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:50:00 | 568.20 | 571.78 | 0.00 | ORB-short ORB[572.15,578.90] vol=1.6x ATR=2.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 12:35:00 | 564.65 | 570.59 | 0.00 | T1 1.5R @ 564.65 |
| Stop hit — per-position SL triggered | 2026-04-21 14:55:00 | 568.20 | 568.46 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:50:00 | 556.30 | 563.66 | 0.00 | ORB-short ORB[562.00,569.95] vol=2.0x ATR=2.95 |
| Stop hit — per-position SL triggered | 2026-04-24 09:55:00 | 559.25 | 563.46 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 550.60 | 554.42 | 0.00 | ORB-short ORB[551.80,559.35] vol=1.9x ATR=2.03 |
| Stop hit — per-position SL triggered | 2026-04-28 11:15:00 | 552.63 | 551.41 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 11:00:00 | 528.10 | 533.82 | 0.00 | ORB-short ORB[535.00,540.25] vol=2.0x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 12:25:00 | 525.24 | 531.33 | 0.00 | T1 1.5R @ 525.24 |
| Stop hit — per-position SL triggered | 2026-05-04 12:50:00 | 528.10 | 530.05 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 10:00:00 | 539.70 | 535.59 | 0.00 | ORB-long ORB[533.10,536.80] vol=1.7x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:05:00 | 542.35 | 535.87 | 0.00 | T1 1.5R @ 542.35 |
| Stop hit — per-position SL triggered | 2026-05-05 10:10:00 | 539.70 | 536.23 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:40:00 | 566.80 | 560.03 | 0.00 | ORB-long ORB[552.20,559.00] vol=6.8x ATR=3.12 |
| Stop hit — per-position SL triggered | 2026-05-07 09:45:00 | 563.68 | 560.47 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-18 11:10:00 | 520.10 | 2026-02-18 11:45:00 | 521.61 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-23 11:10:00 | 500.50 | 2026-02-23 11:25:00 | 498.42 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-23 11:10:00 | 500.50 | 2026-02-23 11:30:00 | 500.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-25 10:50:00 | 494.50 | 2026-02-25 11:10:00 | 492.01 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-02-25 10:50:00 | 494.50 | 2026-02-25 15:20:00 | 490.00 | TARGET_HIT | 0.50 | 0.91% |
| SELL | retest1 | 2026-03-13 09:45:00 | 460.45 | 2026-03-13 10:15:00 | 456.91 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2026-03-13 09:45:00 | 460.45 | 2026-03-13 15:20:00 | 441.45 | TARGET_HIT | 0.50 | 4.13% |
| SELL | retest1 | 2026-03-16 10:35:00 | 440.20 | 2026-03-16 10:55:00 | 442.34 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-03-20 11:00:00 | 413.25 | 2026-03-20 11:05:00 | 411.28 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-03-20 11:00:00 | 413.25 | 2026-03-20 11:10:00 | 413.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-21 10:50:00 | 568.20 | 2026-04-21 12:35:00 | 564.65 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-04-21 10:50:00 | 568.20 | 2026-04-21 14:55:00 | 568.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 09:50:00 | 556.30 | 2026-04-24 09:55:00 | 559.25 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2026-04-28 09:30:00 | 550.60 | 2026-04-28 11:15:00 | 552.63 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-05-04 11:00:00 | 528.10 | 2026-05-04 12:25:00 | 525.24 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-05-04 11:00:00 | 528.10 | 2026-05-04 12:50:00 | 528.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 10:00:00 | 539.70 | 2026-05-05 10:05:00 | 542.35 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-05-05 10:00:00 | 539.70 | 2026-05-05 10:10:00 | 539.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-07 09:40:00 | 566.80 | 2026-05-07 09:45:00 | 563.68 | STOP_HIT | 1.00 | -0.55% |
