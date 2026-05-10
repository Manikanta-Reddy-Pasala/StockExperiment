# Cemindia Projects Ltd. (CEMPRO)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 955.20
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / Stop hits / Partials:** 0 / 7 / 1
- **Avg / median % per leg:** -0.24% / -0.39%
- **Sum % (uncompounded):** -1.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.22% | -1.5% |
| BUY @ 2nd Alert (retest1) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.22% | -1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.40% | -0.4% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.40% | -0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 1 | 12.5% | 0 | 7 | 1 | -0.24% | -1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 09:50:00 | 617.90 | 612.94 | 0.00 | ORB-long ORB[608.55,614.20] vol=1.5x ATR=3.23 |
| Stop hit — per-position SL triggered | 2026-02-13 09:55:00 | 614.67 | 613.34 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:50:00 | 603.80 | 596.99 | 0.00 | ORB-long ORB[589.15,596.55] vol=2.7x ATR=2.87 |
| Stop hit — per-position SL triggered | 2026-02-17 10:00:00 | 600.93 | 598.01 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-03-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:55:00 | 567.40 | 563.45 | 0.00 | ORB-long ORB[558.30,563.85] vol=4.3x ATR=2.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:10:00 | 571.29 | 564.25 | 0.00 | T1 1.5R @ 571.29 |
| Stop hit — per-position SL triggered | 2026-03-06 10:45:00 | 567.40 | 566.39 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:25:00 | 538.40 | 535.45 | 0.00 | ORB-long ORB[530.05,537.45] vol=2.9x ATR=2.55 |
| Stop hit — per-position SL triggered | 2026-03-17 11:05:00 | 535.85 | 535.80 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:35:00 | 639.15 | 642.43 | 0.00 | ORB-short ORB[640.00,646.30] vol=2.1x ATR=2.55 |
| Stop hit — per-position SL triggered | 2026-04-16 09:40:00 | 641.70 | 642.10 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 651.20 | 649.03 | 0.00 | ORB-long ORB[643.25,650.80] vol=2.9x ATR=2.55 |
| Stop hit — per-position SL triggered | 2026-04-22 09:50:00 | 648.65 | 649.73 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 11:10:00 | 669.50 | 666.31 | 0.00 | ORB-long ORB[660.00,667.50] vol=1.6x ATR=2.46 |
| Stop hit — per-position SL triggered | 2026-04-28 11:20:00 | 667.04 | 666.35 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-13 09:50:00 | 617.90 | 2026-02-13 09:55:00 | 614.67 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-02-17 09:50:00 | 603.80 | 2026-02-17 10:00:00 | 600.93 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-03-06 09:55:00 | 567.40 | 2026-03-06 10:10:00 | 571.29 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-03-06 09:55:00 | 567.40 | 2026-03-06 10:45:00 | 567.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:25:00 | 538.40 | 2026-03-17 11:05:00 | 535.85 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-04-16 09:35:00 | 639.15 | 2026-04-16 09:40:00 | 641.70 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-22 09:30:00 | 651.20 | 2026-04-22 09:50:00 | 648.65 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-28 11:10:00 | 669.50 | 2026-04-28 11:20:00 | 667.04 | STOP_HIT | 1.00 | -0.37% |
