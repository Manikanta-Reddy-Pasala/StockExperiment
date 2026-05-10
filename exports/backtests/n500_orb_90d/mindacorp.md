# Minda Corporation Ltd. (MINDACORP)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 537.80
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
| ENTRY1 | 10 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 9
- **Target hits / Stop hits / Partials:** 1 / 9 / 2
- **Avg / median % per leg:** -0.00% / -0.25%
- **Sum % (uncompounded):** -0.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 1 | 6 | 1 | 0.03% | 0.3% |
| BUY @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 1 | 6 | 1 | 0.03% | 0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.07% | -0.3% |
| SELL @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.07% | -0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 3 | 25.0% | 1 | 9 | 2 | -0.00% | -0.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:50:00 | 578.35 | 573.30 | 0.00 | ORB-long ORB[565.60,574.00] vol=1.7x ATR=1.96 |
| Stop hit — per-position SL triggered | 2026-02-20 11:05:00 | 576.39 | 573.54 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 11:05:00 | 572.40 | 574.33 | 0.00 | ORB-short ORB[572.70,581.00] vol=5.5x ATR=1.42 |
| Stop hit — per-position SL triggered | 2026-02-24 11:10:00 | 573.82 | 574.32 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:15:00 | 528.00 | 529.49 | 0.00 | ORB-short ORB[529.15,534.10] vol=3.1x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:25:00 | 526.07 | 529.43 | 0.00 | T1 1.5R @ 526.07 |
| Stop hit — per-position SL triggered | 2026-03-05 12:15:00 | 528.00 | 528.97 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:00:00 | 484.90 | 482.83 | 0.00 | ORB-long ORB[477.00,483.50] vol=3.3x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-03-17 10:30:00 | 483.34 | 483.26 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 11:00:00 | 501.25 | 497.00 | 0.00 | ORB-long ORB[494.35,499.90] vol=2.2x ATR=1.75 |
| Stop hit — per-position SL triggered | 2026-03-19 14:35:00 | 499.50 | 498.93 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 09:45:00 | 497.35 | 501.78 | 0.00 | ORB-short ORB[501.05,508.00] vol=1.5x ATR=1.97 |
| Stop hit — per-position SL triggered | 2026-03-20 09:50:00 | 499.32 | 500.41 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:15:00 | 524.00 | 518.31 | 0.00 | ORB-long ORB[507.00,514.65] vol=1.7x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:35:00 | 527.15 | 520.39 | 0.00 | T1 1.5R @ 527.15 |
| Target hit | 2026-04-17 15:20:00 | 532.75 | 531.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:15:00 | 542.55 | 540.98 | 0.00 | ORB-long ORB[536.65,540.80] vol=1.8x ATR=1.33 |
| Stop hit — per-position SL triggered | 2026-04-21 12:00:00 | 541.22 | 541.27 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 551.00 | 546.36 | 0.00 | ORB-long ORB[541.10,548.70] vol=3.5x ATR=2.35 |
| Stop hit — per-position SL triggered | 2026-04-22 09:40:00 | 548.65 | 546.63 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-05-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:45:00 | 541.45 | 535.35 | 0.00 | ORB-long ORB[531.60,538.00] vol=2.0x ATR=1.82 |
| Stop hit — per-position SL triggered | 2026-05-08 11:00:00 | 539.63 | 536.60 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-20 10:50:00 | 578.35 | 2026-02-20 11:05:00 | 576.39 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-24 11:05:00 | 572.40 | 2026-02-24 11:10:00 | 573.82 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-05 11:15:00 | 528.00 | 2026-03-05 11:25:00 | 526.07 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-03-05 11:15:00 | 528.00 | 2026-03-05 12:15:00 | 528.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:00:00 | 484.90 | 2026-03-17 10:30:00 | 483.34 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-19 11:00:00 | 501.25 | 2026-03-19 14:35:00 | 499.50 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-20 09:45:00 | 497.35 | 2026-03-20 09:50:00 | 499.32 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-17 10:15:00 | 524.00 | 2026-04-17 10:35:00 | 527.15 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-17 10:15:00 | 524.00 | 2026-04-17 15:20:00 | 532.75 | TARGET_HIT | 0.50 | 1.67% |
| BUY | retest1 | 2026-04-21 11:15:00 | 542.55 | 2026-04-21 12:00:00 | 541.22 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-22 09:35:00 | 551.00 | 2026-04-22 09:40:00 | 548.65 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-05-08 10:45:00 | 541.45 | 2026-05-08 11:00:00 | 539.63 | STOP_HIT | 1.00 | -0.34% |
