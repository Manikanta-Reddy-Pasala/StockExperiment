# KPIT Technologies Ltd. (KPITTECH)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 725.00
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
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 4
- **Target hits / Stop hits / Partials:** 3 / 4 / 3
- **Avg / median % per leg:** 0.53% / 0.43%
- **Sum % (uncompounded):** 5.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | -0.12% | -0.6% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | -0.12% | -0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 1.18% | 5.9% |
| SELL @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 2 | 1 | 2 | 1.18% | 5.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 6 | 60.0% | 3 | 4 | 3 | 0.53% | 5.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 974.10 | 970.57 | 0.00 | ORB-long ORB[964.10,973.00] vol=2.7x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:35:00 | 978.14 | 971.55 | 0.00 | T1 1.5R @ 978.14 |
| Target hit | 2026-02-10 10:10:00 | 978.30 | 979.55 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 856.40 | 861.84 | 0.00 | ORB-short ORB[860.00,869.10] vol=2.7x ATR=2.61 |
| Stop hit — per-position SL triggered | 2026-02-19 09:45:00 | 859.01 | 860.70 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:10:00 | 796.20 | 797.68 | 0.00 | ORB-short ORB[797.50,808.40] vol=1.7x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:35:00 | 791.91 | 797.46 | 0.00 | T1 1.5R @ 791.91 |
| Target hit | 2026-02-27 15:20:00 | 770.50 | 787.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-03-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:05:00 | 680.50 | 674.32 | 0.00 | ORB-long ORB[665.25,673.50] vol=1.7x ATR=2.95 |
| Stop hit — per-position SL triggered | 2026-03-25 11:45:00 | 677.55 | 677.93 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 721.85 | 727.74 | 0.00 | ORB-short ORB[723.15,733.95] vol=1.8x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:50:00 | 717.69 | 724.41 | 0.00 | T1 1.5R @ 717.69 |
| Target hit | 2026-04-24 14:35:00 | 708.40 | 706.79 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — BUY (started 2026-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:30:00 | 746.65 | 735.31 | 0.00 | ORB-long ORB[727.65,738.05] vol=2.2x ATR=3.71 |
| Stop hit — per-position SL triggered | 2026-04-30 09:40:00 | 742.94 | 738.77 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-05-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:45:00 | 773.00 | 765.09 | 0.00 | ORB-long ORB[756.55,767.50] vol=1.6x ATR=4.08 |
| Stop hit — per-position SL triggered | 2026-05-04 11:40:00 | 768.92 | 769.30 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:30:00 | 974.10 | 2026-02-10 09:35:00 | 978.14 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-02-10 09:30:00 | 974.10 | 2026-02-10 10:10:00 | 978.30 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-19 09:30:00 | 856.40 | 2026-02-19 09:45:00 | 859.01 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-27 11:10:00 | 796.20 | 2026-02-27 11:35:00 | 791.91 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-02-27 11:10:00 | 796.20 | 2026-02-27 15:20:00 | 770.50 | TARGET_HIT | 0.50 | 3.23% |
| BUY | retest1 | 2026-03-25 10:05:00 | 680.50 | 2026-03-25 11:45:00 | 677.55 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-04-24 09:30:00 | 721.85 | 2026-04-24 09:50:00 | 717.69 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-04-24 09:30:00 | 721.85 | 2026-04-24 14:35:00 | 708.40 | TARGET_HIT | 0.50 | 1.86% |
| BUY | retest1 | 2026-04-30 09:30:00 | 746.65 | 2026-04-30 09:40:00 | 742.94 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-05-04 09:45:00 | 773.00 | 2026-05-04 11:40:00 | 768.92 | STOP_HIT | 1.00 | -0.53% |
