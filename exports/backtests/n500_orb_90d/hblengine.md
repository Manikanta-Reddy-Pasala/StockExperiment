# HBL Engineering Ltd. (HBLENGINE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 850.05
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
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 8
- **Target hits / Stop hits / Partials:** 1 / 8 / 4
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 1.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.11% | -0.6% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.11% | -0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.22% | 1.8% |
| SELL @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.22% | 1.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 5 | 38.5% | 1 | 8 | 4 | 0.09% | 1.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:10:00 | 740.40 | 746.17 | 0.00 | ORB-short ORB[742.80,752.00] vol=1.8x ATR=2.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:15:00 | 736.81 | 743.67 | 0.00 | T1 1.5R @ 736.81 |
| Target hit | 2026-02-18 15:20:00 | 732.50 | 735.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:05:00 | 714.40 | 721.57 | 0.00 | ORB-short ORB[718.00,727.35] vol=1.7x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:30:00 | 710.29 | 719.90 | 0.00 | T1 1.5R @ 710.29 |
| Stop hit — per-position SL triggered | 2026-02-23 11:00:00 | 714.40 | 718.18 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:40:00 | 700.00 | 703.70 | 0.00 | ORB-short ORB[702.55,709.00] vol=1.8x ATR=2.68 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 702.68 | 703.58 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:05:00 | 714.80 | 708.56 | 0.00 | ORB-long ORB[703.80,712.25] vol=2.0x ATR=2.75 |
| Stop hit — per-position SL triggered | 2026-02-25 10:25:00 | 712.05 | 709.15 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:35:00 | 646.85 | 652.22 | 0.00 | ORB-short ORB[650.05,658.85] vol=1.8x ATR=3.17 |
| Stop hit — per-position SL triggered | 2026-03-05 09:40:00 | 650.02 | 651.86 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:55:00 | 675.45 | 662.39 | 0.00 | ORB-long ORB[655.50,664.90] vol=3.6x ATR=3.33 |
| Stop hit — per-position SL triggered | 2026-03-12 11:20:00 | 672.12 | 666.38 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 787.45 | 783.16 | 0.00 | ORB-long ORB[775.60,787.00] vol=1.6x ATR=3.03 |
| Stop hit — per-position SL triggered | 2026-04-21 11:55:00 | 784.42 | 787.24 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:40:00 | 807.70 | 810.48 | 0.00 | ORB-short ORB[807.95,816.90] vol=2.1x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:40:00 | 803.47 | 809.03 | 0.00 | T1 1.5R @ 803.47 |
| Stop hit — per-position SL triggered | 2026-04-28 11:00:00 | 807.70 | 808.83 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:45:00 | 819.30 | 812.15 | 0.00 | ORB-long ORB[805.05,815.45] vol=5.1x ATR=3.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:55:00 | 825.08 | 815.73 | 0.00 | T1 1.5R @ 825.08 |
| Stop hit — per-position SL triggered | 2026-05-06 11:00:00 | 819.30 | 816.58 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-18 10:10:00 | 740.40 | 2026-02-18 11:15:00 | 736.81 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-02-18 10:10:00 | 740.40 | 2026-02-18 15:20:00 | 732.50 | TARGET_HIT | 0.50 | 1.07% |
| SELL | retest1 | 2026-02-23 10:05:00 | 714.40 | 2026-02-23 10:30:00 | 710.29 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-02-23 10:05:00 | 714.40 | 2026-02-23 11:00:00 | 714.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 09:40:00 | 700.00 | 2026-02-24 09:45:00 | 702.68 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-25 10:05:00 | 714.80 | 2026-02-25 10:25:00 | 712.05 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-03-05 09:35:00 | 646.85 | 2026-03-05 09:40:00 | 650.02 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-03-12 10:55:00 | 675.45 | 2026-03-12 11:20:00 | 672.12 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-04-21 09:35:00 | 787.45 | 2026-04-21 11:55:00 | 784.42 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-04-28 09:40:00 | 807.70 | 2026-04-28 10:40:00 | 803.47 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-04-28 09:40:00 | 807.70 | 2026-04-28 11:00:00 | 807.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 10:45:00 | 819.30 | 2026-05-06 10:55:00 | 825.08 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2026-05-06 10:45:00 | 819.30 | 2026-05-06 11:00:00 | 819.30 | STOP_HIT | 0.50 | 0.00% |
