# Titagarh Rail Systems Ltd. (TITAGARH)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 840.00
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
- **Avg / median % per leg:** -0.13% / -0.30%
- **Sum % (uncompounded):** -1.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.13% | -0.6% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.13% | -0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.12% | -0.9% |
| SELL @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.12% | -0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 3 | 25.0% | 1 | 9 | 2 | -0.13% | -1.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:35:00 | 777.95 | 784.03 | 0.00 | ORB-short ORB[783.00,793.60] vol=4.2x ATR=2.63 |
| Stop hit — per-position SL triggered | 2026-02-12 11:00:00 | 780.58 | 782.83 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:10:00 | 761.65 | 766.29 | 0.00 | ORB-short ORB[764.95,771.95] vol=1.6x ATR=2.31 |
| Stop hit — per-position SL triggered | 2026-02-18 10:50:00 | 763.96 | 765.65 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:40:00 | 731.55 | 735.31 | 0.00 | ORB-short ORB[734.65,738.50] vol=1.5x ATR=2.06 |
| Stop hit — per-position SL triggered | 2026-02-25 10:05:00 | 733.61 | 734.81 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:35:00 | 704.85 | 708.09 | 0.00 | ORB-short ORB[705.90,714.75] vol=1.7x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:05:00 | 701.31 | 706.31 | 0.00 | T1 1.5R @ 701.31 |
| Target hit | 2026-02-27 14:40:00 | 702.80 | 702.79 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 718.70 | 715.39 | 0.00 | ORB-long ORB[710.65,717.90] vol=2.7x ATR=2.88 |
| Stop hit — per-position SL triggered | 2026-04-10 09:40:00 | 715.82 | 715.65 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:05:00 | 709.65 | 719.64 | 0.00 | ORB-short ORB[721.20,730.00] vol=1.9x ATR=2.89 |
| Stop hit — per-position SL triggered | 2026-04-16 10:25:00 | 712.54 | 717.89 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:45:00 | 742.40 | 738.53 | 0.00 | ORB-long ORB[733.50,742.00] vol=4.5x ATR=2.33 |
| Stop hit — per-position SL triggered | 2026-04-22 10:55:00 | 740.07 | 738.85 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:15:00 | 799.65 | 791.35 | 0.00 | ORB-long ORB[787.20,796.00] vol=4.0x ATR=2.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:20:00 | 803.23 | 793.47 | 0.00 | T1 1.5R @ 803.23 |
| Stop hit — per-position SL triggered | 2026-04-29 11:25:00 | 799.65 | 793.66 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:00:00 | 778.50 | 773.67 | 0.00 | ORB-long ORB[769.00,777.80] vol=5.0x ATR=2.81 |
| Stop hit — per-position SL triggered | 2026-05-04 11:10:00 | 775.69 | 774.18 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:05:00 | 834.00 | 841.23 | 0.00 | ORB-short ORB[840.10,850.65] vol=1.8x ATR=2.83 |
| Stop hit — per-position SL triggered | 2026-05-07 11:30:00 | 836.83 | 840.68 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 10:35:00 | 777.95 | 2026-02-12 11:00:00 | 780.58 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-18 10:10:00 | 761.65 | 2026-02-18 10:50:00 | 763.96 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-25 09:40:00 | 731.55 | 2026-02-25 10:05:00 | 733.61 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-27 09:35:00 | 704.85 | 2026-02-27 10:05:00 | 701.31 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-02-27 09:35:00 | 704.85 | 2026-02-27 14:40:00 | 702.80 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2026-04-10 09:30:00 | 718.70 | 2026-04-10 09:40:00 | 715.82 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-16 10:05:00 | 709.65 | 2026-04-16 10:25:00 | 712.54 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-22 10:45:00 | 742.40 | 2026-04-22 10:55:00 | 740.07 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-29 11:15:00 | 799.65 | 2026-04-29 11:20:00 | 803.23 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-04-29 11:15:00 | 799.65 | 2026-04-29 11:25:00 | 799.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 11:00:00 | 778.50 | 2026-05-04 11:10:00 | 775.69 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-05-07 11:05:00 | 834.00 | 2026-05-07 11:30:00 | 836.83 | STOP_HIT | 1.00 | -0.34% |
