# Container Corporation of India Ltd. (CONCOR)

## Backtest Summary

- **Window:** 2024-03-07 09:15:00 → 2026-05-08 15:25:00 (38446 bars)
- **Last close:** 533.75
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
| ENTRY1 | 19 |
| ENTRY2 | 0 |
| PARTIAL | 9 |
| TARGET_HIT | 3 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 16
- **Target hits / Stop hits / Partials:** 3 / 16 / 9
- **Avg / median % per leg:** 0.20% / 0.00%
- **Sum % (uncompounded):** 5.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 9 | 47.4% | 3 | 10 | 6 | 0.25% | 4.8% |
| BUY @ 2nd Alert (retest1) | 19 | 9 | 47.4% | 3 | 10 | 6 | 0.25% | 4.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 3 | 33.3% | 0 | 6 | 3 | 0.09% | 0.8% |
| SELL @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 0 | 6 | 3 | 0.09% | 0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 28 | 12 | 42.9% | 3 | 16 | 9 | 0.20% | 5.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-11 09:40:00 | 763.60 | 765.67 | 0.00 | ORB-short ORB[764.64,772.80] vol=3.0x ATR=2.27 |
| Stop hit — per-position SL triggered | 2024-03-11 10:20:00 | 765.87 | 764.46 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-03-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-15 09:30:00 | 685.00 | 690.94 | 0.00 | ORB-short ORB[688.56,696.60] vol=1.7x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 09:40:00 | 680.01 | 687.63 | 0.00 | T1 1.5R @ 680.01 |
| Stop hit — per-position SL triggered | 2024-03-15 09:45:00 | 685.00 | 686.80 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-03-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 10:45:00 | 671.56 | 678.98 | 0.00 | ORB-short ORB[677.64,685.40] vol=2.2x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-19 11:05:00 | 667.77 | 675.09 | 0.00 | T1 1.5R @ 667.77 |
| Stop hit — per-position SL triggered | 2024-03-19 11:10:00 | 671.56 | 674.90 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-03-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 09:45:00 | 685.60 | 683.68 | 0.00 | ORB-long ORB[677.04,684.52] vol=4.4x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-21 10:05:00 | 690.08 | 684.87 | 0.00 | T1 1.5R @ 690.08 |
| Stop hit — per-position SL triggered | 2024-03-21 10:30:00 | 685.60 | 687.10 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-03-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 09:50:00 | 700.80 | 696.09 | 0.00 | ORB-long ORB[692.24,696.68] vol=2.6x ATR=2.69 |
| Stop hit — per-position SL triggered | 2024-03-26 10:05:00 | 698.11 | 697.50 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-03-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 09:35:00 | 709.20 | 704.16 | 0.00 | ORB-long ORB[695.76,702.64] vol=4.1x ATR=2.51 |
| Stop hit — per-position SL triggered | 2024-03-27 09:40:00 | 706.69 | 704.68 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-03-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 09:50:00 | 701.36 | 695.89 | 0.00 | ORB-long ORB[691.68,697.48] vol=1.6x ATR=2.05 |
| Stop hit — per-position SL triggered | 2024-03-28 09:55:00 | 699.31 | 696.25 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-04-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 10:30:00 | 728.36 | 722.54 | 0.00 | ORB-long ORB[719.60,724.32] vol=2.6x ATR=2.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 10:45:00 | 732.30 | 724.20 | 0.00 | T1 1.5R @ 732.30 |
| Stop hit — per-position SL triggered | 2024-04-02 10:55:00 | 728.36 | 724.54 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-04-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:55:00 | 734.36 | 736.84 | 0.00 | ORB-short ORB[734.92,740.88] vol=1.6x ATR=2.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 11:05:00 | 730.52 | 736.49 | 0.00 | T1 1.5R @ 730.52 |
| Stop hit — per-position SL triggered | 2024-04-04 11:10:00 | 734.36 | 736.35 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-04-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-05 10:00:00 | 735.96 | 730.15 | 0.00 | ORB-long ORB[724.36,734.36] vol=1.7x ATR=2.61 |
| Stop hit — per-position SL triggered | 2024-04-05 10:05:00 | 733.35 | 730.69 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-04-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-08 09:45:00 | 737.72 | 743.79 | 0.00 | ORB-short ORB[741.40,749.36] vol=1.5x ATR=2.86 |
| Stop hit — per-position SL triggered | 2024-04-08 09:55:00 | 740.58 | 742.92 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-04-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 10:40:00 | 743.52 | 737.09 | 0.00 | ORB-long ORB[731.64,737.28] vol=1.8x ATR=2.64 |
| Stop hit — per-position SL triggered | 2024-04-09 10:50:00 | 740.88 | 737.28 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 09:30:00 | 750.12 | 746.48 | 0.00 | ORB-long ORB[740.52,748.36] vol=5.2x ATR=2.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-10 09:55:00 | 754.44 | 748.60 | 0.00 | T1 1.5R @ 754.44 |
| Target hit | 2024-04-10 15:20:00 | 777.04 | 768.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2024-04-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-18 11:00:00 | 748.92 | 753.60 | 0.00 | ORB-short ORB[756.48,763.96] vol=1.8x ATR=2.60 |
| Stop hit — per-position SL triggered | 2024-04-18 11:05:00 | 751.52 | 753.24 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-04-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 09:55:00 | 759.88 | 756.11 | 0.00 | ORB-long ORB[751.96,759.56] vol=1.6x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-23 10:05:00 | 763.40 | 757.66 | 0.00 | T1 1.5R @ 763.40 |
| Target hit | 2024-04-23 10:50:00 | 760.20 | 760.73 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — BUY (started 2024-04-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 10:00:00 | 769.96 | 765.90 | 0.00 | ORB-long ORB[758.24,766.88] vol=3.8x ATR=2.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 10:20:00 | 773.96 | 768.21 | 0.00 | T1 1.5R @ 773.96 |
| Target hit | 2024-04-24 11:10:00 | 770.24 | 771.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — BUY (started 2024-05-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 11:10:00 | 835.84 | 832.64 | 0.00 | ORB-long ORB[822.88,834.36] vol=3.5x ATR=2.63 |
| Stop hit — per-position SL triggered | 2024-05-02 11:25:00 | 833.21 | 832.70 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-05-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-03 09:45:00 | 856.40 | 848.34 | 0.00 | ORB-long ORB[838.44,849.96] vol=2.0x ATR=3.58 |
| Stop hit — per-position SL triggered | 2024-05-03 10:05:00 | 852.82 | 851.83 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-05-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-10 09:55:00 | 801.60 | 795.55 | 0.00 | ORB-long ORB[791.84,799.20] vol=2.1x ATR=4.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 11:25:00 | 808.63 | 799.99 | 0.00 | T1 1.5R @ 808.63 |
| Stop hit — per-position SL triggered | 2024-05-10 12:10:00 | 801.60 | 800.85 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-03-11 09:40:00 | 763.60 | 2024-03-11 10:20:00 | 765.87 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-03-15 09:30:00 | 685.00 | 2024-03-15 09:40:00 | 680.01 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2024-03-15 09:30:00 | 685.00 | 2024-03-15 09:45:00 | 685.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-19 10:45:00 | 671.56 | 2024-03-19 11:05:00 | 667.77 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-03-19 10:45:00 | 671.56 | 2024-03-19 11:10:00 | 671.56 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-21 09:45:00 | 685.60 | 2024-03-21 10:05:00 | 690.08 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-03-21 09:45:00 | 685.60 | 2024-03-21 10:30:00 | 685.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-26 09:50:00 | 700.80 | 2024-03-26 10:05:00 | 698.11 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-03-27 09:35:00 | 709.20 | 2024-03-27 09:40:00 | 706.69 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-03-28 09:50:00 | 701.36 | 2024-03-28 09:55:00 | 699.31 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-04-02 10:30:00 | 728.36 | 2024-04-02 10:45:00 | 732.30 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-04-02 10:30:00 | 728.36 | 2024-04-02 10:55:00 | 728.36 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-04 10:55:00 | 734.36 | 2024-04-04 11:05:00 | 730.52 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-04-04 10:55:00 | 734.36 | 2024-04-04 11:10:00 | 734.36 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-05 10:00:00 | 735.96 | 2024-04-05 10:05:00 | 733.35 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-04-08 09:45:00 | 737.72 | 2024-04-08 09:55:00 | 740.58 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-04-09 10:40:00 | 743.52 | 2024-04-09 10:50:00 | 740.88 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-04-10 09:30:00 | 750.12 | 2024-04-10 09:55:00 | 754.44 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-04-10 09:30:00 | 750.12 | 2024-04-10 15:20:00 | 777.04 | TARGET_HIT | 0.50 | 3.59% |
| SELL | retest1 | 2024-04-18 11:00:00 | 748.92 | 2024-04-18 11:05:00 | 751.52 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-04-23 09:55:00 | 759.88 | 2024-04-23 10:05:00 | 763.40 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-04-23 09:55:00 | 759.88 | 2024-04-23 10:50:00 | 760.20 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2024-04-24 10:00:00 | 769.96 | 2024-04-24 10:20:00 | 773.96 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-04-24 10:00:00 | 769.96 | 2024-04-24 11:10:00 | 770.24 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2024-05-02 11:10:00 | 835.84 | 2024-05-02 11:25:00 | 833.21 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-05-03 09:45:00 | 856.40 | 2024-05-03 10:05:00 | 852.82 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-05-10 09:55:00 | 801.60 | 2024-05-10 11:25:00 | 808.63 | PARTIAL | 0.50 | 0.88% |
| BUY | retest1 | 2024-05-10 09:55:00 | 801.60 | 2024-05-10 12:10:00 | 801.60 | STOP_HIT | 0.50 | 0.00% |
