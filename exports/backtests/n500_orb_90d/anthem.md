# Anthem Biosciences Ltd. (ANTHEM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 784.40
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 17
- **Target hits / Stop hits / Partials:** 1 / 17 / 4
- **Avg / median % per leg:** -0.09% / -0.30%
- **Sum % (uncompounded):** -2.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 3 | 18.8% | 0 | 13 | 3 | -0.11% | -1.8% |
| BUY @ 2nd Alert (retest1) | 16 | 3 | 18.8% | 0 | 13 | 3 | -0.11% | -1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | -0.04% | -0.3% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | -0.04% | -0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 5 | 22.7% | 1 | 17 | 4 | -0.09% | -2.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:00:00 | 685.00 | 681.12 | 0.00 | ORB-long ORB[674.45,682.70] vol=2.4x ATR=2.13 |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 682.87 | 681.38 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:45:00 | 691.05 | 688.43 | 0.00 | ORB-long ORB[684.90,690.25] vol=1.9x ATR=2.57 |
| Stop hit — per-position SL triggered | 2026-02-16 10:05:00 | 688.48 | 688.73 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:30:00 | 693.75 | 689.44 | 0.00 | ORB-long ORB[685.00,690.00] vol=1.6x ATR=2.10 |
| Stop hit — per-position SL triggered | 2026-02-17 09:35:00 | 691.65 | 689.74 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:05:00 | 691.20 | 692.40 | 0.00 | ORB-short ORB[691.65,695.25] vol=1.9x ATR=1.64 |
| Stop hit — per-position SL triggered | 2026-02-18 11:20:00 | 692.84 | 691.73 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:40:00 | 693.75 | 690.13 | 0.00 | ORB-long ORB[685.00,691.05] vol=8.1x ATR=2.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:55:00 | 697.12 | 690.92 | 0.00 | T1 1.5R @ 697.12 |
| Stop hit — per-position SL triggered | 2026-02-19 11:55:00 | 693.75 | 692.66 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:55:00 | 698.55 | 695.30 | 0.00 | ORB-long ORB[692.50,697.00] vol=1.8x ATR=2.28 |
| Stop hit — per-position SL triggered | 2026-02-20 10:05:00 | 696.27 | 695.50 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:10:00 | 707.75 | 703.70 | 0.00 | ORB-long ORB[695.45,705.75] vol=2.1x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:40:00 | 711.79 | 707.10 | 0.00 | T1 1.5R @ 711.79 |
| Stop hit — per-position SL triggered | 2026-02-25 10:45:00 | 707.75 | 707.11 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:50:00 | 669.05 | 672.58 | 0.00 | ORB-short ORB[669.85,677.00] vol=1.8x ATR=3.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 10:45:00 | 664.17 | 671.07 | 0.00 | T1 1.5R @ 664.17 |
| Target hit | 2026-03-04 12:30:00 | 668.20 | 667.44 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — BUY (started 2026-03-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:50:00 | 669.80 | 665.75 | 0.00 | ORB-long ORB[660.50,665.95] vol=1.7x ATR=2.55 |
| Stop hit — per-position SL triggered | 2026-03-06 10:35:00 | 667.25 | 667.42 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 10:45:00 | 637.30 | 639.75 | 0.00 | ORB-short ORB[641.70,650.00] vol=2.1x ATR=1.47 |
| Stop hit — per-position SL triggered | 2026-03-17 10:50:00 | 638.77 | 639.67 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 11:00:00 | 648.05 | 644.79 | 0.00 | ORB-long ORB[636.05,643.00] vol=2.7x ATR=1.98 |
| Stop hit — per-position SL triggered | 2026-03-20 11:20:00 | 646.07 | 645.36 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:40:00 | 629.20 | 632.61 | 0.00 | ORB-short ORB[630.70,635.65] vol=3.2x ATR=2.16 |
| Stop hit — per-position SL triggered | 2026-03-23 11:20:00 | 631.36 | 630.89 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:30:00 | 729.45 | 725.86 | 0.00 | ORB-long ORB[721.00,729.20] vol=2.7x ATR=2.59 |
| Stop hit — per-position SL triggered | 2026-04-17 10:05:00 | 726.86 | 726.68 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:10:00 | 747.75 | 742.01 | 0.00 | ORB-long ORB[733.50,741.25] vol=1.6x ATR=2.43 |
| Stop hit — per-position SL triggered | 2026-04-22 11:05:00 | 745.32 | 744.75 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 751.35 | 748.06 | 0.00 | ORB-long ORB[743.40,749.00] vol=2.5x ATR=3.14 |
| Stop hit — per-position SL triggered | 2026-04-24 09:35:00 | 748.21 | 748.13 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:35:00 | 755.90 | 749.22 | 0.00 | ORB-long ORB[739.15,748.45] vol=4.7x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:50:00 | 760.43 | 752.46 | 0.00 | T1 1.5R @ 760.43 |
| Stop hit — per-position SL triggered | 2026-04-27 10:00:00 | 755.90 | 758.37 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 766.50 | 768.08 | 0.00 | ORB-short ORB[768.00,775.05] vol=4.0x ATR=2.37 |
| Stop hit — per-position SL triggered | 2026-04-28 09:45:00 | 768.87 | 768.09 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-05-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:10:00 | 780.15 | 777.08 | 0.00 | ORB-long ORB[769.70,778.80] vol=2.8x ATR=2.48 |
| Stop hit — per-position SL triggered | 2026-05-07 10:35:00 | 777.67 | 777.37 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 10:00:00 | 685.00 | 2026-02-11 10:15:00 | 682.87 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-16 09:45:00 | 691.05 | 2026-02-16 10:05:00 | 688.48 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-17 09:30:00 | 693.75 | 2026-02-17 09:35:00 | 691.65 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-18 10:05:00 | 691.20 | 2026-02-18 11:20:00 | 692.84 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-19 10:40:00 | 693.75 | 2026-02-19 10:55:00 | 697.12 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-19 10:40:00 | 693.75 | 2026-02-19 11:55:00 | 693.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 09:55:00 | 698.55 | 2026-02-20 10:05:00 | 696.27 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-25 10:10:00 | 707.75 | 2026-02-25 10:40:00 | 711.79 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-02-25 10:10:00 | 707.75 | 2026-02-25 10:45:00 | 707.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-04 09:50:00 | 669.05 | 2026-03-04 10:45:00 | 664.17 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2026-03-04 09:50:00 | 669.05 | 2026-03-04 12:30:00 | 668.20 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2026-03-06 09:50:00 | 669.80 | 2026-03-06 10:35:00 | 667.25 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-03-17 10:45:00 | 637.30 | 2026-03-17 10:50:00 | 638.77 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-03-20 11:00:00 | 648.05 | 2026-03-20 11:20:00 | 646.07 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-23 10:40:00 | 629.20 | 2026-03-23 11:20:00 | 631.36 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-17 09:30:00 | 729.45 | 2026-04-17 10:05:00 | 726.86 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-22 10:10:00 | 747.75 | 2026-04-22 11:05:00 | 745.32 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-24 09:30:00 | 751.35 | 2026-04-24 09:35:00 | 748.21 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-27 09:35:00 | 755.90 | 2026-04-27 09:50:00 | 760.43 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-27 09:35:00 | 755.90 | 2026-04-27 10:00:00 | 755.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-28 09:30:00 | 766.50 | 2026-04-28 09:45:00 | 768.87 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-05-07 10:10:00 | 780.15 | 2026-05-07 10:35:00 | 777.67 | STOP_HIT | 1.00 | -0.32% |
