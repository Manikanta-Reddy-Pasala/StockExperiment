# Supreme Petrochem Ltd. (SPLPETRO)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 738.40
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
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 11
- **Target hits / Stop hits / Partials:** 4 / 11 / 7
- **Avg / median % per leg:** 0.91% / 0.36%
- **Sum % (uncompounded):** 20.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 10 | 62.5% | 4 | 6 | 6 | 1.30% | 20.7% |
| BUY @ 2nd Alert (retest1) | 16 | 10 | 62.5% | 4 | 6 | 6 | 1.30% | 20.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.12% | -0.7% |
| SELL @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.12% | -0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 11 | 50.0% | 4 | 11 | 7 | 0.91% | 20.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:30:00 | 606.30 | 603.18 | 0.00 | ORB-long ORB[596.35,602.95] vol=3.4x ATR=2.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:20:00 | 610.76 | 606.69 | 0.00 | T1 1.5R @ 610.76 |
| Target hit | 2026-02-09 15:20:00 | 637.40 | 631.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 628.00 | 633.13 | 0.00 | ORB-short ORB[633.80,640.20] vol=2.9x ATR=2.19 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 630.19 | 631.35 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:50:00 | 630.20 | 626.24 | 0.00 | ORB-long ORB[620.15,625.35] vol=10.9x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:55:00 | 633.48 | 628.88 | 0.00 | T1 1.5R @ 633.48 |
| Stop hit — per-position SL triggered | 2026-02-17 12:30:00 | 630.20 | 630.69 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:05:00 | 632.90 | 630.47 | 0.00 | ORB-long ORB[625.00,631.55] vol=1.9x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:30:00 | 635.15 | 632.11 | 0.00 | T1 1.5R @ 635.15 |
| Stop hit — per-position SL triggered | 2026-02-18 11:00:00 | 632.90 | 632.36 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:05:00 | 633.20 | 637.52 | 0.00 | ORB-short ORB[633.85,640.70] vol=2.1x ATR=1.47 |
| Stop hit — per-position SL triggered | 2026-02-19 12:00:00 | 634.67 | 649.08 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:55:00 | 650.75 | 647.88 | 0.00 | ORB-long ORB[646.85,649.50] vol=3.1x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:10:00 | 653.29 | 654.41 | 0.00 | T1 1.5R @ 653.29 |
| Target hit | 2026-02-27 15:20:00 | 722.75 | 705.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 11:15:00 | 670.75 | 663.48 | 0.00 | ORB-long ORB[659.50,666.45] vol=4.0x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:25:00 | 673.86 | 665.34 | 0.00 | T1 1.5R @ 673.86 |
| Target hit | 2026-03-05 12:45:00 | 673.15 | 673.78 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2026-03-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:35:00 | 685.80 | 683.05 | 0.00 | ORB-long ORB[679.65,684.05] vol=1.6x ATR=2.85 |
| Stop hit — per-position SL triggered | 2026-03-06 10:45:00 | 682.95 | 685.20 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 09:55:00 | 644.70 | 651.10 | 0.00 | ORB-short ORB[655.25,663.40] vol=9.3x ATR=2.91 |
| Stop hit — per-position SL triggered | 2026-03-20 10:00:00 | 647.61 | 648.79 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:45:00 | 751.35 | 746.29 | 0.00 | ORB-long ORB[738.85,749.80] vol=3.1x ATR=3.54 |
| Stop hit — per-position SL triggered | 2026-04-10 10:00:00 | 747.81 | 746.71 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 11:00:00 | 754.30 | 740.19 | 0.00 | ORB-long ORB[730.40,741.65] vol=2.3x ATR=3.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:15:00 | 759.74 | 749.19 | 0.00 | T1 1.5R @ 759.74 |
| Target hit | 2026-04-15 15:20:00 | 775.80 | 764.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 779.35 | 775.12 | 0.00 | ORB-long ORB[769.95,778.00] vol=2.6x ATR=2.55 |
| Stop hit — per-position SL triggered | 2026-04-22 09:45:00 | 776.80 | 776.21 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 788.00 | 779.80 | 0.00 | ORB-long ORB[769.85,781.20] vol=4.2x ATR=5.06 |
| Stop hit — per-position SL triggered | 2026-04-28 09:40:00 | 782.94 | 785.11 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-05-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:30:00 | 749.00 | 757.03 | 0.00 | ORB-short ORB[753.00,761.15] vol=5.5x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 10:40:00 | 744.44 | 754.33 | 0.00 | T1 1.5R @ 744.44 |
| Stop hit — per-position SL triggered | 2026-05-04 11:00:00 | 749.00 | 753.58 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-05-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:35:00 | 732.25 | 736.39 | 0.00 | ORB-short ORB[733.00,743.40] vol=1.6x ATR=2.11 |
| Stop hit — per-position SL triggered | 2026-05-05 11:05:00 | 734.36 | 736.20 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:30:00 | 606.30 | 2026-02-09 11:20:00 | 610.76 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2026-02-09 10:30:00 | 606.30 | 2026-02-09 15:20:00 | 637.40 | TARGET_HIT | 0.50 | 5.13% |
| SELL | retest1 | 2026-02-13 09:30:00 | 628.00 | 2026-02-13 09:40:00 | 630.19 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-17 10:50:00 | 630.20 | 2026-02-17 10:55:00 | 633.48 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-02-17 10:50:00 | 630.20 | 2026-02-17 12:30:00 | 630.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 10:05:00 | 632.90 | 2026-02-18 10:30:00 | 635.15 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-18 10:05:00 | 632.90 | 2026-02-18 11:00:00 | 632.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 11:05:00 | 633.20 | 2026-02-19 12:00:00 | 634.67 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-27 09:55:00 | 650.75 | 2026-02-27 10:10:00 | 653.29 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-27 09:55:00 | 650.75 | 2026-02-27 15:20:00 | 722.75 | TARGET_HIT | 0.50 | 11.06% |
| BUY | retest1 | 2026-03-05 11:15:00 | 670.75 | 2026-03-05 11:25:00 | 673.86 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-03-05 11:15:00 | 670.75 | 2026-03-05 12:45:00 | 673.15 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2026-03-06 09:35:00 | 685.80 | 2026-03-06 10:45:00 | 682.95 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-20 09:55:00 | 644.70 | 2026-03-20 10:00:00 | 647.61 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-10 09:45:00 | 751.35 | 2026-04-10 10:00:00 | 747.81 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-04-15 11:00:00 | 754.30 | 2026-04-15 11:15:00 | 759.74 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2026-04-15 11:00:00 | 754.30 | 2026-04-15 15:20:00 | 775.80 | TARGET_HIT | 0.50 | 2.85% |
| BUY | retest1 | 2026-04-22 09:35:00 | 779.35 | 2026-04-22 09:45:00 | 776.80 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-28 09:30:00 | 788.00 | 2026-04-28 09:40:00 | 782.94 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest1 | 2026-05-04 10:30:00 | 749.00 | 2026-05-04 10:40:00 | 744.44 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-05-04 10:30:00 | 749.00 | 2026-05-04 11:00:00 | 749.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 10:35:00 | 732.25 | 2026-05-05 11:05:00 | 734.36 | STOP_HIT | 1.00 | -0.29% |
