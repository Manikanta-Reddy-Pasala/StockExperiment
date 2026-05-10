# SBI Cards and Payment Services Ltd. (SBICARD)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 645.00
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 13
- **Target hits / Stop hits / Partials:** 1 / 13 / 3
- **Avg / median % per leg:** 0.07% / -0.24%
- **Sum % (uncompounded):** 1.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.38% | -1.9% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.38% | -1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 4 | 33.3% | 1 | 8 | 3 | 0.25% | 3.0% |
| SELL @ 2nd Alert (retest1) | 12 | 4 | 33.3% | 1 | 8 | 3 | 0.25% | 3.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 4 | 23.5% | 1 | 13 | 3 | 0.07% | 1.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:30:00 | 770.45 | 765.09 | 0.00 | ORB-long ORB[755.80,763.00] vol=2.2x ATR=3.99 |
| Stop hit — per-position SL triggered | 2026-02-09 11:25:00 | 766.46 | 766.95 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:50:00 | 773.70 | 766.42 | 0.00 | ORB-long ORB[760.95,767.05] vol=2.2x ATR=1.71 |
| Stop hit — per-position SL triggered | 2026-02-12 11:15:00 | 771.99 | 767.99 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:35:00 | 729.95 | 733.23 | 0.00 | ORB-short ORB[731.55,740.00] vol=2.1x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 10:40:00 | 725.11 | 730.83 | 0.00 | T1 1.5R @ 725.11 |
| Stop hit — per-position SL triggered | 2026-03-04 13:10:00 | 729.95 | 727.47 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:45:00 | 725.95 | 728.29 | 0.00 | ORB-short ORB[729.95,738.00] vol=1.5x ATR=1.86 |
| Stop hit — per-position SL triggered | 2026-03-05 10:50:00 | 727.81 | 728.21 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:35:00 | 698.00 | 693.71 | 0.00 | ORB-long ORB[687.40,693.90] vol=2.7x ATR=2.13 |
| Stop hit — per-position SL triggered | 2026-03-20 11:30:00 | 695.87 | 695.37 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:05:00 | 684.60 | 687.46 | 0.00 | ORB-short ORB[690.20,698.05] vol=1.9x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:40:00 | 681.28 | 686.82 | 0.00 | T1 1.5R @ 681.28 |
| Stop hit — per-position SL triggered | 2026-03-27 13:40:00 | 684.60 | 683.57 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 10:50:00 | 655.00 | 660.06 | 0.00 | ORB-short ORB[661.00,667.00] vol=5.3x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 11:05:00 | 651.19 | 657.56 | 0.00 | T1 1.5R @ 651.19 |
| Target hit | 2026-03-30 15:20:00 | 635.80 | 646.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-04-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:25:00 | 696.50 | 691.07 | 0.00 | ORB-long ORB[680.75,690.75] vol=1.6x ATR=2.21 |
| Stop hit — per-position SL triggered | 2026-04-17 10:45:00 | 694.29 | 691.93 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:45:00 | 676.65 | 679.89 | 0.00 | ORB-short ORB[679.65,687.90] vol=1.7x ATR=2.39 |
| Stop hit — per-position SL triggered | 2026-04-24 10:05:00 | 679.04 | 678.94 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:35:00 | 665.00 | 659.90 | 0.00 | ORB-long ORB[651.30,657.95] vol=1.6x ATR=3.44 |
| Stop hit — per-position SL triggered | 2026-04-28 11:00:00 | 661.56 | 660.98 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:45:00 | 634.50 | 640.45 | 0.00 | ORB-short ORB[642.10,650.00] vol=5.1x ATR=1.99 |
| Stop hit — per-position SL triggered | 2026-04-30 11:20:00 | 636.49 | 638.67 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 11:05:00 | 643.35 | 645.74 | 0.00 | ORB-short ORB[646.05,649.80] vol=1.6x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-05-04 11:30:00 | 644.91 | 645.45 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-05-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:20:00 | 640.00 | 641.77 | 0.00 | ORB-short ORB[641.10,645.95] vol=2.8x ATR=1.60 |
| Stop hit — per-position SL triggered | 2026-05-05 11:25:00 | 641.60 | 640.56 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-05-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:00:00 | 645.00 | 648.16 | 0.00 | ORB-short ORB[649.75,655.75] vol=1.9x ATR=1.45 |
| Stop hit — per-position SL triggered | 2026-05-07 11:30:00 | 646.45 | 647.53 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:30:00 | 770.45 | 2026-02-09 11:25:00 | 766.46 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-02-12 10:50:00 | 773.70 | 2026-02-12 11:15:00 | 771.99 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-03-04 09:35:00 | 729.95 | 2026-03-04 10:40:00 | 725.11 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-03-04 09:35:00 | 729.95 | 2026-03-04 13:10:00 | 729.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 10:45:00 | 725.95 | 2026-03-05 10:50:00 | 727.81 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-20 10:35:00 | 698.00 | 2026-03-20 11:30:00 | 695.87 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-27 11:05:00 | 684.60 | 2026-03-27 11:40:00 | 681.28 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-27 11:05:00 | 684.60 | 2026-03-27 13:40:00 | 684.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-30 10:50:00 | 655.00 | 2026-03-30 11:05:00 | 651.19 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-03-30 10:50:00 | 655.00 | 2026-03-30 15:20:00 | 635.80 | TARGET_HIT | 0.50 | 2.93% |
| BUY | retest1 | 2026-04-17 10:25:00 | 696.50 | 2026-04-17 10:45:00 | 694.29 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-24 09:45:00 | 676.65 | 2026-04-24 10:05:00 | 679.04 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-28 10:35:00 | 665.00 | 2026-04-28 11:00:00 | 661.56 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2026-04-30 10:45:00 | 634.50 | 2026-04-30 11:20:00 | 636.49 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-05-04 11:05:00 | 643.35 | 2026-05-04 11:30:00 | 644.91 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-05-05 10:20:00 | 640.00 | 2026-05-05 11:25:00 | 641.60 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-05-07 11:00:00 | 645.00 | 2026-05-07 11:30:00 | 646.45 | STOP_HIT | 1.00 | -0.23% |
