# AU Small Finance Bank Ltd. (AUBANK)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1051.00
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
| ENTRY1 | 73 |
| ENTRY2 | 0 |
| PARTIAL | 25 |
| TARGET_HIT | 13 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 98 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 60
- **Target hits / Stop hits / Partials:** 13 / 60 / 25
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 7.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 52 | 13 | 25.0% | 3 | 39 | 10 | -0.04% | -2.2% |
| BUY @ 2nd Alert (retest1) | 52 | 13 | 25.0% | 3 | 39 | 10 | -0.04% | -2.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 46 | 25 | 54.3% | 10 | 21 | 15 | 0.22% | 10.1% |
| SELL @ 2nd Alert (retest1) | 46 | 25 | 54.3% | 10 | 21 | 15 | 0.22% | 10.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 98 | 38 | 38.8% | 13 | 60 | 25 | 0.08% | 8.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 11:05:00 | 624.00 | 627.48 | 0.00 | ORB-short ORB[628.05,636.00] vol=1.5x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 11:20:00 | 619.97 | 626.68 | 0.00 | T1 1.5R @ 619.97 |
| Stop hit — per-position SL triggered | 2024-05-13 12:35:00 | 624.00 | 624.92 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:30:00 | 648.20 | 646.06 | 0.00 | ORB-long ORB[642.45,648.00] vol=2.6x ATR=2.10 |
| Stop hit — per-position SL triggered | 2024-05-15 09:35:00 | 646.10 | 646.02 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:15:00 | 615.10 | 622.00 | 0.00 | ORB-short ORB[621.50,628.50] vol=1.8x ATR=1.66 |
| Stop hit — per-position SL triggered | 2024-05-16 11:25:00 | 616.76 | 621.55 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 11:10:00 | 625.95 | 624.03 | 0.00 | ORB-long ORB[621.85,625.40] vol=2.8x ATR=1.57 |
| Stop hit — per-position SL triggered | 2024-05-17 11:20:00 | 624.38 | 624.15 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 09:30:00 | 618.60 | 622.54 | 0.00 | ORB-short ORB[620.55,626.00] vol=2.6x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 09:55:00 | 615.35 | 620.06 | 0.00 | T1 1.5R @ 615.35 |
| Stop hit — per-position SL triggered | 2024-05-21 10:35:00 | 618.60 | 617.44 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:45:00 | 613.40 | 616.00 | 0.00 | ORB-short ORB[615.50,620.55] vol=1.7x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 10:05:00 | 610.78 | 614.99 | 0.00 | T1 1.5R @ 610.78 |
| Target hit | 2024-05-22 15:20:00 | 604.05 | 607.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-05-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 09:40:00 | 628.90 | 626.01 | 0.00 | ORB-long ORB[621.00,628.50] vol=1.9x ATR=2.07 |
| Stop hit — per-position SL triggered | 2024-05-27 09:50:00 | 626.83 | 626.58 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-05-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 09:45:00 | 650.15 | 643.98 | 0.00 | ORB-long ORB[633.10,641.60] vol=3.5x ATR=3.00 |
| Stop hit — per-position SL triggered | 2024-05-29 09:55:00 | 647.15 | 644.80 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-05-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-31 10:05:00 | 649.40 | 646.07 | 0.00 | ORB-long ORB[639.85,648.95] vol=1.9x ATR=2.57 |
| Stop hit — per-position SL triggered | 2024-05-31 10:10:00 | 646.83 | 646.36 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 10:35:00 | 667.15 | 664.19 | 0.00 | ORB-long ORB[660.40,665.65] vol=1.8x ATR=1.98 |
| Stop hit — per-position SL triggered | 2024-06-07 10:40:00 | 665.17 | 664.29 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 09:30:00 | 664.40 | 670.55 | 0.00 | ORB-short ORB[668.05,676.25] vol=3.6x ATR=2.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 09:50:00 | 659.98 | 667.92 | 0.00 | T1 1.5R @ 659.98 |
| Target hit | 2024-06-10 13:40:00 | 656.90 | 656.51 | 0.00 | Trail-exit close>VWAP |

### Cycle 12 — BUY (started 2024-06-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:55:00 | 676.20 | 672.18 | 0.00 | ORB-long ORB[666.00,674.90] vol=1.8x ATR=2.29 |
| Stop hit — per-position SL triggered | 2024-06-12 10:40:00 | 673.91 | 674.94 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 10:40:00 | 689.00 | 684.65 | 0.00 | ORB-long ORB[681.05,688.15] vol=2.0x ATR=2.40 |
| Stop hit — per-position SL triggered | 2024-06-25 11:10:00 | 686.60 | 685.57 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:15:00 | 693.15 | 683.99 | 0.00 | ORB-long ORB[674.75,684.00] vol=2.1x ATR=3.32 |
| Stop hit — per-position SL triggered | 2024-06-26 10:20:00 | 689.83 | 684.51 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 11:05:00 | 675.00 | 671.22 | 0.00 | ORB-long ORB[665.00,672.55] vol=1.9x ATR=1.73 |
| Stop hit — per-position SL triggered | 2024-07-01 14:45:00 | 673.27 | 673.32 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 09:35:00 | 679.05 | 676.39 | 0.00 | ORB-long ORB[673.10,677.50] vol=1.6x ATR=1.86 |
| Stop hit — per-position SL triggered | 2024-07-02 09:40:00 | 677.19 | 676.49 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:50:00 | 670.00 | 673.97 | 0.00 | ORB-short ORB[673.00,678.20] vol=1.5x ATR=2.16 |
| Stop hit — per-position SL triggered | 2024-07-05 10:25:00 | 672.16 | 672.33 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:40:00 | 659.60 | 663.24 | 0.00 | ORB-short ORB[660.40,668.25] vol=2.6x ATR=2.46 |
| Stop hit — per-position SL triggered | 2024-07-08 09:50:00 | 662.06 | 662.89 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:05:00 | 641.00 | 638.15 | 0.00 | ORB-long ORB[633.05,638.90] vol=1.5x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 11:05:00 | 643.86 | 640.14 | 0.00 | T1 1.5R @ 643.86 |
| Stop hit — per-position SL triggered | 2024-07-12 11:35:00 | 641.00 | 640.66 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-07-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 11:00:00 | 627.60 | 629.85 | 0.00 | ORB-short ORB[629.55,635.35] vol=1.7x ATR=1.66 |
| Stop hit — per-position SL triggered | 2024-07-19 11:15:00 | 629.26 | 629.78 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 10:00:00 | 637.15 | 633.18 | 0.00 | ORB-long ORB[625.00,633.55] vol=2.9x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 10:15:00 | 640.13 | 634.62 | 0.00 | T1 1.5R @ 640.13 |
| Target hit | 2024-07-22 15:20:00 | 654.50 | 649.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2024-07-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 09:50:00 | 659.10 | 656.43 | 0.00 | ORB-long ORB[651.15,658.20] vol=2.7x ATR=3.33 |
| Stop hit — per-position SL triggered | 2024-07-24 10:25:00 | 655.77 | 659.43 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-02 10:00:00 | 636.20 | 638.07 | 0.00 | ORB-short ORB[636.50,643.95] vol=3.5x ATR=1.86 |
| Stop hit — per-position SL triggered | 2024-08-02 10:10:00 | 638.06 | 638.03 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 10:25:00 | 627.05 | 628.57 | 0.00 | ORB-short ORB[629.35,636.00] vol=4.3x ATR=1.66 |
| Stop hit — per-position SL triggered | 2024-08-09 10:50:00 | 628.71 | 627.46 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-12 10:05:00 | 610.05 | 614.53 | 0.00 | ORB-short ORB[612.25,621.40] vol=1.9x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 10:25:00 | 606.90 | 612.89 | 0.00 | T1 1.5R @ 606.90 |
| Stop hit — per-position SL triggered | 2024-08-12 10:35:00 | 610.05 | 612.34 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 11:15:00 | 623.70 | 621.50 | 0.00 | ORB-long ORB[619.60,623.05] vol=4.3x ATR=1.64 |
| Stop hit — per-position SL triggered | 2024-08-20 12:05:00 | 622.06 | 622.78 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 09:35:00 | 616.60 | 618.85 | 0.00 | ORB-short ORB[617.80,623.90] vol=1.8x ATR=1.69 |
| Stop hit — per-position SL triggered | 2024-08-21 09:45:00 | 618.29 | 618.25 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 11:00:00 | 634.00 | 633.33 | 0.00 | ORB-long ORB[629.30,633.00] vol=1.5x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-08-22 12:10:00 | 632.37 | 633.35 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:35:00 | 629.55 | 632.46 | 0.00 | ORB-short ORB[630.25,636.00] vol=2.4x ATR=1.69 |
| Stop hit — per-position SL triggered | 2024-08-23 09:55:00 | 631.24 | 631.70 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-06 11:15:00 | 709.70 | 706.06 | 0.00 | ORB-long ORB[700.00,709.50] vol=3.3x ATR=2.41 |
| Stop hit — per-position SL triggered | 2024-09-06 11:40:00 | 707.29 | 706.95 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-09 11:00:00 | 712.95 | 705.93 | 0.00 | ORB-long ORB[696.85,707.40] vol=3.0x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 11:30:00 | 716.16 | 708.17 | 0.00 | T1 1.5R @ 716.16 |
| Stop hit — per-position SL triggered | 2024-09-09 11:45:00 | 712.95 | 708.79 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:55:00 | 721.60 | 718.86 | 0.00 | ORB-long ORB[713.15,719.00] vol=2.4x ATR=1.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 11:00:00 | 724.11 | 719.76 | 0.00 | T1 1.5R @ 724.11 |
| Stop hit — per-position SL triggered | 2024-09-11 11:15:00 | 721.60 | 720.11 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:55:00 | 723.40 | 721.22 | 0.00 | ORB-long ORB[717.25,723.00] vol=4.4x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 11:15:00 | 725.83 | 721.86 | 0.00 | T1 1.5R @ 725.83 |
| Target hit | 2024-09-13 13:55:00 | 723.75 | 723.77 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — BUY (started 2024-09-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 10:40:00 | 722.45 | 719.72 | 0.00 | ORB-long ORB[715.20,720.50] vol=1.8x ATR=1.77 |
| Stop hit — per-position SL triggered | 2024-09-18 10:55:00 | 720.68 | 720.22 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-10-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 10:55:00 | 696.00 | 701.00 | 0.00 | ORB-short ORB[699.30,706.55] vol=2.7x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 11:15:00 | 693.41 | 700.24 | 0.00 | T1 1.5R @ 693.41 |
| Target hit | 2024-10-11 15:20:00 | 689.90 | 694.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2024-10-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:55:00 | 689.00 | 693.29 | 0.00 | ORB-short ORB[694.00,702.40] vol=2.2x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:15:00 | 686.16 | 692.73 | 0.00 | T1 1.5R @ 686.16 |
| Target hit | 2024-10-17 15:20:00 | 687.10 | 689.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2024-10-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 09:50:00 | 683.50 | 680.66 | 0.00 | ORB-long ORB[676.50,683.05] vol=1.8x ATR=2.06 |
| Stop hit — per-position SL triggered | 2024-10-18 09:55:00 | 681.44 | 680.73 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-10-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 11:00:00 | 615.30 | 611.40 | 0.00 | ORB-long ORB[604.50,611.40] vol=1.5x ATR=2.90 |
| Stop hit — per-position SL triggered | 2024-10-28 11:10:00 | 612.40 | 611.50 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-04 10:15:00 | 623.00 | 620.71 | 0.00 | ORB-long ORB[616.40,622.00] vol=5.1x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 10:30:00 | 625.98 | 621.70 | 0.00 | T1 1.5R @ 625.98 |
| Stop hit — per-position SL triggered | 2024-11-04 10:35:00 | 623.00 | 621.80 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-11-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 09:55:00 | 616.65 | 619.10 | 0.00 | ORB-short ORB[620.10,626.85] vol=1.6x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 10:05:00 | 613.59 | 618.62 | 0.00 | T1 1.5R @ 613.59 |
| Target hit | 2024-11-05 14:00:00 | 613.00 | 612.55 | 0.00 | Trail-exit close>VWAP |

### Cycle 41 — SELL (started 2024-11-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 10:50:00 | 607.45 | 608.63 | 0.00 | ORB-short ORB[608.10,614.40] vol=1.9x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 12:00:00 | 605.46 | 608.21 | 0.00 | T1 1.5R @ 605.46 |
| Target hit | 2024-11-07 15:20:00 | 602.00 | 606.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2024-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-18 11:15:00 | 578.95 | 578.25 | 0.00 | ORB-long ORB[570.25,577.70] vol=2.0x ATR=1.74 |
| Stop hit — per-position SL triggered | 2024-11-18 12:40:00 | 577.21 | 578.40 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 11:15:00 | 592.05 | 588.06 | 0.00 | ORB-long ORB[581.05,589.20] vol=3.4x ATR=1.92 |
| Stop hit — per-position SL triggered | 2024-11-21 11:20:00 | 590.13 | 588.68 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-11-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 09:40:00 | 584.95 | 587.78 | 0.00 | ORB-short ORB[586.80,592.05] vol=2.0x ATR=1.53 |
| Stop hit — per-position SL triggered | 2024-11-27 10:15:00 | 586.48 | 587.13 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-11-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 10:45:00 | 584.50 | 587.41 | 0.00 | ORB-short ORB[586.65,591.75] vol=2.5x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 11:20:00 | 582.54 | 585.26 | 0.00 | T1 1.5R @ 582.54 |
| Target hit | 2024-11-29 15:20:00 | 584.45 | 583.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2024-12-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-02 11:00:00 | 579.55 | 579.76 | 0.00 | ORB-short ORB[580.55,586.00] vol=12.0x ATR=1.32 |
| Stop hit — per-position SL triggered | 2024-12-02 11:05:00 | 580.87 | 579.77 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-12-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:50:00 | 591.30 | 586.56 | 0.00 | ORB-long ORB[582.10,587.70] vol=2.9x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-12-03 10:05:00 | 589.59 | 587.73 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-12-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:30:00 | 594.60 | 594.82 | 0.00 | ORB-short ORB[595.00,598.70] vol=2.3x ATR=1.42 |
| Stop hit — per-position SL triggered | 2024-12-05 10:50:00 | 596.02 | 594.90 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-12-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 10:05:00 | 591.65 | 595.30 | 0.00 | ORB-short ORB[596.85,602.00] vol=1.9x ATR=1.81 |
| Stop hit — per-position SL triggered | 2024-12-06 10:20:00 | 593.46 | 594.48 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-12-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:30:00 | 589.50 | 586.86 | 0.00 | ORB-long ORB[581.45,589.00] vol=2.0x ATR=1.53 |
| Stop hit — per-position SL triggered | 2024-12-10 09:35:00 | 587.97 | 586.99 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-12-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 11:10:00 | 574.85 | 580.24 | 0.00 | ORB-short ORB[584.30,588.75] vol=4.6x ATR=1.88 |
| Stop hit — per-position SL triggered | 2024-12-13 11:25:00 | 576.73 | 579.70 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-12-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:35:00 | 573.10 | 575.29 | 0.00 | ORB-short ORB[574.00,580.00] vol=1.6x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 09:45:00 | 571.35 | 574.22 | 0.00 | T1 1.5R @ 571.35 |
| Target hit | 2024-12-17 11:00:00 | 572.40 | 572.17 | 0.00 | Trail-exit close>VWAP |

### Cycle 53 — SELL (started 2024-12-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-19 10:35:00 | 543.95 | 547.87 | 0.00 | ORB-short ORB[547.30,555.00] vol=2.2x ATR=1.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 10:50:00 | 541.44 | 546.88 | 0.00 | T1 1.5R @ 541.44 |
| Stop hit — per-position SL triggered | 2024-12-19 11:15:00 | 543.95 | 546.20 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-23 11:05:00 | 546.50 | 544.12 | 0.00 | ORB-long ORB[539.05,543.60] vol=1.7x ATR=1.52 |
| Stop hit — per-position SL triggered | 2024-12-23 11:15:00 | 544.98 | 544.28 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:30:00 | 548.35 | 546.22 | 0.00 | ORB-long ORB[543.00,548.25] vol=3.1x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 10:20:00 | 550.95 | 547.26 | 0.00 | T1 1.5R @ 550.95 |
| Stop hit — per-position SL triggered | 2024-12-24 11:10:00 | 548.35 | 548.01 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-12-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:45:00 | 548.15 | 553.00 | 0.00 | ORB-short ORB[554.60,557.85] vol=1.6x ATR=1.39 |
| Stop hit — per-position SL triggered | 2024-12-26 14:20:00 | 549.54 | 550.18 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-12-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 09:30:00 | 553.00 | 550.03 | 0.00 | ORB-long ORB[547.00,552.00] vol=2.5x ATR=1.32 |
| Stop hit — per-position SL triggered | 2024-12-30 09:35:00 | 551.68 | 550.33 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-01-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-14 10:55:00 | 575.95 | 572.62 | 0.00 | ORB-long ORB[564.00,571.85] vol=9.6x ATR=1.70 |
| Stop hit — per-position SL triggered | 2025-01-14 12:10:00 | 574.25 | 574.39 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-01-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:35:00 | 575.30 | 576.79 | 0.00 | ORB-short ORB[576.50,580.90] vol=2.0x ATR=1.86 |
| Stop hit — per-position SL triggered | 2025-01-15 10:45:00 | 577.16 | 576.36 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-01-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 09:35:00 | 601.90 | 596.83 | 0.00 | ORB-long ORB[590.00,597.25] vol=2.7x ATR=2.01 |
| Stop hit — per-position SL triggered | 2025-01-16 09:40:00 | 599.89 | 597.51 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 11:15:00 | 599.60 | 598.15 | 0.00 | ORB-long ORB[592.40,599.00] vol=8.4x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 12:40:00 | 602.37 | 598.77 | 0.00 | T1 1.5R @ 602.37 |
| Target hit | 2025-01-20 15:20:00 | 605.60 | 601.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — BUY (started 2025-01-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:35:00 | 599.75 | 593.04 | 0.00 | ORB-long ORB[588.00,591.95] vol=2.0x ATR=2.35 |
| Stop hit — per-position SL triggered | 2025-01-29 10:50:00 | 597.40 | 594.35 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-01-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 11:00:00 | 599.55 | 594.38 | 0.00 | ORB-long ORB[589.25,596.00] vol=2.4x ATR=2.41 |
| Stop hit — per-position SL triggered | 2025-01-31 11:30:00 | 597.14 | 595.26 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-02-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-03 09:30:00 | 599.55 | 597.28 | 0.00 | ORB-long ORB[591.40,598.35] vol=3.7x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-03 10:25:00 | 604.36 | 599.02 | 0.00 | T1 1.5R @ 604.36 |
| Stop hit — per-position SL triggered | 2025-02-03 10:50:00 | 599.55 | 599.60 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-02-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 09:50:00 | 595.00 | 590.10 | 0.00 | ORB-long ORB[582.55,590.00] vol=3.4x ATR=2.41 |
| Stop hit — per-position SL triggered | 2025-02-07 09:55:00 | 592.59 | 590.52 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-02-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 09:40:00 | 531.90 | 535.54 | 0.00 | ORB-short ORB[534.60,539.00] vol=2.5x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 09:55:00 | 529.28 | 534.23 | 0.00 | T1 1.5R @ 529.28 |
| Stop hit — per-position SL triggered | 2025-02-21 11:10:00 | 531.90 | 531.38 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-24 09:35:00 | 537.05 | 535.38 | 0.00 | ORB-long ORB[530.25,536.45] vol=1.9x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-02-24 09:40:00 | 534.97 | 535.17 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-03-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-04 11:05:00 | 553.85 | 551.28 | 0.00 | ORB-long ORB[547.25,553.50] vol=1.6x ATR=1.85 |
| Stop hit — per-position SL triggered | 2025-03-04 11:15:00 | 552.00 | 551.31 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-05 10:15:00 | 544.80 | 548.48 | 0.00 | ORB-short ORB[548.00,551.45] vol=1.6x ATR=1.54 |
| Stop hit — per-position SL triggered | 2025-03-05 10:20:00 | 546.34 | 548.35 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-03-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 10:55:00 | 546.00 | 544.08 | 0.00 | ORB-long ORB[539.20,545.45] vol=4.6x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-10 11:40:00 | 549.15 | 544.48 | 0.00 | T1 1.5R @ 549.15 |
| Stop hit — per-position SL triggered | 2025-03-10 12:35:00 | 546.00 | 545.35 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-03-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 10:40:00 | 510.00 | 512.30 | 0.00 | ORB-short ORB[517.05,523.90] vol=2.7x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 10:45:00 | 506.92 | 511.80 | 0.00 | T1 1.5R @ 506.92 |
| Target hit | 2025-03-12 12:30:00 | 507.70 | 506.65 | 0.00 | Trail-exit close>VWAP |

### Cycle 72 — SELL (started 2025-03-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-17 09:55:00 | 500.00 | 501.99 | 0.00 | ORB-short ORB[500.10,505.40] vol=1.8x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 11:00:00 | 497.37 | 501.29 | 0.00 | T1 1.5R @ 497.37 |
| Target hit | 2025-03-17 15:20:00 | 491.50 | 495.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — BUY (started 2025-03-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 09:30:00 | 558.70 | 554.59 | 0.00 | ORB-long ORB[550.00,556.90] vol=1.5x ATR=2.24 |
| Stop hit — per-position SL triggered | 2025-03-25 09:40:00 | 556.46 | 555.47 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 11:05:00 | 624.00 | 2024-05-13 11:20:00 | 619.97 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-05-13 11:05:00 | 624.00 | 2024-05-13 12:35:00 | 624.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-15 09:30:00 | 648.20 | 2024-05-15 09:35:00 | 646.10 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-05-16 11:15:00 | 615.10 | 2024-05-16 11:25:00 | 616.76 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-05-17 11:10:00 | 625.95 | 2024-05-17 11:20:00 | 624.38 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-05-21 09:30:00 | 618.60 | 2024-05-21 09:55:00 | 615.35 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-05-21 09:30:00 | 618.60 | 2024-05-21 10:35:00 | 618.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-22 09:45:00 | 613.40 | 2024-05-22 10:05:00 | 610.78 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-05-22 09:45:00 | 613.40 | 2024-05-22 15:20:00 | 604.05 | TARGET_HIT | 0.50 | 1.52% |
| BUY | retest1 | 2024-05-27 09:40:00 | 628.90 | 2024-05-27 09:50:00 | 626.83 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-05-29 09:45:00 | 650.15 | 2024-05-29 09:55:00 | 647.15 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-05-31 10:05:00 | 649.40 | 2024-05-31 10:10:00 | 646.83 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-06-07 10:35:00 | 667.15 | 2024-06-07 10:40:00 | 665.17 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-06-10 09:30:00 | 664.40 | 2024-06-10 09:50:00 | 659.98 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-06-10 09:30:00 | 664.40 | 2024-06-10 13:40:00 | 656.90 | TARGET_HIT | 0.50 | 1.13% |
| BUY | retest1 | 2024-06-12 09:55:00 | 676.20 | 2024-06-12 10:40:00 | 673.91 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-06-25 10:40:00 | 689.00 | 2024-06-25 11:10:00 | 686.60 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-06-26 10:15:00 | 693.15 | 2024-06-26 10:20:00 | 689.83 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-07-01 11:05:00 | 675.00 | 2024-07-01 14:45:00 | 673.27 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-07-02 09:35:00 | 679.05 | 2024-07-02 09:40:00 | 677.19 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-07-05 09:50:00 | 670.00 | 2024-07-05 10:25:00 | 672.16 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-07-08 09:40:00 | 659.60 | 2024-07-08 09:50:00 | 662.06 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-12 10:05:00 | 641.00 | 2024-07-12 11:05:00 | 643.86 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-07-12 10:05:00 | 641.00 | 2024-07-12 11:35:00 | 641.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-19 11:00:00 | 627.60 | 2024-07-19 11:15:00 | 629.26 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-07-22 10:00:00 | 637.15 | 2024-07-22 10:15:00 | 640.13 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-07-22 10:00:00 | 637.15 | 2024-07-22 15:20:00 | 654.50 | TARGET_HIT | 0.50 | 2.72% |
| BUY | retest1 | 2024-07-24 09:50:00 | 659.10 | 2024-07-24 10:25:00 | 655.77 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-08-02 10:00:00 | 636.20 | 2024-08-02 10:10:00 | 638.06 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-08-09 10:25:00 | 627.05 | 2024-08-09 10:50:00 | 628.71 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-12 10:05:00 | 610.05 | 2024-08-12 10:25:00 | 606.90 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-08-12 10:05:00 | 610.05 | 2024-08-12 10:35:00 | 610.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-20 11:15:00 | 623.70 | 2024-08-20 12:05:00 | 622.06 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-08-21 09:35:00 | 616.60 | 2024-08-21 09:45:00 | 618.29 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-22 11:00:00 | 634.00 | 2024-08-22 12:10:00 | 632.37 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-08-23 09:35:00 | 629.55 | 2024-08-23 09:55:00 | 631.24 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-06 11:15:00 | 709.70 | 2024-09-06 11:40:00 | 707.29 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-09-09 11:00:00 | 712.95 | 2024-09-09 11:30:00 | 716.16 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-09-09 11:00:00 | 712.95 | 2024-09-09 11:45:00 | 712.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-11 10:55:00 | 721.60 | 2024-09-11 11:00:00 | 724.11 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-09-11 10:55:00 | 721.60 | 2024-09-11 11:15:00 | 721.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-13 10:55:00 | 723.40 | 2024-09-13 11:15:00 | 725.83 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-09-13 10:55:00 | 723.40 | 2024-09-13 13:55:00 | 723.75 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2024-09-18 10:40:00 | 722.45 | 2024-09-18 10:55:00 | 720.68 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-10-11 10:55:00 | 696.00 | 2024-10-11 11:15:00 | 693.41 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-10-11 10:55:00 | 696.00 | 2024-10-11 15:20:00 | 689.90 | TARGET_HIT | 0.50 | 0.88% |
| SELL | retest1 | 2024-10-17 10:55:00 | 689.00 | 2024-10-17 11:15:00 | 686.16 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-10-17 10:55:00 | 689.00 | 2024-10-17 15:20:00 | 687.10 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2024-10-18 09:50:00 | 683.50 | 2024-10-18 09:55:00 | 681.44 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-10-28 11:00:00 | 615.30 | 2024-10-28 11:10:00 | 612.40 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-11-04 10:15:00 | 623.00 | 2024-11-04 10:30:00 | 625.98 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-11-04 10:15:00 | 623.00 | 2024-11-04 10:35:00 | 623.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-05 09:55:00 | 616.65 | 2024-11-05 10:05:00 | 613.59 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-11-05 09:55:00 | 616.65 | 2024-11-05 14:00:00 | 613.00 | TARGET_HIT | 0.50 | 0.59% |
| SELL | retest1 | 2024-11-07 10:50:00 | 607.45 | 2024-11-07 12:00:00 | 605.46 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-11-07 10:50:00 | 607.45 | 2024-11-07 15:20:00 | 602.00 | TARGET_HIT | 0.50 | 0.90% |
| BUY | retest1 | 2024-11-18 11:15:00 | 578.95 | 2024-11-18 12:40:00 | 577.21 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-11-21 11:15:00 | 592.05 | 2024-11-21 11:20:00 | 590.13 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-11-27 09:40:00 | 584.95 | 2024-11-27 10:15:00 | 586.48 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-11-29 10:45:00 | 584.50 | 2024-11-29 11:20:00 | 582.54 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-11-29 10:45:00 | 584.50 | 2024-11-29 15:20:00 | 584.45 | TARGET_HIT | 0.50 | 0.01% |
| SELL | retest1 | 2024-12-02 11:00:00 | 579.55 | 2024-12-02 11:05:00 | 580.87 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-12-03 09:50:00 | 591.30 | 2024-12-03 10:05:00 | 589.59 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-05 10:30:00 | 594.60 | 2024-12-05 10:50:00 | 596.02 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-12-06 10:05:00 | 591.65 | 2024-12-06 10:20:00 | 593.46 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-10 09:30:00 | 589.50 | 2024-12-10 09:35:00 | 587.97 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-12-13 11:10:00 | 574.85 | 2024-12-13 11:25:00 | 576.73 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-12-17 09:35:00 | 573.10 | 2024-12-17 09:45:00 | 571.35 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-12-17 09:35:00 | 573.10 | 2024-12-17 11:00:00 | 572.40 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2024-12-19 10:35:00 | 543.95 | 2024-12-19 10:50:00 | 541.44 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-12-19 10:35:00 | 543.95 | 2024-12-19 11:15:00 | 543.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-23 11:05:00 | 546.50 | 2024-12-23 11:15:00 | 544.98 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-24 09:30:00 | 548.35 | 2024-12-24 10:20:00 | 550.95 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-12-24 09:30:00 | 548.35 | 2024-12-24 11:10:00 | 548.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-26 10:45:00 | 548.15 | 2024-12-26 14:20:00 | 549.54 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-12-30 09:30:00 | 553.00 | 2024-12-30 09:35:00 | 551.68 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-01-14 10:55:00 | 575.95 | 2025-01-14 12:10:00 | 574.25 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-15 09:35:00 | 575.30 | 2025-01-15 10:45:00 | 577.16 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-01-16 09:35:00 | 601.90 | 2025-01-16 09:40:00 | 599.89 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-01-20 11:15:00 | 599.60 | 2025-01-20 12:40:00 | 602.37 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-01-20 11:15:00 | 599.60 | 2025-01-20 15:20:00 | 605.60 | TARGET_HIT | 0.50 | 1.00% |
| BUY | retest1 | 2025-01-29 10:35:00 | 599.75 | 2025-01-29 10:50:00 | 597.40 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-01-31 11:00:00 | 599.55 | 2025-01-31 11:30:00 | 597.14 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-02-03 09:30:00 | 599.55 | 2025-02-03 10:25:00 | 604.36 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2025-02-03 09:30:00 | 599.55 | 2025-02-03 10:50:00 | 599.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-07 09:50:00 | 595.00 | 2025-02-07 09:55:00 | 592.59 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-02-21 09:40:00 | 531.90 | 2025-02-21 09:55:00 | 529.28 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-02-21 09:40:00 | 531.90 | 2025-02-21 11:10:00 | 531.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-24 09:35:00 | 537.05 | 2025-02-24 09:40:00 | 534.97 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-03-04 11:05:00 | 553.85 | 2025-03-04 11:15:00 | 552.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-03-05 10:15:00 | 544.80 | 2025-03-05 10:20:00 | 546.34 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-03-10 10:55:00 | 546.00 | 2025-03-10 11:40:00 | 549.15 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-03-10 10:55:00 | 546.00 | 2025-03-10 12:35:00 | 546.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-12 10:40:00 | 510.00 | 2025-03-12 10:45:00 | 506.92 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-03-12 10:40:00 | 510.00 | 2025-03-12 12:30:00 | 507.70 | TARGET_HIT | 0.50 | 0.45% |
| SELL | retest1 | 2025-03-17 09:55:00 | 500.00 | 2025-03-17 11:00:00 | 497.37 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-03-17 09:55:00 | 500.00 | 2025-03-17 15:20:00 | 491.50 | TARGET_HIT | 0.50 | 1.70% |
| BUY | retest1 | 2025-03-25 09:30:00 | 558.70 | 2025-03-25 09:40:00 | 556.46 | STOP_HIT | 1.00 | -0.40% |
