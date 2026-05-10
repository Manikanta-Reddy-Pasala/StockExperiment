# Intellect Design Arena Ltd. (INTELLECT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 808.00
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 8
- **Target hits / Stop hits / Partials:** 4 / 8 / 6
- **Avg / median % per leg:** 0.29% / 0.53%
- **Sum % (uncompounded):** 5.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.15% | 1.0% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.15% | 1.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.38% | 4.2% |
| SELL @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.38% | 4.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 10 | 55.6% | 4 | 8 | 6 | 0.29% | 5.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:35:00 | 693.55 | 699.99 | 0.00 | ORB-short ORB[696.30,705.70] vol=2.0x ATR=2.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:40:00 | 689.68 | 699.26 | 0.00 | T1 1.5R @ 689.68 |
| Stop hit — per-position SL triggered | 2026-02-25 10:55:00 | 693.55 | 698.25 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:40:00 | 693.30 | 699.23 | 0.00 | ORB-short ORB[696.00,705.30] vol=2.3x ATR=2.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:30:00 | 688.99 | 695.68 | 0.00 | T1 1.5R @ 688.99 |
| Stop hit — per-position SL triggered | 2026-02-27 11:00:00 | 693.30 | 694.75 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-03-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 09:45:00 | 676.55 | 672.52 | 0.00 | ORB-long ORB[666.10,676.00] vol=1.8x ATR=3.82 |
| Stop hit — per-position SL triggered | 2026-03-04 09:50:00 | 672.73 | 672.64 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:55:00 | 664.00 | 669.67 | 0.00 | ORB-short ORB[667.55,676.35] vol=1.7x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 13:05:00 | 659.73 | 664.91 | 0.00 | T1 1.5R @ 659.73 |
| Target hit | 2026-03-05 15:20:00 | 659.00 | 661.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-03-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:30:00 | 677.20 | 680.28 | 0.00 | ORB-short ORB[678.20,686.00] vol=1.9x ATR=2.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:35:00 | 673.62 | 677.63 | 0.00 | T1 1.5R @ 673.62 |
| Target hit | 2026-03-11 15:20:00 | 661.30 | 669.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2026-03-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:10:00 | 661.50 | 655.42 | 0.00 | ORB-long ORB[650.30,660.05] vol=1.9x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 11:55:00 | 664.80 | 657.71 | 0.00 | T1 1.5R @ 664.80 |
| Target hit | 2026-03-12 15:20:00 | 666.10 | 665.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-03-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:55:00 | 658.25 | 662.10 | 0.00 | ORB-short ORB[660.40,666.90] vol=1.9x ATR=2.47 |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 660.72 | 661.67 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:30:00 | 653.40 | 649.14 | 0.00 | ORB-long ORB[645.40,651.00] vol=1.9x ATR=2.80 |
| Stop hit — per-position SL triggered | 2026-03-25 09:35:00 | 650.60 | 649.38 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:50:00 | 675.70 | 683.40 | 0.00 | ORB-short ORB[681.20,690.10] vol=1.8x ATR=2.89 |
| Stop hit — per-position SL triggered | 2026-04-09 10:45:00 | 678.59 | 681.33 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 672.00 | 667.17 | 0.00 | ORB-long ORB[661.00,670.95] vol=1.7x ATR=3.69 |
| Stop hit — per-position SL triggered | 2026-04-15 10:35:00 | 668.31 | 669.34 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:55:00 | 662.55 | 669.83 | 0.00 | ORB-short ORB[668.40,677.50] vol=1.6x ATR=2.77 |
| Stop hit — per-position SL triggered | 2026-04-22 10:10:00 | 665.32 | 668.97 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:00:00 | 700.90 | 695.10 | 0.00 | ORB-long ORB[690.40,698.70] vol=1.8x ATR=3.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 10:15:00 | 705.68 | 697.73 | 0.00 | T1 1.5R @ 705.68 |
| Target hit | 2026-04-23 13:55:00 | 705.75 | 705.92 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-25 10:35:00 | 693.55 | 2026-02-25 10:40:00 | 689.68 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-02-25 10:35:00 | 693.55 | 2026-02-25 10:55:00 | 693.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 09:40:00 | 693.30 | 2026-02-27 10:30:00 | 688.99 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-02-27 09:40:00 | 693.30 | 2026-02-27 11:00:00 | 693.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-04 09:45:00 | 676.55 | 2026-03-04 09:50:00 | 672.73 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2026-03-05 09:55:00 | 664.00 | 2026-03-05 13:05:00 | 659.73 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2026-03-05 09:55:00 | 664.00 | 2026-03-05 15:20:00 | 659.00 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2026-03-11 09:30:00 | 677.20 | 2026-03-11 10:35:00 | 673.62 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-03-11 09:30:00 | 677.20 | 2026-03-11 15:20:00 | 661.30 | TARGET_HIT | 0.50 | 2.35% |
| BUY | retest1 | 2026-03-12 11:10:00 | 661.50 | 2026-03-12 11:55:00 | 664.80 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-03-12 11:10:00 | 661.50 | 2026-03-12 15:20:00 | 666.10 | TARGET_HIT | 0.50 | 0.70% |
| SELL | retest1 | 2026-03-13 09:55:00 | 658.25 | 2026-03-13 10:15:00 | 660.72 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-25 09:30:00 | 653.40 | 2026-03-25 09:35:00 | 650.60 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-04-09 09:50:00 | 675.70 | 2026-04-09 10:45:00 | 678.59 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-15 09:40:00 | 672.00 | 2026-04-15 10:35:00 | 668.31 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2026-04-22 09:55:00 | 662.55 | 2026-04-22 10:10:00 | 665.32 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-23 10:00:00 | 700.90 | 2026-04-23 10:15:00 | 705.68 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-04-23 10:00:00 | 700.90 | 2026-04-23 13:55:00 | 705.75 | TARGET_HIT | 0.50 | 0.69% |
