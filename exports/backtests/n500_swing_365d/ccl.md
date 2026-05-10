# CCL Products (I) Ltd. (CCL)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1123.10
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 6
- **Target hits / Stop hits / Partials:** 1 / 8 / 1
- **Avg / median % per leg:** -0.78% / -0.17%
- **Sum % (uncompounded):** -7.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 4 | 40.0% | 1 | 8 | 1 | -0.78% | -7.8% |
| BUY @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 8 | 1 | -0.78% | -7.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 4 | 40.0% | 1 | 8 | 1 | -0.78% | -7.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 05:30:00 | 884.95 | 714.82 | 837.09 | Stage2 pullback-breakout RSI=65 vol=2.2x ATR=33.00 |
| Stop hit — per-position SL triggered | 2025-07-17 05:30:00 | 888.60 | 731.15 | 868.00 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-08-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 05:30:00 | 894.10 | 744.32 | 861.51 | Stage2 pullback-breakout RSI=62 vol=4.0x ATR=32.71 |
| Stop hit — per-position SL triggered | 2025-08-06 05:30:00 | 845.04 | 748.94 | 871.12 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2025-08-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 05:30:00 | 920.50 | 758.63 | 875.16 | Stage2 pullback-breakout RSI=64 vol=2.4x ATR=33.92 |
| Stop hit — per-position SL triggered | 2025-08-25 05:30:00 | 869.62 | 763.83 | 880.12 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2025-11-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 05:30:00 | 877.15 | 803.56 | 855.03 | Stage2 pullback-breakout RSI=58 vol=1.8x ATR=21.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 05:30:00 | 920.10 | 806.09 | 869.17 | T1 booked 50% @ 920.10 |
| Target hit | 2025-12-01 05:30:00 | 984.20 | 839.82 | 986.94 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2025-12-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 05:30:00 | 1025.30 | 851.65 | 981.92 | Stage2 pullback-breakout RSI=60 vol=2.0x ATR=36.23 |
| Stop hit — per-position SL triggered | 2025-12-18 05:30:00 | 970.95 | 857.04 | 984.30 | SL hit (bars_held=4) |

### Cycle 6 — BUY (started 2026-01-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-19 05:30:00 | 983.75 | 873.64 | 955.31 | Stage2 pullback-breakout RSI=58 vol=1.5x ATR=30.19 |
| Stop hit — per-position SL triggered | 2026-01-21 05:30:00 | 938.47 | 875.06 | 953.48 | SL hit (bars_held=2) |

### Cycle 7 — BUY (started 2026-02-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 05:30:00 | 1002.85 | 882.99 | 960.15 | Stage2 pullback-breakout RSI=61 vol=1.5x ATR=34.34 |
| Stop hit — per-position SL triggered | 2026-02-18 05:30:00 | 1001.15 | 894.79 | 989.99 | Time-stop (10d <3%) |

### Cycle 8 — BUY (started 2026-03-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 05:30:00 | 1045.70 | 910.58 | 1016.19 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=38.21 |
| Stop hit — per-position SL triggered | 2026-03-27 05:30:00 | 1058.70 | 926.96 | 1046.24 | Time-stop (10d <3%) |

### Cycle 9 — BUY (started 2026-04-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 05:30:00 | 1116.90 | 933.11 | 1059.46 | Stage2 pullback-breakout RSI=64 vol=1.6x ATR=41.02 |
| Stop hit — per-position SL triggered | 2026-04-13 05:30:00 | 1055.37 | 941.41 | 1075.80 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-03 05:30:00 | 884.95 | 2025-07-17 05:30:00 | 888.60 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest1 | 2025-08-01 05:30:00 | 894.10 | 2025-08-06 05:30:00 | 845.04 | STOP_HIT | 1.00 | -5.49% |
| BUY | retest1 | 2025-08-19 05:30:00 | 920.50 | 2025-08-25 05:30:00 | 869.62 | STOP_HIT | 1.00 | -5.53% |
| BUY | retest1 | 2025-11-03 05:30:00 | 877.15 | 2025-11-06 05:30:00 | 920.10 | PARTIAL | 0.50 | 4.90% |
| BUY | retest1 | 2025-11-03 05:30:00 | 877.15 | 2025-12-01 05:30:00 | 984.20 | TARGET_HIT | 0.50 | 12.20% |
| BUY | retest1 | 2025-12-12 05:30:00 | 1025.30 | 2025-12-18 05:30:00 | 970.95 | STOP_HIT | 1.00 | -5.30% |
| BUY | retest1 | 2026-01-19 05:30:00 | 983.75 | 2026-01-21 05:30:00 | 938.47 | STOP_HIT | 1.00 | -4.60% |
| BUY | retest1 | 2026-02-04 05:30:00 | 1002.85 | 2026-02-18 05:30:00 | 1001.15 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-03-10 05:30:00 | 1045.70 | 2026-03-27 05:30:00 | 1058.70 | STOP_HIT | 1.00 | 1.24% |
| BUY | retest1 | 2026-04-06 05:30:00 | 1116.90 | 2026-04-13 05:30:00 | 1055.37 | STOP_HIT | 1.00 | -5.51% |
