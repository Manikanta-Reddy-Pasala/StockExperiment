# Finolex Cables Ltd. (FINCABLES)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1113.10
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 2
- **Avg / median % per leg:** 2.51% / 0.00%
- **Sum % (uncompounded):** 20.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 3 | 37.5% | 1 | 5 | 2 | 2.51% | 20.1% |
| BUY @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 1 | 5 | 2 | 2.51% | 20.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 3 | 37.5% | 1 | 5 | 2 | 2.51% | 20.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 00:00:00 | 858.75 | 697.69 | 823.85 | Stage2 pullback-breakout RSI=57 vol=3.0x ATR=29.48 |
| Stop hit — per-position SL triggered | 2023-07-07 00:00:00 | 814.53 | 704.49 | 828.56 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2023-07-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-13 00:00:00 | 865.00 | 709.37 | 829.34 | Stage2 pullback-breakout RSI=59 vol=4.7x ATR=31.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 00:00:00 | 928.44 | 717.00 | 855.24 | T1 booked 50% @ 928.44 |
| Target hit | 2023-09-12 00:00:00 | 1079.50 | 827.07 | 1089.35 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-09-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 00:00:00 | 1195.55 | 856.60 | 1106.91 | Stage2 pullback-breakout RSI=69 vol=6.6x ATR=42.67 |
| Stop hit — per-position SL triggered | 2023-09-29 00:00:00 | 1131.55 | 859.23 | 1108.26 | SL hit (bars_held=1) |

### Cycle 4 — BUY (started 2024-01-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-25 00:00:00 | 1106.80 | 944.97 | 1065.40 | Stage2 pullback-breakout RSI=65 vol=2.1x ATR=33.46 |
| Stop hit — per-position SL triggered | 2024-02-02 00:00:00 | 1056.60 | 952.03 | 1074.44 | SL hit (bars_held=5) |

### Cycle 5 — BUY (started 2024-03-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 00:00:00 | 967.35 | 952.51 | 916.64 | Stage2 pullback-breakout RSI=58 vol=5.3x ATR=35.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 00:00:00 | 1039.01 | 954.88 | 952.00 | T1 booked 50% @ 1039.01 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 967.35 | 957.31 | 972.01 | SL hit (bars_held=12) |

### Cycle 6 — BUY (started 2024-04-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 00:00:00 | 1053.25 | 961.83 | 996.42 | Stage2 pullback-breakout RSI=68 vol=1.7x ATR=30.99 |
| Stop hit — per-position SL triggered | 2024-05-10 00:00:00 | 1006.76 | 967.38 | 1019.27 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-30 00:00:00 | 858.75 | 2023-07-07 00:00:00 | 814.53 | STOP_HIT | 1.00 | -5.15% |
| BUY | retest1 | 2023-07-13 00:00:00 | 865.00 | 2023-07-19 00:00:00 | 928.44 | PARTIAL | 0.50 | 7.33% |
| BUY | retest1 | 2023-07-13 00:00:00 | 865.00 | 2023-09-12 00:00:00 | 1079.50 | TARGET_HIT | 0.50 | 24.80% |
| BUY | retest1 | 2023-09-28 00:00:00 | 1195.55 | 2023-09-29 00:00:00 | 1131.55 | STOP_HIT | 1.00 | -5.35% |
| BUY | retest1 | 2024-01-25 00:00:00 | 1106.80 | 2024-02-02 00:00:00 | 1056.60 | STOP_HIT | 1.00 | -4.54% |
| BUY | retest1 | 2024-03-26 00:00:00 | 967.35 | 2024-04-04 00:00:00 | 1039.01 | PARTIAL | 0.50 | 7.41% |
| BUY | retest1 | 2024-03-26 00:00:00 | 967.35 | 2024-04-15 00:00:00 | 967.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-30 00:00:00 | 1053.25 | 2024-05-10 00:00:00 | 1006.76 | STOP_HIT | 1.00 | -4.41% |
