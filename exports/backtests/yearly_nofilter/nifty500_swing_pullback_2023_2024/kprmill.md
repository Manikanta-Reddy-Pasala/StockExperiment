# K.P.R. Mill Ltd. (KPRMILL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 927.70
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
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / Stop hits / Partials:** 2 / 4 / 2
- **Avg / median % per leg:** 1.42% / 1.55%
- **Sum % (uncompounded):** 11.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 2 | 4 | 2 | 1.42% | 11.3% |
| BUY @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 2 | 4 | 2 | 1.42% | 11.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 4 | 50.0% | 2 | 4 | 2 | 1.42% | 11.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-10 00:00:00 | 661.45 | 598.12 | 638.15 | Stage2 pullback-breakout RSI=59 vol=8.0x ATR=19.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-18 00:00:00 | 699.96 | 601.42 | 649.21 | T1 booked 50% @ 699.96 |
| Target hit | 2023-09-15 00:00:00 | 725.05 | 627.36 | 735.87 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-10-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 00:00:00 | 777.85 | 638.42 | 742.91 | Stage2 pullback-breakout RSI=66 vol=2.0x ATR=24.80 |
| Stop hit — per-position SL triggered | 2023-10-04 00:00:00 | 740.66 | 639.50 | 743.30 | SL hit (bars_held=1) |

### Cycle 3 — BUY (started 2023-10-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 00:00:00 | 793.05 | 648.84 | 748.48 | Stage2 pullback-breakout RSI=65 vol=2.8x ATR=24.92 |
| Stop hit — per-position SL triggered | 2023-10-26 00:00:00 | 755.67 | 658.21 | 775.30 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2023-11-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 00:00:00 | 823.35 | 677.82 | 786.07 | Stage2 pullback-breakout RSI=63 vol=1.8x ATR=26.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-22 00:00:00 | 876.17 | 683.04 | 804.11 | T1 booked 50% @ 876.17 |
| Target hit | 2023-12-07 00:00:00 | 836.15 | 699.83 | 838.27 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-02-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 00:00:00 | 779.35 | 732.49 | 749.62 | Stage2 pullback-breakout RSI=58 vol=2.6x ATR=22.66 |
| Stop hit — per-position SL triggered | 2024-03-11 00:00:00 | 775.30 | 736.92 | 769.47 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-03-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 00:00:00 | 826.25 | 741.85 | 784.12 | Stage2 pullback-breakout RSI=64 vol=2.0x ATR=26.54 |
| Stop hit — per-position SL triggered | 2024-04-10 00:00:00 | 809.25 | 750.41 | 812.86 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-10 00:00:00 | 661.45 | 2023-08-18 00:00:00 | 699.96 | PARTIAL | 0.50 | 5.82% |
| BUY | retest1 | 2023-08-10 00:00:00 | 661.45 | 2023-09-15 00:00:00 | 725.05 | TARGET_HIT | 0.50 | 9.62% |
| BUY | retest1 | 2023-10-03 00:00:00 | 777.85 | 2023-10-04 00:00:00 | 740.66 | STOP_HIT | 1.00 | -4.78% |
| BUY | retest1 | 2023-10-17 00:00:00 | 793.05 | 2023-10-26 00:00:00 | 755.67 | STOP_HIT | 1.00 | -4.71% |
| BUY | retest1 | 2023-11-17 00:00:00 | 823.35 | 2023-11-22 00:00:00 | 876.17 | PARTIAL | 0.50 | 6.42% |
| BUY | retest1 | 2023-11-17 00:00:00 | 823.35 | 2023-12-07 00:00:00 | 836.15 | TARGET_HIT | 0.50 | 1.55% |
| BUY | retest1 | 2024-02-26 00:00:00 | 779.35 | 2024-03-11 00:00:00 | 775.30 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2024-03-26 00:00:00 | 826.25 | 2024-04-10 00:00:00 | 809.25 | STOP_HIT | 1.00 | -2.06% |
