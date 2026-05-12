# Zensar Technolgies Ltd. (ZENSARTECH)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 516.65
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / Stop hits / Partials:** 0 / 5 / 3
- **Avg / median % per leg:** 1.50% / 2.54%
- **Sum % (uncompounded):** 12.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 0 | 5 | 3 | 1.50% | 12.0% |
| BUY @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 0 | 5 | 3 | 1.50% | 12.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 4 | 50.0% | 0 | 5 | 3 | 1.50% | 12.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 00:00:00 | 769.70 | 609.46 | 727.95 | Stage2 pullback-breakout RSI=66 vol=4.7x ATR=28.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 00:00:00 | 827.34 | 611.23 | 733.57 | T1 booked 50% @ 827.34 |
| Stop hit — per-position SL triggered | 2024-07-19 00:00:00 | 769.70 | 615.99 | 743.35 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2024-07-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 00:00:00 | 808.85 | 623.58 | 755.47 | Stage2 pullback-breakout RSI=67 vol=2.3x ATR=36.68 |
| Stop hit — per-position SL triggered | 2024-08-02 00:00:00 | 753.84 | 631.73 | 768.43 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2024-08-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 00:00:00 | 797.60 | 642.29 | 761.94 | Stage2 pullback-breakout RSI=59 vol=3.4x ATR=31.91 |
| Stop hit — per-position SL triggered | 2024-08-30 00:00:00 | 767.90 | 655.41 | 772.34 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-11-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-04 00:00:00 | 712.80 | 676.46 | 697.65 | Stage2 pullback-breakout RSI=55 vol=3.6x ATR=22.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 00:00:00 | 758.49 | 679.70 | 713.42 | T1 booked 50% @ 758.49 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 712.80 | 679.88 | 711.95 | SL hit (bars_held=7) |

### Cycle 5 — BUY (started 2024-11-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 00:00:00 | 770.25 | 683.32 | 721.43 | Stage2 pullback-breakout RSI=66 vol=1.7x ATR=23.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 00:00:00 | 817.38 | 691.15 | 754.49 | T1 booked 50% @ 817.38 |
| Stop hit — per-position SL triggered | 2024-12-18 00:00:00 | 789.85 | 697.50 | 774.95 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-12 00:00:00 | 769.70 | 2024-07-15 00:00:00 | 827.34 | PARTIAL | 0.50 | 7.49% |
| BUY | retest1 | 2024-07-12 00:00:00 | 769.70 | 2024-07-19 00:00:00 | 769.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 00:00:00 | 808.85 | 2024-08-02 00:00:00 | 753.84 | STOP_HIT | 1.00 | -6.80% |
| BUY | retest1 | 2024-08-16 00:00:00 | 797.60 | 2024-08-30 00:00:00 | 767.90 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest1 | 2024-11-04 00:00:00 | 712.80 | 2024-11-12 00:00:00 | 758.49 | PARTIAL | 0.50 | 6.41% |
| BUY | retest1 | 2024-11-04 00:00:00 | 712.80 | 2024-11-13 00:00:00 | 712.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-27 00:00:00 | 770.25 | 2024-12-10 00:00:00 | 817.38 | PARTIAL | 0.50 | 6.12% |
| BUY | retest1 | 2024-11-27 00:00:00 | 770.25 | 2024-12-18 00:00:00 | 789.85 | STOP_HIT | 0.50 | 2.54% |
