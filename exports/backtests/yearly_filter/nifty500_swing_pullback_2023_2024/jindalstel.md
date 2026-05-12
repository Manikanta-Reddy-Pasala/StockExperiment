# Jindal Steel Ltd. (JINDALSTEL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1241.60
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 2
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 1.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.18% | 1.1% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.18% | 1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.18% | 1.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-09 00:00:00 | 678.95 | 560.52 | 647.03 | Stage2 pullback-breakout RSI=68 vol=1.8x ATR=15.79 |
| Stop hit — per-position SL triggered | 2023-08-16 00:00:00 | 655.26 | 564.86 | 654.49 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2024-01-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 00:00:00 | 749.75 | 645.73 | 729.49 | Stage2 pullback-breakout RSI=62 vol=1.7x ATR=18.90 |
| Stop hit — per-position SL triggered | 2024-01-17 00:00:00 | 721.40 | 646.49 | 728.78 | SL hit (bars_held=1) |

### Cycle 3 — BUY (started 2024-01-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 00:00:00 | 746.20 | 652.27 | 725.53 | Stage2 pullback-breakout RSI=58 vol=1.9x ATR=21.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 00:00:00 | 789.55 | 658.95 | 744.97 | T1 booked 50% @ 789.55 |
| Stop hit — per-position SL triggered | 2024-02-09 00:00:00 | 746.20 | 661.13 | 749.29 | SL hit (bars_held=8) |

### Cycle 4 — BUY (started 2024-03-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-01 00:00:00 | 822.85 | 676.24 | 769.21 | Stage2 pullback-breakout RSI=66 vol=2.2x ATR=24.89 |
| Stop hit — per-position SL triggered | 2024-03-13 00:00:00 | 785.51 | 687.37 | 795.51 | SL hit (bars_held=8) |

### Cycle 5 — BUY (started 2024-03-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 00:00:00 | 821.55 | 693.51 | 795.07 | Stage2 pullback-breakout RSI=57 vol=1.5x ATR=29.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 00:00:00 | 879.71 | 702.81 | 822.85 | T1 booked 50% @ 879.71 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-09 00:00:00 | 678.95 | 2023-08-16 00:00:00 | 655.26 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest1 | 2024-01-16 00:00:00 | 749.75 | 2024-01-17 00:00:00 | 721.40 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest1 | 2024-01-30 00:00:00 | 746.20 | 2024-02-07 00:00:00 | 789.55 | PARTIAL | 0.50 | 5.81% |
| BUY | retest1 | 2024-01-30 00:00:00 | 746.20 | 2024-02-09 00:00:00 | 746.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-01 00:00:00 | 822.85 | 2024-03-13 00:00:00 | 785.51 | STOP_HIT | 1.00 | -4.54% |
| BUY | retest1 | 2024-03-21 00:00:00 | 821.55 | 2024-04-02 00:00:00 | 879.71 | PARTIAL | 0.50 | 7.08% |
