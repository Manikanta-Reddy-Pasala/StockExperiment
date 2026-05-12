# Zen Technologies Ltd. (ZENTEC)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1588.50
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
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 2
- **Avg / median % per leg:** 0.39% / 0.00%
- **Sum % (uncompounded):** 2.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 0 | 5 | 2 | 0.39% | 2.7% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 5 | 2 | 0.39% | 2.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 2 | 28.6% | 0 | 5 | 2 | 0.39% | 2.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 00:00:00 | 742.15 | 525.92 | 725.53 | Stage2 pullback-breakout RSI=53 vol=2.0x ATR=32.44 |
| Stop hit — per-position SL triggered | 2023-10-31 00:00:00 | 693.49 | 529.60 | 723.08 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2023-11-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 00:00:00 | 758.55 | 555.38 | 729.30 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=26.76 |
| Stop hit — per-position SL triggered | 2023-12-05 00:00:00 | 754.90 | 574.90 | 749.05 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-12-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 00:00:00 | 810.15 | 599.53 | 755.10 | Stage2 pullback-breakout RSI=67 vol=4.6x ATR=27.72 |
| Stop hit — per-position SL triggered | 2024-01-02 00:00:00 | 768.56 | 607.08 | 766.94 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2024-01-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 00:00:00 | 840.40 | 635.07 | 764.00 | Stage2 pullback-breakout RSI=68 vol=3.9x ATR=28.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 00:00:00 | 897.96 | 639.62 | 782.20 | T1 booked 50% @ 897.96 |
| Stop hit — per-position SL triggered | 2024-02-05 00:00:00 | 840.40 | 641.60 | 787.49 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-02-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-29 00:00:00 | 865.85 | 671.51 | 818.30 | Stage2 pullback-breakout RSI=66 vol=1.6x ATR=34.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-02 00:00:00 | 935.70 | 676.66 | 839.10 | T1 booked 50% @ 935.70 |
| Stop hit — per-position SL triggered | 2024-03-13 00:00:00 | 865.85 | 692.85 | 875.28 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-27 00:00:00 | 742.15 | 2023-10-31 00:00:00 | 693.49 | STOP_HIT | 1.00 | -6.56% |
| BUY | retest1 | 2023-11-20 00:00:00 | 758.55 | 2023-12-05 00:00:00 | 754.90 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2023-12-27 00:00:00 | 810.15 | 2024-01-02 00:00:00 | 768.56 | STOP_HIT | 1.00 | -5.13% |
| BUY | retest1 | 2024-01-31 00:00:00 | 840.40 | 2024-02-02 00:00:00 | 897.96 | PARTIAL | 0.50 | 6.85% |
| BUY | retest1 | 2024-01-31 00:00:00 | 840.40 | 2024-02-05 00:00:00 | 840.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-29 00:00:00 | 865.85 | 2024-03-02 00:00:00 | 935.70 | PARTIAL | 0.50 | 8.07% |
| BUY | retest1 | 2024-02-29 00:00:00 | 865.85 | 2024-03-13 00:00:00 | 865.85 | STOP_HIT | 0.50 | 0.00% |
