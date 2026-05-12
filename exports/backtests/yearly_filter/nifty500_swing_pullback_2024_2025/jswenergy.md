# JSW Energy Ltd. (JSWENERGY)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 554.55
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
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 2
- **Avg / median % per leg:** 3.65% / 7.04%
- **Sum % (uncompounded):** 14.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 2 | 2 | 3.65% | 14.6% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 2 | 2 | 3.65% | 14.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 2 | 2 | 3.65% | 14.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-09 00:00:00 | 730.75 | 596.53 | 708.22 | Stage2 pullback-breakout RSI=58 vol=2.7x ATR=25.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 00:00:00 | 782.17 | 610.33 | 737.77 | T1 booked 50% @ 782.17 |
| Stop hit — per-position SL triggered | 2024-09-27 00:00:00 | 730.75 | 618.42 | 752.03 | SL hit (bars_held=14) |

### Cycle 2 — BUY (started 2024-11-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 00:00:00 | 714.80 | 635.66 | 686.12 | Stage2 pullback-breakout RSI=56 vol=2.8x ATR=27.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 00:00:00 | 769.00 | 638.56 | 698.81 | T1 booked 50% @ 769.00 |
| Stop hit — per-position SL triggered | 2024-11-18 00:00:00 | 714.80 | 641.11 | 705.35 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-09 00:00:00 | 730.75 | 2024-09-20 00:00:00 | 782.17 | PARTIAL | 0.50 | 7.04% |
| BUY | retest1 | 2024-09-09 00:00:00 | 730.75 | 2024-09-27 00:00:00 | 730.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-07 00:00:00 | 714.80 | 2024-11-12 00:00:00 | 769.00 | PARTIAL | 0.50 | 7.58% |
| BUY | retest1 | 2024-11-07 00:00:00 | 714.80 | 2024-11-18 00:00:00 | 714.80 | STOP_HIT | 0.50 | 0.00% |
