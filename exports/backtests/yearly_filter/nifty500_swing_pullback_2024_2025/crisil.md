# CRISIL Ltd. (CRISIL)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 4162.30
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 1
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 0.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.19% | 0.6% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.19% | 0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.19% | 0.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-10-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 00:00:00 | 4711.85 | 4398.33 | 4555.41 | Stage2 pullback-breakout RSI=58 vol=1.8x ATR=130.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 00:00:00 | 4972.25 | 4411.89 | 4618.51 | T1 booked 50% @ 4972.25 |
| Stop hit — per-position SL triggered | 2024-10-18 00:00:00 | 4711.85 | 4414.78 | 4626.46 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2024-12-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-13 00:00:00 | 5817.75 | 4707.68 | 5393.96 | Stage2 pullback-breakout RSI=69 vol=6.5x ATR=191.94 |
| Stop hit — per-position SL triggered | 2024-12-27 00:00:00 | 5529.84 | 4795.69 | 5582.33 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-10-11 00:00:00 | 4711.85 | 2024-10-17 00:00:00 | 4972.25 | PARTIAL | 0.50 | 5.53% |
| BUY | retest1 | 2024-10-11 00:00:00 | 4711.85 | 2024-10-18 00:00:00 | 4711.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-13 00:00:00 | 5817.75 | 2024-12-27 00:00:00 | 5529.84 | STOP_HIT | 1.00 | -4.95% |
