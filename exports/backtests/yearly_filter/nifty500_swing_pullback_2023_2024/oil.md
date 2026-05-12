# Oil India Ltd. (OIL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 455.90
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 0.08% / 5.18%
- **Sum % (uncompounded):** 0.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.08% | 0.3% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.08% | 0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.08% | 0.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 00:00:00 | 194.83 | 168.70 | 187.21 | Stage2 pullback-breakout RSI=64 vol=4.3x ATR=5.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 00:00:00 | 206.46 | 170.45 | 192.61 | T1 booked 50% @ 206.46 |
| Target hit | 2023-10-23 00:00:00 | 204.93 | 174.49 | 205.52 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 00:00:00 | 218.80 | 181.95 | 206.06 | Stage2 pullback-breakout RSI=64 vol=4.8x ATR=7.59 |
| Stop hit — per-position SL triggered | 2023-12-13 00:00:00 | 207.42 | 183.95 | 208.61 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2024-04-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 00:00:00 | 423.77 | 265.37 | 394.21 | Stage2 pullback-breakout RSI=64 vol=2.1x ATR=21.97 |
| Stop hit — per-position SL triggered | 2024-04-19 00:00:00 | 399.90 | 279.40 | 405.14 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-28 00:00:00 | 194.83 | 2023-10-09 00:00:00 | 206.46 | PARTIAL | 0.50 | 5.97% |
| BUY | retest1 | 2023-09-28 00:00:00 | 194.83 | 2023-10-23 00:00:00 | 204.93 | TARGET_HIT | 0.50 | 5.18% |
| BUY | retest1 | 2023-12-04 00:00:00 | 218.80 | 2023-12-13 00:00:00 | 207.42 | STOP_HIT | 1.00 | -5.20% |
| BUY | retest1 | 2024-04-03 00:00:00 | 423.77 | 2024-04-19 00:00:00 | 399.90 | STOP_HIT | 1.00 | -5.63% |
