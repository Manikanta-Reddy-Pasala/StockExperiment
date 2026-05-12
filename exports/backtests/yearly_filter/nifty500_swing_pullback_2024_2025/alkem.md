# Alkem Laboratories Ltd. (ALKEM)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 5520.00
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
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 1
- **Avg / median % per leg:** 4.14% / 5.39%
- **Sum % (uncompounded):** 12.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.14% | 12.4% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.14% | 12.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.14% | 12.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 00:00:00 | 5512.25 | 4847.07 | 5259.66 | Stage2 pullback-breakout RSI=67 vol=1.9x ATR=148.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 00:00:00 | 5809.25 | 4864.80 | 5347.65 | T1 booked 50% @ 5809.25 |
| Target hit | 2024-09-18 00:00:00 | 6090.30 | 5143.42 | 6114.84 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-12-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 00:00:00 | 5633.90 | 5439.79 | 5511.80 | Stage2 pullback-breakout RSI=56 vol=1.6x ATR=129.31 |
| Stop hit — per-position SL triggered | 2025-01-09 00:00:00 | 5439.93 | 5446.33 | 5522.07 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-07 00:00:00 | 5512.25 | 2024-08-09 00:00:00 | 5809.25 | PARTIAL | 0.50 | 5.39% |
| BUY | retest1 | 2024-08-07 00:00:00 | 5512.25 | 2024-09-18 00:00:00 | 6090.30 | TARGET_HIT | 0.50 | 10.49% |
| BUY | retest1 | 2024-12-31 00:00:00 | 5633.90 | 2025-01-09 00:00:00 | 5439.93 | STOP_HIT | 1.00 | -3.44% |
