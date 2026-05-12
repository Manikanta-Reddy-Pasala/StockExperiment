# Bank of Maharashtra (MAHABANK)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 80.32
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -4.40% / -4.35%
- **Sum % (uncompounded):** -17.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.40% | -17.6% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.40% | -17.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.40% | -17.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-01-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-15 00:00:00 | 50.40 | 39.02 | 46.56 | Stage2 pullback-breakout RSI=70 vol=2.9x ATR=1.46 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 48.21 | 39.36 | 47.62 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2024-03-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 00:00:00 | 63.60 | 44.90 | 59.99 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=2.78 |
| Stop hit — per-position SL triggered | 2024-03-12 00:00:00 | 59.43 | 45.54 | 60.29 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2024-03-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 00:00:00 | 62.35 | 46.93 | 59.58 | Stage2 pullback-breakout RSI=59 vol=1.8x ATR=2.75 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 61.65 | 48.62 | 62.67 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-04-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 00:00:00 | 67.65 | 49.80 | 63.56 | Stage2 pullback-breakout RSI=65 vol=3.1x ATR=2.51 |
| Stop hit — per-position SL triggered | 2024-05-09 00:00:00 | 63.88 | 51.16 | 65.47 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-15 00:00:00 | 50.40 | 2024-01-18 00:00:00 | 48.21 | STOP_HIT | 1.00 | -4.35% |
| BUY | retest1 | 2024-03-05 00:00:00 | 63.60 | 2024-03-12 00:00:00 | 59.43 | STOP_HIT | 1.00 | -6.56% |
| BUY | retest1 | 2024-03-28 00:00:00 | 62.35 | 2024-04-15 00:00:00 | 61.65 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest1 | 2024-04-26 00:00:00 | 67.65 | 2024-05-09 00:00:00 | 63.88 | STOP_HIT | 1.00 | -5.57% |
