# Biocon Ltd. (BIOCON)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 381.40
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -4.89% / -4.99%
- **Sum % (uncompounded):** -14.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.89% | -14.7% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.89% | -14.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.89% | -14.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 00:00:00 | 368.50 | 296.59 | 350.29 | Stage2 pullback-breakout RSI=63 vol=1.8x ATR=12.25 |
| Stop hit — per-position SL triggered | 2024-08-02 00:00:00 | 350.12 | 299.63 | 353.55 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2024-09-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 00:00:00 | 379.40 | 309.50 | 355.16 | Stage2 pullback-breakout RSI=69 vol=2.9x ATR=11.25 |
| Stop hit — per-position SL triggered | 2024-09-18 00:00:00 | 362.52 | 316.13 | 369.92 | SL hit (bars_held=10) |

### Cycle 3 — BUY (started 2024-11-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 00:00:00 | 348.70 | 323.51 | 330.00 | Stage2 pullback-breakout RSI=58 vol=5.3x ATR=12.14 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 330.49 | 323.82 | 331.66 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-26 00:00:00 | 368.50 | 2024-08-02 00:00:00 | 350.12 | STOP_HIT | 1.00 | -4.99% |
| BUY | retest1 | 2024-09-04 00:00:00 | 379.40 | 2024-09-18 00:00:00 | 362.52 | STOP_HIT | 1.00 | -4.45% |
| BUY | retest1 | 2024-11-11 00:00:00 | 348.70 | 2024-11-13 00:00:00 | 330.49 | STOP_HIT | 1.00 | -5.22% |
