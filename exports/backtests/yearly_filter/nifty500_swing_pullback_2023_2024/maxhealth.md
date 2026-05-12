# Max Healthcare Institute Ltd. (MAXHEALTH)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1017.60
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** -1.12% / -5.38%
- **Sum % (uncompounded):** -5.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | -1.12% | -5.6% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | -1.12% | -5.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | -1.12% | -5.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 00:00:00 | 592.20 | 521.96 | 573.41 | Stage2 pullback-breakout RSI=57 vol=1.6x ATR=21.44 |
| Stop hit — per-position SL triggered | 2023-10-05 00:00:00 | 560.04 | 522.84 | 572.09 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2023-11-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 00:00:00 | 615.40 | 537.55 | 588.14 | Stage2 pullback-breakout RSI=66 vol=1.5x ATR=17.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-30 00:00:00 | 649.75 | 544.17 | 604.32 | T1 booked 50% @ 649.75 |
| Target hit | 2023-12-20 00:00:00 | 650.25 | 562.00 | 661.58 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-23 00:00:00 | 776.05 | 592.58 | 718.87 | Stage2 pullback-breakout RSI=67 vol=2.8x ATR=27.83 |
| Stop hit — per-position SL triggered | 2024-01-24 00:00:00 | 734.30 | 594.01 | 720.60 | SL hit (bars_held=1) |

### Cycle 4 — BUY (started 2024-04-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 00:00:00 | 838.10 | 690.67 | 802.93 | Stage2 pullback-breakout RSI=58 vol=2.3x ATR=33.85 |
| Stop hit — per-position SL triggered | 2024-05-10 00:00:00 | 787.32 | 700.18 | 808.22 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-03 00:00:00 | 592.20 | 2023-10-05 00:00:00 | 560.04 | STOP_HIT | 1.00 | -5.43% |
| BUY | retest1 | 2023-11-16 00:00:00 | 615.40 | 2023-11-30 00:00:00 | 649.75 | PARTIAL | 0.50 | 5.58% |
| BUY | retest1 | 2023-11-16 00:00:00 | 615.40 | 2023-12-20 00:00:00 | 650.25 | TARGET_HIT | 0.50 | 5.66% |
| BUY | retest1 | 2024-01-23 00:00:00 | 776.05 | 2024-01-24 00:00:00 | 734.30 | STOP_HIT | 1.00 | -5.38% |
| BUY | retest1 | 2024-04-29 00:00:00 | 838.10 | 2024-05-10 00:00:00 | 787.32 | STOP_HIT | 1.00 | -6.06% |
