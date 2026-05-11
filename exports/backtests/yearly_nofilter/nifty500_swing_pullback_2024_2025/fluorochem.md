# Gujarat Fluorochemicals Ltd. (FLUOROCHEM)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
- **Last close:** 3769.10
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
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 4.29% / 7.10%
- **Sum % (uncompounded):** 17.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 2 | 1 | 4.29% | 17.2% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 2 | 1 | 4.29% | 17.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 2 | 1 | 4.29% | 17.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 00:00:00 | 3524.30 | 3263.75 | 3268.75 | Stage2 pullback-breakout RSI=66 vol=9.0x ATR=125.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 00:00:00 | 3774.58 | 3269.17 | 3320.19 | T1 booked 50% @ 3774.58 |
| Target hit | 2024-10-04 00:00:00 | 4065.30 | 3438.78 | 4115.60 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-10-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 00:00:00 | 4659.75 | 3472.15 | 4181.32 | Stage2 pullback-breakout RSI=69 vol=3.5x ATR=210.78 |
| Stop hit — per-position SL triggered | 2024-10-23 00:00:00 | 4343.58 | 3573.72 | 4457.04 | SL hit (bars_held=9) |

### Cycle 3 — BUY (started 2024-12-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 00:00:00 | 4261.25 | 3703.50 | 4065.05 | Stage2 pullback-breakout RSI=60 vol=2.4x ATR=154.63 |
| Stop hit — per-position SL triggered | 2024-12-19 00:00:00 | 4325.25 | 3767.98 | 4259.94 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-05 00:00:00 | 3524.30 | 2024-09-06 00:00:00 | 3774.58 | PARTIAL | 0.50 | 7.10% |
| BUY | retest1 | 2024-09-05 00:00:00 | 3524.30 | 2024-10-04 00:00:00 | 4065.30 | TARGET_HIT | 0.50 | 15.35% |
| BUY | retest1 | 2024-10-10 00:00:00 | 4659.75 | 2024-10-23 00:00:00 | 4343.58 | STOP_HIT | 1.00 | -6.79% |
| BUY | retest1 | 2024-12-05 00:00:00 | 4261.25 | 2024-12-19 00:00:00 | 4325.25 | STOP_HIT | 1.00 | 1.50% |
