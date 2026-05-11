# Hero MotoCorp Ltd. (HEROMOTOCO)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 5322.00
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
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** -0.10% / 0.76%
- **Sum % (uncompounded):** -0.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 3 | 1 | -0.10% | -0.5% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 3 | 1 | -0.10% | -0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 3 | 1 | -0.10% | -0.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 05:30:00 | 5550.50 | 4654.44 | 5335.34 | Stage2 pullback-breakout RSI=68 vol=1.5x ATR=128.42 |
| Stop hit — per-position SL triggered | 2025-10-17 05:30:00 | 5592.50 | 4740.25 | 5476.16 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-11-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 05:30:00 | 5798.50 | 4875.14 | 5509.42 | Stage2 pullback-breakout RSI=67 vol=2.9x ATR=130.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 05:30:00 | 6059.84 | 4938.23 | 5717.81 | T1 booked 50% @ 6059.84 |
| Target hit | 2025-12-09 05:30:00 | 6001.00 | 5059.22 | 6026.61 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2026-02-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 05:30:00 | 5737.50 | 5321.23 | 5605.80 | Stage2 pullback-breakout RSI=56 vol=1.8x ATR=160.41 |
| Stop hit — per-position SL triggered | 2026-03-04 05:30:00 | 5496.88 | 5333.96 | 5615.40 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2026-04-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 05:30:00 | 5466.50 | 5324.24 | 5274.43 | Stage2 pullback-breakout RSI=55 vol=1.5x ATR=184.93 |
| Stop hit — per-position SL triggered | 2026-04-16 05:30:00 | 5189.10 | 5321.49 | 5262.47 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-10-03 05:30:00 | 5550.50 | 2025-10-17 05:30:00 | 5592.50 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest1 | 2025-11-17 05:30:00 | 5798.50 | 2025-11-25 05:30:00 | 6059.84 | PARTIAL | 0.50 | 4.51% |
| BUY | retest1 | 2025-11-17 05:30:00 | 5798.50 | 2025-12-09 05:30:00 | 6001.00 | TARGET_HIT | 0.50 | 3.49% |
| BUY | retest1 | 2026-02-25 05:30:00 | 5737.50 | 2026-03-04 05:30:00 | 5496.88 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest1 | 2026-04-10 05:30:00 | 5466.50 | 2026-04-16 05:30:00 | 5189.10 | STOP_HIT | 1.00 | -5.07% |
