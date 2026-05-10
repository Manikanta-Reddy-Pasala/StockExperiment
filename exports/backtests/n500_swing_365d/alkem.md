# Alkem Laboratories Ltd. (ALKEM)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 5585.50
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 2
- **Avg / median % per leg:** -0.01% / 0.00%
- **Sum % (uncompounded):** -0.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 0 | 4 | 2 | -0.01% | -0.1% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 4 | 2 | -0.01% | -0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 4 | 2 | -0.01% | -0.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 05:30:00 | 5664.00 | 5293.11 | 5528.34 | Stage2 pullback-breakout RSI=65 vol=2.0x ATR=98.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 05:30:00 | 5861.03 | 5316.66 | 5607.09 | T1 booked 50% @ 5861.03 |
| Stop hit — per-position SL triggered | 2025-11-14 05:30:00 | 5664.00 | 5320.70 | 5618.08 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2026-01-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 05:30:00 | 5655.50 | 5410.39 | 5577.34 | Stage2 pullback-breakout RSI=55 vol=1.5x ATR=105.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 05:30:00 | 5865.53 | 5426.32 | 5657.00 | T1 booked 50% @ 5865.53 |
| Stop hit — per-position SL triggered | 2026-01-21 05:30:00 | 5655.50 | 5446.61 | 5705.18 | SL hit (bars_held=10) |

### Cycle 3 — BUY (started 2026-02-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 05:30:00 | 5750.50 | 5475.66 | 5686.97 | Stage2 pullback-breakout RSI=55 vol=1.8x ATR=133.28 |
| Stop hit — per-position SL triggered | 2026-02-13 05:30:00 | 5550.58 | 5486.11 | 5700.30 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2026-02-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 05:30:00 | 5625.00 | 5485.14 | 5579.98 | Stage2 pullback-breakout RSI=51 vol=1.6x ATR=141.53 |
| Stop hit — per-position SL triggered | 2026-03-12 05:30:00 | 5412.70 | 5491.81 | 5556.05 | SL hit (bars_held=10) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-11-04 05:30:00 | 5664.00 | 2025-11-13 05:30:00 | 5861.03 | PARTIAL | 0.50 | 3.48% |
| BUY | retest1 | 2025-11-04 05:30:00 | 5664.00 | 2025-11-14 05:30:00 | 5664.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-06 05:30:00 | 5655.50 | 2026-01-12 05:30:00 | 5865.53 | PARTIAL | 0.50 | 3.71% |
| BUY | retest1 | 2026-01-06 05:30:00 | 5655.50 | 2026-01-21 05:30:00 | 5655.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-09 05:30:00 | 5750.50 | 2026-02-13 05:30:00 | 5550.58 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest1 | 2026-02-25 05:30:00 | 5625.00 | 2026-03-12 05:30:00 | 5412.70 | STOP_HIT | 1.00 | -3.77% |
