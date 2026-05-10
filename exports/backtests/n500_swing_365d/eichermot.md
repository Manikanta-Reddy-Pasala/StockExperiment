# Eicher Motors Ltd. (EICHERMOT)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 7302.50
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
- **Avg / median % per leg:** 1.61% / 0.00%
- **Sum % (uncompounded):** 4.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 1.61% | 4.8% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 1.61% | 4.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 2 | 1 | 1.61% | 4.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 05:30:00 | 7125.50 | 6017.69 | 6895.88 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=142.86 |
| Stop hit — per-position SL triggered | 2025-12-04 05:30:00 | 7100.00 | 6123.26 | 7036.21 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2026-02-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 05:30:00 | 7771.00 | 6545.80 | 7246.64 | Stage2 pullback-breakout RSI=69 vol=4.4x ATR=201.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 05:30:00 | 8173.24 | 6697.94 | 7758.54 | T1 booked 50% @ 8173.24 |
| Stop hit — per-position SL triggered | 2026-03-02 05:30:00 | 7771.00 | 6722.09 | 7786.68 | SL hit (bars_held=13) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-11-20 05:30:00 | 7125.50 | 2025-12-04 05:30:00 | 7100.00 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-11 05:30:00 | 7771.00 | 2026-02-26 05:30:00 | 8173.24 | PARTIAL | 0.50 | 5.18% |
| BUY | retest1 | 2026-02-11 05:30:00 | 7771.00 | 2026-03-02 05:30:00 | 7771.00 | STOP_HIT | 0.50 | 0.00% |
