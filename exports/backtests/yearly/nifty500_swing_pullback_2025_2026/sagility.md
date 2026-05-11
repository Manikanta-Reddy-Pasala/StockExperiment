# Sagility Ltd. (SAGILITY)

## Backtest Summary

- **Window:** 2024-11-12 05:30:00 → 2026-05-08 05:30:00 (367 bars)
- **Last close:** 44.19
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
- **Winners / losers:** 1 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -3.66% / -4.28%
- **Sum % (uncompounded):** -10.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 3 | 0 | -3.66% | -11.0% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -3.66% | -11.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 3 | 0 | -3.66% | -11.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 05:30:00 | 53.28 | 43.65 | 49.96 | Stage2 pullback-breakout RSI=63 vol=8.7x ATR=2.39 |
| Stop hit — per-position SL triggered | 2025-11-20 05:30:00 | 49.70 | 43.92 | 50.12 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2025-12-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 05:30:00 | 52.00 | 45.05 | 50.15 | Stage2 pullback-breakout RSI=60 vol=2.1x ATR=1.68 |
| Stop hit — per-position SL triggered | 2026-01-05 05:30:00 | 52.02 | 45.75 | 51.54 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2026-01-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 05:30:00 | 53.41 | 46.18 | 51.51 | Stage2 pullback-breakout RSI=62 vol=1.9x ATR=1.52 |
| Stop hit — per-position SL triggered | 2026-01-21 05:30:00 | 51.12 | 46.36 | 51.71 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-11-14 05:30:00 | 53.28 | 2025-11-20 05:30:00 | 49.70 | STOP_HIT | 1.00 | -6.72% |
| BUY | retest1 | 2025-12-19 05:30:00 | 52.00 | 2026-01-05 05:30:00 | 52.02 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest1 | 2026-01-16 05:30:00 | 53.41 | 2026-01-21 05:30:00 | 51.12 | STOP_HIT | 1.00 | -4.28% |
