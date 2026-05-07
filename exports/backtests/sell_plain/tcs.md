# TCS (TCS)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2025-11-10 09:15:00 → 2026-05-07 15:15:00 (847 bars)
- **Last close:** 2403.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 4 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 0 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 0 / 0
- **Target hits / Stop hits / Partials:** 0 / 0 / 0
- **Avg / median % per leg:** 0.00% / 0.00%
- **Sum % (uncompounded):** 0.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 2917.90 | 3157.71 | 3158.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 14:15:00 | 2910.00 | 3112.73 | 3134.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 11:15:00 | 2536.20 | 2525.98 | 2687.57 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 2511.00 | 2531.89 | 2675.99 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 11:15:00 | 2511.00 | 2531.44 | 2674.33 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 2480.60 | 2530.50 | 2670.32 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 11:15:00 | 2482.00 | 2529.58 | 2668.47 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-22 11:15:00 | 2517.60 | 2542.44 | 2649.02 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-22 12:15:00 | 2523.30 | 2542.25 | 2648.39 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-23 15:15:00 | 2518.20 | 2541.71 | 2642.95 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:15:00 | 2460.20 | 2540.90 | 2642.04 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 1080m) |

