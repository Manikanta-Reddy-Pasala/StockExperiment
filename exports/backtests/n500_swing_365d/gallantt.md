# Gallantt Ispat Ltd. (GALLANTT)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 869.25
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
- **Avg / median % per leg:** -6.29% / -6.23%
- **Sum % (uncompounded):** -18.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -6.29% | -18.9% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -6.29% | -18.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -6.29% | -18.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 05:30:00 | 701.20 | 493.52 | 639.90 | Stage2 pullback-breakout RSI=62 vol=7.0x ATR=37.29 |
| Stop hit — per-position SL triggered | 2025-09-22 05:30:00 | 671.20 | 511.79 | 665.76 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-11-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 05:30:00 | 656.30 | 538.21 | 581.84 | Stage2 pullback-breakout RSI=61 vol=9.5x ATR=36.62 |
| Stop hit — per-position SL triggered | 2025-11-13 05:30:00 | 601.37 | 540.80 | 592.65 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2026-02-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 05:30:00 | 597.25 | 549.73 | 549.83 | Stage2 pullback-breakout RSI=67 vol=5.2x ATR=24.82 |
| Stop hit — per-position SL triggered | 2026-02-09 05:30:00 | 560.03 | 550.42 | 556.03 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-08 05:30:00 | 701.20 | 2025-09-22 05:30:00 | 671.20 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest1 | 2025-11-10 05:30:00 | 656.30 | 2025-11-13 05:30:00 | 601.37 | STOP_HIT | 1.00 | -8.37% |
| BUY | retest1 | 2026-02-05 05:30:00 | 597.25 | 2026-02-09 05:30:00 | 560.03 | STOP_HIT | 1.00 | -6.23% |
