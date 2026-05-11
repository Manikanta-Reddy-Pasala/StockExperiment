# Chambal Fertilizers & Chemicals Ltd. (CHAMBLFERT)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 455.80
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
- **Avg / median % per leg:** -5.13% / -5.02%
- **Sum % (uncompounded):** -20.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.13% | -20.5% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.13% | -20.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.13% | -20.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 05:30:00 | 525.25 | 433.86 | 508.03 | Stage2 pullback-breakout RSI=57 vol=2.0x ATR=18.07 |
| Stop hit — per-position SL triggered | 2024-10-04 05:30:00 | 498.15 | 437.49 | 513.68 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2024-11-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 05:30:00 | 503.40 | 447.09 | 484.89 | Stage2 pullback-breakout RSI=56 vol=1.5x ATR=16.83 |
| Stop hit — per-position SL triggered | 2024-11-11 05:30:00 | 478.15 | 448.44 | 486.61 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2025-01-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 05:30:00 | 509.30 | 469.10 | 496.36 | Stage2 pullback-breakout RSI=55 vol=1.7x ATR=15.54 |
| Stop hit — per-position SL triggered | 2025-01-27 05:30:00 | 485.99 | 469.46 | 494.64 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2025-02-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-27 05:30:00 | 574.75 | 481.59 | 536.21 | Stage2 pullback-breakout RSI=64 vol=1.8x ATR=22.12 |
| Stop hit — per-position SL triggered | 2025-02-28 05:30:00 | 541.58 | 482.11 | 535.97 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-27 05:30:00 | 525.25 | 2024-10-04 05:30:00 | 498.15 | STOP_HIT | 1.00 | -5.16% |
| BUY | retest1 | 2024-11-06 05:30:00 | 503.40 | 2024-11-11 05:30:00 | 478.15 | STOP_HIT | 1.00 | -5.02% |
| BUY | retest1 | 2025-01-23 05:30:00 | 509.30 | 2025-01-27 05:30:00 | 485.99 | STOP_HIT | 1.00 | -4.58% |
| BUY | retest1 | 2025-02-27 05:30:00 | 574.75 | 2025-02-28 05:30:00 | 541.58 | STOP_HIT | 1.00 | -5.77% |
