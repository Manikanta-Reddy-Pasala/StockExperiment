# Granules India Ltd. (GRANULES)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 752.95
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
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -1.51% / 0.00%
- **Sum % (uncompounded):** -6.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.51% | -6.0% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.51% | -6.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.51% | -6.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-09 00:00:00 | 370.10 | 316.93 | 346.23 | Stage2 pullback-breakout RSI=69 vol=3.8x ATR=10.34 |
| Stop hit — per-position SL triggered | 2023-11-23 00:00:00 | 370.10 | 321.77 | 359.85 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-12-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-18 00:00:00 | 402.30 | 331.84 | 384.20 | Stage2 pullback-breakout RSI=67 vol=2.1x ATR=10.92 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 385.92 | 332.95 | 384.82 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-01-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 00:00:00 | 411.80 | 337.26 | 391.58 | Stage2 pullback-breakout RSI=65 vol=1.9x ATR=12.48 |
| Stop hit — per-position SL triggered | 2024-01-17 00:00:00 | 420.65 | 346.11 | 408.93 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-04-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 00:00:00 | 453.55 | 383.07 | 436.85 | Stage2 pullback-breakout RSI=61 vol=2.1x ATR=12.48 |
| Stop hit — per-position SL triggered | 2024-04-09 00:00:00 | 434.83 | 385.31 | 437.71 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-09 00:00:00 | 370.10 | 2023-11-23 00:00:00 | 370.10 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest1 | 2023-12-18 00:00:00 | 402.30 | 2023-12-20 00:00:00 | 385.92 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest1 | 2024-01-01 00:00:00 | 411.80 | 2024-01-17 00:00:00 | 420.65 | STOP_HIT | 1.00 | 2.15% |
| BUY | retest1 | 2024-04-03 00:00:00 | 453.55 | 2024-04-09 00:00:00 | 434.83 | STOP_HIT | 1.00 | -4.13% |
