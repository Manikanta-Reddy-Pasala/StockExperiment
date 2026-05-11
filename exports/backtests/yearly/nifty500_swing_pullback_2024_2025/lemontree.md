# Lemon Tree Hotels Ltd. (LEMONTREE)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 120.41
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
- **Avg / median % per leg:** -5.78% / -5.81%
- **Sum % (uncompounded):** -23.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.78% | -23.1% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.78% | -23.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.78% | -23.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 05:30:00 | 150.57 | 135.21 | 146.45 | Stage2 pullback-breakout RSI=58 vol=2.4x ATR=4.92 |
| Stop hit — per-position SL triggered | 2024-08-05 05:30:00 | 143.20 | 135.48 | 145.83 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2025-01-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 05:30:00 | 147.69 | 134.24 | 145.05 | Stage2 pullback-breakout RSI=55 vol=1.8x ATR=5.72 |
| Stop hit — per-position SL triggered | 2025-01-22 05:30:00 | 139.11 | 134.36 | 144.18 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2025-02-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 05:30:00 | 147.46 | 134.70 | 140.85 | Stage2 pullback-breakout RSI=58 vol=1.9x ATR=6.24 |
| Stop hit — per-position SL triggered | 2025-02-07 05:30:00 | 138.09 | 134.98 | 141.69 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2025-04-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 05:30:00 | 140.71 | 133.81 | 133.35 | Stage2 pullback-breakout RSI=60 vol=1.6x ATR=5.69 |
| Stop hit — per-position SL triggered | 2025-04-07 05:30:00 | 132.17 | 133.94 | 134.50 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-31 05:30:00 | 150.57 | 2024-08-05 05:30:00 | 143.20 | STOP_HIT | 1.00 | -4.90% |
| BUY | retest1 | 2025-01-20 05:30:00 | 147.69 | 2025-01-22 05:30:00 | 139.11 | STOP_HIT | 1.00 | -5.81% |
| BUY | retest1 | 2025-02-04 05:30:00 | 147.46 | 2025-02-07 05:30:00 | 138.09 | STOP_HIT | 1.00 | -6.35% |
| BUY | retest1 | 2025-04-02 05:30:00 | 140.71 | 2025-04-07 05:30:00 | 132.17 | STOP_HIT | 1.00 | -6.07% |
