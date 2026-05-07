# TRENT (TRENT)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 4298.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT2_SKIP | 2 |
| ALERT3 | 2 |
| PENDING | 5 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 0 |
| ENTRY2 | 3 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -3.28% / -3.12%
- **Sum % (uncompounded):** -9.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.28% | -9.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.28% | -9.8% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.28% | -9.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 15:15:00 | 6995.00 | 6941.72 | 6941.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 12:15:00 | 7038.15 | 6943.67 | 6942.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 14:15:00 | 6932.05 | 6963.80 | 6953.09 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 14:15:00 | 6932.05 | 6963.80 | 6953.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 6932.05 | 6963.80 | 6953.09 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-31 13:15:00 | 7118.65 | 6965.21 | 6954.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 14:15:00 | 7133.80 | 6966.89 | 6955.01 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-07 09:15:00 | 7064.05 | 7020.79 | 6985.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-07 10:15:00 | 7024.35 | 7020.83 | 6985.86 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-01-07 14:15:00 | 6865.90 | 7017.80 | 6985.03 | SL hit (close<static) qty=1.00 sl=6923.05 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 5661.00 | 5388.31 | 5387.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 11:15:00 | 5796.00 | 5459.52 | 5426.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 5541.50 | 5551.36 | 5481.93 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 5541.50 | 5551.36 | 5481.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 5541.50 | 5551.36 | 5481.93 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-13 15:15:00 | 5610.00 | 5552.11 | 5484.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 5610.00 | 5552.69 | 5484.99 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2025-07-04 13:15:00 | 5444.00 | 5831.89 | 5680.35 | SL hit (close<static) qty=1.00 sl=5481.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-18 12:15:00 | 5610.00 | 5377.66 | 5439.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-18 13:15:00 | 5597.50 | 5379.85 | 5440.16 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-04 09:15:00 | 5636.00 | 5394.32 | 5429.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 10:15:00 | 5646.00 | 5396.82 | 5430.80 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-08 12:15:00 | 5470.00 | 5416.44 | 5438.44 | SL hit (close<static) qty=1.00 sl=5481.00 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 12:15:00 | 4330.60 | 3908.61 | 3907.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 13:15:00 | 4362.00 | 3913.12 | 3909.38 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-12-31 14:15:00 | 7133.80 | 2025-01-07 14:15:00 | 6865.90 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest2 | 2025-06-16 09:15:00 | 5610.00 | 2025-07-04 13:15:00 | 5444.00 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-09-04 10:15:00 | 5646.00 | 2025-09-08 12:15:00 | 5470.00 | STOP_HIT | 1.00 | -3.12% |
