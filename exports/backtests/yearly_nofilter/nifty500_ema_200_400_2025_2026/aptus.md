# Aptus Value Housing Finance India Ltd. (APTUS)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 282.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 15
- **Target hits / Stop hits / Partials:** 1 / 15 / 1
- **Avg / median % per leg:** -1.71% / -2.70%
- **Sum % (uncompounded):** -29.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 0 | 0.0% | 0 | 14 | 0 | -2.99% | -41.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 0 | 0.0% | 0 | 14 | 0 | -2.99% | -41.9% |
| SELL (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.24% | 12.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.24% | 12.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 17 | 2 | 11.8% | 1 | 15 | 1 | -1.71% | -29.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 323.85 | 334.63 | 334.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 15:15:00 | 322.00 | 334.50 | 334.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 13:15:00 | 329.95 | 328.98 | 331.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 13:15:00 | 329.95 | 328.98 | 331.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 329.95 | 328.98 | 331.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 14:00:00 | 329.95 | 328.98 | 331.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 319.65 | 317.39 | 322.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:45:00 | 320.75 | 317.39 | 322.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 281.05 | 274.04 | 281.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 15:15:00 | 280.80 | 274.04 | 281.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 280.80 | 274.10 | 281.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 276.95 | 274.10 | 281.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 11:15:00 | 283.30 | 274.34 | 281.65 | SL hit (close>static) qty=1.00 sl=283.00 alert=retest2 |

### Cycle 2 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 265.15 | 243.03 | 243.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 266.15 | 243.65 | 243.32 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-10 13:15:00 | 333.25 | 2025-07-31 13:15:00 | 323.35 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2025-07-08 13:30:00 | 335.80 | 2025-07-31 13:15:00 | 323.35 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2025-07-08 14:15:00 | 334.65 | 2025-08-28 09:15:00 | 329.95 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-07-30 09:15:00 | 334.80 | 2025-08-28 09:15:00 | 329.95 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-07-31 12:45:00 | 339.10 | 2025-08-28 09:15:00 | 329.95 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-07-31 13:15:00 | 339.05 | 2025-08-28 09:15:00 | 329.95 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-08-01 09:30:00 | 340.20 | 2025-08-28 11:15:00 | 325.35 | STOP_HIT | 1.00 | -4.37% |
| BUY | retest2 | 2025-08-01 14:45:00 | 338.95 | 2025-08-28 11:15:00 | 325.35 | STOP_HIT | 1.00 | -4.01% |
| BUY | retest2 | 2025-08-12 10:45:00 | 337.75 | 2025-09-12 13:15:00 | 332.55 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-08-12 11:45:00 | 335.95 | 2025-09-17 11:15:00 | 330.65 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-08-25 14:30:00 | 336.55 | 2025-09-23 14:15:00 | 323.85 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2025-08-26 10:00:00 | 336.95 | 2025-09-23 14:15:00 | 323.85 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2025-09-05 14:15:00 | 337.00 | 2025-09-23 14:15:00 | 323.85 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2025-09-15 09:15:00 | 336.95 | 2025-09-23 14:15:00 | 323.85 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest2 | 2026-02-04 09:15:00 | 276.95 | 2026-02-04 11:15:00 | 283.30 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-02-04 14:15:00 | 274.85 | 2026-02-06 09:15:00 | 261.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 14:15:00 | 274.85 | 2026-02-13 09:15:00 | 247.37 | TARGET_HIT | 0.50 | 10.00% |
