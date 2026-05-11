# Biocon Ltd. (BIOCON)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1521 bars)
- **Last close:** 378.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 13 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 12
- **Target hits / Stop hits / Partials:** 0 / 12 / 0
- **Avg / median % per leg:** -2.70% / -2.48%
- **Sum % (uncompounded):** -32.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.88% | -23.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.88% | -23.1% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.34% | -9.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.34% | -9.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.70% | -32.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 14:15:00 | 365.80 | 380.70 | 380.71 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 391.75 | 378.28 | 378.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 395.55 | 378.45 | 378.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 379.95 | 381.24 | 379.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 09:15:00 | 379.95 | 381.24 | 379.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 379.95 | 381.24 | 379.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 12:30:00 | 385.90 | 382.02 | 380.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 10:45:00 | 385.95 | 384.83 | 382.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 11:30:00 | 388.80 | 384.86 | 382.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 12:30:00 | 385.70 | 384.88 | 382.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 381.85 | 384.86 | 382.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:00:00 | 381.85 | 384.86 | 382.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 372.90 | 384.74 | 382.16 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 372.90 | 384.74 | 382.16 | SL hit (close<static) qty=1.00 sl=378.55 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 371.20 | 380.24 | 380.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 362.85 | 380.07 | 380.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 364.50 | 363.29 | 369.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-21 10:00:00 | 364.50 | 363.29 | 369.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 362.95 | 361.66 | 367.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:45:00 | 362.90 | 361.78 | 367.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:45:00 | 362.25 | 361.78 | 367.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 358.55 | 361.80 | 367.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 362.55 | 361.72 | 367.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 367.60 | 361.86 | 367.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:45:00 | 367.20 | 361.86 | 367.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 367.55 | 361.92 | 367.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:45:00 | 367.50 | 361.92 | 367.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 368.00 | 361.98 | 367.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 15:15:00 | 370.00 | 361.98 | 367.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 370.00 | 362.06 | 367.17 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-05 15:15:00 | 370.00 | 362.06 | 367.17 | SL hit (close>static) qty=1.00 sl=368.20 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-03-09 12:30:00 | 385.90 | 2026-03-16 10:15:00 | 372.90 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2026-03-13 10:45:00 | 385.95 | 2026-03-16 10:15:00 | 372.90 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2026-03-13 11:30:00 | 388.80 | 2026-03-16 10:15:00 | 372.90 | STOP_HIT | 1.00 | -4.09% |
| BUY | retest2 | 2026-03-13 12:30:00 | 385.70 | 2026-03-16 10:15:00 | 372.90 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2026-03-18 09:15:00 | 382.00 | 2026-03-19 09:15:00 | 374.50 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2026-03-20 13:15:00 | 380.70 | 2026-03-23 09:15:00 | 371.25 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2026-03-20 15:00:00 | 381.55 | 2026-03-23 09:15:00 | 371.25 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2026-03-25 11:00:00 | 381.05 | 2026-03-27 09:15:00 | 374.35 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-04-29 13:45:00 | 362.90 | 2026-05-05 15:15:00 | 370.00 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-04-29 14:45:00 | 362.25 | 2026-05-05 15:15:00 | 370.00 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2026-04-30 09:15:00 | 358.55 | 2026-05-05 15:15:00 | 370.00 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2026-05-04 13:15:00 | 362.55 | 2026-05-05 15:15:00 | 370.00 | STOP_HIT | 1.00 | -2.05% |
