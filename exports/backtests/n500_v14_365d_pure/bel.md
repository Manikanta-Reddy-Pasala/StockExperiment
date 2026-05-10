# Bharat Electronics Ltd. (BEL)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 439.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 17 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 0 / 14
- **Target hits / Stop hits / Partials:** 0 / 14 / 0
- **Avg / median % per leg:** -2.58% / -2.61%
- **Sum % (uncompounded):** -36.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.45% | -29.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.45% | -29.4% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.35% | -6.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.35% | -6.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 0 | 0.0% | 0 | 14 | 0 | -2.58% | -36.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 09:15:00 | 375.20 | 382.29 | 382.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 372.70 | 381.44 | 381.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 10:15:00 | 381.05 | 379.46 | 380.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 10:15:00 | 381.05 | 379.46 | 380.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 381.05 | 379.46 | 380.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:45:00 | 380.50 | 379.46 | 380.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 384.20 | 379.51 | 380.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 384.20 | 379.51 | 380.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 386.10 | 379.57 | 380.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:00:00 | 386.10 | 379.57 | 380.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 397.65 | 381.90 | 381.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 400.75 | 382.69 | 382.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 14:15:00 | 407.00 | 407.56 | 400.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 15:00:00 | 407.00 | 407.56 | 400.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 408.05 | 416.02 | 408.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 408.05 | 416.02 | 408.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 407.45 | 415.93 | 408.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:45:00 | 407.80 | 415.93 | 408.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 403.85 | 415.65 | 408.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 403.85 | 415.65 | 408.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 409.80 | 415.39 | 408.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 12:00:00 | 410.95 | 415.35 | 408.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 411.15 | 415.17 | 408.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 15:00:00 | 411.55 | 414.84 | 409.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 407.75 | 414.69 | 409.45 | SL hit (close<static) qty=1.00 sl=408.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 407.75 | 414.69 | 409.45 | SL hit (close<static) qty=1.00 sl=408.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 407.75 | 414.69 | 409.45 | SL hit (close<static) qty=1.00 sl=408.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 390.90 | 405.68 | 405.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 13:15:00 | 389.90 | 405.37 | 405.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 400.50 | 400.36 | 402.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:45:00 | 401.65 | 400.36 | 402.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 402.25 | 400.41 | 402.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:30:00 | 402.20 | 400.41 | 402.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 401.85 | 400.43 | 402.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:30:00 | 402.60 | 400.43 | 402.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 404.35 | 400.46 | 402.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 12:30:00 | 402.10 | 400.54 | 402.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 11:45:00 | 402.25 | 399.57 | 401.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 415.65 | 399.85 | 401.89 | SL hit (close>static) qty=1.00 sl=407.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 415.65 | 399.85 | 401.89 | SL hit (close>static) qty=1.00 sl=407.55 alert=retest2 |

### Cycle 4 — BUY (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 10:15:00 | 420.05 | 403.77 | 403.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 430.40 | 408.91 | 406.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 440.90 | 444.04 | 433.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 10:00:00 | 440.90 | 444.04 | 433.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 432.80 | 443.62 | 433.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 13:30:00 | 439.40 | 442.24 | 433.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 438.60 | 442.12 | 433.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 410.20 | 440.55 | 433.36 | SL hit (close<static) qty=1.00 sl=425.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 410.20 | 440.55 | 433.36 | SL hit (close<static) qty=1.00 sl=425.70 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 09:15:00 | 440.40 | 429.77 | 428.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 10:00:00 | 438.40 | 429.85 | 428.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 435.20 | 440.16 | 435.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:00:00 | 435.20 | 440.16 | 435.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 435.70 | 440.11 | 435.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 09:15:00 | 438.95 | 439.99 | 435.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 11:30:00 | 438.05 | 439.93 | 435.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 14:30:00 | 437.05 | 439.79 | 435.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 438.55 | 439.76 | 435.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 437.40 | 439.68 | 435.55 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 426.60 | 439.52 | 435.52 | SL hit (close<static) qty=1.00 sl=432.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 426.60 | 439.52 | 435.52 | SL hit (close<static) qty=1.00 sl=432.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 426.60 | 439.52 | 435.52 | SL hit (close<static) qty=1.00 sl=432.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 426.60 | 439.52 | 435.52 | SL hit (close<static) qty=1.00 sl=432.95 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:30:00 | 438.10 | 438.24 | 435.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:15:00 | 438.40 | 438.21 | 435.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 15:00:00 | 438.15 | 438.21 | 435.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 434.25 | 438.17 | 435.31 | SL hit (close<static) qty=1.00 sl=435.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 434.25 | 438.17 | 435.31 | SL hit (close<static) qty=1.00 sl=435.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 434.25 | 438.17 | 435.31 | SL hit (close<static) qty=1.00 sl=435.25 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 12:15:00 | 438.75 | 438.15 | 435.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-25 12:00:00 | 410.95 | 2025-12-03 09:15:00 | 407.75 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-11-26 09:15:00 | 411.15 | 2025-12-03 09:15:00 | 407.75 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-11-28 15:00:00 | 411.55 | 2025-12-03 09:15:00 | 407.75 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-12-26 12:30:00 | 402.10 | 2026-01-05 09:15:00 | 415.65 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2026-01-02 11:45:00 | 402.25 | 2026-01-05 09:15:00 | 415.65 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2026-03-17 13:30:00 | 439.40 | 2026-03-23 09:15:00 | 410.20 | STOP_HIT | 1.00 | -6.65% |
| BUY | retest2 | 2026-03-19 10:30:00 | 438.60 | 2026-03-23 09:15:00 | 410.20 | STOP_HIT | 1.00 | -6.48% |
| BUY | retest2 | 2026-04-09 09:15:00 | 440.40 | 2026-04-30 09:15:00 | 426.60 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2026-04-09 10:00:00 | 438.40 | 2026-04-30 09:15:00 | 426.60 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2026-04-28 09:15:00 | 438.95 | 2026-04-30 09:15:00 | 426.60 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2026-04-28 11:30:00 | 438.05 | 2026-04-30 09:15:00 | 426.60 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2026-04-28 14:30:00 | 437.05 | 2026-05-07 09:15:00 | 434.25 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-04-29 09:15:00 | 438.55 | 2026-05-07 09:15:00 | 434.25 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2026-05-06 09:30:00 | 438.10 | 2026-05-07 09:15:00 | 434.25 | STOP_HIT | 1.00 | -0.88% |
