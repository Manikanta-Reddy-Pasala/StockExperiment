# Exide Industries Ltd. (EXIDEIND)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 361.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 31 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 20 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 20
- **Target hits / Stop hits / Partials:** 0 / 20 / 0
- **Avg / median % per leg:** -1.58% / -1.51%
- **Sum % (uncompounded):** -31.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 0 | 0.0% | 0 | 20 | 0 | -1.58% | -31.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 0 | 0.0% | 0 | 20 | 0 | -1.58% | -31.7% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 20 | 0 | 0.0% | 0 | 20 | 0 | -1.58% | -31.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 12:15:00 | 394.50 | 372.82 | 372.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 11:15:00 | 395.85 | 382.56 | 378.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 386.55 | 386.87 | 381.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 11:00:00 | 386.55 | 386.87 | 381.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 380.10 | 386.73 | 381.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 379.00 | 386.73 | 381.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 385.20 | 386.72 | 381.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:30:00 | 386.20 | 386.70 | 381.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 10:15:00 | 385.55 | 386.59 | 381.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 12:15:00 | 380.00 | 386.23 | 381.82 | SL hit (close<static) qty=1.00 sl=380.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-18 12:15:00 | 380.00 | 386.23 | 381.82 | SL hit (close<static) qty=1.00 sl=380.05 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:30:00 | 385.65 | 384.55 | 381.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 14:45:00 | 385.25 | 384.65 | 381.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 384.20 | 385.50 | 382.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:30:00 | 384.60 | 385.50 | 382.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 381.90 | 385.35 | 382.59 | SL hit (close<static) qty=1.00 sl=382.05 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 387.40 | 385.22 | 382.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 15:15:00 | 384.50 | 385.16 | 382.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 10:00:00 | 384.45 | 385.15 | 382.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 383.40 | 385.55 | 383.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:30:00 | 383.80 | 385.55 | 383.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 382.70 | 385.52 | 383.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:30:00 | 382.50 | 385.52 | 383.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 381.35 | 385.48 | 383.22 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-11 14:15:00 | 381.35 | 385.48 | 383.22 | SL hit (close<static) qty=1.00 sl=382.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 14:15:00 | 381.35 | 385.48 | 383.22 | SL hit (close<static) qty=1.00 sl=382.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 14:15:00 | 381.35 | 385.48 | 383.22 | SL hit (close<static) qty=1.00 sl=382.05 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-07-11 14:30:00 | 381.90 | 385.48 | 383.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-11 15:15:00 | 379.75 | 385.42 | 383.20 | SL hit (close<static) qty=1.00 sl=380.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 15:15:00 | 379.75 | 385.42 | 383.20 | SL hit (close<static) qty=1.00 sl=380.05 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 383.55 | 385.17 | 383.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 386.30 | 385.17 | 383.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 09:15:00 | 380.45 | 385.17 | 383.23 | SL hit (close<static) qty=1.00 sl=382.85 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 12:30:00 | 384.35 | 385.11 | 383.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 382.00 | 385.10 | 383.26 | SL hit (close<static) qty=1.00 sl=382.85 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:15:00 | 384.05 | 385.09 | 383.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 384.30 | 385.01 | 383.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 380.80 | 384.97 | 383.26 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 380.80 | 384.97 | 383.26 | SL hit (close<static) qty=1.00 sl=382.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 380.80 | 384.97 | 383.26 | SL hit (close<static) qty=1.00 sl=382.85 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 380.80 | 384.97 | 383.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 383.55 | 384.96 | 383.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 381.90 | 384.96 | 383.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 382.50 | 384.91 | 383.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:00:00 | 382.50 | 384.91 | 383.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 384.10 | 384.91 | 383.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:00:00 | 388.20 | 384.93 | 383.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 14:15:00 | 381.30 | 386.50 | 384.41 | SL hit (close<static) qty=1.00 sl=382.30 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 09:30:00 | 386.45 | 386.23 | 384.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 10:45:00 | 387.40 | 386.79 | 384.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:15:00 | 387.40 | 386.79 | 384.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 384.65 | 386.79 | 384.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 384.65 | 386.79 | 384.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 383.75 | 386.76 | 384.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 381.50 | 386.76 | 384.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 379.65 | 386.69 | 384.80 | SL hit (close<static) qty=1.00 sl=382.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 379.65 | 386.69 | 384.80 | SL hit (close<static) qty=1.00 sl=382.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 379.65 | 386.69 | 384.80 | SL hit (close<static) qty=1.00 sl=382.30 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 384.85 | 386.31 | 384.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:30:00 | 384.70 | 386.31 | 384.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 383.45 | 386.28 | 384.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:30:00 | 381.55 | 386.28 | 384.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 387.35 | 386.29 | 384.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 14:45:00 | 389.95 | 386.35 | 384.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 10:45:00 | 389.75 | 386.47 | 384.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 11:30:00 | 389.80 | 386.54 | 384.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 13:15:00 | 379.80 | 386.53 | 384.87 | SL hit (close<static) qty=1.00 sl=383.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 13:15:00 | 379.80 | 386.53 | 384.87 | SL hit (close<static) qty=1.00 sl=383.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 13:15:00 | 379.80 | 386.53 | 384.87 | SL hit (close<static) qty=1.00 sl=383.40 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 15:15:00 | 376.00 | 383.51 | 383.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 373.85 | 383.41 | 383.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 13:15:00 | 383.50 | 382.33 | 382.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 13:15:00 | 383.50 | 382.33 | 382.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 383.50 | 382.33 | 382.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:00:00 | 383.50 | 382.33 | 382.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 393.75 | 382.44 | 382.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 393.75 | 382.44 | 382.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 396.55 | 383.60 | 383.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 11:15:00 | 397.35 | 383.74 | 383.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 12:15:00 | 408.25 | 408.50 | 399.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 13:00:00 | 408.25 | 408.50 | 399.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 399.20 | 408.00 | 399.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 399.20 | 408.00 | 399.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 398.85 | 407.91 | 399.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 398.85 | 407.91 | 399.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 400.10 | 407.76 | 399.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:45:00 | 399.90 | 407.76 | 399.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 399.60 | 407.68 | 399.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 14:30:00 | 399.00 | 407.68 | 399.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 399.30 | 407.59 | 399.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 396.60 | 407.59 | 399.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 397.15 | 406.52 | 399.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:45:00 | 396.95 | 406.52 | 399.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 398.20 | 402.46 | 398.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 398.45 | 402.46 | 398.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 399.30 | 402.43 | 398.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:45:00 | 396.95 | 402.43 | 398.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 405.00 | 402.37 | 398.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 10:30:00 | 407.70 | 402.41 | 398.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 15:15:00 | 397.00 | 402.30 | 398.88 | SL hit (close<static) qty=1.00 sl=398.10 alert=retest2 |

### Cycle 4 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 379.60 | 397.05 | 397.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 378.20 | 392.44 | 394.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 380.25 | 379.94 | 385.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 15:00:00 | 380.25 | 379.94 | 385.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 322.75 | 309.74 | 321.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:45:00 | 324.00 | 309.74 | 321.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 323.25 | 309.88 | 321.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:00:00 | 323.25 | 309.88 | 321.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 12:15:00 | 324.10 | 310.02 | 321.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:45:00 | 324.30 | 310.02 | 321.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 327.70 | 310.77 | 321.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:00:00 | 327.70 | 310.77 | 321.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 13:15:00 | 365.35 | 328.88 | 328.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 374.30 | 332.31 | 330.49 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-16 11:30:00 | 386.20 | 2025-06-18 12:15:00 | 380.00 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-06-17 10:15:00 | 385.55 | 2025-06-18 12:15:00 | 380.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-06-24 09:30:00 | 385.65 | 2025-07-03 10:15:00 | 381.90 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-06-24 14:45:00 | 385.25 | 2025-07-11 14:15:00 | 381.35 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-07-02 11:30:00 | 384.60 | 2025-07-11 14:15:00 | 381.35 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-07-04 09:15:00 | 387.40 | 2025-07-11 14:15:00 | 381.35 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-07-04 15:15:00 | 384.50 | 2025-07-11 15:15:00 | 379.75 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-07-07 10:00:00 | 384.45 | 2025-07-11 15:15:00 | 379.75 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-07-15 09:15:00 | 386.30 | 2025-07-16 09:15:00 | 380.45 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-07-16 12:30:00 | 384.35 | 2025-07-17 09:15:00 | 382.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-07-17 11:15:00 | 384.05 | 2025-07-18 09:15:00 | 380.80 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-07-18 09:15:00 | 384.30 | 2025-07-18 09:15:00 | 380.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-07-21 10:00:00 | 388.20 | 2025-07-25 14:15:00 | 381.30 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-07-29 09:30:00 | 386.45 | 2025-08-01 09:15:00 | 379.65 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-07-31 10:45:00 | 387.40 | 2025-08-01 09:15:00 | 379.65 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-07-31 11:15:00 | 387.40 | 2025-08-01 09:15:00 | 379.65 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-08-04 14:45:00 | 389.95 | 2025-08-05 13:15:00 | 379.80 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-08-05 10:45:00 | 389.75 | 2025-08-05 13:15:00 | 379.80 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-08-05 11:30:00 | 389.80 | 2025-08-05 13:15:00 | 379.80 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-10-07 10:30:00 | 407.70 | 2025-10-08 15:15:00 | 397.00 | STOP_HIT | 1.00 | -2.62% |
