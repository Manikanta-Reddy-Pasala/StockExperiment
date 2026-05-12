# COALINDIA (COALINDIA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 456.55
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 3 |
| ALERT3 | 34 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 35 |
| PARTIAL | 1 |
| TARGET_HIT | 5 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 29
- **Target hits / Stop hits / Partials:** 5 / 30 / 1
- **Avg / median % per leg:** 0.63% / -0.52%
- **Sum % (uncompounded):** 22.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 5 | 20.8% | 5 | 19 | 0 | 1.22% | 29.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 5 | 20.8% | 5 | 19 | 0 | 1.22% | 29.2% |
| SELL (all) | 12 | 2 | 16.7% | 0 | 11 | 1 | -0.54% | -6.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 2 | 16.7% | 0 | 11 | 1 | -0.54% | -6.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 36 | 7 | 19.4% | 5 | 30 | 1 | 0.63% | 22.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 11:15:00 | 387.05 | 393.34 | 393.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 10:15:00 | 386.00 | 392.94 | 393.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 388.80 | 388.77 | 390.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-18 15:00:00 | 388.80 | 388.77 | 390.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 389.85 | 388.60 | 390.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:45:00 | 391.10 | 388.60 | 390.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 389.70 | 388.61 | 390.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:15:00 | 389.20 | 388.61 | 390.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 390.80 | 388.64 | 390.38 | SL hit (close>static) qty=1.00 sl=390.60 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 15:15:00 | 394.95 | 385.98 | 385.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 396.40 | 386.08 | 385.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 389.40 | 389.80 | 388.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 14:00:00 | 389.40 | 389.80 | 388.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 389.10 | 389.79 | 388.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 389.85 | 389.79 | 388.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 12:15:00 | 386.30 | 389.78 | 388.22 | SL hit (close<static) qty=1.00 sl=388.05 alert=retest2 |

### Cycle 3 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 382.50 | 387.25 | 387.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 381.40 | 387.14 | 387.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 12:15:00 | 387.40 | 386.75 | 386.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 12:15:00 | 387.40 | 386.75 | 386.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 387.40 | 386.75 | 386.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:45:00 | 387.10 | 386.75 | 386.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 387.05 | 386.75 | 386.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 387.05 | 386.75 | 386.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 387.75 | 386.76 | 386.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:30:00 | 387.75 | 386.76 | 386.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 388.05 | 386.78 | 387.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 388.05 | 386.78 | 387.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 388.45 | 386.80 | 387.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:00:00 | 388.45 | 386.80 | 387.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 388.60 | 386.80 | 387.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 388.60 | 386.80 | 387.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 388.70 | 386.82 | 387.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:15:00 | 390.05 | 386.82 | 387.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 392.15 | 387.20 | 387.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 393.70 | 387.32 | 387.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:45:00 | 384.80 | 389.11 | 388.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 381.85 | 389.04 | 388.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:45:00 | 381.15 | 389.04 | 388.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 388.00 | 388.80 | 388.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 390.05 | 388.80 | 388.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 13:15:00 | 388.30 | 388.91 | 388.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 15:00:00 | 388.85 | 388.90 | 388.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 379.30 | 388.81 | 388.16 | SL hit (close<static) qty=1.00 sl=387.45 alert=retest2 |

### Cycle 5 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 374.35 | 387.52 | 387.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 372.95 | 387.38 | 387.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 12:15:00 | 386.80 | 385.73 | 386.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 12:15:00 | 386.80 | 385.73 | 386.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 386.80 | 385.73 | 386.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:00:00 | 386.80 | 385.73 | 386.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 386.35 | 385.73 | 386.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:30:00 | 387.20 | 385.73 | 386.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 386.50 | 385.74 | 386.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 386.50 | 385.74 | 386.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 386.20 | 385.74 | 386.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 385.45 | 385.74 | 386.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 385.65 | 385.74 | 386.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 14:00:00 | 383.85 | 385.74 | 386.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:00:00 | 383.90 | 385.67 | 386.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 388.20 | 385.70 | 386.48 | SL hit (close>static) qty=1.00 sl=387.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 12:15:00 | 402.70 | 384.05 | 384.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 405.35 | 384.80 | 384.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 419.30 | 419.57 | 408.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:15:00 | 418.55 | 419.57 | 408.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 412.45 | 422.91 | 413.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 09:45:00 | 418.90 | 422.11 | 413.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 14:45:00 | 418.25 | 421.66 | 413.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 10:45:00 | 417.90 | 421.52 | 413.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 14:45:00 | 417.85 | 421.33 | 413.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 424.80 | 423.96 | 416.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 431.20 | 423.97 | 416.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-12 10:15:00 | 460.79 | 431.12 | 422.24 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-04 15:00:00 | 394.70 | 2025-06-05 13:15:00 | 393.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-06-05 09:15:00 | 395.95 | 2025-06-05 13:15:00 | 393.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-06-05 11:15:00 | 394.80 | 2025-06-05 13:15:00 | 393.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-06-05 15:00:00 | 395.00 | 2025-06-12 13:15:00 | 393.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-06-27 09:15:00 | 396.15 | 2025-06-30 10:15:00 | 392.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-06-27 11:00:00 | 395.95 | 2025-06-30 10:15:00 | 392.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-06-27 11:30:00 | 395.55 | 2025-06-30 10:15:00 | 392.50 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-06-27 12:00:00 | 395.70 | 2025-06-30 10:15:00 | 392.50 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-06-27 15:00:00 | 394.55 | 2025-06-30 10:15:00 | 392.50 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-06-30 10:00:00 | 395.25 | 2025-06-30 10:15:00 | 392.50 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-07-22 14:15:00 | 389.20 | 2025-07-23 09:15:00 | 390.80 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-07-23 13:30:00 | 389.10 | 2025-07-23 14:15:00 | 390.80 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-07-24 09:15:00 | 389.50 | 2025-08-28 09:15:00 | 370.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 09:15:00 | 389.50 | 2025-09-02 09:15:00 | 383.75 | STOP_HIT | 0.50 | 1.48% |
| SELL | retest2 | 2025-09-03 14:00:00 | 389.25 | 2025-09-04 12:15:00 | 391.10 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-09-29 09:15:00 | 389.85 | 2025-09-29 12:15:00 | 386.30 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-09-29 14:00:00 | 389.70 | 2025-09-29 14:15:00 | 388.00 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-09-29 15:15:00 | 389.40 | 2025-09-30 13:15:00 | 387.80 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-09-30 09:45:00 | 390.00 | 2025-09-30 13:15:00 | 387.80 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-10-01 09:30:00 | 392.15 | 2025-10-03 09:15:00 | 383.00 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-10-01 14:30:00 | 390.50 | 2025-10-03 09:15:00 | 383.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-10-31 09:15:00 | 390.05 | 2025-11-04 09:15:00 | 379.30 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-11-03 13:15:00 | 388.30 | 2025-11-04 09:15:00 | 379.30 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-11-03 15:00:00 | 388.85 | 2025-11-04 09:15:00 | 379.30 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-11-13 14:00:00 | 383.85 | 2025-11-17 09:15:00 | 388.20 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-11-14 10:00:00 | 383.90 | 2025-11-17 09:15:00 | 388.20 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-11-18 15:00:00 | 384.10 | 2025-12-15 11:15:00 | 384.35 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-12-11 13:00:00 | 384.30 | 2025-12-15 11:15:00 | 384.35 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-12-15 09:15:00 | 380.65 | 2025-12-17 10:15:00 | 384.85 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-12-15 10:30:00 | 382.00 | 2025-12-23 09:15:00 | 397.40 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2025-12-16 10:15:00 | 381.40 | 2025-12-23 09:15:00 | 397.40 | STOP_HIT | 1.00 | -4.20% |
| BUY | retest2 | 2026-02-16 09:45:00 | 418.90 | 2026-03-12 10:15:00 | 460.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-18 14:45:00 | 418.25 | 2026-03-12 10:15:00 | 460.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-19 10:45:00 | 417.90 | 2026-03-12 10:15:00 | 459.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-19 14:45:00 | 417.85 | 2026-03-12 10:15:00 | 459.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-04 09:15:00 | 431.20 | 2026-03-13 09:15:00 | 474.32 | TARGET_HIT | 1.00 | 10.00% |
