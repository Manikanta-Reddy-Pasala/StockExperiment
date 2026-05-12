# Gujarat Mineral Development Corporation Ltd. (GMDCLTD)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 685.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 3 |
| ALERT3 | 77 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 50 |
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 57 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 46
- **Target hits / Stop hits / Partials:** 4 / 47 / 6
- **Avg / median % per leg:** -1.19% / -2.39%
- **Sum % (uncompounded):** -67.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 3 | 11.1% | 2 | 24 | 1 | -1.62% | -43.6% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| BUY @ 3rd Alert (retest2) | 25 | 1 | 4.0% | 1 | 24 | 0 | -2.35% | -58.6% |
| SELL (all) | 30 | 8 | 26.7% | 2 | 23 | 5 | -0.81% | -24.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 8 | 26.7% | 2 | 23 | 5 | -0.81% | -24.2% |
| retest1 (combined) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| retest2 (combined) | 55 | 9 | 16.4% | 3 | 47 | 5 | -1.51% | -82.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 09:15:00 | 398.70 | 419.41 | 419.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 14:15:00 | 394.45 | 418.25 | 418.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 09:15:00 | 385.35 | 377.36 | 392.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-02 09:45:00 | 386.15 | 377.36 | 392.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 13:15:00 | 389.30 | 377.79 | 392.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-04 10:45:00 | 387.00 | 379.05 | 392.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-04 11:30:00 | 387.00 | 379.10 | 392.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-04 15:00:00 | 387.00 | 379.40 | 392.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 09:15:00 | 385.55 | 379.51 | 392.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 388.90 | 380.16 | 392.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-08 10:45:00 | 386.95 | 380.23 | 392.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-08 12:30:00 | 386.95 | 380.36 | 392.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-09 12:30:00 | 386.00 | 380.82 | 392.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-09 14:15:00 | 393.40 | 380.97 | 392.04 | SL hit (close>static) qty=1.00 sl=392.85 alert=retest2 |

### Cycle 2 — BUY (started 2024-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 13:15:00 | 429.50 | 398.00 | 397.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-03 10:15:00 | 435.50 | 400.50 | 399.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 12:15:00 | 404.15 | 404.98 | 401.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-07 12:45:00 | 404.70 | 404.98 | 401.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 14:15:00 | 400.50 | 405.61 | 402.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-08 14:30:00 | 400.65 | 405.61 | 402.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 15:15:00 | 400.35 | 405.56 | 402.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 09:15:00 | 406.20 | 405.56 | 402.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-09 10:15:00 | 396.50 | 405.41 | 402.01 | SL hit (close<static) qty=1.00 sl=399.20 alert=retest2 |

### Cycle 3 — SELL (started 2024-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 14:15:00 | 348.35 | 401.60 | 401.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 334.85 | 400.49 | 401.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-14 10:15:00 | 394.00 | 393.03 | 396.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-14 11:15:00 | 394.20 | 393.03 | 396.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 11:15:00 | 393.75 | 393.04 | 396.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 11:30:00 | 394.25 | 393.04 | 396.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 400.60 | 393.12 | 396.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 15:00:00 | 400.60 | 393.12 | 396.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 399.00 | 393.17 | 396.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:15:00 | 397.50 | 393.17 | 396.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 397.90 | 393.25 | 396.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:30:00 | 397.20 | 393.25 | 396.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 411.70 | 393.44 | 396.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 12:00:00 | 411.70 | 393.44 | 396.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 397.30 | 394.43 | 397.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 398.05 | 394.43 | 397.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 398.10 | 394.47 | 397.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:30:00 | 399.60 | 394.47 | 397.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 406.80 | 394.59 | 397.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:00:00 | 406.80 | 394.59 | 397.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 401.85 | 395.67 | 397.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:45:00 | 401.00 | 395.67 | 397.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 396.40 | 395.82 | 397.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:15:00 | 407.15 | 395.82 | 397.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 405.75 | 395.92 | 397.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:15:00 | 408.40 | 395.92 | 397.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 402.75 | 396.42 | 397.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:15:00 | 407.55 | 396.42 | 397.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 400.40 | 396.88 | 398.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 11:30:00 | 402.75 | 396.88 | 398.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 394.30 | 396.86 | 398.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:45:00 | 397.45 | 396.86 | 398.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 399.75 | 396.89 | 398.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 399.75 | 396.89 | 398.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 397.00 | 396.89 | 398.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:45:00 | 398.85 | 396.88 | 397.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 397.50 | 396.89 | 397.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:30:00 | 398.45 | 396.89 | 397.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 395.70 | 396.76 | 397.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:30:00 | 396.40 | 396.76 | 397.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 404.90 | 396.17 | 397.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:00:00 | 404.90 | 396.17 | 397.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 400.40 | 396.21 | 397.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 14:45:00 | 398.10 | 396.35 | 397.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 15:15:00 | 396.05 | 396.35 | 397.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 09:15:00 | 424.55 | 396.97 | 397.79 | SL hit (close>static) qty=1.00 sl=404.95 alert=retest2 |

### Cycle 4 — BUY (started 2024-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 15:15:00 | 423.25 | 398.65 | 398.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 09:15:00 | 428.35 | 398.95 | 398.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 397.90 | 406.61 | 403.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 09:15:00 | 397.90 | 406.61 | 403.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 397.90 | 406.61 | 403.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 397.90 | 406.61 | 403.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 398.50 | 406.53 | 403.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:00:00 | 398.50 | 406.53 | 403.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 397.75 | 406.24 | 403.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 15:00:00 | 397.75 | 406.24 | 403.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 403.35 | 405.33 | 402.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:15:00 | 389.75 | 405.33 | 402.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 397.20 | 405.25 | 402.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 396.55 | 405.25 | 402.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 401.05 | 405.21 | 402.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:30:00 | 397.35 | 405.21 | 402.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 398.10 | 405.14 | 402.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 15:00:00 | 398.10 | 405.14 | 402.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 402.75 | 405.77 | 403.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 15:00:00 | 402.75 | 405.77 | 403.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 15:15:00 | 402.70 | 405.74 | 403.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 09:15:00 | 399.50 | 405.74 | 403.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 399.40 | 405.68 | 403.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 09:30:00 | 397.90 | 405.68 | 403.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 398.85 | 405.61 | 403.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 10:30:00 | 397.95 | 405.61 | 403.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 399.60 | 405.16 | 403.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 399.60 | 405.16 | 403.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 400.60 | 405.11 | 403.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 403.30 | 405.11 | 403.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 10:00:00 | 401.90 | 405.08 | 403.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 10:30:00 | 401.85 | 405.05 | 403.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 12:45:00 | 401.55 | 404.98 | 403.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 403.35 | 404.89 | 403.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 403.35 | 404.89 | 403.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 405.40 | 404.90 | 403.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-02 09:15:00 | 396.95 | 404.70 | 403.05 | SL hit (close<static) qty=1.00 sl=398.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 12:15:00 | 378.00 | 401.47 | 401.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 13:15:00 | 371.90 | 401.17 | 401.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 12:15:00 | 376.70 | 372.44 | 380.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-16 13:00:00 | 376.70 | 372.44 | 380.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 357.50 | 350.77 | 361.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 15:00:00 | 352.80 | 351.02 | 361.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:30:00 | 351.25 | 351.20 | 361.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:15:00 | 353.10 | 351.65 | 361.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 09:15:00 | 371.85 | 352.39 | 360.53 | SL hit (close>static) qty=1.00 sl=370.60 alert=retest2 |

### Cycle 6 — BUY (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 12:15:00 | 326.35 | 286.95 | 286.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 332.55 | 301.39 | 295.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 10:15:00 | 390.60 | 392.01 | 367.35 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 12:00:00 | 396.25 | 387.88 | 371.17 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 13:15:00 | 416.06 | 388.54 | 371.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-07-18 14:15:00 | 435.88 | 389.02 | 371.99 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 7 — SELL (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 10:15:00 | 489.10 | 537.93 | 538.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 11:15:00 | 486.25 | 537.41 | 537.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 11:15:00 | 536.40 | 531.78 | 534.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 11:15:00 | 536.40 | 531.78 | 534.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 536.40 | 531.78 | 534.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:00:00 | 536.40 | 531.78 | 534.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 534.60 | 531.81 | 534.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 14:00:00 | 533.60 | 531.83 | 534.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 526.35 | 531.90 | 534.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 10:15:00 | 506.92 | 530.81 | 534.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 14:15:00 | 500.03 | 529.69 | 533.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 09:15:00 | 540.00 | 526.96 | 531.54 | SL hit (close>ema200) qty=0.50 sl=526.96 alert=retest2 |

### Cycle 8 — BUY (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 13:15:00 | 596.05 | 535.52 | 535.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 14:15:00 | 599.60 | 536.15 | 535.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 11:15:00 | 563.90 | 565.39 | 552.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-09 11:45:00 | 563.05 | 565.39 | 552.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 551.95 | 565.08 | 552.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:30:00 | 554.20 | 565.08 | 552.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 551.50 | 564.95 | 552.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 537.90 | 564.95 | 552.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 544.40 | 564.74 | 552.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 12:30:00 | 563.55 | 564.27 | 552.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 14:30:00 | 558.00 | 565.02 | 554.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 15:15:00 | 556.10 | 565.02 | 554.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 560.20 | 563.43 | 553.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 550.55 | 563.17 | 553.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:15:00 | 552.20 | 563.17 | 553.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 544.45 | 562.99 | 553.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 544.45 | 562.99 | 553.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 541.70 | 562.78 | 553.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:30:00 | 542.40 | 562.78 | 553.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-20 13:15:00 | 533.50 | 562.49 | 553.70 | SL hit (close<static) qty=1.00 sl=534.60 alert=retest2 |

### Cycle 9 — SELL (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 15:15:00 | 550.50 | 562.16 | 562.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 13:15:00 | 545.15 | 561.67 | 561.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 10:15:00 | 563.00 | 556.42 | 559.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 10:15:00 | 563.00 | 556.42 | 559.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 563.00 | 556.42 | 559.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 563.50 | 556.42 | 559.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 561.75 | 556.48 | 559.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:45:00 | 563.15 | 556.48 | 559.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 560.80 | 556.64 | 559.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 554.15 | 556.64 | 559.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 09:15:00 | 576.40 | 555.80 | 558.61 | SL hit (close>static) qty=1.00 sl=560.80 alert=retest2 |

### Cycle 10 — BUY (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 09:15:00 | 574.85 | 561.13 | 561.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 09:15:00 | 585.15 | 561.92 | 561.47 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-04 10:45:00 | 387.00 | 2024-04-09 14:15:00 | 393.40 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-04-04 11:30:00 | 387.00 | 2024-04-09 14:15:00 | 393.40 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-04-04 15:00:00 | 387.00 | 2024-04-09 14:15:00 | 393.40 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-04-05 09:15:00 | 385.55 | 2024-04-09 14:15:00 | 393.40 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-04-08 10:45:00 | 386.95 | 2024-04-09 15:15:00 | 399.25 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2024-04-08 12:30:00 | 386.95 | 2024-04-09 15:15:00 | 399.25 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2024-04-09 12:30:00 | 386.00 | 2024-04-09 15:15:00 | 399.25 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2024-04-15 09:15:00 | 387.30 | 2024-04-16 09:15:00 | 395.65 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-04-15 14:15:00 | 388.50 | 2024-04-16 09:15:00 | 395.65 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-04-16 14:30:00 | 389.05 | 2024-04-18 10:15:00 | 397.10 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-04-18 14:30:00 | 385.55 | 2024-04-23 09:15:00 | 394.15 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2024-04-22 14:00:00 | 389.50 | 2024-04-23 09:15:00 | 394.15 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-05-09 09:15:00 | 406.20 | 2024-05-09 10:15:00 | 396.50 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2024-05-14 12:45:00 | 403.80 | 2024-05-28 09:15:00 | 390.55 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2024-06-03 09:30:00 | 403.10 | 2024-06-04 09:15:00 | 388.85 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest2 | 2024-06-03 11:30:00 | 402.55 | 2024-06-04 09:15:00 | 388.85 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2024-06-03 14:15:00 | 408.60 | 2024-06-04 09:15:00 | 388.85 | STOP_HIT | 1.00 | -4.83% |
| SELL | retest2 | 2024-07-04 14:45:00 | 398.10 | 2024-07-08 09:15:00 | 424.55 | STOP_HIT | 1.00 | -6.64% |
| SELL | retest2 | 2024-07-04 15:15:00 | 396.05 | 2024-07-08 09:15:00 | 424.55 | STOP_HIT | 1.00 | -7.20% |
| BUY | retest2 | 2024-07-31 09:15:00 | 403.30 | 2024-08-02 09:15:00 | 396.95 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-07-31 10:00:00 | 401.90 | 2024-08-02 09:15:00 | 396.95 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-07-31 10:30:00 | 401.85 | 2024-08-02 09:15:00 | 396.95 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-07-31 12:45:00 | 401.55 | 2024-08-02 09:15:00 | 396.95 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-10-23 15:00:00 | 352.80 | 2024-10-30 09:15:00 | 371.85 | STOP_HIT | 1.00 | -5.40% |
| SELL | retest2 | 2024-10-24 09:30:00 | 351.25 | 2024-10-30 09:15:00 | 371.85 | STOP_HIT | 1.00 | -5.86% |
| SELL | retest2 | 2024-10-25 09:15:00 | 353.10 | 2024-10-30 09:15:00 | 371.85 | STOP_HIT | 1.00 | -5.31% |
| SELL | retest2 | 2024-11-11 09:15:00 | 351.50 | 2024-11-13 12:15:00 | 333.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 09:15:00 | 351.50 | 2024-11-27 09:15:00 | 347.00 | STOP_HIT | 0.50 | 1.28% |
| SELL | retest2 | 2024-11-28 15:00:00 | 345.05 | 2024-12-03 09:15:00 | 355.10 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2024-12-18 09:45:00 | 348.95 | 2024-12-20 13:15:00 | 331.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 12:15:00 | 346.30 | 2024-12-20 13:15:00 | 328.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 09:45:00 | 348.95 | 2024-12-30 14:15:00 | 314.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-18 12:15:00 | 346.30 | 2025-01-06 13:15:00 | 311.67 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-07-18 12:00:00 | 396.25 | 2025-07-18 13:15:00 | 416.06 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-07-18 12:00:00 | 396.25 | 2025-07-18 14:15:00 | 435.88 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-09-01 09:15:00 | 410.80 | 2025-09-03 09:15:00 | 451.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-15 14:00:00 | 533.60 | 2025-12-17 10:15:00 | 506.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-16 09:15:00 | 526.35 | 2025-12-17 14:15:00 | 500.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-15 14:00:00 | 533.60 | 2025-12-23 09:15:00 | 540.00 | STOP_HIT | 0.50 | -1.20% |
| SELL | retest2 | 2025-12-16 09:15:00 | 526.35 | 2025-12-23 09:15:00 | 540.00 | STOP_HIT | 0.50 | -2.59% |
| SELL | retest2 | 2025-12-23 11:30:00 | 532.40 | 2025-12-24 09:15:00 | 548.50 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2026-01-12 12:30:00 | 563.55 | 2026-01-20 13:15:00 | 533.50 | STOP_HIT | 1.00 | -5.33% |
| BUY | retest2 | 2026-01-16 14:30:00 | 558.00 | 2026-01-20 13:15:00 | 533.50 | STOP_HIT | 1.00 | -4.39% |
| BUY | retest2 | 2026-01-16 15:15:00 | 556.10 | 2026-01-20 13:15:00 | 533.50 | STOP_HIT | 1.00 | -4.06% |
| BUY | retest2 | 2026-01-20 09:15:00 | 560.20 | 2026-01-20 13:15:00 | 533.50 | STOP_HIT | 1.00 | -4.77% |
| BUY | retest2 | 2026-02-18 13:30:00 | 575.50 | 2026-02-24 12:15:00 | 555.65 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2026-02-19 11:15:00 | 572.90 | 2026-02-24 12:15:00 | 555.65 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2026-02-19 12:15:00 | 569.35 | 2026-02-24 12:15:00 | 555.65 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2026-02-19 13:00:00 | 570.10 | 2026-02-24 12:15:00 | 555.65 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2026-02-20 09:15:00 | 563.35 | 2026-03-02 10:15:00 | 559.05 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-02-23 13:15:00 | 564.65 | 2026-03-02 10:15:00 | 559.05 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2026-02-23 15:00:00 | 564.40 | 2026-03-02 10:15:00 | 559.05 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2026-02-24 15:15:00 | 563.70 | 2026-03-02 12:15:00 | 548.55 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2026-02-26 09:30:00 | 571.35 | 2026-03-02 12:15:00 | 548.55 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2026-02-27 12:45:00 | 567.45 | 2026-03-02 12:15:00 | 548.55 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2026-03-02 09:45:00 | 567.65 | 2026-03-02 12:15:00 | 548.55 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2026-03-19 09:15:00 | 554.15 | 2026-03-20 09:15:00 | 576.40 | STOP_HIT | 1.00 | -4.02% |
