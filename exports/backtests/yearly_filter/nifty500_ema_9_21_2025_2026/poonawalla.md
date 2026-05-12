# Poonawalla Fincorp Ltd. (POONAWALLA)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 461.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 73 |
| ALERT1 | 55 |
| ALERT2 | 53 |
| ALERT2_SKIP | 28 |
| ALERT3 | 158 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 75 |
| PARTIAL | 11 |
| TARGET_HIT | 7 |
| STOP_HIT | 70 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 88 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 45
- **Target hits / Stop hits / Partials:** 7 / 70 / 11
- **Avg / median % per leg:** 1.17% / -0.42%
- **Sum % (uncompounded):** 103.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 18 | 51.4% | 2 | 33 | 0 | 0.42% | 14.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 35 | 18 | 51.4% | 2 | 33 | 0 | 0.42% | 14.8% |
| SELL (all) | 53 | 25 | 47.2% | 5 | 37 | 11 | 1.67% | 88.4% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.29% | -2.6% |
| SELL @ 3rd Alert (retest2) | 51 | 25 | 49.0% | 5 | 35 | 11 | 1.78% | 91.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.29% | -2.6% |
| retest2 (combined) | 86 | 43 | 50.0% | 7 | 68 | 11 | 1.23% | 105.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 388.90 | 377.52 | 377.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 10:15:00 | 392.50 | 385.85 | 383.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 13:15:00 | 390.60 | 390.71 | 388.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 14:00:00 | 390.60 | 390.71 | 388.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 390.25 | 390.42 | 388.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:30:00 | 390.80 | 390.42 | 388.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 390.20 | 394.04 | 392.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 389.50 | 394.04 | 392.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 394.50 | 394.13 | 392.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 11:15:00 | 394.80 | 394.13 | 392.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 13:00:00 | 394.70 | 394.33 | 393.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:30:00 | 394.55 | 393.70 | 393.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 11:00:00 | 394.80 | 393.70 | 393.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 393.80 | 393.72 | 393.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:30:00 | 394.35 | 393.72 | 393.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 396.20 | 394.21 | 393.55 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-22 12:15:00 | 390.40 | 393.47 | 393.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 390.40 | 393.47 | 393.67 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 15:15:00 | 395.30 | 394.03 | 393.88 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 10:15:00 | 393.40 | 393.71 | 393.75 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 396.65 | 394.30 | 394.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 397.10 | 394.86 | 394.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 14:15:00 | 396.85 | 397.81 | 396.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 14:15:00 | 396.85 | 397.81 | 396.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 396.85 | 397.81 | 396.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:45:00 | 397.75 | 397.81 | 396.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 395.10 | 397.27 | 396.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 395.30 | 397.27 | 396.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 394.15 | 396.64 | 396.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 394.15 | 396.64 | 396.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 397.80 | 396.87 | 396.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 399.60 | 396.87 | 396.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 12:00:00 | 399.60 | 397.42 | 396.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 11:00:00 | 398.45 | 398.81 | 397.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 12:45:00 | 398.15 | 398.20 | 397.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 399.60 | 398.48 | 397.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 14:30:00 | 401.00 | 398.98 | 398.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:45:00 | 400.15 | 402.21 | 401.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:30:00 | 400.30 | 401.17 | 400.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 15:15:00 | 403.25 | 404.53 | 404.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 403.25 | 404.53 | 404.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 398.80 | 403.39 | 404.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 403.35 | 403.19 | 403.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 11:15:00 | 403.35 | 403.19 | 403.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 403.35 | 403.19 | 403.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:00:00 | 403.35 | 403.19 | 403.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 400.00 | 402.55 | 403.47 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 14:15:00 | 404.80 | 403.12 | 403.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 15:15:00 | 406.40 | 403.77 | 403.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 14:15:00 | 427.75 | 429.91 | 425.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 14:15:00 | 427.75 | 429.91 | 425.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 427.75 | 429.91 | 425.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 427.75 | 429.91 | 425.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 428.45 | 429.08 | 425.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 426.00 | 429.08 | 425.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 423.25 | 427.70 | 426.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 423.40 | 427.70 | 426.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 426.35 | 427.43 | 426.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 15:15:00 | 426.95 | 427.43 | 426.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 11:00:00 | 428.00 | 427.06 | 426.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 12:00:00 | 428.25 | 427.30 | 426.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 421.25 | 426.16 | 426.03 | SL hit (close<static) qty=1.00 sl=422.95 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 422.65 | 425.46 | 425.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 416.65 | 423.07 | 424.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 417.70 | 417.52 | 420.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 417.70 | 417.52 | 420.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 419.75 | 418.16 | 419.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 419.10 | 418.16 | 419.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 421.75 | 418.88 | 419.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 420.25 | 418.88 | 419.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 421.50 | 419.40 | 420.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:00:00 | 421.50 | 419.40 | 420.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 419.90 | 419.50 | 420.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 15:15:00 | 419.00 | 420.15 | 420.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 09:15:00 | 422.50 | 420.44 | 420.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 09:15:00 | 422.50 | 420.44 | 420.31 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 13:15:00 | 418.85 | 420.06 | 420.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 15:15:00 | 418.55 | 419.70 | 420.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 421.30 | 420.02 | 420.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 421.30 | 420.02 | 420.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 421.30 | 420.02 | 420.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 421.90 | 420.02 | 420.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 419.50 | 419.92 | 420.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:15:00 | 417.40 | 419.92 | 420.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 11:45:00 | 416.90 | 416.40 | 417.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:30:00 | 417.60 | 416.13 | 416.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 424.15 | 417.74 | 417.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 424.15 | 417.74 | 417.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 430.00 | 420.19 | 418.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 09:15:00 | 443.50 | 443.50 | 436.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 10:00:00 | 443.50 | 443.50 | 436.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 435.70 | 441.50 | 436.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:00:00 | 435.70 | 441.50 | 436.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 434.00 | 440.00 | 436.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:45:00 | 434.75 | 440.00 | 436.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 442.80 | 439.51 | 437.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:00:00 | 444.70 | 440.54 | 438.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 15:15:00 | 467.00 | 468.29 | 468.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 15:15:00 | 467.00 | 468.29 | 468.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 466.85 | 467.89 | 468.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 11:15:00 | 469.60 | 468.23 | 468.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 11:15:00 | 469.60 | 468.23 | 468.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 469.60 | 468.23 | 468.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 469.60 | 468.23 | 468.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 469.20 | 468.43 | 468.33 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 466.70 | 468.53 | 468.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 461.00 | 466.39 | 467.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 14:15:00 | 454.70 | 453.20 | 457.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 15:00:00 | 454.70 | 453.20 | 457.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 453.50 | 453.37 | 456.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 11:30:00 | 450.45 | 452.89 | 454.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 09:30:00 | 449.65 | 451.82 | 453.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 450.55 | 450.08 | 451.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 13:15:00 | 455.40 | 452.57 | 452.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 13:15:00 | 455.40 | 452.57 | 452.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 09:15:00 | 463.60 | 454.57 | 453.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 455.80 | 458.06 | 456.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 10:15:00 | 455.80 | 458.06 | 456.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 455.80 | 458.06 | 456.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 455.80 | 458.06 | 456.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 456.95 | 457.84 | 456.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:30:00 | 457.15 | 457.84 | 456.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 456.25 | 457.52 | 456.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:00:00 | 456.25 | 457.52 | 456.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 455.40 | 457.09 | 456.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:45:00 | 455.90 | 457.09 | 456.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 452.95 | 456.27 | 455.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 452.95 | 456.27 | 455.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 15:15:00 | 453.45 | 455.70 | 455.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 449.00 | 454.36 | 455.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 439.20 | 422.38 | 429.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 439.20 | 422.38 | 429.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 439.20 | 422.38 | 429.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 439.20 | 422.38 | 429.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 444.50 | 426.81 | 430.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:00:00 | 444.50 | 426.81 | 430.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 419.65 | 416.27 | 420.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:30:00 | 423.95 | 416.27 | 420.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 420.10 | 417.04 | 420.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:45:00 | 420.55 | 417.04 | 420.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 420.00 | 417.63 | 420.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:45:00 | 420.90 | 417.63 | 420.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 420.00 | 417.20 | 419.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:30:00 | 420.80 | 417.20 | 419.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 422.20 | 418.20 | 419.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:00:00 | 422.20 | 418.20 | 419.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 425.55 | 420.82 | 420.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 13:15:00 | 426.70 | 422.00 | 421.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 421.00 | 421.80 | 421.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 14:15:00 | 421.00 | 421.80 | 421.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 421.00 | 421.80 | 421.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 421.00 | 421.80 | 421.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 426.45 | 422.73 | 421.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 419.20 | 422.73 | 421.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 418.30 | 421.84 | 421.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:45:00 | 416.60 | 421.84 | 421.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 419.10 | 421.29 | 421.04 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 415.75 | 420.19 | 420.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 414.95 | 419.14 | 420.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 15:15:00 | 418.50 | 417.93 | 419.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 15:15:00 | 418.50 | 417.93 | 419.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 418.50 | 417.93 | 419.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 09:30:00 | 416.00 | 418.42 | 419.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 10:15:00 | 424.25 | 419.58 | 419.71 | SL hit (close>static) qty=1.00 sl=420.50 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 11:15:00 | 426.00 | 420.87 | 420.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 12:15:00 | 428.65 | 422.42 | 421.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 12:15:00 | 440.80 | 441.33 | 437.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 13:00:00 | 440.80 | 441.33 | 437.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 441.50 | 441.37 | 437.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:45:00 | 438.10 | 441.37 | 437.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 439.65 | 441.99 | 439.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 439.65 | 441.99 | 439.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 437.40 | 441.07 | 439.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:00:00 | 437.40 | 441.07 | 439.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 437.20 | 440.30 | 439.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:45:00 | 437.70 | 440.30 | 439.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 434.00 | 438.06 | 438.19 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 440.30 | 438.29 | 438.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 442.75 | 439.60 | 438.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 12:15:00 | 453.85 | 453.89 | 450.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 13:00:00 | 453.85 | 453.89 | 450.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 466.70 | 467.64 | 464.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 465.65 | 467.64 | 464.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 465.65 | 467.07 | 464.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:30:00 | 464.75 | 467.07 | 464.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 464.45 | 466.54 | 464.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:45:00 | 464.40 | 466.54 | 464.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 463.80 | 466.00 | 464.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:30:00 | 465.00 | 464.68 | 464.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 10:15:00 | 459.10 | 463.56 | 463.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 10:15:00 | 459.10 | 463.56 | 463.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 11:15:00 | 455.00 | 461.85 | 462.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 457.40 | 455.97 | 459.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 457.40 | 455.97 | 459.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 457.40 | 455.97 | 459.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:45:00 | 458.75 | 455.97 | 459.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 459.75 | 456.73 | 459.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 459.75 | 456.73 | 459.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 463.15 | 458.01 | 459.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:00:00 | 463.15 | 458.01 | 459.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 465.20 | 459.45 | 460.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:00:00 | 465.20 | 459.45 | 460.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 14:15:00 | 465.40 | 461.41 | 460.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 09:15:00 | 468.05 | 463.19 | 461.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 464.90 | 466.32 | 464.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 464.90 | 466.32 | 464.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 464.90 | 466.32 | 464.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 465.30 | 466.32 | 464.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 463.50 | 465.75 | 464.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:30:00 | 464.20 | 465.75 | 464.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 463.40 | 465.28 | 464.28 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 458.55 | 463.44 | 463.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 453.50 | 461.45 | 462.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 13:15:00 | 431.00 | 430.79 | 436.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 14:15:00 | 430.80 | 430.79 | 436.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 434.00 | 429.02 | 431.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:00:00 | 434.00 | 429.02 | 431.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 436.05 | 430.43 | 432.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:30:00 | 435.35 | 430.43 | 432.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 12:15:00 | 439.75 | 433.34 | 433.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 13:15:00 | 441.90 | 435.05 | 434.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 437.85 | 438.19 | 436.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 11:00:00 | 437.85 | 438.19 | 436.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 439.40 | 438.30 | 436.58 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 434.35 | 435.70 | 435.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 11:15:00 | 430.60 | 432.36 | 433.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 10:15:00 | 432.70 | 430.67 | 432.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 10:15:00 | 432.70 | 430.67 | 432.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 432.70 | 430.67 | 432.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 432.70 | 430.67 | 432.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 433.10 | 431.16 | 432.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:30:00 | 434.40 | 431.16 | 432.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 432.15 | 431.36 | 432.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:45:00 | 432.70 | 431.36 | 432.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 431.45 | 431.37 | 432.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 14:15:00 | 429.30 | 431.37 | 432.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 15:15:00 | 430.00 | 431.54 | 432.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 442.85 | 433.56 | 432.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 442.85 | 433.56 | 432.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 446.10 | 436.06 | 434.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 440.80 | 442.95 | 440.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 12:15:00 | 440.80 | 442.95 | 440.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 440.80 | 442.95 | 440.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:00:00 | 440.80 | 442.95 | 440.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 439.75 | 442.31 | 439.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 439.75 | 442.31 | 439.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 438.40 | 441.53 | 439.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 438.40 | 441.53 | 439.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 439.10 | 441.04 | 439.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 441.00 | 441.04 | 439.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:15:00 | 439.20 | 440.50 | 439.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:45:00 | 439.40 | 440.32 | 439.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 437.35 | 439.53 | 439.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 437.35 | 439.53 | 439.63 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 444.85 | 440.05 | 439.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 10:15:00 | 451.20 | 443.68 | 441.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 09:15:00 | 489.10 | 496.72 | 486.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 10:00:00 | 489.10 | 496.72 | 486.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 493.60 | 496.10 | 486.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 488.70 | 496.10 | 486.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 489.25 | 493.58 | 487.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:45:00 | 488.80 | 493.58 | 487.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 487.35 | 492.34 | 487.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:30:00 | 488.55 | 492.34 | 487.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 485.35 | 490.94 | 487.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 15:00:00 | 485.35 | 490.94 | 487.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 485.90 | 489.93 | 487.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 483.45 | 489.93 | 487.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 485.10 | 488.12 | 486.64 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 483.55 | 485.98 | 485.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 15:15:00 | 482.20 | 485.23 | 485.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 487.85 | 485.75 | 485.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 487.85 | 485.75 | 485.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 487.85 | 485.75 | 485.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:30:00 | 484.10 | 485.75 | 485.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 10:15:00 | 487.65 | 486.13 | 486.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 13:15:00 | 491.05 | 488.85 | 487.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 487.20 | 489.36 | 488.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 487.20 | 489.36 | 488.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 487.20 | 489.36 | 488.27 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 12:15:00 | 484.40 | 487.59 | 487.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 482.15 | 486.50 | 487.18 | Break + close below crossover candle low |

### Cycle 33 — BUY (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 09:15:00 | 495.40 | 487.62 | 487.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 12:15:00 | 498.65 | 492.25 | 489.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 11:15:00 | 489.15 | 495.49 | 493.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 11:15:00 | 489.15 | 495.49 | 493.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 489.15 | 495.49 | 493.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:00:00 | 489.15 | 495.49 | 493.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 491.80 | 494.75 | 493.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:30:00 | 488.65 | 494.75 | 493.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 490.20 | 493.84 | 492.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:45:00 | 488.05 | 493.84 | 492.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 496.50 | 494.38 | 493.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 13:30:00 | 502.00 | 495.93 | 494.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-06 09:15:00 | 552.20 | 524.54 | 511.80 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 525.75 | 527.84 | 528.00 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 530.65 | 528.17 | 528.08 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 523.85 | 527.31 | 527.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 520.75 | 526.00 | 527.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 525.50 | 521.37 | 523.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 525.50 | 521.37 | 523.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 525.50 | 521.37 | 523.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 525.50 | 521.37 | 523.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 529.10 | 522.92 | 524.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 529.10 | 522.92 | 524.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 530.50 | 524.43 | 524.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 531.75 | 524.43 | 524.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 538.40 | 527.23 | 526.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 13:15:00 | 540.75 | 529.93 | 527.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 532.70 | 536.89 | 534.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 532.70 | 536.89 | 534.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 532.70 | 536.89 | 534.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:00:00 | 532.70 | 536.89 | 534.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 533.75 | 536.26 | 534.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 12:00:00 | 535.15 | 536.04 | 534.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 12:15:00 | 529.45 | 534.72 | 533.81 | SL hit (close<static) qty=1.00 sl=532.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 526.80 | 532.30 | 532.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 15:15:00 | 525.90 | 531.02 | 532.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 515.55 | 514.82 | 521.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-21 13:45:00 | 517.40 | 514.82 | 521.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 502.55 | 491.25 | 497.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:30:00 | 499.00 | 491.25 | 497.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 502.85 | 493.57 | 498.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:30:00 | 502.25 | 493.57 | 498.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 497.30 | 496.96 | 498.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:45:00 | 498.35 | 496.96 | 498.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 498.65 | 495.69 | 497.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 496.60 | 495.69 | 497.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 499.45 | 496.44 | 497.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 499.45 | 496.44 | 497.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 500.35 | 497.23 | 498.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:30:00 | 501.80 | 497.23 | 498.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 487.80 | 488.25 | 491.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:30:00 | 488.15 | 488.25 | 491.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 487.00 | 485.02 | 487.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:00:00 | 482.15 | 484.44 | 487.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 15:00:00 | 481.65 | 482.70 | 485.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 11:00:00 | 482.70 | 482.67 | 484.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:45:00 | 482.70 | 480.55 | 481.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 458.04 | 464.73 | 470.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 458.56 | 464.73 | 470.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 458.56 | 464.73 | 470.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 12:15:00 | 464.55 | 464.30 | 468.73 | SL hit (close>ema200) qty=0.50 sl=464.30 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 472.95 | 464.87 | 463.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 11:15:00 | 479.45 | 467.79 | 465.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 482.00 | 482.74 | 477.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 14:00:00 | 482.00 | 482.74 | 477.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 469.50 | 479.09 | 476.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 469.50 | 479.09 | 476.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 467.00 | 473.87 | 474.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 465.10 | 468.62 | 470.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 12:15:00 | 468.50 | 468.35 | 469.89 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 14:45:00 | 465.50 | 467.80 | 469.37 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 13:15:00 | 466.50 | 466.12 | 467.77 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 472.00 | 467.30 | 468.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-19 13:15:00 | 472.00 | 467.30 | 468.15 | SL hit (close>ema400) qty=1.00 sl=468.15 alert=retest1 |

### Cycle 41 — BUY (started 2025-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 13:15:00 | 471.30 | 468.28 | 468.01 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 10:15:00 | 463.25 | 467.13 | 467.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 459.15 | 465.53 | 466.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 463.00 | 461.57 | 464.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 10:00:00 | 463.00 | 461.57 | 464.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 460.20 | 461.29 | 463.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 11:45:00 | 458.50 | 460.85 | 463.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:45:00 | 459.30 | 460.55 | 462.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:15:00 | 459.90 | 460.44 | 462.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 15:15:00 | 459.50 | 460.35 | 462.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 468.90 | 461.92 | 462.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 468.90 | 461.92 | 462.76 | SL hit (close>static) qty=1.00 sl=464.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 472.40 | 465.13 | 464.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 09:15:00 | 487.00 | 470.87 | 468.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 13:15:00 | 475.35 | 475.70 | 471.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 14:00:00 | 475.35 | 475.70 | 471.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 480.50 | 482.98 | 479.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:30:00 | 477.75 | 482.98 | 479.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 478.75 | 482.13 | 479.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 478.75 | 482.13 | 479.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 476.25 | 480.95 | 479.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 476.25 | 480.95 | 479.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 474.00 | 479.56 | 478.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:45:00 | 473.75 | 479.56 | 478.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 475.25 | 477.90 | 478.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 467.10 | 475.34 | 476.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 10:15:00 | 465.95 | 465.50 | 469.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:30:00 | 467.15 | 465.50 | 469.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 465.40 | 465.36 | 467.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 465.40 | 465.36 | 467.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 467.55 | 465.79 | 467.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:00:00 | 467.55 | 465.79 | 467.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 467.95 | 466.23 | 467.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:45:00 | 469.70 | 466.23 | 467.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 466.50 | 466.28 | 467.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:45:00 | 468.90 | 466.28 | 467.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 470.15 | 467.05 | 467.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 470.15 | 467.05 | 467.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 470.85 | 467.81 | 468.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 470.85 | 467.81 | 468.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 15:15:00 | 472.90 | 468.83 | 468.66 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 461.90 | 467.44 | 468.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 460.20 | 466.00 | 467.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 452.35 | 452.31 | 458.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:00:00 | 452.35 | 452.31 | 458.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 456.70 | 453.78 | 457.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 451.50 | 454.40 | 456.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 12:30:00 | 452.00 | 451.06 | 452.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:00:00 | 451.15 | 451.47 | 452.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 12:15:00 | 443.95 | 442.61 | 442.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 443.95 | 442.61 | 442.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 448.05 | 444.10 | 443.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 12:15:00 | 471.60 | 472.30 | 466.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 13:00:00 | 471.60 | 472.30 | 466.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 472.45 | 471.05 | 467.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 12:15:00 | 475.80 | 471.83 | 468.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 12:45:00 | 474.20 | 472.60 | 469.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:45:00 | 475.00 | 473.72 | 470.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 10:45:00 | 474.30 | 473.64 | 471.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 473.40 | 473.59 | 471.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 15:00:00 | 475.60 | 473.74 | 471.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:30:00 | 475.70 | 474.17 | 472.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:00:00 | 474.90 | 474.17 | 472.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 11:15:00 | 477.45 | 474.21 | 472.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 479.35 | 483.22 | 481.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 479.35 | 483.22 | 481.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 482.50 | 483.08 | 481.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 479.00 | 481.05 | 481.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 479.00 | 481.05 | 481.30 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 11:15:00 | 483.30 | 481.50 | 481.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 14:15:00 | 485.55 | 482.68 | 482.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 11:15:00 | 483.55 | 484.19 | 483.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 12:00:00 | 483.55 | 484.19 | 483.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 480.35 | 483.42 | 482.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:30:00 | 480.75 | 483.42 | 482.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 13:15:00 | 478.45 | 482.43 | 482.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 475.40 | 480.10 | 481.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 10:15:00 | 453.25 | 452.93 | 457.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 11:15:00 | 456.40 | 452.93 | 457.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 460.15 | 455.48 | 457.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 460.15 | 455.48 | 457.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 460.05 | 456.39 | 457.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 462.60 | 457.69 | 458.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 463.25 | 458.80 | 458.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 465.20 | 460.91 | 459.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 11:15:00 | 465.80 | 466.56 | 463.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 11:45:00 | 464.30 | 466.56 | 463.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 463.95 | 466.04 | 463.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:30:00 | 463.55 | 466.04 | 463.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 465.75 | 465.98 | 463.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 483.50 | 465.28 | 463.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:30:00 | 467.15 | 467.95 | 465.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 09:15:00 | 461.45 | 468.52 | 467.04 | SL hit (close<static) qty=1.00 sl=461.55 alert=retest2 |

### Cycle 52 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 457.35 | 464.57 | 465.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 454.50 | 462.56 | 464.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 443.15 | 441.78 | 448.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 14:15:00 | 448.15 | 444.08 | 447.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 448.15 | 444.08 | 447.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 15:00:00 | 448.15 | 444.08 | 447.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 446.00 | 444.47 | 447.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:15:00 | 439.35 | 444.47 | 447.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 14:15:00 | 417.38 | 427.26 | 436.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-27 13:15:00 | 395.42 | 409.13 | 422.14 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 53 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 406.75 | 395.00 | 394.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 411.00 | 398.20 | 395.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 407.80 | 412.41 | 407.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 407.80 | 412.41 | 407.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 407.80 | 412.41 | 407.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 407.80 | 412.41 | 407.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 406.70 | 411.27 | 407.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 405.25 | 411.27 | 407.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 398.20 | 408.65 | 406.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:00:00 | 398.20 | 408.65 | 406.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 397.70 | 406.46 | 405.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:30:00 | 401.55 | 405.52 | 405.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-06 15:15:00 | 441.71 | 418.79 | 412.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 471.80 | 472.60 | 472.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 466.60 | 471.15 | 471.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 10:15:00 | 477.05 | 470.65 | 471.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 10:15:00 | 477.05 | 470.65 | 471.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 477.05 | 470.65 | 471.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:30:00 | 477.75 | 470.65 | 471.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 473.45 | 471.21 | 471.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 12:30:00 | 468.00 | 470.29 | 471.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 474.40 | 470.06 | 469.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 474.40 | 470.06 | 469.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 15:15:00 | 477.00 | 471.93 | 470.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 482.20 | 483.73 | 478.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 10:45:00 | 483.65 | 483.73 | 478.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 476.35 | 481.28 | 478.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 476.35 | 481.28 | 478.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 473.80 | 479.79 | 478.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 473.80 | 479.79 | 478.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 473.45 | 478.52 | 477.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 476.95 | 478.52 | 477.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 10:15:00 | 473.35 | 476.99 | 477.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 10:15:00 | 473.35 | 476.99 | 477.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 11:15:00 | 472.45 | 476.08 | 476.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 476.60 | 473.81 | 475.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 476.60 | 473.81 | 475.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 476.60 | 473.81 | 475.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 478.10 | 473.81 | 475.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 469.10 | 472.87 | 474.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:45:00 | 468.25 | 472.10 | 474.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:30:00 | 467.90 | 471.26 | 473.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 468.20 | 471.80 | 473.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 10:00:00 | 468.35 | 471.11 | 472.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 462.85 | 468.27 | 470.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:15:00 | 462.25 | 468.27 | 470.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 15:15:00 | 462.10 | 464.05 | 465.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 444.84 | 455.62 | 459.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 444.50 | 455.62 | 459.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 444.79 | 455.62 | 459.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 444.93 | 455.62 | 459.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 439.14 | 455.62 | 459.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 439.00 | 455.62 | 459.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 15:15:00 | 421.43 | 430.63 | 439.67 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 13:15:00 | 419.00 | 412.43 | 412.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-16 14:15:00 | 422.70 | 414.48 | 413.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 09:15:00 | 415.90 | 416.61 | 414.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 415.90 | 416.61 | 414.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 415.90 | 416.61 | 414.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 415.90 | 416.61 | 414.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 410.40 | 415.37 | 414.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 410.40 | 415.37 | 414.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 407.80 | 413.85 | 413.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:00:00 | 407.80 | 413.85 | 413.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 12:15:00 | 411.10 | 413.30 | 413.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 403.15 | 409.65 | 410.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 406.75 | 403.92 | 406.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 406.75 | 403.92 | 406.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 406.75 | 403.92 | 406.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 406.15 | 403.92 | 406.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 399.70 | 403.08 | 406.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:30:00 | 402.15 | 403.08 | 406.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 373.70 | 378.94 | 387.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 372.55 | 377.82 | 386.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 394.45 | 383.09 | 385.40 | SL hit (close>static) qty=1.00 sl=388.30 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 391.50 | 387.56 | 387.13 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 378.80 | 387.30 | 387.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 378.40 | 385.52 | 386.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 14:15:00 | 387.10 | 383.91 | 385.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 14:15:00 | 387.10 | 383.91 | 385.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 387.10 | 383.91 | 385.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 387.10 | 383.91 | 385.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 386.50 | 384.43 | 385.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 380.15 | 384.43 | 385.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 383.00 | 380.47 | 381.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 15:15:00 | 384.80 | 381.60 | 381.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 384.80 | 381.60 | 381.49 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 370.60 | 379.40 | 380.50 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 393.90 | 382.56 | 381.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 15:15:00 | 406.00 | 391.99 | 387.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 10:15:00 | 410.35 | 414.24 | 407.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 11:00:00 | 410.35 | 414.24 | 407.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 11:15:00 | 401.80 | 411.75 | 407.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 12:00:00 | 401.80 | 411.75 | 407.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 402.70 | 409.94 | 406.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 13:30:00 | 405.00 | 409.03 | 406.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:45:00 | 410.75 | 406.87 | 406.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 11:15:00 | 401.35 | 405.52 | 405.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 11:15:00 | 401.35 | 405.52 | 405.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-10 14:15:00 | 398.95 | 402.50 | 404.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 13:15:00 | 405.10 | 399.15 | 401.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 13:15:00 | 405.10 | 399.15 | 401.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 13:15:00 | 405.10 | 399.15 | 401.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 14:00:00 | 405.10 | 399.15 | 401.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 408.35 | 400.99 | 401.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 15:00:00 | 408.35 | 400.99 | 401.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 15:15:00 | 410.00 | 402.79 | 402.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 10:15:00 | 415.05 | 406.40 | 404.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 407.80 | 411.08 | 408.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 407.80 | 411.08 | 408.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 407.80 | 411.08 | 408.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:45:00 | 406.60 | 411.08 | 408.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 403.00 | 409.46 | 407.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:00:00 | 403.00 | 409.46 | 407.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 401.05 | 407.78 | 407.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:45:00 | 400.50 | 407.78 | 407.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 402.15 | 406.65 | 406.69 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 13:15:00 | 413.90 | 407.66 | 406.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 11:15:00 | 419.55 | 412.25 | 409.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 11:15:00 | 421.80 | 422.13 | 418.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-22 11:45:00 | 422.00 | 422.13 | 418.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 424.55 | 423.58 | 420.91 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 416.00 | 420.54 | 420.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 413.45 | 418.64 | 419.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 418.00 | 417.64 | 418.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 421.30 | 417.64 | 418.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 420.00 | 418.11 | 419.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:30:00 | 419.55 | 418.11 | 419.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 420.20 | 418.53 | 419.11 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 420.00 | 419.48 | 419.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 421.70 | 420.01 | 419.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 418.00 | 419.61 | 419.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 10:15:00 | 418.00 | 419.61 | 419.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 418.00 | 419.61 | 419.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:00:00 | 418.00 | 419.61 | 419.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 11:15:00 | 414.65 | 418.61 | 419.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 12:15:00 | 414.00 | 417.69 | 418.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 12:15:00 | 414.65 | 414.19 | 416.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 13:00:00 | 414.65 | 414.19 | 416.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 416.55 | 414.67 | 416.17 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 15:15:00 | 421.80 | 417.55 | 417.31 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 413.70 | 416.78 | 416.98 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 418.90 | 417.25 | 417.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 429.00 | 419.60 | 418.16 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-20 11:15:00 | 394.80 | 2025-05-22 12:15:00 | 390.40 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-05-20 13:00:00 | 394.70 | 2025-05-22 12:15:00 | 390.40 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-05-21 10:30:00 | 394.55 | 2025-05-22 12:15:00 | 390.40 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-05-21 11:00:00 | 394.80 | 2025-05-22 12:15:00 | 390.40 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-05-27 11:15:00 | 399.60 | 2025-06-03 15:15:00 | 403.25 | STOP_HIT | 1.00 | 0.91% |
| BUY | retest2 | 2025-05-27 12:00:00 | 399.60 | 2025-06-03 15:15:00 | 403.25 | STOP_HIT | 1.00 | 0.91% |
| BUY | retest2 | 2025-05-28 11:00:00 | 398.45 | 2025-06-03 15:15:00 | 403.25 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest2 | 2025-05-28 12:45:00 | 398.15 | 2025-06-03 15:15:00 | 403.25 | STOP_HIT | 1.00 | 1.28% |
| BUY | retest2 | 2025-05-28 14:30:00 | 401.00 | 2025-06-03 15:15:00 | 403.25 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2025-05-30 09:45:00 | 400.15 | 2025-06-03 15:15:00 | 403.25 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest2 | 2025-05-30 11:30:00 | 400.30 | 2025-06-03 15:15:00 | 403.25 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2025-06-11 15:15:00 | 426.95 | 2025-06-12 13:15:00 | 421.25 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-06-12 11:00:00 | 428.00 | 2025-06-12 13:15:00 | 421.25 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-06-12 12:00:00 | 428.25 | 2025-06-12 13:15:00 | 421.25 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-06-17 15:15:00 | 419.00 | 2025-06-18 09:15:00 | 422.50 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-06-19 11:15:00 | 417.40 | 2025-06-23 10:15:00 | 424.15 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-06-20 11:45:00 | 416.90 | 2025-06-23 10:15:00 | 424.15 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-06-23 09:30:00 | 417.60 | 2025-06-23 10:15:00 | 424.15 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-06-26 13:00:00 | 444.70 | 2025-07-07 15:15:00 | 467.00 | STOP_HIT | 1.00 | 5.01% |
| SELL | retest2 | 2025-07-16 11:30:00 | 450.45 | 2025-07-18 13:15:00 | 455.40 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-07-17 09:30:00 | 449.65 | 2025-07-18 13:15:00 | 455.40 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-07-18 10:15:00 | 450.55 | 2025-07-18 13:15:00 | 455.40 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-08-04 09:30:00 | 416.00 | 2025-08-04 10:15:00 | 424.25 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-08-21 09:30:00 | 465.00 | 2025-08-21 10:15:00 | 459.10 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-09-09 14:15:00 | 429.30 | 2025-09-10 09:15:00 | 442.85 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2025-09-09 15:15:00 | 430.00 | 2025-09-10 09:15:00 | 442.85 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2025-09-12 09:15:00 | 441.00 | 2025-09-15 09:15:00 | 437.35 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-09-12 13:15:00 | 439.20 | 2025-09-15 09:15:00 | 437.35 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-09-12 13:45:00 | 439.40 | 2025-09-15 09:15:00 | 437.35 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-10-01 13:30:00 | 502.00 | 2025-10-06 09:15:00 | 552.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-17 12:00:00 | 535.15 | 2025-10-17 12:15:00 | 529.45 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-10-31 11:00:00 | 482.15 | 2025-11-07 09:15:00 | 458.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 15:00:00 | 481.65 | 2025-11-07 09:15:00 | 458.56 | PARTIAL | 0.50 | 4.79% |
| SELL | retest2 | 2025-11-03 11:00:00 | 482.70 | 2025-11-07 09:15:00 | 458.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 11:00:00 | 482.15 | 2025-11-07 12:15:00 | 464.55 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2025-10-31 15:00:00 | 481.65 | 2025-11-07 12:15:00 | 464.55 | STOP_HIT | 0.50 | 3.55% |
| SELL | retest2 | 2025-11-03 11:00:00 | 482.70 | 2025-11-07 12:15:00 | 464.55 | STOP_HIT | 0.50 | 3.76% |
| SELL | retest2 | 2025-11-04 11:45:00 | 482.70 | 2025-11-10 09:15:00 | 457.57 | PARTIAL | 0.50 | 5.21% |
| SELL | retest2 | 2025-11-04 11:45:00 | 482.70 | 2025-11-11 11:15:00 | 459.90 | STOP_HIT | 0.50 | 4.72% |
| SELL | retest2 | 2025-11-11 09:30:00 | 456.15 | 2025-11-11 14:15:00 | 469.20 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2025-11-11 11:00:00 | 456.55 | 2025-11-11 14:15:00 | 469.20 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest1 | 2025-11-18 14:45:00 | 465.50 | 2025-11-19 13:15:00 | 472.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest1 | 2025-11-19 13:15:00 | 466.50 | 2025-11-19 13:15:00 | 472.00 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-11-20 09:15:00 | 465.70 | 2025-11-21 13:15:00 | 471.30 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-11-20 12:15:00 | 466.65 | 2025-11-21 13:15:00 | 471.30 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-11-20 14:15:00 | 466.50 | 2025-11-21 13:15:00 | 471.30 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-11-20 14:45:00 | 465.95 | 2025-11-21 13:15:00 | 471.30 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-11-21 09:15:00 | 464.15 | 2025-11-21 13:15:00 | 471.30 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-11-21 10:15:00 | 463.90 | 2025-11-21 13:15:00 | 471.30 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-11-25 11:45:00 | 458.50 | 2025-11-26 09:15:00 | 468.90 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-11-25 12:45:00 | 459.30 | 2025-11-26 09:15:00 | 468.90 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-11-25 14:15:00 | 459.90 | 2025-11-26 09:15:00 | 468.90 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-11-25 15:15:00 | 459.50 | 2025-11-26 09:15:00 | 468.90 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-12-10 10:45:00 | 451.50 | 2025-12-19 12:15:00 | 443.95 | STOP_HIT | 1.00 | 1.67% |
| SELL | retest2 | 2025-12-11 12:30:00 | 452.00 | 2025-12-19 12:15:00 | 443.95 | STOP_HIT | 1.00 | 1.78% |
| SELL | retest2 | 2025-12-12 12:00:00 | 451.15 | 2025-12-19 12:15:00 | 443.95 | STOP_HIT | 1.00 | 1.60% |
| BUY | retest2 | 2025-12-26 12:15:00 | 475.80 | 2026-01-05 10:15:00 | 479.00 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2025-12-26 12:45:00 | 474.20 | 2026-01-05 10:15:00 | 479.00 | STOP_HIT | 1.00 | 1.01% |
| BUY | retest2 | 2025-12-29 09:45:00 | 475.00 | 2026-01-05 10:15:00 | 479.00 | STOP_HIT | 1.00 | 0.84% |
| BUY | retest2 | 2025-12-29 10:45:00 | 474.30 | 2026-01-05 10:15:00 | 479.00 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest2 | 2025-12-29 15:00:00 | 475.60 | 2026-01-05 10:15:00 | 479.00 | STOP_HIT | 1.00 | 0.71% |
| BUY | retest2 | 2025-12-30 09:30:00 | 475.70 | 2026-01-05 10:15:00 | 479.00 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2025-12-30 10:00:00 | 474.90 | 2026-01-05 10:15:00 | 479.00 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2025-12-30 11:15:00 | 477.45 | 2026-01-05 10:15:00 | 479.00 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2026-01-19 09:15:00 | 483.50 | 2026-01-20 09:15:00 | 461.45 | STOP_HIT | 1.00 | -4.56% |
| BUY | retest2 | 2026-01-19 12:30:00 | 467.15 | 2026-01-20 09:15:00 | 461.45 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-01-23 09:15:00 | 439.35 | 2026-01-23 14:15:00 | 417.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 09:15:00 | 439.35 | 2026-01-27 13:15:00 | 395.42 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-05 13:30:00 | 401.55 | 2026-02-06 15:15:00 | 441.71 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-16 12:30:00 | 468.00 | 2026-02-17 13:15:00 | 474.40 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-02-20 09:15:00 | 476.95 | 2026-02-20 10:15:00 | 473.35 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2026-02-23 11:45:00 | 468.25 | 2026-03-02 09:15:00 | 444.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 12:30:00 | 467.90 | 2026-03-02 09:15:00 | 444.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 468.20 | 2026-03-02 09:15:00 | 444.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 10:00:00 | 468.35 | 2026-03-02 09:15:00 | 444.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:15:00 | 462.25 | 2026-03-02 09:15:00 | 439.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 15:15:00 | 462.10 | 2026-03-02 09:15:00 | 439.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 11:45:00 | 468.25 | 2026-03-04 15:15:00 | 421.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-23 12:30:00 | 467.90 | 2026-03-04 15:15:00 | 421.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 468.20 | 2026-03-04 15:15:00 | 421.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 10:00:00 | 468.35 | 2026-03-04 15:15:00 | 421.52 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 11:15:00 | 462.25 | 2026-03-05 09:15:00 | 432.05 | STOP_HIT | 0.50 | 6.53% |
| SELL | retest2 | 2026-02-26 15:15:00 | 462.10 | 2026-03-05 09:15:00 | 432.05 | STOP_HIT | 0.50 | 6.50% |
| SELL | retest2 | 2026-03-24 10:30:00 | 372.55 | 2026-03-25 09:15:00 | 394.45 | STOP_HIT | 1.00 | -5.88% |
| SELL | retest2 | 2026-03-30 09:15:00 | 380.15 | 2026-04-01 15:15:00 | 384.80 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-04-01 13:30:00 | 383.00 | 2026-04-01 15:15:00 | 384.80 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2026-04-09 13:30:00 | 405.00 | 2026-04-10 11:15:00 | 401.35 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2026-04-10 09:45:00 | 410.75 | 2026-04-10 11:15:00 | 401.35 | STOP_HIT | 1.00 | -2.29% |
