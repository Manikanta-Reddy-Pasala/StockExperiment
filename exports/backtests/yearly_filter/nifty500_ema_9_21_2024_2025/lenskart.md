# Lenskart Solutions Ltd. (LENSKART)

## Backtest Summary

- **Window:** 2025-11-10 09:15:00 → 2026-05-11 15:15:00 (861 bars)
- **Last close:** 488.55
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 53 |
| ALERT1 | 33 |
| ALERT2 | 33 |
| ALERT2_SKIP | 17 |
| ALERT3 | 87 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 28 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 25
- **Target hits / Stop hits / Partials:** 1 / 27 / 1
- **Avg / median % per leg:** -1.09% / -1.50%
- **Sum % (uncompounded):** -31.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 1 | 11.1% | 1 | 8 | 0 | -1.23% | -11.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 1 | 11.1% | 1 | 8 | 0 | -1.23% | -11.0% |
| SELL (all) | 20 | 3 | 15.0% | 0 | 19 | 1 | -1.04% | -20.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 3 | 15.0% | 0 | 19 | 1 | -1.04% | -20.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 29 | 4 | 13.8% | 1 | 27 | 1 | -1.09% | -31.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 400.45 | 408.79 | 409.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 391.95 | 405.42 | 407.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 12:15:00 | 406.50 | 404.75 | 406.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 12:15:00 | 406.50 | 404.75 | 406.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 406.50 | 404.75 | 406.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 406.50 | 404.75 | 406.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 404.75 | 404.75 | 406.68 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 415.95 | 408.16 | 407.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 10:15:00 | 418.75 | 410.28 | 408.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 420.30 | 421.71 | 416.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 10:00:00 | 420.30 | 421.71 | 416.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 419.00 | 423.43 | 419.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 417.40 | 423.43 | 419.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 412.20 | 421.19 | 419.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 412.20 | 421.19 | 419.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 414.60 | 419.87 | 418.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:45:00 | 411.75 | 419.87 | 418.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 418.35 | 422.57 | 420.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 418.25 | 422.57 | 420.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 420.60 | 422.18 | 420.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 12:00:00 | 421.65 | 422.07 | 420.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 15:15:00 | 415.00 | 419.45 | 419.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 415.00 | 419.45 | 419.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 11:15:00 | 413.40 | 416.85 | 418.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 14:15:00 | 419.35 | 416.15 | 417.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 14:15:00 | 419.35 | 416.15 | 417.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 419.35 | 416.15 | 417.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 419.35 | 416.15 | 417.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 419.40 | 416.80 | 417.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 414.10 | 416.80 | 417.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 13:15:00 | 423.45 | 416.59 | 416.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 13:15:00 | 423.45 | 416.59 | 416.20 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 09:15:00 | 413.50 | 415.51 | 415.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-26 12:15:00 | 411.00 | 414.47 | 415.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 14:15:00 | 408.20 | 408.09 | 410.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-27 15:00:00 | 408.20 | 408.09 | 410.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 414.80 | 409.34 | 410.64 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 422.20 | 413.60 | 412.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 11:15:00 | 441.80 | 421.43 | 416.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 12:15:00 | 425.95 | 428.21 | 424.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 13:00:00 | 425.95 | 428.21 | 424.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 426.80 | 427.79 | 424.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:45:00 | 425.55 | 427.79 | 424.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 422.50 | 426.73 | 424.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 417.00 | 426.73 | 424.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 413.95 | 424.17 | 423.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 413.95 | 424.17 | 423.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 412.60 | 421.86 | 422.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 14:15:00 | 411.10 | 416.32 | 419.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 408.30 | 406.39 | 410.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 10:00:00 | 408.30 | 406.39 | 410.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 413.70 | 408.42 | 411.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:45:00 | 414.80 | 408.42 | 411.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 413.70 | 409.48 | 411.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:45:00 | 413.40 | 409.48 | 411.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 15:15:00 | 418.05 | 413.05 | 412.65 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 407.70 | 411.98 | 412.20 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 12:15:00 | 415.65 | 411.03 | 410.96 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 15:15:00 | 408.60 | 410.63 | 410.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 13:15:00 | 406.35 | 408.61 | 409.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 14:15:00 | 406.00 | 405.59 | 407.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 15:00:00 | 406.00 | 405.59 | 407.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 406.00 | 405.75 | 406.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 405.50 | 405.83 | 406.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 15:15:00 | 404.00 | 406.41 | 406.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:45:00 | 404.70 | 405.56 | 406.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 09:45:00 | 405.25 | 404.99 | 405.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 405.25 | 405.05 | 405.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 405.20 | 405.05 | 405.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 404.50 | 404.94 | 405.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:15:00 | 403.55 | 404.94 | 405.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 15:15:00 | 402.90 | 404.73 | 405.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 407.30 | 404.95 | 405.15 | SL hit (close>static) qty=1.00 sl=405.60 alert=retest2 |

### Cycle 12 — BUY (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 10:15:00 | 414.75 | 406.91 | 406.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 419.95 | 410.91 | 408.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 462.20 | 464.68 | 448.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 455.30 | 464.68 | 448.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 458.50 | 467.75 | 461.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:45:00 | 458.50 | 467.75 | 461.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 454.95 | 465.19 | 460.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:00:00 | 454.95 | 465.19 | 460.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 452.50 | 459.52 | 458.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:30:00 | 453.85 | 459.52 | 458.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 448.95 | 457.40 | 458.02 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 10:15:00 | 464.25 | 458.91 | 458.61 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 454.45 | 458.50 | 458.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 15:15:00 | 448.15 | 455.79 | 457.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 453.95 | 453.62 | 455.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 13:45:00 | 453.25 | 453.62 | 455.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 457.55 | 454.05 | 455.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:45:00 | 457.30 | 454.05 | 455.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 459.00 | 455.04 | 455.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 12:00:00 | 456.10 | 455.25 | 455.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 12:15:00 | 458.40 | 455.88 | 455.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 12:15:00 | 458.40 | 455.88 | 455.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 13:15:00 | 462.20 | 457.15 | 456.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 10:15:00 | 456.50 | 457.91 | 457.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 456.50 | 457.91 | 457.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 456.50 | 457.91 | 457.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 456.50 | 457.91 | 457.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 454.30 | 457.19 | 456.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 454.30 | 457.19 | 456.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 455.50 | 456.85 | 456.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 455.50 | 456.85 | 456.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 13:15:00 | 453.80 | 456.24 | 456.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-31 14:15:00 | 451.40 | 455.27 | 456.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 455.00 | 447.13 | 450.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 455.00 | 447.13 | 450.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 455.00 | 447.13 | 450.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 455.00 | 447.13 | 450.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 452.45 | 448.20 | 450.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 15:00:00 | 451.55 | 450.72 | 451.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 451.50 | 447.41 | 447.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 451.50 | 447.41 | 447.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-08 09:15:00 | 457.45 | 452.27 | 450.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 11:15:00 | 446.10 | 451.48 | 450.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 11:15:00 | 446.10 | 451.48 | 450.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 446.10 | 451.48 | 450.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:45:00 | 445.40 | 451.48 | 450.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 447.15 | 450.61 | 449.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:45:00 | 447.05 | 450.61 | 449.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 447.50 | 449.11 | 449.24 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 09:15:00 | 453.05 | 449.90 | 449.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-09 10:15:00 | 457.65 | 451.45 | 450.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 15:15:00 | 452.90 | 454.52 | 452.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 15:15:00 | 452.90 | 454.52 | 452.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 452.90 | 454.52 | 452.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 447.35 | 454.52 | 452.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 454.15 | 454.44 | 452.68 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2026-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 12:15:00 | 452.00 | 452.60 | 452.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 14:15:00 | 447.05 | 451.40 | 452.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 452.45 | 449.94 | 451.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 452.45 | 449.94 | 451.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 452.45 | 449.94 | 451.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 452.45 | 449.94 | 451.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 454.90 | 450.94 | 451.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 454.90 | 450.94 | 451.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 458.50 | 452.45 | 452.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 469.35 | 457.70 | 454.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 455.40 | 461.66 | 458.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 13:15:00 | 455.40 | 461.66 | 458.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 455.40 | 461.66 | 458.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 455.40 | 461.66 | 458.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 452.05 | 459.74 | 457.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 453.50 | 459.74 | 457.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 442.95 | 454.92 | 455.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 13:15:00 | 442.00 | 447.95 | 451.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 436.30 | 433.84 | 439.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 436.30 | 433.84 | 439.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 436.30 | 433.84 | 439.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:45:00 | 435.10 | 433.84 | 439.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 433.70 | 433.37 | 437.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:30:00 | 436.05 | 433.37 | 437.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 435.75 | 434.43 | 437.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 435.80 | 434.43 | 437.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 438.60 | 435.26 | 437.51 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 439.60 | 438.45 | 438.36 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 430.85 | 436.92 | 437.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 423.25 | 434.19 | 436.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 430.45 | 430.21 | 433.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 09:45:00 | 431.70 | 430.21 | 433.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 430.40 | 429.94 | 432.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 13:30:00 | 430.75 | 429.94 | 432.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 433.90 | 430.73 | 432.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 14:45:00 | 431.80 | 430.73 | 432.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 435.80 | 431.74 | 432.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 09:15:00 | 430.85 | 431.74 | 432.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 10:15:00 | 437.30 | 433.26 | 433.37 | SL hit (close>static) qty=1.00 sl=436.00 alert=retest2 |

### Cycle 26 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 437.00 | 434.01 | 433.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 11:15:00 | 440.65 | 436.61 | 435.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 10:15:00 | 452.70 | 453.75 | 448.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 10:45:00 | 452.50 | 453.75 | 448.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 447.50 | 452.50 | 448.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 447.50 | 452.50 | 448.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 444.30 | 450.86 | 448.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 440.00 | 450.86 | 448.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 433.00 | 444.66 | 445.73 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 452.75 | 444.82 | 444.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 12:15:00 | 473.95 | 453.06 | 448.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 456.10 | 456.19 | 452.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 11:00:00 | 456.10 | 456.19 | 452.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 450.00 | 454.95 | 451.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:30:00 | 450.15 | 454.95 | 451.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 448.50 | 453.66 | 451.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:45:00 | 447.55 | 453.66 | 451.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 445.70 | 451.63 | 451.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 446.40 | 451.63 | 451.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 442.55 | 449.81 | 450.42 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 453.55 | 450.35 | 449.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 455.35 | 451.35 | 450.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 471.60 | 471.63 | 465.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:45:00 | 468.95 | 471.63 | 465.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 464.85 | 470.48 | 466.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 15:15:00 | 473.00 | 468.50 | 466.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-12 09:15:00 | 520.30 | 478.20 | 471.34 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 09:15:00 | 483.10 | 495.40 | 495.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 14:15:00 | 480.00 | 486.08 | 490.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 10:15:00 | 485.90 | 484.43 | 488.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 10:45:00 | 485.80 | 484.43 | 488.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 485.90 | 485.28 | 487.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 483.50 | 485.65 | 487.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:00:00 | 481.90 | 485.22 | 486.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 494.25 | 483.91 | 485.32 | SL hit (close>static) qty=1.00 sl=488.50 alert=retest2 |

### Cycle 32 — BUY (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 12:15:00 | 489.05 | 486.46 | 486.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 491.30 | 488.62 | 487.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 14:15:00 | 485.10 | 488.43 | 487.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 14:15:00 | 485.10 | 488.43 | 487.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 485.10 | 488.43 | 487.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 485.10 | 488.43 | 487.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2026-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 15:15:00 | 483.70 | 487.48 | 487.53 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 500.90 | 488.72 | 487.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 10:15:00 | 504.20 | 491.81 | 489.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 10:15:00 | 525.40 | 528.31 | 519.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-02 10:45:00 | 526.65 | 528.31 | 519.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 11:15:00 | 517.90 | 526.22 | 519.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 12:00:00 | 517.90 | 526.22 | 519.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 12:15:00 | 521.00 | 525.18 | 519.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 12:30:00 | 517.85 | 525.18 | 519.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 523.70 | 524.88 | 519.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 13:45:00 | 518.65 | 524.88 | 519.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 524.00 | 524.97 | 520.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:15:00 | 518.35 | 524.97 | 520.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 516.40 | 523.25 | 520.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:00:00 | 516.40 | 523.25 | 520.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 518.75 | 522.35 | 520.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:45:00 | 519.40 | 522.35 | 520.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 524.45 | 523.15 | 521.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 13:45:00 | 521.95 | 523.15 | 521.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 520.00 | 522.71 | 521.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:15:00 | 527.25 | 522.71 | 521.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 523.35 | 522.83 | 521.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 531.50 | 522.66 | 521.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 10:15:00 | 532.45 | 524.15 | 522.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 11:30:00 | 532.30 | 526.60 | 524.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 503.70 | 519.72 | 521.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 503.70 | 519.72 | 521.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 497.00 | 504.71 | 509.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 12:15:00 | 501.65 | 501.35 | 506.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 13:00:00 | 501.65 | 501.35 | 506.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 503.10 | 501.65 | 505.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 15:00:00 | 503.10 | 501.65 | 505.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 496.15 | 500.69 | 504.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:30:00 | 502.80 | 500.69 | 504.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 507.55 | 502.06 | 504.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:45:00 | 509.00 | 502.06 | 504.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 509.35 | 503.52 | 504.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:30:00 | 507.70 | 503.52 | 504.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 499.40 | 502.70 | 504.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 13:45:00 | 498.75 | 501.96 | 503.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 15:15:00 | 498.90 | 501.55 | 503.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 10:15:00 | 491.30 | 500.96 | 502.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 13:30:00 | 498.75 | 499.18 | 501.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 502.10 | 498.64 | 500.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-17 12:15:00 | 504.85 | 502.06 | 501.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 504.85 | 502.06 | 501.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 518.10 | 505.57 | 503.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 507.80 | 509.38 | 506.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 507.80 | 509.38 | 506.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 507.80 | 509.38 | 506.96 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 502.75 | 505.96 | 506.11 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 511.80 | 506.39 | 506.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 11:15:00 | 515.15 | 508.14 | 507.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 489.50 | 509.86 | 508.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 489.50 | 509.86 | 508.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 489.50 | 509.86 | 508.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 489.50 | 509.86 | 508.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 486.45 | 505.18 | 506.91 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 513.05 | 505.14 | 504.74 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 15:15:00 | 502.45 | 504.75 | 504.78 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 506.00 | 505.00 | 504.89 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 10:15:00 | 503.05 | 504.61 | 504.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-25 11:15:00 | 496.00 | 502.89 | 503.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 09:15:00 | 499.55 | 498.97 | 501.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:45:00 | 499.80 | 498.97 | 501.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 501.70 | 497.66 | 499.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 501.70 | 497.66 | 499.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 509.70 | 500.07 | 500.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 509.70 | 500.07 | 500.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 15:15:00 | 512.00 | 502.45 | 501.71 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 489.65 | 499.89 | 500.61 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 510.00 | 500.94 | 500.35 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 488.00 | 499.48 | 500.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 10:15:00 | 484.85 | 496.55 | 499.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 09:15:00 | 489.15 | 489.09 | 493.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 489.15 | 489.09 | 493.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 489.15 | 489.09 | 493.50 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 504.70 | 496.28 | 495.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 511.35 | 501.80 | 499.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 14:15:00 | 504.40 | 504.86 | 502.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-08 15:00:00 | 504.40 | 504.86 | 502.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 537.20 | 543.76 | 539.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 15:00:00 | 537.20 | 543.76 | 539.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 15:15:00 | 538.30 | 542.67 | 539.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 09:15:00 | 542.10 | 542.67 | 539.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 533.60 | 540.86 | 538.81 | SL hit (close<static) qty=1.00 sl=537.00 alert=retest2 |

### Cycle 49 — SELL (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 15:15:00 | 536.00 | 538.28 | 538.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 09:15:00 | 523.05 | 535.24 | 537.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-20 14:15:00 | 530.00 | 526.33 | 531.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 14:15:00 | 530.00 | 526.33 | 531.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 530.00 | 526.33 | 531.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 15:00:00 | 530.00 | 526.33 | 531.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 536.00 | 528.26 | 531.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:15:00 | 538.45 | 528.26 | 531.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 540.95 | 530.80 | 532.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:45:00 | 540.90 | 530.80 | 532.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 540.45 | 532.73 | 533.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:00:00 | 540.45 | 532.73 | 533.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 11:15:00 | 537.95 | 533.77 | 533.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 15:15:00 | 542.70 | 537.65 | 535.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 09:15:00 | 533.65 | 541.14 | 539.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 533.65 | 541.14 | 539.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 533.65 | 541.14 | 539.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:00:00 | 533.65 | 541.14 | 539.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 533.10 | 539.53 | 538.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:30:00 | 531.35 | 539.53 | 538.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 12:15:00 | 535.65 | 538.37 | 538.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 13:15:00 | 532.55 | 537.21 | 537.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 14:15:00 | 538.05 | 537.38 | 537.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 14:15:00 | 538.05 | 537.38 | 537.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 538.05 | 537.38 | 537.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:45:00 | 538.65 | 537.38 | 537.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 538.00 | 537.50 | 537.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 540.45 | 537.50 | 537.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 538.00 | 537.60 | 537.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:30:00 | 540.25 | 537.60 | 537.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 529.65 | 536.01 | 537.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:00:00 | 525.60 | 532.10 | 534.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:00:00 | 527.70 | 527.02 | 529.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 14:15:00 | 535.30 | 531.51 | 531.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 14:15:00 | 535.30 | 531.51 | 531.41 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 11:15:00 | 524.75 | 530.60 | 531.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 520.90 | 527.48 | 529.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 525.35 | 518.82 | 522.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 525.35 | 518.82 | 522.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 525.35 | 518.82 | 522.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:30:00 | 526.60 | 518.82 | 522.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 519.05 | 518.86 | 521.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:15:00 | 517.85 | 518.86 | 521.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 10:15:00 | 491.96 | 500.65 | 505.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 15:15:00 | 492.95 | 492.65 | 499.33 | SL hit (close>ema200) qty=0.50 sl=492.65 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-20 12:00:00 | 421.65 | 2025-11-20 15:15:00 | 415.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-11-24 09:15:00 | 414.10 | 2025-11-25 13:15:00 | 423.45 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-12-12 10:45:00 | 405.50 | 2025-12-18 09:15:00 | 407.30 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-12-15 15:15:00 | 404.00 | 2025-12-18 09:15:00 | 407.30 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-12-16 09:45:00 | 404.70 | 2025-12-18 10:15:00 | 414.75 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-12-17 09:45:00 | 405.25 | 2025-12-18 10:15:00 | 414.75 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-12-17 12:15:00 | 403.55 | 2025-12-18 10:15:00 | 414.75 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-12-17 15:15:00 | 402.90 | 2025-12-18 10:15:00 | 414.75 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2025-12-30 12:00:00 | 456.10 | 2025-12-30 12:15:00 | 458.40 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-01-02 15:00:00 | 451.55 | 2026-01-07 10:15:00 | 451.50 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2026-01-28 09:15:00 | 430.85 | 2026-01-28 10:15:00 | 437.30 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-02-11 15:15:00 | 473.00 | 2026-02-12 09:15:00 | 520.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-19 09:15:00 | 483.50 | 2026-02-20 09:15:00 | 494.25 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2026-02-19 12:00:00 | 481.90 | 2026-02-20 09:15:00 | 494.25 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2026-03-06 09:15:00 | 531.50 | 2026-03-09 09:15:00 | 503.70 | STOP_HIT | 1.00 | -5.23% |
| BUY | retest2 | 2026-03-06 10:15:00 | 532.45 | 2026-03-09 09:15:00 | 503.70 | STOP_HIT | 1.00 | -5.40% |
| BUY | retest2 | 2026-03-06 11:30:00 | 532.30 | 2026-03-09 09:15:00 | 503.70 | STOP_HIT | 1.00 | -5.37% |
| SELL | retest2 | 2026-03-13 13:45:00 | 498.75 | 2026-03-17 12:15:00 | 504.85 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-03-13 15:15:00 | 498.90 | 2026-03-17 12:15:00 | 504.85 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-03-16 10:15:00 | 491.30 | 2026-03-17 12:15:00 | 504.85 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2026-03-16 13:30:00 | 498.75 | 2026-03-17 12:15:00 | 504.85 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-04-16 09:15:00 | 542.10 | 2026-04-16 09:15:00 | 533.60 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2026-04-16 13:30:00 | 539.30 | 2026-04-17 13:15:00 | 536.90 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2026-04-17 10:00:00 | 540.25 | 2026-04-17 13:15:00 | 536.90 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-04-17 12:30:00 | 541.30 | 2026-04-17 13:15:00 | 536.90 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-04-27 11:00:00 | 525.60 | 2026-04-28 14:15:00 | 535.30 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2026-04-28 11:00:00 | 527.70 | 2026-04-28 14:15:00 | 535.30 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-05-04 11:15:00 | 517.85 | 2026-05-07 10:15:00 | 491.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-05-04 11:15:00 | 517.85 | 2026-05-07 15:15:00 | 492.95 | STOP_HIT | 0.50 | 4.81% |
