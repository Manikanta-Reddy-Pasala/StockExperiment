# Housing & Urban Development Corporation Ltd. (HUDCO)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 232.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 201 |
| ALERT1 | 142 |
| ALERT2 | 141 |
| ALERT2_SKIP | 59 |
| ALERT3 | 384 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 159 |
| PARTIAL | 26 |
| TARGET_HIT | 20 |
| STOP_HIT | 139 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 185 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 82 / 103
- **Target hits / Stop hits / Partials:** 20 / 139 / 26
- **Avg / median % per leg:** 1.14% / -0.80%
- **Sum % (uncompounded):** 211.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 78 | 27 | 34.6% | 14 | 64 | 0 | 1.01% | 78.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 78 | 27 | 34.6% | 14 | 64 | 0 | 1.01% | 78.6% |
| SELL (all) | 107 | 55 | 51.4% | 6 | 75 | 26 | 1.24% | 132.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 107 | 55 | 51.4% | 6 | 75 | 26 | 1.24% | 132.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 185 | 82 | 44.3% | 20 | 139 | 26 | 1.14% | 211.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 13:15:00 | 55.00 | 54.68 | 54.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-16 09:15:00 | 56.30 | 55.09 | 54.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-17 11:15:00 | 57.75 | 58.11 | 57.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-17 12:00:00 | 57.75 | 58.11 | 57.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 14:15:00 | 57.60 | 57.87 | 57.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-17 14:45:00 | 57.50 | 57.87 | 57.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 14:15:00 | 58.05 | 58.27 | 57.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 14:30:00 | 57.75 | 58.27 | 57.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 09:15:00 | 56.30 | 57.83 | 57.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 10:00:00 | 56.30 | 57.83 | 57.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 10:15:00 | 56.75 | 57.61 | 57.60 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 11:15:00 | 56.90 | 57.47 | 57.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-22 13:15:00 | 56.35 | 56.76 | 57.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-23 10:15:00 | 56.70 | 56.51 | 56.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-23 11:00:00 | 56.70 | 56.51 | 56.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 11:15:00 | 56.05 | 56.41 | 56.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-23 13:15:00 | 55.95 | 56.34 | 56.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-24 09:15:00 | 57.70 | 56.39 | 56.57 | SL hit (close>static) qty=1.00 sl=56.75 alert=retest2 |

### Cycle 3 — BUY (started 2023-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 11:15:00 | 57.45 | 56.79 | 56.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-24 12:15:00 | 58.20 | 57.08 | 56.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 15:15:00 | 57.15 | 57.17 | 56.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-25 09:15:00 | 56.70 | 57.17 | 56.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 56.85 | 57.11 | 56.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 09:30:00 | 56.80 | 57.11 | 56.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 10:15:00 | 56.90 | 57.07 | 56.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 10:30:00 | 56.80 | 57.07 | 56.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 11:15:00 | 56.70 | 56.99 | 56.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 11:30:00 | 56.60 | 56.99 | 56.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2023-05-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 12:15:00 | 56.25 | 56.84 | 56.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-26 11:15:00 | 55.55 | 56.40 | 56.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-29 12:15:00 | 55.65 | 55.61 | 56.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-29 12:15:00 | 55.65 | 55.61 | 56.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 12:15:00 | 55.65 | 55.61 | 56.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-29 14:15:00 | 55.35 | 55.61 | 55.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-30 09:15:00 | 58.85 | 56.34 | 56.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2023-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-30 09:15:00 | 58.85 | 56.34 | 56.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 09:15:00 | 60.50 | 58.70 | 58.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-06 09:15:00 | 60.80 | 61.28 | 60.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-06 10:00:00 | 60.80 | 61.28 | 60.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 11:15:00 | 60.65 | 61.07 | 60.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 13:15:00 | 60.80 | 61.01 | 60.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 14:45:00 | 60.90 | 60.96 | 60.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-07 09:15:00 | 61.15 | 60.92 | 60.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-08 09:45:00 | 60.85 | 61.02 | 60.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 10:15:00 | 60.75 | 60.97 | 60.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 10:45:00 | 60.70 | 60.97 | 60.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-06-08 11:15:00 | 59.30 | 60.63 | 60.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 11:15:00 | 59.30 | 60.63 | 60.71 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 10:15:00 | 60.80 | 59.82 | 59.75 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-14 15:15:00 | 59.50 | 60.12 | 60.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-15 10:15:00 | 59.30 | 59.84 | 59.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 15:15:00 | 59.05 | 59.05 | 59.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-19 09:15:00 | 59.20 | 59.05 | 59.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 09:15:00 | 59.40 | 59.12 | 59.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-19 09:30:00 | 59.95 | 59.12 | 59.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 10:15:00 | 59.00 | 59.09 | 59.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-19 13:30:00 | 58.70 | 59.02 | 59.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-20 10:30:00 | 58.60 | 59.09 | 59.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-20 12:15:00 | 58.75 | 59.03 | 59.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-20 13:00:00 | 58.75 | 58.98 | 59.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 15:15:00 | 59.00 | 58.95 | 59.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:15:00 | 61.25 | 58.95 | 59.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-06-21 09:15:00 | 61.80 | 59.52 | 59.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2023-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 09:15:00 | 61.80 | 59.52 | 59.32 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-06-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 15:15:00 | 59.35 | 59.78 | 59.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 58.50 | 59.52 | 59.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 14:15:00 | 57.85 | 57.77 | 58.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 15:00:00 | 57.85 | 57.77 | 58.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 58.30 | 57.91 | 58.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:30:00 | 58.25 | 57.91 | 58.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 57.85 | 57.90 | 58.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 12:00:00 | 57.75 | 57.87 | 58.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 13:00:00 | 57.70 | 57.84 | 58.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 14:15:00 | 57.70 | 57.83 | 58.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-03 14:15:00 | 58.15 | 57.64 | 57.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2023-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 14:15:00 | 58.15 | 57.64 | 57.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-05 09:15:00 | 59.30 | 58.08 | 57.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-06 13:15:00 | 59.00 | 59.05 | 58.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-06 13:45:00 | 59.00 | 59.05 | 58.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 58.65 | 58.97 | 58.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 09:45:00 | 58.75 | 58.97 | 58.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 58.25 | 58.82 | 58.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 10:45:00 | 58.10 | 58.82 | 58.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 11:15:00 | 58.10 | 58.68 | 58.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 11:30:00 | 57.90 | 58.68 | 58.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2023-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 12:15:00 | 58.05 | 58.55 | 58.57 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 12:15:00 | 59.25 | 58.62 | 58.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-10 13:15:00 | 59.35 | 58.77 | 58.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 10:15:00 | 60.00 | 60.04 | 59.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-12 10:45:00 | 59.95 | 60.04 | 59.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 11:15:00 | 59.60 | 59.95 | 59.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 12:00:00 | 59.60 | 59.95 | 59.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 12:15:00 | 59.75 | 59.91 | 59.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-12 15:15:00 | 59.85 | 59.81 | 59.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 10:45:00 | 59.95 | 59.82 | 59.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-13 13:15:00 | 58.95 | 59.65 | 59.63 | SL hit (close<static) qty=1.00 sl=59.45 alert=retest2 |

### Cycle 14 — SELL (started 2023-07-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 14:15:00 | 58.85 | 59.49 | 59.56 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 12:15:00 | 59.65 | 59.39 | 59.36 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 11:15:00 | 58.70 | 59.23 | 59.30 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 14:15:00 | 59.40 | 59.24 | 59.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-20 09:15:00 | 59.90 | 59.38 | 59.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-25 09:15:00 | 62.25 | 62.35 | 61.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-25 10:00:00 | 62.25 | 62.35 | 61.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 11:15:00 | 63.50 | 63.81 | 63.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 11:30:00 | 63.60 | 63.81 | 63.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 13:15:00 | 63.45 | 63.68 | 63.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 13:30:00 | 63.35 | 63.68 | 63.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 14:15:00 | 63.35 | 63.61 | 63.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 14:30:00 | 63.10 | 63.61 | 63.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 15:15:00 | 63.40 | 63.57 | 63.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 09:15:00 | 62.95 | 63.57 | 63.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 09:15:00 | 63.05 | 63.47 | 63.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 10:15:00 | 62.80 | 63.47 | 63.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 10:15:00 | 63.10 | 63.39 | 63.28 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2023-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 12:15:00 | 62.70 | 63.18 | 63.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-28 13:15:00 | 62.50 | 63.05 | 63.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-28 14:15:00 | 63.35 | 63.11 | 63.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-28 14:15:00 | 63.35 | 63.11 | 63.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 14:15:00 | 63.35 | 63.11 | 63.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 15:00:00 | 63.35 | 63.11 | 63.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 15:15:00 | 63.30 | 63.15 | 63.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 09:15:00 | 63.95 | 63.15 | 63.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 09:15:00 | 63.75 | 63.27 | 63.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 09:15:00 | 65.10 | 64.03 | 63.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 11:15:00 | 65.10 | 65.12 | 64.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-02 11:30:00 | 65.25 | 65.12 | 64.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 12:15:00 | 64.45 | 64.99 | 64.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 12:45:00 | 64.45 | 64.99 | 64.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 63.55 | 64.70 | 64.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 14:00:00 | 63.55 | 64.70 | 64.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 64.00 | 64.56 | 64.41 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-08-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 09:15:00 | 63.60 | 64.21 | 64.27 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 10:15:00 | 65.20 | 64.31 | 64.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 12:15:00 | 66.15 | 64.88 | 64.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-04 14:15:00 | 61.80 | 64.39 | 64.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 14:15:00 | 61.80 | 64.39 | 64.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 14:15:00 | 61.80 | 64.39 | 64.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-04 15:00:00 | 61.80 | 64.39 | 64.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2023-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-04 15:15:00 | 61.85 | 63.88 | 64.13 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 15:15:00 | 64.65 | 64.12 | 64.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-08 09:15:00 | 69.70 | 65.23 | 64.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-09 09:15:00 | 67.40 | 67.81 | 66.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-09 09:30:00 | 67.55 | 67.81 | 66.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 10:15:00 | 66.95 | 67.32 | 66.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 10:30:00 | 66.90 | 67.32 | 66.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 11:15:00 | 66.95 | 67.25 | 66.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-10 12:30:00 | 67.45 | 67.25 | 66.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-10 13:15:00 | 66.75 | 67.15 | 66.93 | SL hit (close<static) qty=1.00 sl=66.90 alert=retest2 |

### Cycle 24 — SELL (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 09:15:00 | 65.90 | 67.10 | 67.12 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-14 15:15:00 | 67.80 | 67.03 | 67.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-16 10:15:00 | 68.55 | 67.47 | 67.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-21 09:15:00 | 72.35 | 72.40 | 71.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-21 10:00:00 | 72.35 | 72.40 | 71.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 11:15:00 | 74.10 | 74.61 | 73.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 11:30:00 | 74.35 | 74.61 | 73.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 12:15:00 | 73.60 | 74.41 | 73.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 13:00:00 | 73.60 | 74.41 | 73.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 13:15:00 | 73.60 | 74.24 | 73.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 14:15:00 | 73.35 | 74.24 | 73.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2023-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 09:15:00 | 72.60 | 73.67 | 73.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 10:15:00 | 71.75 | 73.28 | 73.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 10:15:00 | 72.15 | 72.06 | 72.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-28 10:30:00 | 72.30 | 72.06 | 72.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 13:15:00 | 71.95 | 71.98 | 72.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 14:00:00 | 71.95 | 71.98 | 72.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 73.50 | 72.23 | 72.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 10:00:00 | 73.50 | 72.23 | 72.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 10:15:00 | 73.00 | 72.38 | 72.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 11:45:00 | 72.70 | 72.44 | 72.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-29 13:15:00 | 72.75 | 72.58 | 72.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2023-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 13:15:00 | 72.75 | 72.58 | 72.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 14:15:00 | 74.55 | 72.97 | 72.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 09:15:00 | 75.20 | 75.63 | 74.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-31 10:00:00 | 75.20 | 75.63 | 74.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 75.10 | 75.53 | 74.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 10:30:00 | 74.95 | 75.53 | 74.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 14:15:00 | 75.15 | 75.24 | 74.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 14:45:00 | 74.95 | 75.24 | 74.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 15:15:00 | 74.65 | 75.12 | 74.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 09:15:00 | 74.15 | 75.12 | 74.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 09:15:00 | 74.15 | 74.93 | 74.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 09:45:00 | 74.10 | 74.93 | 74.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 10:15:00 | 74.70 | 74.88 | 74.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-01 13:00:00 | 75.25 | 74.91 | 74.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-08 12:15:00 | 77.05 | 77.46 | 77.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2023-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 12:15:00 | 77.05 | 77.46 | 77.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-11 09:15:00 | 75.40 | 76.88 | 77.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 12:15:00 | 70.75 | 70.62 | 72.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 13:00:00 | 70.75 | 70.62 | 72.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 71.75 | 70.97 | 72.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:30:00 | 72.50 | 70.97 | 72.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 10:15:00 | 71.95 | 71.17 | 72.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 10:30:00 | 71.95 | 71.17 | 72.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 11:15:00 | 72.15 | 71.36 | 72.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 12:00:00 | 72.15 | 71.36 | 72.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 12:15:00 | 71.60 | 71.41 | 72.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 13:45:00 | 71.40 | 71.40 | 71.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-15 09:15:00 | 73.80 | 71.97 | 72.07 | SL hit (close>static) qty=1.00 sl=72.20 alert=retest2 |

### Cycle 29 — BUY (started 2023-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 10:15:00 | 73.50 | 72.27 | 72.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-21 09:15:00 | 74.40 | 73.44 | 73.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 12:15:00 | 73.35 | 73.72 | 73.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 12:15:00 | 73.35 | 73.72 | 73.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 12:15:00 | 73.35 | 73.72 | 73.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 13:00:00 | 73.35 | 73.72 | 73.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 13:15:00 | 72.90 | 73.56 | 73.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 13:30:00 | 73.20 | 73.56 | 73.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 14:15:00 | 72.15 | 73.27 | 73.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 14:45:00 | 71.95 | 73.27 | 73.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2023-09-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 15:15:00 | 72.30 | 73.08 | 73.10 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 09:15:00 | 73.25 | 73.11 | 73.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-22 10:15:00 | 77.25 | 73.94 | 73.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 11:15:00 | 82.50 | 82.59 | 80.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-26 11:45:00 | 82.65 | 82.59 | 80.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 89.75 | 90.60 | 89.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 09:30:00 | 89.90 | 90.60 | 89.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 10:15:00 | 95.55 | 91.59 | 89.92 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 12:15:00 | 88.45 | 90.12 | 90.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 14:15:00 | 87.70 | 89.34 | 89.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 89.80 | 89.10 | 89.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 89.80 | 89.10 | 89.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 89.80 | 89.10 | 89.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 09:45:00 | 89.85 | 89.10 | 89.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 89.25 | 89.13 | 89.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 11:15:00 | 91.00 | 89.13 | 89.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 11:15:00 | 90.05 | 89.32 | 89.69 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 92.15 | 90.17 | 89.98 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 15:15:00 | 90.40 | 90.86 | 90.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-16 14:15:00 | 89.65 | 90.25 | 90.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 91.30 | 90.34 | 90.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 09:15:00 | 91.30 | 90.34 | 90.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 91.30 | 90.34 | 90.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 11:30:00 | 90.30 | 90.37 | 90.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 09:15:00 | 85.78 | 88.54 | 89.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2023-10-18 11:15:00 | 81.27 | 85.98 | 88.18 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 35 — BUY (started 2023-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 11:15:00 | 76.40 | 74.74 | 74.70 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-11-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 10:15:00 | 75.00 | 75.37 | 75.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 12:15:00 | 74.50 | 75.14 | 75.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 09:15:00 | 75.00 | 74.77 | 75.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 09:15:00 | 75.00 | 74.77 | 75.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 75.00 | 74.77 | 75.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 09:45:00 | 75.25 | 74.77 | 75.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 10:15:00 | 75.35 | 74.89 | 75.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:00:00 | 75.35 | 74.89 | 75.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 75.45 | 75.00 | 75.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:45:00 | 75.60 | 75.00 | 75.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2023-11-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 14:15:00 | 75.45 | 75.19 | 75.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 09:15:00 | 76.30 | 75.44 | 75.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 14:15:00 | 75.65 | 75.89 | 75.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 14:15:00 | 75.65 | 75.89 | 75.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 14:15:00 | 75.65 | 75.89 | 75.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 15:00:00 | 75.65 | 75.89 | 75.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 15:15:00 | 76.00 | 75.91 | 75.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 09:15:00 | 79.10 | 75.91 | 75.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-10 10:15:00 | 78.60 | 79.51 | 79.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2023-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 10:15:00 | 78.60 | 79.51 | 79.56 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 15:15:00 | 80.20 | 79.57 | 79.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-12 18:15:00 | 80.95 | 79.84 | 79.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 13:15:00 | 81.75 | 81.78 | 81.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-16 13:45:00 | 81.70 | 81.78 | 81.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 15:15:00 | 81.45 | 81.66 | 81.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 09:15:00 | 81.25 | 81.66 | 81.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 80.70 | 81.47 | 81.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 10:00:00 | 80.70 | 81.47 | 81.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 10:15:00 | 80.70 | 81.31 | 81.21 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 13:15:00 | 80.65 | 81.06 | 81.12 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 09:15:00 | 82.60 | 81.27 | 81.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 14:15:00 | 83.40 | 82.34 | 81.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 11:15:00 | 82.65 | 82.66 | 82.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-22 11:45:00 | 82.85 | 82.66 | 82.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 12:15:00 | 82.40 | 82.61 | 82.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-22 14:15:00 | 82.80 | 82.58 | 82.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-24 09:15:00 | 81.95 | 82.32 | 82.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2023-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 09:15:00 | 81.95 | 82.32 | 82.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 15:15:00 | 81.50 | 81.77 | 81.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 09:15:00 | 82.25 | 81.87 | 81.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 09:15:00 | 82.25 | 81.87 | 81.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 82.25 | 81.87 | 81.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 13:45:00 | 81.50 | 81.80 | 81.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 14:45:00 | 81.55 | 81.73 | 81.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-30 11:15:00 | 85.10 | 82.48 | 82.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2023-11-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 11:15:00 | 85.10 | 82.48 | 82.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 12:15:00 | 86.90 | 83.36 | 82.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 14:15:00 | 85.65 | 85.74 | 84.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-01 15:00:00 | 85.65 | 85.74 | 84.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 11:15:00 | 86.40 | 87.33 | 86.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 11:45:00 | 86.10 | 87.33 | 86.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 12:15:00 | 87.25 | 87.31 | 86.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 13:15:00 | 87.90 | 87.31 | 86.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 14:45:00 | 87.80 | 87.46 | 86.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 15:15:00 | 87.50 | 87.46 | 86.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-06 10:15:00 | 87.70 | 87.42 | 86.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 12:15:00 | 86.65 | 87.29 | 86.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 12:45:00 | 86.95 | 87.29 | 86.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 13:15:00 | 87.00 | 87.23 | 86.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 13:30:00 | 86.80 | 87.23 | 86.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 14:15:00 | 86.70 | 87.13 | 86.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 15:00:00 | 86.70 | 87.13 | 86.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 15:15:00 | 87.00 | 87.10 | 86.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 09:15:00 | 90.05 | 87.10 | 86.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-12 09:15:00 | 96.25 | 91.97 | 91.17 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 103.05 | 108.38 | 109.10 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 09:15:00 | 110.40 | 106.37 | 106.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 14:15:00 | 112.80 | 109.73 | 107.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 09:15:00 | 125.45 | 126.77 | 120.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-29 10:00:00 | 125.45 | 126.77 | 120.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 09:15:00 | 132.75 | 127.32 | 123.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 09:30:00 | 124.90 | 127.32 | 123.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 126.40 | 128.01 | 126.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:30:00 | 126.10 | 128.01 | 126.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 11:15:00 | 126.60 | 127.72 | 126.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 12:30:00 | 127.40 | 127.47 | 126.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 14:00:00 | 127.90 | 127.56 | 126.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 15:00:00 | 127.25 | 127.49 | 126.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 09:15:00 | 128.00 | 127.42 | 126.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 126.85 | 127.30 | 126.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 10:30:00 | 128.85 | 127.57 | 126.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 09:15:00 | 131.90 | 127.24 | 126.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 12:45:00 | 128.80 | 128.30 | 127.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 14:45:00 | 128.65 | 128.46 | 127.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 10:15:00 | 127.05 | 128.13 | 127.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 10:45:00 | 127.10 | 128.13 | 127.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 11:15:00 | 127.00 | 127.90 | 127.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 11:45:00 | 127.00 | 127.90 | 127.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-01-05 13:15:00 | 126.00 | 127.37 | 127.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 13:15:00 | 126.00 | 127.37 | 127.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 12:15:00 | 125.30 | 126.49 | 127.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 11:15:00 | 126.15 | 125.38 | 126.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 11:15:00 | 126.15 | 125.38 | 126.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 11:15:00 | 126.15 | 125.38 | 126.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 11:45:00 | 127.05 | 125.38 | 126.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 12:15:00 | 125.90 | 125.48 | 126.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 13:00:00 | 125.90 | 125.48 | 126.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 13:15:00 | 125.75 | 125.54 | 126.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 13:30:00 | 126.45 | 125.54 | 126.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 14:15:00 | 125.65 | 125.56 | 126.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 15:00:00 | 125.65 | 125.56 | 126.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 15:15:00 | 125.35 | 125.52 | 125.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 09:15:00 | 123.75 | 125.52 | 125.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-11 09:15:00 | 130.35 | 125.47 | 125.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2024-01-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 09:15:00 | 130.35 | 125.47 | 125.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-15 09:15:00 | 133.15 | 127.98 | 127.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 12:15:00 | 128.90 | 132.45 | 130.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 12:15:00 | 128.90 | 132.45 | 130.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 128.90 | 132.45 | 130.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:00:00 | 128.90 | 132.45 | 130.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 13:15:00 | 131.50 | 132.26 | 131.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 09:45:00 | 134.35 | 132.97 | 131.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 10:00:00 | 133.75 | 135.43 | 133.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-19 09:15:00 | 147.78 | 140.53 | 137.20 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 15:15:00 | 149.90 | 154.85 | 154.89 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 13:15:00 | 161.65 | 155.56 | 155.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 14:15:00 | 166.00 | 157.65 | 156.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 12:15:00 | 168.00 | 168.01 | 165.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-30 12:30:00 | 168.25 | 168.01 | 165.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 164.65 | 167.09 | 165.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 14:30:00 | 164.25 | 167.09 | 165.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 15:15:00 | 164.35 | 166.54 | 165.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 09:30:00 | 166.45 | 166.34 | 165.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 10:00:00 | 165.55 | 166.34 | 165.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-01 09:15:00 | 183.09 | 173.00 | 169.34 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2024-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 13:15:00 | 198.20 | 200.22 | 200.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 09:15:00 | 196.70 | 198.85 | 199.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-08 13:15:00 | 199.30 | 198.11 | 198.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 13:15:00 | 199.30 | 198.11 | 198.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 13:15:00 | 199.30 | 198.11 | 198.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 14:00:00 | 199.30 | 198.11 | 198.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 14:15:00 | 197.60 | 198.01 | 198.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-09 09:15:00 | 193.40 | 197.92 | 198.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 09:15:00 | 183.73 | 196.09 | 197.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-09 14:15:00 | 202.85 | 192.97 | 195.00 | SL hit (close>ema200) qty=0.50 sl=192.97 alert=retest2 |

### Cycle 51 — BUY (started 2024-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 12:15:00 | 192.75 | 186.63 | 185.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 13:15:00 | 197.45 | 188.79 | 187.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 09:15:00 | 200.05 | 200.55 | 196.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-16 09:45:00 | 200.05 | 200.55 | 196.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 195.35 | 198.69 | 197.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 09:45:00 | 195.45 | 198.69 | 197.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 10:15:00 | 196.10 | 198.18 | 197.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 11:15:00 | 195.55 | 198.18 | 197.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 11:15:00 | 195.20 | 197.58 | 196.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 11:30:00 | 195.20 | 197.58 | 196.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2024-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 14:15:00 | 192.45 | 195.86 | 196.28 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-20 14:15:00 | 198.80 | 196.43 | 196.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-21 09:15:00 | 201.15 | 197.65 | 196.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-21 13:15:00 | 198.15 | 199.49 | 198.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 13:15:00 | 198.15 | 199.49 | 198.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 13:15:00 | 198.15 | 199.49 | 198.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 14:00:00 | 198.15 | 199.49 | 198.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 14:15:00 | 196.05 | 198.80 | 198.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 15:00:00 | 196.05 | 198.80 | 198.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 15:15:00 | 196.15 | 198.27 | 197.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 09:15:00 | 192.50 | 198.27 | 197.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2024-02-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 10:15:00 | 194.80 | 197.24 | 197.45 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 10:15:00 | 200.50 | 197.66 | 197.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 11:15:00 | 201.70 | 198.47 | 197.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-23 14:15:00 | 199.35 | 199.56 | 198.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-23 15:00:00 | 199.35 | 199.56 | 198.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 15:15:00 | 198.95 | 199.44 | 198.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 09:15:00 | 209.45 | 199.44 | 198.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 205.45 | 200.64 | 199.20 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 09:15:00 | 194.20 | 200.71 | 201.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 10:15:00 | 192.05 | 198.98 | 200.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 12:15:00 | 192.95 | 191.40 | 194.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 13:00:00 | 192.95 | 191.40 | 194.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 195.65 | 192.42 | 194.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:30:00 | 196.85 | 192.42 | 194.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 10:15:00 | 194.85 | 192.90 | 194.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 11:45:00 | 193.85 | 193.12 | 194.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 15:15:00 | 193.90 | 193.62 | 194.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-02 11:15:00 | 196.35 | 194.29 | 194.31 | SL hit (close>static) qty=1.00 sl=195.65 alert=retest2 |

### Cycle 57 — BUY (started 2024-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 12:15:00 | 195.15 | 194.46 | 194.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-04 09:15:00 | 198.50 | 195.27 | 194.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 14:15:00 | 195.45 | 195.95 | 195.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 14:15:00 | 195.45 | 195.95 | 195.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 14:15:00 | 195.45 | 195.95 | 195.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 14:30:00 | 196.90 | 195.95 | 195.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 15:15:00 | 195.30 | 195.82 | 195.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 09:15:00 | 200.85 | 195.82 | 195.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 10:00:00 | 196.80 | 199.27 | 198.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 10:30:00 | 197.55 | 198.46 | 197.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-06 11:15:00 | 192.90 | 197.35 | 197.33 | SL hit (close<static) qty=1.00 sl=194.50 alert=retest2 |

### Cycle 58 — SELL (started 2024-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 12:15:00 | 194.45 | 196.77 | 197.07 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 09:15:00 | 200.30 | 197.34 | 197.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-07 13:15:00 | 201.10 | 198.34 | 197.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 09:15:00 | 198.40 | 199.43 | 198.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 09:15:00 | 198.40 | 199.43 | 198.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 198.40 | 199.43 | 198.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 10:00:00 | 198.40 | 199.43 | 198.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 10:15:00 | 197.40 | 199.02 | 198.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 10:45:00 | 197.45 | 199.02 | 198.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 11:15:00 | 196.00 | 198.42 | 198.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 12:00:00 | 196.00 | 198.42 | 198.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 12:15:00 | 198.60 | 198.45 | 198.21 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 14:15:00 | 195.05 | 197.52 | 197.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 09:15:00 | 190.15 | 195.77 | 196.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 171.60 | 170.69 | 178.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 10:00:00 | 171.60 | 170.69 | 178.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 177.40 | 172.82 | 178.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 12:00:00 | 177.40 | 172.82 | 178.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 12:15:00 | 174.55 | 173.16 | 178.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 13:15:00 | 177.80 | 173.16 | 178.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 13:15:00 | 177.75 | 174.08 | 178.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 13:45:00 | 179.20 | 174.08 | 178.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 180.55 | 175.37 | 178.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 14:30:00 | 179.40 | 175.37 | 178.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 15:15:00 | 180.55 | 176.41 | 178.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 09:15:00 | 180.60 | 176.41 | 178.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 13:15:00 | 183.85 | 176.37 | 177.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 14:00:00 | 183.85 | 176.37 | 177.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2024-03-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 14:15:00 | 190.80 | 179.26 | 178.63 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 15:15:00 | 177.00 | 179.94 | 180.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 09:15:00 | 173.70 | 178.69 | 179.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 12:15:00 | 177.80 | 177.53 | 178.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-20 12:45:00 | 176.55 | 177.53 | 178.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 13:15:00 | 175.80 | 177.18 | 178.56 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 12:15:00 | 181.80 | 179.34 | 179.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 13:15:00 | 184.80 | 180.43 | 179.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 182.60 | 183.50 | 181.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-22 15:00:00 | 182.60 | 183.50 | 181.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 184.70 | 183.62 | 182.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:30:00 | 180.85 | 183.62 | 182.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 14:15:00 | 187.05 | 188.89 | 187.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 15:00:00 | 187.05 | 188.89 | 187.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 15:15:00 | 186.90 | 188.49 | 187.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 09:15:00 | 189.25 | 188.49 | 187.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-04 14:15:00 | 208.18 | 201.77 | 199.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-04-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 12:15:00 | 205.95 | 208.72 | 209.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 14:15:00 | 204.35 | 207.33 | 208.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 09:15:00 | 198.60 | 196.99 | 199.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 09:15:00 | 198.60 | 196.99 | 199.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 198.60 | 196.99 | 199.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:30:00 | 199.40 | 196.99 | 199.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 198.25 | 197.24 | 199.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 10:45:00 | 198.50 | 197.24 | 199.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 12:15:00 | 195.90 | 195.18 | 196.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 14:30:00 | 194.45 | 195.15 | 196.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 10:15:00 | 198.80 | 195.88 | 196.51 | SL hit (close>static) qty=1.00 sl=197.00 alert=retest2 |

### Cycle 65 — BUY (started 2024-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 15:15:00 | 197.40 | 196.40 | 196.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 09:15:00 | 201.50 | 197.42 | 196.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 14:15:00 | 198.85 | 199.54 | 198.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-24 14:15:00 | 198.85 | 199.54 | 198.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 14:15:00 | 198.85 | 199.54 | 198.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-24 15:00:00 | 198.85 | 199.54 | 198.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 15:15:00 | 198.80 | 199.40 | 198.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 09:15:00 | 202.00 | 199.40 | 198.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-26 10:15:00 | 222.20 | 206.21 | 202.71 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-05-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 10:15:00 | 221.10 | 222.98 | 223.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 11:15:00 | 218.55 | 222.10 | 222.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 215.75 | 210.28 | 212.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 09:15:00 | 215.75 | 210.28 | 212.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 215.75 | 210.28 | 212.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 10:00:00 | 215.75 | 210.28 | 212.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 216.25 | 211.48 | 213.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 11:00:00 | 216.25 | 211.48 | 213.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2024-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 11:15:00 | 226.60 | 214.50 | 214.41 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-05-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 09:15:00 | 216.20 | 217.08 | 217.12 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 226.30 | 217.35 | 216.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 233.60 | 224.80 | 220.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 14:15:00 | 229.50 | 230.12 | 226.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 15:00:00 | 229.50 | 230.12 | 226.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 256.85 | 265.96 | 264.53 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 09:15:00 | 257.20 | 262.76 | 263.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 11:15:00 | 253.35 | 259.58 | 261.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 255.80 | 254.69 | 257.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 10:00:00 | 255.80 | 254.69 | 257.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 257.35 | 255.22 | 257.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 11:00:00 | 257.35 | 255.22 | 257.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 258.20 | 255.82 | 257.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 11:30:00 | 258.20 | 255.82 | 257.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 269.00 | 258.45 | 258.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:00:00 | 269.00 | 258.45 | 258.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 13:15:00 | 267.25 | 260.21 | 259.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 13:15:00 | 272.40 | 266.51 | 264.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 269.65 | 281.82 | 276.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 269.65 | 281.82 | 276.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 269.65 | 281.82 | 276.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 262.90 | 281.82 | 276.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 248.75 | 275.21 | 274.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 248.75 | 275.21 | 274.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 229.70 | 266.11 | 269.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 214.45 | 240.76 | 254.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 253.85 | 238.76 | 246.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 253.85 | 238.76 | 246.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 253.85 | 238.76 | 246.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 255.60 | 238.76 | 246.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 258.30 | 242.67 | 247.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 258.30 | 242.67 | 247.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 248.60 | 245.54 | 247.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:30:00 | 250.70 | 245.54 | 247.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 15:15:00 | 249.35 | 246.99 | 247.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:15:00 | 252.05 | 246.99 | 247.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 250.80 | 248.04 | 248.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 10:30:00 | 251.20 | 248.04 | 248.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 11:15:00 | 249.20 | 248.27 | 248.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-07 12:15:00 | 247.20 | 248.27 | 248.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-07 13:00:00 | 247.90 | 248.20 | 248.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 13:15:00 | 249.95 | 248.55 | 248.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2024-06-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 13:15:00 | 249.95 | 248.55 | 248.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 12:15:00 | 255.40 | 250.24 | 249.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 12:15:00 | 278.00 | 278.26 | 271.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 12:30:00 | 277.75 | 278.26 | 271.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 285.15 | 280.48 | 277.11 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 274.60 | 278.81 | 279.21 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 13:15:00 | 285.65 | 279.55 | 279.39 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 09:15:00 | 277.95 | 281.65 | 281.89 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 12:15:00 | 284.75 | 279.24 | 278.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 09:15:00 | 289.75 | 283.96 | 281.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 12:15:00 | 281.85 | 284.02 | 282.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 12:15:00 | 281.85 | 284.02 | 282.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 281.85 | 284.02 | 282.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:45:00 | 282.35 | 284.02 | 282.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 280.85 | 283.39 | 282.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 280.85 | 283.39 | 282.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 283.80 | 283.47 | 282.21 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-06-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 12:15:00 | 280.30 | 281.83 | 281.85 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 284.40 | 282.15 | 281.94 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 14:15:00 | 280.70 | 281.85 | 281.88 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 284.45 | 282.36 | 282.10 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 278.80 | 281.51 | 281.78 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 10:15:00 | 293.40 | 283.73 | 282.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 14:15:00 | 301.80 | 291.08 | 286.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 14:15:00 | 332.80 | 334.53 | 326.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 15:00:00 | 332.80 | 334.53 | 326.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 332.80 | 334.37 | 328.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 13:30:00 | 338.20 | 332.26 | 330.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 10:45:00 | 336.85 | 335.23 | 334.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 12:15:00 | 332.45 | 334.26 | 334.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 12:15:00 | 332.45 | 334.26 | 334.30 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 14:15:00 | 336.50 | 334.47 | 334.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 340.80 | 335.96 | 335.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 11:15:00 | 337.70 | 337.78 | 336.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 12:00:00 | 337.70 | 337.78 | 336.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 336.40 | 337.50 | 336.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 13:00:00 | 336.40 | 337.50 | 336.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 336.05 | 337.21 | 336.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:00:00 | 336.05 | 337.21 | 336.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 335.50 | 336.87 | 336.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:45:00 | 335.35 | 336.87 | 336.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 335.65 | 336.63 | 336.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 327.45 | 336.63 | 336.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 328.30 | 334.96 | 335.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 313.00 | 324.74 | 329.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 327.85 | 320.39 | 323.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 327.85 | 320.39 | 323.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 327.85 | 320.39 | 323.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 327.85 | 320.39 | 323.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 318.75 | 320.06 | 323.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 325.40 | 320.06 | 323.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 324.55 | 321.02 | 323.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:45:00 | 323.30 | 321.02 | 323.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 323.25 | 321.47 | 323.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:45:00 | 321.25 | 321.88 | 323.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:15:00 | 318.70 | 322.02 | 323.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:45:00 | 318.80 | 321.25 | 322.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-23 12:15:00 | 289.12 | 317.94 | 320.97 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 11:15:00 | 321.30 | 315.61 | 315.17 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 09:15:00 | 314.90 | 316.08 | 316.22 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 09:15:00 | 318.15 | 316.54 | 316.32 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 15:15:00 | 315.50 | 316.35 | 316.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 09:15:00 | 313.60 | 315.80 | 316.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 10:15:00 | 308.35 | 307.83 | 310.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 10:15:00 | 308.35 | 307.83 | 310.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 308.35 | 307.83 | 310.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:45:00 | 309.85 | 307.83 | 310.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 13:15:00 | 306.85 | 307.45 | 309.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 13:45:00 | 309.00 | 307.45 | 309.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 298.80 | 295.06 | 300.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 292.40 | 294.89 | 298.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 294.25 | 294.22 | 295.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:45:00 | 293.45 | 293.70 | 294.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:30:00 | 294.20 | 292.74 | 293.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 12:15:00 | 292.10 | 292.61 | 293.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 12:30:00 | 293.25 | 292.61 | 293.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 293.60 | 292.81 | 293.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:00:00 | 293.60 | 292.81 | 293.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 292.10 | 292.67 | 293.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:30:00 | 293.60 | 292.67 | 293.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-12 09:15:00 | 307.50 | 295.55 | 294.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 09:15:00 | 307.50 | 295.55 | 294.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 10:15:00 | 310.75 | 298.59 | 295.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 14:15:00 | 299.80 | 300.73 | 297.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 15:00:00 | 299.80 | 300.73 | 297.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 298.00 | 300.41 | 298.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:45:00 | 297.80 | 300.41 | 298.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 298.00 | 299.93 | 298.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 298.00 | 299.93 | 298.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 14:15:00 | 286.55 | 297.26 | 297.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 14:15:00 | 284.40 | 287.91 | 289.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 288.20 | 287.38 | 288.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 288.20 | 287.38 | 288.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 288.20 | 287.38 | 288.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:30:00 | 287.50 | 287.38 | 288.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 289.00 | 286.56 | 287.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 09:30:00 | 290.80 | 286.56 | 287.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 289.25 | 287.10 | 287.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 10:30:00 | 290.10 | 287.10 | 287.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 285.90 | 286.94 | 287.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 12:30:00 | 287.30 | 286.94 | 287.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 285.25 | 286.22 | 286.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 13:45:00 | 283.80 | 285.01 | 286.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 09:30:00 | 283.70 | 284.20 | 285.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 11:15:00 | 287.30 | 281.48 | 281.55 | SL hit (close>static) qty=1.00 sl=287.10 alert=retest2 |

### Cycle 93 — BUY (started 2024-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 12:15:00 | 296.20 | 284.42 | 282.88 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 15:15:00 | 278.40 | 285.55 | 286.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 09:15:00 | 275.20 | 283.48 | 285.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 15:15:00 | 252.40 | 252.25 | 256.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 09:15:00 | 255.10 | 252.25 | 256.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 251.50 | 252.10 | 255.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 11:45:00 | 250.70 | 252.02 | 253.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 12:45:00 | 249.65 | 251.46 | 253.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 11:15:00 | 250.65 | 249.16 | 251.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 11:45:00 | 250.35 | 249.99 | 251.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 255.50 | 251.09 | 251.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:00:00 | 255.50 | 251.09 | 251.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-12 15:15:00 | 254.20 | 252.45 | 252.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 254.20 | 252.45 | 252.31 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 09:15:00 | 249.95 | 252.01 | 252.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 10:15:00 | 249.60 | 251.53 | 251.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 13:15:00 | 249.70 | 246.43 | 248.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 13:15:00 | 249.70 | 246.43 | 248.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 249.70 | 246.43 | 248.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:00:00 | 249.70 | 246.43 | 248.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 244.85 | 246.11 | 247.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 11:00:00 | 244.40 | 245.51 | 247.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:15:00 | 232.18 | 239.16 | 242.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-20 09:15:00 | 233.15 | 232.96 | 237.55 | SL hit (close>ema200) qty=0.50 sl=232.96 alert=retest2 |

### Cycle 97 — BUY (started 2024-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 12:15:00 | 249.10 | 239.93 | 239.89 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 15:15:00 | 242.20 | 243.08 | 243.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 237.60 | 241.98 | 242.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 10:15:00 | 238.30 | 237.75 | 239.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 10:15:00 | 238.30 | 237.75 | 239.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 238.30 | 237.75 | 239.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:30:00 | 241.40 | 237.75 | 239.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 233.95 | 236.99 | 239.14 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 11:15:00 | 240.70 | 237.95 | 237.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 14:15:00 | 242.00 | 239.56 | 238.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 09:15:00 | 238.81 | 239.45 | 238.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 238.81 | 239.45 | 238.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 238.81 | 239.45 | 238.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:15:00 | 238.15 | 239.45 | 238.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 238.22 | 239.21 | 238.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:30:00 | 238.06 | 239.21 | 238.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 235.46 | 238.46 | 238.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:00:00 | 235.46 | 238.46 | 238.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2024-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 12:15:00 | 235.36 | 237.84 | 238.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 230.96 | 235.59 | 236.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 215.85 | 215.03 | 220.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:45:00 | 217.29 | 215.03 | 220.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 221.51 | 216.32 | 220.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 221.51 | 216.32 | 220.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 221.57 | 217.37 | 220.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:30:00 | 222.73 | 217.37 | 220.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 224.82 | 218.86 | 221.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 224.82 | 218.86 | 221.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 225.93 | 220.28 | 221.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:30:00 | 225.15 | 220.28 | 221.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 230.25 | 223.62 | 223.01 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 10:15:00 | 222.47 | 225.09 | 225.20 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 12:15:00 | 225.57 | 224.21 | 224.20 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 14:15:00 | 223.48 | 224.24 | 224.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 221.74 | 223.65 | 224.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 203.81 | 202.76 | 206.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 11:00:00 | 203.81 | 202.76 | 206.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 204.65 | 202.29 | 204.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:30:00 | 205.30 | 202.29 | 204.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 11:15:00 | 205.51 | 202.94 | 204.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 11:45:00 | 205.49 | 202.94 | 204.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 12:15:00 | 207.46 | 203.84 | 204.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 12:30:00 | 208.00 | 203.84 | 204.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 197.06 | 202.73 | 204.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 10:15:00 | 196.67 | 202.73 | 204.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 09:30:00 | 195.14 | 196.25 | 199.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 15:15:00 | 204.10 | 200.69 | 200.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2024-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 15:15:00 | 204.10 | 200.69 | 200.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 11:15:00 | 205.15 | 202.38 | 201.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 12:15:00 | 215.95 | 216.21 | 212.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 12:45:00 | 215.98 | 216.21 | 212.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 214.81 | 216.64 | 214.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 214.81 | 216.64 | 214.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 214.64 | 216.24 | 214.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:30:00 | 214.38 | 216.24 | 214.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 12:15:00 | 215.90 | 216.17 | 214.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:15:00 | 217.47 | 216.17 | 214.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:45:00 | 216.23 | 217.75 | 216.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 11:45:00 | 216.02 | 217.26 | 216.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 13:15:00 | 220.50 | 223.44 | 223.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 220.50 | 223.44 | 223.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 219.29 | 222.61 | 223.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 221.54 | 221.10 | 222.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 221.54 | 221.10 | 222.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 221.54 | 221.10 | 222.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 10:45:00 | 222.85 | 221.10 | 222.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 218.65 | 220.61 | 221.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 12:45:00 | 217.34 | 219.69 | 221.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 09:30:00 | 216.98 | 217.30 | 219.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 206.47 | 211.78 | 215.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 206.13 | 211.78 | 215.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 207.07 | 205.48 | 209.72 | SL hit (close>ema200) qty=0.50 sl=205.48 alert=retest2 |

### Cycle 107 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 212.21 | 207.35 | 207.05 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 200.00 | 206.10 | 206.78 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 217.61 | 207.39 | 206.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 220.53 | 210.02 | 207.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 11:15:00 | 213.75 | 214.94 | 212.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 12:00:00 | 213.75 | 214.94 | 212.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 13:15:00 | 212.71 | 214.33 | 212.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 09:45:00 | 216.26 | 214.16 | 212.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-28 09:15:00 | 237.89 | 222.78 | 218.00 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 12:15:00 | 248.00 | 249.34 | 249.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 240.70 | 246.92 | 248.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 245.42 | 245.11 | 246.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 15:00:00 | 245.42 | 245.11 | 246.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 249.33 | 245.87 | 246.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:00:00 | 249.33 | 245.87 | 246.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 250.01 | 246.70 | 247.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:00:00 | 250.01 | 246.70 | 247.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 251.32 | 247.88 | 247.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 13:15:00 | 255.55 | 249.41 | 248.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 15:15:00 | 256.50 | 256.72 | 253.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-18 09:15:00 | 253.30 | 256.72 | 253.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 252.02 | 255.78 | 253.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:00:00 | 252.02 | 255.78 | 253.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 252.48 | 255.12 | 253.68 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2024-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 13:15:00 | 249.55 | 252.41 | 252.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 246.00 | 249.95 | 251.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 09:15:00 | 247.39 | 246.74 | 248.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 247.39 | 246.74 | 248.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 247.39 | 246.74 | 248.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:30:00 | 248.88 | 246.74 | 248.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 248.70 | 247.13 | 248.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:00:00 | 248.70 | 247.13 | 248.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 247.20 | 247.14 | 248.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:15:00 | 246.62 | 247.14 | 248.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 09:15:00 | 234.29 | 238.73 | 242.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-27 09:15:00 | 229.73 | 229.29 | 232.85 | SL hit (close>ema200) qty=0.50 sl=229.29 alert=retest2 |

### Cycle 113 — BUY (started 2024-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 12:15:00 | 233.10 | 230.33 | 230.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 13:15:00 | 235.85 | 231.43 | 230.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 244.97 | 245.69 | 242.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 15:00:00 | 244.97 | 245.69 | 242.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 250.93 | 247.12 | 243.48 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 239.10 | 242.43 | 242.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 10:15:00 | 237.56 | 241.42 | 242.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 12:15:00 | 242.10 | 241.35 | 241.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 12:15:00 | 242.10 | 241.35 | 241.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 242.10 | 241.35 | 241.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 12:45:00 | 240.70 | 241.35 | 241.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 242.14 | 241.50 | 241.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:45:00 | 241.10 | 241.50 | 241.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 239.50 | 241.10 | 241.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 15:15:00 | 238.76 | 241.10 | 241.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 14:15:00 | 226.82 | 231.96 | 235.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-10 14:15:00 | 214.88 | 221.11 | 227.37 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 115 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 223.42 | 216.38 | 215.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 13:15:00 | 226.91 | 221.66 | 218.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 232.47 | 233.80 | 230.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 232.47 | 233.80 | 230.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 229.11 | 232.86 | 230.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 229.11 | 232.86 | 230.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 231.80 | 232.65 | 230.53 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 223.20 | 229.76 | 229.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 219.38 | 226.20 | 228.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 229.20 | 224.90 | 226.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 229.20 | 224.90 | 226.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 229.20 | 224.90 | 226.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 14:45:00 | 226.10 | 224.90 | 226.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 228.00 | 225.52 | 227.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 226.47 | 225.52 | 227.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 225.25 | 225.46 | 226.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:15:00 | 218.64 | 225.12 | 226.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 12:15:00 | 219.44 | 224.87 | 226.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 13:30:00 | 221.34 | 223.22 | 224.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 207.71 | 219.06 | 221.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 208.47 | 219.06 | 221.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 210.27 | 219.06 | 221.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 212.83 | 210.13 | 214.09 | SL hit (close>ema200) qty=0.50 sl=210.13 alert=retest2 |

### Cycle 117 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 218.71 | 214.68 | 214.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 220.89 | 217.10 | 215.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 222.83 | 227.85 | 223.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 222.83 | 227.85 | 223.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 222.83 | 227.85 | 223.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 225.00 | 227.85 | 223.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 219.03 | 226.09 | 223.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 221.07 | 226.09 | 223.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 222.22 | 225.31 | 223.02 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 199.20 | 217.50 | 219.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 10:15:00 | 197.65 | 213.53 | 217.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 12:15:00 | 204.48 | 203.27 | 208.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 12:30:00 | 205.00 | 203.27 | 208.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 210.75 | 205.28 | 207.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:30:00 | 209.94 | 205.28 | 207.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 210.50 | 206.32 | 207.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:45:00 | 211.54 | 206.32 | 207.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 13:15:00 | 211.45 | 208.81 | 208.70 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 11:15:00 | 206.97 | 208.51 | 208.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 204.72 | 207.46 | 208.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 09:15:00 | 207.40 | 206.97 | 207.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 09:15:00 | 207.40 | 206.97 | 207.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 207.40 | 206.97 | 207.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 207.40 | 206.97 | 207.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 208.94 | 207.36 | 207.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:00:00 | 208.94 | 207.36 | 207.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 205.59 | 207.01 | 207.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:30:00 | 208.89 | 207.01 | 207.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 207.50 | 207.11 | 207.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 12:45:00 | 207.99 | 207.11 | 207.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 205.44 | 206.77 | 207.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 13:30:00 | 207.37 | 206.77 | 207.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 206.83 | 206.78 | 207.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 15:00:00 | 206.83 | 206.78 | 207.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 201.78 | 205.70 | 206.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 11:00:00 | 201.11 | 204.78 | 206.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 191.05 | 192.82 | 197.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-12 12:15:00 | 194.73 | 192.97 | 196.29 | SL hit (close>ema200) qty=0.50 sl=192.97 alert=retest2 |

### Cycle 121 — BUY (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 09:15:00 | 184.39 | 183.22 | 183.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 187.44 | 184.07 | 183.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 15:15:00 | 188.30 | 189.44 | 187.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 09:15:00 | 184.97 | 189.44 | 187.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 185.14 | 188.58 | 187.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:30:00 | 183.85 | 188.58 | 187.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 12:15:00 | 184.22 | 186.47 | 186.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 182.09 | 185.59 | 186.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 166.00 | 164.89 | 169.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 12:45:00 | 164.27 | 164.89 | 169.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 168.65 | 166.32 | 168.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 168.65 | 166.32 | 168.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 168.76 | 166.81 | 168.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:45:00 | 167.98 | 166.90 | 168.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:45:00 | 167.61 | 167.03 | 168.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 178.05 | 169.93 | 169.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 178.05 | 169.93 | 169.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 182.49 | 176.87 | 173.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 182.08 | 182.30 | 179.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 182.08 | 182.30 | 179.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 180.21 | 181.65 | 180.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:00:00 | 180.21 | 181.65 | 180.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 179.37 | 181.20 | 180.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:15:00 | 178.10 | 181.20 | 180.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 182.20 | 180.89 | 180.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:30:00 | 180.66 | 180.89 | 180.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 181.00 | 180.91 | 180.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:45:00 | 180.90 | 180.91 | 180.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 179.32 | 180.59 | 180.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:00:00 | 179.32 | 180.59 | 180.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 180.77 | 180.63 | 180.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:30:00 | 179.80 | 180.63 | 180.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 179.76 | 180.86 | 180.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:45:00 | 179.77 | 180.86 | 180.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 11:15:00 | 178.85 | 180.46 | 180.49 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 14:15:00 | 181.16 | 180.59 | 180.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 09:15:00 | 182.60 | 181.16 | 180.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 13:15:00 | 181.90 | 181.93 | 181.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-13 14:00:00 | 181.90 | 181.93 | 181.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 181.13 | 181.77 | 181.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:00:00 | 181.13 | 181.77 | 181.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 180.66 | 181.55 | 181.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 186.80 | 181.55 | 181.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-24 09:15:00 | 205.48 | 202.42 | 199.35 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 201.10 | 202.98 | 203.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 200.10 | 202.41 | 202.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 200.84 | 200.46 | 201.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 10:00:00 | 200.84 | 200.46 | 201.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 203.07 | 200.98 | 201.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:45:00 | 203.19 | 200.98 | 201.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 202.85 | 201.35 | 201.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 13:30:00 | 201.91 | 201.56 | 201.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 10:00:00 | 202.00 | 201.42 | 201.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:00:00 | 202.00 | 201.53 | 201.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:00:00 | 201.15 | 201.68 | 201.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 199.51 | 200.95 | 201.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:30:00 | 201.82 | 200.95 | 201.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 203.79 | 201.18 | 201.42 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-01 09:15:00 | 203.79 | 201.18 | 201.42 | SL hit (close>static) qty=1.00 sl=203.64 alert=retest2 |

### Cycle 127 — BUY (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 10:15:00 | 205.80 | 202.11 | 201.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 209.01 | 205.79 | 204.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 203.12 | 207.44 | 206.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 203.12 | 207.44 | 206.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 203.12 | 207.44 | 206.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 203.12 | 207.44 | 206.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 205.00 | 206.95 | 206.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 204.43 | 206.95 | 206.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 202.55 | 205.55 | 205.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 189.49 | 201.51 | 203.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 196.84 | 195.69 | 199.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 201.87 | 195.69 | 199.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 200.58 | 196.67 | 199.27 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2025-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 13:15:00 | 206.60 | 201.06 | 200.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 14:15:00 | 207.00 | 202.25 | 201.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 13:15:00 | 235.75 | 236.77 | 232.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 14:00:00 | 235.75 | 236.77 | 232.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 233.90 | 236.20 | 232.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 14:45:00 | 232.30 | 236.20 | 232.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 233.25 | 235.61 | 232.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 234.20 | 235.61 | 232.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 226.96 | 233.88 | 232.33 | SL hit (close<static) qty=1.00 sl=232.65 alert=retest2 |

### Cycle 130 — SELL (started 2025-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 12:15:00 | 229.10 | 231.15 | 231.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 224.38 | 229.13 | 230.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 225.10 | 223.46 | 226.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 225.10 | 223.46 | 226.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 225.98 | 224.01 | 225.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:45:00 | 225.92 | 224.01 | 225.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 224.74 | 224.16 | 225.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:45:00 | 227.25 | 224.16 | 225.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 225.50 | 224.43 | 225.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 13:30:00 | 225.65 | 224.43 | 225.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 226.36 | 224.81 | 225.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 15:00:00 | 226.36 | 224.81 | 225.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 226.30 | 225.11 | 225.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 229.25 | 225.11 | 225.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 10:15:00 | 228.47 | 226.44 | 226.38 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 10:15:00 | 225.47 | 226.55 | 226.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 11:15:00 | 224.86 | 226.21 | 226.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 226.73 | 224.84 | 225.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 226.73 | 224.84 | 225.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 226.73 | 224.84 | 225.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 226.73 | 224.84 | 225.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 225.93 | 225.05 | 225.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:30:00 | 227.04 | 225.05 | 225.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 223.28 | 224.70 | 225.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 12:15:00 | 222.20 | 224.70 | 225.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 14:15:00 | 221.87 | 224.19 | 224.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 10:15:00 | 226.60 | 224.39 | 224.75 | SL hit (close>static) qty=1.00 sl=225.95 alert=retest2 |

### Cycle 133 — BUY (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 12:15:00 | 228.48 | 225.56 | 225.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 15:15:00 | 229.30 | 227.11 | 226.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 223.95 | 226.47 | 225.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 223.95 | 226.47 | 225.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 223.95 | 226.47 | 225.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:45:00 | 223.63 | 226.47 | 225.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 223.25 | 225.83 | 225.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:00:00 | 223.25 | 225.83 | 225.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 221.65 | 224.99 | 225.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 12:15:00 | 220.54 | 224.10 | 224.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 222.72 | 220.87 | 222.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 11:15:00 | 222.72 | 220.87 | 222.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 222.72 | 220.87 | 222.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 222.55 | 220.87 | 222.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 208.72 | 218.44 | 221.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:45:00 | 215.57 | 218.44 | 221.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 220.69 | 216.93 | 218.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 12:00:00 | 220.69 | 216.93 | 218.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 217.95 | 217.13 | 218.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:15:00 | 217.38 | 217.13 | 218.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 206.51 | 213.14 | 216.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 218.72 | 211.08 | 213.06 | SL hit (close>ema200) qty=0.50 sl=211.08 alert=retest2 |

### Cycle 135 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 221.39 | 215.62 | 214.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 222.04 | 216.90 | 215.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 220.80 | 221.72 | 219.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 15:00:00 | 220.80 | 221.72 | 219.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 219.00 | 221.18 | 219.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 222.25 | 221.18 | 219.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 11:15:00 | 221.60 | 220.99 | 219.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 13:15:00 | 217.71 | 220.14 | 219.60 | SL hit (close<static) qty=1.00 sl=218.95 alert=retest2 |

### Cycle 136 — SELL (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 09:15:00 | 228.80 | 229.85 | 229.93 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 236.87 | 229.45 | 228.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 240.03 | 234.08 | 231.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 11:15:00 | 239.00 | 239.21 | 236.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 12:00:00 | 239.00 | 239.21 | 236.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 237.47 | 238.87 | 237.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 15:15:00 | 237.10 | 238.87 | 237.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 237.10 | 238.52 | 237.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 237.99 | 238.52 | 237.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:45:00 | 238.05 | 238.57 | 237.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 11:30:00 | 237.64 | 238.56 | 237.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 10:15:00 | 237.80 | 238.65 | 238.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 238.07 | 238.54 | 238.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 12:45:00 | 238.85 | 238.58 | 238.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 11:15:00 | 241.95 | 244.91 | 245.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 11:15:00 | 241.95 | 244.91 | 245.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 13:15:00 | 241.67 | 243.78 | 244.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 10:15:00 | 243.30 | 242.89 | 243.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-06 11:00:00 | 243.30 | 242.89 | 243.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 243.99 | 243.11 | 243.91 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 14:15:00 | 246.38 | 244.77 | 244.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 249.60 | 245.92 | 245.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 15:15:00 | 247.10 | 247.15 | 246.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 09:15:00 | 246.13 | 247.15 | 246.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 245.69 | 246.86 | 246.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:45:00 | 244.41 | 246.86 | 246.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 244.79 | 246.45 | 246.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 244.79 | 246.45 | 246.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 244.39 | 246.04 | 245.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:30:00 | 244.29 | 246.04 | 245.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 12:15:00 | 244.24 | 245.68 | 245.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 15:15:00 | 243.00 | 244.72 | 245.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 227.49 | 227.22 | 230.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:00:00 | 227.49 | 227.22 | 230.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 230.22 | 228.76 | 230.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 230.54 | 228.76 | 230.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 229.79 | 228.97 | 230.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 230.32 | 228.97 | 230.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 220.74 | 220.00 | 222.66 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 15:15:00 | 227.84 | 224.18 | 223.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 09:15:00 | 228.96 | 225.14 | 224.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 12:15:00 | 237.89 | 238.07 | 235.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 13:00:00 | 237.89 | 238.07 | 235.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 236.91 | 239.12 | 237.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:00:00 | 236.91 | 239.12 | 237.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 236.86 | 238.67 | 237.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 238.18 | 237.74 | 237.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 12:00:00 | 238.09 | 238.05 | 237.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 15:15:00 | 238.07 | 237.73 | 237.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 236.19 | 237.48 | 237.43 | SL hit (close<static) qty=1.00 sl=236.37 alert=retest2 |

### Cycle 142 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 236.99 | 237.38 | 237.39 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 13:15:00 | 239.79 | 237.79 | 237.56 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 234.99 | 237.54 | 237.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 12:15:00 | 234.05 | 236.54 | 237.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 232.95 | 232.63 | 234.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 09:30:00 | 232.68 | 232.63 | 234.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 227.90 | 227.28 | 229.00 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 12:15:00 | 230.74 | 229.75 | 229.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 231.50 | 230.41 | 230.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 230.01 | 230.50 | 230.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 10:15:00 | 230.01 | 230.50 | 230.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 230.01 | 230.50 | 230.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 230.01 | 230.50 | 230.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 232.50 | 230.90 | 230.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:30:00 | 233.61 | 231.68 | 230.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 12:15:00 | 233.35 | 232.12 | 231.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:30:00 | 233.50 | 232.24 | 231.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 10:00:00 | 234.20 | 232.24 | 231.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 231.99 | 232.97 | 232.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:00:00 | 231.99 | 232.97 | 232.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 233.66 | 233.11 | 232.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-17 10:15:00 | 231.60 | 232.21 | 232.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 10:15:00 | 231.60 | 232.21 | 232.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 11:15:00 | 230.00 | 231.77 | 232.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 11:15:00 | 227.60 | 227.03 | 228.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 12:00:00 | 227.60 | 227.03 | 228.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 228.47 | 227.32 | 228.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:30:00 | 228.42 | 227.32 | 228.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 227.78 | 227.41 | 228.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:45:00 | 227.03 | 227.47 | 228.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 226.52 | 227.47 | 228.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 215.68 | 218.34 | 220.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 215.19 | 218.34 | 220.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 215.82 | 215.81 | 218.15 | SL hit (close>ema200) qty=0.50 sl=215.81 alert=retest2 |

### Cycle 147 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 218.82 | 214.88 | 214.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 14:15:00 | 219.54 | 215.82 | 214.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 09:15:00 | 216.30 | 216.38 | 215.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 10:00:00 | 216.30 | 216.38 | 215.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 214.35 | 217.25 | 216.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 214.35 | 217.25 | 216.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 212.77 | 216.36 | 216.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 212.77 | 216.36 | 216.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 212.99 | 215.68 | 215.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 211.65 | 213.86 | 214.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 210.40 | 210.31 | 212.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 210.40 | 210.31 | 212.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 208.63 | 210.08 | 211.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:15:00 | 208.24 | 210.08 | 211.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:45:00 | 208.33 | 209.02 | 210.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:30:00 | 208.49 | 208.89 | 210.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 211.40 | 210.32 | 210.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 211.40 | 210.32 | 210.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 213.19 | 211.21 | 210.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 211.26 | 211.57 | 211.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 11:15:00 | 211.26 | 211.57 | 211.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 211.26 | 211.57 | 211.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 211.26 | 211.57 | 211.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 211.26 | 211.51 | 211.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:30:00 | 210.97 | 211.51 | 211.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 210.77 | 211.36 | 211.03 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 207.87 | 210.37 | 210.64 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 211.80 | 210.07 | 210.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 10:15:00 | 215.52 | 211.93 | 211.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 10:15:00 | 213.61 | 213.65 | 212.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 11:00:00 | 213.61 | 213.65 | 212.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 214.60 | 213.84 | 212.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:30:00 | 213.17 | 213.84 | 212.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 213.80 | 214.21 | 213.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 213.80 | 214.21 | 213.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 213.10 | 213.99 | 213.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:30:00 | 214.07 | 214.04 | 213.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:00:00 | 214.08 | 214.05 | 213.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 14:15:00 | 211.91 | 213.41 | 213.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 211.91 | 213.41 | 213.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 15:15:00 | 211.45 | 213.02 | 213.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 213.00 | 212.96 | 213.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 10:15:00 | 213.00 | 212.96 | 213.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 213.00 | 212.96 | 213.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 211.24 | 213.03 | 213.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:30:00 | 211.54 | 212.52 | 212.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 13:15:00 | 210.86 | 208.20 | 207.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 210.86 | 208.20 | 207.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 211.44 | 208.85 | 208.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 14:15:00 | 216.23 | 216.24 | 214.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 15:00:00 | 216.23 | 216.24 | 214.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 215.53 | 216.01 | 215.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:30:00 | 215.07 | 216.01 | 215.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 213.87 | 215.59 | 214.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 213.87 | 215.59 | 214.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 214.59 | 215.39 | 214.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 216.76 | 215.39 | 214.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:15:00 | 215.20 | 214.81 | 214.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 216.25 | 216.68 | 216.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 216.25 | 216.68 | 216.71 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 224.13 | 218.07 | 217.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 10:15:00 | 224.75 | 219.40 | 218.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 13:15:00 | 222.60 | 222.64 | 221.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 14:00:00 | 222.60 | 222.64 | 221.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 222.65 | 224.06 | 223.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 222.65 | 224.06 | 223.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 222.80 | 223.81 | 223.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:45:00 | 222.72 | 223.81 | 223.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 236.66 | 236.38 | 234.27 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 10:15:00 | 231.95 | 233.67 | 233.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 11:15:00 | 231.10 | 233.16 | 233.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 221.96 | 221.82 | 224.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:00:00 | 221.96 | 221.82 | 224.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 222.69 | 221.64 | 223.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:15:00 | 220.85 | 221.51 | 223.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 232.27 | 224.12 | 223.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 232.27 | 224.12 | 223.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 12:15:00 | 233.54 | 227.87 | 225.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 231.42 | 232.51 | 230.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:45:00 | 231.85 | 232.51 | 230.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 231.15 | 231.99 | 230.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 12:45:00 | 231.40 | 231.78 | 230.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 13:15:00 | 231.73 | 231.78 | 230.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 15:15:00 | 231.50 | 231.34 | 230.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 10:15:00 | 229.11 | 230.75 | 230.51 | SL hit (close<static) qty=1.00 sl=230.24 alert=retest2 |

### Cycle 158 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 226.92 | 230.10 | 230.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 13:15:00 | 226.35 | 228.57 | 229.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 11:15:00 | 228.36 | 227.17 | 228.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 11:15:00 | 228.36 | 227.17 | 228.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 228.36 | 227.17 | 228.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 228.36 | 227.17 | 228.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 228.00 | 227.33 | 228.30 | EMA400 retest candle locked (from downside) |

### Cycle 159 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 232.98 | 229.24 | 228.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 13:15:00 | 234.00 | 231.47 | 230.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 230.46 | 231.58 | 230.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 230.46 | 231.58 | 230.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 230.46 | 231.58 | 230.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 230.69 | 231.58 | 230.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 229.01 | 231.06 | 230.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 229.01 | 231.06 | 230.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 229.75 | 230.80 | 230.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 14:15:00 | 230.50 | 230.45 | 230.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 14:45:00 | 230.55 | 230.46 | 230.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 227.11 | 229.75 | 230.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 227.11 | 229.75 | 230.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 224.63 | 227.76 | 228.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 226.60 | 226.41 | 227.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 226.60 | 226.41 | 227.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 228.42 | 226.97 | 227.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 228.42 | 226.97 | 227.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 229.37 | 227.45 | 227.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 229.37 | 227.45 | 227.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 229.70 | 228.29 | 228.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 231.85 | 229.00 | 228.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 12:15:00 | 229.15 | 229.31 | 228.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 13:00:00 | 229.15 | 229.31 | 228.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 228.91 | 229.23 | 228.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:30:00 | 228.86 | 229.23 | 228.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 227.84 | 228.95 | 228.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 15:00:00 | 227.84 | 228.95 | 228.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 228.21 | 228.80 | 228.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 227.80 | 228.80 | 228.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 225.75 | 228.19 | 228.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 224.18 | 227.18 | 227.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 226.39 | 225.93 | 226.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 226.39 | 225.93 | 226.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 226.39 | 225.93 | 226.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:15:00 | 226.40 | 225.93 | 226.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 227.70 | 226.28 | 227.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:00:00 | 227.70 | 226.28 | 227.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 228.03 | 226.63 | 227.13 | EMA400 retest candle locked (from downside) |

### Cycle 163 — BUY (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 14:15:00 | 228.45 | 227.60 | 227.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 229.89 | 228.33 | 227.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 11:15:00 | 226.98 | 228.06 | 227.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 11:15:00 | 226.98 | 228.06 | 227.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 226.98 | 228.06 | 227.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:00:00 | 226.98 | 228.06 | 227.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 227.93 | 228.03 | 227.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 13:15:00 | 229.12 | 228.03 | 227.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 14:00:00 | 229.25 | 228.28 | 227.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 15:15:00 | 229.00 | 228.26 | 228.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:30:00 | 229.20 | 228.48 | 228.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 227.80 | 228.35 | 228.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:00:00 | 227.80 | 228.35 | 228.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 227.60 | 228.20 | 228.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:30:00 | 227.31 | 228.20 | 228.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-24 12:15:00 | 226.63 | 227.88 | 227.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 226.63 | 227.88 | 227.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 13:15:00 | 225.57 | 226.62 | 227.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 228.27 | 226.76 | 226.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 228.27 | 226.76 | 226.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 228.27 | 226.76 | 226.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 228.27 | 226.76 | 226.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 231.40 | 227.69 | 227.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 12:15:00 | 232.97 | 229.56 | 228.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 236.75 | 237.71 | 235.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 15:00:00 | 236.75 | 237.71 | 235.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 240.86 | 238.19 | 236.20 | EMA400 retest candle locked (from upside) |

### Cycle 166 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 233.49 | 236.34 | 236.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 228.70 | 233.19 | 234.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 228.00 | 227.53 | 229.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:00:00 | 228.00 | 227.53 | 229.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 229.32 | 227.89 | 229.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 230.00 | 227.89 | 229.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 229.90 | 228.29 | 229.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 231.20 | 228.29 | 229.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 231.25 | 228.88 | 230.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 231.69 | 228.88 | 230.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 233.45 | 229.80 | 230.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 233.56 | 229.80 | 230.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 233.11 | 230.46 | 230.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:15:00 | 234.80 | 230.46 | 230.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 234.16 | 231.20 | 230.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 235.67 | 232.61 | 231.65 | Break + close above crossover candle high |

### Cycle 168 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 224.50 | 231.23 | 231.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 223.30 | 229.64 | 230.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 14:15:00 | 230.80 | 228.67 | 229.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 14:15:00 | 230.80 | 228.67 | 229.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 230.80 | 228.67 | 229.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:00:00 | 230.80 | 228.67 | 229.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 230.85 | 229.11 | 229.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 231.40 | 229.11 | 229.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 228.23 | 227.98 | 228.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:30:00 | 228.73 | 227.98 | 228.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 227.00 | 226.69 | 227.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:30:00 | 227.91 | 226.69 | 227.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 227.04 | 226.76 | 227.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:30:00 | 226.93 | 226.76 | 227.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 227.62 | 226.93 | 227.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 227.62 | 226.93 | 227.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 227.59 | 227.07 | 227.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 231.87 | 227.07 | 227.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 232.68 | 228.19 | 227.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 12:15:00 | 233.08 | 230.14 | 228.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 10:15:00 | 241.50 | 242.09 | 238.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 10:45:00 | 241.75 | 242.09 | 238.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 238.45 | 241.20 | 238.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:00:00 | 238.45 | 241.20 | 238.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 237.40 | 240.44 | 238.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:45:00 | 236.90 | 240.44 | 238.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 237.10 | 239.77 | 238.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 236.50 | 239.77 | 238.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 238.34 | 239.14 | 238.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:00:00 | 238.34 | 239.14 | 238.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 236.98 | 238.71 | 238.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 14:00:00 | 236.98 | 238.71 | 238.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 237.25 | 238.08 | 238.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 232.36 | 236.94 | 237.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 233.47 | 233.08 | 234.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 09:45:00 | 233.30 | 233.08 | 234.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 230.75 | 230.30 | 231.57 | EMA400 retest candle locked (from downside) |

### Cycle 171 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 237.11 | 232.79 | 232.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 239.19 | 236.19 | 234.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 14:15:00 | 239.59 | 240.06 | 238.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 14:45:00 | 239.54 | 240.06 | 238.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 238.37 | 239.72 | 238.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 240.72 | 239.72 | 238.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 11:15:00 | 237.43 | 239.04 | 238.52 | SL hit (close<static) qty=1.00 sl=238.21 alert=retest2 |

### Cycle 172 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 237.30 | 238.15 | 238.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 12:15:00 | 236.49 | 237.66 | 238.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 13:15:00 | 223.07 | 222.27 | 225.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 14:00:00 | 223.07 | 222.27 | 225.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 225.54 | 222.92 | 225.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 225.54 | 222.92 | 225.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 225.00 | 223.34 | 225.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 222.90 | 223.34 | 225.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 14:15:00 | 211.75 | 217.16 | 221.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 214.90 | 214.20 | 216.70 | SL hit (close>ema200) qty=0.50 sl=214.20 alert=retest2 |

### Cycle 173 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 10:15:00 | 214.23 | 213.58 | 213.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 11:15:00 | 215.85 | 214.03 | 213.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 211.94 | 214.08 | 213.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 211.94 | 214.08 | 213.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 211.94 | 214.08 | 213.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 211.94 | 214.08 | 213.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 211.96 | 213.66 | 213.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 211.27 | 213.18 | 213.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 208.32 | 208.22 | 209.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 208.32 | 208.22 | 209.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 207.58 | 207.48 | 208.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 208.34 | 207.48 | 208.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 208.46 | 207.76 | 208.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 208.74 | 207.76 | 208.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 209.23 | 208.06 | 208.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:30:00 | 209.05 | 208.06 | 208.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 211.99 | 208.84 | 208.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 211.99 | 208.84 | 208.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 211.30 | 209.33 | 209.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 214.53 | 210.37 | 209.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 216.60 | 216.95 | 215.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 13:30:00 | 216.70 | 216.95 | 215.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 219.27 | 217.15 | 215.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:30:00 | 220.50 | 218.17 | 216.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 225.08 | 226.94 | 226.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 225.08 | 226.94 | 226.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 224.15 | 225.66 | 226.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 225.70 | 225.45 | 225.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 225.70 | 225.45 | 225.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 225.70 | 225.45 | 225.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:45:00 | 225.60 | 225.45 | 225.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 226.54 | 225.67 | 226.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 226.54 | 225.67 | 226.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 226.79 | 225.89 | 226.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:30:00 | 227.84 | 225.89 | 226.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 226.39 | 226.10 | 226.17 | EMA400 retest candle locked (from downside) |

### Cycle 177 — BUY (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 14:15:00 | 227.17 | 226.32 | 226.26 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 224.77 | 226.09 | 226.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 220.90 | 225.05 | 225.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 216.20 | 215.62 | 218.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:30:00 | 216.09 | 215.62 | 218.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 217.56 | 215.94 | 217.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 217.56 | 215.94 | 217.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 217.55 | 216.26 | 217.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 216.72 | 216.26 | 217.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 215.76 | 216.16 | 217.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 214.35 | 215.67 | 217.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 214.34 | 215.46 | 216.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 213.66 | 215.10 | 216.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 217.93 | 216.76 | 216.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 217.93 | 216.76 | 216.69 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 12:15:00 | 215.50 | 216.61 | 216.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 13:15:00 | 214.92 | 216.27 | 216.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 208.70 | 205.50 | 207.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 208.70 | 205.50 | 207.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 208.70 | 205.50 | 207.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 208.96 | 205.50 | 207.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 206.58 | 205.71 | 207.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 205.78 | 205.71 | 207.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 206.48 | 206.31 | 206.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 13:15:00 | 195.49 | 203.04 | 205.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 13:15:00 | 196.16 | 203.04 | 205.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 14:15:00 | 197.80 | 197.16 | 200.29 | SL hit (close>ema200) qty=0.50 sl=197.16 alert=retest2 |

### Cycle 181 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 204.35 | 200.92 | 200.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 204.66 | 202.23 | 201.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 11:15:00 | 202.35 | 202.49 | 201.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 12:00:00 | 202.35 | 202.49 | 201.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 203.06 | 202.60 | 201.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:30:00 | 202.12 | 202.60 | 201.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — SELL (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 09:15:00 | 194.78 | 201.54 | 201.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 188.99 | 196.72 | 199.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 09:15:00 | 194.43 | 194.29 | 197.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-01 10:00:00 | 194.43 | 194.29 | 197.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 197.27 | 194.89 | 197.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:00:00 | 197.27 | 194.89 | 197.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 194.81 | 194.87 | 196.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 189.66 | 194.87 | 196.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:45:00 | 192.76 | 194.49 | 196.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 14:45:00 | 192.94 | 194.23 | 196.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 10:15:00 | 183.12 | 189.99 | 193.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 10:15:00 | 183.29 | 189.99 | 193.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 11:15:00 | 180.18 | 187.94 | 192.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-02 14:15:00 | 188.00 | 186.07 | 190.16 | SL hit (close>ema200) qty=0.50 sl=186.07 alert=retest2 |

### Cycle 183 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 195.91 | 191.42 | 191.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 196.71 | 193.81 | 192.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 194.76 | 194.94 | 193.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 194.76 | 194.94 | 193.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 193.68 | 194.69 | 193.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 193.09 | 194.69 | 193.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 193.80 | 194.51 | 193.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:30:00 | 194.29 | 194.51 | 193.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 194.35 | 194.48 | 193.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:15:00 | 194.60 | 194.48 | 193.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 14:15:00 | 194.50 | 194.42 | 193.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 189.46 | 193.47 | 193.36 | SL hit (close<static) qty=1.00 sl=193.30 alert=retest2 |

### Cycle 184 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 189.55 | 192.69 | 193.02 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 194.67 | 192.61 | 192.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 195.23 | 194.49 | 193.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 194.24 | 194.55 | 193.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:00:00 | 194.24 | 194.55 | 193.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 194.31 | 194.50 | 193.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:15:00 | 193.48 | 194.50 | 193.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 193.48 | 194.30 | 193.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 192.07 | 194.30 | 193.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 191.16 | 193.67 | 193.68 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 09:15:00 | 195.54 | 193.61 | 193.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 12:15:00 | 199.70 | 194.88 | 194.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 196.12 | 197.69 | 195.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 196.12 | 197.69 | 195.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 196.12 | 197.69 | 195.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 196.99 | 197.69 | 195.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 195.40 | 197.24 | 195.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:00:00 | 195.40 | 197.24 | 195.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 195.85 | 196.96 | 195.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:30:00 | 194.75 | 196.96 | 195.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 195.06 | 196.58 | 195.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 13:00:00 | 195.06 | 196.58 | 195.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 194.97 | 196.26 | 195.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 13:45:00 | 194.91 | 196.26 | 195.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 195.35 | 195.95 | 195.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:15:00 | 195.41 | 195.95 | 195.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 195.48 | 195.86 | 195.66 | EMA400 retest candle locked (from upside) |

### Cycle 188 — SELL (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 11:15:00 | 193.81 | 195.25 | 195.40 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 196.86 | 195.66 | 195.53 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 11:15:00 | 194.82 | 195.47 | 195.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 14:15:00 | 194.02 | 195.00 | 195.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 09:15:00 | 195.90 | 195.07 | 195.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 195.90 | 195.07 | 195.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 195.90 | 195.07 | 195.22 | EMA400 retest candle locked (from downside) |

### Cycle 191 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 195.85 | 195.33 | 195.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 13:15:00 | 196.30 | 195.52 | 195.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 13:15:00 | 196.27 | 196.46 | 196.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 13:15:00 | 196.27 | 196.46 | 196.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 196.27 | 196.46 | 196.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 196.27 | 196.46 | 196.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 194.80 | 196.13 | 195.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 194.80 | 196.13 | 195.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 193.72 | 195.65 | 195.74 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 196.88 | 195.96 | 195.86 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 15:15:00 | 195.36 | 195.75 | 195.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 10:15:00 | 194.82 | 195.61 | 195.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 10:15:00 | 191.36 | 191.09 | 192.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:30:00 | 191.25 | 191.09 | 192.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 190.03 | 190.23 | 191.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 10:15:00 | 189.70 | 190.23 | 191.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 189.47 | 189.91 | 191.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 15:15:00 | 189.00 | 189.33 | 190.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 180.21 | 186.65 | 188.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 180.00 | 186.65 | 188.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 179.55 | 186.65 | 188.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 177.87 | 176.59 | 178.81 | SL hit (close>ema200) qty=0.50 sl=176.59 alert=retest2 |

### Cycle 195 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 179.89 | 176.25 | 175.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 180.00 | 177.46 | 176.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 177.29 | 178.40 | 177.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 177.29 | 178.40 | 177.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 177.29 | 178.40 | 177.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 177.29 | 178.40 | 177.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 177.58 | 178.24 | 177.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 174.91 | 178.24 | 177.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 174.23 | 177.44 | 177.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 174.33 | 177.44 | 177.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 178.56 | 177.73 | 177.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:15:00 | 178.70 | 177.73 | 177.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:15:00 | 178.71 | 177.79 | 177.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 173.35 | 176.63 | 177.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 173.35 | 176.63 | 177.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 173.15 | 174.99 | 176.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 172.34 | 171.56 | 173.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 172.34 | 171.56 | 173.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 172.34 | 171.56 | 173.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 172.83 | 171.56 | 173.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 173.50 | 172.02 | 173.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 173.05 | 172.02 | 173.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 172.39 | 172.09 | 173.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 171.23 | 172.09 | 173.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 174.67 | 172.65 | 172.94 | SL hit (close>static) qty=1.00 sl=174.30 alert=retest2 |

### Cycle 197 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 176.60 | 173.44 | 173.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 177.00 | 174.15 | 173.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 173.97 | 176.46 | 175.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 173.97 | 176.46 | 175.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 173.97 | 176.46 | 175.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 173.97 | 176.46 | 175.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 173.65 | 175.90 | 175.07 | EMA400 retest candle locked (from upside) |

### Cycle 198 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 172.19 | 174.43 | 174.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 170.90 | 173.72 | 174.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 174.61 | 173.62 | 174.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 174.61 | 173.62 | 174.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 174.61 | 173.62 | 174.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 175.96 | 173.62 | 174.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 174.29 | 173.76 | 174.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:15:00 | 175.20 | 173.76 | 174.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 173.89 | 173.78 | 174.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 173.07 | 173.78 | 174.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:15:00 | 173.20 | 173.68 | 173.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:15:00 | 164.54 | 168.82 | 171.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 15:15:00 | 164.42 | 166.90 | 169.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 167.10 | 166.65 | 168.83 | SL hit (close>ema200) qty=0.50 sl=166.65 alert=retest2 |

### Cycle 199 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 177.16 | 169.94 | 169.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 179.25 | 171.80 | 170.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 170.37 | 174.78 | 173.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 170.37 | 174.78 | 173.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 170.37 | 174.78 | 173.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 170.37 | 174.78 | 173.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 169.23 | 173.67 | 172.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 169.17 | 173.67 | 172.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 169.50 | 172.02 | 172.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 167.93 | 170.94 | 171.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 167.13 | 163.51 | 166.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 167.13 | 163.51 | 166.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 167.13 | 163.51 | 166.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 162.20 | 167.22 | 167.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 09:15:00 | 167.49 | 166.59 | 166.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 167.49 | 166.59 | 166.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 15:15:00 | 169.85 | 168.34 | 167.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 167.98 | 168.27 | 167.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 167.98 | 168.27 | 167.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 167.98 | 168.27 | 167.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 11:45:00 | 170.32 | 168.70 | 167.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-13 12:15:00 | 187.35 | 185.20 | 182.95 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-23 13:15:00 | 55.95 | 2023-05-24 09:15:00 | 57.70 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2023-05-29 14:15:00 | 55.35 | 2023-05-30 09:15:00 | 58.85 | STOP_HIT | 1.00 | -6.32% |
| BUY | retest2 | 2023-06-06 13:15:00 | 60.80 | 2023-06-08 11:15:00 | 59.30 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2023-06-06 14:45:00 | 60.90 | 2023-06-08 11:15:00 | 59.30 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2023-06-07 09:15:00 | 61.15 | 2023-06-08 11:15:00 | 59.30 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2023-06-08 09:45:00 | 60.85 | 2023-06-08 11:15:00 | 59.30 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2023-06-19 13:30:00 | 58.70 | 2023-06-21 09:15:00 | 61.80 | STOP_HIT | 1.00 | -5.28% |
| SELL | retest2 | 2023-06-20 10:30:00 | 58.60 | 2023-06-21 09:15:00 | 61.80 | STOP_HIT | 1.00 | -5.46% |
| SELL | retest2 | 2023-06-20 12:15:00 | 58.75 | 2023-06-21 09:15:00 | 61.80 | STOP_HIT | 1.00 | -5.19% |
| SELL | retest2 | 2023-06-20 13:00:00 | 58.75 | 2023-06-21 09:15:00 | 61.80 | STOP_HIT | 1.00 | -5.19% |
| SELL | retest2 | 2023-06-27 12:00:00 | 57.75 | 2023-07-03 14:15:00 | 58.15 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2023-06-27 13:00:00 | 57.70 | 2023-07-03 14:15:00 | 58.15 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2023-06-27 14:15:00 | 57.70 | 2023-07-03 14:15:00 | 58.15 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2023-07-12 15:15:00 | 59.85 | 2023-07-13 13:15:00 | 58.95 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2023-07-13 10:45:00 | 59.95 | 2023-07-13 13:15:00 | 58.95 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2023-08-10 12:30:00 | 67.45 | 2023-08-10 13:15:00 | 66.75 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2023-08-11 09:45:00 | 67.95 | 2023-08-14 09:15:00 | 65.90 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2023-08-29 11:45:00 | 72.70 | 2023-08-29 13:15:00 | 72.75 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2023-09-01 13:00:00 | 75.25 | 2023-09-08 12:15:00 | 77.05 | STOP_HIT | 1.00 | 2.39% |
| SELL | retest2 | 2023-09-14 13:45:00 | 71.40 | 2023-09-15 09:15:00 | 73.80 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2023-10-17 11:30:00 | 90.30 | 2023-10-18 09:15:00 | 85.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-17 11:30:00 | 90.30 | 2023-10-18 11:15:00 | 81.27 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2023-11-06 09:15:00 | 79.10 | 2023-11-10 10:15:00 | 78.60 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2023-11-22 14:15:00 | 82.80 | 2023-11-24 09:15:00 | 81.95 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2023-11-29 13:45:00 | 81.50 | 2023-11-30 11:15:00 | 85.10 | STOP_HIT | 1.00 | -4.42% |
| SELL | retest2 | 2023-11-29 14:45:00 | 81.55 | 2023-11-30 11:15:00 | 85.10 | STOP_HIT | 1.00 | -4.35% |
| BUY | retest2 | 2023-12-05 13:15:00 | 87.90 | 2023-12-12 09:15:00 | 96.25 | TARGET_HIT | 1.00 | 9.50% |
| BUY | retest2 | 2023-12-05 14:45:00 | 87.80 | 2023-12-12 13:15:00 | 96.69 | TARGET_HIT | 1.00 | 10.13% |
| BUY | retest2 | 2023-12-05 15:15:00 | 87.50 | 2023-12-12 13:15:00 | 96.58 | TARGET_HIT | 1.00 | 10.38% |
| BUY | retest2 | 2023-12-06 10:15:00 | 87.70 | 2023-12-12 13:15:00 | 96.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-07 09:15:00 | 90.05 | 2023-12-12 13:15:00 | 99.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-02 12:30:00 | 127.40 | 2024-01-05 13:15:00 | 126.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-01-02 14:00:00 | 127.90 | 2024-01-05 13:15:00 | 126.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-01-02 15:00:00 | 127.25 | 2024-01-05 13:15:00 | 126.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-01-03 09:15:00 | 128.00 | 2024-01-05 13:15:00 | 126.00 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-01-03 10:30:00 | 128.85 | 2024-01-05 13:15:00 | 126.00 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-01-04 09:15:00 | 131.90 | 2024-01-05 13:15:00 | 126.00 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest2 | 2024-01-04 12:45:00 | 128.80 | 2024-01-05 13:15:00 | 126.00 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2024-01-04 14:45:00 | 128.65 | 2024-01-05 13:15:00 | 126.00 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2024-01-10 09:15:00 | 123.75 | 2024-01-11 09:15:00 | 130.35 | STOP_HIT | 1.00 | -5.33% |
| BUY | retest2 | 2024-01-17 09:45:00 | 134.35 | 2024-01-19 09:15:00 | 147.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-18 10:00:00 | 133.75 | 2024-01-19 09:15:00 | 147.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-31 09:30:00 | 166.45 | 2024-02-01 09:15:00 | 183.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-31 10:00:00 | 165.55 | 2024-02-01 09:15:00 | 182.11 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-02-09 09:15:00 | 193.40 | 2024-02-09 09:15:00 | 183.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-09 09:15:00 | 193.40 | 2024-02-09 14:15:00 | 202.85 | STOP_HIT | 0.50 | -4.89% |
| SELL | retest2 | 2024-02-12 09:15:00 | 195.00 | 2024-02-12 09:15:00 | 185.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-12 09:15:00 | 195.00 | 2024-02-13 09:15:00 | 175.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-01 11:45:00 | 193.85 | 2024-03-02 11:15:00 | 196.35 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-03-01 15:15:00 | 193.90 | 2024-03-02 11:15:00 | 196.35 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-03-05 09:15:00 | 200.85 | 2024-03-06 11:15:00 | 192.90 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2024-03-06 10:00:00 | 196.80 | 2024-03-06 11:15:00 | 192.90 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-03-06 10:30:00 | 197.55 | 2024-03-06 11:15:00 | 192.90 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-04-01 09:15:00 | 189.25 | 2024-04-04 14:15:00 | 208.18 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-04-19 14:30:00 | 194.45 | 2024-04-22 10:15:00 | 198.80 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-04-25 09:15:00 | 202.00 | 2024-04-26 10:15:00 | 222.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-07 12:15:00 | 247.20 | 2024-06-07 13:15:00 | 249.95 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-06-07 13:00:00 | 247.90 | 2024-06-07 13:15:00 | 249.95 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-07-11 13:30:00 | 338.20 | 2024-07-15 12:15:00 | 332.45 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-07-15 10:45:00 | 336.85 | 2024-07-15 12:15:00 | 332.45 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-07-23 09:45:00 | 321.25 | 2024-07-23 12:15:00 | 289.12 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-23 11:15:00 | 318.70 | 2024-07-23 12:15:00 | 286.83 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-23 11:45:00 | 318.80 | 2024-07-23 12:15:00 | 286.92 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-25 10:15:00 | 320.70 | 2024-07-25 11:15:00 | 321.30 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-08-06 13:30:00 | 292.40 | 2024-08-12 09:15:00 | 307.50 | STOP_HIT | 1.00 | -5.16% |
| SELL | retest2 | 2024-08-08 09:15:00 | 294.25 | 2024-08-12 09:15:00 | 307.50 | STOP_HIT | 1.00 | -4.50% |
| SELL | retest2 | 2024-08-08 09:45:00 | 293.45 | 2024-08-12 09:15:00 | 307.50 | STOP_HIT | 1.00 | -4.79% |
| SELL | retest2 | 2024-08-09 11:30:00 | 294.20 | 2024-08-12 09:15:00 | 307.50 | STOP_HIT | 1.00 | -4.52% |
| SELL | retest2 | 2024-08-23 13:45:00 | 283.80 | 2024-08-28 11:15:00 | 287.30 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-08-26 09:30:00 | 283.70 | 2024-08-28 11:15:00 | 287.30 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-08-28 11:45:00 | 283.70 | 2024-08-28 12:15:00 | 296.20 | STOP_HIT | 1.00 | -4.41% |
| SELL | retest2 | 2024-09-11 11:45:00 | 250.70 | 2024-09-12 15:15:00 | 254.20 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-09-11 12:45:00 | 249.65 | 2024-09-12 15:15:00 | 254.20 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-09-12 11:15:00 | 250.65 | 2024-09-12 15:15:00 | 254.20 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-09-12 11:45:00 | 250.35 | 2024-09-12 15:15:00 | 254.20 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-09-18 11:00:00 | 244.40 | 2024-09-19 10:15:00 | 232.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-18 11:00:00 | 244.40 | 2024-09-20 09:15:00 | 233.15 | STOP_HIT | 0.50 | 4.60% |
| SELL | retest2 | 2024-10-25 10:15:00 | 196.67 | 2024-10-28 15:15:00 | 204.10 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2024-10-28 09:30:00 | 195.14 | 2024-10-28 15:15:00 | 204.10 | STOP_HIT | 1.00 | -4.59% |
| BUY | retest2 | 2024-11-04 13:15:00 | 217.47 | 2024-11-08 13:15:00 | 220.50 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2024-11-05 09:45:00 | 216.23 | 2024-11-08 13:15:00 | 220.50 | STOP_HIT | 1.00 | 1.97% |
| BUY | retest2 | 2024-11-05 11:45:00 | 216.02 | 2024-11-08 13:15:00 | 220.50 | STOP_HIT | 1.00 | 2.07% |
| SELL | retest2 | 2024-11-11 12:45:00 | 217.34 | 2024-11-13 09:15:00 | 206.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 09:30:00 | 216.98 | 2024-11-13 09:15:00 | 206.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 12:45:00 | 217.34 | 2024-11-14 09:15:00 | 207.07 | STOP_HIT | 0.50 | 4.73% |
| SELL | retest2 | 2024-11-12 09:30:00 | 216.98 | 2024-11-14 09:15:00 | 207.07 | STOP_HIT | 0.50 | 4.57% |
| BUY | retest2 | 2024-11-27 09:45:00 | 216.26 | 2024-11-28 09:15:00 | 237.89 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-20 12:15:00 | 246.62 | 2024-12-24 09:15:00 | 234.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 12:15:00 | 246.62 | 2024-12-27 09:15:00 | 229.73 | STOP_HIT | 0.50 | 6.85% |
| SELL | retest2 | 2025-01-07 15:15:00 | 238.76 | 2025-01-09 14:15:00 | 226.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 15:15:00 | 238.76 | 2025-01-10 14:15:00 | 214.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-23 11:15:00 | 218.64 | 2025-01-27 09:15:00 | 207.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 12:15:00 | 219.44 | 2025-01-27 09:15:00 | 208.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 13:30:00 | 221.34 | 2025-01-27 09:15:00 | 210.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 11:15:00 | 218.64 | 2025-01-28 12:15:00 | 212.83 | STOP_HIT | 0.50 | 2.66% |
| SELL | retest2 | 2025-01-23 12:15:00 | 219.44 | 2025-01-28 12:15:00 | 212.83 | STOP_HIT | 0.50 | 3.01% |
| SELL | retest2 | 2025-01-24 13:30:00 | 221.34 | 2025-01-28 12:15:00 | 212.83 | STOP_HIT | 0.50 | 3.84% |
| SELL | retest2 | 2025-02-10 11:00:00 | 201.11 | 2025-02-12 09:15:00 | 191.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 11:00:00 | 201.11 | 2025-02-12 12:15:00 | 194.73 | STOP_HIT | 0.50 | 3.17% |
| SELL | retest2 | 2025-03-04 11:45:00 | 167.98 | 2025-03-05 09:15:00 | 178.05 | STOP_HIT | 1.00 | -5.99% |
| SELL | retest2 | 2025-03-04 12:45:00 | 167.61 | 2025-03-05 09:15:00 | 178.05 | STOP_HIT | 1.00 | -6.23% |
| BUY | retest2 | 2025-03-17 09:15:00 | 186.80 | 2025-03-24 09:15:00 | 205.48 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-27 13:30:00 | 201.91 | 2025-04-01 09:15:00 | 203.79 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-03-28 10:00:00 | 202.00 | 2025-04-01 09:15:00 | 203.79 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-03-28 11:00:00 | 202.00 | 2025-04-01 09:15:00 | 203.79 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-03-28 13:00:00 | 201.15 | 2025-04-01 09:15:00 | 203.79 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-04-23 09:15:00 | 234.20 | 2025-04-23 09:15:00 | 226.96 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-05-02 12:15:00 | 222.20 | 2025-05-05 10:15:00 | 226.60 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-05-02 14:15:00 | 221.87 | 2025-05-05 10:15:00 | 226.60 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-05-08 13:15:00 | 217.38 | 2025-05-09 09:15:00 | 206.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 13:15:00 | 217.38 | 2025-05-12 09:15:00 | 218.72 | STOP_HIT | 0.50 | -0.62% |
| BUY | retest2 | 2025-05-14 09:15:00 | 222.25 | 2025-05-14 13:15:00 | 217.71 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-05-14 11:15:00 | 221.60 | 2025-05-14 13:15:00 | 217.71 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-05-15 09:15:00 | 221.90 | 2025-05-21 09:15:00 | 228.80 | STOP_HIT | 1.00 | 3.11% |
| BUY | retest2 | 2025-05-15 12:00:00 | 221.51 | 2025-05-21 09:15:00 | 228.80 | STOP_HIT | 1.00 | 3.29% |
| BUY | retest2 | 2025-05-29 09:15:00 | 237.99 | 2025-06-05 11:15:00 | 241.95 | STOP_HIT | 1.00 | 1.66% |
| BUY | retest2 | 2025-05-29 09:45:00 | 238.05 | 2025-06-05 11:15:00 | 241.95 | STOP_HIT | 1.00 | 1.64% |
| BUY | retest2 | 2025-05-29 11:30:00 | 237.64 | 2025-06-05 11:15:00 | 241.95 | STOP_HIT | 1.00 | 1.81% |
| BUY | retest2 | 2025-05-30 10:15:00 | 237.80 | 2025-06-05 11:15:00 | 241.95 | STOP_HIT | 1.00 | 1.75% |
| BUY | retest2 | 2025-05-30 12:45:00 | 238.85 | 2025-06-05 11:15:00 | 241.95 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2025-06-30 09:15:00 | 238.18 | 2025-07-01 09:15:00 | 236.19 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-06-30 12:00:00 | 238.09 | 2025-07-01 09:15:00 | 236.19 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-06-30 15:15:00 | 238.07 | 2025-07-01 09:15:00 | 236.19 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-07-14 09:30:00 | 233.61 | 2025-07-17 10:15:00 | 231.60 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-07-14 12:15:00 | 233.35 | 2025-07-17 10:15:00 | 231.60 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-07-15 09:30:00 | 233.50 | 2025-07-17 10:15:00 | 231.60 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-07-15 10:00:00 | 234.20 | 2025-07-17 10:15:00 | 231.60 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-07-22 09:45:00 | 227.03 | 2025-07-28 13:15:00 | 215.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 10:15:00 | 226.52 | 2025-07-28 13:15:00 | 215.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 09:45:00 | 227.03 | 2025-07-29 12:15:00 | 215.82 | STOP_HIT | 0.50 | 4.94% |
| SELL | retest2 | 2025-07-22 10:15:00 | 226.52 | 2025-07-29 12:15:00 | 215.82 | STOP_HIT | 0.50 | 4.72% |
| SELL | retest2 | 2025-08-08 10:15:00 | 208.24 | 2025-08-12 11:15:00 | 211.40 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-08-08 14:45:00 | 208.33 | 2025-08-12 11:15:00 | 211.40 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-08-11 11:30:00 | 208.49 | 2025-08-12 11:15:00 | 211.40 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-08-22 09:30:00 | 214.07 | 2025-08-22 14:15:00 | 211.91 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-08-22 11:00:00 | 214.08 | 2025-08-22 14:15:00 | 211.91 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-08-26 09:15:00 | 211.24 | 2025-09-01 13:15:00 | 210.86 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-08-26 10:30:00 | 211.54 | 2025-09-01 13:15:00 | 210.86 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2025-09-05 09:15:00 | 216.76 | 2025-09-12 12:15:00 | 216.25 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-09-05 13:15:00 | 215.20 | 2025-09-12 12:15:00 | 216.25 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2025-09-30 13:15:00 | 220.85 | 2025-10-01 09:15:00 | 232.27 | STOP_HIT | 1.00 | -5.17% |
| BUY | retest2 | 2025-10-06 12:45:00 | 231.40 | 2025-10-07 10:15:00 | 229.11 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-10-06 13:15:00 | 231.73 | 2025-10-07 10:15:00 | 229.11 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-10-06 15:15:00 | 231.50 | 2025-10-07 10:15:00 | 229.11 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-10-07 12:45:00 | 231.50 | 2025-10-08 09:15:00 | 228.61 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-10-13 14:15:00 | 230.50 | 2025-10-14 09:15:00 | 227.11 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-10-13 14:45:00 | 230.55 | 2025-10-14 09:15:00 | 227.11 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-10-23 13:15:00 | 229.12 | 2025-10-24 12:15:00 | 226.63 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-10-23 14:00:00 | 229.25 | 2025-10-24 12:15:00 | 226.63 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-10-23 15:15:00 | 229.00 | 2025-10-24 12:15:00 | 226.63 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-10-24 09:30:00 | 229.20 | 2025-10-24 12:15:00 | 226.63 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-12-01 09:15:00 | 240.72 | 2025-12-01 11:15:00 | 237.43 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-12-08 09:15:00 | 222.90 | 2025-12-08 14:15:00 | 211.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 09:15:00 | 222.90 | 2025-12-10 09:15:00 | 214.90 | STOP_HIT | 0.50 | 3.59% |
| BUY | retest2 | 2025-12-26 10:30:00 | 220.50 | 2026-01-05 13:15:00 | 225.08 | STOP_HIT | 1.00 | 2.08% |
| SELL | retest2 | 2026-01-13 12:00:00 | 214.35 | 2026-01-16 09:15:00 | 217.93 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-01-13 12:45:00 | 214.34 | 2026-01-16 09:15:00 | 217.93 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-01-13 14:00:00 | 213.66 | 2026-01-16 09:15:00 | 217.93 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2026-01-22 11:15:00 | 205.78 | 2026-01-23 13:15:00 | 195.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 10:45:00 | 206.48 | 2026-01-23 13:15:00 | 196.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 11:15:00 | 205.78 | 2026-01-27 14:15:00 | 197.80 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2026-01-23 10:45:00 | 206.48 | 2026-01-27 14:15:00 | 197.80 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2026-02-01 12:15:00 | 189.66 | 2026-02-02 10:15:00 | 183.12 | PARTIAL | 0.50 | 3.45% |
| SELL | retest2 | 2026-02-01 12:45:00 | 192.76 | 2026-02-02 10:15:00 | 183.29 | PARTIAL | 0.50 | 4.91% |
| SELL | retest2 | 2026-02-01 14:45:00 | 192.94 | 2026-02-02 11:15:00 | 180.18 | PARTIAL | 0.50 | 6.62% |
| SELL | retest2 | 2026-02-01 12:15:00 | 189.66 | 2026-02-02 14:15:00 | 188.00 | STOP_HIT | 0.50 | 0.88% |
| SELL | retest2 | 2026-02-01 12:45:00 | 192.76 | 2026-02-02 14:15:00 | 188.00 | STOP_HIT | 0.50 | 2.47% |
| SELL | retest2 | 2026-02-01 14:45:00 | 192.94 | 2026-02-02 14:15:00 | 188.00 | STOP_HIT | 0.50 | 2.56% |
| BUY | retest2 | 2026-02-05 13:15:00 | 194.60 | 2026-02-06 09:15:00 | 189.46 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2026-02-05 14:15:00 | 194.50 | 2026-02-06 09:15:00 | 189.46 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2026-02-26 10:15:00 | 189.70 | 2026-03-02 09:15:00 | 180.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 11:30:00 | 189.47 | 2026-03-02 09:15:00 | 180.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 15:15:00 | 189.00 | 2026-03-02 09:15:00 | 179.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 10:15:00 | 189.70 | 2026-03-05 14:15:00 | 177.87 | STOP_HIT | 0.50 | 6.24% |
| SELL | retest2 | 2026-02-26 11:30:00 | 189.47 | 2026-03-05 14:15:00 | 177.87 | STOP_HIT | 0.50 | 6.12% |
| SELL | retest2 | 2026-02-26 15:15:00 | 189.00 | 2026-03-05 14:15:00 | 177.87 | STOP_HIT | 0.50 | 5.89% |
| BUY | retest2 | 2026-03-12 12:15:00 | 178.70 | 2026-03-13 09:15:00 | 173.35 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2026-03-12 13:15:00 | 178.71 | 2026-03-13 09:15:00 | 173.35 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2026-03-17 11:15:00 | 171.23 | 2026-03-18 09:15:00 | 174.67 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2026-03-20 12:15:00 | 173.07 | 2026-03-23 12:15:00 | 164.54 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2026-03-20 14:15:00 | 173.20 | 2026-03-23 15:15:00 | 164.42 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2026-03-20 12:15:00 | 173.07 | 2026-03-24 11:15:00 | 167.10 | STOP_HIT | 0.50 | 3.45% |
| SELL | retest2 | 2026-03-20 14:15:00 | 173.20 | 2026-03-24 11:15:00 | 167.10 | STOP_HIT | 0.50 | 3.52% |
| SELL | retest2 | 2026-04-02 09:15:00 | 162.20 | 2026-04-06 09:15:00 | 167.49 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2026-04-07 11:45:00 | 170.32 | 2026-04-13 12:15:00 | 187.35 | TARGET_HIT | 1.00 | 10.00% |
