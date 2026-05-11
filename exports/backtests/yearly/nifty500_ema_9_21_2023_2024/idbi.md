# IDBI Bank Ltd. (IDBI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 74.79
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 216 |
| ALERT1 | 139 |
| ALERT2 | 136 |
| ALERT2_SKIP | 61 |
| ALERT3 | 366 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 144 |
| PARTIAL | 18 |
| TARGET_HIT | 8 |
| STOP_HIT | 140 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 166 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 56 / 110
- **Target hits / Stop hits / Partials:** 8 / 140 / 18
- **Avg / median % per leg:** 0.34% / -0.58%
- **Sum % (uncompounded):** 56.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 82 | 17 | 20.7% | 3 | 77 | 2 | -0.41% | -33.6% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 1 | 1 | 2 | 6.31% | 25.2% |
| BUY @ 3rd Alert (retest2) | 78 | 13 | 16.7% | 2 | 76 | 0 | -0.75% | -58.8% |
| SELL (all) | 84 | 39 | 46.4% | 5 | 63 | 16 | 1.07% | 90.0% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.26% | -2.5% |
| SELL @ 3rd Alert (retest2) | 82 | 39 | 47.6% | 5 | 61 | 16 | 1.13% | 92.5% |
| retest1 (combined) | 6 | 4 | 66.7% | 1 | 3 | 2 | 3.79% | 22.7% |
| retest2 (combined) | 160 | 52 | 32.5% | 7 | 137 | 16 | 0.21% | 33.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 10:15:00 | 53.95 | 53.49 | 53.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 10:15:00 | 54.55 | 53.82 | 53.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-18 14:15:00 | 56.15 | 56.29 | 55.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-18 15:00:00 | 56.15 | 56.29 | 55.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 09:15:00 | 54.90 | 55.95 | 55.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 10:00:00 | 54.90 | 55.95 | 55.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 10:15:00 | 55.15 | 55.79 | 55.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 11:15:00 | 54.90 | 55.79 | 55.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 11:15:00 | 54.90 | 55.61 | 55.32 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-22 09:15:00 | 54.95 | 55.17 | 55.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 12:15:00 | 54.25 | 54.76 | 54.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-24 14:15:00 | 54.75 | 54.71 | 54.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-24 15:00:00 | 54.75 | 54.71 | 54.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 15:15:00 | 54.75 | 54.72 | 54.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 09:15:00 | 54.60 | 54.72 | 54.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 54.70 | 54.71 | 54.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 09:45:00 | 54.70 | 54.71 | 54.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 10:15:00 | 54.75 | 54.72 | 54.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 10:30:00 | 54.75 | 54.72 | 54.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 11:15:00 | 55.55 | 54.89 | 54.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 11:45:00 | 55.85 | 54.89 | 54.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2023-05-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 12:15:00 | 55.20 | 54.95 | 54.93 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 14:15:00 | 54.75 | 55.04 | 55.04 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 13:15:00 | 55.25 | 55.03 | 55.02 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 15:15:00 | 55.00 | 55.04 | 55.05 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 09:15:00 | 55.10 | 55.05 | 55.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 10:15:00 | 55.30 | 55.10 | 55.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-31 11:15:00 | 55.05 | 55.09 | 55.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 11:15:00 | 55.05 | 55.09 | 55.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 11:15:00 | 55.05 | 55.09 | 55.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 11:45:00 | 54.95 | 55.09 | 55.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2023-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 12:15:00 | 54.90 | 55.05 | 55.06 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 15:15:00 | 55.10 | 55.06 | 55.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 09:15:00 | 55.45 | 55.14 | 55.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 13:15:00 | 55.30 | 55.41 | 55.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 13:15:00 | 55.30 | 55.41 | 55.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 13:15:00 | 55.30 | 55.41 | 55.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-01 14:00:00 | 55.30 | 55.41 | 55.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 14:15:00 | 55.35 | 55.40 | 55.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 09:15:00 | 56.05 | 55.39 | 55.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 15:15:00 | 55.55 | 55.48 | 55.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 12:30:00 | 55.55 | 55.58 | 55.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 13:15:00 | 55.55 | 55.58 | 55.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-06 13:15:00 | 55.45 | 55.56 | 55.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2023-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 13:15:00 | 55.45 | 55.56 | 55.57 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 09:15:00 | 56.15 | 55.64 | 55.60 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-06-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 11:15:00 | 54.70 | 55.55 | 55.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 09:15:00 | 54.10 | 54.89 | 55.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 13:15:00 | 53.70 | 53.67 | 54.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-12 13:45:00 | 53.70 | 53.67 | 54.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 15:15:00 | 54.20 | 53.78 | 54.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 09:15:00 | 54.50 | 53.78 | 54.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 54.20 | 53.87 | 54.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 09:30:00 | 54.50 | 53.87 | 54.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 10:15:00 | 54.30 | 53.95 | 54.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 10:30:00 | 54.35 | 53.95 | 54.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 13:15:00 | 53.80 | 54.01 | 54.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 13:30:00 | 54.00 | 54.01 | 54.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 14:15:00 | 54.30 | 54.07 | 54.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 15:00:00 | 54.30 | 54.07 | 54.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 15:15:00 | 54.00 | 54.06 | 54.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-14 09:15:00 | 54.15 | 54.06 | 54.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 09:15:00 | 54.15 | 54.08 | 54.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-14 10:45:00 | 53.95 | 54.05 | 54.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-14 14:30:00 | 53.90 | 54.11 | 54.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-14 15:15:00 | 53.95 | 54.11 | 54.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-15 11:15:00 | 53.90 | 54.07 | 54.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 53.75 | 53.73 | 53.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-16 12:30:00 | 53.55 | 53.72 | 53.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-16 13:30:00 | 53.55 | 53.69 | 53.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-16 14:00:00 | 53.55 | 53.69 | 53.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-16 15:15:00 | 53.95 | 53.77 | 53.84 | SL hit (close>static) qty=1.00 sl=53.90 alert=retest2 |

### Cycle 13 — BUY (started 2023-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-19 09:15:00 | 54.65 | 53.94 | 53.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 09:15:00 | 55.25 | 54.58 | 54.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-21 15:15:00 | 54.70 | 54.75 | 54.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 09:15:00 | 55.05 | 54.81 | 54.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 55.05 | 54.81 | 54.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 09:30:00 | 54.70 | 54.81 | 54.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 54.70 | 54.88 | 54.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 12:00:00 | 54.70 | 54.88 | 54.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 12:15:00 | 54.55 | 54.82 | 54.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 12:45:00 | 54.45 | 54.82 | 54.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 13:15:00 | 54.60 | 54.77 | 54.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 14:15:00 | 54.50 | 54.77 | 54.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 14:15:00 | 54.55 | 54.73 | 54.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 14:30:00 | 54.60 | 54.73 | 54.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 15:15:00 | 54.45 | 54.67 | 54.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 09:15:00 | 53.70 | 54.67 | 54.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2023-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 09:15:00 | 53.95 | 54.53 | 54.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-26 12:15:00 | 53.30 | 53.67 | 54.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 14:15:00 | 53.60 | 53.60 | 53.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 14:30:00 | 53.60 | 53.60 | 53.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 53.85 | 53.68 | 53.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:30:00 | 53.90 | 53.68 | 53.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 53.55 | 53.65 | 53.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:45:00 | 53.85 | 53.65 | 53.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 09:15:00 | 53.80 | 53.65 | 53.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-28 13:45:00 | 53.60 | 53.67 | 53.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-30 09:15:00 | 54.85 | 53.91 | 53.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2023-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 09:15:00 | 54.85 | 53.91 | 53.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 10:15:00 | 55.10 | 54.15 | 53.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 13:15:00 | 56.65 | 56.65 | 56.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-04 14:00:00 | 56.65 | 56.65 | 56.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 57.50 | 58.12 | 57.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 10:00:00 | 57.50 | 58.12 | 57.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 57.15 | 57.92 | 57.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 10:45:00 | 57.05 | 57.92 | 57.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2023-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 13:15:00 | 57.05 | 57.44 | 57.49 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 09:15:00 | 58.95 | 57.74 | 57.62 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-07-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 12:15:00 | 58.20 | 58.73 | 58.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 13:15:00 | 56.90 | 58.37 | 58.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 13:15:00 | 57.70 | 57.57 | 57.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-14 14:00:00 | 57.70 | 57.57 | 57.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 14:15:00 | 57.75 | 57.60 | 57.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 15:00:00 | 57.75 | 57.60 | 57.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 57.65 | 57.61 | 57.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:45:00 | 57.85 | 57.61 | 57.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 10:15:00 | 57.80 | 57.65 | 57.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 11:00:00 | 57.80 | 57.65 | 57.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 12:15:00 | 57.85 | 57.67 | 57.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 13:00:00 | 57.85 | 57.67 | 57.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 13:15:00 | 57.70 | 57.68 | 57.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-18 11:30:00 | 57.45 | 57.79 | 57.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-19 13:45:00 | 57.50 | 57.45 | 57.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-19 14:15:00 | 57.50 | 57.45 | 57.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-20 09:15:00 | 58.35 | 57.64 | 57.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2023-07-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 09:15:00 | 58.35 | 57.64 | 57.64 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 09:15:00 | 57.45 | 57.69 | 57.71 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 09:15:00 | 58.45 | 57.71 | 57.68 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 10:15:00 | 57.20 | 57.75 | 57.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-25 13:15:00 | 56.85 | 57.44 | 57.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 10:15:00 | 57.20 | 57.20 | 57.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-26 11:00:00 | 57.20 | 57.20 | 57.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 11:15:00 | 57.40 | 57.24 | 57.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 12:00:00 | 57.40 | 57.24 | 57.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 12:15:00 | 57.25 | 57.24 | 57.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 12:30:00 | 57.35 | 57.24 | 57.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 13:15:00 | 57.25 | 57.24 | 57.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 13:30:00 | 57.25 | 57.24 | 57.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 14:15:00 | 57.50 | 57.29 | 57.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 15:00:00 | 57.50 | 57.29 | 57.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 15:15:00 | 57.50 | 57.33 | 57.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 09:15:00 | 58.20 | 57.33 | 57.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2023-07-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 09:15:00 | 58.60 | 57.59 | 57.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 09:15:00 | 61.35 | 58.68 | 58.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 10:15:00 | 61.10 | 61.40 | 60.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-02 11:00:00 | 61.10 | 61.40 | 60.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 59.80 | 60.94 | 60.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 14:00:00 | 59.80 | 60.94 | 60.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 60.50 | 60.85 | 60.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 09:15:00 | 61.15 | 60.78 | 60.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 09:45:00 | 60.80 | 60.95 | 60.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-09 11:15:00 | 63.15 | 64.02 | 64.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2023-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 11:15:00 | 63.15 | 64.02 | 64.12 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 09:15:00 | 64.95 | 63.99 | 63.93 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 10:15:00 | 63.45 | 64.00 | 64.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-16 10:15:00 | 62.65 | 63.40 | 63.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 13:15:00 | 60.80 | 60.78 | 61.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-21 14:00:00 | 60.80 | 60.78 | 61.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 09:15:00 | 61.80 | 60.95 | 61.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 09:45:00 | 61.95 | 60.95 | 61.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 10:15:00 | 61.90 | 61.14 | 61.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 11:00:00 | 61.90 | 61.14 | 61.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2023-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 14:15:00 | 61.65 | 61.40 | 61.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 15:15:00 | 61.80 | 61.48 | 61.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 11:15:00 | 62.25 | 62.31 | 62.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-24 12:00:00 | 62.25 | 62.31 | 62.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 12:15:00 | 61.95 | 62.24 | 62.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 13:00:00 | 61.95 | 62.24 | 62.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 13:15:00 | 61.90 | 62.17 | 62.01 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 09:15:00 | 61.10 | 61.79 | 61.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 10:15:00 | 60.50 | 61.53 | 61.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-30 09:15:00 | 59.85 | 59.84 | 60.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 10:15:00 | 60.55 | 59.98 | 60.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 10:15:00 | 60.55 | 59.98 | 60.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 11:00:00 | 60.55 | 59.98 | 60.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 11:15:00 | 60.10 | 60.01 | 60.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-30 14:00:00 | 59.95 | 60.02 | 60.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-30 14:30:00 | 59.90 | 59.96 | 60.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-31 09:45:00 | 59.90 | 59.88 | 60.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-01 12:45:00 | 59.95 | 59.82 | 59.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 13:15:00 | 60.10 | 59.88 | 59.94 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-09-04 09:15:00 | 63.60 | 60.69 | 60.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2023-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 09:15:00 | 63.60 | 60.69 | 60.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 09:15:00 | 68.65 | 64.56 | 62.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 11:15:00 | 69.55 | 69.71 | 68.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-07 11:30:00 | 69.60 | 69.71 | 68.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 13:15:00 | 70.00 | 70.33 | 69.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 14:00:00 | 70.00 | 70.33 | 69.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 14:15:00 | 70.30 | 70.32 | 70.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 15:15:00 | 70.25 | 70.32 | 70.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 15:15:00 | 70.25 | 70.31 | 70.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:15:00 | 69.15 | 70.31 | 70.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 66.75 | 69.60 | 69.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 64.95 | 66.61 | 67.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 12:15:00 | 66.85 | 66.54 | 67.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 13:00:00 | 66.85 | 66.54 | 67.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 13:15:00 | 68.05 | 66.84 | 67.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 13:45:00 | 67.75 | 66.84 | 67.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 14:15:00 | 67.85 | 67.04 | 67.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 14:30:00 | 68.15 | 67.04 | 67.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 68.40 | 67.41 | 67.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:45:00 | 68.75 | 67.41 | 67.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2023-09-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 11:15:00 | 68.95 | 67.95 | 67.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 09:15:00 | 70.70 | 68.91 | 68.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 14:15:00 | 69.55 | 69.68 | 69.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-15 15:00:00 | 69.55 | 69.68 | 69.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 71.75 | 70.02 | 69.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-18 10:15:00 | 71.95 | 70.02 | 69.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-18 10:45:00 | 72.35 | 70.54 | 69.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-18 13:30:00 | 72.50 | 71.11 | 70.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-20 10:30:00 | 72.00 | 71.70 | 70.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 12:15:00 | 71.00 | 71.53 | 70.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 13:00:00 | 71.00 | 71.53 | 70.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 13:15:00 | 70.55 | 71.33 | 70.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 13:45:00 | 70.50 | 71.33 | 70.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 14:15:00 | 70.75 | 71.21 | 70.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 15:15:00 | 71.00 | 71.21 | 70.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 15:15:00 | 71.00 | 71.17 | 70.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 09:15:00 | 71.60 | 71.17 | 70.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 09:15:00 | 71.40 | 71.22 | 70.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-21 10:30:00 | 72.70 | 71.45 | 71.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-21 14:15:00 | 69.50 | 70.78 | 70.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2023-09-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 14:15:00 | 69.50 | 70.78 | 70.83 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 11:15:00 | 71.75 | 70.94 | 70.86 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-25 11:15:00 | 70.45 | 70.84 | 70.89 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 10:15:00 | 71.00 | 70.90 | 70.89 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 11:15:00 | 70.40 | 70.80 | 70.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-26 12:15:00 | 69.90 | 70.62 | 70.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-27 10:15:00 | 70.40 | 70.17 | 70.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-27 10:15:00 | 70.40 | 70.17 | 70.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 70.40 | 70.17 | 70.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 10:30:00 | 70.55 | 70.17 | 70.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 11:15:00 | 70.00 | 70.14 | 70.40 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 14:15:00 | 71.00 | 70.56 | 70.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 09:15:00 | 71.65 | 70.88 | 70.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 13:15:00 | 70.95 | 71.20 | 70.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 13:15:00 | 70.95 | 71.20 | 70.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 70.95 | 71.20 | 70.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 14:00:00 | 70.95 | 71.20 | 70.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 69.90 | 70.94 | 70.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 15:00:00 | 69.90 | 70.94 | 70.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 70.20 | 70.79 | 70.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 09:15:00 | 70.65 | 70.79 | 70.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-29 15:15:00 | 70.65 | 70.80 | 70.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2023-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 15:15:00 | 70.65 | 70.80 | 70.81 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 09:15:00 | 70.95 | 70.83 | 70.82 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 11:15:00 | 70.45 | 70.77 | 70.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 12:15:00 | 69.65 | 70.32 | 70.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 09:15:00 | 69.25 | 69.07 | 69.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-06 10:15:00 | 69.55 | 69.07 | 69.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 10:15:00 | 69.35 | 69.13 | 69.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-06 11:15:00 | 69.15 | 69.13 | 69.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-06 15:15:00 | 69.80 | 69.42 | 69.53 | SL hit (close>static) qty=1.00 sl=69.75 alert=retest2 |

### Cycle 41 — BUY (started 2023-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 10:15:00 | 69.35 | 68.41 | 68.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 14:15:00 | 70.40 | 69.34 | 69.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 15:15:00 | 70.65 | 70.72 | 70.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-16 09:15:00 | 70.50 | 70.72 | 70.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 70.20 | 70.62 | 70.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 09:45:00 | 70.30 | 70.62 | 70.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 10:15:00 | 70.10 | 70.51 | 70.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 11:00:00 | 70.10 | 70.51 | 70.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 11:15:00 | 69.95 | 70.40 | 70.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 11:45:00 | 69.85 | 70.40 | 70.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 12:15:00 | 69.75 | 70.27 | 70.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 12:45:00 | 69.75 | 70.27 | 70.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 15:15:00 | 70.00 | 70.09 | 70.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 09:15:00 | 70.75 | 70.09 | 70.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 11:00:00 | 70.55 | 70.24 | 70.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 09:30:00 | 71.50 | 70.39 | 70.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 12:15:00 | 70.55 | 70.35 | 70.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 12:15:00 | 70.65 | 70.41 | 70.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-10-18 14:15:00 | 68.35 | 69.96 | 70.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2023-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 14:15:00 | 68.35 | 69.96 | 70.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 11:15:00 | 68.00 | 68.65 | 69.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 14:15:00 | 64.45 | 63.55 | 64.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-25 15:00:00 | 64.45 | 63.55 | 64.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 63.30 | 62.82 | 63.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:30:00 | 63.45 | 62.82 | 63.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 63.85 | 63.03 | 63.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 11:00:00 | 63.85 | 63.03 | 63.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 11:15:00 | 63.35 | 63.09 | 63.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 12:15:00 | 63.20 | 63.09 | 63.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-01 15:15:00 | 60.04 | 60.95 | 61.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-11-02 09:15:00 | 61.90 | 61.14 | 61.59 | SL hit (close>ema200) qty=0.50 sl=61.14 alert=retest2 |

### Cycle 43 — BUY (started 2023-11-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 14:15:00 | 62.50 | 61.86 | 61.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 09:15:00 | 62.95 | 62.18 | 61.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 12:15:00 | 62.20 | 62.30 | 62.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 12:15:00 | 62.20 | 62.30 | 62.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 12:15:00 | 62.20 | 62.30 | 62.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 13:00:00 | 62.20 | 62.30 | 62.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 14:15:00 | 62.20 | 62.28 | 62.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 14:45:00 | 62.10 | 62.28 | 62.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 15:15:00 | 62.05 | 62.23 | 62.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 09:15:00 | 62.30 | 62.23 | 62.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 09:45:00 | 62.40 | 62.28 | 62.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 13:30:00 | 62.25 | 62.45 | 62.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 09:15:00 | 62.35 | 62.39 | 62.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 62.45 | 62.40 | 62.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 09:30:00 | 62.15 | 62.40 | 62.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 11:15:00 | 62.45 | 62.46 | 62.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 11:45:00 | 62.50 | 62.46 | 62.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 12:15:00 | 62.75 | 62.52 | 62.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 12:45:00 | 62.80 | 62.52 | 62.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 13:15:00 | 62.20 | 62.45 | 62.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 14:00:00 | 62.20 | 62.45 | 62.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 14:15:00 | 62.75 | 62.51 | 62.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 09:15:00 | 64.00 | 62.56 | 62.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 15:00:00 | 62.90 | 63.21 | 63.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-10 09:15:00 | 63.00 | 63.12 | 63.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-10 09:15:00 | 62.75 | 63.05 | 63.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2023-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 09:15:00 | 62.75 | 63.05 | 63.05 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 15:15:00 | 63.40 | 63.08 | 63.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-12 18:15:00 | 63.85 | 63.23 | 63.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 13:15:00 | 65.85 | 65.96 | 65.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-16 13:45:00 | 65.75 | 65.96 | 65.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 63.00 | 65.27 | 65.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 10:15:00 | 63.05 | 65.27 | 65.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2023-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 10:15:00 | 63.50 | 64.91 | 65.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 14:15:00 | 62.60 | 63.77 | 64.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 14:15:00 | 62.30 | 62.19 | 62.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-21 15:00:00 | 62.30 | 62.19 | 62.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 61.00 | 60.80 | 61.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 09:30:00 | 61.20 | 60.80 | 61.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 60.95 | 60.78 | 61.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-24 15:00:00 | 60.60 | 60.76 | 61.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-28 10:15:00 | 61.60 | 60.94 | 61.03 | SL hit (close>static) qty=1.00 sl=61.25 alert=retest2 |

### Cycle 47 — BUY (started 2023-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 12:15:00 | 61.60 | 61.15 | 61.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 09:15:00 | 62.75 | 62.10 | 61.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 15:15:00 | 62.45 | 62.55 | 62.33 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 09:15:00 | 63.15 | 62.55 | 62.33 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-07 10:15:00 | 66.31 | 64.76 | 64.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-12-08 10:15:00 | 66.45 | 66.46 | 65.51 | SL hit (close<ema200) qty=0.50 sl=66.46 alert=retest1 |

### Cycle 48 — SELL (started 2023-12-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 12:15:00 | 65.85 | 66.36 | 66.43 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2023-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 09:15:00 | 66.90 | 66.36 | 66.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-18 09:15:00 | 70.85 | 67.63 | 67.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 09:15:00 | 69.10 | 69.41 | 68.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-19 10:00:00 | 69.10 | 69.41 | 68.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 11:15:00 | 69.10 | 69.43 | 69.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 11:45:00 | 69.15 | 69.43 | 69.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 68.35 | 69.21 | 68.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:00:00 | 68.35 | 69.21 | 68.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 66.10 | 68.59 | 68.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 65.35 | 67.94 | 68.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 15:15:00 | 66.05 | 65.97 | 66.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-22 09:15:00 | 66.60 | 65.97 | 66.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 66.70 | 66.12 | 66.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 12:15:00 | 66.25 | 66.28 | 66.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-27 10:30:00 | 66.30 | 66.02 | 66.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-27 11:00:00 | 66.25 | 66.02 | 66.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-27 12:00:00 | 66.25 | 66.07 | 66.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 65.90 | 66.03 | 66.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 12:45:00 | 66.00 | 66.03 | 66.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 14:15:00 | 66.05 | 66.01 | 66.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 15:00:00 | 66.05 | 66.01 | 66.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 09:15:00 | 66.05 | 66.02 | 66.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-28 09:30:00 | 66.55 | 66.02 | 66.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 10:15:00 | 66.20 | 66.05 | 66.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-28 11:00:00 | 66.20 | 66.05 | 66.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 11:15:00 | 66.00 | 66.04 | 66.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-12-29 09:15:00 | 67.50 | 66.41 | 66.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2023-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 09:15:00 | 67.50 | 66.41 | 66.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 09:15:00 | 68.30 | 67.34 | 66.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 14:15:00 | 67.70 | 67.75 | 67.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-01 14:30:00 | 67.80 | 67.75 | 67.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 15:15:00 | 67.50 | 67.70 | 67.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 09:30:00 | 68.25 | 67.67 | 67.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-02 10:15:00 | 66.70 | 67.47 | 67.28 | SL hit (close<static) qty=1.00 sl=67.30 alert=retest2 |

### Cycle 52 — SELL (started 2024-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 10:15:00 | 67.80 | 68.43 | 68.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 14:15:00 | 67.20 | 67.94 | 68.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 09:15:00 | 67.55 | 67.06 | 67.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 09:15:00 | 67.55 | 67.06 | 67.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 67.55 | 67.06 | 67.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 10:15:00 | 68.05 | 67.06 | 67.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 10:15:00 | 67.65 | 67.18 | 67.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 10:30:00 | 67.90 | 67.18 | 67.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 11:15:00 | 66.95 | 67.13 | 67.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 11:30:00 | 66.95 | 67.13 | 67.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 68.00 | 67.15 | 67.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-12 09:45:00 | 68.30 | 67.15 | 67.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2024-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 10:15:00 | 69.25 | 67.57 | 67.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 09:15:00 | 71.95 | 69.46 | 68.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 12:15:00 | 69.85 | 70.05 | 69.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-16 13:00:00 | 69.85 | 70.05 | 69.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 69.75 | 70.17 | 69.59 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 14:15:00 | 68.75 | 69.31 | 69.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 67.80 | 68.92 | 69.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 11:15:00 | 69.15 | 68.88 | 69.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 11:15:00 | 69.15 | 68.88 | 69.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 11:15:00 | 69.15 | 68.88 | 69.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 12:00:00 | 69.15 | 68.88 | 69.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 12:15:00 | 69.80 | 69.06 | 69.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 13:00:00 | 69.80 | 69.06 | 69.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 13:15:00 | 69.45 | 69.14 | 69.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-18 14:15:00 | 69.00 | 69.14 | 69.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-18 15:15:00 | 69.40 | 69.23 | 69.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2024-01-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 15:15:00 | 69.40 | 69.23 | 69.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 09:15:00 | 69.80 | 69.34 | 69.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-19 15:15:00 | 69.50 | 69.55 | 69.42 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-20 09:15:00 | 70.25 | 69.55 | 69.42 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-20 13:15:00 | 73.76 | 71.55 | 70.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-01-20 14:15:00 | 77.28 | 73.06 | 71.28 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 56 — SELL (started 2024-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 09:15:00 | 90.35 | 92.91 | 93.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 10:15:00 | 88.30 | 91.99 | 92.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 10:15:00 | 85.60 | 84.93 | 87.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 11:00:00 | 85.60 | 84.93 | 87.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 12:15:00 | 85.50 | 84.84 | 85.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 13:00:00 | 85.50 | 84.84 | 85.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 86.20 | 85.22 | 85.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 15:00:00 | 86.20 | 85.22 | 85.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 15:15:00 | 86.70 | 85.52 | 85.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 09:15:00 | 87.30 | 85.52 | 85.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2024-02-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 11:15:00 | 88.25 | 86.44 | 86.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 13:15:00 | 91.05 | 87.59 | 86.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 14:15:00 | 91.50 | 91.70 | 90.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-19 15:00:00 | 91.50 | 91.70 | 90.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 13:15:00 | 90.55 | 91.39 | 90.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 13:45:00 | 90.40 | 91.39 | 90.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 14:15:00 | 90.60 | 91.23 | 90.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 14:30:00 | 90.45 | 91.23 | 90.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 15:15:00 | 90.90 | 91.16 | 90.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 09:30:00 | 91.30 | 91.25 | 90.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 11:30:00 | 91.25 | 91.18 | 90.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 14:15:00 | 89.90 | 90.84 | 90.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 14:15:00 | 89.90 | 90.84 | 90.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 09:15:00 | 89.20 | 90.35 | 90.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 11:15:00 | 90.65 | 90.22 | 90.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 11:15:00 | 90.65 | 90.22 | 90.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 11:15:00 | 90.65 | 90.22 | 90.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 12:00:00 | 90.65 | 90.22 | 90.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 12:15:00 | 90.20 | 90.21 | 90.47 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-02-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 14:15:00 | 91.80 | 90.66 | 90.64 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 13:15:00 | 90.50 | 90.80 | 90.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 14:15:00 | 90.05 | 90.65 | 90.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 12:15:00 | 85.95 | 85.62 | 86.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 12:45:00 | 85.90 | 85.62 | 86.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 86.55 | 85.96 | 86.87 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-03-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 14:15:00 | 87.75 | 87.18 | 87.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 09:15:00 | 87.95 | 87.41 | 87.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 10:15:00 | 88.05 | 88.09 | 87.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-04 11:00:00 | 88.05 | 88.09 | 87.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 11:15:00 | 87.65 | 88.00 | 87.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 11:45:00 | 87.75 | 88.00 | 87.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 12:15:00 | 88.00 | 88.00 | 87.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 10:45:00 | 88.40 | 87.83 | 87.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-06 09:15:00 | 87.00 | 88.34 | 88.11 | SL hit (close<static) qty=1.00 sl=87.65 alert=retest2 |

### Cycle 62 — SELL (started 2024-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 11:15:00 | 86.00 | 87.68 | 87.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 14:15:00 | 85.30 | 86.66 | 87.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 86.55 | 86.46 | 87.08 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-07 12:15:00 | 85.50 | 86.20 | 86.84 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-07 15:15:00 | 85.45 | 86.02 | 86.60 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 86.55 | 86.04 | 86.50 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-03-11 09:15:00 | 86.55 | 86.04 | 86.50 | SL hit (close>ema400) qty=1.00 sl=86.50 alert=retest1 |

### Cycle 63 — BUY (started 2024-03-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 15:15:00 | 83.90 | 80.54 | 80.19 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 09:15:00 | 78.65 | 80.79 | 81.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 14:15:00 | 78.40 | 79.65 | 80.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 80.50 | 79.61 | 80.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 80.50 | 79.61 | 80.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 80.50 | 79.61 | 80.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 10:15:00 | 80.75 | 79.61 | 80.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 10:15:00 | 80.75 | 79.84 | 80.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 12:00:00 | 80.40 | 79.95 | 80.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-22 09:15:00 | 81.30 | 80.51 | 80.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2024-03-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 09:15:00 | 81.30 | 80.51 | 80.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 10:15:00 | 81.50 | 80.71 | 80.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 13:15:00 | 81.00 | 81.01 | 80.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-22 14:00:00 | 81.00 | 81.01 | 80.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 12:15:00 | 80.90 | 81.07 | 80.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 12:30:00 | 80.85 | 81.07 | 80.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 13:15:00 | 81.15 | 81.09 | 80.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 13:30:00 | 80.65 | 81.09 | 80.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 14:15:00 | 80.65 | 81.00 | 80.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 15:00:00 | 80.65 | 81.00 | 80.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 15:15:00 | 80.60 | 80.92 | 80.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 09:15:00 | 80.90 | 80.92 | 80.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 10:15:00 | 80.85 | 80.88 | 80.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-27 10:15:00 | 80.45 | 80.79 | 80.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 10:15:00 | 80.45 | 80.79 | 80.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 11:15:00 | 79.90 | 80.61 | 80.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 09:15:00 | 81.00 | 80.32 | 80.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 09:15:00 | 81.00 | 80.32 | 80.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 81.00 | 80.32 | 80.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 09:45:00 | 81.05 | 80.32 | 80.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 10:15:00 | 81.40 | 80.54 | 80.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 10:45:00 | 81.55 | 80.54 | 80.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2024-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 11:15:00 | 81.10 | 80.65 | 80.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 12:15:00 | 82.30 | 80.98 | 80.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 14:15:00 | 81.10 | 81.24 | 80.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 14:15:00 | 81.10 | 81.24 | 80.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 14:15:00 | 81.10 | 81.24 | 80.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 14:45:00 | 81.10 | 81.24 | 80.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 15:15:00 | 80.80 | 81.15 | 80.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 09:15:00 | 82.75 | 81.15 | 80.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-12 12:15:00 | 87.70 | 88.55 | 88.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-04-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 12:15:00 | 87.70 | 88.55 | 88.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 14:15:00 | 86.70 | 88.03 | 88.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 86.60 | 85.80 | 86.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 86.60 | 85.80 | 86.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 86.60 | 85.80 | 86.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:45:00 | 87.40 | 85.80 | 86.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 86.65 | 85.97 | 86.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:15:00 | 86.45 | 85.97 | 86.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 86.15 | 86.00 | 86.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 12:15:00 | 85.90 | 86.00 | 86.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 10:15:00 | 85.95 | 85.81 | 86.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-18 12:15:00 | 87.00 | 86.20 | 86.35 | SL hit (close>static) qty=1.00 sl=86.95 alert=retest2 |

### Cycle 69 — BUY (started 2024-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 13:15:00 | 86.00 | 85.25 | 85.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 86.75 | 85.81 | 85.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 11:15:00 | 86.60 | 86.63 | 86.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-24 11:30:00 | 86.65 | 86.63 | 86.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 87.40 | 86.82 | 86.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 13:00:00 | 87.95 | 87.11 | 86.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 15:15:00 | 88.20 | 87.37 | 86.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 12:15:00 | 89.55 | 90.38 | 90.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-05-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 12:15:00 | 89.55 | 90.38 | 90.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 09:15:00 | 88.70 | 89.70 | 90.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 10:15:00 | 90.00 | 89.76 | 90.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 10:15:00 | 90.00 | 89.76 | 90.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 90.00 | 89.76 | 90.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 11:00:00 | 90.00 | 89.76 | 90.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 11:15:00 | 89.60 | 89.73 | 89.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 12:15:00 | 89.25 | 89.73 | 89.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-08 09:15:00 | 84.79 | 86.42 | 87.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-08 11:15:00 | 86.55 | 86.39 | 87.46 | SL hit (close>ema200) qty=0.50 sl=86.39 alert=retest2 |

### Cycle 71 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 84.35 | 83.50 | 83.43 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 14:15:00 | 83.35 | 83.71 | 83.76 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 10:15:00 | 84.25 | 83.79 | 83.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 13:15:00 | 84.50 | 84.05 | 83.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 11:15:00 | 87.25 | 87.36 | 86.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-22 11:30:00 | 87.25 | 87.36 | 86.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 87.20 | 87.47 | 87.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 14:30:00 | 87.25 | 87.47 | 87.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 87.05 | 87.37 | 87.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:00:00 | 87.05 | 87.37 | 87.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 87.75 | 87.45 | 87.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:30:00 | 88.10 | 87.44 | 87.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 12:15:00 | 88.25 | 87.50 | 87.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 12:15:00 | 86.30 | 88.08 | 87.99 | SL hit (close<static) qty=1.00 sl=87.05 alert=retest2 |

### Cycle 74 — SELL (started 2024-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 13:15:00 | 86.25 | 87.72 | 87.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 13:15:00 | 85.85 | 86.40 | 86.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 12:15:00 | 84.40 | 84.34 | 85.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 12:45:00 | 84.40 | 84.34 | 85.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 85.80 | 84.63 | 85.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 85.80 | 84.63 | 85.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 85.65 | 84.84 | 85.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:45:00 | 85.50 | 84.84 | 85.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 85.35 | 84.94 | 85.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 89.45 | 84.94 | 85.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 91.40 | 86.23 | 85.80 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 79.15 | 86.47 | 87.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 77.70 | 81.72 | 84.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 14:15:00 | 81.20 | 81.11 | 82.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:45:00 | 81.10 | 81.11 | 82.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 84.25 | 81.76 | 82.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 84.05 | 81.76 | 82.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 84.75 | 82.36 | 83.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:45:00 | 84.70 | 82.36 | 83.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 83.40 | 82.86 | 83.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:30:00 | 84.00 | 82.86 | 83.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 15:15:00 | 83.80 | 83.18 | 83.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:15:00 | 84.50 | 83.18 | 83.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 84.45 | 83.44 | 83.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 14:15:00 | 85.75 | 84.37 | 83.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 14:15:00 | 86.75 | 87.06 | 86.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 15:00:00 | 86.75 | 87.06 | 86.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 86.87 | 87.37 | 87.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:00:00 | 86.87 | 87.37 | 87.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 86.89 | 87.27 | 87.03 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 15:15:00 | 86.48 | 86.84 | 86.88 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-06-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 11:15:00 | 87.16 | 86.94 | 86.92 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-06-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 14:15:00 | 86.51 | 86.83 | 86.87 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 09:15:00 | 87.23 | 86.90 | 86.88 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 86.51 | 86.82 | 86.84 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 09:15:00 | 87.66 | 86.89 | 86.85 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 86.36 | 86.77 | 86.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 10:15:00 | 86.19 | 86.65 | 86.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 10:15:00 | 86.10 | 86.01 | 86.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 10:15:00 | 86.10 | 86.01 | 86.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 86.10 | 86.01 | 86.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:00:00 | 86.10 | 86.01 | 86.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 86.09 | 85.98 | 86.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 13:45:00 | 86.12 | 85.98 | 86.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 85.93 | 85.91 | 86.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:00:00 | 85.83 | 85.90 | 86.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 11:45:00 | 85.51 | 85.20 | 85.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 10:15:00 | 84.64 | 84.39 | 84.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 10:15:00 | 84.64 | 84.39 | 84.39 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 83.60 | 84.23 | 84.32 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 11:15:00 | 84.34 | 84.04 | 84.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 12:15:00 | 84.43 | 84.12 | 84.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 11:15:00 | 84.50 | 84.61 | 84.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 12:00:00 | 84.50 | 84.61 | 84.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 84.59 | 84.61 | 84.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:45:00 | 84.39 | 84.61 | 84.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 13:15:00 | 84.37 | 84.56 | 84.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 13:45:00 | 84.36 | 84.56 | 84.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 84.43 | 84.54 | 84.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 14:45:00 | 84.10 | 84.54 | 84.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 84.17 | 84.46 | 84.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 09:15:00 | 84.74 | 84.46 | 84.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-18 09:15:00 | 93.21 | 89.55 | 88.73 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-07-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 15:15:00 | 88.98 | 90.12 | 90.16 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 13:15:00 | 91.28 | 90.24 | 90.17 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 15:15:00 | 89.60 | 90.02 | 90.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 09:15:00 | 88.67 | 89.75 | 89.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 88.08 | 87.63 | 88.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-24 09:45:00 | 88.25 | 87.63 | 88.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 87.09 | 87.52 | 88.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:15:00 | 90.15 | 87.52 | 88.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 90.66 | 88.15 | 88.63 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 13:15:00 | 95.54 | 90.20 | 89.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 15:15:00 | 97.90 | 92.94 | 90.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 09:15:00 | 99.12 | 99.60 | 96.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 09:45:00 | 99.33 | 99.60 | 96.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 103.16 | 104.12 | 103.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:45:00 | 103.35 | 104.12 | 103.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 102.82 | 103.86 | 103.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:45:00 | 102.86 | 103.86 | 103.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 102.80 | 103.65 | 102.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:45:00 | 102.73 | 103.65 | 102.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 12:15:00 | 102.45 | 103.41 | 102.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 14:30:00 | 104.42 | 103.17 | 102.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:45:00 | 103.07 | 103.05 | 102.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 10:15:00 | 102.20 | 102.88 | 102.83 | SL hit (close<static) qty=1.00 sl=102.21 alert=retest2 |

### Cycle 92 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 11:15:00 | 100.79 | 102.46 | 102.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 14:15:00 | 100.14 | 101.55 | 102.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 10:15:00 | 102.10 | 101.17 | 101.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 10:15:00 | 102.10 | 101.17 | 101.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 102.10 | 101.17 | 101.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:45:00 | 102.14 | 101.17 | 101.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 11:15:00 | 101.22 | 101.18 | 101.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 12:15:00 | 100.91 | 101.18 | 101.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 15:00:00 | 100.60 | 101.11 | 101.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 95.86 | 100.31 | 101.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 10:15:00 | 95.57 | 99.25 | 100.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-07 11:15:00 | 94.05 | 93.74 | 95.45 | SL hit (close>ema200) qty=0.50 sl=93.74 alert=retest2 |

### Cycle 93 — BUY (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 11:15:00 | 97.35 | 95.55 | 95.44 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 95.37 | 96.31 | 96.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 94.80 | 96.01 | 96.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 93.82 | 93.51 | 94.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 93.82 | 93.51 | 94.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 93.82 | 93.51 | 94.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 11:00:00 | 93.45 | 93.50 | 94.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 09:15:00 | 97.08 | 94.59 | 94.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 97.08 | 94.59 | 94.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 97.82 | 96.97 | 96.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 13:15:00 | 98.72 | 99.25 | 98.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 13:15:00 | 98.72 | 99.25 | 98.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 98.72 | 99.25 | 98.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:00:00 | 98.72 | 99.25 | 98.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 98.83 | 99.17 | 98.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:45:00 | 98.73 | 99.17 | 98.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 98.44 | 99.02 | 98.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:00:00 | 98.10 | 98.84 | 98.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 97.21 | 98.51 | 98.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:00:00 | 97.21 | 98.51 | 98.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2024-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 11:15:00 | 97.40 | 98.29 | 98.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 14:15:00 | 96.72 | 97.65 | 98.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 09:15:00 | 96.70 | 96.37 | 96.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 96.70 | 96.37 | 96.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 96.70 | 96.37 | 96.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:45:00 | 96.96 | 96.37 | 96.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 97.00 | 96.49 | 96.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 12:30:00 | 96.57 | 96.49 | 96.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 14:15:00 | 96.41 | 96.54 | 96.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:15:00 | 91.74 | 92.48 | 93.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:15:00 | 91.59 | 92.48 | 93.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-09-09 09:15:00 | 86.91 | 89.59 | 91.18 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 97 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 94.39 | 88.95 | 88.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 14:15:00 | 94.60 | 92.52 | 90.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 10:15:00 | 92.90 | 92.96 | 91.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 11:00:00 | 92.90 | 92.96 | 91.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 91.95 | 93.02 | 92.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 91.95 | 93.02 | 92.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 91.48 | 92.71 | 92.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:00:00 | 91.48 | 92.71 | 92.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 91.71 | 92.02 | 91.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 15:15:00 | 91.12 | 92.02 | 91.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 91.12 | 91.84 | 91.80 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 09:15:00 | 90.84 | 91.64 | 91.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 90.42 | 91.23 | 91.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 15:15:00 | 88.70 | 88.62 | 89.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 09:15:00 | 88.60 | 88.62 | 89.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 88.73 | 88.64 | 89.52 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 91.96 | 89.78 | 89.67 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 89.23 | 90.19 | 90.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 11:15:00 | 89.06 | 89.80 | 90.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 89.14 | 88.72 | 89.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 89.14 | 88.72 | 89.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 89.14 | 88.72 | 89.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:30:00 | 89.36 | 88.72 | 89.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 89.31 | 88.84 | 89.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 11:30:00 | 89.01 | 88.85 | 89.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 13:15:00 | 84.56 | 86.02 | 86.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-04 11:15:00 | 85.86 | 85.44 | 86.23 | SL hit (close>ema200) qty=0.50 sl=85.44 alert=retest2 |

### Cycle 101 — BUY (started 2024-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 12:15:00 | 83.57 | 82.98 | 82.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 13:15:00 | 83.87 | 83.16 | 83.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 15:15:00 | 83.03 | 83.16 | 83.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 15:15:00 | 83.03 | 83.16 | 83.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 15:15:00 | 83.03 | 83.16 | 83.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 09:15:00 | 83.82 | 83.16 | 83.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 11:15:00 | 83.36 | 83.71 | 83.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 11:45:00 | 83.45 | 83.65 | 83.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 83.56 | 83.48 | 83.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 09:15:00 | 83.31 | 83.45 | 83.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 09:15:00 | 83.31 | 83.45 | 83.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 10:15:00 | 82.95 | 83.35 | 83.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 10:15:00 | 82.85 | 82.77 | 83.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 10:15:00 | 82.85 | 82.77 | 83.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 82.85 | 82.77 | 83.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:45:00 | 82.94 | 82.77 | 83.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 83.15 | 82.64 | 82.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:45:00 | 83.02 | 82.64 | 82.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 82.92 | 82.70 | 82.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 11:15:00 | 82.60 | 82.70 | 82.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 15:15:00 | 82.80 | 82.70 | 82.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 14:15:00 | 83.29 | 82.30 | 82.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 14:15:00 | 83.29 | 82.30 | 82.22 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 14:15:00 | 81.73 | 82.23 | 82.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 81.55 | 82.09 | 82.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 78.65 | 78.64 | 79.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 11:00:00 | 78.65 | 78.64 | 79.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 14:15:00 | 77.92 | 78.29 | 79.31 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2024-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 12:15:00 | 81.71 | 80.09 | 79.87 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 10:15:00 | 78.59 | 79.96 | 79.97 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-10-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 12:15:00 | 81.56 | 80.15 | 80.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-25 14:15:00 | 82.90 | 80.96 | 80.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-28 14:15:00 | 81.49 | 81.50 | 81.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-28 15:00:00 | 81.49 | 81.50 | 81.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 82.59 | 81.68 | 81.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-30 09:45:00 | 84.28 | 82.68 | 81.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 12:45:00 | 83.58 | 83.58 | 83.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 14:30:00 | 83.55 | 83.54 | 83.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 15:00:00 | 83.72 | 83.54 | 83.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 81.84 | 83.47 | 83.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 81.84 | 83.47 | 83.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-04 10:15:00 | 81.29 | 83.03 | 83.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 81.29 | 83.03 | 83.09 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 84.24 | 82.86 | 82.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 84.48 | 83.54 | 83.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 14:15:00 | 84.95 | 85.66 | 84.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 15:00:00 | 84.95 | 85.66 | 84.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 84.91 | 85.51 | 84.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 84.47 | 85.51 | 84.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 84.20 | 85.25 | 84.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:00:00 | 84.20 | 85.25 | 84.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 83.28 | 84.85 | 84.60 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 12:15:00 | 83.27 | 84.23 | 84.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 82.45 | 83.66 | 84.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 78.66 | 78.52 | 79.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 78.66 | 78.52 | 79.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 78.33 | 77.21 | 77.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:45:00 | 78.02 | 77.21 | 77.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 78.32 | 77.43 | 77.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:15:00 | 78.47 | 77.43 | 77.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 78.07 | 77.56 | 77.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 12:30:00 | 77.94 | 77.70 | 77.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:45:00 | 77.84 | 77.80 | 77.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 80.65 | 77.63 | 77.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 80.65 | 77.63 | 77.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 81.50 | 78.41 | 77.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 82.63 | 82.64 | 81.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 13:00:00 | 82.63 | 82.64 | 81.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 82.35 | 82.61 | 81.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 82.35 | 82.61 | 81.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 82.70 | 82.62 | 82.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:15:00 | 82.90 | 82.62 | 82.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 81.81 | 82.46 | 82.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:00:00 | 81.81 | 82.46 | 82.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 82.32 | 82.43 | 82.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:15:00 | 81.57 | 82.43 | 82.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 81.50 | 82.25 | 81.99 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 10:15:00 | 81.33 | 81.78 | 81.84 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 82.94 | 81.85 | 81.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 86.69 | 83.56 | 82.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 85.68 | 85.70 | 84.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 10:00:00 | 85.68 | 85.70 | 84.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 85.64 | 85.86 | 85.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 85.27 | 85.86 | 85.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 85.11 | 85.67 | 85.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 85.11 | 85.67 | 85.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 85.84 | 85.70 | 85.16 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 09:15:00 | 84.81 | 85.18 | 85.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 11:15:00 | 84.63 | 85.04 | 85.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 10:15:00 | 84.75 | 84.72 | 84.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 10:15:00 | 84.75 | 84.72 | 84.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 84.75 | 84.72 | 84.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:30:00 | 84.75 | 84.72 | 84.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 84.65 | 84.59 | 84.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 15:00:00 | 84.65 | 84.59 | 84.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 83.87 | 84.44 | 84.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:30:00 | 83.40 | 84.25 | 84.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 15:15:00 | 79.23 | 80.31 | 81.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 78.14 | 78.09 | 78.81 | SL hit (close>ema200) qty=0.50 sl=78.09 alert=retest2 |

### Cycle 115 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 76.92 | 76.39 | 76.37 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 11:15:00 | 75.54 | 76.40 | 76.46 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 09:15:00 | 77.86 | 76.70 | 76.57 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 75.27 | 76.64 | 76.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 74.35 | 75.93 | 76.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 12:15:00 | 75.34 | 75.10 | 75.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 13:00:00 | 75.34 | 75.10 | 75.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 75.64 | 75.21 | 75.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:00:00 | 75.64 | 75.21 | 75.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 75.72 | 75.31 | 75.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:30:00 | 75.71 | 75.31 | 75.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 75.69 | 75.39 | 75.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 75.09 | 75.39 | 75.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 71.34 | 73.28 | 74.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 12:15:00 | 67.58 | 69.17 | 71.03 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 119 — BUY (started 2025-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 14:15:00 | 78.16 | 70.27 | 70.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 15:15:00 | 79.48 | 72.11 | 71.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 13:15:00 | 72.37 | 72.45 | 71.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-15 13:30:00 | 72.35 | 72.45 | 71.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 82.99 | 83.54 | 82.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:45:00 | 83.00 | 83.54 | 82.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 82.25 | 83.28 | 82.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 81.02 | 83.28 | 82.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 81.03 | 82.83 | 82.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 80.88 | 82.83 | 82.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 80.24 | 82.31 | 82.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:00:00 | 80.24 | 82.31 | 82.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 78.85 | 81.62 | 81.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 09:15:00 | 77.43 | 79.67 | 80.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 12:15:00 | 78.70 | 76.55 | 77.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 12:15:00 | 78.70 | 76.55 | 77.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 12:15:00 | 78.70 | 76.55 | 77.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 13:00:00 | 78.70 | 76.55 | 77.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 13:15:00 | 78.18 | 76.87 | 77.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 14:15:00 | 80.20 | 76.87 | 77.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 14:15:00 | 80.71 | 77.64 | 78.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 14:45:00 | 81.00 | 77.64 | 78.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 10:15:00 | 78.75 | 78.17 | 78.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 11:00:00 | 78.75 | 78.17 | 78.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 11:15:00 | 79.55 | 78.44 | 78.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 12:15:00 | 79.84 | 78.72 | 78.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-28 14:15:00 | 77.68 | 78.57 | 78.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 14:15:00 | 77.68 | 78.57 | 78.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 77.68 | 78.57 | 78.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-28 15:00:00 | 77.68 | 78.57 | 78.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 15:15:00 | 77.98 | 78.46 | 78.47 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 78.99 | 78.56 | 78.52 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 11:15:00 | 78.33 | 78.49 | 78.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-29 14:15:00 | 77.97 | 78.29 | 78.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 78.99 | 78.35 | 78.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 09:15:00 | 78.99 | 78.35 | 78.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 78.99 | 78.35 | 78.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:45:00 | 79.02 | 78.35 | 78.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 78.73 | 78.43 | 78.43 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 78.99 | 78.54 | 78.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 12:15:00 | 79.45 | 78.72 | 78.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 80.25 | 80.78 | 80.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 80.25 | 80.78 | 80.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 80.25 | 80.78 | 80.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 80.15 | 80.78 | 80.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 79.75 | 80.57 | 80.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 79.69 | 80.57 | 80.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 80.40 | 80.54 | 80.06 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 78.66 | 79.76 | 79.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 77.86 | 79.19 | 79.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 79.64 | 78.69 | 79.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 79.64 | 78.69 | 79.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 79.64 | 78.69 | 79.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 80.13 | 78.69 | 79.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 79.09 | 78.77 | 79.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 12:15:00 | 78.74 | 78.82 | 79.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 09:15:00 | 81.06 | 79.42 | 79.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 81.06 | 79.42 | 79.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 10:15:00 | 81.75 | 79.88 | 79.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 11:15:00 | 80.80 | 80.94 | 80.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 12:00:00 | 80.80 | 80.94 | 80.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 80.45 | 80.81 | 80.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:00:00 | 80.45 | 80.81 | 80.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 80.19 | 80.68 | 80.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:30:00 | 80.04 | 80.68 | 80.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 80.39 | 80.62 | 80.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 80.30 | 80.62 | 80.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 80.03 | 80.50 | 80.40 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 11:15:00 | 79.60 | 80.32 | 80.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 79.05 | 79.99 | 80.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 74.55 | 74.21 | 75.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:45:00 | 74.74 | 74.21 | 75.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 75.41 | 74.60 | 75.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 14:15:00 | 74.27 | 74.87 | 75.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 70.56 | 72.30 | 73.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 12:15:00 | 72.10 | 72.00 | 72.98 | SL hit (close>ema200) qty=0.50 sl=72.00 alert=retest2 |

### Cycle 129 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 72.99 | 72.26 | 72.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 73.35 | 72.48 | 72.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 12:15:00 | 73.99 | 74.09 | 73.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 13:00:00 | 73.99 | 74.09 | 73.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 73.49 | 73.97 | 73.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:30:00 | 73.50 | 73.97 | 73.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 73.31 | 73.84 | 73.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:45:00 | 73.31 | 73.84 | 73.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 72.81 | 73.63 | 73.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 71.81 | 73.63 | 73.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 72.06 | 73.32 | 73.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 11:15:00 | 70.90 | 71.58 | 72.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 69.03 | 68.14 | 69.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 09:15:00 | 69.03 | 68.14 | 69.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 69.03 | 68.14 | 69.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 69.69 | 68.14 | 69.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 68.66 | 68.25 | 69.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:30:00 | 69.21 | 68.25 | 69.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 70.05 | 68.44 | 68.72 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 70.37 | 69.10 | 68.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 70.83 | 69.82 | 69.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 12:15:00 | 72.08 | 72.22 | 71.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 13:00:00 | 72.08 | 72.22 | 71.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 71.70 | 72.24 | 71.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 71.70 | 72.24 | 71.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 71.33 | 72.06 | 71.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 71.33 | 72.06 | 71.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 71.81 | 72.01 | 71.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 14:30:00 | 73.45 | 72.39 | 71.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 14:15:00 | 72.16 | 72.45 | 72.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 14:15:00 | 72.16 | 72.45 | 72.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 15:15:00 | 72.01 | 72.36 | 72.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 72.45 | 72.38 | 72.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 72.45 | 72.38 | 72.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 72.45 | 72.38 | 72.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 72.98 | 72.38 | 72.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 72.55 | 72.42 | 72.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:45:00 | 72.58 | 72.42 | 72.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 72.66 | 72.46 | 72.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 12:30:00 | 72.36 | 72.43 | 72.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 14:15:00 | 72.46 | 72.46 | 72.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 72.93 | 72.47 | 72.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 72.93 | 72.47 | 72.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 74.15 | 73.04 | 72.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 80.06 | 80.36 | 78.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:00:00 | 80.06 | 80.36 | 78.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 78.67 | 79.84 | 78.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 78.40 | 79.84 | 78.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 78.59 | 79.59 | 78.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:15:00 | 78.36 | 79.59 | 78.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 78.91 | 79.45 | 78.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:30:00 | 78.69 | 79.45 | 78.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 78.62 | 79.29 | 78.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 15:00:00 | 78.62 | 79.29 | 78.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 78.42 | 79.11 | 78.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 79.09 | 79.11 | 78.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:45:00 | 78.78 | 79.18 | 78.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 12:00:00 | 78.79 | 79.08 | 78.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 14:15:00 | 77.65 | 78.56 | 78.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 77.65 | 78.56 | 78.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 77.40 | 78.33 | 78.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 09:15:00 | 79.09 | 77.94 | 78.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 09:15:00 | 79.09 | 77.94 | 78.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 79.09 | 77.94 | 78.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:30:00 | 79.09 | 77.94 | 78.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 79.32 | 78.22 | 78.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:30:00 | 79.57 | 78.22 | 78.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 79.22 | 78.42 | 78.34 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 14:15:00 | 77.68 | 78.23 | 78.27 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 78.77 | 78.30 | 78.29 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 77.82 | 78.20 | 78.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 09:15:00 | 77.17 | 77.85 | 78.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 11:15:00 | 78.20 | 77.88 | 78.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 11:15:00 | 78.20 | 77.88 | 78.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 78.20 | 77.88 | 78.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 78.20 | 77.88 | 78.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 78.27 | 77.96 | 78.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:30:00 | 78.45 | 77.96 | 78.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 78.34 | 78.13 | 78.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 78.70 | 78.24 | 78.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 77.91 | 79.44 | 78.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 77.91 | 79.44 | 78.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 77.91 | 79.44 | 78.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 77.91 | 79.44 | 78.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 78.42 | 79.24 | 78.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 78.34 | 79.24 | 78.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 13:15:00 | 77.83 | 78.58 | 78.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 72.73 | 77.36 | 78.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 75.03 | 74.55 | 75.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 75.03 | 74.55 | 75.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 75.03 | 74.55 | 75.92 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 13:15:00 | 78.60 | 76.38 | 76.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 14:15:00 | 79.60 | 78.53 | 77.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 15:15:00 | 85.15 | 85.15 | 83.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 09:15:00 | 85.21 | 85.15 | 83.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 83.72 | 84.87 | 83.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 83.72 | 84.87 | 83.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 83.73 | 84.64 | 83.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 83.27 | 84.64 | 83.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 83.80 | 84.47 | 83.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:00:00 | 83.80 | 84.47 | 83.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 84.04 | 84.39 | 83.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:30:00 | 83.80 | 84.39 | 83.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 84.06 | 84.32 | 83.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:00:00 | 84.06 | 84.32 | 83.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 14:15:00 | 84.02 | 84.26 | 83.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:30:00 | 83.76 | 84.26 | 83.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 15:15:00 | 84.00 | 84.21 | 83.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 09:15:00 | 84.18 | 84.21 | 83.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 14:15:00 | 83.80 | 84.20 | 84.09 | SL hit (close<static) qty=1.00 sl=83.83 alert=retest2 |

### Cycle 142 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 81.25 | 83.53 | 83.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 80.30 | 81.40 | 82.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 11:15:00 | 82.01 | 81.46 | 82.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 11:15:00 | 82.01 | 81.46 | 82.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 82.01 | 81.46 | 82.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:45:00 | 81.80 | 81.46 | 82.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 81.98 | 81.56 | 82.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:30:00 | 82.46 | 81.56 | 82.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 82.25 | 81.70 | 82.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 14:00:00 | 82.25 | 81.70 | 82.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 82.75 | 81.91 | 82.24 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 11:15:00 | 82.65 | 82.43 | 82.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 14:15:00 | 83.24 | 82.67 | 82.54 | Break + close above crossover candle high |

### Cycle 144 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 80.60 | 82.31 | 82.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 79.30 | 80.60 | 81.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 80.10 | 79.95 | 80.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 10:15:00 | 80.40 | 79.95 | 80.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 80.57 | 80.08 | 80.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:00:00 | 80.57 | 80.08 | 80.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 80.70 | 80.20 | 80.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:15:00 | 81.03 | 80.20 | 80.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 81.04 | 80.37 | 80.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:45:00 | 80.38 | 80.57 | 80.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 76.36 | 78.64 | 79.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 78.70 | 78.21 | 78.79 | SL hit (close>ema200) qty=0.50 sl=78.21 alert=retest2 |

### Cycle 145 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 80.49 | 77.90 | 77.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 81.52 | 79.42 | 78.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 11:15:00 | 85.92 | 86.06 | 84.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 12:00:00 | 85.92 | 86.06 | 84.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 89.92 | 90.53 | 89.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:45:00 | 89.52 | 90.53 | 89.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 94.07 | 93.91 | 92.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:45:00 | 94.35 | 93.56 | 92.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 13:15:00 | 94.39 | 93.56 | 92.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 14:00:00 | 94.36 | 93.72 | 92.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 94.45 | 93.80 | 93.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 93.36 | 93.71 | 93.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:00:00 | 93.36 | 93.71 | 93.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 93.15 | 93.60 | 93.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:45:00 | 93.20 | 93.60 | 93.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 92.78 | 93.44 | 93.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:45:00 | 92.83 | 93.44 | 93.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 92.30 | 93.21 | 93.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:00:00 | 92.30 | 93.21 | 93.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-26 13:15:00 | 92.28 | 93.02 | 92.94 | SL hit (close<static) qty=1.00 sl=92.30 alert=retest2 |

### Cycle 146 — SELL (started 2025-05-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 15:15:00 | 91.40 | 92.61 | 92.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 10:15:00 | 91.20 | 92.12 | 92.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 09:15:00 | 92.09 | 91.62 | 92.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 92.09 | 91.62 | 92.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 92.09 | 91.62 | 92.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:30:00 | 92.21 | 91.62 | 92.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 92.62 | 91.82 | 92.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:45:00 | 92.68 | 91.82 | 92.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 92.18 | 91.89 | 92.08 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 14:15:00 | 93.23 | 92.29 | 92.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 15:15:00 | 93.99 | 92.63 | 92.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 15:15:00 | 93.19 | 93.69 | 93.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 15:15:00 | 93.19 | 93.69 | 93.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 93.19 | 93.69 | 93.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 92.92 | 93.69 | 93.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 92.59 | 93.47 | 93.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:30:00 | 92.83 | 93.47 | 93.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 92.66 | 93.31 | 93.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 92.50 | 93.31 | 93.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 98.00 | 98.71 | 97.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 97.25 | 98.71 | 97.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 99.22 | 98.81 | 98.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 14:15:00 | 100.60 | 99.18 | 98.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 15:15:00 | 100.40 | 99.38 | 98.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:30:00 | 100.50 | 101.12 | 100.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 10:00:00 | 100.40 | 101.12 | 100.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 100.20 | 100.93 | 100.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 100.20 | 100.93 | 100.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-10 12:15:00 | 100.00 | 100.61 | 100.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 12:15:00 | 100.00 | 100.61 | 100.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 09:15:00 | 99.45 | 100.18 | 100.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 14:15:00 | 100.20 | 99.78 | 100.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 14:15:00 | 100.20 | 99.78 | 100.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 100.20 | 99.78 | 100.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 14:00:00 | 98.25 | 99.45 | 99.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:15:00 | 93.34 | 95.38 | 97.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 94.82 | 94.77 | 96.17 | SL hit (close>ema200) qty=0.50 sl=94.77 alert=retest2 |

### Cycle 149 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 93.46 | 91.55 | 91.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 11:15:00 | 94.20 | 92.73 | 92.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 103.10 | 103.99 | 102.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 10:00:00 | 103.10 | 103.99 | 102.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 102.97 | 103.51 | 102.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 102.81 | 103.51 | 102.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 103.11 | 103.27 | 102.75 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 13:15:00 | 101.90 | 102.51 | 102.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 14:15:00 | 101.35 | 102.28 | 102.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 12:15:00 | 101.10 | 101.02 | 101.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 12:15:00 | 101.10 | 101.02 | 101.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 101.10 | 101.02 | 101.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:00:00 | 101.10 | 101.02 | 101.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 101.64 | 101.14 | 101.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:00:00 | 101.64 | 101.14 | 101.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 101.69 | 101.25 | 101.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:15:00 | 101.25 | 101.25 | 101.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:00:00 | 101.27 | 101.26 | 101.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 100.25 | 99.80 | 99.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 100.25 | 99.80 | 99.77 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 11:15:00 | 99.05 | 99.69 | 99.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 12:15:00 | 98.97 | 99.55 | 99.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 09:15:00 | 99.00 | 98.93 | 99.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 99.00 | 98.93 | 99.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 99.00 | 98.93 | 99.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:30:00 | 99.85 | 98.93 | 99.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 100.06 | 99.21 | 99.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:00:00 | 100.06 | 99.21 | 99.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 100.41 | 99.45 | 99.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:00:00 | 100.41 | 99.45 | 99.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 100.55 | 99.67 | 99.55 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 98.87 | 99.62 | 99.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 10:15:00 | 98.46 | 99.39 | 99.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 10:15:00 | 96.90 | 96.44 | 97.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 10:15:00 | 96.90 | 96.44 | 97.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 96.90 | 96.44 | 97.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:30:00 | 96.94 | 96.44 | 97.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 96.91 | 96.53 | 97.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 95.72 | 96.57 | 96.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 12:15:00 | 94.48 | 93.86 | 93.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 94.48 | 93.86 | 93.86 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 92.40 | 93.70 | 93.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 91.95 | 92.62 | 93.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 91.13 | 91.09 | 91.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 13:45:00 | 91.03 | 91.09 | 91.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 91.30 | 91.24 | 91.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:15:00 | 91.08 | 91.24 | 91.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 12:15:00 | 91.00 | 89.97 | 89.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — BUY (started 2025-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 12:15:00 | 91.00 | 89.97 | 89.92 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 14:15:00 | 89.79 | 89.99 | 90.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 10:15:00 | 89.10 | 89.82 | 89.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 88.69 | 87.93 | 88.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 88.69 | 87.93 | 88.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 88.69 | 87.93 | 88.34 | EMA400 retest candle locked (from downside) |

### Cycle 159 — BUY (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 11:15:00 | 89.48 | 88.59 | 88.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 12:15:00 | 90.30 | 89.59 | 89.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 14:15:00 | 95.10 | 95.14 | 93.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 14:30:00 | 95.30 | 95.14 | 93.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 92.70 | 94.48 | 94.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 92.20 | 94.48 | 94.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 92.72 | 94.13 | 93.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 92.65 | 94.13 | 93.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 92.53 | 93.81 | 93.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 92.39 | 93.32 | 93.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 15:15:00 | 87.30 | 87.05 | 88.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 09:15:00 | 87.55 | 87.05 | 88.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 89.20 | 87.48 | 88.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:00:00 | 89.20 | 87.48 | 88.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 90.29 | 88.04 | 88.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:45:00 | 90.47 | 88.04 | 88.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — BUY (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 12:15:00 | 89.76 | 88.75 | 88.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 10:15:00 | 90.53 | 89.43 | 89.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 89.99 | 90.17 | 89.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:45:00 | 90.10 | 90.17 | 89.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 90.00 | 90.13 | 89.75 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 88.80 | 89.52 | 89.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 88.31 | 89.21 | 89.42 | Break + close below crossover candle low |

### Cycle 163 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 92.02 | 89.59 | 89.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 10:15:00 | 93.21 | 90.31 | 89.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 15:15:00 | 91.80 | 92.35 | 91.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 15:15:00 | 91.80 | 92.35 | 91.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 91.80 | 92.35 | 91.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:45:00 | 93.10 | 92.97 | 92.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 15:00:00 | 92.80 | 93.81 | 93.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 13:15:00 | 93.01 | 93.26 | 93.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 13:15:00 | 93.01 | 93.26 | 93.28 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 94.05 | 93.35 | 93.31 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 92.78 | 93.31 | 93.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 11:15:00 | 92.60 | 93.03 | 93.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 09:15:00 | 93.48 | 92.80 | 92.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 93.48 | 92.80 | 92.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 93.48 | 92.80 | 92.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:45:00 | 93.42 | 92.80 | 92.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 92.93 | 92.83 | 92.98 | EMA400 retest candle locked (from downside) |

### Cycle 167 — BUY (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 12:15:00 | 94.06 | 93.14 | 93.10 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 92.50 | 93.33 | 93.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 13:15:00 | 92.36 | 92.91 | 93.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 90.10 | 90.07 | 90.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:45:00 | 90.14 | 90.07 | 90.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 89.38 | 89.69 | 90.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 90.08 | 89.69 | 90.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 91.40 | 90.00 | 90.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:45:00 | 91.84 | 90.00 | 90.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 91.39 | 90.28 | 90.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:00:00 | 91.39 | 90.28 | 90.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 12:15:00 | 90.89 | 90.63 | 90.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 92.30 | 91.41 | 91.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 15:15:00 | 92.71 | 92.74 | 92.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 09:15:00 | 92.85 | 92.74 | 92.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 92.51 | 92.66 | 92.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 12:45:00 | 92.78 | 92.68 | 92.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:30:00 | 92.70 | 92.70 | 92.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:00:00 | 92.80 | 92.70 | 92.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 92.75 | 92.58 | 92.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 92.21 | 92.50 | 92.45 | SL hit (close<static) qty=1.00 sl=92.35 alert=retest2 |

### Cycle 170 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 91.60 | 92.32 | 92.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 91.05 | 92.07 | 92.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 12:15:00 | 91.49 | 91.46 | 91.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 12:45:00 | 91.50 | 91.46 | 91.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 91.66 | 91.50 | 91.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 91.66 | 91.50 | 91.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 91.37 | 91.47 | 91.71 | EMA400 retest candle locked (from downside) |

### Cycle 171 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 93.90 | 91.96 | 91.89 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 91.72 | 92.59 | 92.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 91.37 | 92.35 | 92.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 13:15:00 | 92.38 | 92.35 | 92.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 14:00:00 | 92.38 | 92.35 | 92.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 91.75 | 92.23 | 92.47 | EMA400 retest candle locked (from downside) |

### Cycle 173 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 93.24 | 92.55 | 92.52 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 15:15:00 | 92.30 | 92.67 | 92.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 91.55 | 92.41 | 92.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 93.29 | 92.33 | 92.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 93.29 | 92.33 | 92.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 93.29 | 92.33 | 92.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:45:00 | 93.59 | 92.33 | 92.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 93.77 | 92.62 | 92.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 12:15:00 | 95.28 | 93.40 | 92.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 13:15:00 | 94.51 | 95.06 | 94.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 13:15:00 | 94.51 | 95.06 | 94.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 94.51 | 95.06 | 94.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 94.51 | 95.06 | 94.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 94.19 | 94.89 | 94.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 94.19 | 94.89 | 94.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 93.98 | 94.70 | 94.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 94.21 | 94.70 | 94.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 12:00:00 | 94.20 | 94.45 | 94.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 13:15:00 | 93.66 | 94.20 | 94.16 | SL hit (close<static) qty=1.00 sl=93.85 alert=retest2 |

### Cycle 176 — SELL (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 15:15:00 | 93.96 | 94.10 | 94.12 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 94.99 | 94.28 | 94.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 12:15:00 | 95.55 | 94.73 | 94.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 100.74 | 101.42 | 99.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 10:00:00 | 100.74 | 101.42 | 99.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 99.41 | 100.82 | 99.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:00:00 | 99.41 | 100.82 | 99.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 98.95 | 100.44 | 99.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:00:00 | 98.95 | 100.44 | 99.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 97.84 | 99.92 | 99.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:00:00 | 97.84 | 99.92 | 99.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 15:15:00 | 97.90 | 99.21 | 99.35 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 10:15:00 | 100.13 | 99.57 | 99.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 11:15:00 | 105.90 | 100.84 | 100.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 10:15:00 | 102.56 | 102.74 | 101.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-03 11:00:00 | 102.56 | 102.74 | 101.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 102.32 | 102.66 | 101.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:45:00 | 101.73 | 102.66 | 101.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 101.50 | 102.28 | 101.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 101.50 | 102.28 | 101.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 101.50 | 102.12 | 101.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 100.10 | 102.12 | 101.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 99.92 | 101.35 | 101.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 99.23 | 100.63 | 101.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 99.00 | 98.01 | 98.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 11:15:00 | 99.00 | 98.01 | 98.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 99.00 | 98.01 | 98.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:00:00 | 99.00 | 98.01 | 98.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 99.79 | 98.36 | 98.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 99.79 | 98.36 | 98.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 100.30 | 98.75 | 99.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 100.35 | 98.75 | 99.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — BUY (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 15:15:00 | 100.20 | 99.34 | 99.32 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 11:15:00 | 98.94 | 99.25 | 99.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 15:15:00 | 98.79 | 99.14 | 99.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 12:15:00 | 98.65 | 98.58 | 98.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 13:00:00 | 98.65 | 98.58 | 98.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 98.82 | 98.65 | 98.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:00:00 | 98.82 | 98.65 | 98.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 98.85 | 98.69 | 98.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 99.20 | 98.69 | 98.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 98.95 | 98.74 | 98.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 10:15:00 | 98.75 | 98.74 | 98.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 10:45:00 | 98.76 | 98.76 | 98.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 11:15:00 | 101.02 | 99.21 | 99.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 101.02 | 99.21 | 99.06 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 98.16 | 99.15 | 99.24 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 11:15:00 | 100.56 | 99.40 | 99.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 14:15:00 | 100.95 | 100.04 | 99.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 14:15:00 | 102.60 | 103.00 | 102.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 15:00:00 | 102.60 | 103.00 | 102.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 103.10 | 102.94 | 102.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 103.10 | 102.94 | 102.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 103.40 | 103.74 | 103.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 103.40 | 103.74 | 103.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 103.01 | 103.52 | 103.14 | EMA400 retest candle locked (from upside) |

### Cycle 186 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 100.63 | 102.66 | 102.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 12:15:00 | 100.05 | 101.51 | 102.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 101.62 | 100.99 | 101.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 101.62 | 100.99 | 101.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 101.62 | 100.99 | 101.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:30:00 | 98.75 | 100.43 | 101.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 102.83 | 101.47 | 101.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 102.83 | 101.47 | 101.29 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 101.15 | 101.55 | 101.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 100.53 | 101.25 | 101.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 101.61 | 100.86 | 101.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 101.61 | 100.86 | 101.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 101.61 | 100.86 | 101.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:45:00 | 101.66 | 100.86 | 101.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 101.23 | 100.94 | 101.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:30:00 | 100.92 | 100.83 | 101.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:45:00 | 100.98 | 100.56 | 100.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 09:15:00 | 95.93 | 97.27 | 97.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:15:00 | 95.87 | 96.95 | 97.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 95.47 | 95.08 | 96.06 | SL hit (close>ema200) qty=0.50 sl=95.08 alert=retest2 |

### Cycle 189 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 97.49 | 95.53 | 95.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 98.85 | 97.21 | 96.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 98.27 | 98.66 | 97.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 98.27 | 98.66 | 97.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 98.50 | 98.55 | 98.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:45:00 | 98.05 | 98.55 | 98.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 98.78 | 98.98 | 98.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:30:00 | 98.50 | 98.98 | 98.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 98.49 | 98.88 | 98.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:00:00 | 98.49 | 98.88 | 98.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 98.16 | 98.74 | 98.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 13:00:00 | 98.16 | 98.74 | 98.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 97.81 | 98.55 | 98.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 97.81 | 98.55 | 98.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2025-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 15:15:00 | 97.59 | 98.27 | 98.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 96.84 | 97.99 | 98.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 98.10 | 97.88 | 98.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 98.10 | 97.88 | 98.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 98.10 | 97.88 | 98.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 98.10 | 97.88 | 98.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 97.88 | 97.88 | 98.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 13:30:00 | 97.45 | 97.77 | 98.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 15:15:00 | 97.32 | 97.74 | 97.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 97.62 | 97.73 | 97.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 15:15:00 | 98.40 | 97.93 | 97.93 | SL hit (close>static) qty=1.00 sl=98.10 alert=retest2 |

### Cycle 191 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 101.49 | 98.64 | 98.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 103.44 | 101.83 | 101.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 12:15:00 | 111.10 | 111.14 | 108.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 13:00:00 | 111.10 | 111.14 | 108.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 110.28 | 110.88 | 109.27 | EMA400 retest candle locked (from upside) |

### Cycle 192 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 107.90 | 108.75 | 108.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 106.86 | 108.07 | 108.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 106.70 | 105.84 | 106.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 106.70 | 105.84 | 106.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 106.70 | 105.84 | 106.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:00:00 | 106.70 | 105.84 | 106.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 106.09 | 105.89 | 106.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:30:00 | 106.00 | 105.68 | 106.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 14:15:00 | 104.88 | 104.47 | 104.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — BUY (started 2026-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 14:15:00 | 104.88 | 104.47 | 104.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 106.48 | 104.95 | 104.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 104.65 | 105.15 | 104.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 13:15:00 | 104.65 | 105.15 | 104.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 104.65 | 105.15 | 104.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 104.65 | 105.15 | 104.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 104.45 | 105.01 | 104.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:00:00 | 104.45 | 105.01 | 104.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 104.40 | 104.89 | 104.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 99.67 | 104.89 | 104.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 100.02 | 103.92 | 104.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 99.00 | 102.93 | 103.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 13:15:00 | 97.94 | 96.68 | 98.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 13:15:00 | 97.94 | 96.68 | 98.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 97.94 | 96.68 | 98.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:45:00 | 96.64 | 96.68 | 98.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 98.13 | 96.97 | 98.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 14:45:00 | 98.01 | 96.97 | 98.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 98.45 | 97.27 | 98.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 100.89 | 97.27 | 98.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 100.27 | 97.87 | 98.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 100.50 | 97.87 | 98.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 100.08 | 98.66 | 98.66 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 97.52 | 98.83 | 98.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 96.99 | 98.30 | 98.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 97.90 | 97.25 | 97.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 97.90 | 97.25 | 97.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 97.90 | 97.25 | 97.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 98.26 | 97.25 | 97.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 97.90 | 97.38 | 97.86 | EMA400 retest candle locked (from downside) |

### Cycle 197 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 99.79 | 98.21 | 98.18 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 97.81 | 98.36 | 98.42 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 100.15 | 98.57 | 98.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 102.62 | 99.90 | 99.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 100.15 | 100.49 | 99.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 100.15 | 100.49 | 99.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 100.15 | 100.49 | 99.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 100.38 | 100.49 | 99.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 99.26 | 100.24 | 99.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 99.26 | 100.24 | 99.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 98.65 | 99.92 | 99.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 98.65 | 99.92 | 99.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 98.24 | 99.59 | 99.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 99.00 | 99.59 | 99.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:45:00 | 98.73 | 99.42 | 99.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 97.10 | 98.96 | 99.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 97.10 | 98.96 | 99.19 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 100.41 | 99.34 | 99.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 10:15:00 | 106.29 | 102.61 | 101.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 14:15:00 | 103.09 | 108.25 | 106.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 14:15:00 | 103.09 | 108.25 | 106.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 103.09 | 108.25 | 106.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:30:00 | 101.26 | 108.25 | 106.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 102.95 | 107.19 | 106.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 104.80 | 107.19 | 106.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 10:30:00 | 103.54 | 106.13 | 105.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 09:15:00 | 103.35 | 105.64 | 105.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — SELL (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 09:15:00 | 103.35 | 105.64 | 105.70 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 108.63 | 105.95 | 105.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 12:15:00 | 109.21 | 107.07 | 106.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 11:15:00 | 110.74 | 110.88 | 109.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 12:00:00 | 110.74 | 110.88 | 109.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 109.99 | 110.55 | 109.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:45:00 | 110.03 | 110.55 | 109.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 108.35 | 110.06 | 109.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 108.64 | 110.06 | 109.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 110.02 | 110.44 | 110.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 14:30:00 | 111.32 | 110.46 | 110.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 14:15:00 | 110.84 | 112.28 | 112.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 110.84 | 112.28 | 112.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 110.58 | 111.94 | 112.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 09:15:00 | 114.05 | 112.36 | 112.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 114.05 | 112.36 | 112.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 114.05 | 112.36 | 112.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:00:00 | 114.05 | 112.36 | 112.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — BUY (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 10:15:00 | 113.42 | 112.57 | 112.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 09:15:00 | 114.65 | 113.74 | 113.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 11:15:00 | 113.70 | 113.91 | 113.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-24 12:00:00 | 113.70 | 113.91 | 113.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 113.19 | 113.76 | 113.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:00:00 | 113.19 | 113.76 | 113.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 113.15 | 113.64 | 113.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:30:00 | 112.60 | 113.64 | 113.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 114.05 | 113.64 | 113.48 | EMA400 retest candle locked (from upside) |

### Cycle 206 — SELL (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 12:15:00 | 112.47 | 113.30 | 113.35 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 114.10 | 113.47 | 113.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 12:15:00 | 116.59 | 114.49 | 113.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 113.52 | 114.89 | 114.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 113.52 | 114.89 | 114.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 113.52 | 114.89 | 114.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 10:15:00 | 115.50 | 114.89 | 114.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 12:15:00 | 112.35 | 114.12 | 114.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 112.35 | 114.12 | 114.15 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 14:15:00 | 115.17 | 114.21 | 114.18 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 111.14 | 113.71 | 113.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 108.71 | 112.13 | 113.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 111.00 | 110.85 | 112.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 111.00 | 110.85 | 112.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 111.00 | 110.85 | 112.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:15:00 | 110.45 | 110.85 | 112.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 104.93 | 107.80 | 109.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-09 13:15:00 | 99.41 | 103.99 | 106.87 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 211 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 67.65 | 64.89 | 64.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 69.30 | 66.79 | 65.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 69.79 | 69.81 | 68.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 15:00:00 | 69.79 | 69.81 | 68.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 72.10 | 72.65 | 71.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 72.15 | 72.65 | 71.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 72.09 | 72.54 | 71.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 72.09 | 72.54 | 71.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 72.85 | 73.41 | 72.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 73.21 | 73.41 | 72.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 74.23 | 74.58 | 74.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 74.23 | 74.58 | 74.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 13:15:00 | 73.72 | 74.13 | 74.34 | Break + close below crossover candle low |

### Cycle 213 — BUY (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 11:15:00 | 77.05 | 74.28 | 74.28 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 75.84 | 76.59 | 76.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 09:15:00 | 74.94 | 75.94 | 76.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 15:15:00 | 74.85 | 74.62 | 75.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 09:15:00 | 75.39 | 74.62 | 75.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 75.14 | 74.73 | 75.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:45:00 | 74.76 | 74.73 | 75.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:00:00 | 74.77 | 74.74 | 75.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 76.33 | 75.28 | 75.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 215 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 76.33 | 75.28 | 75.19 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 11:15:00 | 74.95 | 75.37 | 75.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 13:15:00 | 74.67 | 75.14 | 75.28 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-06-02 09:15:00 | 56.05 | 2023-06-06 13:15:00 | 55.45 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2023-06-02 15:15:00 | 55.55 | 2023-06-06 13:15:00 | 55.45 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2023-06-06 12:30:00 | 55.55 | 2023-06-06 13:15:00 | 55.45 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2023-06-06 13:15:00 | 55.55 | 2023-06-06 13:15:00 | 55.45 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2023-06-14 10:45:00 | 53.95 | 2023-06-16 15:15:00 | 53.95 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2023-06-14 14:30:00 | 53.90 | 2023-06-16 15:15:00 | 53.95 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2023-06-14 15:15:00 | 53.95 | 2023-06-16 15:15:00 | 53.95 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2023-06-15 11:15:00 | 53.90 | 2023-06-19 09:15:00 | 54.65 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2023-06-16 12:30:00 | 53.55 | 2023-06-19 09:15:00 | 54.65 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2023-06-16 13:30:00 | 53.55 | 2023-06-19 09:15:00 | 54.65 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2023-06-16 14:00:00 | 53.55 | 2023-06-19 09:15:00 | 54.65 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2023-06-28 13:45:00 | 53.60 | 2023-06-30 09:15:00 | 54.85 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2023-07-18 11:30:00 | 57.45 | 2023-07-20 09:15:00 | 58.35 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2023-07-19 13:45:00 | 57.50 | 2023-07-20 09:15:00 | 58.35 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2023-07-19 14:15:00 | 57.50 | 2023-07-20 09:15:00 | 58.35 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2023-08-03 09:15:00 | 61.15 | 2023-08-09 11:15:00 | 63.15 | STOP_HIT | 1.00 | 3.27% |
| BUY | retest2 | 2023-08-03 09:45:00 | 60.80 | 2023-08-09 11:15:00 | 63.15 | STOP_HIT | 1.00 | 3.87% |
| SELL | retest2 | 2023-08-30 14:00:00 | 59.95 | 2023-09-04 09:15:00 | 63.60 | STOP_HIT | 1.00 | -6.09% |
| SELL | retest2 | 2023-08-30 14:30:00 | 59.90 | 2023-09-04 09:15:00 | 63.60 | STOP_HIT | 1.00 | -6.18% |
| SELL | retest2 | 2023-08-31 09:45:00 | 59.90 | 2023-09-04 09:15:00 | 63.60 | STOP_HIT | 1.00 | -6.18% |
| SELL | retest2 | 2023-09-01 12:45:00 | 59.95 | 2023-09-04 09:15:00 | 63.60 | STOP_HIT | 1.00 | -6.09% |
| BUY | retest2 | 2023-09-18 10:15:00 | 71.95 | 2023-09-21 14:15:00 | 69.50 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2023-09-18 10:45:00 | 72.35 | 2023-09-21 14:15:00 | 69.50 | STOP_HIT | 1.00 | -3.94% |
| BUY | retest2 | 2023-09-18 13:30:00 | 72.50 | 2023-09-21 14:15:00 | 69.50 | STOP_HIT | 1.00 | -4.14% |
| BUY | retest2 | 2023-09-20 10:30:00 | 72.00 | 2023-09-21 14:15:00 | 69.50 | STOP_HIT | 1.00 | -3.47% |
| BUY | retest2 | 2023-09-21 10:30:00 | 72.70 | 2023-09-21 14:15:00 | 69.50 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest2 | 2023-09-29 09:15:00 | 70.65 | 2023-09-29 15:15:00 | 70.65 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2023-10-06 11:15:00 | 69.15 | 2023-10-06 15:15:00 | 69.80 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2023-10-09 09:15:00 | 66.15 | 2023-10-11 09:15:00 | 69.80 | STOP_HIT | 1.00 | -5.52% |
| BUY | retest2 | 2023-10-17 09:15:00 | 70.75 | 2023-10-18 14:15:00 | 68.35 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2023-10-17 11:00:00 | 70.55 | 2023-10-18 14:15:00 | 68.35 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2023-10-18 09:30:00 | 71.50 | 2023-10-18 14:15:00 | 68.35 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2023-10-18 12:15:00 | 70.55 | 2023-10-18 14:15:00 | 68.35 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2023-10-27 12:15:00 | 63.20 | 2023-11-01 15:15:00 | 60.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-27 12:15:00 | 63.20 | 2023-11-02 09:15:00 | 61.90 | STOP_HIT | 0.50 | 2.06% |
| BUY | retest2 | 2023-11-06 09:15:00 | 62.30 | 2023-11-10 09:15:00 | 62.75 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2023-11-06 09:45:00 | 62.40 | 2023-11-10 09:15:00 | 62.75 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2023-11-06 13:30:00 | 62.25 | 2023-11-10 09:15:00 | 62.75 | STOP_HIT | 1.00 | 0.80% |
| BUY | retest2 | 2023-11-07 09:15:00 | 62.35 | 2023-11-10 09:15:00 | 62.75 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2023-11-08 09:15:00 | 64.00 | 2023-11-10 09:15:00 | 62.75 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2023-11-09 15:00:00 | 62.90 | 2023-11-10 09:15:00 | 62.75 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2023-11-10 09:15:00 | 63.00 | 2023-11-10 09:15:00 | 62.75 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2023-11-24 15:00:00 | 60.60 | 2023-11-28 10:15:00 | 61.60 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest1 | 2023-12-04 09:15:00 | 63.15 | 2023-12-07 10:15:00 | 66.31 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-12-04 09:15:00 | 63.15 | 2023-12-08 10:15:00 | 66.45 | STOP_HIT | 0.50 | 5.23% |
| BUY | retest2 | 2023-12-11 09:15:00 | 67.10 | 2023-12-13 12:15:00 | 65.85 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2023-12-22 12:15:00 | 66.25 | 2023-12-29 09:15:00 | 67.50 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2023-12-27 10:30:00 | 66.30 | 2023-12-29 09:15:00 | 67.50 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2023-12-27 11:00:00 | 66.25 | 2023-12-29 09:15:00 | 67.50 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2023-12-27 12:00:00 | 66.25 | 2023-12-29 09:15:00 | 67.50 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-01-02 09:30:00 | 68.25 | 2024-01-02 10:15:00 | 66.70 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2024-01-03 10:30:00 | 67.80 | 2024-01-08 10:15:00 | 67.80 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2024-01-04 09:15:00 | 69.35 | 2024-01-08 10:15:00 | 67.80 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-01-08 10:00:00 | 67.90 | 2024-01-08 10:15:00 | 67.80 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2024-01-18 14:15:00 | 69.00 | 2024-01-18 15:15:00 | 69.40 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2024-01-20 09:15:00 | 70.25 | 2024-01-20 13:15:00 | 73.76 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-01-20 09:15:00 | 70.25 | 2024-01-20 14:15:00 | 77.28 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-01-31 09:15:00 | 84.70 | 2024-02-05 09:15:00 | 93.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-21 09:30:00 | 91.30 | 2024-02-21 14:15:00 | 89.90 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-02-21 11:30:00 | 91.25 | 2024-02-21 14:15:00 | 89.90 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-03-05 10:45:00 | 88.40 | 2024-03-06 09:15:00 | 87.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest1 | 2024-03-07 12:15:00 | 85.50 | 2024-03-11 09:15:00 | 86.55 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest1 | 2024-03-07 15:15:00 | 85.45 | 2024-03-11 09:15:00 | 86.55 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-03-11 11:30:00 | 85.35 | 2024-03-13 09:15:00 | 81.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 11:30:00 | 85.35 | 2024-03-13 12:15:00 | 76.81 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-21 12:00:00 | 80.40 | 2024-03-22 09:15:00 | 81.30 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-03-27 09:15:00 | 80.90 | 2024-03-27 10:15:00 | 80.45 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-03-27 10:15:00 | 80.85 | 2024-03-27 10:15:00 | 80.45 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2024-04-01 09:15:00 | 82.75 | 2024-04-12 12:15:00 | 87.70 | STOP_HIT | 1.00 | 5.98% |
| SELL | retest2 | 2024-04-16 12:15:00 | 85.90 | 2024-04-18 12:15:00 | 87.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-04-18 10:15:00 | 85.95 | 2024-04-18 12:15:00 | 87.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-04-18 14:30:00 | 85.55 | 2024-04-22 13:15:00 | 86.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-04-25 13:00:00 | 87.95 | 2024-05-03 12:15:00 | 89.55 | STOP_HIT | 1.00 | 1.82% |
| BUY | retest2 | 2024-04-25 15:15:00 | 88.20 | 2024-05-03 12:15:00 | 89.55 | STOP_HIT | 1.00 | 1.53% |
| SELL | retest2 | 2024-05-06 12:15:00 | 89.25 | 2024-05-08 09:15:00 | 84.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-06 12:15:00 | 89.25 | 2024-05-08 11:15:00 | 86.55 | STOP_HIT | 0.50 | 3.03% |
| BUY | retest2 | 2024-05-27 09:30:00 | 88.10 | 2024-05-28 12:15:00 | 86.30 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-05-27 12:15:00 | 88.25 | 2024-05-28 12:15:00 | 86.30 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-06-25 11:00:00 | 85.83 | 2024-07-02 10:15:00 | 84.64 | STOP_HIT | 1.00 | 1.39% |
| SELL | retest2 | 2024-06-26 11:45:00 | 85.51 | 2024-07-02 10:15:00 | 84.64 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2024-07-09 09:15:00 | 84.74 | 2024-07-18 09:15:00 | 93.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-31 14:30:00 | 104.42 | 2024-08-01 10:15:00 | 102.20 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-08-01 09:45:00 | 103.07 | 2024-08-01 10:15:00 | 102.20 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-08-02 12:15:00 | 100.91 | 2024-08-05 09:15:00 | 95.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-02 15:00:00 | 100.60 | 2024-08-05 10:15:00 | 95.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-02 12:15:00 | 100.91 | 2024-08-07 11:15:00 | 94.05 | STOP_HIT | 0.50 | 6.80% |
| SELL | retest2 | 2024-08-02 15:00:00 | 100.60 | 2024-08-07 11:15:00 | 94.05 | STOP_HIT | 0.50 | 6.51% |
| SELL | retest2 | 2024-08-16 11:00:00 | 93.45 | 2024-08-19 09:15:00 | 97.08 | STOP_HIT | 1.00 | -3.88% |
| SELL | retest2 | 2024-08-28 12:30:00 | 96.57 | 2024-09-06 09:15:00 | 91.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-28 14:15:00 | 96.41 | 2024-09-06 09:15:00 | 91.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-28 12:30:00 | 96.57 | 2024-09-09 09:15:00 | 86.91 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-28 14:15:00 | 96.41 | 2024-09-09 10:15:00 | 86.77 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-27 11:30:00 | 89.01 | 2024-10-03 13:15:00 | 84.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 11:30:00 | 89.01 | 2024-10-04 11:15:00 | 85.86 | STOP_HIT | 0.50 | 3.54% |
| BUY | retest2 | 2024-10-10 09:15:00 | 83.82 | 2024-10-14 09:15:00 | 83.31 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-10-11 11:15:00 | 83.36 | 2024-10-14 09:15:00 | 83.31 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2024-10-11 11:45:00 | 83.45 | 2024-10-14 09:15:00 | 83.31 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2024-10-14 09:15:00 | 83.56 | 2024-10-14 09:15:00 | 83.31 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2024-10-16 11:15:00 | 82.60 | 2024-10-18 14:15:00 | 83.29 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-10-16 15:15:00 | 82.80 | 2024-10-18 14:15:00 | 83.29 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2024-10-30 09:45:00 | 84.28 | 2024-11-04 10:15:00 | 81.29 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2024-10-31 12:45:00 | 83.58 | 2024-11-04 10:15:00 | 81.29 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2024-10-31 14:30:00 | 83.55 | 2024-11-04 10:15:00 | 81.29 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2024-10-31 15:00:00 | 83.72 | 2024-11-04 10:15:00 | 81.29 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2024-11-19 12:30:00 | 77.94 | 2024-11-25 09:15:00 | 80.65 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2024-11-19 14:45:00 | 77.84 | 2024-11-25 09:15:00 | 80.65 | STOP_HIT | 1.00 | -3.61% |
| SELL | retest2 | 2024-12-12 09:30:00 | 83.40 | 2024-12-17 15:15:00 | 79.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 09:30:00 | 83.40 | 2024-12-20 10:15:00 | 78.14 | STOP_HIT | 0.50 | 6.31% |
| SELL | retest2 | 2025-01-08 09:15:00 | 75.09 | 2025-01-10 09:15:00 | 71.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 75.09 | 2025-01-13 12:15:00 | 67.58 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-04 12:15:00 | 78.74 | 2025-02-05 09:15:00 | 81.06 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2025-02-13 14:15:00 | 74.27 | 2025-02-17 09:15:00 | 70.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 14:15:00 | 74.27 | 2025-02-17 12:15:00 | 72.10 | STOP_HIT | 0.50 | 2.92% |
| BUY | retest2 | 2025-03-10 14:30:00 | 73.45 | 2025-03-13 14:15:00 | 72.16 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-03-17 12:30:00 | 72.36 | 2025-03-18 09:15:00 | 72.93 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-03-17 14:15:00 | 72.46 | 2025-03-18 09:15:00 | 72.93 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-03-26 09:15:00 | 79.09 | 2025-03-26 14:15:00 | 77.65 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-03-26 09:45:00 | 78.78 | 2025-03-26 14:15:00 | 77.65 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-03-26 12:00:00 | 78.79 | 2025-03-26 14:15:00 | 77.65 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-04-24 09:15:00 | 84.18 | 2025-04-24 14:15:00 | 83.80 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-05-06 09:45:00 | 80.38 | 2025-05-07 09:15:00 | 76.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:45:00 | 80.38 | 2025-05-08 09:15:00 | 78.70 | STOP_HIT | 0.50 | 2.09% |
| BUY | retest2 | 2025-05-23 12:45:00 | 94.35 | 2025-05-26 13:15:00 | 92.28 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-05-23 13:15:00 | 94.39 | 2025-05-26 13:15:00 | 92.28 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-05-23 14:00:00 | 94.36 | 2025-05-26 13:15:00 | 92.28 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-05-26 09:15:00 | 94.45 | 2025-05-26 13:15:00 | 92.28 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-06-04 14:15:00 | 100.60 | 2025-06-10 12:15:00 | 100.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-06-04 15:15:00 | 100.40 | 2025-06-10 12:15:00 | 100.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-06-10 09:30:00 | 100.50 | 2025-06-10 12:15:00 | 100.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-06-10 10:00:00 | 100.40 | 2025-06-10 12:15:00 | 100.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-06-12 14:00:00 | 98.25 | 2025-06-16 09:15:00 | 93.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 14:00:00 | 98.25 | 2025-06-16 13:15:00 | 94.82 | STOP_HIT | 0.50 | 3.49% |
| SELL | retest2 | 2025-07-07 15:15:00 | 101.25 | 2025-07-14 14:15:00 | 100.25 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2025-07-08 10:00:00 | 101.27 | 2025-07-14 14:15:00 | 100.25 | STOP_HIT | 1.00 | 1.01% |
| SELL | retest2 | 2025-07-25 09:30:00 | 95.72 | 2025-07-30 12:15:00 | 94.48 | STOP_HIT | 1.00 | 1.30% |
| SELL | retest2 | 2025-08-05 10:15:00 | 91.08 | 2025-08-08 12:15:00 | 91.00 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-09-10 09:45:00 | 93.10 | 2025-09-12 13:15:00 | 93.01 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-09-11 15:00:00 | 92.80 | 2025-09-12 13:15:00 | 93.01 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2025-10-07 12:45:00 | 92.78 | 2025-10-08 09:15:00 | 92.21 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-10-07 13:30:00 | 92.70 | 2025-10-08 09:15:00 | 92.21 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-10-07 14:00:00 | 92.80 | 2025-10-08 09:15:00 | 92.21 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-10-08 09:15:00 | 92.75 | 2025-10-08 09:15:00 | 92.21 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-10-24 09:15:00 | 94.21 | 2025-10-24 13:15:00 | 93.66 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-10-24 12:00:00 | 94.20 | 2025-10-24 13:15:00 | 93.66 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-11-12 10:15:00 | 98.75 | 2025-11-12 11:15:00 | 101.02 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-11-12 10:45:00 | 98.76 | 2025-11-12 11:15:00 | 101.02 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-11-24 14:30:00 | 98.75 | 2025-11-26 09:15:00 | 102.83 | STOP_HIT | 1.00 | -4.13% |
| SELL | retest2 | 2025-12-01 11:30:00 | 100.92 | 2025-12-08 09:15:00 | 95.93 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2025-12-02 09:45:00 | 100.98 | 2025-12-08 10:15:00 | 95.87 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2025-12-01 11:30:00 | 100.92 | 2025-12-09 11:15:00 | 95.47 | STOP_HIT | 0.50 | 5.40% |
| SELL | retest2 | 2025-12-02 09:45:00 | 100.98 | 2025-12-09 11:15:00 | 95.47 | STOP_HIT | 0.50 | 5.46% |
| SELL | retest2 | 2025-12-18 13:30:00 | 97.45 | 2025-12-19 15:15:00 | 98.40 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-12-18 15:15:00 | 97.32 | 2025-12-19 15:15:00 | 98.40 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-12-19 11:15:00 | 97.62 | 2025-12-19 15:15:00 | 98.40 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-01-09 11:30:00 | 106.00 | 2026-01-14 14:15:00 | 104.88 | STOP_HIT | 1.00 | 1.06% |
| BUY | retest2 | 2026-02-02 09:15:00 | 99.00 | 2026-02-02 10:15:00 | 97.10 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2026-02-02 09:45:00 | 98.73 | 2026-02-02 10:15:00 | 97.10 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2026-02-06 09:15:00 | 104.80 | 2026-02-09 09:15:00 | 103.35 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-02-06 10:30:00 | 103.54 | 2026-02-09 09:15:00 | 103.35 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2026-02-16 14:30:00 | 111.32 | 2026-02-19 14:15:00 | 110.84 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2026-03-02 10:15:00 | 115.50 | 2026-03-02 12:15:00 | 112.35 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2026-03-05 10:15:00 | 110.45 | 2026-03-09 09:15:00 | 104.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 10:15:00 | 110.45 | 2026-03-09 13:15:00 | 99.41 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-13 10:15:00 | 73.21 | 2026-04-22 10:15:00 | 74.23 | STOP_HIT | 1.00 | 1.39% |
| SELL | retest2 | 2026-05-06 10:45:00 | 74.76 | 2026-05-06 15:15:00 | 76.33 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2026-05-06 12:00:00 | 74.77 | 2026-05-06 15:15:00 | 76.33 | STOP_HIT | 1.00 | -2.09% |
