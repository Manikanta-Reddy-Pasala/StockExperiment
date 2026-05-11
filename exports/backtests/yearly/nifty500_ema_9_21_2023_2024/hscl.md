# Himadri Speciality Chemical Ltd. (HSCL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5317 bars)
- **Last close:** 631.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 204 |
| ALERT1 | 140 |
| ALERT2 | 136 |
| ALERT2_SKIP | 64 |
| ALERT3 | 364 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 180 |
| PARTIAL | 21 |
| TARGET_HIT | 11 |
| STOP_HIT | 176 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 206 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 85 / 121
- **Target hits / Stop hits / Partials:** 11 / 174 / 21
- **Avg / median % per leg:** 0.33% / -0.56%
- **Sum % (uncompounded):** 68.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 83 | 25 | 30.1% | 8 | 75 | 0 | 0.23% | 19.0% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.64% | -2.6% |
| BUY @ 3rd Alert (retest2) | 79 | 25 | 31.6% | 8 | 71 | 0 | 0.27% | 21.5% |
| SELL (all) | 123 | 60 | 48.8% | 3 | 99 | 21 | 0.40% | 49.3% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| SELL @ 3rd Alert (retest2) | 121 | 58 | 47.9% | 2 | 99 | 20 | 0.28% | 34.3% |
| retest1 (combined) | 6 | 2 | 33.3% | 1 | 4 | 1 | 2.07% | 12.4% |
| retest2 (combined) | 200 | 83 | 41.5% | 10 | 170 | 20 | 0.28% | 55.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 13:15:00 | 119.00 | 119.92 | 119.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-15 14:15:00 | 118.70 | 119.68 | 119.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-16 09:15:00 | 120.20 | 119.62 | 119.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-16 09:15:00 | 120.20 | 119.62 | 119.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 09:15:00 | 120.20 | 119.62 | 119.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-16 10:00:00 | 120.20 | 119.62 | 119.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 10:15:00 | 119.30 | 119.55 | 119.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-16 11:15:00 | 118.90 | 119.55 | 119.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-17 11:45:00 | 118.65 | 119.17 | 119.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-17 12:45:00 | 118.80 | 119.08 | 119.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-17 14:00:00 | 118.80 | 119.03 | 119.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 14:15:00 | 119.70 | 119.16 | 119.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-17 15:00:00 | 119.70 | 119.16 | 119.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 15:15:00 | 119.40 | 119.21 | 119.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-18 09:15:00 | 120.25 | 119.21 | 119.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 119.65 | 119.30 | 119.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-18 10:15:00 | 119.35 | 119.30 | 119.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-19 11:15:00 | 120.75 | 118.32 | 118.65 | SL hit (close>static) qty=1.00 sl=120.45 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 13:15:00 | 119.95 | 118.92 | 118.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 14:15:00 | 122.20 | 119.57 | 119.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-22 10:15:00 | 119.90 | 120.10 | 119.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-22 11:00:00 | 119.90 | 120.10 | 119.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 13:15:00 | 119.40 | 119.98 | 119.65 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 09:15:00 | 118.10 | 119.44 | 119.47 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 09:15:00 | 122.75 | 119.37 | 119.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-24 10:15:00 | 124.10 | 120.32 | 119.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 09:15:00 | 127.50 | 127.82 | 125.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-26 09:45:00 | 127.40 | 127.82 | 125.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 124.10 | 126.66 | 126.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-29 09:30:00 | 124.30 | 126.66 | 126.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 10:15:00 | 124.40 | 126.21 | 125.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-29 12:00:00 | 124.50 | 125.86 | 125.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-29 12:15:00 | 124.30 | 125.55 | 125.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 12:15:00 | 124.30 | 125.55 | 125.62 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-30 11:15:00 | 127.80 | 125.73 | 125.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 09:15:00 | 128.60 | 126.39 | 125.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 10:15:00 | 127.00 | 127.36 | 126.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 10:15:00 | 127.00 | 127.36 | 126.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 10:15:00 | 127.00 | 127.36 | 126.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-01 11:00:00 | 127.00 | 127.36 | 126.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 11:15:00 | 126.90 | 127.27 | 126.86 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-06-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-01 14:15:00 | 125.55 | 126.47 | 126.55 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 12:15:00 | 126.85 | 126.62 | 126.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 09:15:00 | 131.85 | 127.73 | 127.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-06 09:15:00 | 129.85 | 130.78 | 129.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-06 10:15:00 | 129.65 | 130.78 | 129.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 10:15:00 | 130.40 | 130.71 | 129.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 10:30:00 | 130.50 | 130.71 | 129.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 13:15:00 | 131.45 | 131.60 | 130.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-07 15:00:00 | 131.95 | 131.67 | 130.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-08 13:30:00 | 133.25 | 131.03 | 130.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-09 10:15:00 | 128.70 | 130.41 | 130.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2023-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 10:15:00 | 128.70 | 130.41 | 130.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-15 09:15:00 | 127.80 | 128.86 | 129.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-15 10:15:00 | 131.00 | 129.28 | 129.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-15 10:15:00 | 131.00 | 129.28 | 129.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 10:15:00 | 131.00 | 129.28 | 129.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-15 10:30:00 | 132.60 | 129.28 | 129.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2023-06-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 11:15:00 | 130.40 | 129.51 | 129.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-15 13:15:00 | 132.65 | 130.32 | 129.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-16 13:15:00 | 133.95 | 133.99 | 132.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-16 13:30:00 | 133.15 | 133.99 | 132.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 15:15:00 | 132.30 | 133.57 | 132.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 09:15:00 | 132.70 | 133.57 | 132.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 09:15:00 | 132.75 | 133.41 | 132.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 10:00:00 | 135.30 | 133.07 | 132.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-22 14:15:00 | 133.55 | 134.48 | 134.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2023-06-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 14:15:00 | 133.55 | 134.48 | 134.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 131.25 | 133.79 | 134.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 13:15:00 | 131.70 | 131.48 | 132.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 14:00:00 | 131.70 | 131.48 | 132.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 132.25 | 131.63 | 132.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 14:30:00 | 132.05 | 131.63 | 132.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 15:15:00 | 132.45 | 131.80 | 132.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:15:00 | 133.45 | 131.80 | 132.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 133.50 | 132.14 | 132.45 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 12:15:00 | 133.30 | 132.67 | 132.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 13:15:00 | 133.65 | 132.86 | 132.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-27 14:15:00 | 132.80 | 132.85 | 132.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 14:15:00 | 132.80 | 132.85 | 132.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 14:15:00 | 132.80 | 132.85 | 132.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 14:45:00 | 132.50 | 132.85 | 132.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 15:15:00 | 132.60 | 132.80 | 132.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-28 09:15:00 | 134.50 | 132.80 | 132.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 12:45:00 | 133.10 | 133.64 | 133.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-30 14:15:00 | 132.85 | 133.40 | 133.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-06-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 14:15:00 | 132.85 | 133.40 | 133.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-30 15:15:00 | 132.65 | 133.25 | 133.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 09:15:00 | 131.15 | 130.19 | 131.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 09:15:00 | 131.15 | 130.19 | 131.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 131.15 | 130.19 | 131.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-05 09:30:00 | 131.00 | 130.19 | 131.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 10:15:00 | 130.00 | 130.15 | 130.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-05 11:15:00 | 129.85 | 130.15 | 130.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-06 09:15:00 | 131.45 | 129.84 | 130.36 | SL hit (close>static) qty=1.00 sl=131.25 alert=retest2 |

### Cycle 14 — BUY (started 2023-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 10:15:00 | 131.90 | 129.17 | 128.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 09:15:00 | 134.70 | 131.38 | 130.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 13:15:00 | 134.90 | 134.96 | 133.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-13 14:00:00 | 134.90 | 134.96 | 133.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 14:15:00 | 135.30 | 135.03 | 133.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 14:30:00 | 133.45 | 135.03 | 133.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 143.40 | 142.50 | 140.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 09:30:00 | 141.45 | 142.50 | 140.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 09:15:00 | 142.40 | 142.94 | 141.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-19 10:00:00 | 142.40 | 142.94 | 141.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 10:15:00 | 141.45 | 142.64 | 141.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-19 11:00:00 | 141.45 | 142.64 | 141.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 11:15:00 | 141.45 | 142.40 | 141.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 09:15:00 | 142.60 | 142.03 | 141.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 10:00:00 | 142.30 | 142.08 | 141.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-20 11:15:00 | 141.00 | 141.77 | 141.67 | SL hit (close<static) qty=1.00 sl=141.05 alert=retest2 |

### Cycle 15 — SELL (started 2023-07-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 12:15:00 | 140.75 | 141.57 | 141.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-20 13:15:00 | 140.15 | 141.29 | 141.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-21 10:15:00 | 141.45 | 140.68 | 141.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 10:15:00 | 141.45 | 140.68 | 141.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 10:15:00 | 141.45 | 140.68 | 141.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-21 10:45:00 | 141.40 | 140.68 | 141.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 11:15:00 | 141.35 | 140.81 | 141.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-21 12:30:00 | 140.35 | 140.75 | 141.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-21 13:30:00 | 140.70 | 140.78 | 141.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-24 11:45:00 | 140.75 | 140.94 | 141.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-24 12:45:00 | 140.70 | 140.88 | 140.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 13:15:00 | 140.70 | 140.84 | 140.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 13:30:00 | 141.05 | 140.84 | 140.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 140.20 | 140.12 | 140.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-25 10:30:00 | 138.60 | 139.78 | 140.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-26 11:30:00 | 139.20 | 139.17 | 139.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-26 13:00:00 | 138.80 | 139.10 | 139.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-31 13:15:00 | 139.30 | 138.13 | 138.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2023-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 13:15:00 | 139.30 | 138.13 | 138.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 14:15:00 | 139.40 | 138.38 | 138.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 10:15:00 | 138.85 | 138.99 | 138.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 10:15:00 | 138.85 | 138.99 | 138.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 138.85 | 138.99 | 138.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 10:45:00 | 138.90 | 138.99 | 138.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 13:15:00 | 138.90 | 139.09 | 138.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 14:00:00 | 138.90 | 139.09 | 138.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 14:15:00 | 139.40 | 139.15 | 138.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-02 09:30:00 | 139.70 | 139.43 | 138.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-02 10:15:00 | 138.20 | 139.18 | 138.91 | SL hit (close<static) qty=1.00 sl=138.75 alert=retest2 |

### Cycle 17 — SELL (started 2023-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 12:15:00 | 136.15 | 138.25 | 138.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 135.05 | 137.61 | 138.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 10:15:00 | 137.50 | 137.31 | 137.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-03 11:00:00 | 137.50 | 137.31 | 137.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 11:15:00 | 139.90 | 137.83 | 138.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 12:00:00 | 139.90 | 137.83 | 138.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2023-08-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 12:15:00 | 140.00 | 138.26 | 138.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 09:15:00 | 141.70 | 139.40 | 138.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-09 10:15:00 | 150.75 | 151.70 | 149.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-09 10:45:00 | 150.75 | 151.70 | 149.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 163.60 | 162.64 | 159.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 09:30:00 | 161.65 | 162.64 | 159.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 10:15:00 | 166.10 | 168.62 | 167.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 10:45:00 | 163.95 | 168.62 | 167.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 11:15:00 | 165.80 | 168.06 | 166.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 12:00:00 | 165.80 | 168.06 | 166.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 12:15:00 | 166.45 | 167.73 | 166.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-18 13:45:00 | 167.55 | 167.78 | 167.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 09:15:00 | 173.15 | 167.45 | 167.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-08-21 15:15:00 | 184.31 | 177.31 | 172.95 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2023-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-11 11:15:00 | 257.55 | 266.21 | 266.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 09:15:00 | 238.85 | 256.57 | 261.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 12:15:00 | 234.15 | 234.08 | 243.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 12:45:00 | 232.45 | 234.08 | 243.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 244.40 | 235.84 | 241.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 10:00:00 | 244.40 | 235.84 | 241.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 10:15:00 | 245.15 | 237.70 | 241.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 10:30:00 | 245.05 | 237.70 | 241.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 11:15:00 | 243.30 | 238.82 | 241.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 12:30:00 | 241.75 | 239.34 | 241.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 09:15:00 | 238.50 | 241.54 | 242.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-18 09:15:00 | 241.60 | 240.66 | 241.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-18 10:00:00 | 241.25 | 240.78 | 241.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 10:15:00 | 240.05 | 240.63 | 240.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-20 09:30:00 | 239.10 | 239.64 | 240.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-21 09:45:00 | 239.50 | 238.26 | 239.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-21 10:15:00 | 239.30 | 238.26 | 239.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-21 10:45:00 | 239.00 | 238.39 | 239.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 11:15:00 | 237.50 | 238.21 | 238.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 11:30:00 | 238.40 | 238.21 | 238.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 14:15:00 | 238.10 | 237.55 | 238.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 15:00:00 | 238.10 | 237.55 | 238.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 15:15:00 | 238.00 | 237.64 | 238.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 09:15:00 | 236.95 | 237.64 | 238.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-25 10:00:00 | 235.90 | 234.73 | 235.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-25 12:15:00 | 241.00 | 236.60 | 236.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2023-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 12:15:00 | 241.00 | 236.60 | 236.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 14:15:00 | 246.55 | 239.42 | 237.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 13:15:00 | 245.75 | 245.84 | 242.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-26 14:00:00 | 245.75 | 245.84 | 242.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 248.85 | 250.09 | 248.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-28 14:15:00 | 251.35 | 250.09 | 248.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-28 14:15:00 | 246.30 | 249.33 | 247.94 | SL hit (close<static) qty=1.00 sl=248.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 11:15:00 | 244.00 | 247.18 | 247.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-03 09:15:00 | 242.50 | 244.64 | 245.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-03 10:15:00 | 250.10 | 245.73 | 246.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 10:15:00 | 250.10 | 245.73 | 246.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 10:15:00 | 250.10 | 245.73 | 246.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-03 10:45:00 | 252.90 | 245.73 | 246.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2023-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 11:15:00 | 254.35 | 247.46 | 246.98 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 14:15:00 | 247.80 | 248.83 | 248.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 09:15:00 | 239.95 | 245.65 | 246.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 243.45 | 239.94 | 242.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 243.45 | 239.94 | 242.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 243.45 | 239.94 | 242.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 09:30:00 | 241.90 | 239.94 | 242.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 243.15 | 240.58 | 242.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 10:30:00 | 244.50 | 240.58 | 242.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 11:15:00 | 243.50 | 241.17 | 242.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 11:30:00 | 244.45 | 241.17 | 242.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 13:15:00 | 244.00 | 242.03 | 242.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 14:00:00 | 244.00 | 242.03 | 242.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 246.75 | 243.59 | 243.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 09:15:00 | 254.50 | 246.77 | 245.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 10:15:00 | 253.80 | 254.30 | 250.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-13 11:00:00 | 253.80 | 254.30 | 250.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 11:15:00 | 256.80 | 256.81 | 255.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-17 11:45:00 | 255.45 | 256.81 | 255.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 14:15:00 | 257.45 | 256.85 | 255.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-17 14:45:00 | 256.10 | 256.85 | 255.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 255.70 | 256.70 | 255.88 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 12:15:00 | 253.50 | 255.08 | 255.25 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 13:15:00 | 256.95 | 255.45 | 255.41 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 14:15:00 | 254.00 | 255.16 | 255.28 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-10-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-19 09:15:00 | 256.00 | 255.46 | 255.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-19 14:15:00 | 261.00 | 256.93 | 256.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-20 11:15:00 | 259.75 | 260.52 | 258.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 11:15:00 | 259.75 | 260.52 | 258.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 11:15:00 | 259.75 | 260.52 | 258.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 12:00:00 | 259.75 | 260.52 | 258.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 12:15:00 | 258.10 | 260.04 | 258.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 13:00:00 | 258.10 | 260.04 | 258.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 13:15:00 | 259.50 | 259.93 | 258.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-20 14:45:00 | 260.05 | 260.03 | 258.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-23 09:15:00 | 248.05 | 257.80 | 257.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2023-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 09:15:00 | 248.05 | 257.80 | 257.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 09:15:00 | 245.65 | 249.56 | 252.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 15:15:00 | 235.00 | 234.57 | 239.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-27 09:15:00 | 239.15 | 234.57 | 239.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 241.00 | 235.85 | 240.02 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 13:15:00 | 241.65 | 240.52 | 240.45 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 14:15:00 | 239.00 | 240.41 | 240.54 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-11-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 09:15:00 | 243.00 | 240.63 | 240.60 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-11-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 10:15:00 | 239.80 | 240.46 | 240.53 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-11-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 12:15:00 | 241.75 | 240.67 | 240.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-01 13:15:00 | 243.95 | 241.33 | 240.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-02 11:15:00 | 241.85 | 242.19 | 241.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 11:15:00 | 241.85 | 242.19 | 241.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 241.85 | 242.19 | 241.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:30:00 | 241.45 | 242.19 | 241.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 12:15:00 | 242.20 | 242.19 | 241.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-02 12:45:00 | 241.75 | 242.19 | 241.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 13:15:00 | 241.60 | 242.07 | 241.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-02 14:00:00 | 241.60 | 242.07 | 241.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 14:15:00 | 242.00 | 242.06 | 241.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-02 14:30:00 | 242.10 | 242.06 | 241.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 15:15:00 | 243.00 | 242.25 | 241.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-03 09:15:00 | 243.50 | 242.25 | 241.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-10 11:15:00 | 249.50 | 251.36 | 251.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 11:15:00 | 249.50 | 251.36 | 251.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 15:15:00 | 248.50 | 250.19 | 250.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-13 09:15:00 | 250.60 | 250.24 | 250.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-13 09:15:00 | 250.60 | 250.24 | 250.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 09:15:00 | 250.60 | 250.24 | 250.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-13 10:00:00 | 250.60 | 250.24 | 250.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 10:15:00 | 251.60 | 250.51 | 250.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-13 10:30:00 | 253.25 | 250.51 | 250.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 11:15:00 | 251.80 | 250.77 | 250.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-13 11:30:00 | 253.10 | 250.77 | 250.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2023-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 12:15:00 | 258.90 | 252.40 | 251.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-13 13:15:00 | 261.50 | 254.22 | 252.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-15 15:15:00 | 257.55 | 258.13 | 256.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-16 09:15:00 | 258.25 | 258.13 | 256.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 09:15:00 | 255.45 | 257.59 | 256.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-16 10:00:00 | 255.45 | 257.59 | 256.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 10:15:00 | 257.00 | 257.48 | 256.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-16 12:45:00 | 259.45 | 257.44 | 256.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-17 11:15:00 | 254.00 | 255.91 | 256.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2023-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 11:15:00 | 254.00 | 255.91 | 256.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 13:15:00 | 252.25 | 254.90 | 255.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 09:15:00 | 254.65 | 254.37 | 255.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 09:15:00 | 254.65 | 254.37 | 255.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 254.65 | 254.37 | 255.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 09:30:00 | 258.05 | 254.37 | 255.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 13:15:00 | 254.25 | 253.90 | 254.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 14:00:00 | 254.25 | 253.90 | 254.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 14:15:00 | 255.30 | 254.18 | 254.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 15:00:00 | 255.30 | 254.18 | 254.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 15:15:00 | 257.65 | 254.88 | 254.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 09:15:00 | 262.50 | 254.88 | 254.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2023-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 09:15:00 | 260.75 | 256.05 | 255.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 11:15:00 | 267.00 | 259.34 | 257.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 12:15:00 | 264.95 | 265.44 | 262.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 12:15:00 | 264.95 | 265.44 | 262.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 12:15:00 | 264.95 | 265.44 | 262.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 12:30:00 | 264.40 | 265.44 | 262.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 10:15:00 | 262.95 | 264.61 | 263.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-23 10:45:00 | 262.25 | 264.61 | 263.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 11:15:00 | 259.95 | 263.68 | 262.76 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-11-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 14:15:00 | 260.20 | 261.97 | 262.11 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 09:15:00 | 264.00 | 262.19 | 262.18 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 10:15:00 | 259.75 | 261.70 | 261.96 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 13:15:00 | 263.95 | 262.42 | 262.25 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-11-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 09:15:00 | 257.50 | 262.17 | 262.73 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-11-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 14:15:00 | 274.75 | 264.25 | 263.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 09:15:00 | 279.00 | 272.70 | 270.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 10:15:00 | 300.30 | 304.26 | 298.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 10:15:00 | 300.30 | 304.26 | 298.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 10:15:00 | 300.30 | 304.26 | 298.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 10:45:00 | 297.30 | 304.26 | 298.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 11:15:00 | 296.85 | 302.78 | 298.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 11:45:00 | 296.30 | 302.78 | 298.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 12:15:00 | 296.00 | 301.43 | 298.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 09:15:00 | 299.75 | 299.13 | 297.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-11 09:15:00 | 295.00 | 298.30 | 297.55 | SL hit (close<static) qty=1.00 sl=295.10 alert=retest2 |

### Cycle 45 — SELL (started 2023-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 12:15:00 | 296.25 | 297.09 | 297.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 15:15:00 | 295.00 | 296.31 | 296.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 09:15:00 | 294.75 | 292.26 | 293.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 294.75 | 292.26 | 293.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 294.75 | 292.26 | 293.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-14 10:15:00 | 292.65 | 292.26 | 293.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-14 11:30:00 | 293.50 | 292.68 | 293.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-19 09:30:00 | 294.10 | 291.69 | 291.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-19 10:15:00 | 292.45 | 291.85 | 291.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2023-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 10:15:00 | 292.45 | 291.85 | 291.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-19 11:15:00 | 293.20 | 292.12 | 291.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 09:15:00 | 291.75 | 293.13 | 292.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 09:15:00 | 291.75 | 293.13 | 292.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 09:15:00 | 291.75 | 293.13 | 292.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 10:00:00 | 291.75 | 293.13 | 292.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 295.20 | 293.55 | 292.85 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 279.60 | 290.11 | 291.41 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2023-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 09:15:00 | 297.10 | 287.47 | 286.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 14:15:00 | 307.05 | 297.60 | 292.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 09:15:00 | 303.20 | 306.48 | 301.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-28 10:00:00 | 303.20 | 306.48 | 301.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 14:15:00 | 385.80 | 388.00 | 376.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 14:45:00 | 382.55 | 388.00 | 376.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 375.00 | 385.71 | 377.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 09:30:00 | 378.50 | 385.71 | 377.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 10:15:00 | 374.90 | 383.55 | 377.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 11:00:00 | 374.90 | 383.55 | 377.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 11:15:00 | 371.95 | 381.23 | 376.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 12:00:00 | 371.95 | 381.23 | 376.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 14:15:00 | 381.00 | 379.71 | 376.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 09:15:00 | 390.15 | 379.96 | 377.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 11:45:00 | 384.00 | 381.55 | 378.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-12 10:15:00 | 373.40 | 377.73 | 377.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2024-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 10:15:00 | 373.40 | 377.73 | 377.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 13:15:00 | 371.00 | 375.36 | 376.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-15 09:15:00 | 374.90 | 374.13 | 375.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 09:15:00 | 374.90 | 374.13 | 375.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 374.90 | 374.13 | 375.75 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2024-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 11:15:00 | 377.50 | 375.83 | 375.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 12:15:00 | 388.00 | 378.26 | 376.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 09:15:00 | 384.50 | 384.92 | 381.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 09:15:00 | 384.50 | 384.92 | 381.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 384.50 | 384.92 | 381.00 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 09:15:00 | 365.00 | 377.54 | 378.95 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-01-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 12:15:00 | 372.00 | 368.92 | 368.75 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 09:15:00 | 365.10 | 369.48 | 369.64 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-01-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 12:15:00 | 369.95 | 368.60 | 368.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 13:15:00 | 370.95 | 369.07 | 368.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 14:15:00 | 370.70 | 371.07 | 370.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 14:15:00 | 370.70 | 371.07 | 370.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 14:15:00 | 370.70 | 371.07 | 370.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 15:15:00 | 370.60 | 371.07 | 370.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 15:15:00 | 370.60 | 370.97 | 370.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 09:15:00 | 371.00 | 370.97 | 370.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 370.00 | 370.78 | 370.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 09:45:00 | 368.35 | 370.78 | 370.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 10:15:00 | 370.90 | 370.80 | 370.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 11:00:00 | 370.90 | 370.80 | 370.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 11:15:00 | 368.90 | 370.42 | 370.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 12:00:00 | 368.90 | 370.42 | 370.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 12:15:00 | 369.80 | 370.30 | 370.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 13:00:00 | 369.80 | 370.30 | 370.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 13:15:00 | 370.00 | 370.24 | 370.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 13:45:00 | 370.15 | 370.24 | 370.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 14:15:00 | 370.70 | 370.33 | 370.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 14:30:00 | 370.00 | 370.33 | 370.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 15:15:00 | 370.00 | 370.26 | 370.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 09:15:00 | 365.25 | 370.26 | 370.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2024-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 09:15:00 | 369.50 | 370.11 | 370.14 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 14:15:00 | 373.50 | 370.30 | 369.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 15:15:00 | 374.90 | 371.22 | 370.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 09:15:00 | 376.95 | 378.10 | 375.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 09:15:00 | 376.95 | 378.10 | 375.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 376.95 | 378.10 | 375.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 09:45:00 | 379.55 | 378.10 | 375.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 372.25 | 376.93 | 375.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 11:00:00 | 372.25 | 376.93 | 375.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 11:15:00 | 372.80 | 376.10 | 374.94 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 09:15:00 | 364.75 | 372.64 | 373.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 09:15:00 | 352.10 | 365.47 | 369.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 13:15:00 | 354.95 | 352.77 | 358.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 14:00:00 | 354.95 | 352.77 | 358.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 355.00 | 353.22 | 357.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 14:30:00 | 356.50 | 353.22 | 357.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 354.00 | 352.53 | 356.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 11:00:00 | 354.00 | 352.53 | 356.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 11:15:00 | 356.00 | 353.22 | 356.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 11:30:00 | 354.90 | 353.22 | 356.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 12:15:00 | 355.50 | 353.68 | 356.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 13:00:00 | 355.50 | 353.68 | 356.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 13:15:00 | 355.25 | 353.99 | 356.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 13:30:00 | 356.20 | 353.99 | 356.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 355.00 | 354.19 | 355.99 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-02-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 10:15:00 | 365.00 | 358.28 | 357.53 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-02-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 09:15:00 | 358.00 | 362.48 | 362.71 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 14:15:00 | 364.40 | 362.15 | 362.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 15:15:00 | 368.40 | 363.40 | 362.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 13:15:00 | 365.50 | 365.75 | 364.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 13:15:00 | 365.50 | 365.75 | 364.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 13:15:00 | 365.50 | 365.75 | 364.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 13:30:00 | 363.30 | 365.75 | 364.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 14:15:00 | 364.00 | 365.40 | 364.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 15:00:00 | 364.00 | 365.40 | 364.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 15:15:00 | 365.00 | 365.32 | 364.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 09:30:00 | 361.00 | 365.25 | 364.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 10:15:00 | 366.20 | 365.44 | 364.60 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 10:15:00 | 356.00 | 363.19 | 364.01 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-02-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 15:15:00 | 371.00 | 361.66 | 361.22 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 09:15:00 | 354.10 | 360.98 | 361.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 340.00 | 351.32 | 354.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 15:15:00 | 343.15 | 343.11 | 348.21 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-07 09:15:00 | 340.00 | 343.11 | 348.21 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 09:15:00 | 323.00 | 333.34 | 340.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-03-12 09:15:00 | 306.00 | 317.46 | 327.58 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 64 — BUY (started 2024-03-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 10:15:00 | 311.50 | 302.59 | 301.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 09:15:00 | 319.80 | 311.40 | 307.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 10:15:00 | 314.90 | 317.44 | 313.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-19 11:00:00 | 314.90 | 317.44 | 313.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 11:15:00 | 318.00 | 317.55 | 313.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 11:30:00 | 311.70 | 317.55 | 313.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 12:15:00 | 317.40 | 317.52 | 314.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 12:30:00 | 318.00 | 317.52 | 314.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 14:15:00 | 318.05 | 317.51 | 314.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 14:45:00 | 315.25 | 317.51 | 314.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 313.50 | 317.13 | 315.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 10:00:00 | 313.50 | 317.13 | 315.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 10:15:00 | 311.90 | 316.08 | 314.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 10:45:00 | 309.95 | 316.08 | 314.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 14:15:00 | 318.00 | 317.55 | 315.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-21 09:15:00 | 323.00 | 317.84 | 316.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-21 10:15:00 | 322.10 | 318.09 | 316.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-21 11:00:00 | 322.95 | 319.06 | 317.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-22 11:15:00 | 316.75 | 317.02 | 317.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-03-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 11:15:00 | 316.75 | 317.02 | 317.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 09:15:00 | 312.00 | 314.80 | 315.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 09:15:00 | 312.00 | 311.14 | 313.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 09:15:00 | 312.00 | 311.14 | 313.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 09:15:00 | 312.00 | 311.14 | 313.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 09:30:00 | 310.00 | 311.14 | 313.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 309.00 | 308.87 | 311.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 15:00:00 | 309.00 | 308.87 | 311.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 309.60 | 309.01 | 310.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 14:45:00 | 303.00 | 306.07 | 308.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-01 14:15:00 | 315.20 | 309.68 | 309.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2024-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 14:15:00 | 315.20 | 309.68 | 309.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 15:15:00 | 316.50 | 311.04 | 309.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 14:15:00 | 333.15 | 333.65 | 328.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-04 15:00:00 | 333.15 | 333.65 | 328.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 326.60 | 332.25 | 330.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 10:00:00 | 326.60 | 332.25 | 330.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 10:15:00 | 328.00 | 331.40 | 330.54 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 14:15:00 | 327.60 | 330.09 | 330.15 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 10:15:00 | 335.00 | 330.62 | 330.33 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-04-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 10:15:00 | 326.85 | 329.96 | 330.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 13:15:00 | 325.00 | 328.03 | 329.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 15:15:00 | 310.40 | 310.35 | 313.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-18 09:15:00 | 313.90 | 310.35 | 313.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 324.00 | 313.08 | 314.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:30:00 | 324.10 | 313.08 | 314.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 324.30 | 315.32 | 315.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 10:30:00 | 324.30 | 315.32 | 315.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2024-04-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 11:15:00 | 324.30 | 317.12 | 316.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-19 09:15:00 | 335.85 | 323.71 | 320.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 12:15:00 | 364.00 | 365.40 | 355.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-24 12:30:00 | 359.90 | 365.40 | 355.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 371.60 | 375.39 | 372.27 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-05-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 11:15:00 | 370.30 | 373.24 | 373.50 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-05-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 14:15:00 | 375.15 | 373.71 | 373.63 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 09:15:00 | 358.00 | 370.62 | 372.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 09:15:00 | 353.60 | 359.67 | 364.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 13:15:00 | 363.60 | 355.51 | 360.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 13:15:00 | 363.60 | 355.51 | 360.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 13:15:00 | 363.60 | 355.51 | 360.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 14:00:00 | 363.60 | 355.51 | 360.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 14:15:00 | 364.90 | 357.38 | 361.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 15:15:00 | 367.00 | 357.38 | 361.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 356.00 | 359.23 | 361.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 09:15:00 | 355.85 | 358.81 | 360.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 09:15:00 | 338.06 | 344.04 | 348.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-14 09:15:00 | 341.85 | 340.95 | 344.16 | SL hit (close>ema200) qty=0.50 sl=340.95 alert=retest2 |

### Cycle 74 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 350.25 | 346.35 | 345.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 357.50 | 349.94 | 347.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 15:15:00 | 354.15 | 354.30 | 351.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 09:15:00 | 353.95 | 354.30 | 351.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 349.30 | 353.11 | 351.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:00:00 | 349.30 | 353.11 | 351.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 349.00 | 352.29 | 351.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 12:15:00 | 350.80 | 352.29 | 351.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 13:45:00 | 350.15 | 351.30 | 350.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 14:15:00 | 349.60 | 351.30 | 350.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 15:15:00 | 349.80 | 350.72 | 350.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 15:15:00 | 349.80 | 350.54 | 350.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 15:15:00 | 349.80 | 350.54 | 350.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-17 13:15:00 | 346.70 | 349.34 | 349.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 14:15:00 | 349.50 | 349.37 | 349.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-17 15:00:00 | 349.50 | 349.37 | 349.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 349.00 | 349.30 | 349.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:15:00 | 353.00 | 349.30 | 349.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 09:15:00 | 355.00 | 350.44 | 350.29 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 11:15:00 | 348.00 | 350.07 | 350.30 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 13:15:00 | 356.05 | 351.10 | 350.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 14:15:00 | 363.10 | 353.50 | 351.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 14:15:00 | 362.05 | 363.40 | 358.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-22 14:30:00 | 362.75 | 363.40 | 358.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 359.40 | 362.39 | 359.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 11:00:00 | 359.40 | 362.39 | 359.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 359.55 | 361.82 | 359.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 11:45:00 | 359.45 | 361.82 | 359.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 12:15:00 | 359.30 | 361.32 | 359.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 12:45:00 | 359.65 | 361.32 | 359.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 356.60 | 360.37 | 359.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 13:45:00 | 356.60 | 360.37 | 359.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 355.85 | 359.47 | 358.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 14:45:00 | 355.65 | 359.47 | 358.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2024-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 15:15:00 | 355.00 | 358.58 | 358.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 11:15:00 | 353.00 | 356.56 | 357.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 346.60 | 342.58 | 345.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 346.60 | 342.58 | 345.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 346.60 | 342.58 | 345.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:00:00 | 346.60 | 342.58 | 345.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 345.80 | 343.22 | 345.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 13:15:00 | 344.55 | 344.09 | 345.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 14:00:00 | 344.25 | 344.12 | 345.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 14:15:00 | 340.00 | 338.98 | 338.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 14:15:00 | 340.00 | 338.98 | 338.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 15:15:00 | 342.60 | 339.70 | 339.29 | Break + close above crossover candle high |

### Cycle 81 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 323.90 | 336.58 | 337.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 307.70 | 323.98 | 330.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 326.45 | 324.48 | 330.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 326.45 | 324.48 | 330.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 326.65 | 324.91 | 329.70 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 336.30 | 331.31 | 330.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 13:15:00 | 340.60 | 338.10 | 335.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 12:15:00 | 354.55 | 355.75 | 350.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 12:45:00 | 353.85 | 355.75 | 350.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 359.60 | 360.18 | 357.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:15:00 | 361.55 | 360.18 | 357.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-20 09:15:00 | 397.71 | 381.93 | 373.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 09:15:00 | 390.65 | 395.33 | 395.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 382.80 | 390.66 | 393.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 13:15:00 | 390.85 | 389.06 | 390.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 13:15:00 | 390.85 | 389.06 | 390.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 390.85 | 389.06 | 390.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 14:00:00 | 390.85 | 389.06 | 390.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 394.05 | 390.06 | 391.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 14:30:00 | 398.00 | 390.06 | 391.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 390.40 | 390.12 | 391.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:15:00 | 398.30 | 390.12 | 391.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 399.40 | 391.98 | 391.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 406.70 | 398.01 | 395.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 399.10 | 399.57 | 396.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 12:15:00 | 399.10 | 399.57 | 396.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 399.10 | 399.57 | 396.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 399.10 | 399.57 | 396.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 405.00 | 405.24 | 402.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:30:00 | 403.10 | 405.24 | 402.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 403.25 | 405.01 | 403.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 13:45:00 | 403.50 | 405.01 | 403.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 402.35 | 404.48 | 403.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 402.35 | 404.48 | 403.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 404.00 | 404.38 | 403.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 402.50 | 404.38 | 403.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 400.80 | 403.67 | 402.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:45:00 | 400.15 | 403.67 | 402.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 400.50 | 403.03 | 402.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 10:30:00 | 400.60 | 403.03 | 402.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 405.00 | 403.41 | 402.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 09:15:00 | 414.55 | 404.74 | 404.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 15:15:00 | 422.05 | 424.49 | 424.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 15:15:00 | 422.05 | 424.49 | 424.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 09:15:00 | 414.90 | 422.57 | 423.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 422.35 | 420.28 | 421.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 422.35 | 420.28 | 421.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 422.35 | 420.28 | 421.60 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2024-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 14:15:00 | 431.95 | 423.98 | 423.03 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 415.90 | 422.72 | 422.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 15:15:00 | 414.00 | 418.84 | 420.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 410.60 | 405.86 | 410.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 410.60 | 405.86 | 410.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 410.60 | 405.86 | 410.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:45:00 | 407.55 | 405.86 | 410.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 409.60 | 406.61 | 410.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 414.95 | 406.61 | 410.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 406.25 | 405.84 | 408.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 391.00 | 407.20 | 408.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 413.00 | 408.14 | 408.45 | SL hit (close>static) qty=1.00 sl=412.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 13:15:00 | 409.85 | 408.73 | 408.68 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 09:15:00 | 404.90 | 408.22 | 408.48 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 419.40 | 408.43 | 407.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 12:15:00 | 422.55 | 413.24 | 410.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 12:15:00 | 417.85 | 419.24 | 415.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 15:15:00 | 419.80 | 418.85 | 416.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 15:15:00 | 419.80 | 418.85 | 416.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 09:15:00 | 429.05 | 418.85 | 416.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-06 09:15:00 | 471.96 | 455.45 | 450.75 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 12:15:00 | 460.70 | 466.29 | 467.02 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 14:15:00 | 477.00 | 468.48 | 467.90 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 461.25 | 468.12 | 468.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 459.00 | 466.30 | 467.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 11:15:00 | 455.80 | 455.76 | 461.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 12:00:00 | 455.80 | 455.76 | 461.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 13:15:00 | 456.65 | 455.99 | 460.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 13:45:00 | 458.85 | 455.99 | 460.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 462.65 | 457.32 | 460.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 14:45:00 | 464.55 | 457.32 | 460.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 465.00 | 458.86 | 460.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:15:00 | 462.00 | 458.86 | 460.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 460.85 | 459.26 | 460.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:45:00 | 462.70 | 459.26 | 460.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 468.40 | 461.08 | 461.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:00:00 | 468.40 | 461.08 | 461.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 12:15:00 | 467.55 | 462.38 | 462.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 482.00 | 467.46 | 464.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 09:15:00 | 482.75 | 483.04 | 478.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 10:00:00 | 482.75 | 483.04 | 478.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 484.70 | 484.40 | 482.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:00:00 | 484.70 | 484.40 | 482.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 483.20 | 484.90 | 482.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:45:00 | 483.05 | 484.90 | 482.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 486.00 | 484.86 | 483.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:45:00 | 483.30 | 484.86 | 483.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 483.10 | 484.51 | 483.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:00:00 | 483.10 | 484.51 | 483.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 483.80 | 484.37 | 483.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:45:00 | 483.90 | 484.37 | 483.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 517.80 | 491.05 | 486.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 13:15:00 | 524.35 | 491.05 | 486.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 13:45:00 | 525.50 | 497.94 | 489.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 11:00:00 | 520.95 | 511.70 | 499.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 12:30:00 | 522.60 | 524.65 | 523.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 13:15:00 | 524.45 | 524.61 | 523.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 14:00:00 | 524.45 | 524.61 | 523.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 525.50 | 524.79 | 524.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 14:30:00 | 522.45 | 524.79 | 524.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 524.50 | 524.73 | 524.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 523.00 | 524.73 | 524.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 519.50 | 523.68 | 523.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:45:00 | 520.35 | 523.68 | 523.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-30 10:15:00 | 518.65 | 522.68 | 523.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 10:15:00 | 518.65 | 522.68 | 523.19 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 09:15:00 | 549.80 | 527.00 | 524.52 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 15:15:00 | 530.00 | 532.95 | 533.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 517.70 | 529.90 | 531.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 12:15:00 | 528.95 | 528.68 | 530.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 13:00:00 | 528.95 | 528.68 | 530.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 531.60 | 529.26 | 530.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:00:00 | 531.60 | 529.26 | 530.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 529.55 | 529.32 | 530.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:15:00 | 531.90 | 529.32 | 530.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 531.90 | 529.84 | 530.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 535.80 | 529.84 | 530.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 532.80 | 530.43 | 531.00 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 11:15:00 | 538.15 | 532.51 | 531.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 12:15:00 | 549.75 | 535.96 | 533.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 09:15:00 | 552.85 | 558.89 | 551.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-12 09:45:00 | 554.50 | 558.89 | 551.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 553.85 | 557.88 | 552.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 14:15:00 | 559.95 | 554.96 | 552.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-17 10:15:00 | 615.95 | 594.96 | 581.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 644.35 | 656.98 | 657.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 640.40 | 653.66 | 655.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 649.50 | 647.31 | 651.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 10:00:00 | 649.50 | 647.31 | 651.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 641.55 | 646.16 | 650.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 12:30:00 | 639.25 | 644.51 | 648.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 607.29 | 634.42 | 642.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 606.60 | 606.05 | 620.30 | SL hit (close>ema200) qty=0.50 sl=606.05 alert=retest2 |

### Cycle 100 — BUY (started 2024-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 12:15:00 | 632.90 | 621.77 | 620.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 639.00 | 628.57 | 624.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 13:15:00 | 629.65 | 629.69 | 626.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 14:15:00 | 628.15 | 629.69 | 626.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 631.45 | 630.04 | 626.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:30:00 | 628.95 | 630.04 | 626.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 631.05 | 629.85 | 627.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 13:15:00 | 640.90 | 629.68 | 627.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 13:45:00 | 634.30 | 632.55 | 630.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 10:30:00 | 635.30 | 633.52 | 632.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 11:15:00 | 638.45 | 633.52 | 632.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 640.00 | 634.82 | 632.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 12:30:00 | 643.45 | 636.05 | 633.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 14:00:00 | 644.55 | 637.75 | 634.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 10:15:00 | 628.80 | 646.85 | 644.28 | SL hit (close<static) qty=1.00 sl=632.50 alert=retest2 |

### Cycle 101 — SELL (started 2024-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 12:15:00 | 623.05 | 638.77 | 640.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 620.00 | 635.02 | 638.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 12:15:00 | 626.50 | 626.01 | 631.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 13:00:00 | 626.50 | 626.01 | 631.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 621.10 | 625.03 | 630.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 15:00:00 | 620.60 | 624.15 | 629.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 10:45:00 | 619.50 | 621.23 | 627.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 589.57 | 606.41 | 616.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 588.52 | 606.41 | 616.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-23 09:15:00 | 558.54 | 588.68 | 601.52 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 102 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 572.80 | 569.55 | 569.13 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 15:15:00 | 566.00 | 568.66 | 568.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 13:15:00 | 565.40 | 567.57 | 568.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 14:15:00 | 572.85 | 568.63 | 568.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 14:15:00 | 572.85 | 568.63 | 568.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 572.85 | 568.63 | 568.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:45:00 | 570.15 | 568.63 | 568.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2024-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 15:15:00 | 574.90 | 569.88 | 569.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 579.55 | 571.82 | 570.20 | Break + close above crossover candle high |

### Cycle 105 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 555.70 | 569.40 | 569.43 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 12:15:00 | 573.70 | 567.75 | 567.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 13:15:00 | 588.65 | 571.93 | 569.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 582.25 | 585.20 | 580.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 582.25 | 585.20 | 580.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 582.20 | 584.60 | 580.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 579.95 | 584.60 | 580.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 579.40 | 583.56 | 580.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 579.40 | 583.56 | 580.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 580.40 | 582.93 | 580.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:00:00 | 580.40 | 582.93 | 580.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 580.20 | 582.38 | 580.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:30:00 | 578.20 | 582.38 | 580.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 577.05 | 581.32 | 580.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 577.05 | 581.32 | 580.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 576.00 | 580.25 | 579.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 573.85 | 580.25 | 579.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 572.50 | 578.70 | 579.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 561.30 | 573.30 | 576.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 10:15:00 | 548.90 | 548.63 | 556.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 11:00:00 | 548.90 | 548.63 | 556.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 510.70 | 503.43 | 509.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:15:00 | 499.25 | 503.69 | 507.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:45:00 | 497.35 | 501.60 | 506.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 09:30:00 | 497.00 | 488.14 | 491.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 13:15:00 | 496.85 | 492.77 | 492.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2024-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 13:15:00 | 496.85 | 492.77 | 492.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 14:15:00 | 499.45 | 494.11 | 492.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 14:15:00 | 522.80 | 524.79 | 516.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 15:00:00 | 522.80 | 524.79 | 516.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 516.00 | 523.08 | 517.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:45:00 | 520.50 | 523.08 | 517.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 517.35 | 521.94 | 517.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 13:45:00 | 523.00 | 521.58 | 518.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-05 13:15:00 | 575.30 | 559.49 | 550.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 11:15:00 | 557.85 | 564.86 | 564.94 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 13:15:00 | 568.90 | 564.91 | 564.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 14:15:00 | 569.75 | 565.88 | 565.05 | Break + close above crossover candle high |

### Cycle 111 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 557.90 | 564.48 | 564.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 11:15:00 | 555.25 | 561.89 | 563.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 14:15:00 | 565.00 | 561.66 | 562.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 14:15:00 | 565.00 | 561.66 | 562.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 565.00 | 561.66 | 562.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 14:45:00 | 565.00 | 561.66 | 562.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 563.00 | 561.93 | 562.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 558.70 | 561.93 | 562.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 12:15:00 | 561.40 | 559.42 | 561.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 14:30:00 | 562.10 | 560.99 | 561.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 15:00:00 | 561.85 | 560.99 | 561.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 560.05 | 560.80 | 561.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 564.15 | 560.80 | 561.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 560.70 | 560.78 | 561.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:00:00 | 557.15 | 560.06 | 560.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 14:00:00 | 559.00 | 559.19 | 560.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 11:30:00 | 558.90 | 557.88 | 559.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 530.76 | 546.45 | 551.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 533.33 | 546.45 | 551.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 534.00 | 546.45 | 551.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 533.76 | 546.45 | 551.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 531.05 | 546.45 | 551.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 530.95 | 546.45 | 551.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 544.45 | 542.04 | 545.72 | SL hit (close>ema200) qty=0.50 sl=542.04 alert=retest2 |

### Cycle 112 — BUY (started 2024-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 15:15:00 | 552.65 | 546.88 | 546.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 10:15:00 | 556.80 | 549.77 | 547.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-24 14:15:00 | 552.10 | 552.10 | 549.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-24 15:00:00 | 552.10 | 552.10 | 549.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 573.20 | 578.36 | 573.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:00:00 | 573.20 | 578.36 | 573.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 573.80 | 577.45 | 573.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:45:00 | 571.90 | 577.45 | 573.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 578.75 | 577.71 | 573.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:30:00 | 574.60 | 577.71 | 573.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 582.00 | 581.98 | 577.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:45:00 | 581.05 | 581.98 | 577.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 578.80 | 582.18 | 579.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 09:45:00 | 587.00 | 583.24 | 580.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 577.80 | 589.20 | 587.61 | SL hit (close<static) qty=1.00 sl=578.10 alert=retest2 |

### Cycle 113 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 570.10 | 585.38 | 586.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 15:15:00 | 561.95 | 573.77 | 579.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 12:15:00 | 571.75 | 570.88 | 576.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 12:45:00 | 568.50 | 570.88 | 576.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 567.10 | 569.93 | 574.02 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 12:15:00 | 582.00 | 574.58 | 573.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-09 13:15:00 | 584.65 | 576.60 | 574.79 | Break + close above crossover candle high |

### Cycle 115 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 561.10 | 574.06 | 574.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 14:15:00 | 550.35 | 559.46 | 565.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 15:15:00 | 549.80 | 548.27 | 554.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 09:15:00 | 554.25 | 548.27 | 554.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 551.20 | 548.86 | 554.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 13:15:00 | 546.35 | 550.20 | 553.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 11:00:00 | 548.25 | 546.72 | 550.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 11:45:00 | 547.70 | 546.72 | 549.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 519.03 | 532.06 | 536.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 520.84 | 532.06 | 536.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 520.32 | 532.06 | 536.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-22 14:15:00 | 532.20 | 529.78 | 533.29 | SL hit (close>ema200) qty=0.50 sl=529.78 alert=retest2 |

### Cycle 116 — BUY (started 2025-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 15:15:00 | 487.05 | 484.78 | 484.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 490.00 | 485.82 | 485.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 13:15:00 | 488.70 | 489.06 | 487.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 13:15:00 | 488.70 | 489.06 | 487.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 13:15:00 | 488.70 | 489.06 | 487.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 14:00:00 | 488.70 | 489.06 | 487.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 493.85 | 490.02 | 487.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 15:15:00 | 494.80 | 490.02 | 487.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 12:15:00 | 483.90 | 490.56 | 489.16 | SL hit (close<static) qty=1.00 sl=487.50 alert=retest2 |

### Cycle 117 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 478.80 | 487.35 | 488.47 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2025-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 14:15:00 | 490.45 | 487.48 | 487.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 501.75 | 490.74 | 488.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 10:15:00 | 504.00 | 505.54 | 499.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 11:00:00 | 504.00 | 505.54 | 499.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 498.90 | 503.49 | 500.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:00:00 | 498.90 | 503.49 | 500.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 502.40 | 503.27 | 500.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:30:00 | 499.25 | 503.27 | 500.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 496.90 | 501.70 | 500.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:45:00 | 495.10 | 501.70 | 500.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 500.80 | 501.52 | 500.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 502.20 | 501.52 | 500.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 13:15:00 | 491.90 | 498.43 | 498.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 491.90 | 498.43 | 498.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 481.65 | 493.45 | 496.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 445.45 | 440.59 | 452.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 10:00:00 | 445.45 | 440.59 | 452.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 447.55 | 441.98 | 451.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:30:00 | 449.55 | 441.98 | 451.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 408.90 | 413.62 | 421.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 10:15:00 | 406.85 | 413.62 | 421.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 12:45:00 | 408.35 | 409.47 | 417.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 13:15:00 | 405.10 | 409.47 | 417.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 432.30 | 415.89 | 417.74 | SL hit (close>static) qty=1.00 sl=421.35 alert=retest2 |

### Cycle 120 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 434.70 | 419.65 | 419.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 12:15:00 | 445.80 | 427.53 | 423.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 443.60 | 445.31 | 439.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 11:00:00 | 443.60 | 445.31 | 439.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 444.50 | 450.48 | 445.02 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2025-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 13:15:00 | 439.15 | 444.86 | 445.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 14:15:00 | 434.95 | 442.88 | 444.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 401.85 | 401.17 | 412.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 401.85 | 401.17 | 412.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 410.45 | 402.42 | 409.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 412.75 | 402.42 | 409.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 407.55 | 403.45 | 409.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 13:15:00 | 404.80 | 404.64 | 408.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 419.45 | 409.53 | 409.86 | SL hit (close>static) qty=1.00 sl=412.70 alert=retest2 |

### Cycle 122 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 426.90 | 413.00 | 411.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 432.30 | 420.70 | 415.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 435.50 | 435.69 | 428.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 09:15:00 | 437.55 | 435.69 | 428.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 441.85 | 436.92 | 429.90 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 420.45 | 430.41 | 430.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 12:15:00 | 417.85 | 427.90 | 429.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 411.50 | 406.82 | 410.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 411.50 | 406.82 | 410.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 411.50 | 406.82 | 410.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:15:00 | 402.55 | 405.46 | 408.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 13:00:00 | 403.00 | 404.97 | 407.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 15:00:00 | 402.25 | 404.34 | 406.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 416.65 | 406.37 | 406.91 | SL hit (close>static) qty=1.00 sl=412.10 alert=retest2 |

### Cycle 124 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 415.80 | 408.26 | 407.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 13:15:00 | 419.50 | 412.73 | 410.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 14:15:00 | 426.80 | 429.61 | 425.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 15:00:00 | 426.80 | 429.61 | 425.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 15:15:00 | 427.10 | 429.11 | 425.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:15:00 | 433.50 | 429.11 | 425.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 15:15:00 | 431.50 | 438.29 | 439.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 431.50 | 438.29 | 439.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 11:15:00 | 428.85 | 434.61 | 437.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 430.15 | 429.35 | 432.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 10:45:00 | 430.75 | 429.35 | 432.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 429.05 | 428.48 | 431.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:30:00 | 431.55 | 428.48 | 431.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 432.15 | 428.70 | 431.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 433.95 | 428.70 | 431.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 435.35 | 430.03 | 431.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:15:00 | 428.80 | 430.68 | 431.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:00:00 | 427.80 | 430.11 | 431.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 12:15:00 | 435.25 | 430.98 | 430.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2025-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 12:15:00 | 435.25 | 430.98 | 430.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 13:15:00 | 437.35 | 432.25 | 431.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 09:15:00 | 427.45 | 432.33 | 431.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 427.45 | 432.33 | 431.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 427.45 | 432.33 | 431.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:00:00 | 436.85 | 433.38 | 432.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 10:30:00 | 438.20 | 449.45 | 447.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 12:15:00 | 434.00 | 444.32 | 445.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 434.00 | 444.32 | 445.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 432.30 | 441.92 | 444.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 416.65 | 416.08 | 425.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 416.65 | 416.08 | 425.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 416.65 | 416.08 | 425.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 414.20 | 415.86 | 424.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 14:45:00 | 414.25 | 415.39 | 421.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 10:15:00 | 433.70 | 418.46 | 418.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 433.70 | 418.46 | 418.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 13:15:00 | 438.45 | 425.95 | 421.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 12:15:00 | 444.95 | 445.70 | 439.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 13:00:00 | 444.95 | 445.70 | 439.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 470.00 | 477.89 | 471.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 470.00 | 477.89 | 471.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 469.50 | 476.21 | 471.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 468.50 | 476.21 | 471.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 466.70 | 474.31 | 470.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:00:00 | 466.70 | 474.31 | 470.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 468.80 | 469.55 | 469.52 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2025-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 11:15:00 | 467.65 | 469.17 | 469.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 12:15:00 | 466.55 | 468.64 | 469.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 14:15:00 | 455.65 | 455.61 | 460.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-25 14:45:00 | 457.00 | 455.61 | 460.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 459.00 | 456.35 | 460.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 459.00 | 456.35 | 460.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 453.00 | 454.60 | 457.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:15:00 | 451.35 | 454.60 | 457.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-29 12:15:00 | 460.15 | 455.83 | 457.22 | SL hit (close>static) qty=1.00 sl=458.50 alert=retest2 |

### Cycle 130 — BUY (started 2025-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 15:15:00 | 462.00 | 458.62 | 458.29 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 454.55 | 457.81 | 457.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 10:15:00 | 450.80 | 456.41 | 457.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 10:15:00 | 435.70 | 435.62 | 441.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 10:15:00 | 435.70 | 435.62 | 441.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 435.70 | 435.62 | 441.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:30:00 | 440.35 | 435.62 | 441.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 440.50 | 437.17 | 440.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 15:15:00 | 441.00 | 437.17 | 440.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 441.00 | 437.93 | 440.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 438.70 | 437.93 | 440.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 431.50 | 436.65 | 439.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 10:15:00 | 429.80 | 436.65 | 439.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 11:30:00 | 430.30 | 429.53 | 430.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 14:00:00 | 430.25 | 430.05 | 430.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 14:45:00 | 424.15 | 430.50 | 430.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 423.00 | 429.00 | 430.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 09:15:00 | 415.00 | 429.00 | 430.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 432.80 | 420.98 | 423.76 | SL hit (close>static) qty=1.00 sl=432.60 alert=retest2 |

### Cycle 132 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 433.20 | 426.82 | 426.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 440.80 | 430.65 | 427.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 15:15:00 | 462.80 | 463.39 | 458.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 09:15:00 | 460.50 | 463.39 | 458.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 459.75 | 462.66 | 458.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 458.85 | 462.66 | 458.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 460.70 | 462.27 | 458.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 14:45:00 | 461.95 | 461.91 | 459.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:00:00 | 463.25 | 462.35 | 460.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 12:45:00 | 461.65 | 461.46 | 460.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:45:00 | 462.20 | 462.20 | 460.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 474.15 | 471.53 | 467.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:30:00 | 475.10 | 472.22 | 468.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:00:00 | 475.00 | 472.22 | 468.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 479.60 | 473.52 | 470.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 11:00:00 | 475.10 | 474.17 | 471.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 474.30 | 476.08 | 474.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:00:00 | 474.30 | 476.08 | 474.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 479.30 | 476.72 | 474.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 14:45:00 | 480.10 | 477.08 | 475.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 481.20 | 477.46 | 475.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:15:00 | 479.50 | 477.84 | 475.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 15:15:00 | 476.00 | 476.39 | 476.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 15:15:00 | 476.00 | 476.39 | 476.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 09:15:00 | 471.10 | 475.34 | 475.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 09:15:00 | 481.20 | 470.47 | 471.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 09:15:00 | 481.20 | 470.47 | 471.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 481.20 | 470.47 | 471.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 10:00:00 | 481.20 | 470.47 | 471.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 476.50 | 471.68 | 472.04 | EMA400 retest candle locked (from downside) |

### Cycle 134 — BUY (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 11:15:00 | 480.75 | 473.49 | 472.83 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 12:15:00 | 471.80 | 472.80 | 472.89 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 475.50 | 473.34 | 473.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 14:15:00 | 482.00 | 475.07 | 473.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 14:15:00 | 496.55 | 497.26 | 491.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 15:00:00 | 496.55 | 497.26 | 491.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 496.50 | 496.52 | 491.82 | EMA400 retest candle locked (from upside) |

### Cycle 137 — SELL (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 13:15:00 | 490.00 | 492.97 | 493.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 11:15:00 | 489.25 | 491.40 | 492.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 490.15 | 489.14 | 490.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 490.15 | 489.14 | 490.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 490.15 | 489.14 | 490.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 493.30 | 489.14 | 490.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 487.05 | 488.72 | 490.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 11:15:00 | 486.50 | 488.72 | 490.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 462.17 | 480.16 | 484.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 469.50 | 467.63 | 472.74 | SL hit (close>ema200) qty=0.50 sl=467.63 alert=retest2 |

### Cycle 138 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 450.40 | 444.88 | 444.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 456.00 | 450.74 | 448.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 504.45 | 509.11 | 494.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 09:30:00 | 507.55 | 509.11 | 494.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 496.00 | 503.21 | 498.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:45:00 | 495.10 | 503.21 | 498.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 495.60 | 501.69 | 498.22 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 09:15:00 | 496.20 | 496.86 | 496.88 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 497.10 | 496.91 | 496.90 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 493.40 | 498.19 | 498.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 12:15:00 | 492.85 | 496.13 | 497.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 493.15 | 490.71 | 493.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 14:15:00 | 493.15 | 490.71 | 493.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 493.15 | 490.71 | 493.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:45:00 | 494.45 | 490.71 | 493.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 491.15 | 490.80 | 492.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 496.65 | 490.80 | 492.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 505.80 | 493.80 | 494.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 505.80 | 493.80 | 494.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 508.00 | 496.64 | 495.35 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 13:15:00 | 500.35 | 504.21 | 504.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 15:15:00 | 497.40 | 502.31 | 503.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 506.15 | 503.07 | 503.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 506.15 | 503.07 | 503.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 506.15 | 503.07 | 503.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 506.15 | 503.07 | 503.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 507.80 | 504.02 | 504.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:30:00 | 510.55 | 504.02 | 504.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 11:15:00 | 507.55 | 504.73 | 504.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 13:15:00 | 509.40 | 505.81 | 504.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 14:15:00 | 514.25 | 519.70 | 514.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 14:15:00 | 514.25 | 519.70 | 514.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 514.25 | 519.70 | 514.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 514.25 | 519.70 | 514.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 512.00 | 518.16 | 514.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 516.25 | 518.16 | 514.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 10:30:00 | 515.75 | 516.10 | 513.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:00:00 | 515.30 | 515.43 | 513.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 521.95 | 514.53 | 513.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 515.85 | 514.95 | 514.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:45:00 | 514.75 | 514.95 | 514.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 514.30 | 515.18 | 514.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 514.30 | 515.18 | 514.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 517.50 | 515.64 | 514.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:30:00 | 518.30 | 515.64 | 514.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 515.60 | 516.00 | 515.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:30:00 | 514.65 | 516.00 | 515.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 513.85 | 515.57 | 515.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 513.20 | 515.57 | 515.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 514.35 | 515.33 | 514.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 514.85 | 515.33 | 514.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 513.85 | 515.03 | 514.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:30:00 | 513.30 | 515.03 | 514.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-18 13:15:00 | 512.85 | 514.60 | 514.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 512.85 | 514.60 | 514.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 508.20 | 513.18 | 514.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 10:15:00 | 514.45 | 513.44 | 514.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 10:15:00 | 514.45 | 513.44 | 514.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 514.45 | 513.44 | 514.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 514.20 | 513.44 | 514.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 512.30 | 513.21 | 513.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 12:45:00 | 511.50 | 512.69 | 513.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:30:00 | 511.40 | 512.27 | 513.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:45:00 | 510.50 | 511.12 | 512.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 13:00:00 | 511.50 | 511.30 | 512.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 511.00 | 511.24 | 512.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:15:00 | 509.25 | 511.24 | 512.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 14:15:00 | 513.75 | 511.74 | 512.25 | SL hit (close>static) qty=1.00 sl=512.35 alert=retest2 |

### Cycle 146 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 515.40 | 512.26 | 512.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 517.80 | 513.37 | 512.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 12:15:00 | 513.20 | 514.19 | 513.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 12:15:00 | 513.20 | 514.19 | 513.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 513.20 | 514.19 | 513.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:00:00 | 513.20 | 514.19 | 513.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 513.00 | 513.95 | 513.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:30:00 | 514.50 | 513.78 | 513.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 508.75 | 512.54 | 512.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 508.75 | 512.54 | 512.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 503.25 | 510.68 | 511.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 485.75 | 485.28 | 491.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 485.75 | 485.28 | 491.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 484.30 | 485.49 | 490.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 11:00:00 | 482.85 | 484.96 | 489.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 09:15:00 | 458.71 | 467.46 | 472.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 11:15:00 | 467.55 | 467.20 | 471.75 | SL hit (close>ema200) qty=0.50 sl=467.20 alert=retest2 |

### Cycle 148 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 462.65 | 453.93 | 453.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 12:15:00 | 475.80 | 460.27 | 456.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 15:15:00 | 471.80 | 472.01 | 466.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 09:15:00 | 474.50 | 472.01 | 466.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 467.20 | 469.86 | 467.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:00:00 | 467.20 | 469.86 | 467.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 467.45 | 469.38 | 467.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:30:00 | 467.05 | 469.38 | 467.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 466.00 | 468.70 | 467.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 472.00 | 468.70 | 467.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 470.95 | 475.68 | 476.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 470.95 | 475.68 | 476.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 15:15:00 | 467.95 | 472.67 | 474.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 452.90 | 450.65 | 454.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 452.90 | 450.65 | 454.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 452.90 | 450.65 | 454.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 455.40 | 450.65 | 454.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 454.15 | 451.48 | 453.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 454.15 | 451.48 | 453.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 452.20 | 451.62 | 453.51 | EMA400 retest candle locked (from downside) |

### Cycle 150 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 462.50 | 455.56 | 454.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 466.50 | 458.99 | 456.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 462.50 | 463.52 | 461.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 11:00:00 | 462.50 | 463.52 | 461.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 461.20 | 463.06 | 461.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:45:00 | 461.95 | 463.06 | 461.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 461.10 | 462.67 | 461.45 | EMA400 retest candle locked (from upside) |

### Cycle 151 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 457.90 | 460.56 | 460.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 454.85 | 458.78 | 459.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 457.25 | 456.89 | 458.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 13:45:00 | 458.05 | 456.89 | 458.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 459.20 | 457.04 | 458.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 459.20 | 457.04 | 458.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 458.50 | 457.34 | 458.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 12:15:00 | 456.10 | 457.52 | 458.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:15:00 | 456.30 | 457.44 | 458.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:30:00 | 455.75 | 456.03 | 457.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 483.50 | 458.84 | 457.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 483.50 | 458.84 | 457.21 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 14:15:00 | 465.85 | 467.13 | 467.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 09:15:00 | 463.95 | 466.39 | 466.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 09:15:00 | 471.20 | 465.98 | 466.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 09:15:00 | 471.20 | 465.98 | 466.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 471.20 | 465.98 | 466.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:45:00 | 473.00 | 465.98 | 466.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 472.00 | 467.19 | 466.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 472.25 | 468.56 | 467.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 469.30 | 469.88 | 468.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 12:15:00 | 469.30 | 469.88 | 468.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 469.30 | 469.88 | 468.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 469.30 | 469.88 | 468.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 469.55 | 469.82 | 468.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:15:00 | 468.55 | 469.82 | 468.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 469.75 | 469.80 | 468.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:30:00 | 468.75 | 469.80 | 468.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 469.30 | 470.59 | 469.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:00:00 | 469.30 | 470.59 | 469.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 470.00 | 470.47 | 469.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:30:00 | 468.75 | 470.47 | 469.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 467.45 | 469.87 | 469.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 467.45 | 469.87 | 469.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 467.85 | 469.46 | 469.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 464.00 | 469.46 | 469.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 463.30 | 468.23 | 468.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 11:15:00 | 461.05 | 465.84 | 467.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 10:15:00 | 461.10 | 456.41 | 459.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 10:15:00 | 461.10 | 456.41 | 459.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 461.10 | 456.41 | 459.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 461.10 | 456.41 | 459.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 471.35 | 459.40 | 460.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 471.35 | 459.40 | 460.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — BUY (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 12:15:00 | 487.65 | 465.05 | 462.81 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 14:15:00 | 461.65 | 463.83 | 463.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 456.05 | 461.90 | 463.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 456.00 | 455.63 | 458.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 456.00 | 455.63 | 458.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 449.15 | 446.26 | 449.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 449.15 | 446.26 | 449.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 446.75 | 446.36 | 448.92 | EMA400 retest candle locked (from downside) |

### Cycle 158 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 451.30 | 449.70 | 449.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 11:15:00 | 463.30 | 452.91 | 451.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 13:15:00 | 462.00 | 462.15 | 459.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 14:00:00 | 462.00 | 462.15 | 459.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 458.80 | 461.53 | 459.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 458.85 | 461.53 | 459.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 458.75 | 460.98 | 459.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 457.80 | 460.98 | 459.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 460.10 | 460.80 | 459.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:30:00 | 458.00 | 460.80 | 459.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 456.35 | 459.91 | 459.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:45:00 | 455.80 | 459.91 | 459.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 455.45 | 459.02 | 458.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:30:00 | 455.85 | 459.02 | 458.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — SELL (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 13:15:00 | 456.05 | 458.43 | 458.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 14:15:00 | 452.30 | 457.20 | 458.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 14:15:00 | 455.40 | 454.86 | 456.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 14:15:00 | 455.40 | 454.86 | 456.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 455.40 | 454.86 | 456.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:30:00 | 457.40 | 454.86 | 456.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 455.00 | 454.89 | 456.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 456.80 | 454.89 | 456.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 453.25 | 454.56 | 455.80 | EMA400 retest candle locked (from downside) |

### Cycle 160 — BUY (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 13:15:00 | 465.25 | 457.98 | 457.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 15:15:00 | 468.00 | 461.44 | 458.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 15:15:00 | 464.00 | 464.61 | 462.22 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 09:15:00 | 469.00 | 464.61 | 462.22 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:30:00 | 470.70 | 466.61 | 463.57 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 467.75 | 469.30 | 466.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:45:00 | 467.80 | 469.30 | 466.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 467.50 | 469.43 | 467.75 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-16 14:15:00 | 467.50 | 469.43 | 467.75 | SL hit (close<ema400) qty=1.00 sl=467.75 alert=retest1 |

### Cycle 161 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 461.70 | 466.02 | 466.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 459.00 | 464.62 | 465.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 12:15:00 | 461.20 | 458.99 | 461.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 12:15:00 | 461.20 | 458.99 | 461.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 461.20 | 458.99 | 461.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 14:15:00 | 456.50 | 459.18 | 461.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 469.65 | 461.74 | 462.10 | SL hit (close>static) qty=1.00 sl=463.00 alert=retest2 |

### Cycle 162 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 467.25 | 462.84 | 462.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 470.45 | 465.79 | 464.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 12:15:00 | 469.40 | 469.93 | 467.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 13:00:00 | 469.40 | 469.93 | 467.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 467.25 | 469.40 | 467.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 466.70 | 469.40 | 467.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 471.55 | 469.83 | 467.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:30:00 | 467.15 | 469.83 | 467.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 468.30 | 469.84 | 468.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:45:00 | 469.45 | 469.84 | 468.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 471.50 | 470.17 | 468.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 10:30:00 | 474.85 | 472.20 | 470.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 11:00:00 | 474.80 | 472.20 | 470.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 11:30:00 | 475.55 | 472.61 | 470.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 12:45:00 | 475.00 | 472.69 | 470.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 486.55 | 485.41 | 482.27 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 480.00 | 481.73 | 481.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 480.00 | 481.73 | 481.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 12:15:00 | 478.30 | 481.04 | 481.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 10:15:00 | 455.40 | 454.66 | 458.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 11:00:00 | 455.40 | 454.66 | 458.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 459.50 | 454.45 | 456.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:45:00 | 460.90 | 454.45 | 456.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 461.00 | 455.76 | 456.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:00:00 | 461.00 | 455.76 | 456.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 461.00 | 457.62 | 457.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 463.35 | 460.23 | 459.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 458.50 | 460.41 | 459.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 12:15:00 | 458.50 | 460.41 | 459.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 458.50 | 460.41 | 459.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:45:00 | 458.50 | 460.41 | 459.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 457.20 | 459.77 | 459.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 457.20 | 459.77 | 459.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 456.20 | 458.48 | 458.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 454.30 | 457.65 | 458.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 456.55 | 455.71 | 456.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 456.55 | 455.71 | 456.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 456.55 | 455.71 | 456.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 456.50 | 455.71 | 456.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 455.55 | 455.68 | 456.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 13:15:00 | 455.50 | 455.77 | 456.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 455.25 | 455.99 | 456.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 10:15:00 | 432.72 | 438.90 | 443.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 10:15:00 | 432.49 | 438.90 | 443.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 445.60 | 432.30 | 437.14 | SL hit (close>ema200) qty=0.50 sl=432.30 alert=retest2 |

### Cycle 166 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 443.35 | 439.59 | 439.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 444.30 | 440.53 | 439.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 444.50 | 445.44 | 443.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 13:00:00 | 444.50 | 445.44 | 443.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 443.35 | 445.02 | 443.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 443.45 | 445.02 | 443.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 445.45 | 445.10 | 443.41 | EMA400 retest candle locked (from upside) |

### Cycle 167 — SELL (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 13:15:00 | 439.95 | 442.69 | 442.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 436.85 | 441.52 | 442.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 09:15:00 | 440.15 | 436.15 | 437.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 440.15 | 436.15 | 437.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 440.15 | 436.15 | 437.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:30:00 | 444.00 | 436.15 | 437.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 442.75 | 437.47 | 438.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:30:00 | 441.55 | 437.47 | 438.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 445.45 | 439.07 | 438.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 13:15:00 | 447.65 | 441.83 | 440.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 15:15:00 | 471.40 | 472.29 | 463.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 09:15:00 | 470.70 | 472.29 | 463.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 462.75 | 469.73 | 464.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 462.75 | 469.73 | 464.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 461.00 | 467.98 | 464.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:30:00 | 460.40 | 467.98 | 464.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 461.30 | 465.72 | 464.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:15:00 | 455.50 | 465.72 | 464.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — SELL (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 10:15:00 | 457.45 | 462.62 | 462.86 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 466.30 | 462.56 | 462.20 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 10:15:00 | 457.60 | 461.67 | 462.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 12:15:00 | 456.00 | 460.15 | 461.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 15:15:00 | 464.25 | 460.78 | 461.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 15:15:00 | 464.25 | 460.78 | 461.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 464.25 | 460.78 | 461.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 467.70 | 460.78 | 461.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 465.30 | 461.68 | 461.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 469.45 | 464.02 | 462.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 10:15:00 | 471.45 | 471.51 | 468.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:30:00 | 471.10 | 471.51 | 468.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 470.00 | 470.79 | 469.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 470.15 | 470.79 | 469.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 468.60 | 470.35 | 469.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:15:00 | 467.45 | 470.35 | 469.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 466.00 | 469.48 | 468.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 466.00 | 469.48 | 468.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 465.60 | 468.24 | 468.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 464.45 | 466.60 | 467.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 465.45 | 464.40 | 465.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 465.45 | 464.40 | 465.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 465.45 | 464.40 | 465.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:45:00 | 461.30 | 463.67 | 465.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 468.15 | 464.00 | 465.00 | SL hit (close>static) qty=1.00 sl=467.05 alert=retest2 |

### Cycle 174 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 478.40 | 466.88 | 466.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 484.20 | 475.31 | 471.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 13:15:00 | 480.40 | 481.01 | 476.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 480.80 | 480.80 | 477.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 480.80 | 480.80 | 477.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 490.55 | 481.10 | 479.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:15:00 | 485.45 | 484.42 | 481.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 13:15:00 | 480.45 | 481.08 | 481.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 13:15:00 | 480.45 | 481.08 | 481.16 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 488.45 | 481.63 | 481.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 15:15:00 | 491.20 | 483.54 | 482.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 15:15:00 | 485.40 | 486.79 | 484.87 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 09:15:00 | 489.20 | 486.79 | 484.87 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:00:00 | 489.30 | 487.15 | 485.36 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 485.40 | 486.66 | 485.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-01 12:15:00 | 485.40 | 486.66 | 485.44 | SL hit (close<ema400) qty=1.00 sl=485.44 alert=retest1 |

### Cycle 177 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 481.75 | 488.78 | 488.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 481.40 | 487.31 | 488.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 484.30 | 483.21 | 485.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 10:00:00 | 484.30 | 483.21 | 485.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 485.00 | 483.70 | 485.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 485.00 | 483.70 | 485.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 483.55 | 483.67 | 485.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 13:15:00 | 482.30 | 483.67 | 485.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:00:00 | 483.00 | 482.62 | 484.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 458.19 | 468.27 | 473.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 458.85 | 468.27 | 473.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 14:15:00 | 471.85 | 468.59 | 472.03 | SL hit (close>ema200) qty=0.50 sl=468.59 alert=retest2 |

### Cycle 178 — BUY (started 2026-01-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 13:15:00 | 475.25 | 473.43 | 473.36 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 12:15:00 | 473.25 | 473.46 | 473.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 14:15:00 | 471.55 | 472.95 | 473.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 474.10 | 472.77 | 473.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 474.10 | 472.77 | 473.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 474.10 | 472.77 | 473.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:15:00 | 480.40 | 472.77 | 473.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 475.55 | 473.33 | 473.29 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 11:15:00 | 472.30 | 473.12 | 473.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 12:15:00 | 471.25 | 472.75 | 473.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 14:15:00 | 451.85 | 448.75 | 454.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 15:00:00 | 451.85 | 448.75 | 454.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 457.90 | 450.58 | 455.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 467.70 | 450.58 | 455.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 461.20 | 452.70 | 455.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 459.45 | 452.70 | 455.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:00:00 | 459.40 | 456.08 | 456.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 10:15:00 | 457.70 | 456.40 | 456.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 10:15:00 | 457.70 | 456.40 | 456.24 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 452.95 | 455.61 | 455.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 450.40 | 454.57 | 455.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 449.60 | 447.12 | 450.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 15:00:00 | 449.60 | 447.12 | 450.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 451.30 | 448.42 | 450.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:00:00 | 451.30 | 448.42 | 450.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 451.10 | 448.95 | 450.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:45:00 | 453.60 | 448.95 | 450.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 450.70 | 449.30 | 450.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 12:30:00 | 449.50 | 449.38 | 450.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:45:00 | 449.15 | 448.78 | 449.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 12:00:00 | 447.20 | 448.41 | 449.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 13:45:00 | 449.45 | 448.68 | 449.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 14:15:00 | 455.65 | 450.08 | 449.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 14:15:00 | 455.65 | 450.08 | 449.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 15:15:00 | 470.50 | 454.16 | 451.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 459.00 | 459.20 | 456.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 10:00:00 | 459.00 | 459.20 | 456.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 456.60 | 458.85 | 456.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 456.60 | 458.85 | 456.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 453.15 | 457.71 | 456.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 453.20 | 457.71 | 456.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 452.40 | 456.65 | 456.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 452.40 | 456.65 | 456.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 447.75 | 454.87 | 455.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 446.00 | 453.09 | 454.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 11:15:00 | 454.45 | 451.70 | 453.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 11:15:00 | 454.45 | 451.70 | 453.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 454.45 | 451.70 | 453.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:00:00 | 454.45 | 451.70 | 453.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 12:15:00 | 469.50 | 455.26 | 454.78 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 455.95 | 458.98 | 459.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 11:15:00 | 453.15 | 457.81 | 458.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 12:15:00 | 461.20 | 458.49 | 459.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 12:15:00 | 461.20 | 458.49 | 459.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 461.20 | 458.49 | 459.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:30:00 | 460.75 | 458.49 | 459.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 457.15 | 458.22 | 458.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 13:30:00 | 459.40 | 458.22 | 458.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 454.50 | 457.07 | 458.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 10:15:00 | 452.70 | 457.07 | 458.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 12:30:00 | 453.50 | 455.86 | 457.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 461.05 | 456.48 | 456.96 | SL hit (close>static) qty=1.00 sl=458.75 alert=retest2 |

### Cycle 188 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 460.70 | 457.90 | 457.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 461.60 | 458.64 | 457.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 15:15:00 | 453.20 | 458.04 | 457.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 15:15:00 | 453.20 | 458.04 | 457.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 453.20 | 458.04 | 457.80 | EMA400 retest candle locked (from upside) |

### Cycle 189 — SELL (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 09:15:00 | 455.00 | 457.43 | 457.55 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2026-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 14:15:00 | 461.50 | 457.81 | 457.57 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 14:15:00 | 455.45 | 457.43 | 457.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 452.75 | 456.18 | 456.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 10:15:00 | 452.35 | 452.27 | 454.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 10:45:00 | 452.15 | 452.27 | 454.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 454.60 | 452.94 | 454.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:30:00 | 454.90 | 452.94 | 454.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 453.30 | 453.01 | 453.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 13:30:00 | 456.90 | 453.01 | 453.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 450.15 | 452.44 | 453.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 449.05 | 452.05 | 453.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 10:15:00 | 459.50 | 453.48 | 453.76 | SL hit (close>static) qty=1.00 sl=453.65 alert=retest2 |

### Cycle 192 — BUY (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 12:15:00 | 456.05 | 454.06 | 453.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 14:15:00 | 473.00 | 457.96 | 455.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 12:15:00 | 467.00 | 468.79 | 465.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 12:15:00 | 467.00 | 468.79 | 465.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 467.00 | 468.79 | 465.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:00:00 | 467.00 | 468.79 | 465.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 467.25 | 469.89 | 466.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 09:30:00 | 480.15 | 474.36 | 469.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 478.20 | 476.79 | 473.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 11:45:00 | 477.75 | 475.95 | 473.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 15:15:00 | 483.30 | 486.45 | 486.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — SELL (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 15:15:00 | 483.30 | 486.45 | 486.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 09:15:00 | 480.50 | 485.26 | 485.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 15:15:00 | 485.00 | 482.44 | 483.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 15:15:00 | 485.00 | 482.44 | 483.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 485.00 | 482.44 | 483.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 472.10 | 483.26 | 483.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 448.50 | 459.99 | 469.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 448.50 | 446.72 | 456.48 | SL hit (close>ema200) qty=0.50 sl=446.72 alert=retest2 |

### Cycle 194 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 451.45 | 445.25 | 444.98 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 435.65 | 443.79 | 444.59 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 15:15:00 | 449.50 | 445.27 | 444.71 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 437.85 | 443.79 | 444.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 436.55 | 442.34 | 443.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 14:15:00 | 439.65 | 439.33 | 441.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-13 14:45:00 | 440.55 | 439.33 | 441.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 437.00 | 438.87 | 440.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 09:15:00 | 427.20 | 438.87 | 440.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 09:15:00 | 429.80 | 434.90 | 436.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 446.30 | 439.33 | 438.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 446.30 | 439.33 | 438.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 450.80 | 443.61 | 440.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 438.05 | 448.35 | 445.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 438.05 | 448.35 | 445.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 438.05 | 448.35 | 445.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:45:00 | 446.70 | 445.83 | 444.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 432.50 | 448.07 | 448.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 432.50 | 448.07 | 448.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 430.30 | 440.00 | 444.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 13:15:00 | 440.50 | 440.10 | 443.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 14:00:00 | 440.50 | 440.10 | 443.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 432.00 | 437.19 | 441.40 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 450.20 | 442.77 | 442.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 461.15 | 446.49 | 444.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 457.50 | 458.16 | 452.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 10:15:00 | 454.20 | 457.37 | 452.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 454.20 | 457.37 | 452.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 454.40 | 457.37 | 452.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 452.50 | 456.39 | 452.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 452.50 | 456.39 | 452.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 451.35 | 455.38 | 452.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:45:00 | 451.15 | 455.38 | 452.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 445.30 | 452.55 | 451.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 445.30 | 452.55 | 451.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 442.00 | 450.44 | 450.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 430.90 | 446.53 | 449.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 14:15:00 | 441.25 | 439.84 | 444.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 15:00:00 | 441.25 | 439.84 | 444.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 443.40 | 440.55 | 444.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 462.35 | 440.55 | 444.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 458.60 | 444.16 | 445.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:15:00 | 454.95 | 444.16 | 445.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 457.60 | 446.85 | 446.52 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 432.05 | 445.45 | 446.56 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 451.95 | 445.01 | 444.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 471.40 | 453.80 | 449.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 15:15:00 | 463.50 | 465.12 | 458.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:15:00 | 463.75 | 465.12 | 458.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 458.00 | 463.70 | 458.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 458.00 | 463.70 | 458.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 459.75 | 462.91 | 458.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 14:00:00 | 461.20 | 461.28 | 458.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 14:30:00 | 465.60 | 461.82 | 459.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-21 10:15:00 | 507.32 | 495.66 | 490.89 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-16 11:15:00 | 118.90 | 2023-05-19 11:15:00 | 120.75 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2023-05-17 11:45:00 | 118.65 | 2023-05-19 11:15:00 | 120.75 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2023-05-17 12:45:00 | 118.80 | 2023-05-19 11:15:00 | 120.75 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2023-05-17 14:00:00 | 118.80 | 2023-05-19 11:15:00 | 120.75 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2023-05-18 10:15:00 | 119.35 | 2023-05-19 13:15:00 | 119.95 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2023-05-19 13:15:00 | 119.40 | 2023-05-19 13:15:00 | 119.95 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2023-05-29 12:00:00 | 124.50 | 2023-05-29 12:15:00 | 124.30 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2023-06-07 15:00:00 | 131.95 | 2023-06-09 10:15:00 | 128.70 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2023-06-08 13:30:00 | 133.25 | 2023-06-09 10:15:00 | 128.70 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2023-06-20 10:00:00 | 135.30 | 2023-06-22 14:15:00 | 133.55 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2023-06-28 09:15:00 | 134.50 | 2023-06-30 14:15:00 | 132.85 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2023-06-30 12:45:00 | 133.10 | 2023-06-30 14:15:00 | 132.85 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2023-07-05 11:15:00 | 129.85 | 2023-07-06 09:15:00 | 131.45 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2023-07-06 10:30:00 | 129.85 | 2023-07-11 09:15:00 | 132.90 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2023-07-06 11:00:00 | 129.50 | 2023-07-11 09:15:00 | 132.90 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2023-07-07 09:30:00 | 129.75 | 2023-07-11 09:15:00 | 132.90 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2023-07-10 11:30:00 | 126.50 | 2023-07-11 09:15:00 | 132.90 | STOP_HIT | 1.00 | -5.06% |
| SELL | retest2 | 2023-07-10 12:00:00 | 126.20 | 2023-07-11 09:15:00 | 132.90 | STOP_HIT | 1.00 | -5.31% |
| BUY | retest2 | 2023-07-20 09:15:00 | 142.60 | 2023-07-20 11:15:00 | 141.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2023-07-20 10:00:00 | 142.30 | 2023-07-20 11:15:00 | 141.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2023-07-21 12:30:00 | 140.35 | 2023-07-31 13:15:00 | 139.30 | STOP_HIT | 1.00 | 0.75% |
| SELL | retest2 | 2023-07-21 13:30:00 | 140.70 | 2023-07-31 13:15:00 | 139.30 | STOP_HIT | 1.00 | 1.00% |
| SELL | retest2 | 2023-07-24 11:45:00 | 140.75 | 2023-07-31 13:15:00 | 139.30 | STOP_HIT | 1.00 | 1.03% |
| SELL | retest2 | 2023-07-24 12:45:00 | 140.70 | 2023-07-31 13:15:00 | 139.30 | STOP_HIT | 1.00 | 1.00% |
| SELL | retest2 | 2023-07-25 10:30:00 | 138.60 | 2023-07-31 13:15:00 | 139.30 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2023-07-26 11:30:00 | 139.20 | 2023-07-31 13:15:00 | 139.30 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2023-07-26 13:00:00 | 138.80 | 2023-07-31 13:15:00 | 139.30 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2023-08-02 09:30:00 | 139.70 | 2023-08-02 10:15:00 | 138.20 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2023-08-18 13:45:00 | 167.55 | 2023-08-21 15:15:00 | 184.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-21 09:15:00 | 173.15 | 2023-08-23 09:15:00 | 190.47 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-09-14 12:30:00 | 241.75 | 2023-09-25 12:15:00 | 241.00 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2023-09-15 09:15:00 | 238.50 | 2023-09-25 12:15:00 | 241.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2023-09-18 09:15:00 | 241.60 | 2023-09-25 12:15:00 | 241.00 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2023-09-18 10:00:00 | 241.25 | 2023-09-25 12:15:00 | 241.00 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2023-09-20 09:30:00 | 239.10 | 2023-09-25 12:15:00 | 241.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2023-09-21 09:45:00 | 239.50 | 2023-09-25 12:15:00 | 241.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2023-09-21 10:15:00 | 239.30 | 2023-09-25 12:15:00 | 241.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2023-09-21 10:45:00 | 239.00 | 2023-09-25 12:15:00 | 241.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2023-09-22 09:15:00 | 236.95 | 2023-09-25 12:15:00 | 241.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2023-09-25 10:00:00 | 235.90 | 2023-09-25 12:15:00 | 241.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2023-09-28 14:15:00 | 251.35 | 2023-09-28 14:15:00 | 246.30 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2023-10-20 14:45:00 | 260.05 | 2023-10-23 09:15:00 | 248.05 | STOP_HIT | 1.00 | -4.61% |
| BUY | retest2 | 2023-11-03 09:15:00 | 243.50 | 2023-11-10 11:15:00 | 249.50 | STOP_HIT | 1.00 | 2.46% |
| BUY | retest2 | 2023-11-16 12:45:00 | 259.45 | 2023-11-17 11:15:00 | 254.00 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2023-12-11 09:15:00 | 299.75 | 2023-12-11 09:15:00 | 295.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2023-12-14 10:15:00 | 292.65 | 2023-12-19 10:15:00 | 292.45 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2023-12-14 11:30:00 | 293.50 | 2023-12-19 10:15:00 | 292.45 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest2 | 2023-12-19 09:30:00 | 294.10 | 2023-12-19 10:15:00 | 292.45 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2024-01-11 09:15:00 | 390.15 | 2024-01-12 10:15:00 | 373.40 | STOP_HIT | 1.00 | -4.29% |
| BUY | retest2 | 2024-01-11 11:45:00 | 384.00 | 2024-01-12 10:15:00 | 373.40 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest1 | 2024-03-07 09:15:00 | 340.00 | 2024-03-11 09:15:00 | 323.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-03-07 09:15:00 | 340.00 | 2024-03-12 09:15:00 | 306.00 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-03-21 09:15:00 | 323.00 | 2024-03-22 11:15:00 | 316.75 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-03-21 10:15:00 | 322.10 | 2024-03-22 11:15:00 | 316.75 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-03-21 11:00:00 | 322.95 | 2024-03-22 11:15:00 | 316.75 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-03-28 14:45:00 | 303.00 | 2024-04-01 14:15:00 | 315.20 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2024-05-09 09:15:00 | 355.85 | 2024-05-13 09:15:00 | 338.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-09 09:15:00 | 355.85 | 2024-05-14 09:15:00 | 341.85 | STOP_HIT | 0.50 | 3.93% |
| BUY | retest2 | 2024-05-16 12:15:00 | 350.80 | 2024-05-16 15:15:00 | 349.80 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-05-16 13:45:00 | 350.15 | 2024-05-16 15:15:00 | 349.80 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-05-16 14:15:00 | 349.60 | 2024-05-16 15:15:00 | 349.80 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2024-05-16 15:15:00 | 349.80 | 2024-05-16 15:15:00 | 349.80 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-05-29 13:15:00 | 344.55 | 2024-06-03 14:15:00 | 340.00 | STOP_HIT | 1.00 | 1.32% |
| SELL | retest2 | 2024-05-29 14:00:00 | 344.25 | 2024-06-03 14:15:00 | 340.00 | STOP_HIT | 1.00 | 1.23% |
| BUY | retest2 | 2024-06-13 12:15:00 | 361.55 | 2024-06-20 09:15:00 | 397.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-09 09:15:00 | 414.55 | 2024-07-12 15:15:00 | 422.05 | STOP_HIT | 1.00 | 1.81% |
| SELL | retest2 | 2024-07-23 12:15:00 | 391.00 | 2024-07-24 09:15:00 | 413.00 | STOP_HIT | 1.00 | -5.63% |
| BUY | retest2 | 2024-07-30 09:15:00 | 429.05 | 2024-08-06 09:15:00 | 471.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-23 13:15:00 | 524.35 | 2024-08-30 10:15:00 | 518.65 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-08-23 13:45:00 | 525.50 | 2024-08-30 10:15:00 | 518.65 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-08-26 11:00:00 | 520.95 | 2024-08-30 10:15:00 | 518.65 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2024-08-29 12:30:00 | 522.60 | 2024-08-30 10:15:00 | 518.65 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-09-12 14:15:00 | 559.95 | 2024-09-17 10:15:00 | 615.95 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-04 12:30:00 | 639.25 | 2024-10-07 09:15:00 | 607.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 12:30:00 | 639.25 | 2024-10-08 10:15:00 | 606.60 | STOP_HIT | 0.50 | 5.11% |
| BUY | retest2 | 2024-10-11 13:15:00 | 640.90 | 2024-10-17 10:15:00 | 628.80 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-10-14 13:45:00 | 634.30 | 2024-10-17 10:15:00 | 628.80 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-10-15 10:30:00 | 635.30 | 2024-10-17 12:15:00 | 623.05 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-10-15 11:15:00 | 638.45 | 2024-10-17 12:15:00 | 623.05 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2024-10-15 12:30:00 | 643.45 | 2024-10-17 12:15:00 | 623.05 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2024-10-15 14:00:00 | 644.55 | 2024-10-17 12:15:00 | 623.05 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2024-10-18 15:00:00 | 620.60 | 2024-10-22 10:15:00 | 589.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 10:45:00 | 619.50 | 2024-10-22 10:15:00 | 588.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 15:00:00 | 620.60 | 2024-10-23 09:15:00 | 558.54 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 10:45:00 | 619.50 | 2024-10-23 09:15:00 | 557.55 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-19 14:15:00 | 499.25 | 2024-11-26 13:15:00 | 496.85 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2024-11-19 14:45:00 | 497.35 | 2024-11-26 13:15:00 | 496.85 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2024-11-25 09:30:00 | 497.00 | 2024-11-26 13:15:00 | 496.85 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2024-11-29 13:45:00 | 523.00 | 2024-12-05 13:15:00 | 575.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-13 09:15:00 | 558.70 | 2024-12-19 09:15:00 | 530.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 12:15:00 | 561.40 | 2024-12-19 09:15:00 | 533.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 14:30:00 | 562.10 | 2024-12-19 09:15:00 | 534.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 15:00:00 | 561.85 | 2024-12-19 09:15:00 | 533.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 11:00:00 | 557.15 | 2024-12-19 09:15:00 | 531.05 | PARTIAL | 0.50 | 4.68% |
| SELL | retest2 | 2024-12-16 14:00:00 | 559.00 | 2024-12-19 09:15:00 | 530.95 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2024-12-13 09:15:00 | 558.70 | 2024-12-20 10:15:00 | 544.45 | STOP_HIT | 0.50 | 2.55% |
| SELL | retest2 | 2024-12-13 12:15:00 | 561.40 | 2024-12-20 10:15:00 | 544.45 | STOP_HIT | 0.50 | 3.02% |
| SELL | retest2 | 2024-12-13 14:30:00 | 562.10 | 2024-12-20 10:15:00 | 544.45 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2024-12-13 15:00:00 | 561.85 | 2024-12-20 10:15:00 | 544.45 | STOP_HIT | 0.50 | 3.10% |
| SELL | retest2 | 2024-12-16 11:00:00 | 557.15 | 2024-12-20 10:15:00 | 544.45 | STOP_HIT | 0.50 | 2.28% |
| SELL | retest2 | 2024-12-16 14:00:00 | 559.00 | 2024-12-20 10:15:00 | 544.45 | STOP_HIT | 0.50 | 2.60% |
| SELL | retest2 | 2024-12-17 11:30:00 | 558.90 | 2024-12-23 15:15:00 | 552.65 | STOP_HIT | 1.00 | 1.12% |
| BUY | retest2 | 2025-01-02 09:45:00 | 587.00 | 2025-01-06 09:15:00 | 577.80 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-01-06 09:30:00 | 584.45 | 2025-01-06 10:15:00 | 570.10 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-01-15 13:15:00 | 546.35 | 2025-01-22 09:15:00 | 519.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 11:00:00 | 548.25 | 2025-01-22 09:15:00 | 520.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 11:45:00 | 547.70 | 2025-01-22 09:15:00 | 520.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-15 13:15:00 | 546.35 | 2025-01-22 14:15:00 | 532.20 | STOP_HIT | 0.50 | 2.59% |
| SELL | retest2 | 2025-01-16 11:00:00 | 548.25 | 2025-01-22 14:15:00 | 532.20 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2025-01-16 11:45:00 | 547.70 | 2025-01-22 14:15:00 | 532.20 | STOP_HIT | 0.50 | 2.83% |
| BUY | retest2 | 2025-01-31 15:15:00 | 494.80 | 2025-02-01 12:15:00 | 483.90 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-02-01 14:00:00 | 494.05 | 2025-02-03 09:15:00 | 486.95 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-02-01 14:45:00 | 497.75 | 2025-02-03 09:15:00 | 486.95 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-02-07 11:15:00 | 502.20 | 2025-02-07 13:15:00 | 491.90 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-02-18 10:15:00 | 406.85 | 2025-02-19 09:15:00 | 432.30 | STOP_HIT | 1.00 | -6.26% |
| SELL | retest2 | 2025-02-18 12:45:00 | 408.35 | 2025-02-19 09:15:00 | 432.30 | STOP_HIT | 1.00 | -5.87% |
| SELL | retest2 | 2025-02-18 13:15:00 | 405.10 | 2025-02-19 09:15:00 | 432.30 | STOP_HIT | 1.00 | -6.71% |
| SELL | retest2 | 2025-03-04 13:15:00 | 404.80 | 2025-03-05 09:15:00 | 419.45 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2025-03-13 14:15:00 | 402.55 | 2025-03-18 09:15:00 | 416.65 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2025-03-17 13:00:00 | 403.00 | 2025-03-18 09:15:00 | 416.65 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2025-03-17 15:00:00 | 402.25 | 2025-03-18 09:15:00 | 416.65 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2025-03-21 09:15:00 | 433.50 | 2025-03-25 15:15:00 | 431.50 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-03-28 12:15:00 | 428.80 | 2025-04-01 12:15:00 | 435.25 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-03-28 13:00:00 | 427.80 | 2025-04-01 12:15:00 | 435.25 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-04-02 12:00:00 | 436.85 | 2025-04-04 12:15:00 | 434.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-04-04 10:30:00 | 438.20 | 2025-04-04 12:15:00 | 434.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-04-08 10:30:00 | 414.20 | 2025-04-11 10:15:00 | 433.70 | STOP_HIT | 1.00 | -4.71% |
| SELL | retest2 | 2025-04-08 14:45:00 | 414.25 | 2025-04-11 10:15:00 | 433.70 | STOP_HIT | 1.00 | -4.70% |
| SELL | retest2 | 2025-04-29 10:15:00 | 451.35 | 2025-04-29 12:15:00 | 460.15 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-05-06 10:15:00 | 429.80 | 2025-05-12 09:15:00 | 432.80 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-05-08 11:30:00 | 430.30 | 2025-05-12 12:15:00 | 433.20 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-05-08 14:00:00 | 430.25 | 2025-05-12 12:15:00 | 433.20 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-05-08 14:45:00 | 424.15 | 2025-05-12 12:15:00 | 433.20 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-05-09 09:15:00 | 415.00 | 2025-05-12 12:15:00 | 433.20 | STOP_HIT | 1.00 | -4.39% |
| BUY | retest2 | 2025-05-20 14:45:00 | 461.95 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | 3.04% |
| BUY | retest2 | 2025-05-21 10:00:00 | 463.25 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | 2.75% |
| BUY | retest2 | 2025-05-21 12:45:00 | 461.65 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | 3.11% |
| BUY | retest2 | 2025-05-21 13:45:00 | 462.20 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | 2.99% |
| BUY | retest2 | 2025-05-23 11:30:00 | 475.10 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-05-23 12:00:00 | 475.00 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest2 | 2025-05-26 09:15:00 | 479.60 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-05-26 11:00:00 | 475.10 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-05-27 14:45:00 | 480.10 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-05-28 09:15:00 | 481.20 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-05-28 10:15:00 | 479.50 | 2025-05-29 15:15:00 | 476.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-06-12 11:15:00 | 486.50 | 2025-06-13 09:15:00 | 462.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 11:15:00 | 486.50 | 2025-06-16 14:15:00 | 469.50 | STOP_HIT | 0.50 | 3.49% |
| BUY | retest2 | 2025-07-16 09:15:00 | 516.25 | 2025-07-18 13:15:00 | 512.85 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-07-16 10:30:00 | 515.75 | 2025-07-18 13:15:00 | 512.85 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-07-16 13:00:00 | 515.30 | 2025-07-18 13:15:00 | 512.85 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-07-17 09:15:00 | 521.95 | 2025-07-18 13:15:00 | 512.85 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-07-21 12:45:00 | 511.50 | 2025-07-22 14:15:00 | 513.75 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-07-21 13:30:00 | 511.40 | 2025-07-23 09:15:00 | 513.15 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-07-22 09:45:00 | 510.50 | 2025-07-23 14:15:00 | 514.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-07-22 13:00:00 | 511.50 | 2025-07-23 14:15:00 | 514.50 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-07-22 14:15:00 | 509.25 | 2025-07-23 15:15:00 | 515.40 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-07-23 09:15:00 | 510.05 | 2025-07-23 15:15:00 | 515.40 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-07-23 10:30:00 | 510.20 | 2025-07-23 15:15:00 | 515.40 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-07-23 11:15:00 | 509.75 | 2025-07-23 15:15:00 | 515.40 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-07-24 14:30:00 | 514.50 | 2025-07-25 09:15:00 | 508.75 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-07-30 11:00:00 | 482.85 | 2025-08-04 09:15:00 | 458.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 11:00:00 | 482.85 | 2025-08-04 11:15:00 | 467.55 | STOP_HIT | 0.50 | 3.17% |
| BUY | retest2 | 2025-08-18 09:15:00 | 472.00 | 2025-08-22 11:15:00 | 470.95 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-09-08 12:15:00 | 456.10 | 2025-09-10 09:15:00 | 483.50 | STOP_HIT | 1.00 | -6.01% |
| SELL | retest2 | 2025-09-08 13:15:00 | 456.30 | 2025-09-10 09:15:00 | 483.50 | STOP_HIT | 1.00 | -5.96% |
| SELL | retest2 | 2025-09-08 14:30:00 | 455.75 | 2025-09-10 09:15:00 | 483.50 | STOP_HIT | 1.00 | -6.09% |
| BUY | retest1 | 2025-10-15 09:15:00 | 469.00 | 2025-10-16 14:15:00 | 467.50 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-10-15 10:30:00 | 470.70 | 2025-10-16 14:15:00 | 467.50 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-10-20 14:15:00 | 456.50 | 2025-10-21 13:15:00 | 469.65 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2025-10-28 10:30:00 | 474.85 | 2025-11-03 11:15:00 | 480.00 | STOP_HIT | 1.00 | 1.08% |
| BUY | retest2 | 2025-10-28 11:00:00 | 474.80 | 2025-11-03 11:15:00 | 480.00 | STOP_HIT | 1.00 | 1.10% |
| BUY | retest2 | 2025-10-28 11:30:00 | 475.55 | 2025-11-03 11:15:00 | 480.00 | STOP_HIT | 1.00 | 0.94% |
| BUY | retest2 | 2025-10-28 12:45:00 | 475.00 | 2025-11-03 11:15:00 | 480.00 | STOP_HIT | 1.00 | 1.05% |
| SELL | retest2 | 2025-11-17 13:15:00 | 455.50 | 2025-11-24 10:15:00 | 432.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 09:15:00 | 455.25 | 2025-11-24 10:15:00 | 432.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 13:15:00 | 455.50 | 2025-11-25 09:15:00 | 445.60 | STOP_HIT | 0.50 | 2.17% |
| SELL | retest2 | 2025-11-18 09:15:00 | 455.25 | 2025-11-25 09:15:00 | 445.60 | STOP_HIT | 0.50 | 2.12% |
| SELL | retest2 | 2025-12-19 11:45:00 | 461.30 | 2025-12-19 13:15:00 | 468.15 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-12-26 09:15:00 | 490.55 | 2025-12-29 13:15:00 | 480.45 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-12-26 13:15:00 | 485.45 | 2025-12-29 13:15:00 | 480.45 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest1 | 2026-01-01 09:15:00 | 489.20 | 2026-01-01 12:15:00 | 485.40 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest1 | 2026-01-01 11:00:00 | 489.30 | 2026-01-01 12:15:00 | 485.40 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-01-02 09:15:00 | 489.55 | 2026-01-06 09:15:00 | 481.75 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-01-07 13:15:00 | 482.30 | 2026-01-12 11:15:00 | 458.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:00:00 | 483.00 | 2026-01-12 11:15:00 | 458.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 13:15:00 | 482.30 | 2026-01-12 14:15:00 | 471.85 | STOP_HIT | 0.50 | 2.17% |
| SELL | retest2 | 2026-01-08 10:00:00 | 483.00 | 2026-01-12 14:15:00 | 471.85 | STOP_HIT | 0.50 | 2.31% |
| SELL | retest2 | 2026-01-22 10:15:00 | 459.45 | 2026-01-23 10:15:00 | 457.70 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2026-01-23 10:00:00 | 459.40 | 2026-01-23 10:15:00 | 457.70 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2026-01-28 12:30:00 | 449.50 | 2026-01-29 14:15:00 | 455.65 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-01-29 09:45:00 | 449.15 | 2026-01-29 14:15:00 | 455.65 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-01-29 12:00:00 | 447.20 | 2026-01-29 14:15:00 | 455.65 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-01-29 13:45:00 | 449.45 | 2026-01-29 14:15:00 | 455.65 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-02-06 10:15:00 | 452.70 | 2026-02-09 10:15:00 | 461.05 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-02-06 12:30:00 | 453.50 | 2026-02-09 10:15:00 | 461.05 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-02-16 09:15:00 | 449.05 | 2026-02-16 10:15:00 | 459.50 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-02-19 09:30:00 | 480.15 | 2026-02-25 15:15:00 | 483.30 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2026-02-20 09:15:00 | 478.20 | 2026-02-25 15:15:00 | 483.30 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2026-02-20 11:45:00 | 477.75 | 2026-02-25 15:15:00 | 483.30 | STOP_HIT | 1.00 | 1.16% |
| SELL | retest2 | 2026-03-02 09:15:00 | 472.10 | 2026-03-04 09:15:00 | 448.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 472.10 | 2026-03-05 09:15:00 | 448.50 | STOP_HIT | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-16 09:15:00 | 427.20 | 2026-03-17 13:15:00 | 446.30 | STOP_HIT | 1.00 | -4.47% |
| SELL | retest2 | 2026-03-17 09:15:00 | 429.80 | 2026-03-17 13:15:00 | 446.30 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2026-03-19 12:45:00 | 446.70 | 2026-03-23 09:15:00 | 432.50 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2026-04-09 14:00:00 | 461.20 | 2026-04-21 10:15:00 | 507.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-09 14:30:00 | 465.60 | 2026-04-21 10:15:00 | 512.16 | TARGET_HIT | 1.00 | 10.00% |
