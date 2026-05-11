# Bharat Electronics Ltd. (BEL)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 439.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 201 |
| ALERT1 | 145 |
| ALERT2 | 143 |
| ALERT2_SKIP | 66 |
| ALERT3 | 393 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 157 |
| PARTIAL | 11 |
| TARGET_HIT | 14 |
| STOP_HIT | 149 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 174 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 59 / 115
- **Target hits / Stop hits / Partials:** 14 / 149 / 11
- **Avg / median % per leg:** 0.52% / -0.82%
- **Sum % (uncompounded):** 89.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 77 | 24 | 31.2% | 14 | 63 | 0 | 1.29% | 99.4% |
| BUY @ 2nd Alert (retest1) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.49% | -9.0% |
| BUY @ 3rd Alert (retest2) | 71 | 24 | 33.8% | 14 | 57 | 0 | 1.53% | 108.3% |
| SELL (all) | 97 | 35 | 36.1% | 0 | 86 | 11 | -0.10% | -9.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.70% | -2.7% |
| SELL @ 3rd Alert (retest2) | 96 | 35 | 36.5% | 0 | 85 | 11 | -0.07% | -6.9% |
| retest1 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.67% | -11.7% |
| retest2 (combined) | 167 | 59 | 35.3% | 14 | 142 | 11 | 0.61% | 101.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 12:15:00 | 108.30 | 107.75 | 107.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 09:15:00 | 109.35 | 108.42 | 108.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-17 13:15:00 | 108.35 | 108.55 | 108.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-17 14:00:00 | 108.35 | 108.55 | 108.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 14:15:00 | 108.30 | 108.50 | 108.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-17 15:00:00 | 108.30 | 108.50 | 108.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 15:15:00 | 108.20 | 108.44 | 108.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-18 09:15:00 | 108.65 | 108.44 | 108.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-18 10:15:00 | 108.40 | 108.42 | 108.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-18 14:15:00 | 107.80 | 108.25 | 108.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-05-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 14:15:00 | 107.80 | 108.25 | 108.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 15:15:00 | 107.60 | 108.12 | 108.21 | Break + close below crossover candle low |

### Cycle 3 — BUY (started 2023-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 09:15:00 | 111.30 | 108.05 | 107.93 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 15:15:00 | 107.95 | 108.67 | 108.73 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 13:15:00 | 109.10 | 108.79 | 108.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 09:15:00 | 110.30 | 109.10 | 108.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 11:15:00 | 111.40 | 111.41 | 110.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-30 12:00:00 | 111.40 | 111.41 | 110.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 15:15:00 | 111.15 | 111.38 | 111.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 09:15:00 | 110.55 | 111.38 | 111.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 110.85 | 111.27 | 110.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 11:30:00 | 111.65 | 111.28 | 111.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-06-12 09:15:00 | 122.82 | 120.02 | 118.71 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 15:15:00 | 122.90 | 124.34 | 124.47 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-06-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-22 09:15:00 | 125.70 | 124.62 | 124.58 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 11:15:00 | 123.45 | 124.54 | 124.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 122.35 | 123.79 | 124.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 13:15:00 | 121.30 | 121.22 | 122.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 14:00:00 | 121.30 | 121.22 | 122.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 119.80 | 121.08 | 121.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 10:45:00 | 119.30 | 120.64 | 121.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-28 10:00:00 | 119.35 | 119.29 | 120.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-30 10:15:00 | 122.50 | 120.64 | 120.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2023-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 10:15:00 | 122.50 | 120.64 | 120.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 11:15:00 | 123.20 | 121.15 | 120.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 12:15:00 | 123.20 | 123.49 | 122.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-03 13:00:00 | 123.20 | 123.49 | 122.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 123.20 | 123.53 | 122.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 10:00:00 | 123.20 | 123.53 | 122.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 10:15:00 | 122.95 | 123.41 | 122.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 11:00:00 | 122.95 | 123.41 | 122.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 11:15:00 | 122.75 | 123.28 | 122.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 11:45:00 | 122.70 | 123.28 | 122.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 12:15:00 | 122.30 | 123.08 | 122.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 13:00:00 | 122.30 | 123.08 | 122.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 13:15:00 | 122.40 | 122.95 | 122.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 14:00:00 | 122.40 | 122.95 | 122.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 14:15:00 | 122.20 | 122.80 | 122.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 14:45:00 | 121.70 | 122.80 | 122.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 10:15:00 | 122.50 | 122.71 | 122.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 11:00:00 | 122.50 | 122.71 | 122.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 11:15:00 | 122.50 | 122.67 | 122.66 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-05 12:15:00 | 121.75 | 122.49 | 122.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-05 13:15:00 | 121.65 | 122.32 | 122.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-06 09:15:00 | 123.35 | 122.43 | 122.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-06 09:15:00 | 123.35 | 122.43 | 122.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 09:15:00 | 123.35 | 122.43 | 122.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-06 09:45:00 | 123.25 | 122.43 | 122.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2023-07-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 10:15:00 | 123.95 | 122.73 | 122.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 14:15:00 | 124.85 | 123.64 | 123.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 12:15:00 | 124.20 | 124.45 | 123.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-07 13:00:00 | 124.20 | 124.45 | 123.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 15:15:00 | 124.00 | 124.26 | 123.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:15:00 | 123.00 | 124.26 | 123.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 123.60 | 124.13 | 123.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:30:00 | 123.40 | 124.13 | 123.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 123.60 | 124.02 | 123.82 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 12:15:00 | 122.75 | 123.61 | 123.66 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 09:15:00 | 125.60 | 123.79 | 123.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 12:15:00 | 128.35 | 125.61 | 124.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 11:15:00 | 127.25 | 127.48 | 126.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-13 11:30:00 | 127.25 | 127.48 | 126.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 12:15:00 | 127.05 | 127.40 | 126.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 13:00:00 | 127.05 | 127.40 | 126.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 13:15:00 | 125.35 | 126.99 | 126.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 14:00:00 | 125.35 | 126.99 | 126.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 14:15:00 | 125.60 | 126.71 | 126.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 14:45:00 | 125.40 | 126.71 | 126.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2023-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 09:15:00 | 125.20 | 126.26 | 126.35 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 15:15:00 | 127.10 | 126.37 | 126.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 12:15:00 | 127.45 | 126.76 | 126.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 15:15:00 | 126.95 | 126.95 | 126.69 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-18 09:15:00 | 127.55 | 126.95 | 126.69 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 11:15:00 | 125.35 | 126.77 | 126.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-18 11:15:00 | 125.35 | 126.77 | 126.69 | SL hit (close<ema400) qty=1.00 sl=126.69 alert=retest1 |

### Cycle 16 — SELL (started 2023-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 13:15:00 | 126.50 | 126.61 | 126.62 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 14:15:00 | 127.15 | 126.72 | 126.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-18 15:15:00 | 127.35 | 126.85 | 126.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-19 12:15:00 | 126.95 | 127.01 | 126.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-19 12:15:00 | 126.95 | 127.01 | 126.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 12:15:00 | 126.95 | 127.01 | 126.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-19 12:45:00 | 126.85 | 127.01 | 126.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 13:15:00 | 127.15 | 127.04 | 126.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-19 13:30:00 | 126.85 | 127.04 | 126.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 15:15:00 | 127.00 | 127.02 | 126.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 09:15:00 | 126.75 | 127.02 | 126.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 126.30 | 126.87 | 126.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 10:00:00 | 126.30 | 126.87 | 126.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2023-07-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 10:15:00 | 126.05 | 126.71 | 126.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 09:15:00 | 125.55 | 126.13 | 126.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 09:15:00 | 127.25 | 125.73 | 125.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 09:15:00 | 127.25 | 125.73 | 125.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 127.25 | 125.73 | 125.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 10:00:00 | 127.25 | 125.73 | 125.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 10:15:00 | 126.50 | 125.88 | 126.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-24 11:30:00 | 126.05 | 125.91 | 126.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-24 12:30:00 | 126.10 | 125.97 | 126.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-24 13:30:00 | 125.95 | 125.99 | 126.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-24 15:00:00 | 126.00 | 125.99 | 126.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 15:15:00 | 126.20 | 126.03 | 126.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 09:15:00 | 126.35 | 126.03 | 126.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 126.10 | 126.05 | 126.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 09:30:00 | 126.45 | 126.05 | 126.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 125.90 | 126.02 | 126.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 10:45:00 | 126.15 | 126.02 | 126.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-07-25 11:15:00 | 126.30 | 126.07 | 126.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2023-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 11:15:00 | 126.30 | 126.07 | 126.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 12:15:00 | 126.70 | 126.20 | 126.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 09:15:00 | 126.60 | 126.61 | 126.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 09:15:00 | 126.60 | 126.61 | 126.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 126.60 | 126.61 | 126.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-26 09:30:00 | 126.25 | 126.61 | 126.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 10:15:00 | 126.15 | 126.52 | 126.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-26 10:45:00 | 126.35 | 126.52 | 126.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 11:15:00 | 126.10 | 126.44 | 126.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-26 11:30:00 | 125.90 | 126.44 | 126.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2023-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 14:15:00 | 125.90 | 126.20 | 126.24 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-07-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 09:15:00 | 127.20 | 126.39 | 126.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 13:15:00 | 128.40 | 127.02 | 126.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 12:15:00 | 130.75 | 130.90 | 129.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-01 12:30:00 | 130.65 | 130.90 | 129.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 128.95 | 130.38 | 130.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 09:45:00 | 129.25 | 130.38 | 130.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 127.50 | 129.80 | 129.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:00:00 | 127.50 | 129.80 | 129.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2023-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 11:15:00 | 127.10 | 129.26 | 129.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 12:15:00 | 126.25 | 128.66 | 129.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 09:15:00 | 125.85 | 125.38 | 126.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 09:15:00 | 125.85 | 125.38 | 126.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 125.85 | 125.38 | 126.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 09:45:00 | 126.30 | 125.38 | 126.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 11:15:00 | 127.25 | 125.84 | 126.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 12:00:00 | 127.25 | 125.84 | 126.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 12:15:00 | 127.30 | 126.13 | 126.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 13:00:00 | 127.30 | 126.13 | 126.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2023-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 10:15:00 | 127.35 | 126.93 | 126.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 12:15:00 | 128.00 | 127.16 | 127.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 13:15:00 | 130.20 | 131.28 | 130.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 13:15:00 | 130.20 | 131.28 | 130.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 13:15:00 | 130.20 | 131.28 | 130.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 14:00:00 | 130.20 | 131.28 | 130.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 14:15:00 | 129.50 | 130.92 | 130.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 15:00:00 | 129.50 | 130.92 | 130.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 15:15:00 | 130.00 | 130.74 | 130.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-11 09:15:00 | 130.20 | 130.74 | 130.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-14 14:15:00 | 130.40 | 130.88 | 130.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2023-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 14:15:00 | 130.40 | 130.88 | 130.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-16 10:15:00 | 130.15 | 130.65 | 130.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 12:15:00 | 131.65 | 130.78 | 130.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 12:15:00 | 131.65 | 130.78 | 130.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 12:15:00 | 131.65 | 130.78 | 130.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 13:00:00 | 131.65 | 130.78 | 130.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2023-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 13:15:00 | 131.40 | 130.90 | 130.89 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-08-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 10:15:00 | 130.10 | 130.75 | 130.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-17 11:15:00 | 129.50 | 130.50 | 130.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 11:15:00 | 128.55 | 128.03 | 128.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-21 12:00:00 | 128.55 | 128.03 | 128.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 12:15:00 | 128.55 | 128.14 | 128.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 12:30:00 | 128.65 | 128.14 | 128.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 13:15:00 | 128.60 | 128.23 | 128.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 14:00:00 | 128.60 | 128.23 | 128.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 14:15:00 | 129.10 | 128.40 | 128.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 15:00:00 | 129.10 | 128.40 | 128.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 15:15:00 | 129.65 | 128.65 | 128.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 09:15:00 | 130.60 | 128.65 | 128.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2023-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 09:15:00 | 131.20 | 129.16 | 129.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 10:15:00 | 133.25 | 129.98 | 129.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 14:15:00 | 133.35 | 133.55 | 132.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-23 15:00:00 | 133.35 | 133.55 | 132.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 12:15:00 | 133.80 | 134.33 | 133.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-25 13:30:00 | 134.45 | 134.24 | 133.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-25 14:15:00 | 133.35 | 134.06 | 133.66 | SL hit (close<static) qty=1.00 sl=133.50 alert=retest2 |

### Cycle 28 — SELL (started 2023-08-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 09:15:00 | 132.35 | 134.24 | 134.44 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 10:15:00 | 136.25 | 134.64 | 134.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 11:15:00 | 137.15 | 135.14 | 134.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 12:15:00 | 139.70 | 140.08 | 138.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-05 12:30:00 | 139.40 | 140.08 | 138.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 09:15:00 | 139.90 | 139.93 | 139.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 09:30:00 | 139.25 | 139.93 | 139.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 11:15:00 | 139.05 | 139.64 | 139.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 11:30:00 | 138.70 | 139.64 | 139.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 12:15:00 | 138.50 | 139.41 | 139.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 12:30:00 | 138.45 | 139.41 | 139.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 13:15:00 | 138.40 | 139.21 | 138.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 13:30:00 | 138.45 | 139.21 | 138.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 10:15:00 | 139.20 | 138.98 | 138.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-08 09:15:00 | 141.15 | 139.58 | 139.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 09:15:00 | 136.30 | 142.58 | 142.27 | SL hit (close<static) qty=1.00 sl=138.55 alert=retest2 |

### Cycle 30 — SELL (started 2023-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 10:15:00 | 138.50 | 141.76 | 141.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 135.20 | 139.53 | 140.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 15:15:00 | 136.05 | 135.88 | 137.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-14 09:15:00 | 137.35 | 135.88 | 137.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 137.35 | 136.17 | 137.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:45:00 | 138.25 | 136.17 | 137.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 10:15:00 | 137.35 | 136.41 | 137.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 10:30:00 | 137.50 | 136.41 | 137.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 11:15:00 | 137.35 | 136.60 | 137.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 12:00:00 | 137.35 | 136.60 | 137.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 12:15:00 | 136.50 | 136.58 | 137.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 13:45:00 | 135.65 | 136.45 | 137.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 09:45:00 | 136.20 | 136.57 | 137.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 10:15:00 | 136.15 | 136.57 | 137.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 11:15:00 | 136.20 | 136.51 | 136.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 14:15:00 | 135.65 | 136.10 | 136.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-15 14:30:00 | 136.45 | 136.10 | 136.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 139.80 | 136.76 | 136.83 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-09-18 09:15:00 | 139.80 | 136.76 | 136.83 | SL hit (close>static) qty=1.00 sl=137.40 alert=retest2 |

### Cycle 31 — BUY (started 2023-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 10:15:00 | 141.60 | 137.73 | 137.26 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 13:15:00 | 136.50 | 138.00 | 138.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 12:15:00 | 136.05 | 137.09 | 137.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 12:15:00 | 136.60 | 135.97 | 136.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 12:15:00 | 136.60 | 135.97 | 136.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 12:15:00 | 136.60 | 135.97 | 136.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 12:45:00 | 136.50 | 135.97 | 136.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 13:15:00 | 136.35 | 136.05 | 136.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 14:15:00 | 135.45 | 136.05 | 136.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-25 09:15:00 | 134.90 | 136.08 | 136.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-26 09:15:00 | 137.40 | 136.12 | 136.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2023-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 09:15:00 | 137.40 | 136.12 | 136.12 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 134.90 | 136.47 | 136.60 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 11:15:00 | 137.30 | 136.76 | 136.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 12:15:00 | 137.80 | 136.97 | 136.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 09:15:00 | 137.40 | 138.49 | 138.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 09:15:00 | 137.40 | 138.49 | 138.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 137.40 | 138.49 | 138.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 09:30:00 | 137.60 | 138.49 | 138.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 10:15:00 | 137.70 | 138.33 | 137.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-04 14:45:00 | 138.55 | 138.15 | 137.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 10:30:00 | 139.05 | 138.69 | 138.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-09 09:15:00 | 137.25 | 138.66 | 138.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 137.25 | 138.66 | 138.67 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 11:15:00 | 139.60 | 138.09 | 138.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 12:15:00 | 140.10 | 138.49 | 138.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 11:15:00 | 139.05 | 139.19 | 138.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-11 11:30:00 | 139.05 | 139.19 | 138.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 12:15:00 | 138.65 | 139.08 | 138.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 12:45:00 | 138.60 | 139.08 | 138.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 13:15:00 | 137.60 | 138.78 | 138.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 13:45:00 | 137.65 | 138.78 | 138.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 14:15:00 | 137.90 | 138.61 | 138.59 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-10-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-11 15:15:00 | 137.80 | 138.45 | 138.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 09:15:00 | 136.80 | 137.97 | 138.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-13 13:15:00 | 137.55 | 137.55 | 137.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-13 14:15:00 | 137.90 | 137.55 | 137.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 14:15:00 | 137.40 | 137.52 | 137.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-13 14:30:00 | 137.95 | 137.52 | 137.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 137.40 | 137.43 | 137.76 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 11:15:00 | 138.35 | 137.81 | 137.75 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 11:15:00 | 137.05 | 137.79 | 137.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 136.45 | 137.20 | 137.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 13:15:00 | 137.00 | 136.86 | 137.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-19 14:00:00 | 137.00 | 136.86 | 137.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 14:15:00 | 136.90 | 136.87 | 137.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 15:00:00 | 136.90 | 136.87 | 137.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 15:15:00 | 136.85 | 136.86 | 137.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 09:30:00 | 136.40 | 136.74 | 137.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 12:15:00 | 129.58 | 132.12 | 133.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-26 15:15:00 | 129.30 | 129.08 | 130.67 | SL hit (close>ema200) qty=0.50 sl=129.08 alert=retest2 |

### Cycle 41 — BUY (started 2023-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 15:15:00 | 132.50 | 131.39 | 131.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 12:15:00 | 132.70 | 131.68 | 131.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 14:15:00 | 132.25 | 133.31 | 132.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 14:15:00 | 132.25 | 133.31 | 132.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 14:15:00 | 132.25 | 133.31 | 132.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 15:00:00 | 132.25 | 133.31 | 132.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 15:15:00 | 132.40 | 133.13 | 132.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 09:15:00 | 133.95 | 133.13 | 132.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-21 12:15:00 | 142.50 | 143.44 | 143.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2023-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 12:15:00 | 142.50 | 143.44 | 143.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 09:15:00 | 142.35 | 143.00 | 143.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-22 15:15:00 | 142.10 | 141.96 | 142.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-23 09:15:00 | 142.20 | 141.96 | 142.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 141.10 | 141.79 | 142.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 10:15:00 | 140.90 | 141.79 | 142.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-24 12:45:00 | 140.60 | 140.43 | 140.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 09:30:00 | 140.85 | 140.51 | 140.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 12:30:00 | 140.90 | 140.84 | 140.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 13:15:00 | 141.45 | 140.96 | 140.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 13:30:00 | 141.55 | 140.96 | 140.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-11-28 14:15:00 | 141.25 | 141.02 | 141.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2023-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 14:15:00 | 141.25 | 141.02 | 141.00 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2023-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-29 14:15:00 | 140.95 | 141.02 | 141.03 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 15:15:00 | 141.20 | 141.06 | 141.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 09:15:00 | 142.70 | 141.39 | 141.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 10:15:00 | 152.25 | 152.70 | 150.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-05 11:00:00 | 152.25 | 152.70 | 150.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 11:15:00 | 150.40 | 152.24 | 150.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 11:45:00 | 150.40 | 152.24 | 150.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 12:15:00 | 150.75 | 151.94 | 150.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 15:00:00 | 152.10 | 151.80 | 150.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-15 09:15:00 | 167.31 | 164.19 | 162.82 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2023-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 15:15:00 | 166.80 | 170.96 | 171.01 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 10:15:00 | 171.70 | 170.98 | 170.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 12:15:00 | 172.70 | 171.33 | 171.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 12:15:00 | 180.10 | 180.85 | 178.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 12:30:00 | 179.90 | 180.85 | 178.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 09:15:00 | 182.70 | 184.35 | 183.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 09:45:00 | 182.00 | 184.35 | 183.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 181.00 | 183.68 | 183.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:30:00 | 180.25 | 183.68 | 183.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2024-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 12:15:00 | 181.10 | 182.66 | 182.81 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 12:15:00 | 183.45 | 182.73 | 182.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 09:15:00 | 184.15 | 183.01 | 182.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 13:15:00 | 185.00 | 186.15 | 185.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 13:15:00 | 185.00 | 186.15 | 185.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 13:15:00 | 185.00 | 186.15 | 185.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 14:00:00 | 185.00 | 186.15 | 185.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 14:15:00 | 186.05 | 186.13 | 185.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-08 09:15:00 | 186.55 | 186.12 | 185.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-08 11:15:00 | 183.90 | 185.36 | 185.12 | SL hit (close<static) qty=1.00 sl=184.05 alert=retest2 |

### Cycle 50 — SELL (started 2024-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 14:15:00 | 183.30 | 184.64 | 184.82 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 10:15:00 | 185.40 | 184.93 | 184.92 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 09:15:00 | 183.15 | 184.75 | 184.90 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-01-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 15:15:00 | 186.00 | 184.74 | 184.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-11 09:15:00 | 186.10 | 185.01 | 184.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-11 11:15:00 | 185.10 | 185.24 | 185.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 11:15:00 | 185.10 | 185.24 | 185.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 11:15:00 | 185.10 | 185.24 | 185.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 11:45:00 | 185.20 | 185.24 | 185.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 12:15:00 | 185.40 | 185.28 | 185.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 12:30:00 | 184.95 | 185.28 | 185.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 13:15:00 | 184.10 | 185.04 | 184.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 14:00:00 | 184.10 | 185.04 | 184.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 14:15:00 | 184.40 | 184.91 | 184.90 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-01-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-11 15:15:00 | 184.35 | 184.80 | 184.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 10:15:00 | 183.65 | 184.50 | 184.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-15 09:15:00 | 185.15 | 184.13 | 184.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 09:15:00 | 185.15 | 184.13 | 184.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 185.15 | 184.13 | 184.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-15 09:30:00 | 184.85 | 184.13 | 184.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 10:15:00 | 185.90 | 184.49 | 184.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-15 11:00:00 | 185.90 | 184.49 | 184.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2024-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 11:15:00 | 186.70 | 184.93 | 184.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-15 12:15:00 | 188.95 | 185.73 | 185.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 11:15:00 | 187.40 | 187.80 | 186.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-16 12:00:00 | 187.40 | 187.80 | 186.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 186.25 | 187.49 | 186.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:00:00 | 186.25 | 187.49 | 186.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 13:15:00 | 186.35 | 187.26 | 186.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:30:00 | 186.40 | 187.26 | 186.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 14:15:00 | 187.00 | 187.21 | 186.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 14:30:00 | 186.45 | 187.21 | 186.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 15:15:00 | 186.50 | 187.07 | 186.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 09:15:00 | 186.25 | 187.07 | 186.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 187.05 | 187.06 | 186.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 11:15:00 | 188.05 | 187.14 | 186.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-17 12:15:00 | 184.95 | 186.42 | 186.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2024-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 12:15:00 | 184.95 | 186.42 | 186.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 183.55 | 185.60 | 186.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 11:15:00 | 187.00 | 185.81 | 186.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 11:15:00 | 187.00 | 185.81 | 186.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 11:15:00 | 187.00 | 185.81 | 186.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 12:00:00 | 187.00 | 185.81 | 186.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 12:15:00 | 186.85 | 186.01 | 186.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 12:30:00 | 185.75 | 186.01 | 186.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2024-01-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 13:15:00 | 187.90 | 186.39 | 186.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-18 15:15:00 | 189.00 | 187.15 | 186.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 09:15:00 | 191.25 | 192.48 | 191.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 191.25 | 192.48 | 191.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 191.25 | 192.48 | 191.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:00:00 | 191.25 | 192.48 | 191.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 189.45 | 191.87 | 190.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:30:00 | 189.05 | 191.87 | 190.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 11:15:00 | 187.05 | 190.91 | 190.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 12:00:00 | 187.05 | 190.91 | 190.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2024-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 12:15:00 | 187.05 | 190.14 | 190.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 11:15:00 | 186.45 | 188.29 | 189.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 13:15:00 | 189.05 | 188.43 | 189.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 13:15:00 | 189.05 | 188.43 | 189.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 189.05 | 188.43 | 189.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:00:00 | 189.05 | 188.43 | 189.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 189.05 | 188.55 | 189.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:45:00 | 189.45 | 188.55 | 189.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 189.40 | 188.72 | 189.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:15:00 | 190.65 | 188.72 | 189.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 190.85 | 189.15 | 189.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:30:00 | 191.85 | 189.15 | 189.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 11:15:00 | 189.15 | 189.16 | 189.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 11:45:00 | 189.45 | 189.16 | 189.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 12:15:00 | 188.85 | 189.09 | 189.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 12:30:00 | 189.35 | 189.09 | 189.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 13:15:00 | 188.75 | 189.03 | 189.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 14:00:00 | 188.75 | 189.03 | 189.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2024-01-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 14:15:00 | 190.05 | 189.23 | 189.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 10:15:00 | 190.40 | 189.54 | 189.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-29 14:15:00 | 190.40 | 190.69 | 190.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-29 14:15:00 | 190.40 | 190.69 | 190.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 14:15:00 | 190.40 | 190.69 | 190.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-29 14:45:00 | 189.65 | 190.69 | 190.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 09:15:00 | 191.25 | 190.96 | 190.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 09:30:00 | 191.15 | 190.96 | 190.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 10:15:00 | 189.35 | 190.64 | 190.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 11:00:00 | 189.35 | 190.64 | 190.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 11:15:00 | 189.35 | 190.38 | 190.14 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 13:15:00 | 189.20 | 189.95 | 189.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 14:15:00 | 188.05 | 189.57 | 189.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-01 09:15:00 | 186.90 | 186.73 | 187.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 09:15:00 | 186.90 | 186.73 | 187.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 186.90 | 186.73 | 187.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-01 09:30:00 | 187.20 | 186.73 | 187.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 187.10 | 184.86 | 186.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 10:00:00 | 187.10 | 184.86 | 186.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 10:15:00 | 185.45 | 184.98 | 186.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-02 12:15:00 | 184.90 | 185.08 | 185.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-02 14:45:00 | 184.60 | 184.87 | 185.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-05 10:00:00 | 184.60 | 184.76 | 185.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-05 10:45:00 | 184.85 | 184.70 | 185.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 182.00 | 182.82 | 183.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 11:00:00 | 181.10 | 182.48 | 183.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-08 11:15:00 | 184.95 | 183.05 | 183.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2024-02-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 11:15:00 | 184.95 | 183.05 | 183.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-08 12:15:00 | 186.85 | 183.81 | 183.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 09:15:00 | 182.10 | 183.98 | 183.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 182.10 | 183.98 | 183.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 182.10 | 183.98 | 183.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 10:00:00 | 182.10 | 183.98 | 183.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 178.25 | 182.83 | 183.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 11:15:00 | 175.85 | 179.13 | 180.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 14:15:00 | 176.45 | 176.06 | 177.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 15:00:00 | 176.45 | 176.06 | 177.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 179.50 | 176.82 | 177.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 10:00:00 | 179.50 | 176.82 | 177.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 180.25 | 177.51 | 177.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 11:00:00 | 180.25 | 177.51 | 177.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2024-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 12:15:00 | 179.85 | 178.36 | 178.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 14:15:00 | 181.30 | 179.22 | 178.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 14:15:00 | 188.40 | 188.93 | 187.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-19 15:00:00 | 188.40 | 188.93 | 187.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 188.50 | 188.85 | 187.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 10:15:00 | 191.25 | 188.85 | 187.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 11:30:00 | 189.70 | 189.08 | 187.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 09:15:00 | 189.70 | 189.85 | 189.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-27 09:15:00 | 208.67 | 204.41 | 201.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 09:15:00 | 207.80 | 212.04 | 212.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 10:15:00 | 204.90 | 210.61 | 211.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 194.65 | 194.47 | 199.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 11:00:00 | 194.65 | 194.47 | 199.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 188.00 | 193.08 | 196.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 10:15:00 | 185.65 | 193.08 | 196.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 10:15:00 | 186.40 | 189.17 | 190.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 11:15:00 | 186.35 | 188.83 | 190.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 13:15:00 | 186.70 | 188.03 | 189.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 187.65 | 186.34 | 187.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 13:00:00 | 187.65 | 186.34 | 187.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 13:15:00 | 187.10 | 186.49 | 187.68 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-03-21 09:15:00 | 194.10 | 188.35 | 188.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 194.10 | 188.35 | 188.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 10:15:00 | 195.60 | 189.80 | 188.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 15:15:00 | 199.00 | 200.00 | 198.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 15:15:00 | 199.00 | 200.00 | 198.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 199.00 | 200.00 | 198.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 09:15:00 | 200.70 | 200.00 | 198.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 09:45:00 | 200.60 | 200.00 | 198.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 11:00:00 | 201.70 | 200.34 | 198.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-02 09:15:00 | 220.77 | 210.06 | 205.59 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-04-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 09:15:00 | 230.40 | 233.30 | 233.46 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 15:15:00 | 233.65 | 233.18 | 233.15 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 09:15:00 | 232.80 | 233.11 | 233.11 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 10:15:00 | 234.00 | 233.29 | 233.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 11:15:00 | 235.70 | 233.77 | 233.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 15:15:00 | 236.25 | 236.43 | 235.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 09:15:00 | 238.35 | 236.43 | 235.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 238.80 | 236.91 | 235.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 09:45:00 | 240.65 | 238.10 | 236.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 11:15:00 | 240.15 | 238.46 | 237.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 11:45:00 | 240.35 | 238.89 | 237.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-29 12:15:00 | 235.95 | 237.37 | 237.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 12:15:00 | 235.95 | 237.37 | 237.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-29 13:15:00 | 235.25 | 236.95 | 237.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-02 11:15:00 | 235.25 | 234.76 | 235.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 11:15:00 | 235.25 | 234.76 | 235.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 11:15:00 | 235.25 | 234.76 | 235.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 12:00:00 | 235.25 | 234.76 | 235.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 12:15:00 | 235.65 | 234.93 | 235.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 12:45:00 | 235.65 | 234.93 | 235.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 13:15:00 | 235.65 | 235.08 | 235.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 13:45:00 | 236.15 | 235.08 | 235.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 14:15:00 | 234.55 | 234.97 | 235.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 11:15:00 | 234.15 | 235.10 | 235.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 15:15:00 | 234.30 | 234.11 | 234.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 09:30:00 | 232.90 | 233.52 | 234.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 13:00:00 | 233.25 | 233.26 | 234.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 14:15:00 | 231.95 | 233.02 | 233.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 11:00:00 | 229.85 | 232.47 | 233.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 13:30:00 | 231.15 | 229.78 | 230.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:30:00 | 229.90 | 230.55 | 230.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 09:15:00 | 222.44 | 226.61 | 227.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 09:15:00 | 222.59 | 226.61 | 227.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 10:15:00 | 221.25 | 226.02 | 227.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 10:15:00 | 221.59 | 226.02 | 227.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-14 09:15:00 | 226.95 | 225.51 | 226.53 | SL hit (close>ema200) qty=0.50 sl=225.51 alert=retest2 |

### Cycle 71 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 229.50 | 227.37 | 227.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 230.95 | 228.49 | 227.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 11:15:00 | 232.35 | 232.56 | 230.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 11:15:00 | 232.35 | 232.56 | 230.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 232.35 | 232.56 | 230.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:45:00 | 231.75 | 232.56 | 230.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 230.10 | 232.07 | 230.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:00:00 | 230.10 | 232.07 | 230.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 229.70 | 231.60 | 230.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:45:00 | 229.50 | 231.60 | 230.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 290.90 | 293.93 | 290.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:00:00 | 290.90 | 293.93 | 290.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 291.35 | 293.41 | 290.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:15:00 | 290.00 | 293.41 | 290.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 289.05 | 292.54 | 290.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:00:00 | 289.05 | 292.54 | 290.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 289.95 | 292.02 | 290.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:45:00 | 286.80 | 292.02 | 290.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 289.20 | 290.89 | 290.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 09:30:00 | 291.00 | 290.87 | 290.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 11:15:00 | 290.85 | 290.67 | 290.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 13:00:00 | 291.20 | 290.89 | 290.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 15:15:00 | 291.55 | 292.27 | 291.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 291.55 | 292.12 | 291.75 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-31 09:15:00 | 287.75 | 291.25 | 291.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2024-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 09:15:00 | 287.75 | 291.25 | 291.39 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 12:15:00 | 293.10 | 291.45 | 291.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 13:15:00 | 295.75 | 292.31 | 291.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 298.35 | 309.77 | 304.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 298.35 | 309.77 | 304.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 298.35 | 309.77 | 304.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 293.35 | 309.77 | 304.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 270.85 | 301.99 | 301.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 270.85 | 301.99 | 301.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 243.00 | 290.19 | 295.76 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 10:15:00 | 279.20 | 273.64 | 273.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 11:15:00 | 281.15 | 275.14 | 273.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 10:15:00 | 288.85 | 289.89 | 287.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 11:00:00 | 288.85 | 289.89 | 287.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 290.00 | 289.91 | 287.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:15:00 | 291.50 | 289.91 | 287.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 11:15:00 | 320.65 | 312.02 | 304.19 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-06-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 13:15:00 | 307.30 | 309.76 | 309.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 14:15:00 | 304.95 | 308.80 | 309.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 09:15:00 | 309.30 | 308.37 | 309.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 309.30 | 308.37 | 309.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 309.30 | 308.37 | 309.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:00:00 | 309.30 | 308.37 | 309.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 309.95 | 308.69 | 309.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:45:00 | 309.90 | 308.69 | 309.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 310.80 | 309.11 | 309.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:00:00 | 310.80 | 309.11 | 309.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 310.90 | 309.47 | 309.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:30:00 | 310.85 | 309.47 | 309.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2024-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 13:15:00 | 310.90 | 309.75 | 309.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 313.10 | 310.48 | 310.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 12:15:00 | 310.65 | 310.82 | 310.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 12:15:00 | 310.65 | 310.82 | 310.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 310.65 | 310.82 | 310.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:45:00 | 310.50 | 310.82 | 310.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 311.20 | 310.89 | 310.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 13:30:00 | 310.60 | 310.89 | 310.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 309.85 | 310.69 | 310.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 14:45:00 | 309.55 | 310.69 | 310.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 309.70 | 310.49 | 310.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:15:00 | 310.40 | 310.49 | 310.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 09:15:00 | 308.60 | 310.11 | 310.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 11:15:00 | 308.00 | 309.46 | 309.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 305.50 | 305.16 | 306.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 09:15:00 | 305.50 | 305.16 | 306.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 305.50 | 305.16 | 306.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 10:15:00 | 304.95 | 305.16 | 306.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 14:00:00 | 304.50 | 304.53 | 305.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 14:30:00 | 304.75 | 304.47 | 305.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 15:15:00 | 309.80 | 305.54 | 306.04 | SL hit (close>static) qty=1.00 sl=308.00 alert=retest2 |

### Cycle 79 — BUY (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 13:15:00 | 306.45 | 306.35 | 306.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 14:15:00 | 308.00 | 306.68 | 306.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 305.65 | 306.65 | 306.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 305.65 | 306.65 | 306.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 305.65 | 306.65 | 306.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:00:00 | 305.65 | 306.65 | 306.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 306.10 | 306.54 | 306.49 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 11:15:00 | 305.05 | 306.24 | 306.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 12:15:00 | 304.70 | 305.94 | 306.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-02 14:15:00 | 306.30 | 305.89 | 306.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 14:15:00 | 306.30 | 305.89 | 306.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 14:15:00 | 306.30 | 305.89 | 306.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 14:45:00 | 306.20 | 305.89 | 306.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 305.80 | 305.87 | 306.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:15:00 | 306.70 | 305.87 | 306.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 09:15:00 | 310.00 | 306.70 | 306.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 14:15:00 | 314.60 | 310.07 | 308.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 10:15:00 | 331.40 | 331.55 | 326.69 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 14:45:00 | 334.85 | 332.49 | 328.67 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-10 09:15:00 | 338.00 | 332.85 | 329.18 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 331.35 | 332.61 | 329.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 328.15 | 332.61 | 329.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 332.25 | 332.54 | 329.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 09:15:00 | 336.90 | 332.74 | 330.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 11:00:00 | 335.25 | 333.38 | 331.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 11:45:00 | 334.95 | 333.61 | 331.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 12:15:00 | 334.90 | 333.61 | 331.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 332.50 | 334.52 | 333.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-12 11:15:00 | 332.50 | 334.52 | 333.32 | SL hit (close<ema400) qty=1.00 sl=333.32 alert=retest1 |

### Cycle 82 — SELL (started 2024-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 12:15:00 | 331.90 | 332.81 | 332.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 14:15:00 | 331.40 | 332.35 | 332.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 309.75 | 309.58 | 314.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 09:30:00 | 309.60 | 309.58 | 314.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 312.30 | 310.83 | 313.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 313.55 | 310.83 | 313.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 313.20 | 311.47 | 313.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 311.50 | 311.47 | 313.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 311.85 | 311.55 | 313.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 288.05 | 311.65 | 313.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 11:15:00 | 308.75 | 304.59 | 304.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 11:15:00 | 308.75 | 304.59 | 304.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 309.70 | 306.33 | 305.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 14:15:00 | 318.15 | 318.97 | 315.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 15:00:00 | 318.15 | 318.97 | 315.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 316.85 | 318.11 | 316.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:30:00 | 316.40 | 318.11 | 316.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 316.25 | 317.69 | 316.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:00:00 | 316.25 | 317.69 | 316.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 315.85 | 317.32 | 316.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 315.85 | 317.32 | 316.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 316.30 | 317.12 | 316.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:45:00 | 315.00 | 316.75 | 316.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 312.60 | 315.92 | 315.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:45:00 | 312.00 | 315.92 | 315.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 11:15:00 | 310.50 | 314.84 | 315.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 12:15:00 | 310.00 | 313.87 | 314.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 296.40 | 294.86 | 300.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 296.40 | 294.86 | 300.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 296.40 | 294.86 | 300.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 290.25 | 293.18 | 297.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 09:15:00 | 299.15 | 297.20 | 296.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 299.15 | 297.20 | 296.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 302.90 | 299.32 | 298.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 299.40 | 300.95 | 299.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 299.40 | 300.95 | 299.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 299.40 | 300.95 | 299.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:45:00 | 299.10 | 300.95 | 299.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 302.90 | 301.34 | 300.16 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 14:15:00 | 295.50 | 299.81 | 300.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 293.60 | 297.98 | 299.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 297.60 | 295.35 | 296.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 297.60 | 295.35 | 296.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 297.60 | 295.35 | 296.85 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 300.80 | 297.89 | 297.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 303.30 | 298.97 | 298.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 09:15:00 | 300.60 | 301.86 | 300.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 09:15:00 | 300.60 | 301.86 | 300.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 300.60 | 301.86 | 300.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:00:00 | 300.60 | 301.86 | 300.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 299.50 | 301.39 | 300.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:30:00 | 298.85 | 301.39 | 300.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 300.15 | 301.14 | 300.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 298.60 | 301.14 | 300.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 303.70 | 304.38 | 303.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:30:00 | 304.10 | 304.38 | 303.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 305.30 | 306.33 | 305.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:45:00 | 304.85 | 306.33 | 305.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 305.95 | 306.26 | 305.88 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2024-08-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 13:15:00 | 303.45 | 305.22 | 305.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 14:15:00 | 301.10 | 304.40 | 305.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 10:15:00 | 299.10 | 297.02 | 298.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 10:15:00 | 299.10 | 297.02 | 298.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 299.10 | 297.02 | 298.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:45:00 | 299.40 | 297.02 | 298.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 298.20 | 297.25 | 298.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:45:00 | 298.45 | 297.25 | 298.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 12:15:00 | 298.95 | 297.59 | 298.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 13:00:00 | 298.95 | 297.59 | 298.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 298.95 | 297.86 | 298.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 14:15:00 | 301.05 | 297.86 | 298.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 299.30 | 298.15 | 298.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 14:30:00 | 299.45 | 298.15 | 298.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 299.50 | 298.42 | 298.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:15:00 | 298.60 | 298.42 | 298.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 297.85 | 298.31 | 298.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 11:45:00 | 296.20 | 297.80 | 298.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 15:15:00 | 296.85 | 297.01 | 297.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 10:15:00 | 299.80 | 298.15 | 298.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 10:15:00 | 299.80 | 298.15 | 298.07 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 09:15:00 | 294.00 | 297.59 | 297.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 10:15:00 | 293.10 | 296.69 | 297.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 280.65 | 280.62 | 284.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 15:00:00 | 280.65 | 280.62 | 284.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 282.90 | 281.64 | 284.64 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2024-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 09:15:00 | 288.30 | 285.59 | 285.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 09:15:00 | 289.90 | 287.88 | 286.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 11:15:00 | 290.90 | 290.98 | 289.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 11:45:00 | 291.20 | 290.98 | 289.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 289.70 | 290.69 | 289.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 15:00:00 | 289.70 | 290.69 | 289.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 290.00 | 290.55 | 289.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:15:00 | 290.45 | 290.55 | 289.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 12:15:00 | 290.80 | 290.18 | 289.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 09:15:00 | 286.10 | 289.67 | 289.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 286.10 | 289.67 | 289.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 11:15:00 | 284.75 | 288.08 | 288.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 275.40 | 274.59 | 278.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 10:15:00 | 276.50 | 274.59 | 278.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 279.05 | 275.48 | 278.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:00:00 | 279.05 | 275.48 | 278.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 276.60 | 275.70 | 278.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:30:00 | 275.40 | 276.07 | 278.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:15:00 | 276.30 | 276.07 | 278.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 09:15:00 | 282.60 | 277.82 | 278.36 | SL hit (close>static) qty=1.00 sl=279.40 alert=retest2 |

### Cycle 93 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 283.35 | 278.93 | 278.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 13:15:00 | 285.00 | 281.66 | 280.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 289.30 | 289.33 | 286.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 11:45:00 | 289.25 | 289.33 | 286.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 290.85 | 290.65 | 289.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:30:00 | 289.35 | 290.65 | 289.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 286.20 | 290.40 | 289.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:00:00 | 286.20 | 290.40 | 289.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 286.25 | 289.57 | 289.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:30:00 | 285.15 | 289.57 | 289.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 287.20 | 289.09 | 289.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 14:15:00 | 285.30 | 287.60 | 288.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 281.00 | 280.65 | 282.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 12:00:00 | 281.00 | 280.65 | 282.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 276.00 | 271.55 | 274.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 276.00 | 271.55 | 274.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 275.75 | 272.39 | 274.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:30:00 | 276.70 | 272.39 | 274.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 277.20 | 273.35 | 274.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 277.20 | 273.35 | 274.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 278.25 | 274.33 | 275.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 278.25 | 274.33 | 275.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 281.00 | 276.57 | 276.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 282.10 | 277.68 | 276.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 14:15:00 | 281.50 | 281.51 | 279.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-09 15:00:00 | 281.50 | 281.51 | 279.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 12:15:00 | 285.40 | 286.56 | 285.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 13:00:00 | 285.40 | 286.56 | 285.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 285.40 | 286.33 | 285.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 13:30:00 | 284.65 | 286.33 | 285.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 285.70 | 286.20 | 285.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 14:30:00 | 285.15 | 286.20 | 285.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 15:15:00 | 285.60 | 286.08 | 285.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:15:00 | 286.70 | 286.08 | 285.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 13:00:00 | 286.35 | 287.12 | 286.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 14:15:00 | 286.10 | 286.78 | 286.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 14:15:00 | 285.70 | 286.56 | 286.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 14:15:00 | 285.70 | 286.56 | 286.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 283.30 | 285.76 | 286.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 11:15:00 | 286.50 | 285.63 | 286.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 11:15:00 | 286.50 | 285.63 | 286.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 286.50 | 285.63 | 286.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:00:00 | 286.50 | 285.63 | 286.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 286.60 | 285.82 | 286.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:30:00 | 287.00 | 285.82 | 286.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 284.40 | 285.54 | 285.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 09:15:00 | 278.10 | 285.21 | 285.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 10:45:00 | 283.80 | 284.66 | 285.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 15:15:00 | 287.00 | 285.85 | 285.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2024-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 15:15:00 | 287.00 | 285.85 | 285.74 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 10:15:00 | 285.35 | 285.64 | 285.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 283.00 | 285.11 | 285.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 271.70 | 271.24 | 274.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 10:00:00 | 271.70 | 271.24 | 274.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 13:15:00 | 271.65 | 267.18 | 269.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 14:00:00 | 271.65 | 267.18 | 269.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 14:15:00 | 272.35 | 268.21 | 270.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 15:15:00 | 275.60 | 268.21 | 270.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 275.60 | 269.69 | 270.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:15:00 | 270.95 | 269.69 | 270.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 269.95 | 269.63 | 270.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:30:00 | 271.10 | 269.63 | 270.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 269.25 | 269.56 | 270.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 12:45:00 | 268.05 | 269.34 | 270.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 14:00:00 | 267.85 | 269.05 | 269.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-29 09:15:00 | 274.20 | 270.31 | 270.35 | SL hit (close>static) qty=1.00 sl=271.00 alert=retest2 |

### Cycle 99 — BUY (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 10:15:00 | 272.25 | 270.70 | 270.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 11:15:00 | 276.10 | 271.78 | 271.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 284.85 | 286.54 | 282.15 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-01 18:00:00 | 288.50 | 286.18 | 284.03 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-01 18:30:00 | 289.00 | 286.67 | 284.44 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 284.15 | 286.16 | 284.42 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 284.15 | 286.16 | 284.42 | SL hit (close<ema400) qty=1.00 sl=284.42 alert=retest1 |

### Cycle 100 — SELL (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 10:15:00 | 280.55 | 283.19 | 283.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 11:15:00 | 278.90 | 282.33 | 283.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 285.10 | 282.50 | 283.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 285.10 | 282.50 | 283.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 285.10 | 282.50 | 283.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 285.10 | 282.50 | 283.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 286.65 | 283.33 | 283.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 286.65 | 283.33 | 283.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 285.55 | 283.78 | 283.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 292.10 | 285.44 | 284.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 11:15:00 | 298.65 | 300.04 | 296.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 12:00:00 | 298.65 | 300.04 | 296.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 297.65 | 299.14 | 297.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:30:00 | 298.35 | 299.14 | 297.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 297.00 | 298.71 | 297.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 297.85 | 298.71 | 297.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 297.70 | 298.51 | 297.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 11:00:00 | 302.80 | 299.37 | 297.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 14:15:00 | 300.40 | 299.61 | 298.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 12:15:00 | 295.10 | 297.48 | 297.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 12:15:00 | 295.10 | 297.48 | 297.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 13:15:00 | 293.05 | 296.60 | 297.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 12:15:00 | 283.25 | 283.18 | 287.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 13:00:00 | 283.25 | 283.18 | 287.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 282.15 | 282.28 | 285.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:15:00 | 278.45 | 281.31 | 283.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 15:00:00 | 278.55 | 281.28 | 282.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 13:15:00 | 282.50 | 279.41 | 279.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 282.50 | 279.41 | 279.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 294.75 | 282.91 | 281.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 14:15:00 | 305.70 | 306.01 | 302.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 15:00:00 | 305.70 | 306.01 | 302.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 305.00 | 306.75 | 305.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:30:00 | 305.25 | 306.75 | 305.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 304.60 | 306.32 | 305.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:30:00 | 303.65 | 306.32 | 305.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 12:15:00 | 304.60 | 305.98 | 305.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 12:45:00 | 305.05 | 305.98 | 305.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 306.15 | 306.01 | 305.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 14:15:00 | 306.95 | 306.01 | 305.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 11:15:00 | 313.50 | 314.27 | 314.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 11:15:00 | 313.50 | 314.27 | 314.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 14:15:00 | 313.00 | 313.96 | 314.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 313.90 | 312.74 | 313.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 11:15:00 | 313.90 | 312.74 | 313.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 313.90 | 312.74 | 313.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:00:00 | 313.90 | 312.74 | 313.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 315.20 | 313.23 | 313.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 315.20 | 313.23 | 313.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 315.85 | 313.76 | 313.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:45:00 | 315.90 | 313.76 | 313.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 315.80 | 314.17 | 313.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 14:15:00 | 316.20 | 315.14 | 314.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 10:15:00 | 315.65 | 315.73 | 315.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 10:30:00 | 316.70 | 315.73 | 315.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 314.70 | 315.53 | 315.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:00:00 | 314.70 | 315.53 | 315.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 314.15 | 315.25 | 314.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:00:00 | 314.15 | 315.25 | 314.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 311.45 | 314.49 | 314.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 14:15:00 | 310.65 | 313.72 | 314.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 10:15:00 | 300.65 | 300.00 | 303.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 11:00:00 | 300.65 | 300.00 | 303.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 295.75 | 294.01 | 296.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 13:15:00 | 292.75 | 293.88 | 295.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 11:30:00 | 293.00 | 293.56 | 294.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 10:45:00 | 292.35 | 294.27 | 294.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 14:45:00 | 291.50 | 293.26 | 294.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 290.80 | 288.97 | 290.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:00:00 | 290.80 | 288.97 | 290.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 291.35 | 289.45 | 290.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:45:00 | 291.45 | 289.45 | 290.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 291.30 | 289.82 | 290.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 12:30:00 | 291.25 | 289.82 | 290.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 293.85 | 290.62 | 291.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 293.85 | 290.62 | 291.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-31 15:15:00 | 293.50 | 291.60 | 291.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2024-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 15:15:00 | 293.50 | 291.60 | 291.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 13:15:00 | 293.80 | 292.36 | 291.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 292.60 | 292.86 | 292.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 292.60 | 292.86 | 292.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 292.60 | 292.86 | 292.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:45:00 | 293.10 | 292.86 | 292.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 292.50 | 292.81 | 292.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:45:00 | 292.30 | 292.81 | 292.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 293.15 | 292.88 | 292.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 13:15:00 | 294.25 | 292.88 | 292.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 14:15:00 | 292.00 | 294.09 | 293.76 | SL hit (close<static) qty=1.00 sl=292.15 alert=retest2 |

### Cycle 108 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 289.45 | 292.82 | 293.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 284.80 | 291.21 | 292.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 12:15:00 | 286.40 | 285.39 | 287.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 12:45:00 | 285.95 | 285.39 | 287.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 287.45 | 285.80 | 287.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:30:00 | 287.05 | 285.80 | 287.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 287.10 | 286.06 | 287.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:45:00 | 287.95 | 286.06 | 287.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 269.75 | 266.82 | 269.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:00:00 | 269.75 | 266.82 | 269.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 270.55 | 267.56 | 269.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 268.35 | 267.56 | 269.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 271.30 | 268.53 | 269.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 271.20 | 268.53 | 269.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 270.70 | 268.97 | 269.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 11:30:00 | 271.20 | 268.97 | 269.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 267.55 | 268.78 | 269.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 14:15:00 | 266.70 | 268.78 | 269.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 14:45:00 | 266.90 | 268.63 | 269.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 274.00 | 269.60 | 269.65 | SL hit (close>static) qty=1.00 sl=269.65 alert=retest2 |

### Cycle 109 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 275.65 | 270.81 | 270.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 281.00 | 276.01 | 273.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 282.60 | 284.37 | 281.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 282.60 | 284.37 | 281.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 281.20 | 283.73 | 281.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:30:00 | 281.90 | 283.73 | 281.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 282.45 | 283.48 | 281.53 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 270.20 | 279.06 | 280.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 268.40 | 276.93 | 278.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 274.60 | 272.40 | 275.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 274.60 | 272.40 | 275.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 274.60 | 272.40 | 275.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 274.60 | 272.40 | 275.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 275.05 | 272.93 | 275.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 274.80 | 272.93 | 275.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 275.25 | 273.39 | 275.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:30:00 | 274.50 | 273.39 | 275.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 274.95 | 273.70 | 275.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:30:00 | 275.55 | 273.70 | 275.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 274.75 | 273.91 | 275.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:45:00 | 276.00 | 273.91 | 275.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 271.50 | 273.53 | 274.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:30:00 | 269.30 | 272.39 | 273.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 264.20 | 271.96 | 273.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 255.84 | 263.47 | 267.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 265.60 | 260.69 | 263.53 | SL hit (close>ema200) qty=0.50 sl=260.69 alert=retest2 |

### Cycle 111 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 267.45 | 264.71 | 264.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 15:15:00 | 268.00 | 265.37 | 264.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 290.00 | 291.29 | 284.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 11:45:00 | 289.90 | 291.29 | 284.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 273.35 | 287.70 | 283.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 273.35 | 287.70 | 283.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 281.60 | 286.48 | 283.08 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 267.80 | 281.21 | 281.39 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 12:15:00 | 282.70 | 278.89 | 278.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 14:15:00 | 284.75 | 280.69 | 279.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 284.90 | 287.40 | 284.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 284.90 | 287.40 | 284.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 284.90 | 287.40 | 284.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:00:00 | 284.90 | 287.40 | 284.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 285.15 | 286.95 | 284.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:45:00 | 285.00 | 286.95 | 284.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 284.95 | 286.55 | 284.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:45:00 | 284.95 | 286.55 | 284.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 283.90 | 286.02 | 284.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 283.90 | 286.02 | 284.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 280.35 | 284.89 | 284.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:00:00 | 280.35 | 284.89 | 284.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 14:15:00 | 279.50 | 283.81 | 283.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 275.70 | 281.58 | 282.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 263.30 | 261.46 | 265.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 10:00:00 | 263.30 | 261.46 | 265.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 264.00 | 262.51 | 265.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 12:00:00 | 264.00 | 262.51 | 265.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 13:15:00 | 262.10 | 262.68 | 264.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 14:30:00 | 261.30 | 262.49 | 264.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:15:00 | 261.15 | 262.49 | 264.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 248.23 | 255.11 | 259.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 248.09 | 251.56 | 256.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 248.70 | 248.47 | 253.00 | SL hit (close>ema200) qty=0.50 sl=248.47 alert=retest2 |

### Cycle 115 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 253.15 | 249.56 | 249.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 253.50 | 250.35 | 249.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 255.30 | 256.83 | 254.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 255.30 | 256.83 | 254.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 256.65 | 256.63 | 255.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:30:00 | 255.60 | 256.63 | 255.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 255.00 | 256.33 | 255.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 253.10 | 256.33 | 255.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 255.05 | 256.07 | 255.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:15:00 | 255.50 | 255.86 | 255.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 14:30:00 | 256.35 | 256.24 | 255.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 10:15:00 | 254.50 | 256.18 | 256.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 10:15:00 | 254.50 | 256.18 | 256.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 11:15:00 | 253.05 | 255.56 | 255.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 11:15:00 | 249.95 | 247.51 | 249.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 11:15:00 | 249.95 | 247.51 | 249.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 11:15:00 | 249.95 | 247.51 | 249.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 11:45:00 | 248.98 | 247.51 | 249.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 256.34 | 249.28 | 250.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:00:00 | 256.34 | 249.28 | 250.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-03-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 13:15:00 | 258.01 | 251.02 | 250.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 09:15:00 | 263.13 | 255.29 | 253.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 272.40 | 272.52 | 268.92 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 09:15:00 | 277.46 | 272.52 | 268.92 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 275.03 | 276.75 | 274.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:00:00 | 275.03 | 276.75 | 274.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 272.67 | 275.94 | 274.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-10 14:15:00 | 272.67 | 275.94 | 274.40 | SL hit (close<ema400) qty=1.00 sl=274.40 alert=retest1 |

### Cycle 118 — SELL (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 11:15:00 | 298.65 | 300.93 | 301.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 13:15:00 | 295.95 | 299.49 | 300.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 09:15:00 | 284.40 | 284.36 | 289.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 280.45 | 285.31 | 287.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 280.45 | 285.31 | 287.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 14:00:00 | 280.25 | 283.10 | 285.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 09:15:00 | 268.30 | 282.06 | 284.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 266.24 | 279.03 | 283.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-07 15:15:00 | 274.00 | 273.15 | 277.75 | SL hit (close>ema200) qty=0.50 sl=273.15 alert=retest2 |

### Cycle 119 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 282.10 | 279.34 | 279.22 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 09:15:00 | 277.95 | 279.07 | 279.11 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 11:15:00 | 281.00 | 279.43 | 279.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 285.60 | 281.28 | 280.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 291.70 | 292.72 | 290.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 291.70 | 292.72 | 290.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 298.00 | 303.58 | 303.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 298.00 | 303.58 | 303.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 296.10 | 302.09 | 302.54 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 11:15:00 | 305.95 | 302.56 | 302.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 317.00 | 306.73 | 304.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 312.95 | 314.13 | 310.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 315.65 | 314.35 | 312.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 315.65 | 314.35 | 312.25 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 311.05 | 313.22 | 313.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 10:15:00 | 309.25 | 311.41 | 312.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 15:15:00 | 310.10 | 309.88 | 311.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-08 09:15:00 | 310.60 | 309.88 | 311.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 310.60 | 310.03 | 311.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:45:00 | 308.50 | 310.07 | 310.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 318.45 | 310.59 | 310.78 | SL hit (close>static) qty=1.00 sl=312.50 alert=retest2 |

### Cycle 125 — BUY (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 10:15:00 | 315.40 | 311.55 | 311.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 321.50 | 317.23 | 314.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 15:15:00 | 363.00 | 363.77 | 358.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 09:15:00 | 360.90 | 363.77 | 358.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 364.45 | 363.90 | 359.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 375.40 | 367.84 | 363.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 384.65 | 386.54 | 386.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 14:15:00 | 384.65 | 386.54 | 386.72 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 13:15:00 | 387.75 | 386.83 | 386.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 10:15:00 | 389.05 | 387.59 | 387.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 386.45 | 387.56 | 387.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 12:15:00 | 386.45 | 387.56 | 387.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 386.45 | 387.56 | 387.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 12:45:00 | 386.15 | 387.56 | 387.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 386.40 | 387.33 | 387.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 14:00:00 | 386.40 | 387.33 | 387.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 385.95 | 386.87 | 386.96 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 389.50 | 387.12 | 386.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 11:15:00 | 391.60 | 389.44 | 388.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 391.20 | 391.39 | 389.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 391.20 | 391.39 | 389.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 391.20 | 391.39 | 389.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:45:00 | 391.00 | 391.39 | 389.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 389.65 | 390.89 | 389.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 15:15:00 | 390.90 | 390.73 | 390.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 393.00 | 394.17 | 394.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 393.00 | 394.17 | 394.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 391.90 | 393.72 | 394.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 09:15:00 | 395.70 | 391.33 | 392.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 395.70 | 391.33 | 392.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 395.70 | 391.33 | 392.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:00:00 | 395.70 | 391.33 | 392.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 396.10 | 392.29 | 392.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:45:00 | 395.40 | 392.29 | 392.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 393.25 | 392.51 | 392.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 14:00:00 | 393.25 | 392.51 | 392.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 394.10 | 392.83 | 392.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 394.10 | 392.83 | 392.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 15:15:00 | 395.15 | 393.30 | 393.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 09:15:00 | 397.05 | 394.05 | 393.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 12:15:00 | 401.55 | 402.35 | 399.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 12:45:00 | 401.25 | 402.35 | 399.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 404.15 | 402.53 | 400.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 10:15:00 | 404.65 | 402.53 | 400.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 12:15:00 | 395.65 | 399.97 | 400.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 395.65 | 399.97 | 400.22 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 403.45 | 400.34 | 400.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 11:15:00 | 406.20 | 401.51 | 400.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 418.20 | 418.23 | 414.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 14:45:00 | 418.20 | 418.23 | 414.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 410.60 | 416.83 | 414.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:45:00 | 411.25 | 416.83 | 414.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 411.50 | 415.76 | 414.08 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2025-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 13:15:00 | 408.15 | 412.73 | 412.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 14:15:00 | 406.60 | 411.50 | 412.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 409.95 | 409.14 | 410.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 14:15:00 | 409.95 | 409.14 | 410.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 409.95 | 409.14 | 410.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:45:00 | 410.20 | 409.14 | 410.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 409.85 | 409.28 | 410.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 412.60 | 409.28 | 410.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 415.40 | 410.50 | 410.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:00:00 | 415.40 | 410.50 | 410.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 10:15:00 | 413.65 | 411.13 | 411.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 15:15:00 | 416.40 | 413.46 | 412.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 428.35 | 429.21 | 424.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 09:45:00 | 429.35 | 429.21 | 424.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 424.00 | 427.70 | 424.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 424.55 | 427.70 | 424.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 425.15 | 427.19 | 424.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 14:30:00 | 425.20 | 427.11 | 424.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:45:00 | 429.50 | 427.51 | 425.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 418.90 | 425.94 | 426.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 418.90 | 425.94 | 426.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 10:15:00 | 415.05 | 418.27 | 419.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 412.00 | 408.85 | 410.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 412.00 | 408.85 | 410.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 412.00 | 408.85 | 410.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:00:00 | 412.00 | 408.85 | 410.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 410.15 | 409.11 | 410.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 11:45:00 | 409.60 | 409.24 | 410.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 14:15:00 | 402.95 | 401.26 | 401.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 14:15:00 | 402.95 | 401.26 | 401.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 15:15:00 | 403.45 | 401.70 | 401.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 399.25 | 401.21 | 401.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 399.25 | 401.21 | 401.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 399.25 | 401.21 | 401.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 399.25 | 401.21 | 401.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 398.40 | 400.65 | 400.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 395.35 | 397.51 | 398.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 15:15:00 | 395.80 | 395.76 | 397.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:15:00 | 394.80 | 395.76 | 397.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 388.80 | 386.96 | 390.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 388.80 | 386.96 | 390.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 388.50 | 387.27 | 390.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:45:00 | 388.85 | 387.27 | 390.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 390.00 | 387.91 | 389.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 389.85 | 387.91 | 389.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 386.65 | 387.66 | 389.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 385.95 | 387.66 | 389.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 15:15:00 | 385.65 | 387.41 | 388.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 13:45:00 | 385.40 | 385.87 | 387.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 13:15:00 | 389.25 | 385.13 | 384.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 389.25 | 385.13 | 384.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 14:15:00 | 389.40 | 385.98 | 385.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 09:15:00 | 385.25 | 386.46 | 385.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 09:15:00 | 385.25 | 386.46 | 385.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 385.25 | 386.46 | 385.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:45:00 | 384.40 | 386.46 | 385.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 385.50 | 386.27 | 385.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 384.95 | 386.27 | 385.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 384.75 | 385.97 | 385.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:00:00 | 384.75 | 385.97 | 385.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 385.20 | 385.81 | 385.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:00:00 | 385.20 | 385.81 | 385.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 386.85 | 386.02 | 385.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 10:15:00 | 389.50 | 386.61 | 385.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 12:15:00 | 389.00 | 387.25 | 386.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 15:00:00 | 390.00 | 388.62 | 387.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 13:15:00 | 385.15 | 386.77 | 386.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 13:15:00 | 385.15 | 386.77 | 386.91 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 388.25 | 387.20 | 387.09 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 09:15:00 | 384.70 | 386.70 | 386.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 09:15:00 | 383.20 | 385.07 | 385.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 10:15:00 | 386.55 | 385.36 | 385.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 10:15:00 | 386.55 | 385.36 | 385.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 386.55 | 385.36 | 385.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:45:00 | 386.15 | 385.36 | 385.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 383.95 | 385.08 | 385.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 12:30:00 | 382.35 | 384.70 | 385.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 382.10 | 384.23 | 385.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 10:00:00 | 382.35 | 383.85 | 384.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:30:00 | 381.70 | 381.87 | 383.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 386.20 | 382.47 | 383.37 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-13 12:15:00 | 385.65 | 383.91 | 383.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 12:15:00 | 385.65 | 383.91 | 383.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 13:15:00 | 389.25 | 384.98 | 384.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 383.25 | 385.78 | 384.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 383.25 | 385.78 | 384.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 383.25 | 385.78 | 384.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 383.25 | 385.78 | 384.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 383.15 | 385.25 | 384.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 383.15 | 385.25 | 384.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 384.90 | 384.99 | 384.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:45:00 | 386.35 | 385.33 | 384.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 11:15:00 | 382.25 | 384.71 | 384.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 382.25 | 384.71 | 384.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 09:15:00 | 378.65 | 382.50 | 383.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 15:15:00 | 380.60 | 380.51 | 381.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 09:15:00 | 377.75 | 380.51 | 381.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 376.40 | 379.69 | 381.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 10:30:00 | 374.05 | 378.78 | 380.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 11:45:00 | 374.70 | 375.05 | 377.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 14:00:00 | 375.30 | 375.28 | 376.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 15:00:00 | 374.70 | 376.29 | 376.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 373.15 | 375.53 | 376.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:15:00 | 372.55 | 374.05 | 375.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 371.55 | 369.10 | 368.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 371.55 | 369.10 | 368.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 373.75 | 370.60 | 369.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 375.85 | 377.76 | 376.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 10:15:00 | 375.85 | 377.76 | 376.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 375.85 | 377.76 | 376.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 375.85 | 377.76 | 376.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 374.90 | 377.19 | 376.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:30:00 | 375.05 | 377.19 | 376.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 372.25 | 375.10 | 375.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 370.80 | 373.90 | 374.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 15:15:00 | 371.75 | 371.64 | 373.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:15:00 | 371.30 | 371.64 | 373.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 373.00 | 371.91 | 373.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:45:00 | 370.70 | 371.87 | 372.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 378.95 | 373.38 | 372.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 378.95 | 373.38 | 372.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 381.05 | 374.91 | 373.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 14:15:00 | 384.95 | 385.84 | 382.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 15:00:00 | 384.95 | 385.84 | 382.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 405.70 | 409.02 | 408.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:00:00 | 405.70 | 409.02 | 408.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 406.30 | 408.48 | 408.04 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 13:15:00 | 406.85 | 407.60 | 407.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 405.40 | 406.83 | 407.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 402.80 | 399.59 | 401.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 402.80 | 399.59 | 401.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 402.80 | 399.59 | 401.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:45:00 | 403.30 | 399.59 | 401.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 405.15 | 400.70 | 402.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:00:00 | 405.15 | 400.70 | 402.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 405.25 | 401.61 | 402.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:30:00 | 405.65 | 401.61 | 402.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 403.65 | 402.92 | 402.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 402.20 | 402.92 | 402.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 09:15:00 | 403.40 | 403.02 | 402.99 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 401.40 | 402.69 | 402.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 400.05 | 402.17 | 402.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 405.90 | 400.32 | 401.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 405.90 | 400.32 | 401.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 405.90 | 400.32 | 401.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 405.50 | 400.32 | 401.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 404.70 | 401.20 | 401.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:45:00 | 405.20 | 401.20 | 401.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2025-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 13:15:00 | 402.60 | 401.85 | 401.79 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 14:15:00 | 400.10 | 401.50 | 401.64 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 09:15:00 | 404.90 | 402.10 | 401.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 11:15:00 | 407.30 | 404.97 | 403.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 13:15:00 | 405.25 | 405.37 | 404.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 14:00:00 | 405.25 | 405.37 | 404.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 412.00 | 407.07 | 405.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 13:30:00 | 413.95 | 410.86 | 409.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 14:30:00 | 413.90 | 411.29 | 409.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 414.20 | 411.64 | 409.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 403.95 | 409.28 | 409.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 403.95 | 409.28 | 409.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 403.30 | 406.36 | 407.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 406.30 | 405.84 | 407.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 10:15:00 | 406.30 | 405.84 | 407.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 406.30 | 405.84 | 407.30 | EMA400 retest candle locked (from downside) |

### Cycle 155 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 414.00 | 408.79 | 408.18 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 15:15:00 | 409.65 | 409.95 | 409.96 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 09:15:00 | 411.05 | 410.17 | 410.06 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 407.20 | 409.58 | 409.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 404.30 | 408.52 | 409.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 406.45 | 405.16 | 407.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 406.45 | 405.16 | 407.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 406.80 | 405.49 | 407.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 407.30 | 405.49 | 407.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 408.00 | 405.99 | 407.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 408.00 | 405.99 | 407.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 409.40 | 406.67 | 407.31 | EMA400 retest candle locked (from downside) |

### Cycle 159 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 411.80 | 408.32 | 407.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 418.85 | 412.45 | 410.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 414.20 | 414.35 | 411.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 13:00:00 | 414.20 | 414.35 | 411.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 413.00 | 414.08 | 412.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 412.20 | 414.08 | 412.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 412.90 | 413.67 | 412.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 415.30 | 413.67 | 412.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 415.30 | 418.54 | 418.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 415.30 | 418.54 | 418.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 09:15:00 | 412.50 | 416.06 | 417.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 10:15:00 | 409.70 | 409.13 | 411.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-30 11:00:00 | 409.70 | 409.13 | 411.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 414.00 | 410.39 | 411.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:45:00 | 414.35 | 410.39 | 411.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 409.10 | 410.29 | 410.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:30:00 | 411.35 | 410.29 | 410.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — BUY (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 12:15:00 | 420.50 | 412.34 | 411.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 13:15:00 | 424.70 | 414.81 | 412.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 418.60 | 420.69 | 418.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 10:00:00 | 418.60 | 420.69 | 418.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 417.70 | 420.09 | 418.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:45:00 | 417.55 | 420.09 | 418.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 417.05 | 419.49 | 418.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:45:00 | 417.10 | 419.49 | 418.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — SELL (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 15:15:00 | 415.75 | 417.34 | 417.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 412.10 | 416.29 | 417.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 411.35 | 410.64 | 413.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:00:00 | 411.35 | 410.64 | 413.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 411.25 | 410.76 | 412.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:30:00 | 412.80 | 410.76 | 412.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 413.35 | 411.28 | 412.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 413.35 | 411.28 | 412.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 413.65 | 411.76 | 412.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 413.85 | 411.76 | 412.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 418.00 | 413.68 | 413.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 424.30 | 417.39 | 415.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 10:15:00 | 424.20 | 424.60 | 421.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 11:00:00 | 424.20 | 424.60 | 421.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 424.70 | 424.40 | 422.11 | EMA400 retest candle locked (from upside) |

### Cycle 164 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 420.20 | 421.45 | 421.61 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 10:15:00 | 423.70 | 421.97 | 421.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 12:15:00 | 423.85 | 422.48 | 422.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 10:15:00 | 423.70 | 424.34 | 423.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 11:15:00 | 422.60 | 424.34 | 423.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 422.90 | 424.05 | 423.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 12:45:00 | 424.45 | 424.01 | 423.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 13:30:00 | 424.20 | 424.21 | 423.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 14:15:00 | 420.65 | 423.16 | 423.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 420.65 | 423.16 | 423.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 418.50 | 421.98 | 422.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 12:15:00 | 421.10 | 420.96 | 422.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 12:15:00 | 421.10 | 420.96 | 422.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 421.10 | 420.96 | 422.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:45:00 | 421.05 | 420.96 | 422.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 423.40 | 421.50 | 422.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:00:00 | 423.40 | 421.50 | 422.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 423.25 | 421.85 | 422.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 425.20 | 421.85 | 422.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 425.05 | 422.49 | 422.48 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 419.00 | 422.39 | 422.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 12:15:00 | 416.80 | 420.58 | 421.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 10:15:00 | 409.80 | 408.78 | 412.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 10:30:00 | 409.80 | 408.78 | 412.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 410.05 | 410.09 | 412.32 | EMA400 retest candle locked (from downside) |

### Cycle 169 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 415.50 | 412.93 | 412.85 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 409.75 | 412.75 | 413.15 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 11:15:00 | 414.00 | 413.17 | 413.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 13:15:00 | 417.95 | 414.48 | 413.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 415.10 | 415.40 | 414.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 10:00:00 | 415.10 | 415.40 | 414.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 413.20 | 414.96 | 414.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 413.20 | 414.96 | 414.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 412.10 | 414.39 | 414.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:30:00 | 412.45 | 414.39 | 414.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 411.80 | 413.87 | 413.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 407.75 | 412.23 | 413.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 406.95 | 406.77 | 409.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 09:30:00 | 406.70 | 406.77 | 409.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 409.00 | 407.21 | 408.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 408.80 | 407.21 | 408.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 408.20 | 407.41 | 408.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 12:15:00 | 406.35 | 407.58 | 408.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 14:45:00 | 406.20 | 407.40 | 408.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 14:15:00 | 386.03 | 394.43 | 400.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 14:15:00 | 385.89 | 394.43 | 400.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 390.50 | 390.30 | 395.69 | SL hit (close>ema200) qty=0.50 sl=390.30 alert=retest2 |

### Cycle 173 — BUY (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 11:15:00 | 390.90 | 390.27 | 390.21 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 385.25 | 389.39 | 389.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 382.15 | 385.38 | 386.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 12:15:00 | 384.55 | 384.45 | 385.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 12:15:00 | 384.55 | 384.45 | 385.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 384.55 | 384.45 | 385.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 385.80 | 384.45 | 385.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 389.65 | 384.99 | 385.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 390.40 | 384.99 | 385.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 390.50 | 386.09 | 386.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:45:00 | 390.15 | 386.09 | 386.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 389.60 | 386.80 | 386.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 391.40 | 388.22 | 387.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 399.15 | 399.37 | 396.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 401.80 | 399.37 | 396.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 398.65 | 401.36 | 399.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 398.65 | 401.36 | 399.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 398.55 | 400.80 | 399.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 399.20 | 400.80 | 399.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 399.25 | 400.22 | 399.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 399.25 | 400.22 | 399.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 398.20 | 399.81 | 399.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:30:00 | 397.90 | 399.81 | 399.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 397.05 | 399.26 | 399.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 13:15:00 | 396.30 | 398.67 | 399.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 393.40 | 393.21 | 395.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 15:00:00 | 393.40 | 393.21 | 395.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 395.95 | 393.76 | 395.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 398.90 | 393.76 | 395.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 397.85 | 394.58 | 395.66 | EMA400 retest candle locked (from downside) |

### Cycle 177 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 399.20 | 396.76 | 396.51 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 12:15:00 | 395.55 | 396.78 | 396.81 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 14:15:00 | 398.15 | 396.83 | 396.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 404.85 | 398.60 | 397.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 411.05 | 412.28 | 409.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 411.05 | 412.28 | 409.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 412.15 | 412.47 | 410.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:45:00 | 411.25 | 412.47 | 410.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 415.70 | 417.05 | 414.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 415.70 | 417.05 | 414.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 415.65 | 416.77 | 414.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 419.80 | 416.77 | 414.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 10:15:00 | 418.50 | 417.97 | 416.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 10:15:00 | 413.15 | 417.00 | 416.52 | SL hit (close<static) qty=1.00 sl=414.50 alert=retest2 |

### Cycle 180 — SELL (started 2026-01-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 12:15:00 | 415.30 | 416.16 | 416.19 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 416.80 | 416.29 | 416.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 10:15:00 | 417.85 | 417.09 | 416.68 | Break + close above crossover candle high |

### Cycle 182 — SELL (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 11:15:00 | 413.00 | 416.28 | 416.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 12:15:00 | 412.00 | 415.42 | 415.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 417.95 | 414.90 | 415.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 417.95 | 414.90 | 415.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 417.95 | 414.90 | 415.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 417.95 | 414.90 | 415.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 420.05 | 415.93 | 415.82 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 10:15:00 | 412.15 | 416.03 | 416.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 11:15:00 | 411.65 | 415.15 | 415.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 09:15:00 | 412.90 | 412.00 | 413.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 412.90 | 412.00 | 413.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 412.90 | 412.00 | 413.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 09:15:00 | 402.30 | 410.54 | 411.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 12:15:00 | 414.65 | 409.53 | 409.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — BUY (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 12:15:00 | 414.65 | 409.53 | 409.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 417.45 | 411.93 | 410.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 413.30 | 414.05 | 412.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 12:00:00 | 413.30 | 414.05 | 412.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 411.15 | 413.47 | 411.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 411.15 | 413.47 | 411.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 410.00 | 412.77 | 411.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:30:00 | 408.90 | 412.77 | 411.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 413.70 | 413.09 | 412.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 11:00:00 | 413.70 | 413.09 | 412.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 412.45 | 413.24 | 412.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 14:00:00 | 412.45 | 413.24 | 412.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 415.40 | 413.67 | 412.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 15:15:00 | 417.70 | 413.67 | 412.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-01 09:15:00 | 459.47 | 449.75 | 443.01 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 186 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 426.05 | 438.51 | 439.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 421.95 | 435.20 | 438.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 432.60 | 431.01 | 434.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 14:00:00 | 432.60 | 431.01 | 434.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 440.30 | 432.87 | 434.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 440.30 | 432.87 | 434.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 440.80 | 434.46 | 435.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 445.65 | 434.46 | 435.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 439.95 | 436.02 | 436.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 11:15:00 | 439.40 | 436.02 | 436.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 438.75 | 436.56 | 436.31 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 432.55 | 437.56 | 437.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 427.40 | 432.48 | 434.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 435.55 | 431.15 | 432.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 435.55 | 431.15 | 432.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 435.55 | 431.15 | 432.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 435.55 | 431.15 | 432.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 439.10 | 432.74 | 433.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 439.10 | 432.74 | 433.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 438.85 | 433.96 | 433.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 09:15:00 | 441.75 | 437.54 | 436.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 12:15:00 | 441.40 | 441.66 | 439.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 12:15:00 | 441.40 | 441.66 | 439.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 441.40 | 441.66 | 439.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 13:00:00 | 441.40 | 441.66 | 439.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 440.45 | 441.42 | 440.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 14:00:00 | 440.45 | 441.42 | 440.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 435.65 | 440.26 | 439.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 15:00:00 | 435.65 | 440.26 | 439.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 435.00 | 439.21 | 439.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:15:00 | 434.50 | 439.21 | 439.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 435.30 | 438.43 | 438.84 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 443.00 | 438.20 | 438.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 447.70 | 441.55 | 439.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 15:15:00 | 446.90 | 447.26 | 444.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 09:15:00 | 445.35 | 447.26 | 444.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 443.25 | 446.46 | 444.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:45:00 | 443.30 | 446.46 | 444.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 442.80 | 445.73 | 444.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 442.90 | 445.73 | 444.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 439.25 | 443.16 | 443.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 435.70 | 441.67 | 442.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 09:15:00 | 441.90 | 440.65 | 442.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 441.90 | 440.65 | 442.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 441.90 | 440.65 | 442.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:45:00 | 444.25 | 440.65 | 442.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 443.75 | 441.27 | 442.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 13:00:00 | 440.80 | 441.67 | 442.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 15:00:00 | 441.15 | 441.61 | 442.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:00:00 | 441.25 | 441.44 | 441.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 15:15:00 | 439.50 | 438.05 | 437.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — BUY (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 15:15:00 | 439.50 | 438.05 | 437.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 441.45 | 438.73 | 438.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 11:15:00 | 443.55 | 444.91 | 442.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 11:15:00 | 443.55 | 444.91 | 442.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 443.55 | 444.91 | 442.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:30:00 | 443.75 | 444.91 | 442.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 444.95 | 445.13 | 443.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 444.95 | 445.13 | 443.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 443.00 | 444.70 | 443.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 453.00 | 444.70 | 443.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 452.15 | 446.19 | 444.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 10:30:00 | 457.55 | 452.10 | 450.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 12:15:00 | 453.25 | 459.21 | 459.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — SELL (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 12:15:00 | 453.25 | 459.21 | 459.45 | EMA200 below EMA400 |

### Cycle 195 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 463.15 | 458.92 | 458.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 463.95 | 460.57 | 459.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 455.20 | 460.00 | 459.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 12:15:00 | 455.20 | 460.00 | 459.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 455.20 | 460.00 | 459.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 455.20 | 460.00 | 459.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 455.55 | 459.11 | 459.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 453.65 | 458.02 | 458.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 14:15:00 | 454.40 | 453.81 | 455.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 15:00:00 | 454.40 | 453.81 | 455.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 440.20 | 433.86 | 436.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 440.20 | 433.86 | 436.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 439.50 | 434.99 | 436.96 | EMA400 retest candle locked (from downside) |

### Cycle 197 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 445.20 | 439.16 | 438.55 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 436.60 | 438.81 | 439.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 434.25 | 437.90 | 438.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 412.00 | 411.46 | 419.50 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:30:00 | 409.55 | 411.06 | 418.59 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 420.60 | 414.79 | 417.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 420.60 | 414.79 | 417.29 | SL hit (close>ema400) qty=1.00 sl=417.29 alert=retest1 |

### Cycle 199 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 422.35 | 411.67 | 411.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 09:15:00 | 428.40 | 425.27 | 421.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 424.10 | 425.04 | 421.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 424.10 | 425.04 | 421.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 438.40 | 432.64 | 428.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 11:30:00 | 441.20 | 435.85 | 431.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 443.55 | 438.63 | 434.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:30:00 | 441.65 | 441.79 | 439.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 13:15:00 | 449.80 | 455.26 | 455.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — SELL (started 2026-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 13:15:00 | 449.80 | 455.26 | 455.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 445.30 | 452.19 | 454.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 12:15:00 | 448.10 | 448.09 | 450.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-23 13:15:00 | 449.50 | 448.09 | 450.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 450.30 | 448.53 | 450.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 13:45:00 | 452.90 | 448.53 | 450.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 449.80 | 448.79 | 450.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:15:00 | 445.95 | 448.98 | 450.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 438.00 | 434.12 | 433.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 438.00 | 434.12 | 433.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 15:15:00 | 440.60 | 438.11 | 436.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 12:15:00 | 438.60 | 439.40 | 437.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 12:15:00 | 438.60 | 439.40 | 437.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 438.60 | 439.40 | 437.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 438.60 | 439.40 | 437.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 438.50 | 439.22 | 438.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 14:30:00 | 440.40 | 439.34 | 438.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-18 09:15:00 | 108.65 | 2023-05-18 14:15:00 | 107.80 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2023-05-18 10:15:00 | 108.40 | 2023-05-18 14:15:00 | 107.80 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2023-05-31 11:30:00 | 111.65 | 2023-06-12 09:15:00 | 122.82 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-06-27 10:45:00 | 119.30 | 2023-06-30 10:15:00 | 122.50 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2023-06-28 10:00:00 | 119.35 | 2023-06-30 10:15:00 | 122.50 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest1 | 2023-07-18 09:15:00 | 127.55 | 2023-07-18 11:15:00 | 125.35 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2023-07-24 11:30:00 | 126.05 | 2023-07-25 11:15:00 | 126.30 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2023-07-24 12:30:00 | 126.10 | 2023-07-25 11:15:00 | 126.30 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2023-07-24 13:30:00 | 125.95 | 2023-07-25 11:15:00 | 126.30 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2023-07-24 15:00:00 | 126.00 | 2023-07-25 11:15:00 | 126.30 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2023-08-11 09:15:00 | 130.20 | 2023-08-14 14:15:00 | 130.40 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2023-08-25 13:30:00 | 134.45 | 2023-08-25 14:15:00 | 133.35 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2023-08-28 09:15:00 | 135.20 | 2023-08-31 09:15:00 | 132.35 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2023-09-08 09:15:00 | 141.15 | 2023-09-12 09:15:00 | 136.30 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2023-09-14 13:45:00 | 135.65 | 2023-09-18 09:15:00 | 139.80 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2023-09-15 09:45:00 | 136.20 | 2023-09-18 09:15:00 | 139.80 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2023-09-15 10:15:00 | 136.15 | 2023-09-18 09:15:00 | 139.80 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2023-09-15 11:15:00 | 136.20 | 2023-09-18 09:15:00 | 139.80 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2023-09-22 14:15:00 | 135.45 | 2023-09-26 09:15:00 | 137.40 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2023-09-25 09:15:00 | 134.90 | 2023-09-26 09:15:00 | 137.40 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2023-10-04 14:45:00 | 138.55 | 2023-10-09 09:15:00 | 137.25 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2023-10-05 10:30:00 | 139.05 | 2023-10-09 09:15:00 | 137.25 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2023-10-20 09:30:00 | 136.40 | 2023-10-25 12:15:00 | 129.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 09:30:00 | 136.40 | 2023-10-26 15:15:00 | 129.30 | STOP_HIT | 0.50 | 5.21% |
| BUY | retest2 | 2023-11-02 09:15:00 | 133.95 | 2023-11-21 12:15:00 | 142.50 | STOP_HIT | 1.00 | 6.38% |
| SELL | retest2 | 2023-11-23 10:15:00 | 140.90 | 2023-11-28 14:15:00 | 141.25 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2023-11-24 12:45:00 | 140.60 | 2023-11-28 14:15:00 | 141.25 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2023-11-28 09:30:00 | 140.85 | 2023-11-28 14:15:00 | 141.25 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2023-11-28 12:30:00 | 140.90 | 2023-11-28 14:15:00 | 141.25 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2023-12-05 15:00:00 | 152.10 | 2023-12-15 09:15:00 | 167.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-08 09:15:00 | 186.55 | 2024-01-08 11:15:00 | 183.90 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-01-17 11:15:00 | 188.05 | 2024-01-17 12:15:00 | 184.95 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-02-02 12:15:00 | 184.90 | 2024-02-08 11:15:00 | 184.95 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2024-02-02 14:45:00 | 184.60 | 2024-02-08 11:15:00 | 184.95 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-02-05 10:00:00 | 184.60 | 2024-02-08 11:15:00 | 184.95 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-02-05 10:45:00 | 184.85 | 2024-02-08 11:15:00 | 184.95 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2024-02-07 11:00:00 | 181.10 | 2024-02-08 11:15:00 | 184.95 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-02-20 10:15:00 | 191.25 | 2024-02-27 09:15:00 | 208.67 | TARGET_HIT | 1.00 | 9.11% |
| BUY | retest2 | 2024-02-20 11:30:00 | 189.70 | 2024-02-27 09:15:00 | 208.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-22 09:15:00 | 189.70 | 2024-03-04 09:15:00 | 210.38 | TARGET_HIT | 1.00 | 10.90% |
| SELL | retest2 | 2024-03-15 10:15:00 | 185.65 | 2024-03-21 09:15:00 | 194.10 | STOP_HIT | 1.00 | -4.55% |
| SELL | retest2 | 2024-03-19 10:15:00 | 186.40 | 2024-03-21 09:15:00 | 194.10 | STOP_HIT | 1.00 | -4.13% |
| SELL | retest2 | 2024-03-19 11:15:00 | 186.35 | 2024-03-21 09:15:00 | 194.10 | STOP_HIT | 1.00 | -4.16% |
| SELL | retest2 | 2024-03-19 13:15:00 | 186.70 | 2024-03-21 09:15:00 | 194.10 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2024-03-28 09:15:00 | 200.70 | 2024-04-02 09:15:00 | 220.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-28 09:45:00 | 200.60 | 2024-04-02 09:15:00 | 220.66 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-28 11:00:00 | 201.70 | 2024-04-02 10:15:00 | 221.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-26 09:45:00 | 240.65 | 2024-04-29 12:15:00 | 235.95 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2024-04-26 11:15:00 | 240.15 | 2024-04-29 12:15:00 | 235.95 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-04-26 11:45:00 | 240.35 | 2024-04-29 12:15:00 | 235.95 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-05-03 11:15:00 | 234.15 | 2024-05-13 09:15:00 | 222.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 15:15:00 | 234.30 | 2024-05-13 09:15:00 | 222.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-06 09:30:00 | 232.90 | 2024-05-13 10:15:00 | 221.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-06 13:00:00 | 233.25 | 2024-05-13 10:15:00 | 221.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 11:15:00 | 234.15 | 2024-05-14 09:15:00 | 226.95 | STOP_HIT | 0.50 | 3.07% |
| SELL | retest2 | 2024-05-03 15:15:00 | 234.30 | 2024-05-14 09:15:00 | 226.95 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2024-05-06 09:30:00 | 232.90 | 2024-05-14 09:15:00 | 226.95 | STOP_HIT | 0.50 | 2.55% |
| SELL | retest2 | 2024-05-06 13:00:00 | 233.25 | 2024-05-14 09:15:00 | 226.95 | STOP_HIT | 0.50 | 2.70% |
| SELL | retest2 | 2024-05-07 11:00:00 | 229.85 | 2024-05-14 12:15:00 | 229.50 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2024-05-08 13:30:00 | 231.15 | 2024-05-14 12:15:00 | 229.50 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2024-05-09 10:30:00 | 229.90 | 2024-05-14 12:15:00 | 229.50 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2024-05-29 09:30:00 | 291.00 | 2024-05-31 09:15:00 | 287.75 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-05-29 11:15:00 | 290.85 | 2024-05-31 09:15:00 | 287.75 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-05-29 13:00:00 | 291.20 | 2024-05-31 09:15:00 | 287.75 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-05-30 15:15:00 | 291.55 | 2024-05-31 09:15:00 | 287.75 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-06-13 12:15:00 | 291.50 | 2024-06-18 11:15:00 | 320.65 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-28 10:15:00 | 304.95 | 2024-06-28 15:15:00 | 309.80 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-06-28 14:00:00 | 304.50 | 2024-06-28 15:15:00 | 309.80 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-06-28 14:30:00 | 304.75 | 2024-06-28 15:15:00 | 309.80 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest1 | 2024-07-09 14:45:00 | 334.85 | 2024-07-12 11:15:00 | 332.50 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest1 | 2024-07-10 09:15:00 | 338.00 | 2024-07-12 11:15:00 | 332.50 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-07-11 09:15:00 | 336.90 | 2024-07-15 12:15:00 | 331.90 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-07-11 11:00:00 | 335.25 | 2024-07-15 12:15:00 | 331.90 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-07-11 11:45:00 | 334.95 | 2024-07-15 12:15:00 | 331.90 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-07-11 12:15:00 | 334.90 | 2024-07-15 12:15:00 | 331.90 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-07-23 12:15:00 | 288.05 | 2024-07-26 11:15:00 | 308.75 | STOP_HIT | 1.00 | -7.19% |
| SELL | retest2 | 2024-08-06 13:30:00 | 290.25 | 2024-08-08 09:15:00 | 299.15 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2024-09-02 11:45:00 | 296.20 | 2024-09-04 10:15:00 | 299.80 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-09-02 15:15:00 | 296.85 | 2024-09-04 10:15:00 | 299.80 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-09-16 09:15:00 | 290.45 | 2024-09-17 09:15:00 | 286.10 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-09-16 12:15:00 | 290.80 | 2024-09-17 09:15:00 | 286.10 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-09-20 13:30:00 | 275.40 | 2024-09-23 09:15:00 | 282.60 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2024-09-20 14:15:00 | 276.30 | 2024-09-23 09:15:00 | 282.60 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-10-15 09:15:00 | 286.70 | 2024-10-16 14:15:00 | 285.70 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-10-16 13:00:00 | 286.35 | 2024-10-16 14:15:00 | 285.70 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-10-16 14:15:00 | 286.10 | 2024-10-16 14:15:00 | 285.70 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2024-10-18 09:15:00 | 278.10 | 2024-10-18 15:15:00 | 287.00 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2024-10-18 10:45:00 | 283.80 | 2024-10-18 15:15:00 | 287.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-10-28 12:45:00 | 268.05 | 2024-10-29 09:15:00 | 274.20 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-10-28 14:00:00 | 267.85 | 2024-10-29 09:15:00 | 274.20 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest1 | 2024-11-01 18:00:00 | 288.50 | 2024-11-04 09:15:00 | 284.15 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest1 | 2024-11-01 18:30:00 | 289.00 | 2024-11-04 09:15:00 | 284.15 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-11-11 11:00:00 | 302.80 | 2024-11-12 12:15:00 | 295.10 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2024-11-11 14:15:00 | 300.40 | 2024-11-12 12:15:00 | 295.10 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-11-18 14:15:00 | 278.45 | 2024-11-22 13:15:00 | 282.50 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-11-19 15:00:00 | 278.55 | 2024-11-22 13:15:00 | 282.50 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-12-02 14:15:00 | 306.95 | 2024-12-12 11:15:00 | 313.50 | STOP_HIT | 1.00 | 2.13% |
| SELL | retest2 | 2024-12-24 13:15:00 | 292.75 | 2024-12-31 15:15:00 | 293.50 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2024-12-26 11:30:00 | 293.00 | 2024-12-31 15:15:00 | 293.50 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2024-12-27 10:45:00 | 292.35 | 2024-12-31 15:15:00 | 293.50 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2024-12-27 14:45:00 | 291.50 | 2024-12-31 15:15:00 | 293.50 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-01-02 13:15:00 | 294.25 | 2025-01-03 14:15:00 | 292.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-01-15 14:15:00 | 266.70 | 2025-01-16 09:15:00 | 274.00 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2025-01-15 14:45:00 | 266.90 | 2025-01-16 09:15:00 | 274.00 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-01-24 14:30:00 | 269.30 | 2025-01-28 09:15:00 | 255.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 14:30:00 | 269.30 | 2025-01-29 09:15:00 | 265.60 | STOP_HIT | 0.50 | 1.37% |
| SELL | retest2 | 2025-01-27 09:15:00 | 264.20 | 2025-01-29 14:15:00 | 267.45 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-02-13 14:30:00 | 261.30 | 2025-02-14 13:15:00 | 248.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 15:15:00 | 261.15 | 2025-02-17 09:15:00 | 248.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 14:30:00 | 261.30 | 2025-02-17 14:15:00 | 248.70 | STOP_HIT | 0.50 | 4.82% |
| SELL | retest2 | 2025-02-13 15:15:00 | 261.15 | 2025-02-17 14:15:00 | 248.70 | STOP_HIT | 0.50 | 4.77% |
| BUY | retest2 | 2025-02-24 11:15:00 | 255.50 | 2025-02-27 10:15:00 | 254.50 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-02-24 14:30:00 | 256.35 | 2025-02-27 10:15:00 | 254.50 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest1 | 2025-03-07 09:15:00 | 277.46 | 2025-03-10 14:15:00 | 272.67 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-03-11 14:15:00 | 276.95 | 2025-03-24 10:15:00 | 304.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-11 14:45:00 | 276.68 | 2025-03-24 10:15:00 | 304.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-12 09:30:00 | 276.93 | 2025-03-24 10:15:00 | 304.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-12 10:15:00 | 276.68 | 2025-03-24 10:15:00 | 304.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-17 13:30:00 | 279.31 | 2025-04-01 11:15:00 | 298.65 | STOP_HIT | 1.00 | 6.92% |
| BUY | retest2 | 2025-03-17 14:45:00 | 279.67 | 2025-04-01 11:15:00 | 298.65 | STOP_HIT | 1.00 | 6.79% |
| SELL | retest2 | 2025-04-04 14:00:00 | 280.25 | 2025-04-07 09:15:00 | 266.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 14:00:00 | 280.25 | 2025-04-07 15:15:00 | 274.00 | STOP_HIT | 0.50 | 2.23% |
| SELL | retest2 | 2025-04-07 09:15:00 | 268.30 | 2025-04-08 15:15:00 | 282.10 | STOP_HIT | 1.00 | -5.14% |
| SELL | retest2 | 2025-04-08 10:30:00 | 278.40 | 2025-04-08 15:15:00 | 282.10 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-04-08 13:30:00 | 279.95 | 2025-04-08 15:15:00 | 282.10 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-05-08 13:45:00 | 308.50 | 2025-05-09 09:15:00 | 318.45 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2025-05-21 09:30:00 | 375.40 | 2025-05-30 14:15:00 | 384.65 | STOP_HIT | 1.00 | 2.46% |
| BUY | retest2 | 2025-06-06 15:15:00 | 390.90 | 2025-06-12 10:15:00 | 393.00 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2025-06-18 10:15:00 | 404.65 | 2025-06-19 12:15:00 | 395.65 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-07-02 14:30:00 | 425.20 | 2025-07-07 09:15:00 | 418.90 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-07-03 09:45:00 | 429.50 | 2025-07-07 09:15:00 | 418.90 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-07-15 11:45:00 | 409.60 | 2025-07-22 14:15:00 | 402.95 | STOP_HIT | 1.00 | 1.62% |
| SELL | retest2 | 2025-07-30 10:15:00 | 385.95 | 2025-08-04 13:15:00 | 389.25 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-07-30 15:15:00 | 385.65 | 2025-08-04 13:15:00 | 389.25 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-07-31 13:45:00 | 385.40 | 2025-08-04 13:15:00 | 389.25 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-08-06 10:15:00 | 389.50 | 2025-08-07 13:15:00 | 385.15 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-08-06 12:15:00 | 389.00 | 2025-08-07 13:15:00 | 385.15 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-08-06 15:00:00 | 390.00 | 2025-08-07 13:15:00 | 385.15 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-08-11 12:30:00 | 382.35 | 2025-08-13 12:15:00 | 385.65 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-08-12 09:15:00 | 382.10 | 2025-08-13 12:15:00 | 385.65 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-08-12 10:00:00 | 382.35 | 2025-08-13 12:15:00 | 385.65 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-08-12 14:30:00 | 381.70 | 2025-08-13 12:15:00 | 385.65 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-08-18 09:45:00 | 386.35 | 2025-08-18 11:15:00 | 382.25 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-08-20 10:30:00 | 374.05 | 2025-09-01 09:15:00 | 371.55 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2025-08-21 11:45:00 | 374.70 | 2025-09-01 09:15:00 | 371.55 | STOP_HIT | 1.00 | 0.84% |
| SELL | retest2 | 2025-08-21 14:00:00 | 375.30 | 2025-09-01 09:15:00 | 371.55 | STOP_HIT | 1.00 | 1.00% |
| SELL | retest2 | 2025-08-22 15:00:00 | 374.70 | 2025-09-01 09:15:00 | 371.55 | STOP_HIT | 1.00 | 0.84% |
| SELL | retest2 | 2025-08-25 14:15:00 | 372.55 | 2025-09-01 09:15:00 | 371.55 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-09-09 11:45:00 | 370.70 | 2025-09-10 09:15:00 | 378.95 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-10-06 13:30:00 | 413.95 | 2025-10-08 10:15:00 | 403.95 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-10-06 14:30:00 | 413.90 | 2025-10-08 10:15:00 | 403.95 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-10-07 09:15:00 | 414.20 | 2025-10-08 10:15:00 | 403.95 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-10-20 09:15:00 | 415.30 | 2025-10-27 12:15:00 | 415.30 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-11-17 12:45:00 | 424.45 | 2025-11-18 14:15:00 | 420.65 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-11-17 13:30:00 | 424.20 | 2025-11-18 14:15:00 | 420.65 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-12-05 12:15:00 | 406.35 | 2025-12-08 14:15:00 | 386.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 14:45:00 | 406.20 | 2025-12-08 14:15:00 | 385.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 12:15:00 | 406.35 | 2025-12-09 12:15:00 | 390.50 | STOP_HIT | 0.50 | 3.90% |
| SELL | retest2 | 2025-12-05 14:45:00 | 406.20 | 2025-12-09 12:15:00 | 390.50 | STOP_HIT | 0.50 | 3.87% |
| BUY | retest2 | 2026-01-09 09:15:00 | 419.80 | 2026-01-12 10:15:00 | 413.15 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-01-12 10:15:00 | 418.50 | 2026-01-12 10:15:00 | 413.15 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-01-21 09:15:00 | 402.30 | 2026-01-22 12:15:00 | 414.65 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2026-01-27 15:15:00 | 417.70 | 2026-02-01 09:15:00 | 459.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-01 12:30:00 | 431.80 | 2026-02-01 14:15:00 | 426.05 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-02-20 13:00:00 | 440.80 | 2026-02-25 15:15:00 | 439.50 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2026-02-20 15:00:00 | 441.15 | 2026-02-25 15:15:00 | 439.50 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2026-02-23 10:00:00 | 441.25 | 2026-02-25 15:15:00 | 439.50 | STOP_HIT | 1.00 | 0.40% |
| BUY | retest2 | 2026-03-05 10:30:00 | 457.55 | 2026-03-09 12:15:00 | 453.25 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest1 | 2026-03-24 10:30:00 | 409.55 | 2026-03-25 09:15:00 | 420.60 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2026-03-25 14:45:00 | 414.60 | 2026-04-01 09:15:00 | 426.90 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2026-04-09 11:30:00 | 441.20 | 2026-04-21 13:15:00 | 449.80 | STOP_HIT | 1.00 | 1.95% |
| BUY | retest2 | 2026-04-10 09:15:00 | 443.55 | 2026-04-21 13:15:00 | 449.80 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2026-04-13 11:30:00 | 441.65 | 2026-04-21 13:15:00 | 449.80 | STOP_HIT | 1.00 | 1.85% |
| SELL | retest2 | 2026-04-24 09:15:00 | 445.95 | 2026-05-06 09:15:00 | 438.00 | STOP_HIT | 1.00 | 1.78% |
