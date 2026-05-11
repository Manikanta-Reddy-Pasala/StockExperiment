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
| CROSSOVER | 233 |
| ALERT1 | 147 |
| ALERT2 | 145 |
| ALERT2_SKIP | 74 |
| ALERT3 | 357 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 151 |
| PARTIAL | 18 |
| TARGET_HIT | 11 |
| STOP_HIT | 148 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 177 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 65 / 112
- **Target hits / Stop hits / Partials:** 11 / 148 / 18
- **Avg / median % per leg:** 0.21% / -1.01%
- **Sum % (uncompounded):** 37.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 89 | 23 | 25.8% | 7 | 82 | 0 | -0.30% | -27.0% |
| BUY @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 6 | 0 | -1.19% | -7.1% |
| BUY @ 3rd Alert (retest2) | 83 | 22 | 26.5% | 7 | 76 | 0 | -0.24% | -19.8% |
| SELL (all) | 88 | 42 | 47.7% | 4 | 66 | 18 | 0.73% | 64.1% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.33% | 1.3% |
| SELL @ 3rd Alert (retest2) | 84 | 40 | 47.6% | 4 | 63 | 17 | 0.75% | 62.7% |
| retest1 (combined) | 10 | 3 | 30.0% | 0 | 9 | 1 | -0.58% | -5.8% |
| retest2 (combined) | 167 | 62 | 37.1% | 11 | 139 | 17 | 0.26% | 42.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 15:15:00 | 154.45 | 155.59 | 155.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 11:15:00 | 153.50 | 154.98 | 155.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 12:15:00 | 157.85 | 155.56 | 155.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 12:15:00 | 157.85 | 155.56 | 155.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 12:15:00 | 157.85 | 155.56 | 155.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-17 12:30:00 | 159.20 | 155.56 | 155.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-05-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 13:15:00 | 164.60 | 157.37 | 156.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 14:15:00 | 167.70 | 159.43 | 157.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-18 15:15:00 | 165.55 | 165.86 | 162.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 09:15:00 | 161.95 | 165.08 | 162.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 09:15:00 | 161.95 | 165.08 | 162.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 10:00:00 | 161.95 | 165.08 | 162.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 10:15:00 | 162.35 | 164.53 | 162.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 10:30:00 | 162.20 | 164.53 | 162.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 11:15:00 | 164.80 | 164.59 | 162.90 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-05-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 09:15:00 | 162.50 | 163.79 | 163.94 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 14:15:00 | 165.10 | 164.15 | 164.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 10:15:00 | 166.35 | 164.62 | 164.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 13:15:00 | 166.60 | 166.99 | 166.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-29 14:00:00 | 166.60 | 166.99 | 166.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 14:15:00 | 166.20 | 166.84 | 166.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-29 14:45:00 | 166.25 | 166.84 | 166.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 15:15:00 | 166.00 | 166.67 | 166.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 09:30:00 | 165.30 | 166.28 | 166.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2023-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 10:15:00 | 163.75 | 165.77 | 165.85 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 10:15:00 | 167.75 | 165.51 | 165.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 13:15:00 | 169.20 | 166.98 | 166.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 09:15:00 | 167.15 | 168.50 | 167.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 09:15:00 | 167.15 | 168.50 | 167.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 167.15 | 168.50 | 167.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 10:00:00 | 167.15 | 168.50 | 167.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 10:15:00 | 167.00 | 168.20 | 167.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 11:00:00 | 167.00 | 168.20 | 167.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2023-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 15:15:00 | 167.10 | 167.41 | 167.42 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 09:15:00 | 169.85 | 167.90 | 167.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 10:15:00 | 173.55 | 169.03 | 168.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 15:15:00 | 169.95 | 170.14 | 169.18 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-06 09:15:00 | 171.45 | 170.14 | 169.18 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 10:15:00 | 170.10 | 170.21 | 169.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 10:30:00 | 169.50 | 170.21 | 169.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 11:15:00 | 169.35 | 170.04 | 169.38 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-06-06 11:15:00 | 169.35 | 170.04 | 169.38 | SL hit (close<ema400) qty=1.00 sl=169.38 alert=retest1 |

### Cycle 9 — SELL (started 2023-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-07 10:15:00 | 167.25 | 169.09 | 169.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-07 11:15:00 | 166.65 | 168.60 | 168.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 13:15:00 | 163.70 | 163.57 | 164.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-12 14:00:00 | 163.70 | 163.57 | 164.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 163.60 | 163.42 | 164.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-13 11:00:00 | 163.00 | 163.34 | 164.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-20 10:15:00 | 161.80 | 161.00 | 161.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2023-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 10:15:00 | 161.80 | 161.00 | 161.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 09:15:00 | 171.90 | 163.50 | 162.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 11:15:00 | 168.75 | 170.07 | 167.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-22 12:00:00 | 168.75 | 170.07 | 167.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 12:15:00 | 167.15 | 169.48 | 167.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 13:00:00 | 167.15 | 169.48 | 167.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 13:15:00 | 167.60 | 169.11 | 167.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 14:15:00 | 166.75 | 169.11 | 167.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 14:15:00 | 167.25 | 168.73 | 167.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 14:30:00 | 167.00 | 168.73 | 167.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 15:15:00 | 167.50 | 168.49 | 167.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 09:15:00 | 165.40 | 168.49 | 167.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 166.50 | 168.09 | 167.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 09:45:00 | 166.70 | 168.09 | 167.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 10:15:00 | 167.20 | 167.91 | 167.30 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-26 09:15:00 | 164.50 | 167.04 | 167.12 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 12:15:00 | 166.40 | 165.70 | 165.70 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-06-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 15:15:00 | 165.10 | 165.70 | 165.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-03 09:15:00 | 164.90 | 165.54 | 165.64 | Break + close below crossover candle low |

### Cycle 14 — BUY (started 2023-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 10:15:00 | 168.10 | 166.05 | 165.87 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 11:15:00 | 165.30 | 166.02 | 166.10 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 10:15:00 | 169.00 | 166.40 | 166.16 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 11:15:00 | 165.75 | 167.35 | 167.47 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 09:15:00 | 167.85 | 167.19 | 167.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 12:15:00 | 170.35 | 168.36 | 167.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 13:15:00 | 168.70 | 170.78 | 169.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 13:15:00 | 168.70 | 170.78 | 169.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 13:15:00 | 168.70 | 170.78 | 169.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 14:00:00 | 168.70 | 170.78 | 169.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 14:15:00 | 170.05 | 170.63 | 169.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 09:30:00 | 171.15 | 170.37 | 169.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 11:00:00 | 171.20 | 170.53 | 169.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-21 12:15:00 | 176.45 | 177.50 | 177.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2023-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 12:15:00 | 176.45 | 177.50 | 177.54 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 13:15:00 | 180.50 | 178.10 | 177.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 14:15:00 | 184.30 | 179.34 | 178.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-25 10:15:00 | 184.35 | 184.58 | 182.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-25 10:30:00 | 184.45 | 184.58 | 182.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 11:15:00 | 182.95 | 184.25 | 182.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 11:45:00 | 182.65 | 184.25 | 182.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 12:15:00 | 182.60 | 183.92 | 182.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 13:15:00 | 182.45 | 183.92 | 182.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 13:15:00 | 183.50 | 183.84 | 182.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 13:30:00 | 182.50 | 183.84 | 182.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 14:15:00 | 182.55 | 183.58 | 182.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 15:00:00 | 182.55 | 183.58 | 182.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 15:15:00 | 182.00 | 183.26 | 182.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 09:15:00 | 183.50 | 183.26 | 182.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 10:00:00 | 182.60 | 183.13 | 182.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 10:45:00 | 182.60 | 182.84 | 182.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-26 14:15:00 | 180.75 | 182.11 | 182.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 14:15:00 | 180.75 | 182.11 | 182.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-26 15:15:00 | 179.80 | 181.65 | 182.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-27 09:15:00 | 182.10 | 181.74 | 182.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 09:15:00 | 182.10 | 181.74 | 182.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 182.10 | 181.74 | 182.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 09:30:00 | 183.60 | 181.74 | 182.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 10:15:00 | 181.75 | 181.74 | 182.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 11:15:00 | 181.45 | 181.74 | 182.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 11:45:00 | 181.40 | 181.57 | 181.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-31 11:00:00 | 181.50 | 179.98 | 180.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-31 11:45:00 | 180.85 | 180.14 | 180.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 14:15:00 | 181.15 | 180.22 | 180.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 14:30:00 | 181.40 | 180.22 | 180.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-07-31 15:15:00 | 181.35 | 180.45 | 180.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2023-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 15:15:00 | 181.35 | 180.45 | 180.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 09:15:00 | 184.05 | 181.17 | 180.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 13:15:00 | 177.35 | 181.89 | 181.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 13:15:00 | 177.35 | 181.89 | 181.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 13:15:00 | 177.35 | 181.89 | 181.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 14:00:00 | 177.35 | 181.89 | 181.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2023-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 14:15:00 | 171.60 | 179.83 | 180.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 168.10 | 172.51 | 175.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 11:15:00 | 171.60 | 170.74 | 173.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-03 11:45:00 | 171.80 | 170.74 | 173.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 172.15 | 171.17 | 172.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 13:15:00 | 171.55 | 171.68 | 172.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 13:45:00 | 171.05 | 171.52 | 172.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-09 09:15:00 | 173.00 | 170.61 | 170.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2023-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 09:15:00 | 173.00 | 170.61 | 170.55 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-08-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 13:15:00 | 169.25 | 170.72 | 170.82 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 09:15:00 | 174.70 | 171.32 | 171.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-11 14:15:00 | 176.50 | 173.90 | 172.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-14 15:15:00 | 175.25 | 175.63 | 174.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-16 09:15:00 | 175.35 | 175.63 | 174.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 174.85 | 175.47 | 174.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 09:30:00 | 174.20 | 175.47 | 174.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 10:15:00 | 174.50 | 175.28 | 174.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 10:30:00 | 174.45 | 175.28 | 174.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 11:15:00 | 172.95 | 174.81 | 174.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 12:00:00 | 172.95 | 174.81 | 174.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 12:15:00 | 172.50 | 174.35 | 174.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 13:00:00 | 172.50 | 174.35 | 174.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2023-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-16 13:15:00 | 172.50 | 173.98 | 174.02 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 15:15:00 | 174.65 | 174.15 | 174.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 09:15:00 | 176.75 | 174.67 | 174.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 13:15:00 | 214.30 | 215.09 | 206.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-24 14:00:00 | 214.30 | 215.09 | 206.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 10:15:00 | 212.25 | 214.21 | 209.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 10:30:00 | 210.45 | 214.21 | 209.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 247.90 | 250.54 | 246.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:30:00 | 249.10 | 250.54 | 246.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 11:15:00 | 245.80 | 249.14 | 246.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 11:45:00 | 245.50 | 249.14 | 246.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 12:15:00 | 246.40 | 248.59 | 246.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-31 13:30:00 | 248.30 | 249.01 | 246.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-04 11:15:00 | 245.65 | 248.24 | 248.09 | SL hit (close<static) qty=1.00 sl=245.75 alert=retest2 |

### Cycle 29 — SELL (started 2023-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 12:15:00 | 242.65 | 247.12 | 247.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-04 13:15:00 | 241.65 | 246.03 | 247.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-05 10:15:00 | 253.10 | 244.92 | 245.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 10:15:00 | 253.10 | 244.92 | 245.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 10:15:00 | 253.10 | 244.92 | 245.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-05 11:00:00 | 253.10 | 244.92 | 245.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2023-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 11:15:00 | 259.10 | 247.76 | 247.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 11:15:00 | 265.95 | 257.86 | 253.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 09:15:00 | 261.50 | 261.51 | 257.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-07 09:30:00 | 262.15 | 261.51 | 257.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 258.25 | 260.84 | 259.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 09:45:00 | 259.10 | 260.84 | 259.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 10:15:00 | 257.25 | 260.12 | 258.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 10:30:00 | 256.70 | 260.12 | 258.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 12:15:00 | 257.65 | 259.35 | 258.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 12:30:00 | 257.80 | 259.35 | 258.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 13:15:00 | 257.20 | 258.92 | 258.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 13:45:00 | 257.75 | 258.92 | 258.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 14:15:00 | 257.75 | 258.69 | 258.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 09:15:00 | 268.30 | 258.45 | 258.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-11 14:15:00 | 295.13 | 276.73 | 268.60 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2023-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 15:15:00 | 263.95 | 267.52 | 267.78 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-09-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 12:15:00 | 280.30 | 270.08 | 268.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-13 14:15:00 | 291.10 | 275.95 | 271.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-14 11:15:00 | 279.50 | 282.85 | 276.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 11:15:00 | 279.50 | 282.85 | 276.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 11:15:00 | 279.50 | 282.85 | 276.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-14 11:45:00 | 278.30 | 282.85 | 276.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 12:15:00 | 280.65 | 282.41 | 277.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-14 12:30:00 | 277.65 | 282.41 | 277.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 15:15:00 | 281.00 | 281.66 | 278.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-15 09:15:00 | 286.40 | 281.66 | 278.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-21 14:15:00 | 284.00 | 287.73 | 287.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-09-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 14:15:00 | 284.00 | 287.73 | 287.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 09:15:00 | 269.70 | 283.28 | 285.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 13:15:00 | 269.55 | 268.96 | 273.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-25 14:00:00 | 269.55 | 268.96 | 273.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 274.50 | 269.94 | 273.13 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 09:15:00 | 277.05 | 273.96 | 273.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 12:15:00 | 284.80 | 277.54 | 275.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 13:15:00 | 282.65 | 283.48 | 280.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-28 14:00:00 | 282.65 | 283.48 | 280.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 12:15:00 | 316.60 | 323.85 | 315.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 13:00:00 | 316.60 | 323.85 | 315.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 324.85 | 327.65 | 323.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-06 09:30:00 | 322.15 | 327.65 | 323.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 10:15:00 | 324.70 | 327.06 | 323.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-06 11:15:00 | 325.00 | 327.06 | 323.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 11:15:00 | 327.35 | 327.12 | 324.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-06 13:30:00 | 333.70 | 328.36 | 325.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-09 10:15:00 | 322.50 | 326.74 | 325.49 | SL hit (close<static) qty=1.00 sl=323.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 12:15:00 | 319.65 | 324.48 | 324.63 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 09:15:00 | 355.60 | 328.61 | 326.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 10:15:00 | 357.90 | 334.47 | 329.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 15:15:00 | 407.00 | 407.13 | 396.62 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 09:15:00 | 415.80 | 407.13 | 396.62 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 410.40 | 415.56 | 412.45 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-10-18 09:15:00 | 410.40 | 415.56 | 412.45 | SL hit (close<ema400) qty=1.00 sl=412.45 alert=retest1 |

### Cycle 37 — SELL (started 2023-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 11:15:00 | 399.55 | 409.83 | 410.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 12:15:00 | 383.95 | 398.08 | 402.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 15:15:00 | 360.15 | 359.76 | 369.80 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 09:15:00 | 349.25 | 359.76 | 369.80 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 360.00 | 356.30 | 363.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-26 14:45:00 | 362.10 | 356.30 | 363.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 15:15:00 | 363.00 | 357.64 | 363.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:15:00 | 371.65 | 357.64 | 363.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 366.80 | 359.47 | 363.38 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-10-27 09:15:00 | 366.80 | 359.47 | 363.38 | SL hit (close>ema400) qty=1.00 sl=363.38 alert=retest1 |

### Cycle 38 — BUY (started 2023-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 14:15:00 | 367.90 | 365.68 | 365.43 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 09:15:00 | 361.60 | 364.93 | 365.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 11:15:00 | 348.25 | 357.11 | 360.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-03 09:15:00 | 331.30 | 330.05 | 339.90 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-03 15:00:00 | 326.75 | 328.74 | 335.47 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 09:15:00 | 310.41 | 320.08 | 326.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 10:15:00 | 318.70 | 319.80 | 325.66 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-11-07 13:15:00 | 319.20 | 319.00 | 323.79 | SL hit (close>ema200) qty=0.50 sl=319.00 alert=retest1 |

### Cycle 40 — BUY (started 2023-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 10:15:00 | 345.45 | 326.80 | 325.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 11:15:00 | 354.25 | 332.29 | 328.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-10 10:15:00 | 361.40 | 362.44 | 353.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-10 11:00:00 | 361.40 | 362.44 | 353.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 09:15:00 | 396.65 | 390.31 | 384.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-22 10:15:00 | 413.00 | 394.17 | 392.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-24 13:15:00 | 401.60 | 408.16 | 408.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2023-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 13:15:00 | 401.60 | 408.16 | 408.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 14:15:00 | 386.50 | 403.83 | 406.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 11:15:00 | 401.05 | 399.47 | 403.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 11:15:00 | 401.05 | 399.47 | 403.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 11:15:00 | 401.05 | 399.47 | 403.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 11:45:00 | 402.85 | 399.47 | 403.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 13:15:00 | 406.30 | 400.78 | 403.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 13:45:00 | 408.85 | 400.78 | 403.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 14:15:00 | 406.50 | 401.93 | 403.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 15:00:00 | 406.50 | 401.93 | 403.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2023-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 10:15:00 | 407.00 | 404.67 | 404.59 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-29 11:15:00 | 403.95 | 404.53 | 404.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-29 12:15:00 | 402.80 | 404.18 | 404.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 09:15:00 | 402.50 | 401.89 | 403.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 09:15:00 | 402.50 | 401.89 | 403.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 402.50 | 401.89 | 403.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-30 10:15:00 | 406.25 | 401.89 | 403.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 10:15:00 | 407.00 | 402.91 | 403.38 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-11-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 11:15:00 | 408.00 | 403.93 | 403.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 14:15:00 | 412.50 | 406.31 | 404.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 11:15:00 | 408.90 | 409.51 | 407.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-01 12:00:00 | 408.90 | 409.51 | 407.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 12:15:00 | 426.90 | 412.99 | 408.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 09:15:00 | 441.50 | 415.26 | 411.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 09:15:00 | 435.70 | 428.58 | 421.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 10:45:00 | 432.50 | 429.42 | 423.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 12:00:00 | 432.05 | 429.95 | 423.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 14:15:00 | 423.90 | 428.59 | 424.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 15:00:00 | 423.90 | 428.59 | 424.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 15:15:00 | 429.00 | 428.67 | 425.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 09:30:00 | 423.40 | 428.35 | 425.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 10:15:00 | 425.45 | 427.77 | 425.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 11:00:00 | 425.45 | 427.77 | 425.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 11:15:00 | 430.50 | 428.31 | 425.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 11:30:00 | 426.50 | 428.31 | 425.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 12:15:00 | 424.50 | 427.55 | 425.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 13:00:00 | 424.50 | 427.55 | 425.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 13:15:00 | 432.95 | 428.63 | 426.39 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-12-07 14:15:00 | 422.85 | 426.02 | 426.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2023-12-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 14:15:00 | 422.85 | 426.02 | 426.34 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 11:15:00 | 433.40 | 427.54 | 426.84 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 13:15:00 | 415.90 | 424.78 | 425.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 14:15:00 | 405.10 | 415.05 | 419.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 09:15:00 | 409.40 | 405.62 | 410.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 09:15:00 | 409.40 | 405.62 | 410.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 409.40 | 405.62 | 410.66 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2023-12-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 09:15:00 | 416.60 | 413.34 | 412.90 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2023-12-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-14 13:15:00 | 409.10 | 412.31 | 412.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-15 10:15:00 | 407.00 | 410.85 | 411.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-19 09:15:00 | 409.30 | 405.82 | 407.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 09:15:00 | 409.30 | 405.82 | 407.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 409.30 | 405.82 | 407.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 09:30:00 | 409.30 | 405.82 | 407.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 414.10 | 407.48 | 407.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:30:00 | 415.85 | 407.48 | 407.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2023-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 11:15:00 | 413.80 | 408.74 | 408.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-19 13:15:00 | 417.00 | 411.09 | 409.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 11:15:00 | 412.35 | 413.05 | 411.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 11:15:00 | 412.35 | 413.05 | 411.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 11:15:00 | 412.35 | 413.05 | 411.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 11:45:00 | 412.75 | 413.05 | 411.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 406.00 | 411.64 | 410.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:00:00 | 406.00 | 411.64 | 410.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 391.75 | 407.66 | 409.03 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2023-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 11:15:00 | 409.65 | 405.16 | 404.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 10:15:00 | 411.85 | 407.45 | 406.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 12:15:00 | 410.75 | 411.91 | 409.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 12:15:00 | 410.75 | 411.91 | 409.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 410.75 | 411.91 | 409.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 12:30:00 | 410.15 | 411.91 | 409.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 13:15:00 | 410.80 | 411.69 | 409.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 13:30:00 | 410.40 | 411.69 | 409.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 14:15:00 | 409.85 | 411.32 | 409.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 14:45:00 | 410.00 | 411.32 | 409.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 15:15:00 | 410.90 | 411.24 | 409.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-28 10:15:00 | 413.45 | 410.80 | 409.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-28 11:00:00 | 412.50 | 411.14 | 410.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-28 13:45:00 | 418.30 | 411.28 | 410.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-29 09:15:00 | 406.55 | 410.02 | 410.00 | SL hit (close<static) qty=1.00 sl=407.40 alert=retest2 |

### Cycle 53 — SELL (started 2023-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 10:15:00 | 407.20 | 409.46 | 409.75 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 10:15:00 | 414.15 | 409.67 | 409.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 11:15:00 | 430.40 | 413.81 | 411.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 10:15:00 | 425.95 | 426.54 | 420.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-02 10:30:00 | 425.95 | 426.54 | 420.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 10:15:00 | 471.70 | 476.90 | 472.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 10:45:00 | 469.50 | 476.90 | 472.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 11:15:00 | 470.50 | 475.62 | 472.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 11:30:00 | 469.65 | 475.62 | 472.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 12:15:00 | 469.85 | 474.46 | 471.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 13:00:00 | 469.85 | 474.46 | 471.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2024-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 10:15:00 | 468.05 | 470.24 | 470.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 11:15:00 | 466.10 | 469.41 | 470.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 09:15:00 | 475.65 | 464.76 | 465.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 09:15:00 | 475.65 | 464.76 | 465.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 475.65 | 464.76 | 465.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 10:00:00 | 475.65 | 464.76 | 465.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2024-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 10:15:00 | 473.95 | 466.60 | 466.44 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 12:15:00 | 459.60 | 466.06 | 466.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 452.90 | 461.26 | 463.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 464.50 | 459.26 | 460.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 464.50 | 459.26 | 460.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 464.50 | 459.26 | 460.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 09:45:00 | 464.60 | 459.26 | 460.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 10:15:00 | 463.60 | 460.12 | 461.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 10:30:00 | 464.45 | 460.12 | 461.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2024-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 12:15:00 | 469.65 | 462.41 | 461.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-20 10:15:00 | 481.40 | 471.28 | 466.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 09:15:00 | 465.30 | 473.06 | 470.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 465.30 | 473.06 | 470.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 465.30 | 473.06 | 470.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:00:00 | 465.30 | 473.06 | 470.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 463.65 | 471.18 | 469.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:30:00 | 461.45 | 471.18 | 469.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2024-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 11:15:00 | 456.00 | 468.14 | 468.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 14:15:00 | 440.50 | 459.64 | 464.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 13:15:00 | 450.20 | 449.06 | 455.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 14:00:00 | 450.20 | 449.06 | 455.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 459.00 | 451.04 | 456.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 15:00:00 | 459.00 | 451.04 | 456.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 460.00 | 452.84 | 456.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:45:00 | 459.90 | 454.36 | 456.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 457.25 | 454.94 | 456.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 10:30:00 | 459.40 | 454.94 | 456.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 11:15:00 | 459.10 | 455.77 | 457.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 11:30:00 | 460.20 | 455.77 | 457.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 13:15:00 | 457.55 | 456.44 | 457.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 14:00:00 | 457.55 | 456.44 | 457.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 14:15:00 | 458.05 | 456.76 | 457.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 14:30:00 | 457.00 | 456.76 | 457.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 15:15:00 | 458.60 | 457.13 | 457.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 09:15:00 | 461.45 | 457.13 | 457.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2024-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 09:15:00 | 467.15 | 459.13 | 458.24 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 12:15:00 | 454.85 | 459.56 | 459.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 14:15:00 | 450.55 | 457.15 | 458.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 09:15:00 | 457.15 | 456.41 | 458.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-31 09:30:00 | 456.70 | 456.41 | 458.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 10:15:00 | 454.50 | 456.02 | 457.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-31 11:15:00 | 454.00 | 456.02 | 457.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-31 11:45:00 | 453.10 | 456.08 | 457.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-31 13:15:00 | 453.05 | 455.74 | 457.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-31 14:15:00 | 453.10 | 455.44 | 457.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 14:15:00 | 458.05 | 455.97 | 457.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-31 15:00:00 | 458.05 | 455.97 | 457.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 15:15:00 | 457.00 | 456.17 | 457.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-01 09:15:00 | 457.05 | 456.17 | 457.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 456.85 | 456.31 | 457.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-01 09:30:00 | 457.90 | 456.31 | 457.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 10:15:00 | 453.95 | 455.84 | 456.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-01 11:45:00 | 451.70 | 455.25 | 456.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-01 15:00:00 | 453.15 | 453.92 | 455.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-02 09:15:00 | 469.75 | 456.78 | 456.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2024-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 09:15:00 | 469.75 | 456.78 | 456.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-02 10:15:00 | 477.25 | 460.87 | 458.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-06 10:15:00 | 489.00 | 489.40 | 480.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-06 10:30:00 | 491.20 | 489.40 | 480.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 11:15:00 | 487.90 | 490.70 | 486.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-07 11:30:00 | 486.50 | 490.70 | 486.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 12:15:00 | 484.80 | 489.52 | 486.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-07 12:45:00 | 484.00 | 489.52 | 486.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 13:15:00 | 482.85 | 488.19 | 485.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-07 14:00:00 | 482.85 | 488.19 | 485.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 14:15:00 | 482.75 | 487.10 | 485.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-07 14:30:00 | 483.90 | 487.10 | 485.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 15:15:00 | 482.90 | 486.26 | 485.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 09:30:00 | 482.30 | 485.18 | 484.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2024-02-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 10:15:00 | 477.80 | 483.70 | 484.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 09:15:00 | 463.00 | 478.48 | 481.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 14:15:00 | 398.70 | 393.37 | 411.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 15:00:00 | 398.70 | 393.37 | 411.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 12:15:00 | 403.45 | 397.40 | 406.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 12:45:00 | 402.50 | 397.40 | 406.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 406.30 | 400.12 | 406.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 14:30:00 | 406.75 | 400.12 | 406.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 15:15:00 | 407.30 | 401.55 | 406.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 09:15:00 | 412.00 | 401.55 | 406.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 09:15:00 | 417.05 | 404.65 | 407.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 10:00:00 | 417.05 | 404.65 | 407.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 10:15:00 | 417.90 | 407.30 | 408.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-15 11:45:00 | 415.70 | 409.21 | 409.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-15 12:15:00 | 416.15 | 410.60 | 409.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2024-02-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 12:15:00 | 416.15 | 410.60 | 409.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 14:15:00 | 418.00 | 413.09 | 411.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 09:15:00 | 410.05 | 413.22 | 411.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 09:15:00 | 410.05 | 413.22 | 411.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 09:15:00 | 410.05 | 413.22 | 411.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 09:45:00 | 409.55 | 413.22 | 411.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 10:15:00 | 408.25 | 412.22 | 411.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 11:00:00 | 408.25 | 412.22 | 411.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 13:15:00 | 408.35 | 410.94 | 410.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 13:30:00 | 409.45 | 410.94 | 410.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2024-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 14:15:00 | 407.80 | 410.32 | 410.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 09:15:00 | 400.65 | 408.16 | 409.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-19 13:15:00 | 415.95 | 407.26 | 408.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 13:15:00 | 415.95 | 407.26 | 408.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 13:15:00 | 415.95 | 407.26 | 408.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 13:45:00 | 417.60 | 407.26 | 408.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 14:15:00 | 407.00 | 407.21 | 408.30 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-20 12:15:00 | 410.65 | 408.86 | 408.82 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 11:15:00 | 407.50 | 409.07 | 409.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 14:15:00 | 402.80 | 406.75 | 407.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 14:15:00 | 414.55 | 403.51 | 404.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 14:15:00 | 414.55 | 403.51 | 404.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 414.55 | 403.51 | 404.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 15:00:00 | 414.55 | 403.51 | 404.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2024-02-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 15:15:00 | 416.00 | 406.01 | 405.93 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 13:15:00 | 404.90 | 408.54 | 408.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 13:15:00 | 401.35 | 405.42 | 406.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 12:15:00 | 393.35 | 391.85 | 396.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 12:45:00 | 393.40 | 391.85 | 396.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 398.85 | 392.74 | 395.36 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-03-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 13:15:00 | 398.90 | 396.77 | 396.72 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 15:15:00 | 395.10 | 396.98 | 397.09 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-05 09:15:00 | 398.70 | 397.32 | 397.23 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 10:15:00 | 396.30 | 397.12 | 397.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 11:15:00 | 395.60 | 396.82 | 397.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 13:15:00 | 394.95 | 390.24 | 392.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 13:15:00 | 394.95 | 390.24 | 392.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 13:15:00 | 394.95 | 390.24 | 392.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 13:45:00 | 402.00 | 390.24 | 392.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 396.60 | 391.51 | 393.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 14:45:00 | 399.40 | 391.51 | 393.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 09:15:00 | 402.55 | 394.44 | 394.27 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 10:15:00 | 390.70 | 394.79 | 395.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 11:15:00 | 388.25 | 393.48 | 394.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 340.45 | 339.60 | 355.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 10:00:00 | 340.45 | 339.60 | 355.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 12:15:00 | 344.50 | 342.15 | 353.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 12:30:00 | 346.35 | 342.15 | 353.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 343.65 | 344.13 | 350.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 11:30:00 | 338.80 | 342.09 | 348.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-18 11:15:00 | 360.30 | 351.09 | 350.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2024-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 11:15:00 | 360.30 | 351.09 | 350.38 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 11:15:00 | 348.05 | 350.58 | 350.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 14:15:00 | 341.60 | 348.14 | 349.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 10:15:00 | 347.50 | 346.39 | 348.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-20 11:00:00 | 347.50 | 346.39 | 348.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 13:15:00 | 345.75 | 345.90 | 347.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 14:00:00 | 345.75 | 345.90 | 347.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 353.15 | 346.61 | 347.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 10:00:00 | 353.15 | 346.61 | 347.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 356.90 | 348.67 | 348.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 14:15:00 | 361.90 | 353.78 | 350.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 355.40 | 357.92 | 355.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 355.40 | 357.92 | 355.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 355.40 | 357.92 | 355.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:30:00 | 354.00 | 357.92 | 355.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 10:15:00 | 356.30 | 357.60 | 355.60 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2024-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 15:15:00 | 352.50 | 354.77 | 354.79 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 09:15:00 | 355.00 | 354.82 | 354.81 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 10:15:00 | 352.60 | 354.37 | 354.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 11:15:00 | 350.00 | 353.50 | 354.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 12:15:00 | 353.85 | 353.57 | 354.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-27 13:00:00 | 353.85 | 353.57 | 354.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 13:15:00 | 354.55 | 353.77 | 354.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 13:45:00 | 355.15 | 353.77 | 354.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 348.20 | 352.65 | 353.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-27 15:15:00 | 346.20 | 352.65 | 353.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 14:45:00 | 345.00 | 349.90 | 351.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-01 10:15:00 | 367.15 | 353.58 | 352.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2024-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 10:15:00 | 367.15 | 353.58 | 352.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 12:15:00 | 370.25 | 358.85 | 355.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 09:15:00 | 387.50 | 388.83 | 382.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-04 10:00:00 | 387.50 | 388.83 | 382.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 12:15:00 | 390.75 | 388.39 | 383.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 09:15:00 | 393.25 | 388.09 | 386.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-09 13:15:00 | 383.80 | 386.74 | 386.72 | SL hit (close<static) qty=1.00 sl=383.85 alert=retest2 |

### Cycle 83 — SELL (started 2024-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 10:15:00 | 393.65 | 400.60 | 400.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 14:15:00 | 388.50 | 394.97 | 397.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 395.65 | 394.35 | 397.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 395.65 | 394.35 | 397.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 395.65 | 394.35 | 397.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:30:00 | 396.25 | 394.35 | 397.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 394.70 | 394.42 | 396.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:45:00 | 395.60 | 394.42 | 396.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 391.00 | 390.74 | 393.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 14:30:00 | 385.55 | 390.50 | 392.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 14:15:00 | 389.35 | 389.08 | 389.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 14:15:00 | 391.45 | 389.55 | 389.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2024-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 14:15:00 | 391.45 | 389.55 | 389.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 394.15 | 390.22 | 389.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 09:15:00 | 416.40 | 417.00 | 411.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 10:00:00 | 416.40 | 417.00 | 411.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 426.85 | 432.21 | 428.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 10:00:00 | 426.85 | 432.21 | 428.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 426.80 | 431.13 | 427.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 10:30:00 | 426.40 | 431.13 | 427.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 11:15:00 | 428.85 | 430.67 | 428.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 12:15:00 | 430.00 | 430.67 | 428.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 13:30:00 | 429.45 | 430.11 | 428.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-30 14:15:00 | 420.70 | 428.23 | 427.54 | SL hit (close<static) qty=1.00 sl=425.05 alert=retest2 |

### Cycle 85 — SELL (started 2024-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 15:15:00 | 420.40 | 426.66 | 426.89 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-05-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 11:15:00 | 430.50 | 425.63 | 425.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-03 12:15:00 | 441.70 | 428.84 | 426.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-06 09:15:00 | 429.50 | 436.13 | 431.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 09:15:00 | 429.50 | 436.13 | 431.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 429.50 | 436.13 | 431.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 10:00:00 | 429.50 | 436.13 | 431.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 434.00 | 435.71 | 431.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 12:00:00 | 437.70 | 436.11 | 432.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 10:15:00 | 419.60 | 428.93 | 430.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 10:15:00 | 419.60 | 428.93 | 430.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 11:15:00 | 412.80 | 425.71 | 428.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 11:15:00 | 417.30 | 415.31 | 420.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 11:45:00 | 416.70 | 415.31 | 420.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 419.90 | 416.23 | 420.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:45:00 | 420.50 | 416.23 | 420.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 416.85 | 416.35 | 420.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 14:15:00 | 409.20 | 416.35 | 420.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 14:15:00 | 388.74 | 397.46 | 406.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-10 14:15:00 | 391.25 | 389.96 | 397.49 | SL hit (close>ema200) qty=0.50 sl=389.96 alert=retest2 |

### Cycle 88 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 402.40 | 395.78 | 395.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 407.00 | 398.03 | 396.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 10:15:00 | 410.00 | 410.07 | 406.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 11:00:00 | 410.00 | 410.07 | 406.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 409.95 | 409.75 | 407.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 412.10 | 409.70 | 407.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:45:00 | 412.95 | 410.33 | 407.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-22 15:15:00 | 422.00 | 423.65 | 423.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 15:15:00 | 422.00 | 423.65 | 423.70 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 09:15:00 | 426.60 | 424.24 | 423.97 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 12:15:00 | 421.35 | 423.42 | 423.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 14:15:00 | 418.85 | 422.10 | 422.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 13:15:00 | 393.05 | 392.11 | 397.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 13:30:00 | 395.00 | 392.11 | 397.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 386.15 | 386.29 | 390.41 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 406.90 | 393.60 | 392.25 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 351.60 | 390.07 | 392.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 328.00 | 377.65 | 387.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 355.70 | 354.60 | 366.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:00:00 | 355.70 | 354.60 | 366.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 374.90 | 358.83 | 365.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 374.90 | 358.83 | 365.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 376.00 | 362.27 | 366.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:45:00 | 377.70 | 362.27 | 366.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 371.40 | 365.90 | 367.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:30:00 | 372.10 | 365.90 | 367.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 15:15:00 | 372.00 | 368.01 | 367.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 10:15:00 | 374.05 | 369.76 | 368.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 15:15:00 | 388.00 | 388.10 | 383.72 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:15:00 | 391.55 | 388.10 | 383.72 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 12:30:00 | 390.10 | 389.59 | 385.96 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 389.70 | 389.50 | 387.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:30:00 | 388.60 | 389.50 | 387.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 386.60 | 388.92 | 387.02 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-13 10:15:00 | 386.60 | 388.92 | 387.02 | SL hit (close<ema400) qty=1.00 sl=387.02 alert=retest1 |

### Cycle 95 — SELL (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 15:15:00 | 399.00 | 401.53 | 401.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 14:15:00 | 396.90 | 399.96 | 400.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 09:15:00 | 405.75 | 400.55 | 400.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 09:15:00 | 405.75 | 400.55 | 400.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 405.75 | 400.55 | 400.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:15:00 | 408.40 | 400.55 | 400.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 10:15:00 | 407.35 | 401.91 | 401.46 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 15:15:00 | 401.00 | 401.95 | 402.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 10:15:00 | 399.75 | 401.32 | 401.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 14:15:00 | 399.75 | 399.27 | 400.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-27 15:00:00 | 399.75 | 399.27 | 400.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 397.00 | 398.82 | 400.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:45:00 | 398.85 | 398.30 | 399.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 397.50 | 398.14 | 399.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:30:00 | 398.45 | 398.14 | 399.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 395.70 | 395.56 | 397.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:30:00 | 396.40 | 395.56 | 397.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 395.75 | 395.31 | 396.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 11:30:00 | 392.70 | 394.54 | 395.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 09:15:00 | 404.90 | 394.54 | 394.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 09:15:00 | 404.90 | 394.54 | 394.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 09:15:00 | 424.55 | 405.32 | 401.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 12:15:00 | 421.45 | 423.16 | 416.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-09 13:00:00 | 421.45 | 423.16 | 416.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 412.80 | 420.40 | 416.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 15:00:00 | 412.80 | 420.40 | 416.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 415.00 | 419.32 | 416.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 09:15:00 | 426.25 | 419.32 | 416.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 14:15:00 | 417.45 | 420.58 | 420.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 14:15:00 | 417.45 | 420.58 | 420.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 15:15:00 | 416.50 | 419.77 | 420.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 12:15:00 | 423.30 | 418.13 | 419.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 12:15:00 | 423.30 | 418.13 | 419.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 423.30 | 418.13 | 419.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 12:45:00 | 422.70 | 418.13 | 419.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 422.35 | 418.97 | 419.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 13:30:00 | 423.50 | 418.97 | 419.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2024-07-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 15:15:00 | 422.00 | 419.81 | 419.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 424.20 | 420.69 | 420.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 12:15:00 | 422.80 | 422.80 | 421.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 12:15:00 | 422.80 | 422.80 | 421.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 422.80 | 422.80 | 421.44 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2024-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 15:15:00 | 416.40 | 420.03 | 420.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 413.00 | 418.62 | 419.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 10:15:00 | 404.30 | 398.39 | 401.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 10:15:00 | 404.30 | 398.39 | 401.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 404.30 | 398.39 | 401.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:30:00 | 403.50 | 398.39 | 401.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 403.35 | 399.38 | 401.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:00:00 | 403.35 | 399.38 | 401.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 397.20 | 398.94 | 401.33 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 409.65 | 402.68 | 402.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 13:15:00 | 418.20 | 407.54 | 404.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 14:15:00 | 408.15 | 410.27 | 408.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 14:15:00 | 408.15 | 410.27 | 408.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 408.15 | 410.27 | 408.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 15:00:00 | 408.15 | 410.27 | 408.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 409.00 | 410.02 | 408.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 411.00 | 410.02 | 408.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 10:15:00 | 404.90 | 408.67 | 407.83 | SL hit (close<static) qty=1.00 sl=406.30 alert=retest2 |

### Cycle 103 — SELL (started 2024-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 12:15:00 | 404.55 | 407.23 | 407.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-26 14:15:00 | 402.75 | 405.98 | 406.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 09:15:00 | 407.45 | 400.57 | 402.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 09:15:00 | 407.45 | 400.57 | 402.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 407.45 | 400.57 | 402.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:00:00 | 407.45 | 400.57 | 402.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 407.65 | 401.98 | 402.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:45:00 | 408.10 | 401.98 | 402.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 399.60 | 402.44 | 402.97 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 09:15:00 | 409.45 | 402.80 | 402.62 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 15:15:00 | 400.00 | 402.32 | 402.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 396.95 | 401.25 | 402.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 382.75 | 380.59 | 387.66 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 13:30:00 | 374.65 | 378.06 | 384.18 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 378.20 | 374.75 | 377.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-08 09:15:00 | 378.20 | 374.75 | 377.46 | SL hit (close>ema400) qty=1.00 sl=377.46 alert=retest1 |

### Cycle 106 — BUY (started 2024-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 12:15:00 | 364.90 | 362.86 | 362.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 13:15:00 | 365.40 | 363.37 | 362.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 15:15:00 | 367.95 | 367.96 | 366.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 15:15:00 | 367.95 | 367.96 | 366.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 367.95 | 367.96 | 366.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 12:30:00 | 369.85 | 368.39 | 366.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 14:00:00 | 370.65 | 368.84 | 367.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 11:15:00 | 368.95 | 371.19 | 371.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 11:15:00 | 368.95 | 371.19 | 371.34 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 09:15:00 | 372.65 | 370.83 | 370.75 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 14:15:00 | 369.65 | 370.74 | 370.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 368.05 | 370.08 | 370.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 15:15:00 | 367.40 | 366.95 | 368.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 09:15:00 | 367.50 | 366.95 | 368.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 366.90 | 366.94 | 368.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 11:45:00 | 365.80 | 367.30 | 367.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 12:15:00 | 368.15 | 367.47 | 367.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2024-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 12:15:00 | 368.15 | 367.47 | 367.47 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 13:15:00 | 367.35 | 367.45 | 367.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 14:15:00 | 366.20 | 367.20 | 367.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 09:15:00 | 367.45 | 367.21 | 367.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 367.45 | 367.21 | 367.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 367.45 | 367.21 | 367.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 13:00:00 | 365.00 | 366.51 | 366.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 14:30:00 | 364.25 | 365.86 | 366.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 10:30:00 | 364.95 | 365.57 | 366.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 11:00:00 | 365.05 | 365.57 | 366.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 366.00 | 365.16 | 365.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 09:15:00 | 375.00 | 365.16 | 365.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 373.40 | 366.81 | 366.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 09:15:00 | 373.40 | 366.81 | 366.43 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 10:15:00 | 363.80 | 367.67 | 367.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 12:15:00 | 360.55 | 365.39 | 366.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 13:15:00 | 367.40 | 364.58 | 365.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 13:15:00 | 367.40 | 364.58 | 365.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 367.40 | 364.58 | 365.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 14:00:00 | 367.40 | 364.58 | 365.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 373.00 | 366.26 | 365.98 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 10:15:00 | 363.75 | 366.16 | 366.43 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 369.90 | 366.48 | 366.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 12:15:00 | 376.70 | 370.36 | 368.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 372.15 | 372.30 | 370.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 10:00:00 | 372.15 | 372.30 | 370.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 373.55 | 374.21 | 372.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:45:00 | 373.10 | 374.21 | 372.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 372.40 | 373.61 | 372.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:00:00 | 372.40 | 373.61 | 372.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 369.70 | 372.83 | 372.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:45:00 | 370.10 | 372.83 | 372.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 370.50 | 372.36 | 372.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 09:15:00 | 372.45 | 372.29 | 372.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 09:15:00 | 366.55 | 371.14 | 371.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 09:15:00 | 366.55 | 371.14 | 371.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-20 09:15:00 | 362.80 | 368.35 | 369.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 09:15:00 | 366.50 | 364.58 | 365.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 09:15:00 | 366.50 | 364.58 | 365.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 366.50 | 364.58 | 365.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:15:00 | 369.10 | 364.58 | 365.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 366.70 | 365.01 | 365.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 11:15:00 | 365.75 | 365.01 | 365.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 10:00:00 | 365.00 | 365.14 | 365.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 14:15:00 | 347.46 | 351.54 | 355.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 14:15:00 | 346.75 | 351.54 | 355.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-07 09:15:00 | 329.18 | 337.48 | 344.76 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 118 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 332.25 | 328.36 | 327.98 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 325.85 | 327.74 | 327.86 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 10:15:00 | 329.15 | 328.07 | 327.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 14:15:00 | 340.00 | 330.41 | 329.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 14:15:00 | 337.85 | 338.15 | 334.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 15:00:00 | 337.85 | 338.15 | 334.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 334.50 | 337.61 | 335.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 334.50 | 337.61 | 335.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 339.40 | 337.97 | 335.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:30:00 | 344.00 | 340.27 | 337.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 12:45:00 | 343.30 | 341.11 | 338.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 15:00:00 | 354.00 | 343.92 | 340.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-22 10:15:00 | 343.15 | 350.15 | 351.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 343.15 | 350.15 | 351.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 338.40 | 347.80 | 349.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 354.00 | 340.56 | 344.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 09:15:00 | 354.00 | 340.56 | 344.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 354.00 | 340.56 | 344.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:00:00 | 354.00 | 340.56 | 344.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 354.00 | 343.24 | 345.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:15:00 | 367.00 | 343.24 | 345.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2024-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 12:15:00 | 361.90 | 349.26 | 347.88 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 10:15:00 | 341.30 | 352.60 | 352.73 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 354.65 | 351.24 | 350.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 14:15:00 | 356.95 | 353.02 | 351.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 13:15:00 | 367.10 | 368.47 | 365.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 14:00:00 | 367.10 | 368.47 | 365.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 371.65 | 369.11 | 365.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:30:00 | 367.15 | 369.11 | 365.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 360.00 | 368.47 | 366.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 360.00 | 368.47 | 366.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 355.35 | 365.85 | 365.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 355.35 | 365.85 | 365.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 357.45 | 364.17 | 365.01 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 368.95 | 363.60 | 363.16 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 360.00 | 366.34 | 367.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 15:15:00 | 358.95 | 362.76 | 364.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 11:15:00 | 355.00 | 352.49 | 356.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 11:15:00 | 355.00 | 352.49 | 356.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 355.00 | 352.49 | 356.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:30:00 | 359.40 | 352.49 | 356.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 341.15 | 331.59 | 334.01 | EMA400 retest candle locked (from downside) |

### Cycle 128 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 340.50 | 335.72 | 335.51 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 332.90 | 335.35 | 335.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 320.50 | 332.38 | 334.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 10:15:00 | 326.00 | 324.93 | 328.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 11:00:00 | 326.00 | 324.93 | 328.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 326.30 | 325.33 | 327.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:30:00 | 326.80 | 325.33 | 327.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 328.20 | 325.91 | 327.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 15:00:00 | 328.20 | 325.91 | 327.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 330.50 | 326.83 | 327.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:15:00 | 346.75 | 326.83 | 327.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 344.90 | 330.45 | 329.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 351.05 | 347.28 | 343.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 14:15:00 | 345.05 | 348.81 | 346.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 14:15:00 | 345.05 | 348.81 | 346.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 345.05 | 348.81 | 346.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 345.05 | 348.81 | 346.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 347.00 | 348.45 | 346.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 348.55 | 348.45 | 346.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:45:00 | 348.95 | 347.91 | 346.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 11:15:00 | 343.50 | 346.66 | 345.86 | SL hit (close<static) qty=1.00 sl=344.30 alert=retest2 |

### Cycle 131 — SELL (started 2024-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 14:15:00 | 361.25 | 364.15 | 364.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 354.05 | 361.71 | 363.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-17 09:15:00 | 357.25 | 356.35 | 358.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-17 10:00:00 | 357.25 | 356.35 | 358.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 356.95 | 356.47 | 357.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:45:00 | 358.15 | 356.47 | 357.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 336.65 | 340.68 | 345.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:30:00 | 334.60 | 337.78 | 342.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 13:15:00 | 317.87 | 321.68 | 324.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-31 10:15:00 | 319.40 | 318.78 | 322.03 | SL hit (close>ema200) qty=0.50 sl=318.78 alert=retest2 |

### Cycle 132 — BUY (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 11:15:00 | 324.90 | 322.40 | 322.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 326.70 | 323.99 | 323.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 327.05 | 328.85 | 327.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 14:15:00 | 327.05 | 328.85 | 327.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 327.05 | 328.85 | 327.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 327.05 | 328.85 | 327.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 328.90 | 328.86 | 327.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 321.60 | 328.86 | 327.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 320.20 | 327.13 | 326.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 320.20 | 327.13 | 326.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 314.50 | 324.60 | 325.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 308.30 | 318.73 | 322.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 13:15:00 | 315.25 | 314.65 | 317.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 14:00:00 | 315.25 | 314.65 | 317.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 314.05 | 313.72 | 315.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 11:00:00 | 312.60 | 313.92 | 315.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 13:15:00 | 296.97 | 306.16 | 310.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 14:15:00 | 281.34 | 291.23 | 299.22 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 134 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 330.45 | 302.28 | 298.49 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 310.10 | 319.21 | 320.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 307.00 | 315.37 | 318.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 314.35 | 312.97 | 315.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 09:45:00 | 314.50 | 312.97 | 315.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 304.40 | 310.79 | 313.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:00:00 | 303.75 | 307.83 | 311.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 15:15:00 | 301.00 | 307.09 | 310.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 13:15:00 | 288.56 | 295.74 | 302.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 285.95 | 290.67 | 298.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 13:15:00 | 290.80 | 288.01 | 294.27 | SL hit (close>ema200) qty=0.50 sl=288.01 alert=retest2 |

### Cycle 136 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 321.70 | 298.24 | 296.18 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 307.05 | 315.05 | 315.38 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 317.20 | 313.45 | 313.00 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 13:15:00 | 311.65 | 313.85 | 313.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 309.75 | 312.99 | 313.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 282.50 | 282.34 | 289.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:00:00 | 282.50 | 282.34 | 289.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 285.35 | 282.85 | 287.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 286.55 | 282.85 | 287.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 283.15 | 282.91 | 286.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:30:00 | 285.20 | 282.91 | 286.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 266.35 | 258.66 | 262.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 266.35 | 258.66 | 262.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 266.30 | 260.19 | 263.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 13:00:00 | 263.20 | 261.77 | 263.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 13:45:00 | 263.85 | 262.48 | 263.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 14:15:00 | 262.60 | 262.48 | 263.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 09:15:00 | 269.50 | 264.64 | 264.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 09:15:00 | 269.50 | 264.64 | 264.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 14:15:00 | 271.65 | 268.67 | 266.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 268.85 | 269.01 | 267.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:45:00 | 269.35 | 269.01 | 267.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 268.65 | 269.17 | 267.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:30:00 | 268.10 | 269.17 | 267.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 267.30 | 268.80 | 267.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:30:00 | 267.35 | 268.80 | 267.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 267.00 | 268.44 | 267.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 261.40 | 268.44 | 267.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 261.85 | 267.12 | 267.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 11:15:00 | 259.50 | 262.02 | 263.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 13:15:00 | 240.45 | 238.46 | 243.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 14:00:00 | 240.45 | 238.46 | 243.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 246.32 | 239.95 | 243.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 246.32 | 239.95 | 243.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 244.56 | 240.87 | 243.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:30:00 | 243.86 | 240.92 | 243.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 247.70 | 242.54 | 243.09 | SL hit (close>static) qty=1.00 sl=247.30 alert=retest2 |

### Cycle 142 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 249.03 | 243.84 | 243.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 255.20 | 248.69 | 246.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 13:15:00 | 259.66 | 259.67 | 255.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 14:00:00 | 259.66 | 259.67 | 255.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 258.93 | 260.11 | 256.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 257.77 | 260.11 | 256.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 256.43 | 259.03 | 256.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 256.43 | 259.03 | 256.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 258.28 | 258.88 | 257.04 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 252.24 | 256.06 | 256.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 12:15:00 | 248.27 | 250.42 | 251.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 252.39 | 249.50 | 250.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 252.39 | 249.50 | 250.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 252.39 | 249.50 | 250.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:30:00 | 251.94 | 249.50 | 250.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 251.40 | 249.88 | 250.86 | EMA400 retest candle locked (from downside) |

### Cycle 144 — BUY (started 2025-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 15:15:00 | 252.17 | 251.30 | 251.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 259.17 | 252.88 | 251.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 14:15:00 | 271.39 | 271.50 | 266.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 15:00:00 | 271.39 | 271.50 | 266.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 268.00 | 272.76 | 271.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 268.00 | 272.76 | 271.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 267.96 | 271.80 | 270.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 267.72 | 271.80 | 270.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — SELL (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 14:15:00 | 265.76 | 269.56 | 269.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 15:15:00 | 265.23 | 268.69 | 269.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 264.84 | 263.58 | 265.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 264.84 | 263.58 | 265.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 264.84 | 263.58 | 265.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:45:00 | 265.03 | 263.58 | 265.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 263.00 | 263.47 | 265.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:30:00 | 264.49 | 263.47 | 265.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 264.70 | 263.80 | 265.35 | EMA400 retest candle locked (from downside) |

### Cycle 146 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 271.50 | 266.07 | 266.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 12:15:00 | 273.70 | 269.87 | 268.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 09:15:00 | 270.70 | 271.63 | 269.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 270.70 | 271.63 | 269.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 270.70 | 271.63 | 269.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:30:00 | 276.85 | 273.42 | 271.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 13:00:00 | 276.95 | 273.42 | 271.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 10:30:00 | 278.50 | 282.07 | 279.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 13:15:00 | 276.30 | 279.56 | 278.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 14:15:00 | 273.20 | 277.30 | 277.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 14:15:00 | 273.20 | 277.30 | 277.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 254.00 | 272.16 | 275.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 264.00 | 262.41 | 267.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 267.50 | 262.41 | 267.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 263.70 | 262.66 | 267.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 262.10 | 263.03 | 267.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 262.40 | 265.51 | 266.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 12:15:00 | 273.15 | 266.18 | 266.66 | SL hit (close>static) qty=1.00 sl=272.55 alert=retest2 |

### Cycle 148 — BUY (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 13:15:00 | 272.05 | 267.35 | 267.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 273.80 | 269.03 | 268.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 15:15:00 | 328.90 | 328.96 | 322.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 09:15:00 | 328.00 | 328.96 | 322.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 320.00 | 327.17 | 322.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 320.00 | 327.17 | 322.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 323.00 | 326.33 | 322.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 11:30:00 | 324.20 | 325.73 | 322.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 14:45:00 | 324.75 | 324.81 | 323.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 311.95 | 322.13 | 322.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 311.95 | 322.13 | 322.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 307.60 | 319.23 | 321.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 316.45 | 313.47 | 316.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 316.45 | 313.47 | 316.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 316.45 | 313.47 | 316.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 316.45 | 313.47 | 316.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 315.05 | 313.61 | 315.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 15:00:00 | 315.05 | 313.61 | 315.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 315.25 | 313.94 | 315.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 319.00 | 313.94 | 315.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 316.60 | 314.47 | 315.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:30:00 | 314.85 | 314.65 | 315.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 13:30:00 | 315.00 | 314.55 | 315.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 12:15:00 | 299.25 | 304.36 | 306.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 14:15:00 | 299.11 | 302.01 | 305.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 11:15:00 | 301.60 | 299.89 | 302.85 | SL hit (close>ema200) qty=0.50 sl=299.89 alert=retest2 |

### Cycle 150 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 309.10 | 304.54 | 304.24 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 298.00 | 303.47 | 304.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 291.85 | 301.15 | 303.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 311.65 | 299.06 | 300.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 311.65 | 299.06 | 300.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 311.65 | 299.06 | 300.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 311.65 | 299.06 | 300.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 312.45 | 301.73 | 301.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 314.55 | 305.88 | 303.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 310.40 | 313.20 | 309.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 13:15:00 | 310.40 | 313.20 | 309.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 310.40 | 313.20 | 309.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 310.40 | 313.20 | 309.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 311.15 | 312.79 | 309.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 316.05 | 312.43 | 309.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-16 12:15:00 | 347.66 | 334.06 | 326.25 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 347.20 | 350.24 | 350.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 14:15:00 | 346.95 | 349.58 | 350.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 352.95 | 349.92 | 350.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 352.95 | 349.92 | 350.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 352.95 | 349.92 | 350.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 354.00 | 349.92 | 350.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 354.55 | 350.85 | 350.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 355.40 | 352.26 | 351.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 10:15:00 | 358.50 | 358.62 | 356.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 10:15:00 | 358.50 | 358.62 | 356.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 358.50 | 358.62 | 356.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:30:00 | 355.70 | 358.62 | 356.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 355.20 | 358.00 | 356.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:45:00 | 353.40 | 358.00 | 356.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 355.05 | 357.41 | 356.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:00:00 | 355.05 | 357.41 | 356.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 356.40 | 357.21 | 356.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:45:00 | 354.80 | 357.21 | 356.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 356.00 | 356.97 | 356.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:00:00 | 356.00 | 356.97 | 356.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 355.00 | 356.57 | 356.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 14:15:00 | 357.40 | 356.57 | 356.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 15:00:00 | 357.00 | 356.66 | 356.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:00:00 | 357.50 | 361.35 | 360.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 13:15:00 | 357.65 | 359.18 | 359.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 357.65 | 359.18 | 359.35 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 09:15:00 | 365.75 | 360.09 | 359.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 11:15:00 | 370.40 | 364.63 | 362.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 362.55 | 364.91 | 363.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 362.55 | 364.91 | 363.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 362.55 | 364.91 | 363.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 361.25 | 364.91 | 363.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 364.30 | 364.79 | 363.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 11:15:00 | 366.25 | 364.79 | 363.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-06 12:15:00 | 402.88 | 388.66 | 380.16 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 403.95 | 407.29 | 407.47 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 11:15:00 | 412.80 | 408.15 | 407.73 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 09:15:00 | 398.25 | 406.47 | 407.19 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 414.05 | 407.18 | 407.13 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 403.70 | 407.52 | 407.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 14:15:00 | 401.45 | 405.72 | 406.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 390.60 | 388.70 | 393.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 11:00:00 | 390.60 | 388.70 | 393.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 388.60 | 388.91 | 392.78 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 399.40 | 394.03 | 393.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 403.65 | 397.36 | 395.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 397.70 | 398.48 | 396.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 397.70 | 398.48 | 396.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 397.15 | 398.21 | 396.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 406.50 | 398.21 | 396.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 411.10 | 414.35 | 414.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 411.10 | 414.35 | 414.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 14:15:00 | 408.75 | 412.72 | 413.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 14:15:00 | 391.60 | 391.40 | 394.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 14:15:00 | 391.60 | 391.40 | 394.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 391.60 | 391.40 | 394.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 15:15:00 | 389.85 | 391.40 | 394.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:30:00 | 389.85 | 390.67 | 393.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 11:15:00 | 396.25 | 382.92 | 381.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 11:15:00 | 396.25 | 382.92 | 381.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 12:15:00 | 413.40 | 389.02 | 384.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 14:15:00 | 457.20 | 458.48 | 446.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 14:45:00 | 456.60 | 458.48 | 446.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 443.00 | 452.53 | 449.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 443.00 | 452.53 | 449.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 440.80 | 450.18 | 449.14 | EMA400 retest candle locked (from upside) |

### Cycle 165 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 438.75 | 447.90 | 448.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 14:15:00 | 436.10 | 442.68 | 445.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 411.50 | 392.59 | 397.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 411.50 | 392.59 | 397.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 411.50 | 392.59 | 397.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:45:00 | 416.20 | 392.59 | 397.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 419.35 | 397.94 | 399.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 419.35 | 397.94 | 399.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 11:15:00 | 415.90 | 401.53 | 400.71 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 398.00 | 404.36 | 404.92 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 10:15:00 | 410.05 | 405.13 | 404.60 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 404.85 | 407.69 | 407.80 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 419.00 | 409.32 | 408.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 09:15:00 | 433.85 | 418.61 | 415.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 15:15:00 | 423.30 | 425.26 | 420.92 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:15:00 | 427.85 | 425.26 | 420.92 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 428.20 | 430.60 | 428.26 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-19 14:15:00 | 428.20 | 430.60 | 428.26 | SL hit (close<ema400) qty=1.00 sl=428.26 alert=retest1 |

### Cycle 171 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 420.40 | 426.28 | 426.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 15:15:00 | 419.00 | 424.83 | 426.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 434.25 | 426.71 | 426.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 434.25 | 426.71 | 426.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 434.25 | 426.71 | 426.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 445.90 | 426.71 | 426.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 435.00 | 428.37 | 427.70 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 426.50 | 428.89 | 429.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 423.85 | 427.88 | 428.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 416.30 | 415.75 | 419.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 10:00:00 | 416.30 | 415.75 | 419.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 417.00 | 415.96 | 419.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:30:00 | 415.15 | 415.54 | 418.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 14:45:00 | 414.75 | 414.84 | 417.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 423.75 | 411.76 | 413.89 | SL hit (close>static) qty=1.00 sl=420.95 alert=retest2 |

### Cycle 174 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 432.20 | 415.84 | 415.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 439.90 | 429.48 | 423.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 456.05 | 456.99 | 450.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 14:45:00 | 455.25 | 456.99 | 450.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 470.30 | 459.45 | 452.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 10:15:00 | 483.70 | 459.45 | 452.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-08 09:15:00 | 532.07 | 505.22 | 482.66 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 15:15:00 | 514.75 | 522.48 | 522.61 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 528.40 | 523.66 | 523.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 10:15:00 | 539.45 | 526.82 | 524.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 09:15:00 | 546.35 | 555.44 | 548.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 546.35 | 555.44 | 548.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 546.35 | 555.44 | 548.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 544.30 | 555.44 | 548.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 547.10 | 553.77 | 548.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:30:00 | 546.80 | 553.77 | 548.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 546.40 | 552.30 | 547.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:45:00 | 546.00 | 552.30 | 547.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 547.90 | 551.42 | 547.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:45:00 | 550.50 | 551.42 | 547.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 547.90 | 550.72 | 547.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:00:00 | 547.90 | 550.72 | 547.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 547.00 | 549.97 | 547.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:45:00 | 546.25 | 549.97 | 547.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 548.50 | 549.68 | 547.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 555.65 | 549.68 | 547.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 15:15:00 | 551.55 | 550.31 | 549.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 543.60 | 549.17 | 548.96 | SL hit (close<static) qty=1.00 sl=546.95 alert=retest2 |

### Cycle 177 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 547.00 | 548.73 | 548.79 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 553.75 | 549.74 | 549.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 09:15:00 | 573.65 | 555.04 | 551.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 563.95 | 572.74 | 567.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 14:15:00 | 563.95 | 572.74 | 567.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 563.95 | 572.74 | 567.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 15:00:00 | 563.95 | 572.74 | 567.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 567.00 | 571.59 | 567.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 579.55 | 571.59 | 567.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-23 12:15:00 | 637.50 | 594.31 | 580.09 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 179 — SELL (started 2025-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 13:15:00 | 586.50 | 600.05 | 601.88 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 12:15:00 | 607.00 | 601.61 | 601.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 13:15:00 | 608.95 | 603.08 | 602.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 15:15:00 | 601.60 | 603.10 | 602.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 15:15:00 | 601.60 | 603.10 | 602.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 601.60 | 603.10 | 602.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 610.95 | 603.10 | 602.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 611.95 | 604.87 | 603.24 | EMA400 retest candle locked (from upside) |

### Cycle 181 — SELL (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 13:15:00 | 588.75 | 600.91 | 601.98 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 602.60 | 602.07 | 602.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 609.70 | 603.60 | 602.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 12:15:00 | 605.95 | 615.38 | 610.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 12:15:00 | 605.95 | 615.38 | 610.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 605.95 | 615.38 | 610.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:00:00 | 605.95 | 615.38 | 610.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 611.60 | 614.63 | 610.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:45:00 | 614.55 | 614.37 | 610.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 618.75 | 614.11 | 610.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 11:45:00 | 615.25 | 616.18 | 612.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 10:15:00 | 591.95 | 607.79 | 609.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 10:15:00 | 591.95 | 607.79 | 609.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 12:15:00 | 585.50 | 600.43 | 606.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 603.00 | 594.89 | 601.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 603.00 | 594.89 | 601.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 603.00 | 594.89 | 601.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 603.00 | 594.89 | 601.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 594.00 | 594.71 | 600.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 594.00 | 594.71 | 600.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 604.85 | 593.04 | 596.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 608.35 | 593.04 | 596.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 612.40 | 596.91 | 598.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:30:00 | 609.65 | 596.91 | 598.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 613.15 | 600.16 | 599.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 13:15:00 | 621.10 | 604.85 | 601.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 11:15:00 | 613.60 | 615.17 | 609.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 11:30:00 | 614.30 | 615.17 | 609.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 610.20 | 613.63 | 609.41 | EMA400 retest candle locked (from upside) |

### Cycle 185 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 593.45 | 605.31 | 606.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 588.55 | 596.73 | 600.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 595.40 | 592.33 | 596.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 595.40 | 592.33 | 596.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 595.40 | 592.33 | 596.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 595.40 | 592.33 | 596.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 596.15 | 593.09 | 596.53 | EMA400 retest candle locked (from downside) |

### Cycle 186 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 602.15 | 598.22 | 598.15 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 14:15:00 | 592.20 | 597.91 | 598.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 10:15:00 | 579.05 | 591.12 | 594.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 13:15:00 | 573.65 | 571.07 | 578.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 13:15:00 | 573.65 | 571.07 | 578.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 573.65 | 571.07 | 578.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:30:00 | 576.80 | 571.07 | 578.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 575.65 | 571.99 | 578.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:45:00 | 578.15 | 571.99 | 578.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 578.60 | 573.31 | 578.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 586.80 | 575.77 | 579.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 581.95 | 577.24 | 578.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 581.95 | 577.24 | 578.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 580.55 | 577.90 | 578.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:45:00 | 585.50 | 577.90 | 578.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 582.40 | 578.82 | 578.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:00:00 | 582.40 | 578.82 | 578.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — BUY (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 10:15:00 | 581.10 | 579.27 | 579.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 11:15:00 | 593.80 | 582.18 | 580.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 15:15:00 | 584.45 | 584.64 | 582.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 15:15:00 | 584.45 | 584.64 | 582.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 584.45 | 584.64 | 582.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 581.45 | 584.64 | 582.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 579.30 | 583.58 | 582.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:45:00 | 580.50 | 583.58 | 582.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 577.00 | 582.26 | 581.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:00:00 | 577.00 | 582.26 | 581.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 584.00 | 582.16 | 581.73 | EMA400 retest candle locked (from upside) |

### Cycle 189 — SELL (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 14:15:00 | 578.30 | 581.39 | 581.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 15:15:00 | 577.25 | 580.56 | 581.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 583.50 | 581.15 | 581.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 583.50 | 581.15 | 581.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 583.50 | 581.15 | 581.26 | EMA400 retest candle locked (from downside) |

### Cycle 190 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 585.95 | 582.11 | 581.69 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 15:15:00 | 577.40 | 581.38 | 581.58 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 592.65 | 583.63 | 582.59 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 593.70 | 597.16 | 597.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 14:15:00 | 593.00 | 596.33 | 596.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 580.80 | 575.83 | 582.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:00:00 | 580.80 | 575.83 | 582.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 583.40 | 577.34 | 582.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:45:00 | 584.00 | 577.34 | 582.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 587.10 | 579.29 | 582.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 587.10 | 579.29 | 582.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 594.00 | 582.23 | 583.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 592.00 | 582.23 | 583.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 600.25 | 585.84 | 585.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 607.65 | 592.39 | 588.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 15:15:00 | 597.30 | 598.98 | 594.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 09:15:00 | 595.25 | 598.98 | 594.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 594.75 | 598.13 | 594.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:45:00 | 592.10 | 598.13 | 594.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 590.85 | 596.68 | 594.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:30:00 | 590.45 | 596.68 | 594.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 592.95 | 595.93 | 594.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:15:00 | 593.70 | 595.93 | 594.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 13:15:00 | 594.00 | 595.13 | 593.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 13:45:00 | 595.15 | 595.69 | 594.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 599.05 | 595.73 | 595.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 599.00 | 596.38 | 595.72 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-13 13:15:00 | 592.45 | 595.38 | 595.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 592.45 | 595.38 | 595.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 588.80 | 594.06 | 594.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 10:15:00 | 593.30 | 592.73 | 593.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 10:15:00 | 593.30 | 592.73 | 593.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 593.30 | 592.73 | 593.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:00:00 | 593.30 | 592.73 | 593.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 567.00 | 586.70 | 590.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:30:00 | 558.60 | 586.70 | 590.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 485.95 | 483.69 | 488.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 494.35 | 483.69 | 488.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 492.20 | 485.39 | 489.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 12:15:00 | 488.10 | 487.88 | 489.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 12:15:00 | 501.80 | 490.67 | 490.87 | SL hit (close>static) qty=1.00 sl=500.80 alert=retest2 |

### Cycle 196 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 504.90 | 493.51 | 492.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 528.20 | 500.45 | 495.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 544.80 | 547.02 | 529.93 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 09:45:00 | 557.55 | 545.78 | 537.03 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 544.00 | 546.57 | 543.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:30:00 | 543.95 | 546.57 | 543.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 543.40 | 545.94 | 543.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-02 13:15:00 | 543.40 | 545.94 | 543.56 | SL hit (close<ema400) qty=1.00 sl=543.56 alert=retest1 |

### Cycle 197 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 529.55 | 541.02 | 541.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 12:15:00 | 524.15 | 534.43 | 538.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 533.25 | 529.41 | 534.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 533.25 | 529.41 | 534.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 533.25 | 529.41 | 534.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 535.60 | 529.41 | 534.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 530.70 | 529.67 | 533.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 11:45:00 | 526.50 | 529.10 | 533.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 522.70 | 528.08 | 531.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:15:00 | 500.17 | 512.27 | 520.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:15:00 | 496.56 | 508.84 | 517.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-09 09:15:00 | 473.85 | 495.13 | 506.89 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 198 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 524.25 | 498.68 | 496.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 530.25 | 517.49 | 508.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 524.70 | 529.16 | 520.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:30:00 | 527.40 | 529.16 | 520.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 520.75 | 524.35 | 521.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:30:00 | 521.45 | 524.35 | 521.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 520.90 | 523.66 | 521.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 517.95 | 523.66 | 521.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 509.70 | 520.87 | 520.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 509.70 | 520.87 | 520.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 507.85 | 518.27 | 518.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 11:15:00 | 503.55 | 515.32 | 517.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 09:15:00 | 514.35 | 508.95 | 512.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 09:15:00 | 514.35 | 508.95 | 512.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 514.35 | 508.95 | 512.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 514.35 | 508.95 | 512.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 509.65 | 509.09 | 512.52 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 516.80 | 513.73 | 513.59 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 12:15:00 | 512.80 | 513.43 | 513.47 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 514.20 | 513.48 | 513.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 540.00 | 521.97 | 518.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 526.40 | 527.23 | 522.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 14:45:00 | 528.55 | 527.23 | 522.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 585.25 | 592.01 | 579.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 14:15:00 | 593.90 | 589.85 | 581.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 15:00:00 | 597.00 | 591.28 | 583.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 15:15:00 | 606.10 | 613.28 | 613.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 15:15:00 | 606.10 | 613.28 | 613.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 588.00 | 608.22 | 611.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 561.80 | 554.44 | 567.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 561.80 | 554.44 | 567.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 561.80 | 554.44 | 567.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:45:00 | 558.30 | 554.44 | 567.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 563.60 | 557.32 | 566.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:30:00 | 566.15 | 557.32 | 566.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 572.60 | 561.61 | 567.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 565.65 | 563.95 | 567.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 564.35 | 564.15 | 567.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:45:00 | 566.80 | 564.28 | 566.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 09:15:00 | 576.05 | 568.32 | 568.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — BUY (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 09:15:00 | 576.05 | 568.32 | 568.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 10:15:00 | 580.85 | 570.83 | 569.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 13:15:00 | 571.45 | 572.42 | 570.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 13:15:00 | 571.45 | 572.42 | 570.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 571.45 | 572.42 | 570.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:00:00 | 571.45 | 572.42 | 570.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 572.05 | 572.34 | 570.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:30:00 | 572.05 | 572.34 | 570.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 568.45 | 571.65 | 570.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:30:00 | 567.65 | 571.65 | 570.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 566.30 | 570.58 | 570.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:00:00 | 566.30 | 570.58 | 570.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — SELL (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 11:15:00 | 562.95 | 569.06 | 569.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 12:15:00 | 559.60 | 567.16 | 568.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-20 09:15:00 | 550.50 | 546.76 | 553.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 550.50 | 546.76 | 553.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 550.50 | 546.76 | 553.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:30:00 | 554.55 | 546.76 | 553.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 550.55 | 547.52 | 553.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:30:00 | 554.00 | 547.52 | 553.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 544.45 | 546.90 | 552.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:30:00 | 550.50 | 546.90 | 552.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 543.55 | 542.31 | 546.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 543.55 | 542.31 | 546.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 539.15 | 539.47 | 543.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:45:00 | 534.40 | 538.21 | 542.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:45:00 | 535.15 | 537.38 | 541.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:30:00 | 533.75 | 537.50 | 539.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 09:45:00 | 534.35 | 529.09 | 531.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 548.05 | 532.88 | 532.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:30:00 | 547.00 | 532.88 | 532.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-28 11:15:00 | 555.35 | 537.37 | 534.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 555.35 | 537.37 | 534.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 571.85 | 554.13 | 544.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 11:15:00 | 582.85 | 587.93 | 572.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 12:00:00 | 582.85 | 587.93 | 572.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 582.00 | 584.79 | 573.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 10:00:00 | 597.50 | 585.01 | 576.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:30:00 | 585.00 | 590.10 | 581.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 14:30:00 | 585.95 | 587.61 | 581.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 561.00 | 578.08 | 578.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 561.00 | 578.08 | 578.31 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 590.00 | 578.42 | 577.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 594.30 | 585.70 | 581.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 13:15:00 | 609.45 | 611.45 | 602.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 13:30:00 | 610.45 | 611.45 | 602.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 605.50 | 610.88 | 604.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:15:00 | 601.25 | 610.88 | 604.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 604.80 | 609.66 | 604.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 598.20 | 609.66 | 604.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 602.10 | 608.15 | 604.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:00:00 | 602.10 | 608.15 | 604.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 597.50 | 606.02 | 603.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 597.50 | 606.02 | 603.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — SELL (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 15:15:00 | 599.05 | 602.66 | 602.77 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 617.20 | 605.57 | 604.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 637.20 | 611.89 | 607.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 11:15:00 | 620.70 | 622.86 | 616.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 12:00:00 | 620.70 | 622.86 | 616.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 609.00 | 620.09 | 616.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:30:00 | 606.95 | 620.09 | 616.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 608.00 | 617.67 | 615.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:00:00 | 608.00 | 617.67 | 615.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — SELL (started 2026-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 15:15:00 | 605.95 | 613.23 | 613.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 595.00 | 609.58 | 612.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 09:15:00 | 597.70 | 597.47 | 603.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 597.70 | 597.47 | 603.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 597.70 | 597.47 | 603.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 13:00:00 | 592.90 | 597.13 | 601.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:30:00 | 592.30 | 595.81 | 600.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 563.25 | 576.06 | 585.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 562.68 | 576.06 | 585.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 554.55 | 548.48 | 558.94 | SL hit (close>ema200) qty=0.50 sl=548.48 alert=retest2 |

### Cycle 212 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 575.50 | 565.09 | 563.71 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 557.00 | 564.35 | 564.50 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 09:15:00 | 571.50 | 565.78 | 565.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 10:15:00 | 579.05 | 568.43 | 566.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 13:15:00 | 570.10 | 571.20 | 568.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 13:15:00 | 570.10 | 571.20 | 568.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 570.10 | 571.20 | 568.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 574.60 | 570.18 | 568.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 567.50 | 569.65 | 568.34 | SL hit (close<static) qty=1.00 sl=567.90 alert=retest2 |

### Cycle 215 — SELL (started 2026-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 12:15:00 | 562.60 | 567.20 | 567.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 561.30 | 564.68 | 566.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 15:15:00 | 563.70 | 560.97 | 563.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 15:15:00 | 563.70 | 560.97 | 563.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 563.70 | 560.97 | 563.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 571.70 | 560.97 | 563.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 571.90 | 563.16 | 563.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:15:00 | 570.00 | 563.16 | 563.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 216 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 569.35 | 565.37 | 564.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 572.45 | 566.35 | 565.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 11:15:00 | 566.85 | 572.06 | 569.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 11:15:00 | 566.85 | 572.06 | 569.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 566.85 | 572.06 | 569.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:00:00 | 566.85 | 572.06 | 569.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 566.50 | 570.95 | 569.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:30:00 | 565.50 | 570.95 | 569.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 569.60 | 570.95 | 569.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 569.60 | 570.95 | 569.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 567.00 | 570.16 | 569.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 566.00 | 570.16 | 569.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 564.30 | 568.99 | 569.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 540.75 | 553.29 | 559.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 551.60 | 538.74 | 546.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 551.60 | 538.74 | 546.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 551.60 | 538.74 | 546.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:45:00 | 552.80 | 538.74 | 546.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 550.60 | 541.11 | 547.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:15:00 | 546.55 | 541.11 | 547.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 12:30:00 | 547.40 | 542.75 | 546.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 09:15:00 | 553.65 | 545.54 | 546.96 | SL hit (close>static) qty=1.00 sl=553.30 alert=retest2 |

### Cycle 218 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 551.50 | 547.66 | 547.64 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 546.00 | 547.66 | 547.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 530.70 | 544.27 | 546.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 541.75 | 535.06 | 539.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 541.75 | 535.06 | 539.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 541.75 | 535.06 | 539.08 | EMA400 retest candle locked (from downside) |

### Cycle 220 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 546.40 | 540.49 | 540.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 550.50 | 542.49 | 541.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 547.00 | 548.46 | 545.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:00:00 | 547.00 | 548.46 | 545.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 545.15 | 547.80 | 545.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 545.15 | 547.80 | 545.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 543.60 | 546.96 | 545.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 543.60 | 546.96 | 545.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 541.40 | 545.85 | 544.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 540.65 | 545.85 | 544.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 538.10 | 544.30 | 544.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:45:00 | 551.35 | 545.52 | 544.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 11:15:00 | 541.20 | 547.19 | 547.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 221 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 541.20 | 547.19 | 547.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 532.05 | 544.16 | 545.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 532.10 | 531.43 | 536.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 533.45 | 531.43 | 536.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 540.50 | 533.18 | 536.46 | EMA400 retest candle locked (from downside) |

### Cycle 222 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 546.05 | 538.52 | 538.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 13:15:00 | 555.00 | 541.82 | 539.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 550.80 | 555.79 | 550.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 550.80 | 555.79 | 550.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 550.80 | 555.79 | 550.70 | EMA400 retest candle locked (from upside) |

### Cycle 223 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 537.25 | 547.43 | 547.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 532.95 | 544.53 | 546.60 | Break + close below crossover candle low |

### Cycle 224 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 576.40 | 549.59 | 548.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 10:15:00 | 585.90 | 556.85 | 551.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 12:15:00 | 569.65 | 571.48 | 564.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 13:00:00 | 569.65 | 571.48 | 564.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 562.30 | 569.25 | 565.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 578.85 | 569.25 | 565.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 571.60 | 569.25 | 565.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 11:30:00 | 571.00 | 569.71 | 566.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 11:15:00 | 564.30 | 573.14 | 573.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 225 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 564.30 | 573.14 | 573.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 561.00 | 568.65 | 570.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 09:15:00 | 574.85 | 569.89 | 571.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 574.85 | 569.89 | 571.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 574.85 | 569.89 | 571.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:00:00 | 574.85 | 569.89 | 571.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 565.50 | 569.01 | 570.79 | EMA400 retest candle locked (from downside) |

### Cycle 226 — BUY (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 13:15:00 | 574.65 | 572.15 | 571.96 | EMA200 above EMA400 |

### Cycle 227 — SELL (started 2026-03-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 14:15:00 | 567.50 | 571.22 | 571.55 | EMA200 below EMA400 |

### Cycle 228 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 585.15 | 573.01 | 572.25 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 12:15:00 | 572.55 | 574.91 | 575.22 | EMA200 below EMA400 |

### Cycle 230 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 582.80 | 576.48 | 575.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 591.55 | 581.32 | 578.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 594.25 | 595.89 | 590.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 15:00:00 | 594.25 | 595.89 | 590.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 591.00 | 594.91 | 590.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 602.40 | 594.91 | 590.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 597.00 | 602.83 | 602.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 09:15:00 | 662.64 | 636.98 | 623.57 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 231 — SELL (started 2026-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 13:15:00 | 692.35 | 697.33 | 697.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 14:15:00 | 689.80 | 695.83 | 697.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 698.65 | 695.49 | 696.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 698.65 | 695.49 | 696.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 698.65 | 695.49 | 696.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:30:00 | 696.00 | 695.49 | 696.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 692.65 | 694.92 | 696.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 11:15:00 | 691.50 | 694.92 | 696.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 14:30:00 | 690.70 | 691.86 | 694.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:30:00 | 691.50 | 691.81 | 693.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:00:00 | 690.70 | 691.81 | 693.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 687.00 | 690.85 | 693.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 14:00:00 | 683.20 | 688.92 | 691.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 13:15:00 | 656.92 | 672.00 | 680.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 13:15:00 | 656.16 | 672.00 | 680.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 13:15:00 | 656.92 | 672.00 | 680.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 13:15:00 | 656.16 | 672.00 | 680.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 687.00 | 674.53 | 679.76 | SL hit (close>ema200) qty=0.50 sl=674.53 alert=retest2 |

### Cycle 232 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 694.30 | 683.10 | 682.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 13:15:00 | 696.90 | 685.86 | 684.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 730.20 | 733.55 | 721.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:00:00 | 730.20 | 733.55 | 721.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 721.65 | 733.04 | 727.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 721.65 | 733.04 | 727.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 727.55 | 731.94 | 727.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 15:15:00 | 734.70 | 725.39 | 725.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 09:30:00 | 729.00 | 728.32 | 726.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 13:15:00 | 719.40 | 725.04 | 725.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 233 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 719.40 | 725.04 | 725.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 14:15:00 | 716.45 | 723.32 | 724.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 15:15:00 | 702.00 | 700.86 | 709.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-07 09:15:00 | 704.85 | 700.86 | 709.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 705.25 | 701.74 | 709.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 695.50 | 704.07 | 707.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-06 09:15:00 | 171.45 | 2023-06-06 11:15:00 | 169.35 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2023-06-06 15:00:00 | 170.00 | 2023-06-07 09:15:00 | 168.35 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2023-06-07 09:15:00 | 170.00 | 2023-06-07 09:15:00 | 168.35 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2023-06-13 11:00:00 | 163.00 | 2023-06-20 10:15:00 | 161.80 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2023-07-14 09:30:00 | 171.15 | 2023-07-21 12:15:00 | 176.45 | STOP_HIT | 1.00 | 3.10% |
| BUY | retest2 | 2023-07-14 11:00:00 | 171.20 | 2023-07-21 12:15:00 | 176.45 | STOP_HIT | 1.00 | 3.07% |
| BUY | retest2 | 2023-07-26 09:15:00 | 183.50 | 2023-07-26 14:15:00 | 180.75 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2023-07-26 10:00:00 | 182.60 | 2023-07-26 14:15:00 | 180.75 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2023-07-26 10:45:00 | 182.60 | 2023-07-26 14:15:00 | 180.75 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2023-07-27 11:15:00 | 181.45 | 2023-07-31 15:15:00 | 181.35 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2023-07-27 11:45:00 | 181.40 | 2023-07-31 15:15:00 | 181.35 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2023-07-31 11:00:00 | 181.50 | 2023-07-31 15:15:00 | 181.35 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2023-07-31 11:45:00 | 180.85 | 2023-07-31 15:15:00 | 181.35 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2023-08-04 13:15:00 | 171.55 | 2023-08-09 09:15:00 | 173.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2023-08-04 13:45:00 | 171.05 | 2023-08-09 09:15:00 | 173.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2023-08-31 13:30:00 | 248.30 | 2023-09-04 11:15:00 | 245.65 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2023-09-11 09:15:00 | 268.30 | 2023-09-11 14:15:00 | 295.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-15 09:15:00 | 286.40 | 2023-09-21 14:15:00 | 284.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2023-10-06 13:30:00 | 333.70 | 2023-10-09 10:15:00 | 322.50 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest1 | 2023-10-16 09:15:00 | 415.80 | 2023-10-18 09:15:00 | 410.40 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest1 | 2023-10-26 09:15:00 | 349.25 | 2023-10-27 09:15:00 | 366.80 | STOP_HIT | 1.00 | -5.03% |
| SELL | retest1 | 2023-11-03 15:00:00 | 326.75 | 2023-11-07 09:15:00 | 310.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2023-11-03 15:00:00 | 326.75 | 2023-11-07 13:15:00 | 319.20 | STOP_HIT | 0.50 | 2.31% |
| BUY | retest2 | 2023-11-22 10:15:00 | 413.00 | 2023-11-24 13:15:00 | 401.60 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2023-12-04 09:15:00 | 441.50 | 2023-12-07 14:15:00 | 422.85 | STOP_HIT | 1.00 | -4.22% |
| BUY | retest2 | 2023-12-05 09:15:00 | 435.70 | 2023-12-07 14:15:00 | 422.85 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2023-12-05 10:45:00 | 432.50 | 2023-12-07 14:15:00 | 422.85 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2023-12-05 12:00:00 | 432.05 | 2023-12-07 14:15:00 | 422.85 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2023-12-28 10:15:00 | 413.45 | 2023-12-29 09:15:00 | 406.55 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2023-12-28 11:00:00 | 412.50 | 2023-12-29 09:15:00 | 406.55 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2023-12-28 13:45:00 | 418.30 | 2023-12-29 09:15:00 | 406.55 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2024-01-31 11:15:00 | 454.00 | 2024-02-02 09:15:00 | 469.75 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2024-01-31 11:45:00 | 453.10 | 2024-02-02 09:15:00 | 469.75 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2024-01-31 13:15:00 | 453.05 | 2024-02-02 09:15:00 | 469.75 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2024-01-31 14:15:00 | 453.10 | 2024-02-02 09:15:00 | 469.75 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2024-02-01 11:45:00 | 451.70 | 2024-02-02 09:15:00 | 469.75 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest2 | 2024-02-01 15:00:00 | 453.15 | 2024-02-02 09:15:00 | 469.75 | STOP_HIT | 1.00 | -3.66% |
| SELL | retest2 | 2024-02-15 11:45:00 | 415.70 | 2024-02-15 12:15:00 | 416.15 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2024-03-15 11:30:00 | 338.80 | 2024-03-18 11:15:00 | 360.30 | STOP_HIT | 1.00 | -6.35% |
| SELL | retest2 | 2024-03-27 15:15:00 | 346.20 | 2024-04-01 10:15:00 | 367.15 | STOP_HIT | 1.00 | -6.05% |
| SELL | retest2 | 2024-03-28 14:45:00 | 345.00 | 2024-04-01 10:15:00 | 367.15 | STOP_HIT | 1.00 | -6.42% |
| BUY | retest2 | 2024-04-08 09:15:00 | 393.25 | 2024-04-09 13:15:00 | 383.80 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2024-04-09 15:00:00 | 393.40 | 2024-04-15 10:15:00 | 393.65 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2024-04-18 14:30:00 | 385.55 | 2024-04-22 14:15:00 | 391.45 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-04-22 14:15:00 | 389.35 | 2024-04-22 14:15:00 | 391.45 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-04-30 12:15:00 | 430.00 | 2024-04-30 14:15:00 | 420.70 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-04-30 13:30:00 | 429.45 | 2024-04-30 14:15:00 | 420.70 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-05-06 12:00:00 | 437.70 | 2024-05-07 10:15:00 | 419.60 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest2 | 2024-05-08 14:15:00 | 409.20 | 2024-05-09 14:15:00 | 388.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-08 14:15:00 | 409.20 | 2024-05-10 14:15:00 | 391.25 | STOP_HIT | 0.50 | 4.39% |
| BUY | retest2 | 2024-05-17 09:15:00 | 412.10 | 2024-05-22 15:15:00 | 422.00 | STOP_HIT | 1.00 | 2.40% |
| BUY | retest2 | 2024-05-17 09:45:00 | 412.95 | 2024-05-22 15:15:00 | 422.00 | STOP_HIT | 1.00 | 2.19% |
| BUY | retest1 | 2024-06-12 09:15:00 | 391.55 | 2024-06-13 10:15:00 | 386.60 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest1 | 2024-06-12 12:30:00 | 390.10 | 2024-06-13 10:15:00 | 386.60 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-06-13 15:15:00 | 387.60 | 2024-06-21 15:15:00 | 399.00 | STOP_HIT | 1.00 | 2.94% |
| BUY | retest2 | 2024-06-14 09:30:00 | 391.60 | 2024-06-21 15:15:00 | 399.00 | STOP_HIT | 1.00 | 1.89% |
| SELL | retest2 | 2024-07-02 11:30:00 | 392.70 | 2024-07-04 09:15:00 | 404.90 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2024-07-10 09:15:00 | 426.25 | 2024-07-12 14:15:00 | 417.45 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-07-26 09:15:00 | 411.00 | 2024-07-26 10:15:00 | 404.90 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest1 | 2024-08-06 13:30:00 | 374.65 | 2024-08-08 09:15:00 | 378.20 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-08-08 13:15:00 | 373.75 | 2024-08-19 12:15:00 | 364.90 | STOP_HIT | 1.00 | 2.37% |
| SELL | retest2 | 2024-08-08 13:45:00 | 372.60 | 2024-08-19 12:15:00 | 364.90 | STOP_HIT | 1.00 | 2.07% |
| BUY | retest2 | 2024-08-21 12:30:00 | 369.85 | 2024-08-26 11:15:00 | 368.95 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2024-08-21 14:00:00 | 370.65 | 2024-08-26 11:15:00 | 368.95 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2024-09-02 11:45:00 | 365.80 | 2024-09-03 12:15:00 | 368.15 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-09-04 13:00:00 | 365.00 | 2024-09-06 09:15:00 | 373.40 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2024-09-04 14:30:00 | 364.25 | 2024-09-06 09:15:00 | 373.40 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2024-09-05 10:30:00 | 364.95 | 2024-09-06 09:15:00 | 373.40 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2024-09-05 11:00:00 | 365.05 | 2024-09-06 09:15:00 | 373.40 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2024-09-19 09:15:00 | 372.45 | 2024-09-19 09:15:00 | 366.55 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-09-24 11:15:00 | 365.75 | 2024-10-03 14:15:00 | 347.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-25 10:00:00 | 365.00 | 2024-10-03 14:15:00 | 346.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-24 11:15:00 | 365.75 | 2024-10-07 09:15:00 | 329.18 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-25 10:00:00 | 365.00 | 2024-10-07 09:15:00 | 328.50 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-10-16 09:30:00 | 344.00 | 2024-10-22 10:15:00 | 343.15 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2024-10-16 12:45:00 | 343.30 | 2024-10-22 10:15:00 | 343.15 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2024-10-16 15:00:00 | 354.00 | 2024-10-22 10:15:00 | 343.15 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2024-11-29 09:15:00 | 348.55 | 2024-11-29 11:15:00 | 343.50 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-11-29 09:45:00 | 348.95 | 2024-11-29 11:15:00 | 343.50 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-12-02 10:00:00 | 348.50 | 2024-12-12 14:15:00 | 361.25 | STOP_HIT | 1.00 | 3.66% |
| BUY | retest2 | 2024-12-02 15:15:00 | 353.00 | 2024-12-12 14:15:00 | 361.25 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest2 | 2024-12-06 10:15:00 | 362.50 | 2024-12-12 14:15:00 | 361.25 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2024-12-06 12:15:00 | 362.15 | 2024-12-12 14:15:00 | 361.25 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2024-12-10 09:30:00 | 362.25 | 2024-12-12 14:15:00 | 361.25 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2024-12-10 14:15:00 | 362.60 | 2024-12-12 14:15:00 | 361.25 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-12-11 13:45:00 | 370.50 | 2024-12-12 14:15:00 | 361.25 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2024-12-20 12:30:00 | 334.60 | 2024-12-30 13:15:00 | 317.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 12:30:00 | 334.60 | 2024-12-31 10:15:00 | 319.40 | STOP_HIT | 0.50 | 4.54% |
| SELL | retest2 | 2025-01-09 11:00:00 | 312.60 | 2025-01-10 13:15:00 | 296.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 11:00:00 | 312.60 | 2025-01-13 14:15:00 | 281.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 14:00:00 | 303.75 | 2025-01-27 13:15:00 | 288.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 15:15:00 | 301.00 | 2025-01-28 09:15:00 | 285.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 14:00:00 | 303.75 | 2025-01-28 13:15:00 | 290.80 | STOP_HIT | 0.50 | 4.26% |
| SELL | retest2 | 2025-01-24 15:15:00 | 301.00 | 2025-01-28 13:15:00 | 290.80 | STOP_HIT | 0.50 | 3.39% |
| SELL | retest2 | 2025-02-19 13:00:00 | 263.20 | 2025-02-20 09:15:00 | 269.50 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-02-19 13:45:00 | 263.85 | 2025-02-20 09:15:00 | 269.50 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-02-19 14:15:00 | 262.60 | 2025-02-20 09:15:00 | 269.50 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2025-03-04 11:30:00 | 243.86 | 2025-03-05 09:15:00 | 247.70 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-04-02 12:30:00 | 276.85 | 2025-04-04 14:15:00 | 273.20 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-04-02 13:00:00 | 276.95 | 2025-04-04 14:15:00 | 273.20 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-04-04 10:30:00 | 278.50 | 2025-04-04 14:15:00 | 273.20 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-04-04 13:15:00 | 276.30 | 2025-04-04 14:15:00 | 273.20 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-04-08 10:30:00 | 262.10 | 2025-04-09 12:15:00 | 273.15 | STOP_HIT | 1.00 | -4.22% |
| SELL | retest2 | 2025-04-09 09:15:00 | 262.40 | 2025-04-09 12:15:00 | 273.15 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2025-04-23 11:30:00 | 324.20 | 2025-04-25 09:15:00 | 311.95 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2025-04-23 14:45:00 | 324.75 | 2025-04-25 09:15:00 | 311.95 | STOP_HIT | 1.00 | -3.94% |
| SELL | retest2 | 2025-04-29 11:30:00 | 314.85 | 2025-05-06 12:15:00 | 299.25 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2025-04-29 13:30:00 | 315.00 | 2025-05-06 14:15:00 | 299.11 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2025-04-29 11:30:00 | 314.85 | 2025-05-07 11:15:00 | 301.60 | STOP_HIT | 0.50 | 4.21% |
| SELL | retest2 | 2025-04-29 13:30:00 | 315.00 | 2025-05-07 11:15:00 | 301.60 | STOP_HIT | 0.50 | 4.25% |
| BUY | retest2 | 2025-05-14 09:15:00 | 316.05 | 2025-05-16 12:15:00 | 347.66 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-28 14:15:00 | 357.40 | 2025-05-30 13:15:00 | 357.65 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2025-05-28 15:00:00 | 357.00 | 2025-05-30 13:15:00 | 357.65 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2025-05-30 11:00:00 | 357.50 | 2025-05-30 13:15:00 | 357.65 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2025-06-04 11:15:00 | 366.25 | 2025-06-06 12:15:00 | 402.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-25 09:15:00 | 406.50 | 2025-07-03 12:15:00 | 411.10 | STOP_HIT | 1.00 | 1.13% |
| SELL | retest2 | 2025-07-09 15:15:00 | 389.85 | 2025-07-18 11:15:00 | 396.25 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-07-10 09:30:00 | 389.85 | 2025-07-18 11:15:00 | 396.25 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest1 | 2025-08-18 09:15:00 | 427.85 | 2025-08-19 14:15:00 | 428.20 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2025-08-20 13:15:00 | 429.45 | 2025-08-20 14:15:00 | 420.40 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-08-28 12:30:00 | 415.15 | 2025-09-01 09:15:00 | 423.75 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-08-28 14:45:00 | 414.75 | 2025-09-01 09:15:00 | 423.75 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-09-05 10:15:00 | 483.70 | 2025-09-08 09:15:00 | 532.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-17 09:15:00 | 555.65 | 2025-09-18 09:15:00 | 543.60 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-09-17 15:15:00 | 551.55 | 2025-09-18 09:15:00 | 543.60 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-09-23 09:15:00 | 579.55 | 2025-09-23 12:15:00 | 637.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-03 14:45:00 | 614.55 | 2025-10-07 10:15:00 | 591.95 | STOP_HIT | 1.00 | -3.68% |
| BUY | retest2 | 2025-10-06 09:15:00 | 618.75 | 2025-10-07 10:15:00 | 591.95 | STOP_HIT | 1.00 | -4.33% |
| BUY | retest2 | 2025-10-06 11:45:00 | 615.25 | 2025-10-07 10:15:00 | 591.95 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2025-11-11 12:15:00 | 593.70 | 2025-11-13 13:15:00 | 592.45 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-11-11 13:15:00 | 594.00 | 2025-11-13 13:15:00 | 592.45 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-11-11 13:45:00 | 595.15 | 2025-11-13 13:15:00 | 592.45 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-11-13 09:15:00 | 599.05 | 2025-11-13 13:15:00 | 592.45 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-11-26 12:15:00 | 488.10 | 2025-11-26 12:15:00 | 501.80 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest1 | 2025-12-01 09:45:00 | 557.55 | 2025-12-02 13:15:00 | 543.40 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-12-04 11:45:00 | 526.50 | 2025-12-08 10:15:00 | 500.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 09:15:00 | 522.70 | 2025-12-08 11:15:00 | 496.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 11:45:00 | 526.50 | 2025-12-09 09:15:00 | 473.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-05 09:15:00 | 522.70 | 2025-12-09 14:15:00 | 494.05 | STOP_HIT | 0.50 | 5.48% |
| BUY | retest2 | 2025-12-30 14:15:00 | 593.90 | 2026-01-07 15:15:00 | 606.10 | STOP_HIT | 1.00 | 2.05% |
| BUY | retest2 | 2025-12-30 15:00:00 | 597.00 | 2026-01-07 15:15:00 | 606.10 | STOP_HIT | 1.00 | 1.52% |
| SELL | retest2 | 2026-01-13 12:00:00 | 565.65 | 2026-01-14 09:15:00 | 576.05 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-01-13 12:45:00 | 564.35 | 2026-01-14 09:15:00 | 576.05 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-01-13 13:45:00 | 566.80 | 2026-01-14 09:15:00 | 576.05 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2026-01-22 11:45:00 | 534.40 | 2026-01-28 11:15:00 | 555.35 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2026-01-22 12:45:00 | 535.15 | 2026-01-28 11:15:00 | 555.35 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2026-01-23 11:30:00 | 533.75 | 2026-01-28 11:15:00 | 555.35 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2026-01-28 09:45:00 | 534.35 | 2026-01-28 11:15:00 | 555.35 | STOP_HIT | 1.00 | -3.93% |
| BUY | retest2 | 2026-02-01 10:00:00 | 597.50 | 2026-02-02 10:15:00 | 561.00 | STOP_HIT | 1.00 | -6.11% |
| BUY | retest2 | 2026-02-01 12:30:00 | 585.00 | 2026-02-02 10:15:00 | 561.00 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2026-02-01 14:30:00 | 585.95 | 2026-02-02 10:15:00 | 561.00 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2026-02-12 13:00:00 | 592.90 | 2026-02-16 09:15:00 | 563.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 14:30:00 | 592.30 | 2026-02-16 09:15:00 | 562.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 13:00:00 | 592.90 | 2026-02-18 09:15:00 | 554.55 | STOP_HIT | 0.50 | 6.47% |
| SELL | retest2 | 2026-02-12 14:30:00 | 592.30 | 2026-02-18 09:15:00 | 554.55 | STOP_HIT | 0.50 | 6.37% |
| BUY | retest2 | 2026-02-23 09:15:00 | 574.60 | 2026-02-23 09:15:00 | 567.50 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-03-05 11:15:00 | 546.55 | 2026-03-06 09:15:00 | 553.65 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-03-05 12:30:00 | 547.40 | 2026-03-06 09:15:00 | 553.65 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-03-06 10:45:00 | 546.60 | 2026-03-06 12:15:00 | 551.50 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2026-03-12 10:45:00 | 551.35 | 2026-03-13 11:15:00 | 541.20 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2026-03-24 09:15:00 | 578.85 | 2026-03-27 11:15:00 | 564.30 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2026-03-24 10:15:00 | 571.60 | 2026-03-27 11:15:00 | 564.30 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2026-03-24 11:30:00 | 571.00 | 2026-03-27 11:15:00 | 564.30 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-04-08 09:15:00 | 602.40 | 2026-04-16 09:15:00 | 662.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 10:15:00 | 597.00 | 2026-04-16 09:15:00 | 656.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-22 11:15:00 | 691.50 | 2026-04-24 13:15:00 | 656.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-22 14:30:00 | 690.70 | 2026-04-24 13:15:00 | 656.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 09:30:00 | 691.50 | 2026-04-24 13:15:00 | 656.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 10:00:00 | 690.70 | 2026-04-24 13:15:00 | 656.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-22 11:15:00 | 691.50 | 2026-04-27 09:15:00 | 687.00 | STOP_HIT | 0.50 | 0.65% |
| SELL | retest2 | 2026-04-22 14:30:00 | 690.70 | 2026-04-27 09:15:00 | 687.00 | STOP_HIT | 0.50 | 0.54% |
| SELL | retest2 | 2026-04-23 09:30:00 | 691.50 | 2026-04-27 09:15:00 | 687.00 | STOP_HIT | 0.50 | 0.65% |
| SELL | retest2 | 2026-04-23 10:00:00 | 690.70 | 2026-04-27 09:15:00 | 687.00 | STOP_HIT | 0.50 | 0.54% |
| SELL | retest2 | 2026-04-23 14:00:00 | 683.20 | 2026-04-27 12:15:00 | 694.30 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-04-27 10:30:00 | 686.10 | 2026-04-27 12:15:00 | 694.30 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-05-04 15:15:00 | 734.70 | 2026-05-05 13:15:00 | 719.40 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2026-05-05 09:30:00 | 729.00 | 2026-05-05 13:15:00 | 719.40 | STOP_HIT | 1.00 | -1.32% |
