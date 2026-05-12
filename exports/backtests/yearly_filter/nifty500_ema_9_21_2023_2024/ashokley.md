# Ashok Leyland Ltd. (ASHOKLEY)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 168.77
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 218 |
| ALERT1 | 152 |
| ALERT2 | 152 |
| ALERT2_SKIP | 81 |
| ALERT3 | 416 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 194 |
| PARTIAL | 10 |
| TARGET_HIT | 6 |
| STOP_HIT | 202 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 210 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 65 / 145
- **Target hits / Stop hits / Partials:** 6 / 194 / 10
- **Avg / median % per leg:** 0.09% / -0.66%
- **Sum % (uncompounded):** 18.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 103 | 24 | 23.3% | 5 | 97 | 1 | -0.05% | -5.2% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 3 | 1 | 2.82% | 11.3% |
| BUY @ 3rd Alert (retest2) | 99 | 20 | 20.2% | 5 | 94 | 0 | -0.17% | -16.4% |
| SELL (all) | 107 | 41 | 38.3% | 1 | 97 | 9 | 0.23% | 24.1% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 3 | 0 | -0.24% | -0.7% |
| SELL @ 3rd Alert (retest2) | 104 | 39 | 37.5% | 1 | 94 | 9 | 0.24% | 24.8% |
| retest1 (combined) | 7 | 6 | 85.7% | 0 | 6 | 1 | 1.51% | 10.5% |
| retest2 (combined) | 203 | 59 | 29.1% | 6 | 188 | 9 | 0.04% | 8.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 12:15:00 | 76.38 | 76.60 | 76.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-23 13:15:00 | 76.35 | 76.55 | 76.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-24 14:15:00 | 75.43 | 75.36 | 75.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-24 15:00:00 | 75.43 | 75.36 | 75.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 72.83 | 72.87 | 73.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-30 10:45:00 | 72.55 | 72.84 | 73.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-01 09:15:00 | 73.15 | 73.06 | 73.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-06-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 09:15:00 | 73.15 | 73.06 | 73.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 10:15:00 | 73.65 | 73.17 | 73.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 09:15:00 | 72.72 | 73.33 | 73.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 09:15:00 | 72.72 | 73.33 | 73.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 72.72 | 73.33 | 73.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 09:30:00 | 72.75 | 73.33 | 73.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 11:15:00 | 72.95 | 73.23 | 73.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 12:00:00 | 72.95 | 73.23 | 73.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2023-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 12:15:00 | 72.97 | 73.18 | 73.20 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 09:15:00 | 74.33 | 73.37 | 73.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 10:15:00 | 74.70 | 73.64 | 73.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-09 13:15:00 | 76.40 | 76.48 | 76.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-09 14:00:00 | 76.40 | 76.48 | 76.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 14:15:00 | 76.18 | 76.42 | 76.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 14:30:00 | 76.25 | 76.42 | 76.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 15:15:00 | 76.10 | 76.35 | 76.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-12 09:15:00 | 76.25 | 76.35 | 76.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 76.53 | 76.39 | 76.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-13 09:15:00 | 77.08 | 76.38 | 76.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-06-19 09:15:00 | 84.79 | 82.28 | 80.65 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 15:15:00 | 82.10 | 82.60 | 82.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 80.10 | 81.87 | 82.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 12:15:00 | 81.35 | 80.90 | 81.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 12:15:00 | 81.35 | 80.90 | 81.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 12:15:00 | 81.35 | 80.90 | 81.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 13:00:00 | 81.35 | 80.90 | 81.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 13:15:00 | 81.75 | 81.07 | 81.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 13:30:00 | 81.70 | 81.07 | 81.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 82.23 | 81.30 | 81.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 14:45:00 | 82.28 | 81.30 | 81.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2023-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 09:15:00 | 81.98 | 81.53 | 81.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 13:15:00 | 83.03 | 82.36 | 82.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 11:15:00 | 83.83 | 83.94 | 83.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-03 11:30:00 | 83.90 | 83.94 | 83.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 13:15:00 | 83.53 | 83.82 | 83.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 13:30:00 | 83.60 | 83.82 | 83.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 14:15:00 | 83.53 | 83.76 | 83.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 14:45:00 | 83.48 | 83.76 | 83.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 15:15:00 | 84.00 | 83.81 | 83.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 09:15:00 | 83.48 | 83.81 | 83.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 82.65 | 83.58 | 83.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 10:00:00 | 82.65 | 83.58 | 83.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 10:15:00 | 82.28 | 83.32 | 83.28 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 11:15:00 | 82.05 | 83.06 | 83.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 12:15:00 | 81.80 | 82.81 | 83.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 11:15:00 | 82.43 | 82.20 | 82.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-05 12:00:00 | 82.43 | 82.20 | 82.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 12:15:00 | 81.90 | 82.14 | 82.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-06 12:30:00 | 81.75 | 82.26 | 82.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-07 09:30:00 | 81.75 | 82.29 | 82.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-07 10:30:00 | 81.47 | 82.14 | 82.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-07 12:00:00 | 81.58 | 82.03 | 82.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 14:15:00 | 82.03 | 81.86 | 82.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-07 14:45:00 | 82.13 | 81.86 | 82.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 82.90 | 82.07 | 82.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-07-10 09:15:00 | 82.90 | 82.07 | 82.15 | SL hit (close>static) qty=1.00 sl=82.60 alert=retest2 |

### Cycle 8 — BUY (started 2023-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 12:15:00 | 82.48 | 82.09 | 82.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 13:15:00 | 82.55 | 82.19 | 82.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 13:15:00 | 84.68 | 85.14 | 84.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-13 14:00:00 | 84.68 | 85.14 | 84.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 09:15:00 | 84.90 | 85.09 | 84.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-14 09:30:00 | 84.68 | 85.09 | 84.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 11:15:00 | 85.93 | 86.32 | 85.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 12:00:00 | 85.93 | 86.32 | 85.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 12:15:00 | 86.20 | 86.29 | 85.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 13:15:00 | 86.38 | 86.29 | 85.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-19 14:45:00 | 86.50 | 86.47 | 86.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 10:00:00 | 86.55 | 86.49 | 86.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 11:00:00 | 86.50 | 86.49 | 86.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 11:15:00 | 86.68 | 86.53 | 86.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 13:30:00 | 87.05 | 86.65 | 86.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-28 11:15:00 | 90.20 | 90.67 | 90.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2023-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 11:15:00 | 90.20 | 90.67 | 90.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-28 12:15:00 | 89.50 | 90.44 | 90.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 09:15:00 | 90.43 | 90.13 | 90.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 09:15:00 | 90.43 | 90.13 | 90.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 90.43 | 90.13 | 90.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 09:30:00 | 90.45 | 90.13 | 90.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 10:15:00 | 91.03 | 90.31 | 90.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 11:00:00 | 91.03 | 90.31 | 90.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2023-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 11:15:00 | 91.25 | 90.50 | 90.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 14:15:00 | 91.88 | 90.95 | 90.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 09:15:00 | 91.83 | 91.84 | 91.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-02 10:00:00 | 91.83 | 91.84 | 91.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 91.50 | 91.77 | 91.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 10:30:00 | 91.45 | 91.77 | 91.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 90.85 | 91.59 | 91.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 12:00:00 | 90.85 | 91.59 | 91.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 12:15:00 | 90.53 | 91.38 | 91.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 13:00:00 | 90.53 | 91.38 | 91.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 89.63 | 91.03 | 91.17 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 11:15:00 | 91.53 | 90.92 | 90.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 11:15:00 | 93.18 | 91.97 | 91.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 13:15:00 | 93.00 | 93.59 | 92.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-10 14:00:00 | 93.00 | 93.59 | 92.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 14:15:00 | 93.23 | 93.52 | 92.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-11 09:15:00 | 94.73 | 93.40 | 92.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-11 11:30:00 | 93.55 | 93.55 | 93.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-11 14:30:00 | 93.55 | 93.50 | 93.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-14 10:45:00 | 93.75 | 93.39 | 93.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 11:15:00 | 92.88 | 93.29 | 93.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 11:45:00 | 92.78 | 93.29 | 93.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 12:15:00 | 92.88 | 93.21 | 93.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 12:45:00 | 92.68 | 93.21 | 93.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 14:15:00 | 93.40 | 93.24 | 93.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-16 11:15:00 | 94.18 | 93.30 | 93.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-18 11:15:00 | 92.93 | 93.70 | 93.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 11:15:00 | 92.93 | 93.70 | 93.75 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 12:15:00 | 93.90 | 93.69 | 93.67 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-21 14:15:00 | 93.33 | 93.65 | 93.66 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 09:15:00 | 93.95 | 93.69 | 93.67 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-08-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 09:15:00 | 93.48 | 93.78 | 93.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 10:15:00 | 92.93 | 93.61 | 93.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-25 09:15:00 | 93.65 | 93.24 | 93.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 09:15:00 | 93.65 | 93.24 | 93.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 93.65 | 93.24 | 93.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-25 10:00:00 | 93.65 | 93.24 | 93.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 10:15:00 | 92.63 | 93.12 | 93.36 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 11:15:00 | 93.95 | 93.39 | 93.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 09:15:00 | 94.08 | 93.71 | 93.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-29 14:15:00 | 94.10 | 94.14 | 93.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-29 14:45:00 | 93.98 | 94.14 | 93.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 14:15:00 | 94.00 | 94.52 | 94.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 14:45:00 | 93.78 | 94.52 | 94.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 15:15:00 | 94.00 | 94.42 | 94.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:30:00 | 93.98 | 94.32 | 94.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 93.63 | 94.18 | 94.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 11:00:00 | 93.63 | 94.18 | 94.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2023-08-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 11:15:00 | 93.45 | 94.04 | 94.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 12:15:00 | 92.98 | 93.83 | 93.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-04 09:15:00 | 92.28 | 92.24 | 92.76 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-04 10:15:00 | 92.08 | 92.24 | 92.76 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-04 13:45:00 | 91.95 | 92.05 | 92.49 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 09:15:00 | 92.05 | 92.04 | 92.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-05 09:45:00 | 92.30 | 92.04 | 92.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 09:15:00 | 91.53 | 91.31 | 91.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 09:45:00 | 91.45 | 91.31 | 91.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 10:15:00 | 91.43 | 91.34 | 91.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 10:30:00 | 91.63 | 91.34 | 91.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 91.73 | 91.31 | 91.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-09-08 09:15:00 | 91.73 | 91.31 | 91.46 | SL hit (close>ema400) qty=1.00 sl=91.46 alert=retest1 |

### Cycle 20 — BUY (started 2023-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 11:15:00 | 92.55 | 91.61 | 91.58 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 10:15:00 | 91.13 | 91.74 | 91.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 11:15:00 | 89.45 | 91.29 | 91.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 15:15:00 | 89.98 | 89.85 | 90.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-14 09:15:00 | 90.15 | 89.85 | 90.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 89.73 | 89.83 | 90.31 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 10:15:00 | 90.95 | 90.28 | 90.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 13:15:00 | 91.63 | 90.69 | 90.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-20 10:15:00 | 91.20 | 91.43 | 91.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 10:15:00 | 91.20 | 91.43 | 91.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 10:15:00 | 91.20 | 91.43 | 91.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 10:30:00 | 91.35 | 91.43 | 91.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 11:15:00 | 91.48 | 91.44 | 91.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-20 12:15:00 | 91.58 | 91.44 | 91.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-20 13:15:00 | 91.05 | 91.36 | 91.19 | SL hit (close<static) qty=1.00 sl=91.18 alert=retest2 |

### Cycle 23 — SELL (started 2023-09-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 10:15:00 | 90.53 | 91.06 | 91.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 11:15:00 | 89.95 | 90.83 | 90.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 12:15:00 | 89.98 | 89.45 | 89.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 12:15:00 | 89.98 | 89.45 | 89.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 12:15:00 | 89.98 | 89.45 | 89.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 13:00:00 | 89.98 | 89.45 | 89.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 13:15:00 | 90.10 | 89.58 | 89.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 13:30:00 | 90.30 | 89.58 | 89.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 14:15:00 | 91.38 | 89.94 | 90.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 15:00:00 | 91.38 | 89.94 | 90.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2023-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 15:15:00 | 91.38 | 90.23 | 90.13 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 12:15:00 | 89.93 | 90.07 | 90.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 09:15:00 | 89.30 | 89.61 | 89.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 09:15:00 | 88.10 | 88.07 | 88.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-29 10:00:00 | 88.10 | 88.07 | 88.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 13:15:00 | 88.73 | 88.34 | 88.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 14:00:00 | 88.73 | 88.34 | 88.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 14:15:00 | 88.40 | 88.35 | 88.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 15:00:00 | 88.40 | 88.35 | 88.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 89.30 | 88.58 | 88.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-03 09:30:00 | 89.65 | 88.58 | 88.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 13:15:00 | 88.60 | 88.74 | 88.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-04 09:15:00 | 87.93 | 88.76 | 88.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-10 10:15:00 | 87.58 | 86.25 | 86.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2023-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 10:15:00 | 87.58 | 86.25 | 86.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 88.13 | 86.93 | 86.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 09:15:00 | 87.73 | 87.76 | 87.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-12 10:00:00 | 87.73 | 87.76 | 87.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 12:15:00 | 87.18 | 87.59 | 87.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 13:00:00 | 87.18 | 87.59 | 87.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 13:15:00 | 87.00 | 87.47 | 87.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 13:45:00 | 86.98 | 87.47 | 87.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 15:15:00 | 87.35 | 87.41 | 87.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 09:15:00 | 87.38 | 87.41 | 87.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 88.00 | 87.53 | 87.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 11:00:00 | 88.10 | 87.64 | 87.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 13:15:00 | 88.05 | 87.78 | 87.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 14:45:00 | 88.23 | 87.83 | 87.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 09:15:00 | 88.18 | 87.82 | 87.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 13:15:00 | 88.20 | 88.46 | 88.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-17 14:00:00 | 88.20 | 88.46 | 88.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 14:15:00 | 88.23 | 88.42 | 88.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 15:15:00 | 88.45 | 88.42 | 88.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 12:15:00 | 87.95 | 88.46 | 88.34 | SL hit (close<static) qty=1.00 sl=88.15 alert=retest2 |

### Cycle 27 — SELL (started 2023-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 15:15:00 | 87.95 | 88.22 | 88.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 86.95 | 87.97 | 88.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 13:15:00 | 88.18 | 87.83 | 87.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 13:15:00 | 88.18 | 87.83 | 87.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 13:15:00 | 88.18 | 87.83 | 87.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 14:00:00 | 88.18 | 87.83 | 87.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 14:15:00 | 88.38 | 87.94 | 88.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 15:00:00 | 88.38 | 87.94 | 88.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 15:15:00 | 88.18 | 87.99 | 88.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 09:15:00 | 88.30 | 87.99 | 88.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 88.08 | 88.00 | 88.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 10:30:00 | 87.68 | 87.90 | 87.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 83.30 | 84.49 | 85.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-27 09:15:00 | 83.83 | 83.61 | 84.37 | SL hit (close>ema200) qty=0.50 sl=83.61 alert=retest2 |

### Cycle 28 — BUY (started 2023-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 10:15:00 | 84.90 | 84.11 | 84.05 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 12:15:00 | 83.70 | 83.96 | 83.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 09:15:00 | 83.38 | 83.79 | 83.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 09:15:00 | 83.60 | 83.14 | 83.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 09:15:00 | 83.60 | 83.14 | 83.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 83.60 | 83.14 | 83.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 09:30:00 | 83.63 | 83.14 | 83.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 10:15:00 | 83.75 | 83.27 | 83.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 10:30:00 | 83.85 | 83.27 | 83.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 12:15:00 | 83.70 | 83.35 | 83.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 13:00:00 | 83.70 | 83.35 | 83.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 13:15:00 | 84.25 | 83.53 | 83.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 14:00:00 | 84.25 | 83.53 | 83.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2023-11-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 14:15:00 | 84.23 | 83.67 | 83.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 09:15:00 | 84.75 | 83.97 | 83.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 14:15:00 | 84.10 | 84.15 | 83.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-03 14:45:00 | 84.28 | 84.15 | 83.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 84.50 | 84.53 | 84.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 09:15:00 | 85.05 | 84.42 | 84.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-17 13:15:00 | 86.80 | 87.05 | 87.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2023-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 13:15:00 | 86.80 | 87.05 | 87.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-20 09:15:00 | 86.08 | 86.85 | 86.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 09:15:00 | 86.50 | 86.32 | 86.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 09:15:00 | 86.50 | 86.32 | 86.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 86.50 | 86.32 | 86.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 09:30:00 | 86.70 | 86.32 | 86.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 10:15:00 | 86.78 | 86.41 | 86.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 10:30:00 | 86.83 | 86.41 | 86.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 11:15:00 | 87.13 | 86.56 | 86.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 12:00:00 | 87.13 | 86.56 | 86.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2023-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 12:15:00 | 87.75 | 86.80 | 86.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 13:15:00 | 88.53 | 87.14 | 86.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-23 14:15:00 | 89.03 | 89.07 | 88.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-23 15:00:00 | 89.03 | 89.07 | 88.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 88.35 | 88.92 | 88.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 09:45:00 | 88.25 | 88.92 | 88.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 88.63 | 88.86 | 88.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-24 11:15:00 | 88.83 | 88.86 | 88.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-01 14:15:00 | 88.38 | 90.14 | 90.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-01 14:15:00 | 88.38 | 90.14 | 90.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-01 15:15:00 | 87.98 | 89.71 | 89.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-05 13:15:00 | 87.75 | 87.63 | 88.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-05 13:45:00 | 87.78 | 87.63 | 88.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 11:15:00 | 88.05 | 87.66 | 88.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-06 11:45:00 | 88.13 | 87.66 | 88.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 12:15:00 | 87.90 | 87.71 | 88.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-06 12:30:00 | 88.33 | 87.71 | 88.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 13:15:00 | 88.30 | 87.82 | 88.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-06 13:45:00 | 88.38 | 87.82 | 88.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 14:15:00 | 88.35 | 87.93 | 88.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-07 09:15:00 | 87.75 | 88.01 | 88.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-07 09:15:00 | 88.70 | 88.15 | 88.17 | SL hit (close>static) qty=1.00 sl=88.55 alert=retest2 |

### Cycle 34 — BUY (started 2023-12-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 11:15:00 | 88.83 | 88.29 | 88.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 14:15:00 | 88.93 | 88.47 | 88.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 09:15:00 | 88.43 | 88.52 | 88.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 09:15:00 | 88.43 | 88.52 | 88.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 09:15:00 | 88.43 | 88.52 | 88.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 10:00:00 | 88.43 | 88.52 | 88.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 10:15:00 | 87.75 | 88.36 | 88.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 11:00:00 | 87.75 | 88.36 | 88.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2023-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 11:15:00 | 87.85 | 88.26 | 88.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 13:15:00 | 87.23 | 87.95 | 88.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 09:15:00 | 88.03 | 87.85 | 88.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 09:15:00 | 88.03 | 87.85 | 88.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 88.03 | 87.85 | 88.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 09:30:00 | 87.98 | 87.85 | 88.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 10:15:00 | 87.55 | 87.79 | 87.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-12 10:15:00 | 87.33 | 87.68 | 87.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-12 12:00:00 | 87.40 | 87.59 | 87.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-14 10:15:00 | 87.98 | 87.22 | 87.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2023-12-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 10:15:00 | 87.98 | 87.22 | 87.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 11:15:00 | 88.28 | 87.44 | 87.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 10:15:00 | 87.70 | 87.82 | 87.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-15 10:45:00 | 87.80 | 87.82 | 87.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 11:15:00 | 87.13 | 87.68 | 87.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 12:00:00 | 87.13 | 87.68 | 87.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 12:15:00 | 87.20 | 87.58 | 87.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 12:30:00 | 87.08 | 87.58 | 87.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2023-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 14:15:00 | 87.08 | 87.41 | 87.45 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 14:15:00 | 87.55 | 87.45 | 87.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-20 09:15:00 | 88.10 | 87.71 | 87.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 13:15:00 | 86.00 | 87.74 | 87.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 13:15:00 | 86.00 | 87.74 | 87.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 86.00 | 87.74 | 87.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 14:00:00 | 86.00 | 87.74 | 87.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 84.93 | 87.18 | 87.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 84.78 | 86.70 | 87.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 86.05 | 85.46 | 86.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 86.05 | 85.46 | 86.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 86.05 | 85.46 | 86.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:00:00 | 86.05 | 85.46 | 86.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 86.30 | 85.63 | 86.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 11:00:00 | 86.30 | 85.63 | 86.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 11:15:00 | 86.00 | 85.70 | 86.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 13:30:00 | 85.75 | 85.79 | 86.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-26 09:15:00 | 87.35 | 86.18 | 86.20 | SL hit (close>static) qty=1.00 sl=86.38 alert=retest2 |

### Cycle 40 — BUY (started 2023-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 10:15:00 | 87.00 | 86.34 | 86.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 09:15:00 | 87.75 | 86.89 | 86.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 12:15:00 | 86.88 | 87.06 | 86.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 12:15:00 | 86.88 | 87.06 | 86.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 86.88 | 87.06 | 86.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 12:30:00 | 86.73 | 87.06 | 86.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 13:15:00 | 87.03 | 87.06 | 86.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 13:30:00 | 86.85 | 87.06 | 86.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 09:15:00 | 91.18 | 92.21 | 90.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:00:00 | 91.18 | 92.21 | 90.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 90.25 | 91.82 | 90.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:30:00 | 90.50 | 91.82 | 90.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 11:15:00 | 90.28 | 91.51 | 90.83 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2024-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 09:15:00 | 89.55 | 90.52 | 90.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 13:15:00 | 88.95 | 89.36 | 89.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 09:15:00 | 88.20 | 87.98 | 88.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 09:15:00 | 88.20 | 87.98 | 88.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 88.20 | 87.98 | 88.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-12 10:00:00 | 87.88 | 88.18 | 88.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-15 09:45:00 | 87.85 | 88.13 | 88.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-15 11:15:00 | 87.80 | 88.09 | 88.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-16 09:15:00 | 88.80 | 88.32 | 88.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2024-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 09:15:00 | 88.80 | 88.32 | 88.29 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 12:15:00 | 87.65 | 88.24 | 88.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 09:15:00 | 86.25 | 87.63 | 87.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 10:15:00 | 86.48 | 86.19 | 86.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-18 11:00:00 | 86.48 | 86.19 | 86.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 12:15:00 | 86.95 | 86.37 | 86.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 13:00:00 | 86.95 | 86.37 | 86.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 13:15:00 | 86.60 | 86.42 | 86.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 14:15:00 | 86.38 | 86.62 | 86.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-20 10:15:00 | 87.43 | 86.76 | 86.77 | SL hit (close>static) qty=1.00 sl=87.10 alert=retest2 |

### Cycle 44 — BUY (started 2024-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 11:15:00 | 87.20 | 86.85 | 86.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-20 13:15:00 | 87.50 | 87.04 | 86.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 09:15:00 | 86.75 | 87.03 | 86.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 86.75 | 87.03 | 86.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 86.75 | 87.03 | 86.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:00:00 | 86.75 | 87.03 | 86.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2024-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 10:15:00 | 86.03 | 86.83 | 86.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 11:15:00 | 85.55 | 86.58 | 86.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 14:15:00 | 85.45 | 85.16 | 85.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 15:00:00 | 85.45 | 85.16 | 85.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 85.40 | 85.26 | 85.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:30:00 | 85.85 | 85.26 | 85.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 85.70 | 85.13 | 85.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 10:00:00 | 85.70 | 85.13 | 85.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 10:15:00 | 85.60 | 85.22 | 85.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 11:15:00 | 85.98 | 85.22 | 85.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2024-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 11:15:00 | 86.80 | 85.54 | 85.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 15:15:00 | 87.10 | 86.26 | 85.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 86.38 | 86.45 | 86.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 14:15:00 | 86.38 | 86.45 | 86.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 86.38 | 86.45 | 86.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 14:30:00 | 86.28 | 86.45 | 86.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 87.00 | 86.54 | 86.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 13:45:00 | 87.10 | 86.72 | 86.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 14:30:00 | 87.25 | 86.98 | 86.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 10:00:00 | 87.15 | 87.32 | 87.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 11:00:00 | 87.15 | 87.28 | 87.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 11:15:00 | 87.38 | 87.30 | 87.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 11:30:00 | 87.20 | 87.30 | 87.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 12:15:00 | 86.95 | 87.23 | 87.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 12:30:00 | 87.18 | 87.23 | 87.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 13:15:00 | 86.70 | 87.13 | 87.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 13:30:00 | 86.65 | 87.13 | 87.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 14:15:00 | 87.05 | 87.11 | 87.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-05 09:15:00 | 88.10 | 87.10 | 87.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-05 11:15:00 | 87.43 | 87.10 | 87.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-05 11:45:00 | 87.93 | 87.39 | 87.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-08 10:15:00 | 87.18 | 88.70 | 88.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2024-02-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 10:15:00 | 87.18 | 88.70 | 88.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 09:15:00 | 86.03 | 87.73 | 88.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-12 13:15:00 | 86.63 | 86.34 | 86.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-12 14:00:00 | 86.63 | 86.34 | 86.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 13:15:00 | 86.68 | 86.16 | 86.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 09:15:00 | 85.70 | 86.28 | 86.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 13:45:00 | 86.08 | 86.29 | 86.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 14:45:00 | 86.15 | 86.38 | 86.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-14 15:15:00 | 87.00 | 86.50 | 86.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2024-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 15:15:00 | 87.00 | 86.50 | 86.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 10:15:00 | 87.38 | 86.77 | 86.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 13:15:00 | 87.25 | 87.42 | 87.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 13:15:00 | 87.25 | 87.42 | 87.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 13:15:00 | 87.25 | 87.42 | 87.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 14:00:00 | 87.25 | 87.42 | 87.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 15:15:00 | 87.03 | 87.33 | 87.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 09:15:00 | 86.70 | 87.33 | 87.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 86.35 | 87.14 | 87.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 10:00:00 | 86.35 | 87.14 | 87.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2024-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 10:15:00 | 86.33 | 86.97 | 87.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 14:15:00 | 86.08 | 86.56 | 86.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-20 10:15:00 | 87.10 | 86.58 | 86.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 10:15:00 | 87.10 | 86.58 | 86.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 87.10 | 86.58 | 86.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-20 10:45:00 | 87.05 | 86.58 | 86.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 11:15:00 | 86.78 | 86.62 | 86.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-20 13:15:00 | 86.65 | 86.70 | 86.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-21 10:15:00 | 86.50 | 86.65 | 86.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-21 13:15:00 | 86.55 | 86.68 | 86.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-23 10:15:00 | 86.95 | 86.29 | 86.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 10:15:00 | 86.95 | 86.29 | 86.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 11:15:00 | 87.03 | 86.44 | 86.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 09:15:00 | 86.18 | 86.66 | 86.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 09:15:00 | 86.18 | 86.66 | 86.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 86.18 | 86.66 | 86.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 10:00:00 | 86.18 | 86.66 | 86.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 10:15:00 | 86.50 | 86.63 | 86.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 10:30:00 | 86.00 | 86.63 | 86.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 11:15:00 | 86.95 | 86.69 | 86.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 12:15:00 | 87.10 | 86.69 | 86.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 12:45:00 | 87.20 | 86.82 | 86.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-27 09:15:00 | 87.38 | 87.01 | 86.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-27 10:30:00 | 87.08 | 87.04 | 86.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 11:15:00 | 86.73 | 86.98 | 86.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 12:00:00 | 86.73 | 86.98 | 86.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 12:15:00 | 86.95 | 86.97 | 86.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 12:30:00 | 86.80 | 86.97 | 86.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 13:15:00 | 86.40 | 86.86 | 86.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 13:45:00 | 86.45 | 86.86 | 86.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 14:15:00 | 86.95 | 86.88 | 86.81 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-28 10:15:00 | 86.30 | 86.72 | 86.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 10:15:00 | 86.30 | 86.72 | 86.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 11:15:00 | 85.83 | 86.54 | 86.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 85.13 | 84.88 | 85.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 15:00:00 | 85.13 | 84.88 | 85.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 85.85 | 85.10 | 85.47 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2024-03-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 15:15:00 | 86.00 | 85.64 | 85.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 12:15:00 | 86.23 | 85.90 | 85.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 14:15:00 | 86.00 | 86.01 | 85.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 14:15:00 | 86.00 | 86.01 | 85.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 14:15:00 | 86.00 | 86.01 | 85.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 14:30:00 | 85.98 | 86.01 | 85.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 15:15:00 | 86.00 | 86.00 | 85.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 09:15:00 | 86.55 | 86.00 | 85.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 86.20 | 86.04 | 85.92 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 84.98 | 85.84 | 85.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 09:15:00 | 84.50 | 85.01 | 85.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 81.63 | 81.45 | 82.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 10:45:00 | 81.55 | 81.45 | 82.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 80.88 | 81.53 | 82.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 10:15:00 | 80.38 | 81.53 | 82.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 10:45:00 | 80.53 | 81.31 | 82.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 13:00:00 | 80.60 | 81.02 | 81.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 10:00:00 | 80.50 | 80.88 | 81.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 14:15:00 | 81.33 | 80.90 | 81.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 15:00:00 | 81.33 | 80.90 | 81.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 15:15:00 | 81.72 | 81.06 | 81.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 09:15:00 | 81.00 | 81.06 | 81.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 09:15:00 | 82.05 | 81.05 | 80.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 82.05 | 81.05 | 80.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 12:15:00 | 82.23 | 81.57 | 81.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 83.23 | 83.23 | 82.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-22 15:00:00 | 83.23 | 83.23 | 82.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 87.15 | 87.68 | 87.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-03 12:15:00 | 87.38 | 87.45 | 87.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-03 15:15:00 | 87.35 | 87.40 | 87.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-04 11:15:00 | 86.43 | 86.94 | 86.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2024-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 11:15:00 | 86.43 | 86.94 | 86.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-05 09:15:00 | 86.20 | 86.64 | 86.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 12:15:00 | 87.00 | 86.70 | 86.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 12:15:00 | 87.00 | 86.70 | 86.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 12:15:00 | 87.00 | 86.70 | 86.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 13:00:00 | 87.00 | 86.70 | 86.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2024-04-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 13:15:00 | 87.40 | 86.84 | 86.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-05 14:15:00 | 88.13 | 87.10 | 86.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 10:15:00 | 88.00 | 88.30 | 87.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 10:15:00 | 88.00 | 88.30 | 87.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 10:15:00 | 88.00 | 88.30 | 87.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 11:00:00 | 88.00 | 88.30 | 87.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 12:15:00 | 88.25 | 88.28 | 87.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 12:30:00 | 88.08 | 88.28 | 87.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 13:15:00 | 87.58 | 88.14 | 87.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 14:00:00 | 87.58 | 88.14 | 87.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 14:15:00 | 88.08 | 88.13 | 87.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-10 09:30:00 | 88.30 | 88.08 | 87.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-10 10:15:00 | 88.43 | 88.08 | 87.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 10:15:00 | 87.80 | 88.59 | 88.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2024-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 10:15:00 | 87.80 | 88.59 | 88.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 12:15:00 | 87.20 | 88.16 | 88.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 10:15:00 | 87.95 | 87.77 | 88.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 11:00:00 | 87.95 | 87.77 | 88.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 88.33 | 87.88 | 88.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:45:00 | 88.15 | 87.88 | 88.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 12:15:00 | 88.03 | 87.91 | 88.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 09:15:00 | 87.25 | 88.05 | 88.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-23 09:15:00 | 87.15 | 86.22 | 86.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 09:15:00 | 87.15 | 86.22 | 86.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 14:15:00 | 87.35 | 86.73 | 86.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 15:15:00 | 88.58 | 88.75 | 88.17 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 09:15:00 | 89.05 | 88.75 | 88.17 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-26 12:15:00 | 93.50 | 90.99 | 89.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-04-29 14:15:00 | 92.63 | 92.68 | 91.48 | SL hit (close<ema200) qty=0.50 sl=92.68 alert=retest1 |

### Cycle 59 — SELL (started 2024-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 12:15:00 | 96.48 | 98.81 | 99.08 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-09 09:15:00 | 99.95 | 98.95 | 98.82 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-05-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 13:15:00 | 98.20 | 98.66 | 98.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 14:15:00 | 97.20 | 98.37 | 98.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 09:15:00 | 99.00 | 98.32 | 98.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 09:15:00 | 99.00 | 98.32 | 98.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 99.00 | 98.32 | 98.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 10:00:00 | 99.00 | 98.32 | 98.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 99.53 | 98.56 | 98.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 10:30:00 | 99.73 | 98.56 | 98.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2024-05-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 13:15:00 | 99.10 | 98.71 | 98.66 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 09:15:00 | 96.75 | 98.45 | 98.57 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 15:15:00 | 99.50 | 98.60 | 98.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 09:15:00 | 100.55 | 98.99 | 98.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 09:15:00 | 100.90 | 101.01 | 100.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 10:00:00 | 100.90 | 101.01 | 100.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 104.30 | 105.04 | 104.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 104.23 | 105.04 | 104.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 104.83 | 105.00 | 104.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 12:15:00 | 105.33 | 104.95 | 104.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 10:15:00 | 103.63 | 104.29 | 104.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 10:15:00 | 103.63 | 104.29 | 104.29 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 12:15:00 | 104.88 | 104.38 | 104.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 13:15:00 | 106.13 | 104.73 | 104.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 14:15:00 | 104.65 | 106.09 | 105.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 14:15:00 | 104.65 | 106.09 | 105.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 104.65 | 106.09 | 105.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 10:30:00 | 109.40 | 107.17 | 106.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 12:15:00 | 110.40 | 110.86 | 110.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2024-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 12:15:00 | 110.40 | 110.86 | 110.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 14:15:00 | 109.80 | 110.52 | 110.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 10:15:00 | 111.08 | 110.53 | 110.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 10:15:00 | 111.08 | 110.53 | 110.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 111.08 | 110.53 | 110.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:45:00 | 110.58 | 110.53 | 110.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 111.33 | 110.69 | 110.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:00:00 | 111.33 | 110.69 | 110.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2024-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 12:15:00 | 112.33 | 111.02 | 110.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 13:15:00 | 112.85 | 111.39 | 111.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 114.23 | 116.14 | 114.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 114.23 | 116.14 | 114.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 114.23 | 116.14 | 114.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 113.15 | 116.14 | 114.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 108.15 | 114.54 | 113.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 108.15 | 114.54 | 113.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 100.53 | 111.74 | 112.63 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 113.65 | 110.44 | 110.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 14:15:00 | 115.75 | 114.33 | 112.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 10:15:00 | 114.74 | 114.78 | 113.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 11:00:00 | 114.74 | 114.78 | 113.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 117.90 | 118.90 | 117.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:00:00 | 117.90 | 118.90 | 117.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 118.14 | 118.75 | 117.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 13:30:00 | 118.69 | 118.63 | 118.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 118.93 | 118.55 | 118.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 11:30:00 | 118.82 | 118.70 | 118.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 116.65 | 119.31 | 119.26 | SL hit (close<static) qty=1.00 sl=117.78 alert=retest2 |

### Cycle 71 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 116.80 | 118.81 | 119.04 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 12:15:00 | 120.53 | 118.64 | 118.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 121.75 | 119.87 | 119.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 15:15:00 | 120.76 | 120.82 | 120.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-26 09:15:00 | 120.63 | 120.82 | 120.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 120.71 | 120.80 | 120.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 11:45:00 | 121.46 | 120.93 | 120.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 09:30:00 | 121.97 | 120.76 | 120.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 14:15:00 | 119.31 | 120.47 | 120.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 14:15:00 | 119.31 | 120.47 | 120.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 11:15:00 | 117.47 | 119.21 | 119.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 12:15:00 | 114.38 | 114.10 | 115.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-05 12:45:00 | 114.50 | 114.10 | 115.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 115.03 | 114.29 | 115.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 13:30:00 | 115.01 | 114.29 | 115.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 114.80 | 114.39 | 115.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 10:45:00 | 114.19 | 114.43 | 114.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 11:30:00 | 113.82 | 114.26 | 114.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 14:15:00 | 114.12 | 114.22 | 114.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 15:15:00 | 114.08 | 114.23 | 114.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 112.32 | 113.83 | 114.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 10:30:00 | 111.90 | 113.59 | 114.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 13:15:00 | 114.04 | 113.41 | 113.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2024-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 13:15:00 | 114.04 | 113.41 | 113.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 15:15:00 | 114.15 | 113.66 | 113.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 14:15:00 | 114.03 | 114.10 | 113.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 14:30:00 | 114.10 | 114.10 | 113.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 114.65 | 114.21 | 113.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 113.58 | 114.21 | 113.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 114.06 | 114.18 | 113.91 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 112.55 | 113.83 | 113.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 112.05 | 113.16 | 113.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 114.00 | 113.14 | 113.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 114.00 | 113.14 | 113.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 114.00 | 113.14 | 113.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 114.00 | 113.14 | 113.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 115.33 | 113.57 | 113.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 115.42 | 113.57 | 113.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 11:15:00 | 115.24 | 113.91 | 113.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 12:15:00 | 116.40 | 114.41 | 113.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 12:15:00 | 113.73 | 114.94 | 114.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 113.73 | 114.94 | 114.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 113.73 | 114.94 | 114.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 114.28 | 114.94 | 114.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 114.79 | 114.91 | 114.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:30:00 | 113.60 | 114.91 | 114.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 114.90 | 114.91 | 114.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 14:30:00 | 115.42 | 114.90 | 114.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 09:30:00 | 115.58 | 115.33 | 114.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 13:00:00 | 115.51 | 115.38 | 115.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 14:15:00 | 115.95 | 115.35 | 115.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 116.10 | 115.50 | 115.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 121.18 | 115.63 | 115.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-29 09:15:00 | 126.96 | 122.93 | 119.96 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 13:15:00 | 123.95 | 126.25 | 126.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 123.48 | 125.05 | 125.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 123.70 | 122.67 | 123.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 123.70 | 122.67 | 123.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 123.70 | 122.67 | 123.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:30:00 | 122.08 | 122.60 | 123.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 15:00:00 | 121.38 | 122.60 | 123.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 12:15:00 | 124.28 | 123.53 | 123.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 12:15:00 | 124.28 | 123.53 | 123.44 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 15:15:00 | 123.00 | 123.39 | 123.40 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 125.85 | 123.88 | 123.62 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 124.38 | 125.56 | 125.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 11:15:00 | 124.05 | 125.26 | 125.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 125.05 | 124.45 | 124.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 125.05 | 124.45 | 124.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 125.05 | 124.45 | 124.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:30:00 | 125.40 | 124.45 | 124.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 125.45 | 124.65 | 124.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:45:00 | 125.70 | 124.65 | 124.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 126.20 | 124.96 | 125.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:00:00 | 126.20 | 124.96 | 125.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 12:15:00 | 126.38 | 125.24 | 125.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 13:15:00 | 127.20 | 125.63 | 125.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 130.07 | 130.53 | 130.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 130.07 | 130.53 | 130.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 130.07 | 130.53 | 130.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 129.98 | 130.53 | 130.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 130.35 | 130.50 | 130.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 12:00:00 | 130.82 | 130.56 | 130.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 10:15:00 | 129.45 | 130.25 | 130.18 | SL hit (close<static) qty=1.00 sl=129.93 alert=retest2 |

### Cycle 83 — SELL (started 2024-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 11:15:00 | 129.50 | 130.10 | 130.12 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 09:15:00 | 130.93 | 130.19 | 130.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 10:15:00 | 131.90 | 130.53 | 130.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 15:15:00 | 130.95 | 131.08 | 130.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 15:15:00 | 130.95 | 131.08 | 130.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 130.95 | 131.08 | 130.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:30:00 | 130.25 | 130.90 | 130.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 130.80 | 130.88 | 130.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:45:00 | 130.28 | 130.88 | 130.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 130.73 | 130.85 | 130.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 12:30:00 | 131.03 | 130.83 | 130.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 13:15:00 | 131.03 | 130.83 | 130.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 14:15:00 | 130.28 | 130.72 | 130.66 | SL hit (close<static) qty=1.00 sl=130.38 alert=retest2 |

### Cycle 85 — SELL (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 15:15:00 | 130.00 | 130.58 | 130.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 128.90 | 130.24 | 130.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 12:15:00 | 127.35 | 127.20 | 128.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 12:45:00 | 127.33 | 127.20 | 128.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 129.03 | 127.47 | 128.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 129.03 | 127.47 | 128.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 128.25 | 127.63 | 128.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 09:15:00 | 126.70 | 127.63 | 128.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 14:15:00 | 120.36 | 122.08 | 122.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-12 13:15:00 | 122.05 | 121.75 | 122.28 | SL hit (close>ema200) qty=0.50 sl=121.75 alert=retest2 |

### Cycle 86 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 123.08 | 122.55 | 122.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 09:15:00 | 123.48 | 122.91 | 122.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 11:15:00 | 122.70 | 122.90 | 122.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 11:15:00 | 122.70 | 122.90 | 122.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 122.70 | 122.90 | 122.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:00:00 | 122.70 | 122.90 | 122.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 123.20 | 122.96 | 122.81 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2024-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 15:15:00 | 122.00 | 122.60 | 122.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 09:15:00 | 119.20 | 121.92 | 122.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 12:15:00 | 118.25 | 118.17 | 119.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 12:45:00 | 118.45 | 118.17 | 119.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 118.85 | 118.50 | 119.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:45:00 | 118.95 | 118.50 | 119.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 118.80 | 118.56 | 118.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:30:00 | 118.95 | 118.56 | 118.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 118.93 | 118.63 | 118.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:45:00 | 118.98 | 118.63 | 118.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 118.88 | 118.68 | 118.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:45:00 | 118.90 | 118.68 | 118.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 118.73 | 118.69 | 118.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 14:15:00 | 118.20 | 118.69 | 118.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 118.95 | 118.74 | 118.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 14:45:00 | 118.95 | 118.74 | 118.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 15:15:00 | 118.98 | 118.79 | 118.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:15:00 | 118.93 | 118.79 | 118.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 118.90 | 118.81 | 118.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:30:00 | 119.30 | 118.81 | 118.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 118.50 | 118.75 | 118.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 10:30:00 | 118.60 | 118.75 | 118.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 11:15:00 | 118.95 | 118.79 | 118.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 11:30:00 | 118.88 | 118.79 | 118.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 12:15:00 | 118.68 | 118.77 | 118.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 12:30:00 | 118.95 | 118.77 | 118.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 13:15:00 | 118.80 | 118.77 | 118.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 14:00:00 | 118.80 | 118.77 | 118.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 14:15:00 | 118.35 | 118.69 | 118.83 | EMA400 retest candle locked (from downside) |

### Cycle 88 — BUY (started 2024-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 15:15:00 | 119.25 | 118.66 | 118.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 09:15:00 | 119.85 | 118.90 | 118.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 12:15:00 | 119.45 | 119.97 | 119.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 12:15:00 | 119.45 | 119.97 | 119.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 119.45 | 119.97 | 119.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 13:00:00 | 119.45 | 119.97 | 119.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 119.60 | 119.89 | 119.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 14:30:00 | 120.15 | 119.87 | 119.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 10:00:00 | 120.18 | 119.90 | 119.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 13:15:00 | 119.28 | 119.53 | 119.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 13:15:00 | 119.28 | 119.53 | 119.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 14:15:00 | 117.93 | 119.21 | 119.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 119.89 | 119.13 | 119.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 119.89 | 119.13 | 119.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 119.89 | 119.13 | 119.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:45:00 | 120.25 | 119.13 | 119.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 118.60 | 119.02 | 119.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 117.85 | 119.09 | 119.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 13:15:00 | 111.96 | 114.56 | 116.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 13:15:00 | 111.47 | 111.44 | 112.58 | SL hit (close>ema200) qty=0.50 sl=111.44 alert=retest2 |

### Cycle 90 — BUY (started 2024-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 15:15:00 | 112.73 | 111.97 | 111.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 09:15:00 | 113.97 | 112.37 | 112.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 12:15:00 | 113.32 | 113.57 | 113.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 13:00:00 | 113.32 | 113.57 | 113.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 114.37 | 113.73 | 113.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 13:30:00 | 113.13 | 113.73 | 113.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 112.64 | 113.69 | 113.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 112.64 | 113.69 | 113.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 113.45 | 113.64 | 113.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:30:00 | 113.31 | 113.64 | 113.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 113.06 | 113.53 | 113.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:00:00 | 113.06 | 113.53 | 113.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 112.76 | 113.37 | 113.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:00:00 | 112.76 | 113.37 | 113.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2024-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 15:15:00 | 112.69 | 113.12 | 113.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 09:15:00 | 112.17 | 112.93 | 113.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 13:15:00 | 112.34 | 112.33 | 112.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-16 14:00:00 | 112.34 | 112.33 | 112.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 112.06 | 112.28 | 112.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 14:30:00 | 112.54 | 112.28 | 112.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 111.03 | 112.02 | 112.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 12:30:00 | 110.53 | 111.20 | 111.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 10:45:00 | 110.45 | 110.43 | 111.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:30:00 | 110.58 | 111.21 | 111.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:30:00 | 110.17 | 110.88 | 111.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 108.90 | 107.63 | 108.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:00:00 | 108.90 | 107.63 | 108.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 108.04 | 107.71 | 108.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:30:00 | 108.75 | 107.71 | 108.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 11:15:00 | 108.17 | 107.80 | 108.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 11:30:00 | 108.13 | 107.80 | 108.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 12:15:00 | 108.42 | 107.93 | 108.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 12:45:00 | 108.39 | 107.93 | 108.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 13:15:00 | 108.80 | 108.10 | 108.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 13:45:00 | 108.73 | 108.10 | 108.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 108.82 | 108.34 | 108.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 09:15:00 | 109.01 | 108.34 | 108.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 106.37 | 107.94 | 108.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 10:30:00 | 105.98 | 107.61 | 108.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 10:00:00 | 106.15 | 107.09 | 107.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 12:15:00 | 106.16 | 106.78 | 107.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 13:30:00 | 106.13 | 106.55 | 107.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 09:15:00 | 105.00 | 106.10 | 106.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 09:15:00 | 104.93 | 106.10 | 106.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 09:15:00 | 105.05 | 106.10 | 106.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 09:15:00 | 104.66 | 106.10 | 106.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-30 10:15:00 | 105.19 | 104.90 | 105.62 | SL hit (close>ema200) qty=0.50 sl=104.90 alert=retest2 |

### Cycle 92 — BUY (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 13:15:00 | 105.97 | 104.49 | 104.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 14:15:00 | 106.09 | 104.81 | 104.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 15:15:00 | 107.65 | 107.75 | 106.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 09:15:00 | 106.36 | 107.75 | 106.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 106.37 | 107.47 | 106.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 13:15:00 | 108.98 | 107.10 | 106.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-13 10:15:00 | 108.74 | 110.17 | 110.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-13 10:15:00 | 108.20 | 109.77 | 109.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 108.20 | 109.77 | 109.95 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 12:15:00 | 110.26 | 109.33 | 109.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 111.00 | 109.86 | 109.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 110.45 | 110.89 | 110.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 15:00:00 | 110.45 | 110.89 | 110.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 110.28 | 110.77 | 110.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 108.40 | 110.77 | 110.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 107.80 | 110.18 | 110.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:00:00 | 107.80 | 110.18 | 110.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 10:15:00 | 109.09 | 109.96 | 109.97 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 09:15:00 | 111.30 | 109.93 | 109.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 13:15:00 | 112.27 | 110.97 | 110.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 12:15:00 | 116.15 | 116.29 | 114.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 13:00:00 | 116.15 | 116.29 | 114.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 116.58 | 117.06 | 116.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 11:30:00 | 116.69 | 117.06 | 116.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 116.00 | 116.84 | 116.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:00:00 | 116.00 | 116.84 | 116.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 116.60 | 116.80 | 116.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 14:15:00 | 116.68 | 116.80 | 116.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 14:15:00 | 115.77 | 116.59 | 116.30 | SL hit (close<static) qty=1.00 sl=115.93 alert=retest2 |

### Cycle 97 — SELL (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 10:15:00 | 115.83 | 116.11 | 116.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 09:15:00 | 114.14 | 115.47 | 115.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 114.97 | 114.71 | 115.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-02 15:00:00 | 114.97 | 114.71 | 115.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 116.29 | 115.05 | 115.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:00:00 | 116.29 | 115.05 | 115.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 116.37 | 115.31 | 115.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 11:00:00 | 116.37 | 115.31 | 115.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 11:15:00 | 116.45 | 115.54 | 115.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 13:15:00 | 117.35 | 116.03 | 115.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 10:15:00 | 116.60 | 116.64 | 116.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 10:45:00 | 116.44 | 116.64 | 116.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 116.69 | 116.65 | 116.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:00:00 | 116.69 | 116.65 | 116.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 116.58 | 116.73 | 116.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:15:00 | 116.40 | 116.73 | 116.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 116.04 | 116.59 | 116.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:00:00 | 116.04 | 116.59 | 116.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 116.55 | 116.59 | 116.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:30:00 | 115.97 | 116.59 | 116.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 116.02 | 116.48 | 116.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:00:00 | 116.02 | 116.48 | 116.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 116.49 | 116.49 | 116.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 115.71 | 116.49 | 116.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 116.32 | 116.45 | 116.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:15:00 | 116.28 | 116.45 | 116.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 115.51 | 116.26 | 116.31 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 12:15:00 | 116.80 | 116.34 | 116.34 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-12-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 15:15:00 | 116.12 | 116.31 | 116.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 09:15:00 | 115.15 | 116.08 | 116.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 10:15:00 | 115.10 | 114.94 | 115.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-10 11:00:00 | 115.10 | 114.94 | 115.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 115.23 | 114.75 | 115.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:00:00 | 115.23 | 114.75 | 115.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 115.10 | 114.82 | 115.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 11:15:00 | 115.40 | 114.82 | 115.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 11:15:00 | 115.50 | 114.96 | 115.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 12:00:00 | 115.50 | 114.96 | 115.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 12:15:00 | 115.62 | 115.09 | 115.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 12:45:00 | 115.85 | 115.09 | 115.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 14:15:00 | 115.57 | 115.27 | 115.23 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 11:15:00 | 115.05 | 115.20 | 115.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 114.08 | 114.92 | 115.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 114.95 | 114.72 | 114.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 11:15:00 | 114.95 | 114.72 | 114.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 114.95 | 114.72 | 114.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:00:00 | 114.95 | 114.72 | 114.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 115.25 | 114.83 | 114.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 115.25 | 114.83 | 114.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 115.28 | 114.92 | 115.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 115.28 | 114.92 | 115.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 116.05 | 115.14 | 115.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 15:15:00 | 116.20 | 115.36 | 115.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 11:15:00 | 116.47 | 116.59 | 116.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 12:00:00 | 116.47 | 116.59 | 116.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 116.00 | 116.47 | 116.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:45:00 | 116.16 | 116.47 | 116.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 115.63 | 116.30 | 116.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:00:00 | 115.63 | 116.30 | 116.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 115.38 | 116.12 | 115.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:45:00 | 115.36 | 116.12 | 115.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 09:15:00 | 112.94 | 115.35 | 115.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 10:15:00 | 112.30 | 114.74 | 115.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 09:15:00 | 110.15 | 109.29 | 110.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 09:15:00 | 110.15 | 109.29 | 110.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 110.15 | 109.29 | 110.34 | EMA400 retest candle locked (from downside) |

### Cycle 106 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 111.46 | 110.24 | 110.15 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 109.57 | 110.28 | 110.33 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 112.00 | 110.62 | 110.48 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 11:15:00 | 110.04 | 110.45 | 110.45 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 12:15:00 | 110.97 | 110.43 | 110.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 13:15:00 | 111.86 | 110.71 | 110.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 116.64 | 116.66 | 115.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 09:30:00 | 116.27 | 116.66 | 115.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 114.44 | 116.22 | 115.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 114.44 | 116.22 | 115.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 114.44 | 115.86 | 115.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 114.03 | 115.86 | 115.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 112.96 | 114.67 | 114.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 11:15:00 | 112.19 | 113.64 | 114.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 103.55 | 103.11 | 105.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 13:15:00 | 105.21 | 104.11 | 104.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 105.21 | 104.11 | 104.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:00:00 | 105.21 | 104.11 | 104.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 104.99 | 104.29 | 104.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:15:00 | 104.43 | 104.48 | 105.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 11:00:00 | 104.70 | 104.50 | 104.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 11:15:00 | 104.67 | 104.18 | 104.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 11:45:00 | 104.74 | 104.30 | 104.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 12:15:00 | 104.51 | 104.35 | 104.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 13:00:00 | 104.51 | 104.35 | 104.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 13:15:00 | 104.50 | 104.38 | 104.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 14:15:00 | 104.11 | 104.38 | 104.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 14:15:00 | 104.72 | 104.05 | 103.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2025-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 14:15:00 | 104.72 | 104.05 | 103.99 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 102.62 | 103.74 | 103.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 102.43 | 103.13 | 103.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 09:15:00 | 103.94 | 103.16 | 103.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 09:15:00 | 103.94 | 103.16 | 103.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 103.94 | 103.16 | 103.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 10:00:00 | 103.94 | 103.16 | 103.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 103.91 | 103.31 | 103.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:00:00 | 103.91 | 103.31 | 103.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 102.66 | 103.18 | 103.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 09:15:00 | 102.51 | 103.29 | 103.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 10:15:00 | 104.12 | 103.60 | 103.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 104.12 | 103.60 | 103.54 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 09:15:00 | 102.58 | 103.54 | 103.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 102.01 | 103.08 | 103.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 09:15:00 | 99.90 | 99.75 | 101.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 10:00:00 | 99.90 | 99.75 | 101.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 10:15:00 | 101.56 | 100.11 | 101.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 11:00:00 | 101.56 | 100.11 | 101.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 101.94 | 100.48 | 101.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:15:00 | 101.77 | 100.48 | 101.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2025-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 14:15:00 | 102.55 | 101.58 | 101.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 104.51 | 102.86 | 102.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 105.17 | 107.62 | 106.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 105.17 | 107.62 | 106.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 105.17 | 107.62 | 106.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:00:00 | 105.17 | 107.62 | 106.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 103.19 | 106.74 | 106.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 103.19 | 106.74 | 106.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 102.99 | 105.99 | 105.73 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 103.07 | 105.40 | 105.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 100.65 | 104.12 | 104.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 104.20 | 102.47 | 103.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 104.20 | 102.47 | 103.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 104.20 | 102.47 | 103.35 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 13:15:00 | 105.55 | 104.04 | 103.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 106.19 | 104.94 | 104.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 14:15:00 | 105.55 | 105.62 | 104.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 15:00:00 | 105.55 | 105.62 | 104.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 105.23 | 105.43 | 105.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:00:00 | 105.23 | 105.43 | 105.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 104.72 | 105.29 | 105.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:00:00 | 104.72 | 105.29 | 105.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 104.62 | 105.16 | 104.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 104.62 | 105.16 | 104.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 104.86 | 104.95 | 104.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 105.16 | 104.95 | 104.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 105.91 | 105.14 | 105.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:15:00 | 104.86 | 105.14 | 105.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 107.40 | 105.59 | 105.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:30:00 | 106.20 | 105.59 | 105.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 104.96 | 105.75 | 105.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 104.96 | 105.75 | 105.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 104.96 | 105.59 | 105.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:45:00 | 104.81 | 105.59 | 105.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2025-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 12:15:00 | 104.89 | 105.33 | 105.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 13:15:00 | 104.61 | 105.18 | 105.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 102.59 | 102.12 | 103.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:00:00 | 102.59 | 102.12 | 103.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 102.95 | 102.37 | 103.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 102.81 | 102.37 | 103.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 107.00 | 103.30 | 103.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:45:00 | 106.10 | 103.30 | 103.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2025-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 14:15:00 | 109.49 | 104.54 | 104.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 10:15:00 | 110.35 | 107.21 | 105.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 10:15:00 | 107.42 | 108.25 | 107.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 10:15:00 | 107.42 | 108.25 | 107.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 107.42 | 108.25 | 107.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 11:00:00 | 107.42 | 108.25 | 107.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 11:15:00 | 106.96 | 107.99 | 107.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 12:00:00 | 106.96 | 107.99 | 107.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 12:15:00 | 107.59 | 107.91 | 107.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-14 14:30:00 | 107.89 | 107.45 | 107.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-14 15:15:00 | 106.45 | 107.25 | 106.97 | SL hit (close<static) qty=1.00 sl=106.66 alert=retest2 |

### Cycle 121 — SELL (started 2025-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 15:15:00 | 110.90 | 111.51 | 111.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 11:15:00 | 110.52 | 111.09 | 111.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 14:15:00 | 111.57 | 111.04 | 111.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 14:15:00 | 111.57 | 111.04 | 111.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 111.57 | 111.04 | 111.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 15:00:00 | 111.57 | 111.04 | 111.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 111.45 | 111.12 | 111.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:15:00 | 112.29 | 111.12 | 111.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 113.70 | 111.64 | 111.47 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 109.23 | 111.70 | 111.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 14:15:00 | 106.64 | 109.06 | 110.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 09:15:00 | 104.16 | 103.84 | 105.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-05 09:30:00 | 105.09 | 103.84 | 105.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 105.93 | 104.43 | 105.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:00:00 | 105.93 | 104.43 | 105.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 104.75 | 104.49 | 105.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 13:15:00 | 104.44 | 104.49 | 105.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 15:00:00 | 104.61 | 104.47 | 105.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-06 10:15:00 | 106.58 | 105.12 | 105.27 | SL hit (close>static) qty=1.00 sl=106.25 alert=retest2 |

### Cycle 124 — BUY (started 2025-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 11:15:00 | 106.55 | 105.41 | 105.38 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-03-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 12:15:00 | 104.90 | 105.40 | 105.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 103.79 | 104.92 | 105.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 11:15:00 | 98.72 | 98.59 | 99.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 12:00:00 | 98.72 | 98.59 | 99.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 98.93 | 98.55 | 99.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:45:00 | 99.30 | 98.55 | 99.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 99.23 | 98.72 | 99.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:45:00 | 98.82 | 98.72 | 99.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 99.37 | 98.90 | 99.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:00:00 | 99.37 | 98.90 | 99.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 99.21 | 98.96 | 99.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 15:15:00 | 99.50 | 98.96 | 99.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 99.50 | 99.07 | 99.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:15:00 | 100.64 | 99.07 | 99.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 100.50 | 99.35 | 99.42 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 100.42 | 99.57 | 99.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 101.24 | 99.90 | 99.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 15:15:00 | 102.25 | 102.71 | 101.75 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:15:00 | 103.93 | 102.71 | 101.75 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 10:15:00 | 103.69 | 102.78 | 101.87 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 104.65 | 103.85 | 102.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:15:00 | 104.93 | 103.85 | 102.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 12:15:00 | 104.98 | 105.98 | 105.43 | SL hit (close<ema400) qty=1.00 sl=105.43 alert=retest1 |

### Cycle 127 — SELL (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 14:15:00 | 104.03 | 106.00 | 106.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 09:15:00 | 102.65 | 105.09 | 105.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 13:15:00 | 104.00 | 103.10 | 103.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 13:15:00 | 104.00 | 103.10 | 103.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 104.00 | 103.10 | 103.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 13:45:00 | 103.60 | 103.10 | 103.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 104.40 | 103.36 | 103.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 15:00:00 | 104.40 | 103.36 | 103.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 15:15:00 | 104.55 | 103.60 | 103.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 104.00 | 103.60 | 103.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 10:45:00 | 104.13 | 103.79 | 104.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 12:45:00 | 104.20 | 104.01 | 104.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 14:15:00 | 104.39 | 104.15 | 104.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 104.39 | 104.15 | 104.14 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 09:15:00 | 103.34 | 104.00 | 104.08 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 13:15:00 | 104.86 | 104.19 | 104.13 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 102.55 | 104.10 | 104.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 14:15:00 | 102.49 | 103.34 | 103.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 99.69 | 99.40 | 100.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 99.69 | 99.40 | 100.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 99.69 | 99.40 | 100.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 99.15 | 99.51 | 100.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 11:15:00 | 102.05 | 100.69 | 100.84 | SL hit (close>static) qty=1.00 sl=101.75 alert=retest2 |

### Cycle 132 — BUY (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 12:15:00 | 102.43 | 101.04 | 100.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 103.80 | 102.09 | 101.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 14:15:00 | 111.32 | 111.39 | 110.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 15:00:00 | 111.32 | 111.39 | 110.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 113.21 | 114.53 | 113.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 113.21 | 114.53 | 113.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 112.72 | 114.17 | 113.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:30:00 | 112.66 | 114.17 | 113.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 112.65 | 113.42 | 113.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 15:00:00 | 112.65 | 113.42 | 113.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 15:15:00 | 112.58 | 113.25 | 113.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:15:00 | 112.28 | 113.25 | 113.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 09:15:00 | 112.50 | 113.10 | 113.18 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 14:15:00 | 113.72 | 113.26 | 113.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 114.91 | 113.63 | 113.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 11:15:00 | 113.52 | 113.70 | 113.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 11:15:00 | 113.52 | 113.70 | 113.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 113.52 | 113.70 | 113.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 12:00:00 | 113.52 | 113.70 | 113.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 113.40 | 113.64 | 113.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 13:15:00 | 113.55 | 113.64 | 113.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 13:15:00 | 113.83 | 113.68 | 113.49 | EMA400 retest candle locked (from upside) |

### Cycle 135 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 112.75 | 113.40 | 113.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 10:15:00 | 111.82 | 112.90 | 113.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 111.94 | 111.28 | 112.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 111.94 | 111.28 | 112.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 111.94 | 111.28 | 112.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:45:00 | 112.09 | 111.28 | 112.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 111.89 | 111.40 | 112.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:30:00 | 112.14 | 111.40 | 112.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 111.50 | 111.42 | 112.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 12:15:00 | 111.12 | 111.42 | 112.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 13:15:00 | 112.24 | 111.60 | 111.98 | SL hit (close>static) qty=1.00 sl=112.05 alert=retest2 |

### Cycle 136 — BUY (started 2025-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 12:15:00 | 112.50 | 111.62 | 111.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 13:15:00 | 113.09 | 111.91 | 111.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 10:15:00 | 112.22 | 112.40 | 112.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 10:15:00 | 112.22 | 112.40 | 112.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 112.22 | 112.40 | 112.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:00:00 | 112.22 | 112.40 | 112.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 111.34 | 112.18 | 112.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:00:00 | 111.34 | 112.18 | 112.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 110.63 | 111.87 | 111.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 14:15:00 | 110.40 | 111.57 | 111.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 13:15:00 | 110.59 | 110.49 | 111.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 14:00:00 | 110.59 | 110.49 | 111.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 110.62 | 110.59 | 110.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 114.08 | 110.59 | 110.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 113.25 | 111.12 | 111.17 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 113.81 | 111.66 | 111.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 114.37 | 112.58 | 111.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 11:15:00 | 118.85 | 118.96 | 117.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 12:00:00 | 118.85 | 118.96 | 117.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 121.68 | 121.28 | 120.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:45:00 | 122.43 | 121.53 | 120.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 12:30:00 | 122.30 | 121.73 | 120.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:00:00 | 122.46 | 121.87 | 120.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 119.35 | 121.49 | 121.05 | SL hit (close<static) qty=1.00 sl=119.75 alert=retest2 |

### Cycle 139 — SELL (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 11:15:00 | 118.50 | 120.52 | 120.66 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 120.56 | 119.75 | 119.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 12:15:00 | 120.76 | 120.07 | 119.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 13:15:00 | 119.55 | 119.97 | 119.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 13:15:00 | 119.55 | 119.97 | 119.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 119.55 | 119.97 | 119.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:00:00 | 119.55 | 119.97 | 119.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 119.41 | 119.86 | 119.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 15:00:00 | 119.41 | 119.86 | 119.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 15:15:00 | 119.29 | 119.74 | 119.75 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 13:15:00 | 120.05 | 119.72 | 119.72 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 117.75 | 119.47 | 119.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 10:15:00 | 116.48 | 117.54 | 117.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 118.07 | 117.36 | 117.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 118.07 | 117.36 | 117.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 118.07 | 117.36 | 117.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 118.07 | 117.36 | 117.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 120.62 | 118.01 | 117.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 122.03 | 120.40 | 119.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 11:15:00 | 121.21 | 121.26 | 120.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 12:00:00 | 121.21 | 121.26 | 120.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 120.76 | 121.18 | 120.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:00:00 | 120.76 | 121.18 | 120.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 120.68 | 121.08 | 120.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:30:00 | 120.46 | 121.08 | 120.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 120.67 | 121.00 | 120.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 121.00 | 121.00 | 120.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 120.75 | 120.95 | 120.63 | EMA400 retest candle locked (from upside) |

### Cycle 145 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 119.76 | 120.45 | 120.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 118.90 | 119.99 | 120.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 117.49 | 117.47 | 118.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 117.49 | 117.47 | 118.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 117.95 | 117.50 | 117.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:30:00 | 118.00 | 117.50 | 117.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 117.88 | 117.58 | 117.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 117.45 | 117.61 | 117.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:00:00 | 117.63 | 117.62 | 117.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 117.35 | 117.61 | 117.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:15:00 | 117.55 | 117.47 | 117.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 117.68 | 117.52 | 117.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:45:00 | 117.23 | 117.51 | 117.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:45:00 | 117.24 | 117.49 | 117.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:15:00 | 117.30 | 117.49 | 117.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:00:00 | 117.37 | 117.30 | 117.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 117.00 | 116.61 | 116.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 117.14 | 116.61 | 116.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 117.63 | 116.82 | 117.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 117.63 | 116.82 | 117.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 117.35 | 116.92 | 117.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 117.63 | 116.92 | 117.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 117.05 | 116.96 | 117.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:45:00 | 117.10 | 116.96 | 117.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 117.62 | 117.09 | 117.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 117.62 | 117.09 | 117.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-20 15:15:00 | 117.40 | 117.15 | 117.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 15:15:00 | 117.40 | 117.15 | 117.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 118.36 | 117.52 | 117.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 124.81 | 124.89 | 123.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 15:00:00 | 124.81 | 124.89 | 123.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 124.73 | 124.92 | 124.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 14:30:00 | 125.78 | 125.00 | 124.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 125.75 | 125.08 | 124.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:15:00 | 125.63 | 125.09 | 124.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 125.65 | 125.55 | 125.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 125.25 | 125.46 | 125.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:30:00 | 125.68 | 125.46 | 125.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-04 11:15:00 | 124.65 | 125.30 | 125.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 124.65 | 125.30 | 125.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 123.90 | 124.58 | 124.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 12:15:00 | 124.63 | 124.57 | 124.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 12:15:00 | 124.63 | 124.57 | 124.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 124.63 | 124.57 | 124.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 124.63 | 124.57 | 124.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 124.98 | 124.58 | 124.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 124.98 | 124.58 | 124.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 125.00 | 124.67 | 124.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 125.50 | 124.67 | 124.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 125.63 | 124.86 | 124.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 09:15:00 | 126.10 | 125.46 | 125.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 11:15:00 | 124.98 | 125.45 | 125.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 11:15:00 | 124.98 | 125.45 | 125.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 124.98 | 125.45 | 125.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 124.98 | 125.45 | 125.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 124.95 | 125.35 | 125.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:45:00 | 124.95 | 125.35 | 125.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 124.98 | 125.27 | 125.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 124.80 | 125.27 | 125.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 124.75 | 125.07 | 125.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 124.05 | 124.87 | 125.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 125.00 | 124.16 | 124.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 125.00 | 124.16 | 124.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 125.00 | 124.16 | 124.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 124.95 | 124.16 | 124.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 124.98 | 124.32 | 124.55 | EMA400 retest candle locked (from downside) |

### Cycle 150 — BUY (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 11:15:00 | 126.60 | 124.78 | 124.74 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 10:15:00 | 124.35 | 125.07 | 125.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 09:15:00 | 123.75 | 124.56 | 124.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 123.90 | 123.08 | 123.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 123.90 | 123.08 | 123.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 123.90 | 123.08 | 123.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:00:00 | 123.90 | 123.08 | 123.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 124.60 | 123.39 | 123.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 124.55 | 123.39 | 123.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 12:15:00 | 125.05 | 123.95 | 123.88 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 123.55 | 124.10 | 124.12 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 12:15:00 | 124.35 | 124.14 | 124.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 10:15:00 | 125.15 | 124.51 | 124.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 12:15:00 | 124.55 | 124.58 | 124.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 12:15:00 | 124.55 | 124.58 | 124.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 124.55 | 124.58 | 124.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:00:00 | 124.55 | 124.58 | 124.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 124.45 | 124.68 | 124.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 124.60 | 124.68 | 124.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 124.00 | 124.54 | 124.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 124.00 | 124.54 | 124.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 123.60 | 124.35 | 124.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 11:15:00 | 123.15 | 124.11 | 124.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 124.55 | 123.31 | 123.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 124.55 | 123.31 | 123.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 124.55 | 123.31 | 123.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 124.55 | 123.31 | 123.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 123.55 | 123.36 | 123.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:30:00 | 123.10 | 123.25 | 123.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 12:30:00 | 123.50 | 123.21 | 123.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 13:00:00 | 123.40 | 123.21 | 123.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 15:00:00 | 123.25 | 123.21 | 123.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 123.10 | 123.19 | 123.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 122.15 | 123.19 | 123.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 122.40 | 123.03 | 123.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 121.15 | 122.34 | 122.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 13:45:00 | 121.15 | 121.44 | 122.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 11:15:00 | 122.71 | 121.50 | 121.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 11:15:00 | 122.71 | 121.50 | 121.35 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 120.81 | 121.66 | 121.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 120.44 | 120.95 | 121.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 119.92 | 119.88 | 120.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 119.92 | 119.88 | 120.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 117.94 | 119.46 | 120.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:45:00 | 117.56 | 118.96 | 119.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 120.06 | 119.07 | 119.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 120.06 | 119.07 | 119.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 120.59 | 119.66 | 119.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 119.64 | 119.91 | 119.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 14:15:00 | 119.64 | 119.91 | 119.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 119.64 | 119.91 | 119.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 119.64 | 119.91 | 119.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 120.00 | 119.93 | 119.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 119.86 | 119.88 | 119.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 120.10 | 119.92 | 119.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:45:00 | 119.40 | 119.92 | 119.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 132.00 | 132.63 | 131.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 131.64 | 132.63 | 131.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 131.06 | 132.32 | 131.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 131.06 | 132.32 | 131.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 130.95 | 132.04 | 131.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 130.15 | 132.04 | 131.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 130.36 | 131.34 | 131.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:30:00 | 130.86 | 131.26 | 131.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 130.98 | 131.25 | 131.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 130.98 | 131.25 | 131.25 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 12:15:00 | 131.43 | 131.28 | 131.26 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 14:15:00 | 131.00 | 131.21 | 131.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 15:15:00 | 130.62 | 131.10 | 131.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 10:15:00 | 131.17 | 131.01 | 131.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 10:15:00 | 131.17 | 131.01 | 131.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 131.17 | 131.01 | 131.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:30:00 | 131.41 | 131.01 | 131.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 11:15:00 | 132.10 | 131.22 | 131.21 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 130.60 | 131.17 | 131.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 129.76 | 130.42 | 130.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 13:15:00 | 128.12 | 127.64 | 128.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 14:00:00 | 128.12 | 127.64 | 128.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 128.04 | 127.72 | 128.49 | EMA400 retest candle locked (from downside) |

### Cycle 164 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 132.48 | 129.28 | 129.05 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 128.70 | 129.50 | 129.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 127.87 | 128.92 | 129.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 129.76 | 128.99 | 129.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 129.76 | 128.99 | 129.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 129.76 | 128.99 | 129.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:45:00 | 130.29 | 128.99 | 129.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 129.80 | 129.15 | 129.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:15:00 | 130.92 | 129.15 | 129.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 11:15:00 | 131.30 | 129.58 | 129.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 12:15:00 | 132.01 | 130.07 | 129.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 11:15:00 | 134.97 | 135.86 | 134.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 11:45:00 | 135.05 | 135.86 | 134.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 134.41 | 135.27 | 134.49 | EMA400 retest candle locked (from upside) |

### Cycle 167 — SELL (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 09:15:00 | 133.86 | 134.12 | 134.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 10:15:00 | 132.89 | 133.87 | 134.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 14:15:00 | 133.51 | 133.48 | 133.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 14:15:00 | 133.51 | 133.48 | 133.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 133.51 | 133.48 | 133.75 | EMA400 retest candle locked (from downside) |

### Cycle 168 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 135.27 | 133.96 | 133.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 11:15:00 | 135.44 | 134.26 | 134.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 134.25 | 134.48 | 134.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 14:15:00 | 134.25 | 134.48 | 134.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 134.25 | 134.48 | 134.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:45:00 | 134.25 | 134.48 | 134.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 134.00 | 134.38 | 134.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 135.96 | 134.38 | 134.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 10:15:00 | 133.63 | 134.31 | 134.21 | SL hit (close<static) qty=1.00 sl=133.71 alert=retest2 |

### Cycle 169 — SELL (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 13:15:00 | 134.00 | 134.14 | 134.15 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 135.08 | 134.27 | 134.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 136.65 | 134.94 | 134.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 11:15:00 | 136.32 | 136.46 | 135.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 12:00:00 | 136.32 | 136.46 | 135.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 136.53 | 136.47 | 135.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:45:00 | 136.49 | 136.47 | 135.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 139.48 | 140.57 | 139.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 139.30 | 140.57 | 139.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 139.44 | 140.34 | 139.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 143.62 | 140.34 | 139.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 12:15:00 | 141.10 | 141.76 | 141.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 12:15:00 | 141.10 | 141.76 | 141.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 140.36 | 141.35 | 141.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 09:15:00 | 142.14 | 141.31 | 141.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 142.14 | 141.31 | 141.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 142.14 | 141.31 | 141.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:45:00 | 142.62 | 141.31 | 141.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 141.90 | 141.42 | 141.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 11:30:00 | 141.39 | 141.51 | 141.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:30:00 | 141.35 | 141.45 | 141.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 15:15:00 | 141.80 | 141.59 | 141.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 15:15:00 | 141.80 | 141.59 | 141.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 09:15:00 | 143.16 | 141.91 | 141.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 11:15:00 | 141.47 | 141.89 | 141.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 11:15:00 | 141.47 | 141.89 | 141.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 141.47 | 141.89 | 141.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:00:00 | 141.47 | 141.89 | 141.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 142.85 | 142.09 | 141.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 13:45:00 | 143.09 | 142.27 | 141.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 14:45:00 | 142.90 | 142.31 | 142.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 09:30:00 | 143.08 | 142.46 | 142.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 10:15:00 | 143.04 | 142.46 | 142.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 141.90 | 142.44 | 142.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:45:00 | 142.11 | 142.44 | 142.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 142.27 | 142.41 | 142.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 14:00:00 | 142.54 | 142.44 | 142.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 10:15:00 | 141.25 | 142.29 | 142.24 | SL hit (close<static) qty=1.00 sl=141.46 alert=retest2 |

### Cycle 173 — SELL (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 12:15:00 | 141.66 | 142.14 | 142.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 13:15:00 | 141.31 | 141.98 | 142.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 14:15:00 | 139.92 | 139.86 | 140.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-03 15:00:00 | 139.92 | 139.86 | 140.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 138.75 | 139.62 | 140.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 12:00:00 | 137.99 | 139.15 | 140.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 140.80 | 139.86 | 139.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — BUY (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 09:15:00 | 140.80 | 139.86 | 139.77 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 12:15:00 | 139.28 | 139.70 | 139.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 137.87 | 139.16 | 139.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 15:15:00 | 138.84 | 138.65 | 139.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 09:15:00 | 138.67 | 138.65 | 139.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 137.99 | 138.51 | 138.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 10:45:00 | 137.86 | 138.40 | 138.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:15:00 | 137.81 | 138.40 | 138.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 13:30:00 | 137.89 | 138.20 | 138.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 15:00:00 | 137.48 | 138.06 | 138.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 137.68 | 137.37 | 137.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 137.65 | 137.37 | 137.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 136.79 | 137.25 | 137.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:45:00 | 135.90 | 136.89 | 137.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 11:15:00 | 137.04 | 136.48 | 136.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 11:15:00 | 137.04 | 136.48 | 136.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 12:15:00 | 137.57 | 136.70 | 136.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 136.76 | 136.92 | 136.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 136.76 | 136.92 | 136.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 136.76 | 136.92 | 136.74 | EMA400 retest candle locked (from upside) |

### Cycle 177 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 135.41 | 136.45 | 136.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 13:15:00 | 134.99 | 135.95 | 136.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 136.35 | 135.63 | 136.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 136.35 | 135.63 | 136.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 136.35 | 135.63 | 136.02 | EMA400 retest candle locked (from downside) |

### Cycle 178 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 138.95 | 136.69 | 136.40 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 15:15:00 | 136.60 | 137.03 | 137.08 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 139.97 | 137.62 | 137.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 13:15:00 | 140.69 | 139.02 | 138.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 10:15:00 | 139.89 | 139.91 | 138.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 10:45:00 | 139.86 | 139.91 | 138.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 138.98 | 139.60 | 139.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:00:00 | 138.98 | 139.60 | 139.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 140.18 | 139.72 | 139.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:45:00 | 140.39 | 139.75 | 139.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:00:00 | 140.40 | 139.78 | 139.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 13:30:00 | 140.35 | 139.97 | 139.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 14:00:00 | 140.36 | 139.97 | 139.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 141.58 | 141.63 | 141.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 141.55 | 141.63 | 141.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 140.44 | 141.39 | 140.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:45:00 | 140.03 | 141.39 | 140.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 139.89 | 141.09 | 140.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:00:00 | 139.89 | 141.09 | 140.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-03 13:15:00 | 140.05 | 140.72 | 140.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — SELL (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 13:15:00 | 140.05 | 140.72 | 140.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 139.58 | 140.28 | 140.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 12:15:00 | 140.15 | 139.99 | 140.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 12:15:00 | 140.15 | 139.99 | 140.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 140.15 | 139.99 | 140.30 | EMA400 retest candle locked (from downside) |

### Cycle 182 — BUY (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 09:15:00 | 142.46 | 140.62 | 140.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 12:15:00 | 144.70 | 142.86 | 142.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 13:15:00 | 144.87 | 144.96 | 143.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 13:15:00 | 144.87 | 144.96 | 143.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 144.87 | 144.96 | 143.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 144.87 | 144.96 | 143.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 142.50 | 144.47 | 143.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 142.50 | 144.47 | 143.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 142.22 | 144.02 | 143.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 149.40 | 144.02 | 143.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 15:15:00 | 147.05 | 147.88 | 147.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 147.05 | 147.88 | 147.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 146.11 | 147.52 | 147.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 13:15:00 | 146.25 | 145.89 | 146.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 13:15:00 | 146.25 | 145.89 | 146.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 146.25 | 145.89 | 146.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 14:00:00 | 146.25 | 145.89 | 146.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 146.26 | 145.97 | 146.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:15:00 | 146.85 | 145.97 | 146.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 146.85 | 146.14 | 146.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 145.39 | 146.14 | 146.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 148.40 | 145.63 | 145.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 09:15:00 | 148.40 | 145.63 | 145.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 15:15:00 | 149.22 | 147.85 | 146.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 12:15:00 | 156.91 | 157.20 | 154.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 12:45:00 | 156.77 | 157.20 | 154.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 160.91 | 160.93 | 159.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 12:30:00 | 162.16 | 161.40 | 160.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 14:45:00 | 162.67 | 161.77 | 160.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:30:00 | 162.49 | 162.10 | 160.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 160.23 | 160.66 | 160.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 11:15:00 | 160.23 | 160.66 | 160.69 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 15:15:00 | 160.83 | 160.69 | 160.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 09:15:00 | 161.85 | 160.92 | 160.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 10:15:00 | 160.89 | 160.92 | 160.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 10:15:00 | 160.89 | 160.92 | 160.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 160.89 | 160.92 | 160.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 160.89 | 160.92 | 160.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 159.37 | 160.61 | 160.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 159.00 | 160.29 | 160.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 13:15:00 | 159.20 | 158.98 | 159.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 14:00:00 | 159.20 | 158.98 | 159.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 159.47 | 159.08 | 159.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 159.47 | 159.08 | 159.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 159.31 | 159.13 | 159.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 159.15 | 159.13 | 159.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 159.82 | 159.27 | 159.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:15:00 | 158.66 | 159.27 | 159.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:30:00 | 158.50 | 159.12 | 159.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:30:00 | 158.42 | 158.92 | 159.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 11:15:00 | 160.53 | 159.34 | 159.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 160.53 | 159.34 | 159.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 163.45 | 160.65 | 160.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 165.97 | 166.81 | 165.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 165.97 | 166.81 | 165.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 165.97 | 166.81 | 165.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:45:00 | 168.00 | 166.77 | 166.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 10:15:00 | 173.50 | 175.17 | 175.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 173.50 | 175.17 | 175.27 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 09:15:00 | 176.25 | 174.89 | 174.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 10:15:00 | 177.18 | 175.35 | 175.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 12:15:00 | 187.78 | 188.25 | 186.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 12:45:00 | 188.23 | 188.25 | 186.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 186.75 | 187.95 | 186.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 186.75 | 187.95 | 186.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 186.96 | 187.68 | 186.41 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 185.16 | 186.18 | 186.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 12:15:00 | 184.99 | 185.82 | 186.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 13:15:00 | 185.82 | 185.82 | 185.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 13:15:00 | 185.82 | 185.82 | 185.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 185.82 | 185.82 | 185.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:00:00 | 185.82 | 185.82 | 185.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 186.04 | 185.86 | 186.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 15:00:00 | 186.04 | 185.86 | 186.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 186.14 | 185.92 | 186.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 184.60 | 185.92 | 186.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 188.66 | 185.57 | 185.61 | SL hit (close>static) qty=1.00 sl=186.33 alert=retest2 |

### Cycle 192 — BUY (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 10:15:00 | 189.90 | 186.44 | 186.00 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2026-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 11:15:00 | 184.48 | 186.40 | 186.46 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 187.20 | 186.57 | 186.52 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 09:15:00 | 186.09 | 186.48 | 186.48 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 186.76 | 186.53 | 186.50 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 12:15:00 | 185.91 | 186.39 | 186.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 13:15:00 | 184.70 | 186.05 | 186.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 10:15:00 | 186.45 | 185.84 | 186.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 10:15:00 | 186.45 | 185.84 | 186.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 186.45 | 185.84 | 186.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 186.45 | 185.84 | 186.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 186.58 | 185.99 | 186.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:30:00 | 187.29 | 185.99 | 186.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 187.24 | 186.24 | 186.23 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 09:15:00 | 184.85 | 186.22 | 186.25 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2026-01-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 13:15:00 | 187.89 | 186.13 | 185.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 15:15:00 | 188.49 | 186.89 | 186.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 186.32 | 186.77 | 186.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 186.32 | 186.77 | 186.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 186.32 | 186.77 | 186.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 185.56 | 186.77 | 186.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 185.70 | 186.56 | 186.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 185.70 | 186.56 | 186.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 184.20 | 186.09 | 186.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 184.20 | 186.09 | 186.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 183.27 | 185.52 | 185.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 180.92 | 184.60 | 185.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 182.46 | 181.48 | 183.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 12:15:00 | 182.46 | 181.48 | 183.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 182.46 | 181.48 | 183.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 182.46 | 181.48 | 183.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 186.92 | 182.33 | 182.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 188.45 | 182.33 | 182.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 186.15 | 183.66 | 183.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 13:15:00 | 188.86 | 185.11 | 184.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 09:15:00 | 191.24 | 192.46 | 189.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 191.24 | 192.46 | 189.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 191.24 | 192.46 | 189.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 09:15:00 | 196.38 | 191.79 | 190.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 15:00:00 | 195.30 | 192.09 | 191.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 11:15:00 | 194.27 | 193.14 | 191.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:45:00 | 194.48 | 193.18 | 192.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 194.03 | 193.55 | 192.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 194.78 | 194.12 | 193.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 13:15:00 | 194.88 | 194.48 | 193.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 189.36 | 193.76 | 193.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 189.36 | 193.76 | 193.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 187.29 | 191.72 | 192.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 191.65 | 190.61 | 191.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 14:00:00 | 191.65 | 190.61 | 191.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 194.39 | 191.36 | 192.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 194.39 | 191.36 | 192.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 195.00 | 192.09 | 192.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 199.06 | 192.09 | 192.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 197.64 | 193.20 | 192.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 203.59 | 198.73 | 196.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 11:15:00 | 200.34 | 200.80 | 199.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 12:00:00 | 200.34 | 200.80 | 199.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 199.79 | 200.93 | 199.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 199.58 | 200.93 | 199.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 200.46 | 200.84 | 199.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 199.68 | 200.84 | 199.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 200.10 | 200.77 | 200.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 200.10 | 200.77 | 200.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 205.08 | 210.13 | 207.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:00:00 | 205.08 | 210.13 | 207.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 205.98 | 209.30 | 207.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:30:00 | 211.01 | 208.69 | 207.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 11:15:00 | 205.26 | 208.17 | 208.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 11:15:00 | 205.26 | 208.17 | 208.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 204.98 | 206.74 | 207.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 205.83 | 205.49 | 206.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 205.83 | 205.49 | 206.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 205.12 | 205.42 | 206.14 | EMA400 retest candle locked (from downside) |

### Cycle 206 — BUY (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 15:15:00 | 207.50 | 206.50 | 206.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 208.78 | 206.96 | 206.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 206.00 | 207.66 | 207.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 206.00 | 207.66 | 207.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 206.00 | 207.66 | 207.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 206.00 | 207.66 | 207.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 207.59 | 207.64 | 207.31 | EMA400 retest candle locked (from upside) |

### Cycle 207 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 205.50 | 207.01 | 207.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 203.10 | 206.23 | 206.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 205.75 | 205.48 | 206.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 10:15:00 | 205.75 | 205.48 | 206.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 205.75 | 205.48 | 206.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:45:00 | 206.11 | 205.48 | 206.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 208.13 | 206.01 | 206.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:45:00 | 207.93 | 206.01 | 206.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 207.22 | 206.25 | 206.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 14:15:00 | 206.98 | 206.48 | 206.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 14:15:00 | 208.95 | 206.97 | 206.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 14:15:00 | 208.95 | 206.97 | 206.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 210.98 | 207.99 | 207.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 13:15:00 | 209.77 | 209.93 | 209.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 13:15:00 | 209.77 | 209.93 | 209.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 209.77 | 209.93 | 209.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:30:00 | 209.10 | 209.93 | 209.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 210.18 | 210.71 | 210.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:00:00 | 210.18 | 210.71 | 210.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 211.88 | 210.95 | 210.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:30:00 | 209.95 | 210.95 | 210.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 211.04 | 211.31 | 210.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:45:00 | 210.95 | 211.31 | 210.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 210.77 | 211.20 | 210.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 13:00:00 | 210.77 | 211.20 | 210.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 211.53 | 211.27 | 210.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 15:00:00 | 211.76 | 211.37 | 210.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:00:00 | 212.27 | 211.53 | 211.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 15:15:00 | 211.80 | 211.78 | 211.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 09:30:00 | 212.33 | 211.81 | 211.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 10:15:00 | 210.50 | 211.55 | 211.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 10:45:00 | 210.92 | 211.55 | 211.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 207.80 | 210.80 | 211.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 207.80 | 210.80 | 211.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 204.08 | 208.52 | 209.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 12:15:00 | 201.46 | 201.28 | 203.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 12:30:00 | 201.65 | 201.28 | 203.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 202.59 | 201.46 | 203.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 202.59 | 201.46 | 203.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 193.29 | 189.62 | 193.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 11:00:00 | 190.12 | 192.05 | 193.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 180.61 | 186.57 | 189.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-13 10:15:00 | 171.11 | 177.01 | 182.32 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 210 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 178.36 | 175.78 | 175.69 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 171.62 | 175.76 | 176.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 171.15 | 174.17 | 175.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 15:15:00 | 169.40 | 169.24 | 171.22 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 09:15:00 | 163.82 | 169.24 | 171.22 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 162.71 | 163.45 | 166.44 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 166.03 | 163.90 | 165.89 | SL hit (close>ema400) qty=1.00 sl=165.89 alert=retest1 |

### Cycle 212 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 172.24 | 167.78 | 167.18 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 163.57 | 167.16 | 167.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 162.95 | 165.58 | 166.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 11:15:00 | 157.67 | 157.09 | 160.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 12:00:00 | 157.67 | 157.09 | 160.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 150.80 | 148.79 | 150.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 15:00:00 | 150.80 | 148.79 | 150.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 15:15:00 | 150.70 | 149.17 | 150.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:15:00 | 149.61 | 149.17 | 150.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 11:15:00 | 151.55 | 149.72 | 150.52 | SL hit (close>static) qty=1.00 sl=151.52 alert=retest2 |

### Cycle 214 — BUY (started 2026-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 14:15:00 | 153.10 | 150.91 | 150.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 170.20 | 155.02 | 152.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 172.56 | 175.06 | 171.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 172.56 | 175.06 | 171.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 172.56 | 175.06 | 171.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 177.66 | 173.11 | 171.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 12:15:00 | 174.20 | 175.56 | 175.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 13:15:00 | 174.59 | 175.23 | 175.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 13:15:00 | 173.75 | 174.93 | 174.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 215 — SELL (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 13:15:00 | 173.75 | 174.93 | 174.94 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-04-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 11:15:00 | 176.33 | 175.12 | 175.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 12:15:00 | 176.88 | 175.47 | 175.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 175.38 | 175.63 | 175.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 15:15:00 | 175.38 | 175.63 | 175.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 175.38 | 175.63 | 175.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 177.34 | 175.63 | 175.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 10:00:00 | 176.65 | 176.70 | 176.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 172.95 | 177.50 | 177.18 | SL hit (close<static) qty=1.00 sl=175.04 alert=retest2 |

### Cycle 217 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 173.58 | 176.71 | 176.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 12:15:00 | 172.19 | 175.17 | 176.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 170.48 | 169.94 | 171.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 13:00:00 | 170.48 | 169.94 | 171.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 170.83 | 170.00 | 170.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:45:00 | 171.80 | 170.00 | 170.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 170.23 | 170.04 | 170.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 169.53 | 170.04 | 170.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 11:15:00 | 161.05 | 164.05 | 166.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 162.89 | 162.80 | 164.77 | SL hit (close>ema200) qty=0.50 sl=162.80 alert=retest2 |

### Cycle 218 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 167.81 | 162.64 | 162.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 168.40 | 163.80 | 162.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 168.43 | 168.87 | 166.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 10:00:00 | 168.43 | 168.87 | 166.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-15 09:15:00 | 75.20 | 2023-05-23 12:15:00 | 76.38 | STOP_HIT | 1.00 | 1.57% |
| SELL | retest2 | 2023-05-30 10:45:00 | 72.55 | 2023-06-01 09:15:00 | 73.15 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2023-06-13 09:15:00 | 77.08 | 2023-06-19 09:15:00 | 84.79 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-07-06 12:30:00 | 81.75 | 2023-07-10 09:15:00 | 82.90 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2023-07-07 09:30:00 | 81.75 | 2023-07-10 09:15:00 | 82.90 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2023-07-07 10:30:00 | 81.47 | 2023-07-10 09:15:00 | 82.90 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2023-07-07 12:00:00 | 81.58 | 2023-07-10 09:15:00 | 82.90 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2023-07-10 12:45:00 | 81.63 | 2023-07-11 11:15:00 | 82.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2023-07-10 15:00:00 | 81.68 | 2023-07-11 11:15:00 | 82.50 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2023-07-18 13:15:00 | 86.38 | 2023-07-28 11:15:00 | 90.20 | STOP_HIT | 1.00 | 4.42% |
| BUY | retest2 | 2023-07-19 14:45:00 | 86.50 | 2023-07-28 11:15:00 | 90.20 | STOP_HIT | 1.00 | 4.28% |
| BUY | retest2 | 2023-07-20 10:00:00 | 86.55 | 2023-07-28 11:15:00 | 90.20 | STOP_HIT | 1.00 | 4.22% |
| BUY | retest2 | 2023-07-20 11:00:00 | 86.50 | 2023-07-28 11:15:00 | 90.20 | STOP_HIT | 1.00 | 4.28% |
| BUY | retest2 | 2023-07-20 13:30:00 | 87.05 | 2023-07-28 11:15:00 | 90.20 | STOP_HIT | 1.00 | 3.62% |
| BUY | retest2 | 2023-08-11 09:15:00 | 94.73 | 2023-08-18 11:15:00 | 92.93 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2023-08-11 11:30:00 | 93.55 | 2023-08-18 11:15:00 | 92.93 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2023-08-11 14:30:00 | 93.55 | 2023-08-18 11:15:00 | 92.93 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2023-08-14 10:45:00 | 93.75 | 2023-08-18 11:15:00 | 92.93 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2023-08-16 11:15:00 | 94.18 | 2023-08-18 11:15:00 | 92.93 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest1 | 2023-09-04 10:15:00 | 92.08 | 2023-09-08 09:15:00 | 91.73 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest1 | 2023-09-04 13:45:00 | 91.95 | 2023-09-08 09:15:00 | 91.73 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2023-09-08 10:30:00 | 91.43 | 2023-09-08 11:15:00 | 92.55 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2023-09-20 12:15:00 | 91.58 | 2023-09-20 13:15:00 | 91.05 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2023-10-04 09:15:00 | 87.93 | 2023-10-10 10:15:00 | 87.58 | STOP_HIT | 1.00 | 0.40% |
| BUY | retest2 | 2023-10-13 11:00:00 | 88.10 | 2023-10-18 12:15:00 | 87.95 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2023-10-13 13:15:00 | 88.05 | 2023-10-18 15:15:00 | 87.95 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2023-10-13 14:45:00 | 88.23 | 2023-10-18 15:15:00 | 87.95 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2023-10-16 09:15:00 | 88.18 | 2023-10-18 15:15:00 | 87.95 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2023-10-17 15:15:00 | 88.45 | 2023-10-18 15:15:00 | 87.95 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2023-10-20 10:30:00 | 87.68 | 2023-10-26 09:15:00 | 83.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 10:30:00 | 87.68 | 2023-10-27 09:15:00 | 83.83 | STOP_HIT | 0.50 | 4.39% |
| BUY | retest2 | 2023-11-08 09:15:00 | 85.05 | 2023-11-17 13:15:00 | 86.80 | STOP_HIT | 1.00 | 2.06% |
| BUY | retest2 | 2023-11-24 11:15:00 | 88.83 | 2023-12-01 14:15:00 | 88.38 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2023-12-07 09:15:00 | 87.75 | 2023-12-07 09:15:00 | 88.70 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2023-12-07 11:00:00 | 88.20 | 2023-12-07 11:15:00 | 88.83 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2023-12-12 10:15:00 | 87.33 | 2023-12-14 10:15:00 | 87.98 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2023-12-12 12:00:00 | 87.40 | 2023-12-14 10:15:00 | 87.98 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2023-12-22 13:30:00 | 85.75 | 2023-12-26 09:15:00 | 87.35 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-01-12 10:00:00 | 87.88 | 2024-01-16 09:15:00 | 88.80 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-01-15 09:45:00 | 87.85 | 2024-01-16 09:15:00 | 88.80 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-01-15 11:15:00 | 87.80 | 2024-01-16 09:15:00 | 88.80 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-01-19 14:15:00 | 86.38 | 2024-01-20 10:15:00 | 87.43 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-01-31 13:45:00 | 87.10 | 2024-02-08 10:15:00 | 87.18 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2024-01-31 14:30:00 | 87.25 | 2024-02-08 10:15:00 | 87.18 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2024-02-02 10:00:00 | 87.15 | 2024-02-08 10:15:00 | 87.18 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2024-02-02 11:00:00 | 87.15 | 2024-02-08 10:15:00 | 87.18 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2024-02-05 09:15:00 | 88.10 | 2024-02-08 10:15:00 | 87.18 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-02-05 11:15:00 | 87.43 | 2024-02-08 10:15:00 | 87.18 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-02-05 11:45:00 | 87.93 | 2024-02-08 10:15:00 | 87.18 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-02-14 09:15:00 | 85.70 | 2024-02-14 15:15:00 | 87.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-02-14 13:45:00 | 86.08 | 2024-02-14 15:15:00 | 87.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-02-14 14:45:00 | 86.15 | 2024-02-14 15:15:00 | 87.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-02-20 13:15:00 | 86.65 | 2024-02-23 10:15:00 | 86.95 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-02-21 10:15:00 | 86.50 | 2024-02-23 10:15:00 | 86.95 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-02-21 13:15:00 | 86.55 | 2024-02-23 10:15:00 | 86.95 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-02-26 12:15:00 | 87.10 | 2024-02-28 10:15:00 | 86.30 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-02-26 12:45:00 | 87.20 | 2024-02-28 10:15:00 | 86.30 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-02-27 09:15:00 | 87.38 | 2024-02-28 10:15:00 | 86.30 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-02-27 10:30:00 | 87.08 | 2024-02-28 10:15:00 | 86.30 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-03-15 10:15:00 | 80.38 | 2024-03-21 09:15:00 | 82.05 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-03-15 10:45:00 | 80.53 | 2024-03-21 09:15:00 | 82.05 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-03-15 13:00:00 | 80.60 | 2024-03-21 09:15:00 | 82.05 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-03-18 10:00:00 | 80.50 | 2024-03-21 09:15:00 | 82.05 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-03-19 09:15:00 | 81.00 | 2024-03-21 09:15:00 | 82.05 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-04-03 12:15:00 | 87.38 | 2024-04-04 11:15:00 | 86.43 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-04-03 15:15:00 | 87.35 | 2024-04-04 11:15:00 | 86.43 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-04-10 09:30:00 | 88.30 | 2024-04-15 10:15:00 | 87.80 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-04-10 10:15:00 | 88.43 | 2024-04-15 10:15:00 | 87.80 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-04-18 09:15:00 | 87.25 | 2024-04-23 09:15:00 | 87.15 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest1 | 2024-04-26 09:15:00 | 89.05 | 2024-04-26 12:15:00 | 93.50 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-04-26 09:15:00 | 89.05 | 2024-04-29 14:15:00 | 92.63 | STOP_HIT | 0.50 | 4.02% |
| BUY | retest2 | 2024-05-22 12:15:00 | 105.33 | 2024-05-23 10:15:00 | 103.63 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-05-27 10:30:00 | 109.40 | 2024-05-30 12:15:00 | 110.40 | STOP_HIT | 1.00 | 0.91% |
| BUY | retest2 | 2024-06-13 13:30:00 | 118.69 | 2024-06-19 09:15:00 | 116.65 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-06-14 09:15:00 | 118.93 | 2024-06-19 09:15:00 | 116.65 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-06-14 11:30:00 | 118.82 | 2024-06-19 09:15:00 | 116.65 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2024-06-26 11:45:00 | 121.46 | 2024-07-01 14:15:00 | 119.31 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-06-28 09:30:00 | 121.97 | 2024-07-01 14:15:00 | 119.31 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2024-07-08 10:45:00 | 114.19 | 2024-07-15 13:15:00 | 114.04 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2024-07-08 11:30:00 | 113.82 | 2024-07-15 13:15:00 | 114.04 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-07-09 14:15:00 | 114.12 | 2024-07-15 13:15:00 | 114.04 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2024-07-09 15:15:00 | 114.08 | 2024-07-15 13:15:00 | 114.04 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2024-07-10 10:30:00 | 111.90 | 2024-07-15 13:15:00 | 114.04 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-07-24 14:30:00 | 115.42 | 2024-07-29 09:15:00 | 126.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-25 09:30:00 | 115.58 | 2024-07-29 09:15:00 | 127.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-25 13:00:00 | 115.51 | 2024-07-29 09:15:00 | 127.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-25 14:15:00 | 115.95 | 2024-07-29 09:15:00 | 127.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-26 09:15:00 | 121.18 | 2024-08-01 13:15:00 | 123.95 | STOP_HIT | 1.00 | 2.29% |
| SELL | retest2 | 2024-08-06 14:30:00 | 122.08 | 2024-08-08 12:15:00 | 124.28 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-08-06 15:00:00 | 121.38 | 2024-08-08 12:15:00 | 124.28 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2024-08-23 12:00:00 | 130.82 | 2024-08-26 10:15:00 | 129.45 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-08-28 12:30:00 | 131.03 | 2024-08-28 14:15:00 | 130.28 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-08-28 13:15:00 | 131.03 | 2024-08-28 14:15:00 | 130.28 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2024-09-02 09:15:00 | 126.70 | 2024-09-11 14:15:00 | 120.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-02 09:15:00 | 126.70 | 2024-09-12 13:15:00 | 122.05 | STOP_HIT | 0.50 | 3.67% |
| BUY | retest2 | 2024-09-27 14:30:00 | 120.15 | 2024-09-30 13:15:00 | 119.28 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-09-30 10:00:00 | 120.18 | 2024-09-30 13:15:00 | 119.28 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-10-03 09:15:00 | 117.85 | 2024-10-04 13:15:00 | 111.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 117.85 | 2024-10-08 13:15:00 | 111.47 | STOP_HIT | 0.50 | 5.41% |
| SELL | retest2 | 2024-10-17 12:30:00 | 110.53 | 2024-10-29 09:15:00 | 105.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 10:45:00 | 110.45 | 2024-10-29 09:15:00 | 104.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:30:00 | 110.58 | 2024-10-29 09:15:00 | 105.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:30:00 | 110.17 | 2024-10-29 09:15:00 | 104.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 12:30:00 | 110.53 | 2024-10-30 10:15:00 | 105.19 | STOP_HIT | 0.50 | 4.83% |
| SELL | retest2 | 2024-10-18 10:45:00 | 110.45 | 2024-10-30 10:15:00 | 105.19 | STOP_HIT | 0.50 | 4.76% |
| SELL | retest2 | 2024-10-21 09:30:00 | 110.58 | 2024-10-30 10:15:00 | 105.19 | STOP_HIT | 0.50 | 4.87% |
| SELL | retest2 | 2024-10-21 12:30:00 | 110.17 | 2024-10-30 10:15:00 | 105.19 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2024-10-25 10:30:00 | 105.98 | 2024-11-05 13:15:00 | 105.97 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2024-10-28 10:00:00 | 106.15 | 2024-11-05 13:15:00 | 105.97 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2024-10-28 12:15:00 | 106.16 | 2024-11-05 13:15:00 | 105.97 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2024-10-28 13:30:00 | 106.13 | 2024-11-05 13:15:00 | 105.97 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2024-11-08 13:15:00 | 108.98 | 2024-11-13 10:15:00 | 108.20 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-11-13 10:15:00 | 108.74 | 2024-11-13 10:15:00 | 108.20 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-11-28 14:15:00 | 116.68 | 2024-11-28 14:15:00 | 115.77 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-01-15 09:15:00 | 104.43 | 2025-01-20 14:15:00 | 104.72 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-01-15 11:00:00 | 104.70 | 2025-01-20 14:15:00 | 104.72 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-01-16 11:15:00 | 104.67 | 2025-01-20 14:15:00 | 104.72 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-01-16 11:45:00 | 104.74 | 2025-01-20 14:15:00 | 104.72 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2025-01-16 14:15:00 | 104.11 | 2025-01-20 14:15:00 | 104.72 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-01-23 09:15:00 | 102.51 | 2025-01-23 10:15:00 | 104.12 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-02-14 14:30:00 | 107.89 | 2025-02-14 15:15:00 | 106.45 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-02-17 10:15:00 | 108.47 | 2025-02-21 15:15:00 | 110.90 | STOP_HIT | 1.00 | 2.24% |
| SELL | retest2 | 2025-03-05 13:15:00 | 104.44 | 2025-03-06 10:15:00 | 106.58 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-03-05 15:00:00 | 104.61 | 2025-03-06 10:15:00 | 106.58 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest1 | 2025-03-20 09:15:00 | 103.93 | 2025-03-25 12:15:00 | 104.98 | STOP_HIT | 1.00 | 1.01% |
| BUY | retest1 | 2025-03-20 10:15:00 | 103.69 | 2025-03-25 12:15:00 | 104.98 | STOP_HIT | 1.00 | 1.24% |
| BUY | retest2 | 2025-03-21 10:15:00 | 104.93 | 2025-03-27 14:15:00 | 104.03 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-03-25 13:00:00 | 104.98 | 2025-03-27 14:15:00 | 104.03 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-04-02 09:15:00 | 104.00 | 2025-04-02 14:15:00 | 104.39 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-04-02 10:45:00 | 104.13 | 2025-04-02 14:15:00 | 104.39 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-04-02 12:45:00 | 104.20 | 2025-04-02 14:15:00 | 104.39 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-04-08 10:30:00 | 99.15 | 2025-04-09 11:15:00 | 102.05 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-05-05 12:15:00 | 111.12 | 2025-05-05 13:15:00 | 112.24 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-05-06 11:00:00 | 111.27 | 2025-05-07 10:15:00 | 112.19 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-05-06 14:00:00 | 110.77 | 2025-05-07 10:15:00 | 112.19 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-05-21 10:45:00 | 122.43 | 2025-05-22 09:15:00 | 119.35 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-05-21 12:30:00 | 122.30 | 2025-05-22 09:15:00 | 119.35 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-05-21 14:00:00 | 122.46 | 2025-05-22 09:15:00 | 119.35 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-06-17 09:15:00 | 117.45 | 2025-06-20 15:15:00 | 117.40 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-06-17 10:00:00 | 117.63 | 2025-06-20 15:15:00 | 117.40 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-06-17 11:45:00 | 117.35 | 2025-06-20 15:15:00 | 117.40 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-06-18 10:15:00 | 117.55 | 2025-06-20 15:15:00 | 117.40 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-06-18 11:45:00 | 117.23 | 2025-06-20 15:15:00 | 117.40 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-06-18 12:45:00 | 117.24 | 2025-06-20 15:15:00 | 117.40 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-06-18 14:15:00 | 117.30 | 2025-06-20 15:15:00 | 117.40 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-06-19 10:00:00 | 117.37 | 2025-06-20 15:15:00 | 117.40 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-07-01 14:30:00 | 125.78 | 2025-07-04 11:15:00 | 124.65 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-07-02 09:15:00 | 125.75 | 2025-07-04 11:15:00 | 124.65 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-02 11:15:00 | 125.63 | 2025-07-04 11:15:00 | 124.65 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-07-04 09:15:00 | 125.65 | 2025-07-04 11:15:00 | 124.65 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-07-28 11:30:00 | 123.10 | 2025-08-04 11:15:00 | 122.71 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-07-29 12:30:00 | 123.50 | 2025-08-04 11:15:00 | 122.71 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2025-07-29 13:00:00 | 123.40 | 2025-08-04 11:15:00 | 122.71 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2025-07-29 15:00:00 | 123.25 | 2025-08-04 11:15:00 | 122.71 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-07-31 09:15:00 | 121.15 | 2025-08-04 11:15:00 | 122.71 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-07-31 13:45:00 | 121.15 | 2025-08-04 11:15:00 | 122.71 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-08-08 10:45:00 | 117.56 | 2025-08-11 14:15:00 | 120.06 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-08-22 11:30:00 | 130.86 | 2025-08-25 09:15:00 | 130.98 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-09-15 09:15:00 | 135.96 | 2025-09-15 10:15:00 | 133.63 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-09-23 09:15:00 | 143.62 | 2025-09-25 12:15:00 | 141.10 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-09-26 11:30:00 | 141.39 | 2025-09-26 15:15:00 | 141.80 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-09-26 13:30:00 | 141.35 | 2025-09-26 15:15:00 | 141.80 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-09-29 13:45:00 | 143.09 | 2025-10-01 10:15:00 | 141.25 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-09-29 14:45:00 | 142.90 | 2025-10-01 10:15:00 | 141.25 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-09-30 09:30:00 | 143.08 | 2025-10-01 10:15:00 | 141.25 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-09-30 10:15:00 | 143.04 | 2025-10-01 10:15:00 | 141.25 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-09-30 14:00:00 | 142.54 | 2025-10-01 10:15:00 | 141.25 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-10-06 12:00:00 | 137.99 | 2025-10-08 09:15:00 | 140.80 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-10-10 10:45:00 | 137.86 | 2025-10-16 11:15:00 | 137.04 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2025-10-10 11:15:00 | 137.81 | 2025-10-16 11:15:00 | 137.04 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2025-10-10 13:30:00 | 137.89 | 2025-10-16 11:15:00 | 137.04 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2025-10-10 15:00:00 | 137.48 | 2025-10-16 11:15:00 | 137.04 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-10-14 10:45:00 | 135.90 | 2025-10-16 11:15:00 | 137.04 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-10-29 09:45:00 | 140.39 | 2025-11-03 13:15:00 | 140.05 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-10-30 12:00:00 | 140.40 | 2025-11-03 13:15:00 | 140.05 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-10-30 13:30:00 | 140.35 | 2025-11-03 13:15:00 | 140.05 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-10-30 14:00:00 | 140.36 | 2025-11-03 13:15:00 | 140.05 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-11-13 09:15:00 | 149.40 | 2025-11-18 15:15:00 | 147.05 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-11-21 09:15:00 | 145.39 | 2025-11-25 09:15:00 | 148.40 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-12-03 12:30:00 | 162.16 | 2025-12-05 11:15:00 | 160.23 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-12-03 14:45:00 | 162.67 | 2025-12-05 11:15:00 | 160.23 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-12-04 09:30:00 | 162.49 | 2025-12-05 11:15:00 | 160.23 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-12-10 11:15:00 | 158.66 | 2025-12-11 11:15:00 | 160.53 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-12-10 12:30:00 | 158.50 | 2025-12-11 11:15:00 | 160.53 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-12-10 13:30:00 | 158.42 | 2025-12-11 11:15:00 | 160.53 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-12-18 11:45:00 | 168.00 | 2025-12-26 10:15:00 | 173.50 | STOP_HIT | 1.00 | 3.27% |
| SELL | retest2 | 2026-01-08 09:15:00 | 184.60 | 2026-01-09 09:15:00 | 188.66 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2026-01-28 09:15:00 | 196.38 | 2026-02-01 14:15:00 | 189.36 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2026-01-28 15:00:00 | 195.30 | 2026-02-01 14:15:00 | 189.36 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2026-01-29 11:15:00 | 194.27 | 2026-02-01 14:15:00 | 189.36 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2026-01-29 13:45:00 | 194.48 | 2026-02-01 14:15:00 | 189.36 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2026-01-30 10:30:00 | 194.78 | 2026-02-01 14:15:00 | 189.36 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2026-01-30 13:15:00 | 194.88 | 2026-02-01 14:15:00 | 189.36 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-02-12 09:30:00 | 211.01 | 2026-02-13 11:15:00 | 205.26 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2026-02-20 14:15:00 | 206.98 | 2026-02-20 14:15:00 | 208.95 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2026-02-26 15:00:00 | 211.76 | 2026-03-02 11:15:00 | 207.80 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2026-02-27 11:00:00 | 212.27 | 2026-03-02 11:15:00 | 207.80 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2026-02-27 15:15:00 | 211.80 | 2026-03-02 11:15:00 | 207.80 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2026-03-02 09:30:00 | 212.33 | 2026-03-02 11:15:00 | 207.80 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2026-03-11 11:00:00 | 190.12 | 2026-03-12 09:15:00 | 180.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 11:00:00 | 190.12 | 2026-03-13 10:15:00 | 171.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2026-03-23 09:15:00 | 163.82 | 2026-03-24 12:15:00 | 166.03 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-04-07 09:15:00 | 149.61 | 2026-04-07 11:15:00 | 151.55 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2026-04-15 09:15:00 | 177.66 | 2026-04-17 13:15:00 | 173.75 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2026-04-17 12:15:00 | 174.20 | 2026-04-17 13:15:00 | 173.75 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2026-04-17 13:15:00 | 174.59 | 2026-04-17 13:15:00 | 173.75 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2026-04-21 09:15:00 | 177.34 | 2026-04-23 09:15:00 | 172.95 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2026-04-22 10:00:00 | 176.65 | 2026-04-23 09:15:00 | 172.95 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2026-04-28 11:15:00 | 169.53 | 2026-04-30 11:15:00 | 161.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-28 11:15:00 | 169.53 | 2026-05-04 09:15:00 | 162.89 | STOP_HIT | 0.50 | 3.92% |
