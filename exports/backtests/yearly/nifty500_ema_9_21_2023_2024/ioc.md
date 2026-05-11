# Indian Oil Corporation Ltd. (IOC)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 144.88
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 207 |
| ALERT1 | 140 |
| ALERT2 | 138 |
| ALERT2_SKIP | 64 |
| ALERT3 | 379 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 172 |
| PARTIAL | 16 |
| TARGET_HIT | 1 |
| STOP_HIT | 178 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 195 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 78 / 117
- **Target hits / Stop hits / Partials:** 1 / 178 / 16
- **Avg / median % per leg:** 0.43% / -0.54%
- **Sum % (uncompounded):** 83.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 92 | 32 | 34.8% | 1 | 90 | 1 | -0.38% | -34.8% |
| BUY @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 1 | 6 | 1 | 0.23% | 1.9% |
| BUY @ 3rd Alert (retest2) | 84 | 30 | 35.7% | 0 | 84 | 0 | -0.44% | -36.7% |
| SELL (all) | 103 | 46 | 44.7% | 0 | 88 | 15 | 1.15% | 118.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 103 | 46 | 44.7% | 0 | 88 | 15 | 1.15% | 118.7% |
| retest1 (combined) | 8 | 2 | 25.0% | 1 | 6 | 1 | 0.23% | 1.9% |
| retest2 (combined) | 187 | 76 | 40.6% | 0 | 172 | 15 | 0.44% | 82.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 10:15:00 | 84.35 | 83.89 | 83.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-16 09:15:00 | 85.30 | 84.39 | 84.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-18 09:15:00 | 87.20 | 87.30 | 86.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 09:15:00 | 87.20 | 87.30 | 86.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 87.20 | 87.30 | 86.49 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 09:15:00 | 84.95 | 86.27 | 86.33 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 10:15:00 | 87.00 | 86.24 | 86.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 14:15:00 | 87.15 | 86.70 | 86.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 09:15:00 | 89.40 | 89.41 | 88.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-25 09:30:00 | 89.20 | 89.41 | 88.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 13:15:00 | 89.05 | 89.23 | 88.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 14:00:00 | 89.05 | 89.23 | 88.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 14:15:00 | 89.10 | 89.20 | 88.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 14:30:00 | 88.70 | 89.20 | 88.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 90.15 | 89.38 | 89.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 09:15:00 | 90.45 | 90.03 | 89.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 11:45:00 | 90.35 | 90.28 | 89.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 12:45:00 | 90.40 | 90.28 | 89.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 13:30:00 | 90.40 | 90.29 | 90.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 90.90 | 90.43 | 90.16 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-06-02 12:15:00 | 90.20 | 90.41 | 90.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 12:15:00 | 90.20 | 90.41 | 90.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-02 13:15:00 | 90.05 | 90.34 | 90.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-05 15:15:00 | 89.40 | 89.37 | 89.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-06 09:15:00 | 89.55 | 89.37 | 89.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 09:15:00 | 89.45 | 89.39 | 89.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-06 09:30:00 | 89.70 | 89.39 | 89.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 10:15:00 | 89.55 | 89.42 | 89.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-06 10:45:00 | 89.60 | 89.42 | 89.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 09:15:00 | 90.30 | 89.51 | 89.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 10:00:00 | 90.30 | 89.51 | 89.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2023-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 10:15:00 | 90.35 | 89.68 | 89.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 11:15:00 | 90.60 | 89.86 | 89.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 11:15:00 | 90.50 | 90.55 | 90.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 11:15:00 | 90.50 | 90.55 | 90.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 90.50 | 90.55 | 90.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:00:00 | 90.50 | 90.55 | 90.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 14:15:00 | 90.15 | 90.45 | 90.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 15:00:00 | 90.15 | 90.45 | 90.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 15:15:00 | 89.95 | 90.35 | 90.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-09 09:15:00 | 90.60 | 90.35 | 90.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-16 14:15:00 | 92.10 | 92.45 | 92.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-16 14:15:00 | 92.10 | 92.45 | 92.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 12:15:00 | 91.75 | 92.19 | 92.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 09:15:00 | 92.45 | 92.11 | 92.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 09:15:00 | 92.45 | 92.11 | 92.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 92.45 | 92.11 | 92.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 10:00:00 | 92.45 | 92.11 | 92.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 10:15:00 | 92.35 | 92.16 | 92.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 11:15:00 | 92.55 | 92.16 | 92.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 11:15:00 | 92.15 | 92.16 | 92.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 11:30:00 | 92.40 | 92.16 | 92.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 92.50 | 92.12 | 92.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 10:00:00 | 92.50 | 92.12 | 92.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 10:15:00 | 92.25 | 92.15 | 92.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 11:45:00 | 92.10 | 92.17 | 92.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 12:15:00 | 92.20 | 92.17 | 92.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 14:00:00 | 91.90 | 92.07 | 92.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-22 10:15:00 | 92.70 | 92.26 | 92.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2023-06-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-22 10:15:00 | 92.70 | 92.26 | 92.20 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 14:15:00 | 91.95 | 92.14 | 92.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 91.40 | 91.99 | 92.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 15:15:00 | 89.75 | 89.66 | 90.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-27 09:15:00 | 89.95 | 89.66 | 90.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 90.35 | 89.80 | 90.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:00:00 | 90.35 | 89.80 | 90.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 90.00 | 89.84 | 90.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 11:00:00 | 90.00 | 89.84 | 90.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 09:15:00 | 89.85 | 89.59 | 89.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-28 10:30:00 | 89.60 | 89.57 | 89.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-28 14:15:00 | 90.60 | 89.84 | 89.93 | SL hit (close>static) qty=1.00 sl=90.20 alert=retest2 |

### Cycle 9 — BUY (started 2023-06-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 15:15:00 | 91.00 | 90.07 | 90.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 10:15:00 | 91.35 | 90.49 | 90.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 14:15:00 | 94.70 | 94.76 | 93.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-04 14:30:00 | 94.60 | 94.76 | 93.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 94.65 | 94.70 | 93.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 12:15:00 | 95.20 | 94.86 | 94.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 13:45:00 | 95.20 | 94.95 | 94.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-13 10:15:00 | 97.95 | 98.08 | 98.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2023-07-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 10:15:00 | 97.95 | 98.08 | 98.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 11:15:00 | 97.35 | 97.93 | 98.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 09:15:00 | 97.25 | 97.10 | 97.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-14 10:00:00 | 97.25 | 97.10 | 97.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 10:15:00 | 97.30 | 97.14 | 97.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-14 12:30:00 | 96.90 | 97.05 | 97.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-17 10:15:00 | 97.75 | 97.14 | 97.28 | SL hit (close>static) qty=1.00 sl=97.65 alert=retest2 |

### Cycle 11 — BUY (started 2023-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 09:15:00 | 98.05 | 97.39 | 97.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 09:15:00 | 98.50 | 97.74 | 97.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 15:15:00 | 99.60 | 99.76 | 99.36 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 09:15:00 | 100.05 | 99.76 | 99.36 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 99.50 | 99.71 | 99.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 11:00:00 | 99.50 | 99.71 | 99.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 11:15:00 | 98.95 | 99.56 | 99.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-25 11:15:00 | 98.95 | 99.56 | 99.36 | SL hit (close<ema400) qty=1.00 sl=99.36 alert=retest1 |

### Cycle 12 — SELL (started 2023-07-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 13:15:00 | 99.00 | 99.32 | 99.32 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 14:15:00 | 99.60 | 99.37 | 99.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 09:15:00 | 99.75 | 99.50 | 99.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-27 10:15:00 | 99.35 | 99.47 | 99.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 10:15:00 | 99.35 | 99.47 | 99.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 10:15:00 | 99.35 | 99.47 | 99.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 10:30:00 | 99.40 | 99.47 | 99.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 11:15:00 | 99.05 | 99.39 | 99.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 12:00:00 | 99.05 | 99.39 | 99.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2023-07-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 12:15:00 | 98.60 | 99.23 | 99.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-28 09:15:00 | 96.10 | 98.42 | 98.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-01 09:15:00 | 94.70 | 94.41 | 95.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-01 10:00:00 | 94.70 | 94.41 | 95.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 93.45 | 93.55 | 93.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 10:15:00 | 93.25 | 93.55 | 93.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 13:00:00 | 93.25 | 93.47 | 93.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 10:15:00 | 93.30 | 92.86 | 93.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-09 14:15:00 | 93.85 | 92.99 | 92.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2023-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 14:15:00 | 93.85 | 92.99 | 92.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 15:15:00 | 94.00 | 93.19 | 93.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 13:15:00 | 93.80 | 93.87 | 93.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-11 14:00:00 | 93.80 | 93.87 | 93.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 14:15:00 | 93.45 | 93.78 | 93.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 15:00:00 | 93.45 | 93.78 | 93.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 15:15:00 | 93.35 | 93.70 | 93.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 09:15:00 | 91.85 | 93.70 | 93.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 09:15:00 | 92.10 | 93.38 | 93.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 12:15:00 | 91.35 | 91.96 | 92.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 11:15:00 | 91.55 | 91.48 | 91.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-21 11:45:00 | 91.55 | 91.48 | 91.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 12:15:00 | 91.80 | 91.55 | 91.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 13:00:00 | 91.80 | 91.55 | 91.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 14:15:00 | 92.00 | 91.64 | 91.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 15:00:00 | 92.00 | 91.64 | 91.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 15:15:00 | 92.15 | 91.74 | 91.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 09:15:00 | 92.25 | 91.74 | 91.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 10:15:00 | 91.75 | 91.78 | 91.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 09:45:00 | 91.70 | 91.80 | 91.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-24 09:15:00 | 92.10 | 91.61 | 91.70 | SL hit (close>static) qty=1.00 sl=92.00 alert=retest2 |

### Cycle 17 — BUY (started 2023-08-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 10:15:00 | 92.50 | 91.79 | 91.77 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 15:15:00 | 91.75 | 91.95 | 91.96 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 09:15:00 | 92.30 | 92.02 | 91.99 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-28 15:15:00 | 91.90 | 91.98 | 91.98 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 09:15:00 | 92.25 | 92.03 | 92.01 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-30 09:15:00 | 91.20 | 91.88 | 91.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 09:15:00 | 90.15 | 90.94 | 91.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 10:15:00 | 89.75 | 89.69 | 90.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-01 10:45:00 | 90.00 | 89.69 | 90.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 90.10 | 89.46 | 89.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 09:45:00 | 90.20 | 89.46 | 89.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 10:15:00 | 90.45 | 89.66 | 89.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 11:00:00 | 90.45 | 89.66 | 89.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 12:15:00 | 90.15 | 89.79 | 89.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 12:30:00 | 90.15 | 89.79 | 89.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 13:15:00 | 90.00 | 89.84 | 89.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 13:30:00 | 90.15 | 89.84 | 89.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 14:15:00 | 90.05 | 89.88 | 89.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 14:45:00 | 90.05 | 89.88 | 89.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 15:15:00 | 90.20 | 89.94 | 89.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-05 09:15:00 | 90.90 | 89.94 | 89.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2023-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 09:15:00 | 91.25 | 90.20 | 90.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 10:15:00 | 91.30 | 90.42 | 90.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 11:15:00 | 90.85 | 90.98 | 90.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-06 11:15:00 | 90.85 | 90.98 | 90.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 11:15:00 | 90.85 | 90.98 | 90.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 11:30:00 | 90.65 | 90.98 | 90.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 12:15:00 | 91.30 | 91.05 | 90.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 12:45:00 | 91.00 | 91.05 | 90.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 93.15 | 94.20 | 93.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 92.90 | 94.20 | 93.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 93.25 | 94.01 | 93.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:30:00 | 93.30 | 94.01 | 93.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 11:15:00 | 92.65 | 93.74 | 93.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 12:00:00 | 92.65 | 93.74 | 93.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 12:15:00 | 92.00 | 93.39 | 93.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 12:30:00 | 91.45 | 93.39 | 93.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2023-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 14:15:00 | 91.55 | 92.82 | 92.98 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-09-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 15:15:00 | 93.15 | 92.90 | 92.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 09:15:00 | 93.55 | 93.03 | 92.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 09:15:00 | 93.05 | 94.00 | 93.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 09:15:00 | 93.05 | 94.00 | 93.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 09:15:00 | 93.05 | 94.00 | 93.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-15 09:30:00 | 93.05 | 94.00 | 93.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 10:15:00 | 92.35 | 93.67 | 93.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-15 11:00:00 | 92.35 | 93.67 | 93.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2023-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 11:15:00 | 92.25 | 93.39 | 93.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 10:15:00 | 91.70 | 92.36 | 92.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-20 11:15:00 | 92.40 | 92.37 | 92.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-20 12:00:00 | 92.40 | 92.37 | 92.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 12:15:00 | 92.30 | 92.36 | 92.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-20 13:30:00 | 91.95 | 92.23 | 92.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-21 10:15:00 | 93.10 | 92.39 | 92.50 | SL hit (close>static) qty=1.00 sl=92.70 alert=retest2 |

### Cycle 27 — BUY (started 2023-09-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-21 15:15:00 | 92.85 | 92.55 | 92.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-22 09:15:00 | 92.95 | 92.63 | 92.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-22 13:15:00 | 92.70 | 92.79 | 92.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 13:15:00 | 92.70 | 92.79 | 92.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 13:15:00 | 92.70 | 92.79 | 92.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-22 14:00:00 | 92.70 | 92.79 | 92.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 14:15:00 | 92.40 | 92.71 | 92.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-22 15:00:00 | 92.40 | 92.71 | 92.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2023-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 15:15:00 | 92.15 | 92.60 | 92.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-25 13:15:00 | 91.90 | 92.25 | 92.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 09:15:00 | 92.40 | 92.20 | 92.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 09:15:00 | 92.40 | 92.20 | 92.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 92.40 | 92.20 | 92.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:45:00 | 92.40 | 92.20 | 92.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 92.75 | 92.31 | 92.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 10:30:00 | 92.75 | 92.31 | 92.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 11:15:00 | 92.25 | 92.30 | 92.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 09:15:00 | 91.50 | 92.19 | 92.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-11 10:15:00 | 89.95 | 88.80 | 88.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2023-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 10:15:00 | 89.95 | 88.80 | 88.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 10:15:00 | 90.35 | 89.77 | 89.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 15:15:00 | 90.70 | 90.72 | 90.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-16 09:15:00 | 90.55 | 90.72 | 90.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 90.60 | 90.69 | 90.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 09:30:00 | 90.50 | 90.69 | 90.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 91.25 | 91.56 | 91.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 11:00:00 | 91.25 | 91.56 | 91.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 11:15:00 | 91.00 | 91.45 | 91.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 12:00:00 | 91.00 | 91.45 | 91.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 12:15:00 | 91.00 | 91.36 | 91.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 12:30:00 | 90.95 | 91.36 | 91.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 90.65 | 91.17 | 91.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 15:00:00 | 90.65 | 91.17 | 91.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2023-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 15:15:00 | 90.75 | 91.08 | 91.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 10:15:00 | 90.30 | 90.85 | 90.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 09:15:00 | 90.80 | 90.67 | 90.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-20 10:00:00 | 90.80 | 90.67 | 90.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 10:15:00 | 90.20 | 90.58 | 90.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 11:30:00 | 89.80 | 90.43 | 90.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 12:30:00 | 89.75 | 90.28 | 90.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 09:15:00 | 89.85 | 90.21 | 90.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-30 12:15:00 | 87.90 | 87.42 | 87.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2023-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 12:15:00 | 87.90 | 87.42 | 87.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 13:15:00 | 88.45 | 87.63 | 87.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 11:15:00 | 87.75 | 87.97 | 87.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 11:15:00 | 87.75 | 87.97 | 87.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 11:15:00 | 87.75 | 87.97 | 87.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 11:45:00 | 87.90 | 87.97 | 87.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 12:15:00 | 87.85 | 87.94 | 87.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 12:45:00 | 87.70 | 87.94 | 87.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 13:15:00 | 89.30 | 88.21 | 87.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 13:30:00 | 87.90 | 88.21 | 87.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 98.30 | 102.20 | 102.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-10 10:15:00 | 98.60 | 102.20 | 102.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-10 10:15:00 | 98.20 | 101.40 | 101.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2023-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 10:15:00 | 98.20 | 101.40 | 101.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 14:15:00 | 97.90 | 99.48 | 100.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-13 14:15:00 | 99.05 | 98.57 | 99.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-13 15:00:00 | 99.05 | 98.57 | 99.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 99.45 | 98.79 | 99.35 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-11-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 12:15:00 | 100.85 | 99.76 | 99.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 13:15:00 | 101.55 | 100.12 | 99.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 11:15:00 | 103.45 | 103.63 | 102.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-17 11:30:00 | 103.60 | 103.63 | 102.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 12:15:00 | 102.60 | 103.42 | 102.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 12:45:00 | 102.30 | 103.42 | 102.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 13:15:00 | 102.65 | 103.27 | 102.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 13:45:00 | 102.40 | 103.27 | 102.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 14:15:00 | 102.60 | 103.13 | 102.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-17 15:15:00 | 103.20 | 103.13 | 102.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-20 14:15:00 | 102.80 | 103.11 | 102.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-21 09:15:00 | 101.25 | 102.68 | 102.67 | SL hit (close<static) qty=1.00 sl=101.50 alert=retest2 |

### Cycle 34 — SELL (started 2023-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 10:15:00 | 101.70 | 102.48 | 102.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-21 14:15:00 | 101.05 | 101.95 | 102.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-22 09:15:00 | 102.10 | 101.84 | 102.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 09:15:00 | 102.10 | 101.84 | 102.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 102.10 | 101.84 | 102.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-22 09:45:00 | 102.20 | 101.84 | 102.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 10:15:00 | 101.85 | 101.84 | 102.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 12:00:00 | 101.45 | 101.76 | 102.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 12:00:00 | 101.50 | 101.27 | 101.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-23 14:15:00 | 103.90 | 101.93 | 101.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2023-11-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 14:15:00 | 103.90 | 101.93 | 101.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 11:15:00 | 104.60 | 103.22 | 102.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 15:15:00 | 103.55 | 103.56 | 102.95 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 09:15:00 | 104.55 | 103.56 | 102.95 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-30 14:15:00 | 109.78 | 109.16 | 107.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2023-12-04 09:15:00 | 115.01 | 111.88 | 110.30 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 36 — SELL (started 2023-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 13:15:00 | 117.55 | 118.93 | 119.05 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 14:15:00 | 119.60 | 118.63 | 118.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 09:15:00 | 119.95 | 119.06 | 118.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 15:15:00 | 124.70 | 124.78 | 123.84 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 09:15:00 | 125.40 | 124.78 | 123.84 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 121.40 | 124.25 | 124.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-12-20 13:15:00 | 121.40 | 124.25 | 124.00 | SL hit (close<ema400) qty=1.00 sl=124.00 alert=retest1 |

### Cycle 38 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 119.90 | 123.38 | 123.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 119.85 | 122.68 | 123.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 12:15:00 | 123.00 | 122.53 | 122.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 12:15:00 | 123.00 | 122.53 | 122.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 12:15:00 | 123.00 | 122.53 | 122.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 12:45:00 | 123.20 | 122.53 | 122.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 13:15:00 | 122.80 | 122.58 | 122.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 13:30:00 | 123.15 | 122.58 | 122.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 14:15:00 | 124.20 | 122.91 | 123.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 15:00:00 | 124.20 | 122.91 | 123.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 15:15:00 | 124.35 | 123.20 | 123.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:15:00 | 123.95 | 123.20 | 123.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2023-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 09:15:00 | 124.35 | 123.43 | 123.31 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-22 13:15:00 | 122.95 | 123.24 | 123.26 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 14:15:00 | 123.75 | 123.34 | 123.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 09:15:00 | 124.95 | 123.70 | 123.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 11:15:00 | 126.70 | 126.83 | 125.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 12:00:00 | 126.70 | 126.83 | 125.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 126.40 | 126.74 | 125.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 12:30:00 | 126.25 | 126.74 | 125.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 130.40 | 130.68 | 128.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 09:45:00 | 132.20 | 130.41 | 129.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 12:00:00 | 132.15 | 131.08 | 130.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 12:30:00 | 133.40 | 131.46 | 130.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-05 09:15:00 | 133.80 | 131.32 | 131.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 14:15:00 | 132.70 | 132.40 | 131.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-08 09:15:00 | 135.65 | 132.46 | 131.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-08 13:45:00 | 133.20 | 133.10 | 132.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 09:15:00 | 133.30 | 132.99 | 132.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 10:00:00 | 133.10 | 133.02 | 132.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 14:15:00 | 131.75 | 132.85 | 132.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 14:45:00 | 132.10 | 132.85 | 132.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 15:15:00 | 132.30 | 132.74 | 132.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 09:15:00 | 128.75 | 132.74 | 132.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-01-10 09:15:00 | 130.15 | 132.22 | 132.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 09:15:00 | 130.15 | 132.22 | 132.45 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-01-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 11:15:00 | 132.80 | 131.85 | 131.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-12 09:15:00 | 133.40 | 132.75 | 132.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 143.25 | 143.76 | 141.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 09:15:00 | 143.25 | 143.76 | 141.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 143.25 | 143.76 | 141.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 09:45:00 | 140.70 | 143.76 | 141.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 144.95 | 147.59 | 146.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:00:00 | 144.95 | 147.59 | 146.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 143.85 | 146.84 | 146.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:30:00 | 143.10 | 146.84 | 146.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 11:15:00 | 141.40 | 145.75 | 145.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 12:15:00 | 141.05 | 144.81 | 145.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 14:15:00 | 143.15 | 141.37 | 142.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 14:15:00 | 143.15 | 141.37 | 142.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 143.15 | 141.37 | 142.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:45:00 | 143.70 | 141.37 | 142.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 144.05 | 141.91 | 142.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:15:00 | 144.15 | 141.91 | 142.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 143.00 | 142.52 | 142.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 13:30:00 | 142.55 | 143.02 | 143.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-25 14:15:00 | 143.35 | 143.09 | 143.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-01-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 14:15:00 | 143.35 | 143.09 | 143.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 09:15:00 | 145.55 | 143.58 | 143.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 147.70 | 148.59 | 146.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-30 14:45:00 | 148.10 | 148.59 | 146.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 12:15:00 | 147.05 | 148.03 | 147.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 12:45:00 | 146.95 | 148.03 | 147.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 13:15:00 | 146.35 | 147.69 | 147.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 13:45:00 | 146.60 | 147.69 | 147.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 148.70 | 148.11 | 147.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:00:00 | 148.70 | 148.11 | 147.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 183.75 | 189.27 | 185.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 10:00:00 | 183.75 | 189.27 | 185.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 180.10 | 187.44 | 184.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 11:00:00 | 180.10 | 187.44 | 184.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 13:15:00 | 183.30 | 185.24 | 184.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 13:30:00 | 182.40 | 185.24 | 184.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2024-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 09:15:00 | 177.05 | 182.74 | 183.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 10:15:00 | 175.40 | 181.27 | 182.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 10:15:00 | 177.35 | 177.22 | 179.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 11:00:00 | 177.35 | 177.22 | 179.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 11:15:00 | 180.30 | 176.51 | 177.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 12:00:00 | 180.30 | 176.51 | 177.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 12:15:00 | 180.35 | 177.27 | 177.80 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2024-02-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 13:15:00 | 182.10 | 178.24 | 178.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 14:15:00 | 183.60 | 179.31 | 178.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 13:15:00 | 187.55 | 188.51 | 186.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-16 14:00:00 | 187.55 | 188.51 | 186.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 15:15:00 | 186.75 | 187.93 | 186.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 09:45:00 | 187.45 | 187.97 | 186.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-20 09:15:00 | 185.70 | 188.13 | 187.39 | SL hit (close<static) qty=1.00 sl=186.15 alert=retest2 |

### Cycle 48 — SELL (started 2024-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 12:15:00 | 186.15 | 187.22 | 187.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 13:15:00 | 184.80 | 186.73 | 187.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-26 13:15:00 | 176.00 | 175.10 | 177.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-26 13:45:00 | 175.45 | 175.10 | 177.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 14:15:00 | 177.65 | 175.61 | 177.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 14:30:00 | 177.75 | 175.61 | 177.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 15:15:00 | 177.40 | 175.97 | 177.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 09:15:00 | 177.95 | 175.97 | 177.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 176.20 | 176.01 | 177.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 10:15:00 | 175.75 | 176.01 | 177.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 11:30:00 | 175.40 | 175.67 | 176.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 15:15:00 | 166.96 | 169.72 | 172.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-29 09:15:00 | 166.63 | 169.01 | 171.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-29 15:15:00 | 167.05 | 166.88 | 169.32 | SL hit (close>ema200) qty=0.50 sl=166.88 alert=retest2 |

### Cycle 49 — BUY (started 2024-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 11:15:00 | 172.20 | 170.05 | 169.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-04 09:15:00 | 173.60 | 171.13 | 170.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 172.95 | 174.79 | 173.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 09:15:00 | 172.95 | 174.79 | 173.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 172.95 | 174.79 | 173.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 172.95 | 174.79 | 173.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 174.85 | 174.81 | 173.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 14:00:00 | 175.90 | 174.92 | 174.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-07 10:45:00 | 176.05 | 175.96 | 175.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-11 09:15:00 | 176.05 | 175.14 | 174.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-11 12:30:00 | 175.85 | 175.40 | 175.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 13:15:00 | 174.90 | 175.30 | 175.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 14:15:00 | 174.90 | 175.30 | 175.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 14:15:00 | 174.00 | 175.04 | 175.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 15:00:00 | 174.00 | 175.04 | 175.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-03-11 15:15:00 | 174.00 | 174.83 | 174.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2024-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 15:15:00 | 174.00 | 174.83 | 174.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 11:15:00 | 173.35 | 174.37 | 174.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-12 13:15:00 | 174.50 | 174.07 | 174.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 13:15:00 | 174.50 | 174.07 | 174.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 13:15:00 | 174.50 | 174.07 | 174.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-12 14:00:00 | 174.50 | 174.07 | 174.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 14:15:00 | 172.80 | 173.81 | 174.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 09:15:00 | 171.20 | 173.63 | 174.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 14:15:00 | 162.64 | 167.79 | 170.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-14 10:15:00 | 168.15 | 167.06 | 169.55 | SL hit (close>ema200) qty=0.50 sl=167.06 alert=retest2 |

### Cycle 51 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 162.30 | 159.28 | 159.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 14:15:00 | 164.60 | 161.47 | 160.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 13:15:00 | 168.30 | 168.53 | 166.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-27 13:30:00 | 168.60 | 168.53 | 166.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 166.50 | 168.12 | 166.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 15:00:00 | 166.50 | 168.12 | 166.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 166.70 | 167.84 | 166.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 09:15:00 | 167.55 | 167.84 | 166.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-04 13:15:00 | 169.60 | 171.40 | 171.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2024-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 13:15:00 | 169.60 | 171.40 | 171.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-05 09:15:00 | 165.85 | 169.83 | 170.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-08 09:15:00 | 170.95 | 168.84 | 169.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-08 09:15:00 | 170.95 | 168.84 | 169.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 170.95 | 168.84 | 169.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-08 09:30:00 | 170.95 | 168.84 | 169.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 10:15:00 | 170.45 | 169.16 | 169.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-08 10:30:00 | 171.10 | 169.16 | 169.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 12:15:00 | 170.60 | 169.58 | 169.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-08 12:45:00 | 170.60 | 169.58 | 169.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 14:15:00 | 170.25 | 169.82 | 169.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-08 14:30:00 | 170.65 | 169.82 | 169.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2024-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 15:15:00 | 170.25 | 169.90 | 169.90 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 10:15:00 | 169.20 | 169.80 | 169.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 13:15:00 | 168.10 | 169.36 | 169.63 | Break + close below crossover candle low |

### Cycle 55 — BUY (started 2024-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 09:15:00 | 172.40 | 169.81 | 169.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 10:15:00 | 173.85 | 170.62 | 170.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 12:15:00 | 170.30 | 172.07 | 171.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 12:15:00 | 170.30 | 172.07 | 171.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 12:15:00 | 170.30 | 172.07 | 171.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 13:00:00 | 170.30 | 172.07 | 171.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 13:15:00 | 170.05 | 171.67 | 171.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 13:30:00 | 170.10 | 171.67 | 171.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2024-04-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 15:15:00 | 170.30 | 171.16 | 171.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 09:15:00 | 166.80 | 170.29 | 170.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 170.50 | 168.54 | 169.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 170.50 | 168.54 | 169.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 170.50 | 168.54 | 169.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:00:00 | 170.50 | 168.54 | 169.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 170.50 | 168.93 | 169.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:00:00 | 170.50 | 168.93 | 169.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 168.90 | 168.92 | 169.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:30:00 | 170.30 | 168.92 | 169.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 13:15:00 | 169.00 | 168.83 | 169.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 14:00:00 | 169.00 | 168.83 | 169.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 15:15:00 | 169.00 | 168.88 | 169.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:15:00 | 173.85 | 168.88 | 169.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2024-04-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 09:15:00 | 173.20 | 169.75 | 169.61 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-04-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 09:15:00 | 164.75 | 169.27 | 169.69 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 10:15:00 | 172.10 | 168.87 | 168.82 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 14:15:00 | 168.10 | 169.39 | 169.48 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 09:15:00 | 170.60 | 169.37 | 169.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 09:15:00 | 171.75 | 170.35 | 169.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 14:15:00 | 168.85 | 175.05 | 174.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 14:15:00 | 168.85 | 175.05 | 174.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 168.85 | 175.05 | 174.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 14:30:00 | 169.95 | 175.05 | 174.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2024-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 15:15:00 | 168.95 | 173.83 | 173.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 09:15:00 | 166.00 | 170.20 | 171.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 163.00 | 162.38 | 165.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 09:45:00 | 163.10 | 162.38 | 165.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 157.00 | 158.41 | 159.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 10:30:00 | 155.10 | 158.01 | 159.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-15 09:15:00 | 160.95 | 159.33 | 159.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 160.95 | 159.33 | 159.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 163.05 | 161.38 | 160.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 161.20 | 161.78 | 160.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 14:00:00 | 161.20 | 161.78 | 160.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 165.70 | 166.21 | 164.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 165.65 | 166.21 | 164.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 165.40 | 165.96 | 165.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:00:00 | 165.40 | 165.96 | 165.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 165.30 | 165.83 | 165.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 13:15:00 | 165.75 | 165.83 | 165.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 13:45:00 | 165.75 | 165.78 | 165.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 14:15:00 | 165.80 | 165.78 | 165.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 12:15:00 | 167.05 | 168.10 | 168.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 167.05 | 168.10 | 168.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 10:15:00 | 165.55 | 166.88 | 167.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 163.05 | 162.66 | 163.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 13:45:00 | 162.75 | 162.66 | 163.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 65 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 173.35 | 164.73 | 164.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 174.65 | 169.20 | 166.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 170.00 | 171.74 | 169.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 170.00 | 171.74 | 169.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 170.00 | 171.74 | 169.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 166.95 | 171.74 | 169.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 160.80 | 169.55 | 168.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 160.80 | 169.55 | 168.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 157.80 | 167.20 | 167.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 13:15:00 | 156.80 | 163.65 | 165.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 158.20 | 157.52 | 160.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:00:00 | 158.20 | 157.52 | 160.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 165.15 | 159.53 | 160.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 164.80 | 159.53 | 160.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 164.55 | 161.87 | 161.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 13:15:00 | 168.80 | 166.30 | 165.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 10:15:00 | 168.70 | 168.82 | 167.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 11:00:00 | 168.70 | 168.82 | 167.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 169.87 | 169.97 | 169.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 15:15:00 | 170.00 | 169.68 | 169.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 167.73 | 169.34 | 169.30 | SL hit (close<static) qty=1.00 sl=169.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 167.60 | 168.99 | 169.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 15:15:00 | 166.67 | 167.68 | 168.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 168.42 | 167.83 | 168.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 168.42 | 167.83 | 168.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 168.42 | 167.83 | 168.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:45:00 | 168.28 | 167.83 | 168.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 169.15 | 168.09 | 168.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:45:00 | 169.36 | 168.09 | 168.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 169.14 | 168.30 | 168.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:45:00 | 169.04 | 168.30 | 168.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 168.98 | 168.54 | 168.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 15:00:00 | 168.98 | 168.54 | 168.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 15:15:00 | 169.05 | 168.64 | 168.62 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 10:15:00 | 168.16 | 168.56 | 168.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 167.00 | 168.16 | 168.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 15:15:00 | 163.90 | 163.68 | 164.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 09:15:00 | 165.65 | 163.68 | 164.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 166.35 | 164.21 | 164.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:30:00 | 166.09 | 164.21 | 164.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 166.00 | 164.57 | 164.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:30:00 | 166.40 | 164.57 | 164.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2024-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 11:15:00 | 165.74 | 164.80 | 164.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 166.44 | 165.50 | 165.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 167.95 | 168.03 | 167.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 12:15:00 | 167.95 | 168.03 | 167.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 167.95 | 168.03 | 167.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 167.95 | 168.03 | 167.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 13:15:00 | 170.20 | 170.92 | 170.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 13:45:00 | 170.00 | 170.92 | 170.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 170.00 | 170.74 | 170.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 15:00:00 | 170.00 | 170.74 | 170.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 170.36 | 170.66 | 170.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 09:15:00 | 171.31 | 170.66 | 170.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 169.80 | 170.85 | 170.80 | SL hit (close<static) qty=1.00 sl=169.81 alert=retest2 |

### Cycle 72 — SELL (started 2024-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 12:15:00 | 170.22 | 170.65 | 170.71 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 14:15:00 | 171.92 | 170.87 | 170.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 15:15:00 | 172.15 | 171.12 | 170.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 09:15:00 | 168.63 | 172.67 | 172.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 09:15:00 | 168.63 | 172.67 | 172.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 168.63 | 172.67 | 172.20 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 10:15:00 | 168.05 | 171.75 | 171.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 11:15:00 | 167.33 | 170.86 | 171.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 09:15:00 | 169.07 | 168.75 | 169.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-15 10:00:00 | 169.07 | 168.75 | 169.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 169.71 | 169.00 | 169.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 13:45:00 | 169.61 | 169.00 | 169.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 169.67 | 169.13 | 169.70 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 10:15:00 | 171.47 | 170.13 | 170.05 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 11:15:00 | 169.31 | 170.18 | 170.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 166.46 | 169.12 | 169.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 167.24 | 166.90 | 168.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 167.24 | 166.90 | 168.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 167.45 | 167.01 | 167.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 168.02 | 167.01 | 167.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 168.05 | 167.22 | 167.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:45:00 | 168.05 | 167.22 | 167.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 167.69 | 167.31 | 167.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:15:00 | 168.31 | 167.31 | 167.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 168.14 | 167.48 | 167.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:30:00 | 166.95 | 167.45 | 167.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 11:00:00 | 167.56 | 166.76 | 167.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 12:30:00 | 167.46 | 167.22 | 167.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 13:15:00 | 168.52 | 167.48 | 167.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 13:15:00 | 168.52 | 167.48 | 167.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 10:15:00 | 172.70 | 168.83 | 168.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 14:15:00 | 181.20 | 182.31 | 179.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 15:00:00 | 181.20 | 182.31 | 179.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 180.57 | 181.93 | 180.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:00:00 | 180.57 | 181.93 | 180.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 181.90 | 181.92 | 180.83 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 13:15:00 | 180.02 | 180.43 | 180.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 15:15:00 | 179.50 | 180.17 | 180.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 13:15:00 | 178.75 | 178.51 | 179.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 13:15:00 | 178.75 | 178.51 | 179.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 13:15:00 | 178.75 | 178.51 | 179.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 14:00:00 | 178.75 | 178.51 | 179.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 171.59 | 170.27 | 172.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:30:00 | 172.21 | 170.27 | 172.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 172.45 | 170.97 | 172.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:00:00 | 172.45 | 170.97 | 172.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 172.15 | 171.20 | 172.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:15:00 | 172.15 | 171.20 | 172.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 172.10 | 171.38 | 172.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:15:00 | 172.30 | 171.38 | 172.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 172.30 | 171.57 | 172.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 171.50 | 171.57 | 172.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 09:15:00 | 170.11 | 166.95 | 166.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 170.11 | 166.95 | 166.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 171.83 | 169.71 | 168.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 14:15:00 | 173.89 | 173.96 | 172.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 14:30:00 | 173.84 | 173.96 | 172.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 172.86 | 174.10 | 173.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 172.86 | 174.10 | 173.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 173.22 | 173.92 | 173.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 173.95 | 173.92 | 173.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 10:00:00 | 173.45 | 173.83 | 173.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 12:00:00 | 173.38 | 173.57 | 173.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 14:30:00 | 173.40 | 173.40 | 173.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 09:15:00 | 172.09 | 173.16 | 173.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 09:15:00 | 172.09 | 173.16 | 173.25 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-08-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 11:15:00 | 174.09 | 173.45 | 173.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 09:15:00 | 175.43 | 173.82 | 173.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 13:15:00 | 174.29 | 174.30 | 173.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 13:45:00 | 174.19 | 174.30 | 173.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 173.71 | 174.18 | 173.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 15:00:00 | 173.71 | 174.18 | 173.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 173.78 | 174.10 | 173.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 173.97 | 174.10 | 173.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 173.66 | 173.91 | 173.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 11:15:00 | 173.78 | 173.91 | 173.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 11:45:00 | 174.09 | 173.91 | 173.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 14:15:00 | 176.09 | 177.00 | 177.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 14:15:00 | 176.09 | 177.00 | 177.03 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 09:15:00 | 179.02 | 177.24 | 177.13 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 13:15:00 | 176.40 | 176.94 | 177.02 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 09:15:00 | 180.28 | 177.68 | 177.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 10:15:00 | 183.70 | 178.89 | 177.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 15:15:00 | 180.74 | 180.88 | 179.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 09:15:00 | 180.25 | 180.88 | 179.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 176.68 | 180.04 | 179.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 176.68 | 180.04 | 179.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 175.15 | 179.06 | 178.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:30:00 | 176.19 | 179.06 | 178.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2024-09-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 11:15:00 | 176.51 | 178.55 | 178.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 12:15:00 | 174.50 | 176.09 | 177.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 12:15:00 | 175.63 | 175.38 | 176.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 13:00:00 | 175.63 | 175.38 | 176.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 176.06 | 175.52 | 176.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 13:45:00 | 175.93 | 175.52 | 176.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 175.46 | 175.51 | 176.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:45:00 | 174.97 | 175.21 | 175.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 10:45:00 | 174.90 | 173.47 | 173.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 11:15:00 | 174.28 | 173.63 | 173.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2024-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 11:15:00 | 174.28 | 173.63 | 173.57 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 14:15:00 | 173.06 | 173.48 | 173.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 10:15:00 | 172.57 | 173.20 | 173.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 13:15:00 | 171.25 | 171.24 | 172.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-17 13:45:00 | 171.43 | 171.24 | 172.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 167.14 | 166.14 | 167.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 14:30:00 | 166.90 | 166.14 | 167.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 15:15:00 | 167.05 | 166.32 | 167.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:15:00 | 167.71 | 166.32 | 167.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 169.13 | 166.89 | 167.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 10:00:00 | 169.13 | 166.89 | 167.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 169.86 | 167.48 | 167.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 170.14 | 169.18 | 168.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 14:15:00 | 169.71 | 170.01 | 169.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 15:00:00 | 169.71 | 170.01 | 169.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 168.80 | 169.77 | 169.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 168.80 | 169.77 | 169.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 170.25 | 169.86 | 169.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 11:30:00 | 170.73 | 169.94 | 169.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 13:15:00 | 170.69 | 169.98 | 169.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 13:00:00 | 170.72 | 169.91 | 169.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 14:30:00 | 170.70 | 170.25 | 169.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 178.00 | 178.98 | 177.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:45:00 | 177.65 | 178.98 | 177.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 174.90 | 178.10 | 177.47 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-03 11:15:00 | 172.01 | 176.42 | 176.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 172.01 | 176.42 | 176.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 171.58 | 175.45 | 176.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 13:15:00 | 163.89 | 163.88 | 166.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 14:00:00 | 163.89 | 163.88 | 166.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 166.00 | 164.49 | 165.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:30:00 | 165.20 | 164.86 | 165.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 14:15:00 | 164.93 | 165.04 | 165.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:00:00 | 165.20 | 165.33 | 165.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:30:00 | 164.92 | 165.26 | 165.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 164.78 | 163.90 | 164.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:30:00 | 164.79 | 163.90 | 164.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 164.08 | 163.93 | 164.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 12:15:00 | 163.62 | 163.98 | 164.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 12:45:00 | 163.10 | 163.77 | 164.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 14:15:00 | 165.55 | 164.22 | 164.37 | SL hit (close>static) qty=1.00 sl=165.24 alert=retest2 |

### Cycle 91 — BUY (started 2024-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 15:15:00 | 166.00 | 164.57 | 164.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 09:15:00 | 168.25 | 165.31 | 164.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 166.50 | 167.81 | 167.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 166.50 | 167.81 | 167.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 166.50 | 167.81 | 167.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 166.50 | 167.81 | 167.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 165.59 | 167.37 | 167.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:15:00 | 165.48 | 167.37 | 167.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2024-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 12:15:00 | 165.01 | 166.57 | 166.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 164.19 | 166.09 | 166.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 164.55 | 164.48 | 165.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 13:45:00 | 164.52 | 164.48 | 165.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 165.39 | 164.66 | 165.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 15:00:00 | 165.39 | 164.66 | 165.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 165.25 | 164.78 | 165.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 163.45 | 164.78 | 165.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 162.18 | 164.26 | 165.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:30:00 | 161.50 | 163.19 | 164.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 153.42 | 156.89 | 159.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 148.41 | 148.07 | 150.59 | SL hit (close>ema200) qty=0.50 sl=148.07 alert=retest2 |

### Cycle 93 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 143.80 | 141.14 | 141.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 144.34 | 142.25 | 141.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 09:15:00 | 142.09 | 143.63 | 143.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 09:15:00 | 142.09 | 143.63 | 143.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 142.09 | 143.63 | 143.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:15:00 | 141.94 | 143.63 | 143.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 141.30 | 143.16 | 142.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:00:00 | 141.30 | 143.16 | 142.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 140.68 | 142.66 | 142.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 13:15:00 | 140.10 | 141.84 | 142.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 140.04 | 139.95 | 140.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 140.04 | 139.95 | 140.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 140.04 | 139.95 | 140.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 13:30:00 | 139.56 | 140.02 | 140.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 14:30:00 | 139.35 | 139.75 | 140.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 09:15:00 | 132.58 | 133.55 | 134.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 09:15:00 | 132.38 | 133.55 | 134.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-22 11:15:00 | 132.05 | 131.65 | 132.72 | SL hit (close>ema200) qty=0.50 sl=131.65 alert=retest2 |

### Cycle 95 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 137.06 | 133.25 | 133.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 14:15:00 | 138.85 | 136.89 | 136.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 13:15:00 | 138.29 | 138.35 | 137.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 13:45:00 | 138.33 | 138.35 | 137.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 137.73 | 138.22 | 137.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 137.73 | 138.22 | 137.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 138.04 | 138.19 | 137.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 14:30:00 | 138.96 | 138.14 | 137.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 14:15:00 | 137.49 | 137.64 | 137.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 14:15:00 | 137.49 | 137.64 | 137.65 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 15:15:00 | 137.99 | 137.71 | 137.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 139.00 | 137.97 | 137.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 138.17 | 139.47 | 139.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 09:15:00 | 138.17 | 139.47 | 139.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 138.17 | 139.47 | 139.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:45:00 | 138.39 | 139.47 | 139.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 138.38 | 139.25 | 139.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 12:00:00 | 139.18 | 139.24 | 139.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 12:15:00 | 142.57 | 142.68 | 142.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 12:15:00 | 142.57 | 142.68 | 142.68 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 13:15:00 | 142.80 | 142.70 | 142.69 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 14:15:00 | 141.45 | 142.45 | 142.58 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 13:15:00 | 144.74 | 142.92 | 142.71 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 10:15:00 | 141.82 | 142.73 | 142.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 141.24 | 142.43 | 142.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 10:15:00 | 138.90 | 137.91 | 139.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 11:00:00 | 138.90 | 137.91 | 139.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 139.77 | 138.28 | 139.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 12:00:00 | 139.77 | 138.28 | 139.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 139.63 | 138.55 | 139.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 13:00:00 | 139.63 | 138.55 | 139.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 139.33 | 138.71 | 139.41 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2024-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 11:15:00 | 140.95 | 139.81 | 139.73 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 138.84 | 139.62 | 139.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 13:15:00 | 138.08 | 139.31 | 139.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 10:15:00 | 138.98 | 138.59 | 139.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 10:15:00 | 138.98 | 138.59 | 139.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 138.98 | 138.59 | 139.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:00:00 | 138.98 | 138.59 | 139.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 139.04 | 138.68 | 139.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:45:00 | 139.31 | 138.68 | 139.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 138.14 | 138.57 | 138.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 13:15:00 | 138.01 | 138.57 | 138.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 14:30:00 | 138.05 | 138.31 | 138.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 15:15:00 | 138.00 | 138.44 | 138.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 10:45:00 | 137.79 | 138.30 | 138.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 138.03 | 137.95 | 138.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:30:00 | 137.72 | 137.85 | 138.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 12:15:00 | 136.90 | 136.45 | 136.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 12:15:00 | 136.90 | 136.45 | 136.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 13:15:00 | 137.09 | 136.57 | 136.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 136.43 | 136.68 | 136.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 136.43 | 136.68 | 136.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 136.43 | 136.68 | 136.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:15:00 | 136.47 | 136.68 | 136.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 136.30 | 136.60 | 136.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:00:00 | 136.30 | 136.60 | 136.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 136.56 | 136.59 | 136.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:30:00 | 135.84 | 136.59 | 136.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 137.21 | 136.72 | 136.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 13:30:00 | 137.63 | 136.95 | 136.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 10:45:00 | 137.35 | 137.40 | 137.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 135.20 | 137.50 | 137.36 | SL hit (close<static) qty=1.00 sl=136.43 alert=retest2 |

### Cycle 106 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 133.40 | 136.68 | 137.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 132.85 | 134.96 | 136.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 135.33 | 134.55 | 135.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 135.33 | 134.55 | 135.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 135.33 | 134.55 | 135.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 132.86 | 134.43 | 134.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 11:00:00 | 133.11 | 134.14 | 134.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 11:30:00 | 133.02 | 133.87 | 134.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 126.22 | 130.30 | 131.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 126.45 | 130.30 | 131.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 126.37 | 130.30 | 131.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 124.71 | 124.67 | 127.64 | SL hit (close>ema200) qty=0.50 sl=124.67 alert=retest2 |

### Cycle 107 — BUY (started 2025-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 09:15:00 | 127.70 | 127.26 | 127.24 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 10:15:00 | 126.40 | 127.09 | 127.16 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 12:15:00 | 127.62 | 127.26 | 127.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 13:15:00 | 128.39 | 127.49 | 127.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 14:15:00 | 130.97 | 131.15 | 129.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 14:45:00 | 131.15 | 131.15 | 129.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 129.84 | 130.87 | 130.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 130.10 | 130.87 | 130.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 129.85 | 130.66 | 130.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:15:00 | 129.47 | 130.66 | 130.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 128.81 | 130.29 | 129.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:00:00 | 128.81 | 130.29 | 129.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 130.27 | 130.16 | 129.96 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 15:15:00 | 129.76 | 129.87 | 129.88 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 10:15:00 | 130.19 | 129.89 | 129.88 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 11:15:00 | 129.76 | 129.87 | 129.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 12:15:00 | 129.30 | 129.75 | 129.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 14:15:00 | 123.64 | 123.60 | 124.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 15:00:00 | 123.64 | 123.60 | 124.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 125.88 | 124.11 | 124.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 10:00:00 | 125.88 | 124.11 | 124.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 127.08 | 124.71 | 124.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:00:00 | 127.08 | 124.71 | 124.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 127.02 | 125.17 | 125.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 12:15:00 | 127.51 | 125.64 | 125.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 127.45 | 128.05 | 127.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 127.45 | 128.05 | 127.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 127.45 | 128.05 | 127.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 127.46 | 128.05 | 127.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 125.39 | 127.52 | 127.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:45:00 | 125.00 | 127.52 | 127.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 125.64 | 127.14 | 127.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:30:00 | 125.19 | 127.14 | 127.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 125.48 | 126.81 | 126.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 121.63 | 125.56 | 126.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 124.12 | 122.65 | 124.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 124.12 | 122.65 | 124.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 124.12 | 122.65 | 124.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 124.12 | 122.65 | 124.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 123.50 | 122.82 | 124.00 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 127.44 | 124.54 | 124.40 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 124.93 | 125.70 | 125.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 123.60 | 125.11 | 125.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 120.62 | 120.50 | 121.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:00:00 | 120.62 | 120.50 | 121.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 121.54 | 120.75 | 121.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:30:00 | 121.87 | 120.75 | 121.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 121.30 | 120.86 | 121.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:00:00 | 120.94 | 120.91 | 121.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 114.89 | 117.48 | 118.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 13:15:00 | 117.35 | 117.05 | 118.23 | SL hit (close>ema200) qty=0.50 sl=117.05 alert=retest2 |

### Cycle 117 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 119.70 | 118.43 | 118.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 121.02 | 119.71 | 119.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 121.60 | 121.79 | 120.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 121.60 | 121.79 | 120.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 121.52 | 121.60 | 120.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:30:00 | 121.03 | 121.60 | 120.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 120.68 | 121.42 | 121.02 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 120.14 | 120.77 | 120.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 119.84 | 120.37 | 120.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 114.21 | 113.53 | 115.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 114.21 | 113.53 | 115.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 115.70 | 114.33 | 115.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 115.70 | 114.33 | 115.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 117.60 | 114.98 | 115.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 117.60 | 114.98 | 115.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 12:15:00 | 116.83 | 115.68 | 115.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 13:15:00 | 118.16 | 116.18 | 115.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 124.33 | 124.89 | 122.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 124.33 | 124.89 | 122.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 123.48 | 124.44 | 123.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 123.48 | 124.44 | 123.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 122.69 | 124.09 | 123.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 122.69 | 124.09 | 123.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 122.59 | 123.79 | 123.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:45:00 | 122.68 | 123.79 | 123.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 121.64 | 123.06 | 123.09 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 10:15:00 | 124.08 | 123.17 | 123.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 15:15:00 | 125.05 | 123.96 | 123.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-17 09:15:00 | 124.25 | 125.32 | 124.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 124.25 | 125.32 | 124.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 124.25 | 125.32 | 124.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 124.09 | 125.32 | 124.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 123.80 | 125.01 | 124.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:30:00 | 123.48 | 125.01 | 124.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 12:15:00 | 123.59 | 124.60 | 124.69 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 14:15:00 | 125.09 | 124.70 | 124.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 15:15:00 | 125.35 | 124.83 | 124.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 131.52 | 132.74 | 131.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:00:00 | 131.52 | 132.74 | 131.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 131.25 | 132.44 | 131.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 131.26 | 132.44 | 131.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 131.13 | 132.18 | 131.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 130.55 | 132.18 | 131.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 132.03 | 132.15 | 131.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 13:30:00 | 132.33 | 132.20 | 131.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 133.05 | 131.74 | 131.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:30:00 | 132.43 | 131.87 | 131.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 13:15:00 | 129.93 | 131.18 | 131.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 13:15:00 | 129.93 | 131.18 | 131.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 129.11 | 130.77 | 131.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 130.35 | 129.73 | 130.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 130.35 | 129.73 | 130.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 130.35 | 129.73 | 130.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 130.35 | 129.73 | 130.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 130.31 | 129.85 | 130.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 130.69 | 129.85 | 130.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 130.09 | 130.01 | 130.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:15:00 | 129.90 | 130.01 | 130.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:00:00 | 129.86 | 128.96 | 129.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 11:15:00 | 130.80 | 129.40 | 129.63 | SL hit (close>static) qty=1.00 sl=130.70 alert=retest2 |

### Cycle 125 — BUY (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 13:15:00 | 131.04 | 129.96 | 129.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 15:15:00 | 131.45 | 130.45 | 130.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 09:15:00 | 129.30 | 130.22 | 130.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 129.30 | 130.22 | 130.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 129.30 | 130.22 | 130.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 14:15:00 | 130.94 | 130.32 | 130.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 11:45:00 | 131.20 | 130.96 | 130.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 15:00:00 | 130.99 | 131.00 | 130.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 10:30:00 | 130.98 | 130.85 | 130.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 11:15:00 | 130.01 | 130.68 | 130.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 11:45:00 | 130.27 | 130.68 | 130.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-04 12:15:00 | 129.60 | 130.46 | 130.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 129.60 | 130.46 | 130.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 125.40 | 129.32 | 129.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 13:15:00 | 128.32 | 128.30 | 129.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 13:15:00 | 128.32 | 128.30 | 129.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 13:15:00 | 128.32 | 128.30 | 129.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 13:45:00 | 128.68 | 128.30 | 129.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 129.13 | 128.47 | 129.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 15:00:00 | 129.13 | 128.47 | 129.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 127.99 | 128.37 | 129.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 130.73 | 128.37 | 129.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 128.45 | 128.39 | 129.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:30:00 | 129.36 | 128.39 | 129.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 130.01 | 128.71 | 129.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 11:00:00 | 130.01 | 128.71 | 129.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 130.51 | 129.07 | 129.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:00:00 | 130.51 | 129.07 | 129.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2025-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 13:15:00 | 131.30 | 129.73 | 129.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 14:15:00 | 131.86 | 130.82 | 130.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 13:15:00 | 131.47 | 131.58 | 130.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-11 14:00:00 | 131.47 | 131.58 | 130.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 15:15:00 | 133.19 | 132.75 | 132.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 09:15:00 | 133.64 | 132.75 | 132.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 11:15:00 | 133.55 | 132.90 | 132.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 13:00:00 | 133.32 | 133.05 | 132.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 13:45:00 | 133.33 | 133.12 | 132.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 136.63 | 138.08 | 137.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 136.63 | 138.08 | 137.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 136.76 | 137.82 | 137.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 136.50 | 137.82 | 137.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 136.59 | 137.57 | 137.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:00:00 | 136.59 | 137.57 | 137.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 137.25 | 137.51 | 137.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 13:45:00 | 137.65 | 137.53 | 137.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 14:45:00 | 137.58 | 137.57 | 137.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 135.28 | 137.59 | 137.54 | SL hit (close<static) qty=1.00 sl=136.29 alert=retest2 |

### Cycle 128 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 133.70 | 136.81 | 137.19 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 137.19 | 136.45 | 136.42 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 14:15:00 | 135.73 | 136.41 | 136.44 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 09:15:00 | 138.05 | 136.63 | 136.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 09:15:00 | 143.35 | 138.66 | 137.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 146.55 | 147.08 | 144.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 10:00:00 | 146.55 | 147.08 | 144.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 144.80 | 146.07 | 144.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:00:00 | 144.80 | 146.07 | 144.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 144.12 | 145.68 | 144.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:45:00 | 143.71 | 145.68 | 144.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 143.90 | 145.33 | 144.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 144.09 | 145.33 | 144.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 144.24 | 144.77 | 144.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:15:00 | 144.45 | 144.77 | 144.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 14:30:00 | 144.37 | 144.44 | 144.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 15:00:00 | 144.37 | 144.44 | 144.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 10:15:00 | 143.40 | 144.23 | 144.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 10:15:00 | 143.40 | 144.23 | 144.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 11:15:00 | 142.98 | 143.98 | 144.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 143.03 | 140.85 | 141.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 143.03 | 140.85 | 141.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 143.03 | 140.85 | 141.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 14:15:00 | 142.08 | 141.88 | 141.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-13 09:15:00 | 141.85 | 142.04 | 142.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-13 09:15:00 | 142.16 | 142.06 | 142.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 09:15:00 | 142.16 | 142.06 | 142.05 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 10:15:00 | 141.64 | 141.98 | 142.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 13:15:00 | 141.35 | 141.80 | 141.92 | Break + close below crossover candle low |

### Cycle 135 — BUY (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 09:15:00 | 142.96 | 142.00 | 141.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 144.70 | 143.29 | 142.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 14:15:00 | 144.58 | 145.21 | 144.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 14:15:00 | 144.58 | 145.21 | 144.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 144.58 | 145.21 | 144.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 15:00:00 | 144.58 | 145.21 | 144.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 144.78 | 145.12 | 144.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 09:15:00 | 145.25 | 145.12 | 144.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:00:00 | 145.49 | 145.20 | 144.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 12:45:00 | 145.01 | 145.10 | 144.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 144.42 | 144.96 | 144.68 | SL hit (close<static) qty=1.00 sl=144.43 alert=retest2 |

### Cycle 136 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 143.32 | 144.38 | 144.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 142.90 | 143.95 | 144.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 13:15:00 | 143.09 | 142.79 | 143.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 13:30:00 | 142.81 | 142.79 | 143.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 143.36 | 142.90 | 143.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:00:00 | 143.36 | 142.90 | 143.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 143.40 | 143.00 | 143.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 143.40 | 143.00 | 143.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 143.24 | 143.05 | 143.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:30:00 | 143.73 | 143.05 | 143.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 143.40 | 143.12 | 143.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 143.40 | 143.12 | 143.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 143.00 | 143.10 | 143.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:15:00 | 143.59 | 143.10 | 143.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 143.31 | 143.14 | 143.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:45:00 | 143.55 | 143.14 | 143.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 142.35 | 142.98 | 143.21 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 144.49 | 143.33 | 143.32 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-05-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 15:15:00 | 143.40 | 143.66 | 143.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 142.82 | 143.49 | 143.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 11:15:00 | 143.62 | 143.51 | 143.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 11:15:00 | 143.62 | 143.51 | 143.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 143.62 | 143.51 | 143.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:30:00 | 143.62 | 143.51 | 143.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 143.63 | 143.53 | 143.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:45:00 | 143.74 | 143.53 | 143.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 143.05 | 143.44 | 143.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 142.88 | 143.43 | 143.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 10:15:00 | 142.94 | 143.37 | 143.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 10:15:00 | 143.92 | 143.48 | 143.52 | SL hit (close>static) qty=1.00 sl=143.66 alert=retest2 |

### Cycle 139 — BUY (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 13:15:00 | 143.70 | 143.54 | 143.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 14:15:00 | 144.21 | 143.68 | 143.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 09:15:00 | 143.59 | 143.76 | 143.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 143.59 | 143.76 | 143.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 143.59 | 143.76 | 143.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:30:00 | 143.59 | 143.76 | 143.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 142.96 | 143.60 | 143.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:00:00 | 142.96 | 143.60 | 143.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 11:15:00 | 143.25 | 143.53 | 143.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 09:15:00 | 142.63 | 143.37 | 143.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 12:15:00 | 143.00 | 142.51 | 142.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 12:15:00 | 143.00 | 142.51 | 142.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 143.00 | 142.51 | 142.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:45:00 | 142.89 | 142.51 | 142.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 143.05 | 142.62 | 142.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:30:00 | 143.23 | 142.62 | 142.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 143.25 | 142.87 | 142.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 143.26 | 142.87 | 142.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 142.23 | 142.71 | 142.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 10:30:00 | 142.58 | 142.71 | 142.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 140.40 | 140.36 | 141.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 140.40 | 140.36 | 141.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 141.18 | 140.53 | 141.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 10:45:00 | 140.55 | 140.49 | 141.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 12:15:00 | 140.59 | 140.52 | 141.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 10:30:00 | 140.45 | 140.08 | 140.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 13:00:00 | 140.59 | 140.24 | 140.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 140.62 | 140.32 | 140.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 14:15:00 | 140.87 | 140.32 | 140.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 140.72 | 140.40 | 140.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 14:30:00 | 140.56 | 140.40 | 140.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-09 09:15:00 | 142.21 | 140.82 | 140.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 142.21 | 140.82 | 140.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 12:15:00 | 142.83 | 141.71 | 141.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 12:15:00 | 142.61 | 142.66 | 142.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 14:15:00 | 142.26 | 142.59 | 142.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 142.26 | 142.59 | 142.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:45:00 | 142.04 | 142.59 | 142.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 142.10 | 142.49 | 142.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 145.20 | 142.49 | 142.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 142.02 | 144.27 | 143.61 | SL hit (close<static) qty=1.00 sl=142.10 alert=retest2 |

### Cycle 142 — SELL (started 2025-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 09:15:00 | 140.80 | 142.96 | 143.22 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 11:15:00 | 142.36 | 141.92 | 141.89 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 13:15:00 | 141.63 | 141.83 | 141.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 09:15:00 | 140.60 | 141.55 | 141.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 14:15:00 | 140.83 | 140.83 | 141.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-18 14:30:00 | 140.75 | 140.83 | 141.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 139.89 | 140.63 | 141.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:30:00 | 139.35 | 140.46 | 140.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:30:00 | 139.44 | 140.29 | 140.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:15:00 | 139.40 | 140.29 | 140.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 13:45:00 | 139.47 | 138.51 | 138.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 139.94 | 138.79 | 138.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 143.95 | 139.99 | 139.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 143.95 | 139.99 | 139.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 14:15:00 | 146.49 | 144.25 | 143.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 11:15:00 | 147.00 | 147.17 | 145.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 12:00:00 | 147.00 | 147.17 | 145.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 146.54 | 146.92 | 146.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 146.34 | 146.92 | 146.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 147.61 | 147.51 | 147.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 147.21 | 147.51 | 147.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 147.60 | 147.85 | 147.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 148.75 | 147.85 | 147.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 14:15:00 | 150.35 | 151.73 | 151.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 14:15:00 | 150.35 | 151.73 | 151.82 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 15:15:00 | 152.10 | 151.63 | 151.62 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 151.15 | 151.53 | 151.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 150.89 | 151.40 | 151.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 11:15:00 | 151.45 | 151.41 | 151.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 12:00:00 | 151.45 | 151.41 | 151.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 151.60 | 151.45 | 151.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:30:00 | 151.57 | 151.45 | 151.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 150.60 | 151.28 | 151.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:30:00 | 151.10 | 151.28 | 151.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 151.17 | 151.07 | 151.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:30:00 | 151.15 | 151.07 | 151.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 150.63 | 150.98 | 151.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 150.12 | 150.98 | 151.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 14:15:00 | 151.47 | 150.91 | 150.92 | SL hit (close>static) qty=1.00 sl=151.40 alert=retest2 |

### Cycle 149 — BUY (started 2025-07-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 15:15:00 | 151.45 | 151.02 | 150.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 151.73 | 151.16 | 151.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 151.40 | 151.61 | 151.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 151.40 | 151.61 | 151.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 151.40 | 151.61 | 151.37 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 15:15:00 | 150.80 | 151.29 | 151.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 150.64 | 151.16 | 151.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 10:15:00 | 150.44 | 150.29 | 150.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 10:30:00 | 150.23 | 150.29 | 150.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 150.55 | 150.34 | 150.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 150.55 | 150.34 | 150.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 150.75 | 150.42 | 150.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:45:00 | 150.76 | 150.42 | 150.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 150.82 | 150.50 | 150.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:00:00 | 150.82 | 150.50 | 150.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 151.04 | 150.61 | 150.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 151.04 | 150.61 | 150.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 151.47 | 150.87 | 150.80 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 151.10 | 151.41 | 151.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 149.16 | 150.96 | 151.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 147.76 | 147.50 | 148.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 147.76 | 147.50 | 148.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 148.15 | 147.74 | 148.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 147.37 | 147.74 | 148.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 146.90 | 147.57 | 148.07 | EMA400 retest candle locked (from downside) |

### Cycle 153 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 149.17 | 148.34 | 148.28 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 144.60 | 147.68 | 148.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 144.25 | 146.01 | 146.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 142.16 | 141.87 | 143.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:00:00 | 142.16 | 141.87 | 143.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 139.81 | 141.65 | 142.96 | EMA400 retest candle locked (from downside) |

### Cycle 155 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 143.00 | 141.38 | 141.29 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 140.82 | 141.96 | 142.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 140.11 | 141.23 | 141.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 10:15:00 | 141.09 | 140.56 | 140.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 10:15:00 | 141.09 | 140.56 | 140.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 141.09 | 140.56 | 140.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 141.09 | 140.56 | 140.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 141.19 | 140.69 | 140.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:00:00 | 141.19 | 140.69 | 140.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 141.18 | 140.79 | 140.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:15:00 | 141.35 | 140.79 | 140.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 141.96 | 141.16 | 141.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 11:15:00 | 142.12 | 141.68 | 141.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 14:15:00 | 141.37 | 141.78 | 141.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 14:15:00 | 141.37 | 141.78 | 141.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 141.37 | 141.78 | 141.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 141.37 | 141.78 | 141.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 141.91 | 141.80 | 141.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 142.37 | 141.80 | 141.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 13:15:00 | 141.97 | 142.07 | 141.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 140.54 | 141.62 | 141.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 140.54 | 141.62 | 141.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 140.16 | 140.92 | 141.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 13:15:00 | 140.09 | 139.85 | 140.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 14:00:00 | 140.09 | 139.85 | 140.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 139.25 | 139.76 | 140.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:30:00 | 138.79 | 139.61 | 140.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 15:00:00 | 138.94 | 139.51 | 139.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 138.33 | 139.45 | 139.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:45:00 | 138.95 | 139.21 | 139.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 138.27 | 138.82 | 139.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 137.98 | 138.72 | 139.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:45:00 | 137.93 | 137.79 | 138.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 139.61 | 138.37 | 138.47 | SL hit (close>static) qty=1.00 sl=139.49 alert=retest2 |

### Cycle 159 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 139.56 | 138.61 | 138.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 140.22 | 138.93 | 138.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 13:15:00 | 140.10 | 140.20 | 139.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 14:00:00 | 140.10 | 140.20 | 139.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 140.41 | 140.30 | 139.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:15:00 | 139.97 | 140.30 | 139.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 139.90 | 140.22 | 139.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 139.90 | 140.22 | 139.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 139.77 | 140.13 | 139.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 139.77 | 140.13 | 139.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 139.63 | 140.03 | 139.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:15:00 | 139.84 | 140.03 | 139.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 139.32 | 139.86 | 139.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 139.32 | 139.86 | 139.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 139.62 | 139.81 | 139.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 138.84 | 139.62 | 139.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 140.04 | 139.61 | 139.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 140.04 | 139.61 | 139.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 140.04 | 139.61 | 139.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 140.04 | 139.61 | 139.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 139.65 | 139.62 | 139.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:15:00 | 140.00 | 139.62 | 139.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 140.00 | 139.69 | 139.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 141.03 | 139.69 | 139.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 141.54 | 140.06 | 139.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 142.94 | 141.50 | 141.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 142.96 | 143.54 | 142.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 13:00:00 | 142.96 | 143.54 | 142.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 142.95 | 143.42 | 142.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:45:00 | 142.85 | 143.42 | 142.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 142.63 | 143.27 | 142.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:45:00 | 142.60 | 143.27 | 142.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 142.81 | 143.17 | 142.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 142.84 | 143.17 | 142.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 143.08 | 143.16 | 142.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 15:00:00 | 144.10 | 143.23 | 142.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 147.28 | 147.84 | 147.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 13:15:00 | 147.28 | 147.84 | 147.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 146.80 | 147.64 | 147.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 147.70 | 147.51 | 147.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 147.70 | 147.51 | 147.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 147.70 | 147.51 | 147.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:00:00 | 147.70 | 147.51 | 147.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 147.81 | 147.57 | 147.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:30:00 | 147.62 | 147.57 | 147.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 147.45 | 147.54 | 147.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:15:00 | 147.71 | 147.54 | 147.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 147.35 | 147.50 | 147.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:00:00 | 147.35 | 147.50 | 147.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 147.15 | 147.43 | 147.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:45:00 | 147.52 | 147.43 | 147.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 148.76 | 145.97 | 146.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 148.22 | 145.97 | 146.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 148.91 | 146.56 | 146.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 149.74 | 146.56 | 146.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 11:15:00 | 148.51 | 146.95 | 146.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 13:15:00 | 149.45 | 147.77 | 147.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 11:15:00 | 149.75 | 149.81 | 149.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 11:45:00 | 149.82 | 149.81 | 149.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 149.20 | 149.68 | 149.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:30:00 | 149.15 | 149.68 | 149.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 149.79 | 149.70 | 149.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 09:15:00 | 150.43 | 149.69 | 149.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 148.99 | 149.40 | 149.18 | SL hit (close<static) qty=1.00 sl=149.01 alert=retest2 |

### Cycle 164 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 152.46 | 154.26 | 154.43 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 155.40 | 154.42 | 154.35 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 12:15:00 | 153.30 | 154.20 | 154.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 13:15:00 | 152.66 | 153.89 | 154.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 153.94 | 153.48 | 153.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 10:15:00 | 153.94 | 153.48 | 153.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 153.94 | 153.48 | 153.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 153.94 | 153.48 | 153.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 153.97 | 153.58 | 153.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 153.97 | 153.58 | 153.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 154.35 | 153.73 | 153.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:30:00 | 154.65 | 153.73 | 153.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 153.31 | 153.65 | 153.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:30:00 | 153.05 | 153.41 | 153.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:00:00 | 153.04 | 153.44 | 153.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:30:00 | 153.03 | 153.53 | 153.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 13:15:00 | 154.03 | 153.53 | 153.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 13:15:00 | 154.03 | 153.53 | 153.47 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 151.36 | 153.28 | 153.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 14:15:00 | 149.94 | 151.52 | 152.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 153.71 | 151.28 | 151.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 153.71 | 151.28 | 151.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 153.71 | 151.28 | 151.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 153.71 | 151.28 | 151.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 153.79 | 151.78 | 151.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:00:00 | 153.79 | 151.78 | 151.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 153.06 | 152.04 | 151.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 12:15:00 | 154.29 | 152.49 | 152.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 15:15:00 | 167.00 | 167.35 | 165.65 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 09:15:00 | 168.62 | 167.35 | 165.65 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 168.64 | 168.59 | 167.54 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 167.26 | 168.16 | 167.76 | SL hit (close<ema400) qty=1.00 sl=167.76 alert=retest1 |

### Cycle 170 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 171.04 | 172.01 | 172.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 14:15:00 | 169.50 | 170.53 | 171.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 165.15 | 164.59 | 165.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 165.15 | 164.59 | 165.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 165.15 | 164.59 | 165.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:30:00 | 165.48 | 164.59 | 165.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 165.90 | 164.85 | 165.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 165.90 | 164.85 | 165.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 165.68 | 165.02 | 165.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:30:00 | 165.71 | 165.02 | 165.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 164.85 | 164.98 | 165.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 13:15:00 | 164.68 | 164.98 | 165.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:30:00 | 164.19 | 164.92 | 165.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 13:15:00 | 164.33 | 162.98 | 162.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — BUY (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 13:15:00 | 164.33 | 162.98 | 162.92 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 12:15:00 | 162.70 | 163.01 | 163.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 162.15 | 162.84 | 162.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 10:15:00 | 162.79 | 162.70 | 162.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 10:15:00 | 162.79 | 162.70 | 162.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 162.79 | 162.70 | 162.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:45:00 | 162.76 | 162.70 | 162.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 162.72 | 162.70 | 162.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:45:00 | 162.82 | 162.70 | 162.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 163.20 | 162.80 | 162.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:00:00 | 163.20 | 162.80 | 162.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 163.26 | 162.89 | 162.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 163.77 | 163.07 | 162.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 10:15:00 | 162.96 | 163.15 | 163.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 10:15:00 | 162.96 | 163.15 | 163.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 162.96 | 163.15 | 163.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 162.96 | 163.15 | 163.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 162.86 | 163.09 | 163.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:15:00 | 162.71 | 163.09 | 163.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 162.39 | 162.95 | 162.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 161.65 | 162.69 | 162.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 162.65 | 162.20 | 162.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 162.65 | 162.20 | 162.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 162.65 | 162.20 | 162.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 162.65 | 162.20 | 162.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 161.99 | 162.16 | 162.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:15:00 | 162.70 | 162.16 | 162.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 162.78 | 162.28 | 162.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 162.62 | 162.28 | 162.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 163.08 | 162.44 | 162.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 163.08 | 162.44 | 162.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 162.95 | 162.54 | 162.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 163.80 | 162.54 | 162.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 164.97 | 163.03 | 162.79 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 161.25 | 162.91 | 162.98 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 164.27 | 162.78 | 162.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 164.36 | 163.10 | 162.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 10:15:00 | 166.96 | 167.03 | 165.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 11:00:00 | 166.96 | 167.03 | 165.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 161.98 | 166.76 | 166.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 161.98 | 166.76 | 166.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 10:15:00 | 162.18 | 165.85 | 166.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 13:15:00 | 161.71 | 163.99 | 165.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 162.61 | 162.05 | 163.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 15:00:00 | 162.61 | 162.05 | 163.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 164.34 | 162.58 | 163.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:00:00 | 164.34 | 162.58 | 163.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 163.50 | 162.76 | 163.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:15:00 | 163.46 | 162.76 | 163.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 14:30:00 | 163.45 | 163.34 | 163.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 09:15:00 | 164.33 | 163.57 | 163.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — BUY (started 2025-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 09:15:00 | 164.33 | 163.57 | 163.55 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 163.15 | 163.49 | 163.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 09:15:00 | 162.60 | 163.32 | 163.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 161.70 | 160.75 | 161.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 161.70 | 160.75 | 161.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 161.70 | 160.75 | 161.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:45:00 | 161.94 | 160.75 | 161.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 161.80 | 160.96 | 161.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:30:00 | 161.85 | 160.96 | 161.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 161.28 | 161.21 | 161.50 | EMA400 retest candle locked (from downside) |

### Cycle 181 — BUY (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 09:15:00 | 162.16 | 161.74 | 161.69 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 12:15:00 | 161.36 | 161.63 | 161.65 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 164.47 | 162.16 | 161.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 10:15:00 | 165.54 | 162.84 | 162.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 12:15:00 | 165.01 | 165.58 | 164.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 13:00:00 | 165.01 | 165.58 | 164.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 165.24 | 165.55 | 164.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:45:00 | 164.99 | 165.55 | 164.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 165.52 | 166.09 | 165.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:30:00 | 165.23 | 166.09 | 165.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 163.89 | 165.65 | 165.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 163.89 | 165.65 | 165.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 165.24 | 165.57 | 165.47 | EMA400 retest candle locked (from upside) |

### Cycle 184 — SELL (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 15:15:00 | 164.70 | 165.39 | 165.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 162.67 | 164.85 | 165.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 164.23 | 163.67 | 164.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 14:15:00 | 164.23 | 163.67 | 164.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 164.23 | 163.67 | 164.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 15:00:00 | 164.23 | 163.67 | 164.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 163.90 | 163.71 | 164.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 162.92 | 163.71 | 164.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 163.53 | 163.68 | 164.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:00:00 | 162.49 | 163.39 | 163.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 13:15:00 | 162.33 | 163.25 | 163.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 15:15:00 | 162.50 | 163.11 | 163.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 159.27 | 158.06 | 157.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 159.27 | 158.06 | 157.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 159.55 | 158.35 | 158.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 12:15:00 | 160.05 | 160.12 | 159.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 13:00:00 | 160.05 | 160.12 | 159.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 160.04 | 160.68 | 159.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:45:00 | 160.04 | 160.68 | 159.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 160.28 | 160.60 | 159.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:30:00 | 159.91 | 160.60 | 159.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 159.90 | 160.46 | 159.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 12:45:00 | 159.74 | 160.46 | 159.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 160.33 | 160.44 | 159.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:15:00 | 159.79 | 160.44 | 159.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 160.95 | 160.54 | 160.06 | EMA400 retest candle locked (from upside) |

### Cycle 186 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 159.13 | 159.76 | 159.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 158.19 | 159.32 | 159.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 159.35 | 159.05 | 159.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 159.35 | 159.05 | 159.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 159.35 | 159.05 | 159.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:30:00 | 157.68 | 158.70 | 159.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 14:00:00 | 157.77 | 158.66 | 159.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 161.04 | 159.18 | 159.20 | SL hit (close>static) qty=1.00 sl=159.90 alert=retest2 |

### Cycle 187 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 160.00 | 159.35 | 159.27 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 13:15:00 | 158.58 | 159.14 | 159.20 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 160.16 | 159.26 | 159.23 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 158.62 | 159.13 | 159.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 157.76 | 158.86 | 159.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 157.85 | 157.52 | 158.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 09:45:00 | 157.75 | 157.52 | 158.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 157.76 | 157.57 | 158.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:30:00 | 157.68 | 157.57 | 158.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 158.25 | 157.71 | 158.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:00:00 | 158.25 | 157.71 | 158.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 12:15:00 | 157.94 | 157.75 | 158.15 | EMA400 retest candle locked (from downside) |

### Cycle 191 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 161.76 | 159.00 | 158.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 163.90 | 159.98 | 159.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 162.84 | 162.90 | 161.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 162.84 | 162.90 | 161.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 162.84 | 162.90 | 161.88 | EMA400 retest candle locked (from upside) |

### Cycle 192 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 160.18 | 161.82 | 161.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 159.90 | 161.44 | 161.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 161.33 | 160.79 | 161.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 161.33 | 160.79 | 161.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 161.33 | 160.79 | 161.25 | EMA400 retest candle locked (from downside) |

### Cycle 193 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 164.42 | 161.92 | 161.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 164.88 | 162.51 | 161.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 11:15:00 | 174.58 | 174.63 | 172.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 12:00:00 | 174.58 | 174.63 | 172.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 175.61 | 178.81 | 178.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 175.61 | 178.81 | 178.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 177.40 | 178.53 | 178.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 177.50 | 178.53 | 178.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 14:45:00 | 177.57 | 178.34 | 178.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 176.69 | 177.65 | 177.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 176.69 | 177.65 | 177.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 173.91 | 176.48 | 177.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 175.18 | 175.15 | 176.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 175.18 | 175.15 | 176.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 175.55 | 174.47 | 175.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 175.55 | 174.47 | 175.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 175.75 | 174.73 | 175.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 176.74 | 174.73 | 175.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 177.43 | 175.64 | 175.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 13:15:00 | 177.80 | 176.61 | 176.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 175.72 | 177.02 | 176.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 175.72 | 177.02 | 176.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 175.72 | 177.02 | 176.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 175.72 | 177.02 | 176.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 176.79 | 176.97 | 176.47 | EMA400 retest candle locked (from upside) |

### Cycle 196 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 174.03 | 176.09 | 176.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 173.84 | 175.64 | 175.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 174.24 | 174.14 | 174.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 14:00:00 | 174.24 | 174.14 | 174.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 175.60 | 174.27 | 174.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:15:00 | 176.00 | 174.27 | 174.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 176.54 | 174.72 | 174.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 176.54 | 174.72 | 174.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 175.79 | 175.19 | 175.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 15:15:00 | 176.40 | 175.58 | 175.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 179.81 | 185.41 | 184.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 179.81 | 185.41 | 184.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 179.81 | 185.41 | 184.32 | EMA400 retest candle locked (from upside) |

### Cycle 198 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 179.36 | 183.35 | 183.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 178.24 | 182.33 | 183.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 174.65 | 173.36 | 176.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 174.65 | 173.36 | 176.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 161.73 | 162.36 | 165.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 14:00:00 | 160.05 | 161.43 | 164.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:45:00 | 160.45 | 160.81 | 163.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 15:00:00 | 160.33 | 160.88 | 162.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 14:30:00 | 160.38 | 160.55 | 161.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 152.05 | 155.91 | 158.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 152.43 | 155.91 | 158.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 152.31 | 155.91 | 158.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 152.36 | 155.91 | 158.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 147.39 | 147.37 | 149.86 | SL hit (close>ema200) qty=0.50 sl=147.37 alert=retest2 |

### Cycle 199 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 142.12 | 135.26 | 134.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 11:15:00 | 144.18 | 138.18 | 135.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 141.15 | 141.21 | 138.56 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 143.04 | 141.84 | 140.20 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 13:30:00 | 143.07 | 142.45 | 141.06 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 15:00:00 | 142.96 | 142.56 | 141.23 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 139.19 | 141.95 | 141.19 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 139.19 | 141.95 | 141.19 | SL hit (close<ema400) qty=1.00 sl=141.19 alert=retest1 |

### Cycle 200 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 140.14 | 140.80 | 140.81 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 14:15:00 | 140.96 | 140.83 | 140.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 145.08 | 141.74 | 141.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 12:15:00 | 144.35 | 144.50 | 143.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 13:00:00 | 144.35 | 144.50 | 143.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 144.28 | 144.43 | 143.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 144.64 | 144.41 | 143.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:45:00 | 144.56 | 144.57 | 143.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 13:15:00 | 145.56 | 146.56 | 146.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 145.56 | 146.56 | 146.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 145.41 | 146.33 | 146.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 145.39 | 144.30 | 145.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 145.39 | 144.30 | 145.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 145.39 | 144.30 | 145.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 146.04 | 144.30 | 145.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 145.73 | 144.59 | 145.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 145.69 | 144.59 | 145.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 146.22 | 145.40 | 145.35 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 144.78 | 145.31 | 145.37 | EMA200 below EMA400 |

### Cycle 205 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 146.37 | 145.58 | 145.48 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 144.45 | 145.42 | 145.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 142.35 | 144.81 | 145.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 143.12 | 142.90 | 143.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 143.12 | 142.90 | 143.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 143.12 | 142.90 | 143.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 142.17 | 143.00 | 143.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 14:30:00 | 142.35 | 142.76 | 143.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 141.20 | 142.76 | 143.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 14:30:00 | 142.37 | 142.13 | 142.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 144.08 | 142.56 | 142.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 144.14 | 143.08 | 143.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 144.14 | 143.08 | 143.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 146.75 | 143.93 | 143.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 145.23 | 146.26 | 145.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 145.23 | 146.26 | 145.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 145.23 | 146.26 | 145.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 145.05 | 146.26 | 145.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 145.48 | 146.10 | 145.43 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-30 09:15:00 | 90.45 | 2023-06-02 12:15:00 | 90.20 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2023-05-30 11:45:00 | 90.35 | 2023-06-02 12:15:00 | 90.20 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2023-05-30 12:45:00 | 90.40 | 2023-06-02 12:15:00 | 90.20 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2023-05-30 13:30:00 | 90.40 | 2023-06-02 12:15:00 | 90.20 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2023-06-09 09:15:00 | 90.60 | 2023-06-16 14:15:00 | 92.10 | STOP_HIT | 1.00 | 1.66% |
| SELL | retest2 | 2023-06-21 11:45:00 | 92.10 | 2023-06-22 10:15:00 | 92.70 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-06-21 12:15:00 | 92.20 | 2023-06-22 10:15:00 | 92.70 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2023-06-21 14:00:00 | 91.90 | 2023-06-22 10:15:00 | 92.70 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2023-06-28 10:30:00 | 89.60 | 2023-06-28 14:15:00 | 90.60 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2023-07-05 12:15:00 | 95.20 | 2023-07-13 10:15:00 | 97.95 | STOP_HIT | 1.00 | 2.89% |
| BUY | retest2 | 2023-07-05 13:45:00 | 95.20 | 2023-07-13 10:15:00 | 97.95 | STOP_HIT | 1.00 | 2.89% |
| SELL | retest2 | 2023-07-14 12:30:00 | 96.90 | 2023-07-17 10:15:00 | 97.75 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest1 | 2023-07-25 09:15:00 | 100.05 | 2023-07-25 11:15:00 | 98.95 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2023-08-04 10:15:00 | 93.25 | 2023-08-09 14:15:00 | 93.85 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2023-08-04 13:00:00 | 93.25 | 2023-08-09 14:15:00 | 93.85 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2023-08-08 10:15:00 | 93.30 | 2023-08-09 14:15:00 | 93.85 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2023-08-23 09:45:00 | 91.70 | 2023-08-24 09:15:00 | 92.10 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2023-09-20 13:30:00 | 91.95 | 2023-09-21 10:15:00 | 93.10 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2023-09-21 12:45:00 | 92.05 | 2023-09-21 14:15:00 | 92.75 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2023-09-27 09:15:00 | 91.50 | 2023-10-11 10:15:00 | 89.95 | STOP_HIT | 1.00 | 1.69% |
| SELL | retest2 | 2023-10-20 11:30:00 | 89.80 | 2023-10-30 12:15:00 | 87.90 | STOP_HIT | 1.00 | 2.12% |
| SELL | retest2 | 2023-10-20 12:30:00 | 89.75 | 2023-10-30 12:15:00 | 87.90 | STOP_HIT | 1.00 | 2.06% |
| SELL | retest2 | 2023-10-23 09:15:00 | 89.85 | 2023-10-30 12:15:00 | 87.90 | STOP_HIT | 1.00 | 2.17% |
| BUY | retest2 | 2023-11-10 10:15:00 | 98.60 | 2023-11-10 10:15:00 | 98.20 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2023-11-17 15:15:00 | 103.20 | 2023-11-21 09:15:00 | 101.25 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2023-11-20 14:15:00 | 102.80 | 2023-11-21 09:15:00 | 101.25 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2023-11-22 12:00:00 | 101.45 | 2023-11-23 14:15:00 | 103.90 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2023-11-23 12:00:00 | 101.50 | 2023-11-23 14:15:00 | 103.90 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest1 | 2023-11-28 09:15:00 | 104.55 | 2023-11-30 14:15:00 | 109.78 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-11-28 09:15:00 | 104.55 | 2023-12-04 09:15:00 | 115.01 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2023-12-11 09:15:00 | 119.15 | 2023-12-11 12:15:00 | 117.50 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest1 | 2023-12-20 09:15:00 | 125.40 | 2023-12-20 13:15:00 | 121.40 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2024-01-01 09:45:00 | 132.20 | 2024-01-10 09:15:00 | 130.15 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-01-03 12:00:00 | 132.15 | 2024-01-10 09:15:00 | 130.15 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-01-03 12:30:00 | 133.40 | 2024-01-10 09:15:00 | 130.15 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2024-01-05 09:15:00 | 133.80 | 2024-01-10 09:15:00 | 130.15 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2024-01-08 09:15:00 | 135.65 | 2024-01-10 09:15:00 | 130.15 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2024-01-08 13:45:00 | 133.20 | 2024-01-10 09:15:00 | 130.15 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2024-01-09 09:15:00 | 133.30 | 2024-01-10 09:15:00 | 130.15 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2024-01-09 10:00:00 | 133.10 | 2024-01-10 09:15:00 | 130.15 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2024-01-25 13:30:00 | 142.55 | 2024-01-25 14:15:00 | 143.35 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-02-19 09:45:00 | 187.45 | 2024-02-20 09:15:00 | 185.70 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-02-20 13:30:00 | 188.60 | 2024-02-21 12:15:00 | 186.15 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-02-27 10:15:00 | 175.75 | 2024-02-28 15:15:00 | 166.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-27 11:30:00 | 175.40 | 2024-02-29 09:15:00 | 166.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-27 10:15:00 | 175.75 | 2024-02-29 15:15:00 | 167.05 | STOP_HIT | 0.50 | 4.95% |
| SELL | retest2 | 2024-02-27 11:30:00 | 175.40 | 2024-02-29 15:15:00 | 167.05 | STOP_HIT | 0.50 | 4.76% |
| BUY | retest2 | 2024-03-06 14:00:00 | 175.90 | 2024-03-11 15:15:00 | 174.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-03-07 10:45:00 | 176.05 | 2024-03-11 15:15:00 | 174.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-03-11 09:15:00 | 176.05 | 2024-03-11 15:15:00 | 174.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-03-11 12:30:00 | 175.85 | 2024-03-11 15:15:00 | 174.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-03-13 09:15:00 | 171.20 | 2024-03-13 14:15:00 | 162.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-13 09:15:00 | 171.20 | 2024-03-14 10:15:00 | 168.15 | STOP_HIT | 0.50 | 1.78% |
| BUY | retest2 | 2024-03-28 09:15:00 | 167.55 | 2024-04-04 13:15:00 | 169.60 | STOP_HIT | 1.00 | 1.22% |
| SELL | retest2 | 2024-05-13 10:30:00 | 155.10 | 2024-05-15 09:15:00 | 160.95 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2024-05-22 13:15:00 | 165.75 | 2024-05-28 12:15:00 | 167.05 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2024-05-22 13:45:00 | 165.75 | 2024-05-28 12:15:00 | 167.05 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2024-05-22 14:15:00 | 165.80 | 2024-05-28 12:15:00 | 167.05 | STOP_HIT | 1.00 | 0.75% |
| BUY | retest2 | 2024-06-18 15:15:00 | 170.00 | 2024-06-19 09:15:00 | 167.73 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-07-09 09:15:00 | 171.31 | 2024-07-10 10:15:00 | 169.80 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-07-10 11:45:00 | 170.40 | 2024-07-10 12:15:00 | 170.22 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2024-07-23 09:30:00 | 166.95 | 2024-07-24 13:15:00 | 168.52 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-07-24 11:00:00 | 167.56 | 2024-07-24 13:15:00 | 168.52 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2024-07-24 12:30:00 | 167.46 | 2024-07-24 13:15:00 | 168.52 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-08-08 09:15:00 | 171.50 | 2024-08-19 09:15:00 | 170.11 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2024-08-26 09:15:00 | 173.95 | 2024-08-27 09:15:00 | 172.09 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-08-26 10:00:00 | 173.45 | 2024-08-27 09:15:00 | 172.09 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-08-26 12:00:00 | 173.38 | 2024-08-27 09:15:00 | 172.09 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-08-26 14:30:00 | 173.40 | 2024-08-27 09:15:00 | 172.09 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-08-29 11:15:00 | 173.78 | 2024-09-03 14:15:00 | 176.09 | STOP_HIT | 1.00 | 1.33% |
| BUY | retest2 | 2024-08-29 11:45:00 | 174.09 | 2024-09-03 14:15:00 | 176.09 | STOP_HIT | 1.00 | 1.15% |
| SELL | retest2 | 2024-09-11 09:45:00 | 174.97 | 2024-09-13 11:15:00 | 174.28 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2024-09-13 10:45:00 | 174.90 | 2024-09-13 11:15:00 | 174.28 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2024-09-25 11:30:00 | 170.73 | 2024-10-03 11:15:00 | 172.01 | STOP_HIT | 1.00 | 0.75% |
| BUY | retest2 | 2024-09-25 13:15:00 | 170.69 | 2024-10-03 11:15:00 | 172.01 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest2 | 2024-09-26 13:00:00 | 170.72 | 2024-10-03 11:15:00 | 172.01 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2024-09-26 14:30:00 | 170.70 | 2024-10-03 11:15:00 | 172.01 | STOP_HIT | 1.00 | 0.77% |
| SELL | retest2 | 2024-10-09 11:30:00 | 165.20 | 2024-10-14 14:15:00 | 165.55 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-10-09 14:15:00 | 164.93 | 2024-10-14 14:15:00 | 165.55 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2024-10-10 11:00:00 | 165.20 | 2024-10-14 15:15:00 | 166.00 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2024-10-10 11:30:00 | 164.92 | 2024-10-14 15:15:00 | 166.00 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-10-14 12:15:00 | 163.62 | 2024-10-14 15:15:00 | 166.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-10-14 12:45:00 | 163.10 | 2024-10-14 15:15:00 | 166.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-10-21 11:30:00 | 161.50 | 2024-10-23 09:15:00 | 153.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 11:30:00 | 161.50 | 2024-10-28 10:15:00 | 148.41 | STOP_HIT | 0.50 | 8.11% |
| SELL | retest2 | 2024-11-12 13:30:00 | 139.56 | 2024-11-21 09:15:00 | 132.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 14:30:00 | 139.35 | 2024-11-21 09:15:00 | 132.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 13:30:00 | 139.56 | 2024-11-22 11:15:00 | 132.05 | STOP_HIT | 0.50 | 5.38% |
| SELL | retest2 | 2024-11-12 14:30:00 | 139.35 | 2024-11-22 11:15:00 | 132.05 | STOP_HIT | 0.50 | 5.24% |
| BUY | retest2 | 2024-11-29 14:30:00 | 138.96 | 2024-12-02 14:15:00 | 137.49 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-12-05 12:00:00 | 139.18 | 2024-12-12 12:15:00 | 142.57 | STOP_HIT | 1.00 | 2.44% |
| SELL | retest2 | 2024-12-23 13:15:00 | 138.01 | 2025-01-01 12:15:00 | 136.90 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2024-12-23 14:30:00 | 138.05 | 2025-01-01 12:15:00 | 136.90 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2024-12-24 15:15:00 | 138.00 | 2025-01-01 12:15:00 | 136.90 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2024-12-26 10:45:00 | 137.79 | 2025-01-01 12:15:00 | 136.90 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2024-12-27 11:30:00 | 137.72 | 2025-01-01 12:15:00 | 136.90 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2025-01-02 13:30:00 | 137.63 | 2025-01-06 09:15:00 | 135.20 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-01-03 10:45:00 | 137.35 | 2025-01-06 09:15:00 | 135.20 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-01-09 09:15:00 | 132.86 | 2025-01-13 09:15:00 | 126.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 11:00:00 | 133.11 | 2025-01-13 09:15:00 | 126.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 11:30:00 | 133.02 | 2025-01-13 09:15:00 | 126.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 09:15:00 | 132.86 | 2025-01-14 09:15:00 | 124.71 | STOP_HIT | 0.50 | 6.13% |
| SELL | retest2 | 2025-01-09 11:00:00 | 133.11 | 2025-01-14 09:15:00 | 124.71 | STOP_HIT | 0.50 | 6.31% |
| SELL | retest2 | 2025-01-09 11:30:00 | 133.02 | 2025-01-14 09:15:00 | 124.71 | STOP_HIT | 0.50 | 6.25% |
| SELL | retest2 | 2025-02-13 13:00:00 | 120.94 | 2025-02-17 09:15:00 | 114.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:00:00 | 120.94 | 2025-02-17 13:15:00 | 117.35 | STOP_HIT | 0.50 | 2.97% |
| BUY | retest2 | 2025-03-25 13:30:00 | 132.33 | 2025-03-26 13:15:00 | 129.93 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-03-26 09:15:00 | 133.05 | 2025-03-26 13:15:00 | 129.93 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-03-26 10:30:00 | 132.43 | 2025-03-26 13:15:00 | 129.93 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-03-28 11:15:00 | 129.90 | 2025-04-01 11:15:00 | 130.80 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-04-01 10:00:00 | 129.86 | 2025-04-01 11:15:00 | 130.80 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-04-02 14:15:00 | 130.94 | 2025-04-04 12:15:00 | 129.60 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-04-03 11:45:00 | 131.20 | 2025-04-04 12:15:00 | 129.60 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-04-03 15:00:00 | 130.99 | 2025-04-04 12:15:00 | 129.60 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-04-04 10:30:00 | 130.98 | 2025-04-04 12:15:00 | 129.60 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-04-16 09:15:00 | 133.64 | 2025-04-25 09:15:00 | 135.28 | STOP_HIT | 1.00 | 1.23% |
| BUY | retest2 | 2025-04-16 11:15:00 | 133.55 | 2025-04-25 09:15:00 | 135.28 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2025-04-16 13:00:00 | 133.32 | 2025-04-25 10:15:00 | 133.70 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-04-16 13:45:00 | 133.33 | 2025-04-25 10:15:00 | 133.70 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-04-23 13:45:00 | 137.65 | 2025-04-25 10:15:00 | 133.70 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-04-23 14:45:00 | 137.58 | 2025-04-25 10:15:00 | 133.70 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2025-05-07 11:15:00 | 144.45 | 2025-05-08 10:15:00 | 143.40 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-05-07 14:30:00 | 144.37 | 2025-05-08 10:15:00 | 143.40 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-05-07 15:00:00 | 144.37 | 2025-05-08 10:15:00 | 143.40 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-05-12 14:15:00 | 142.08 | 2025-05-13 09:15:00 | 142.16 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-05-13 09:15:00 | 141.85 | 2025-05-13 09:15:00 | 142.16 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-05-19 09:15:00 | 145.25 | 2025-05-19 13:15:00 | 144.42 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-05-19 10:00:00 | 145.49 | 2025-05-19 13:15:00 | 144.42 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-05-19 12:45:00 | 145.01 | 2025-05-19 13:15:00 | 144.42 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-05-28 09:15:00 | 142.88 | 2025-05-28 10:15:00 | 143.92 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-05-28 10:15:00 | 142.94 | 2025-05-28 10:15:00 | 143.92 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-06-05 10:45:00 | 140.55 | 2025-06-09 09:15:00 | 142.21 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-06-05 12:15:00 | 140.59 | 2025-06-09 09:15:00 | 142.21 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-06-06 10:30:00 | 140.45 | 2025-06-09 09:15:00 | 142.21 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-06-06 13:00:00 | 140.59 | 2025-06-09 09:15:00 | 142.21 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-06-11 09:15:00 | 145.20 | 2025-06-12 09:15:00 | 142.02 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-06-12 10:15:00 | 142.40 | 2025-06-13 09:15:00 | 140.80 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-06-19 10:30:00 | 139.35 | 2025-06-24 09:15:00 | 143.95 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2025-06-19 11:30:00 | 139.44 | 2025-06-24 09:15:00 | 143.95 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2025-06-19 12:15:00 | 139.40 | 2025-06-24 09:15:00 | 143.95 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2025-06-23 13:45:00 | 139.47 | 2025-06-24 09:15:00 | 143.95 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2025-07-04 09:15:00 | 148.75 | 2025-07-09 14:15:00 | 150.35 | STOP_HIT | 1.00 | 1.08% |
| SELL | retest2 | 2025-07-14 11:15:00 | 150.12 | 2025-07-15 14:15:00 | 151.47 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-08-21 09:15:00 | 142.37 | 2025-08-22 09:15:00 | 140.54 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-08-21 13:15:00 | 141.97 | 2025-08-22 09:15:00 | 140.54 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-08-26 10:30:00 | 138.79 | 2025-09-01 14:15:00 | 139.61 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-08-26 15:00:00 | 138.94 | 2025-09-01 14:15:00 | 139.61 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-08-28 09:15:00 | 138.33 | 2025-09-01 15:15:00 | 139.56 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-08-28 12:45:00 | 138.95 | 2025-09-01 15:15:00 | 139.56 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-08-29 12:15:00 | 137.98 | 2025-09-01 15:15:00 | 139.56 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-09-01 11:45:00 | 137.93 | 2025-09-01 15:15:00 | 139.56 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-09-15 15:00:00 | 144.10 | 2025-09-24 13:15:00 | 147.28 | STOP_HIT | 1.00 | 2.21% |
| BUY | retest2 | 2025-10-03 09:15:00 | 150.43 | 2025-10-03 11:15:00 | 148.99 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-10-03 13:15:00 | 149.87 | 2025-10-13 09:15:00 | 152.46 | STOP_HIT | 1.00 | 1.73% |
| SELL | retest2 | 2025-10-16 09:30:00 | 153.05 | 2025-10-20 13:15:00 | 154.03 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-10-16 13:00:00 | 153.04 | 2025-10-20 13:15:00 | 154.03 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-10-17 11:30:00 | 153.03 | 2025-10-20 13:15:00 | 154.03 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest1 | 2025-11-04 09:15:00 | 168.62 | 2025-11-07 09:15:00 | 167.26 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-11-07 13:15:00 | 169.35 | 2025-11-18 10:15:00 | 171.04 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2025-11-10 09:15:00 | 170.79 | 2025-11-18 10:15:00 | 171.04 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2025-11-10 11:15:00 | 169.40 | 2025-11-18 10:15:00 | 171.04 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2025-11-10 13:00:00 | 169.28 | 2025-11-18 10:15:00 | 171.04 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2025-11-11 11:30:00 | 170.12 | 2025-11-18 10:15:00 | 171.04 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2025-11-26 13:15:00 | 164.68 | 2025-12-03 13:15:00 | 164.33 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-11-27 09:30:00 | 164.19 | 2025-12-03 13:15:00 | 164.33 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-12-22 11:15:00 | 163.46 | 2025-12-23 09:15:00 | 164.33 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-12-22 14:30:00 | 163.45 | 2025-12-23 09:15:00 | 164.33 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-01-07 12:00:00 | 162.49 | 2026-01-14 11:15:00 | 159.27 | STOP_HIT | 1.00 | 1.98% |
| SELL | retest2 | 2026-01-07 13:15:00 | 162.33 | 2026-01-14 11:15:00 | 159.27 | STOP_HIT | 1.00 | 1.89% |
| SELL | retest2 | 2026-01-07 15:15:00 | 162.50 | 2026-01-14 11:15:00 | 159.27 | STOP_HIT | 1.00 | 1.99% |
| SELL | retest2 | 2026-01-21 10:30:00 | 157.68 | 2026-01-22 09:15:00 | 161.04 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2026-01-21 14:00:00 | 157.77 | 2026-01-22 09:15:00 | 161.04 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2026-02-12 13:15:00 | 177.50 | 2026-02-13 10:15:00 | 176.69 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2026-02-12 14:45:00 | 177.57 | 2026-02-13 10:15:00 | 176.69 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-03-10 14:00:00 | 160.05 | 2026-03-16 09:15:00 | 152.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:45:00 | 160.45 | 2026-03-16 09:15:00 | 152.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 15:00:00 | 160.33 | 2026-03-16 09:15:00 | 152.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 14:30:00 | 160.38 | 2026-03-16 09:15:00 | 152.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 14:00:00 | 160.05 | 2026-03-18 10:15:00 | 147.39 | STOP_HIT | 0.50 | 7.91% |
| SELL | retest2 | 2026-03-11 10:45:00 | 160.45 | 2026-03-18 10:15:00 | 147.39 | STOP_HIT | 0.50 | 8.14% |
| SELL | retest2 | 2026-03-11 15:00:00 | 160.33 | 2026-03-18 10:15:00 | 147.39 | STOP_HIT | 0.50 | 8.07% |
| SELL | retest2 | 2026-03-12 14:30:00 | 160.38 | 2026-03-18 10:15:00 | 147.39 | STOP_HIT | 0.50 | 8.10% |
| SELL | retest2 | 2026-03-23 09:15:00 | 140.83 | 2026-04-02 09:15:00 | 133.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-23 09:15:00 | 140.83 | 2026-04-02 14:15:00 | 134.38 | STOP_HIT | 0.50 | 4.58% |
| BUY | retest1 | 2026-04-10 09:30:00 | 143.04 | 2026-04-13 09:15:00 | 139.19 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest1 | 2026-04-10 13:30:00 | 143.07 | 2026-04-13 09:15:00 | 139.19 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest1 | 2026-04-10 15:00:00 | 142.96 | 2026-04-13 09:15:00 | 139.19 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2026-04-13 10:15:00 | 139.72 | 2026-04-13 13:15:00 | 140.14 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2026-04-13 10:45:00 | 139.70 | 2026-04-13 13:15:00 | 140.14 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2026-04-17 09:15:00 | 144.64 | 2026-04-23 13:15:00 | 145.56 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2026-04-17 09:45:00 | 144.56 | 2026-04-23 13:15:00 | 145.56 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2026-05-04 13:15:00 | 142.17 | 2026-05-06 11:15:00 | 144.14 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-05-04 14:30:00 | 142.35 | 2026-05-06 11:15:00 | 144.14 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-05-05 09:15:00 | 141.20 | 2026-05-06 11:15:00 | 144.14 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2026-05-05 14:30:00 | 142.37 | 2026-05-06 11:15:00 | 144.14 | STOP_HIT | 1.00 | -1.24% |
