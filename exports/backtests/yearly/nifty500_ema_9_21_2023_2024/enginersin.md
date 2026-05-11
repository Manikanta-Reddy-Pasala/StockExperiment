# Engineers India Ltd. (ENGINERSIN)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 256.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 207 |
| ALERT1 | 147 |
| ALERT2 | 143 |
| ALERT2_SKIP | 79 |
| ALERT3 | 387 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 183 |
| PARTIAL | 43 |
| TARGET_HIT | 25 |
| STOP_HIT | 167 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 235 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 114 / 121
- **Target hits / Stop hits / Partials:** 25 / 167 / 43
- **Avg / median % per leg:** 1.40% / -0.02%
- **Sum % (uncompounded):** 328.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 75 | 18 | 24.0% | 13 | 62 | 0 | -0.07% | -5.4% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 4 | 0 | -0.39% | -1.6% |
| BUY @ 3rd Alert (retest2) | 71 | 16 | 22.5% | 13 | 58 | 0 | -0.05% | -3.8% |
| SELL (all) | 160 | 96 | 60.0% | 12 | 105 | 43 | 2.08% | 333.5% |
| SELL @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.72% | -3.6% |
| SELL @ 3rd Alert (retest2) | 155 | 96 | 61.9% | 12 | 100 | 43 | 2.17% | 337.1% |
| retest1 (combined) | 9 | 2 | 22.2% | 0 | 9 | 0 | -0.57% | -5.2% |
| retest2 (combined) | 226 | 112 | 49.6% | 25 | 158 | 43 | 1.47% | 333.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 11:15:00 | 97.35 | 98.07 | 98.11 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 11:15:00 | 101.25 | 98.42 | 98.12 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 11:15:00 | 98.60 | 99.03 | 99.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 13:15:00 | 96.85 | 98.54 | 98.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 11:15:00 | 100.30 | 97.89 | 98.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 11:15:00 | 100.30 | 97.89 | 98.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 11:15:00 | 100.30 | 97.89 | 98.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 12:00:00 | 100.30 | 97.89 | 98.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 12:15:00 | 100.15 | 98.34 | 98.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 13:00:00 | 100.15 | 98.34 | 98.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2023-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 13:15:00 | 100.40 | 98.75 | 98.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 09:15:00 | 100.90 | 100.06 | 99.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 09:15:00 | 106.55 | 108.62 | 106.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 09:15:00 | 106.55 | 108.62 | 106.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 106.55 | 108.62 | 106.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 10:15:00 | 106.65 | 108.62 | 106.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 10:15:00 | 106.20 | 108.13 | 106.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 10:45:00 | 105.60 | 108.13 | 106.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 11:15:00 | 105.40 | 107.59 | 106.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 11:45:00 | 105.50 | 107.59 | 106.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2023-05-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 15:15:00 | 104.30 | 105.96 | 105.99 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 09:15:00 | 106.25 | 106.02 | 106.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-29 13:15:00 | 112.10 | 108.17 | 107.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-31 09:15:00 | 109.95 | 110.06 | 109.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-31 10:15:00 | 109.30 | 110.06 | 109.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 11:15:00 | 109.15 | 109.75 | 109.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 11:45:00 | 108.55 | 109.75 | 109.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 12:15:00 | 109.15 | 109.63 | 109.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 14:00:00 | 109.80 | 109.66 | 109.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 14:30:00 | 109.80 | 109.88 | 109.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 15:00:00 | 110.75 | 109.88 | 109.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 09:45:00 | 109.85 | 109.95 | 109.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 10:15:00 | 109.20 | 109.80 | 109.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 10:45:00 | 108.95 | 109.80 | 109.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 11:15:00 | 109.30 | 109.70 | 109.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 11:30:00 | 109.15 | 109.70 | 109.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-06-02 12:15:00 | 108.95 | 109.55 | 109.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2023-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 12:15:00 | 108.95 | 109.55 | 109.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-02 13:15:00 | 108.60 | 109.36 | 109.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-05 11:15:00 | 108.50 | 108.22 | 108.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 11:15:00 | 108.50 | 108.22 | 108.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 11:15:00 | 108.50 | 108.22 | 108.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-05 12:00:00 | 108.50 | 108.22 | 108.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 12:15:00 | 110.00 | 108.58 | 108.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-05 12:45:00 | 110.40 | 108.58 | 108.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 13:15:00 | 110.00 | 108.86 | 109.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-05 13:30:00 | 110.55 | 108.86 | 109.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2023-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 15:15:00 | 110.00 | 109.13 | 109.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-06 09:15:00 | 110.05 | 109.31 | 109.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-06 14:15:00 | 110.20 | 110.30 | 109.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-06 14:45:00 | 110.20 | 110.30 | 109.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 110.30 | 111.83 | 111.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 11:45:00 | 110.10 | 111.83 | 111.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 110.40 | 111.55 | 111.25 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 15:15:00 | 110.25 | 111.04 | 111.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 12:15:00 | 109.85 | 110.45 | 110.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 09:15:00 | 110.25 | 110.10 | 110.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 09:15:00 | 110.25 | 110.10 | 110.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 110.25 | 110.10 | 110.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 09:30:00 | 110.10 | 110.10 | 110.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 10:15:00 | 110.00 | 110.08 | 110.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-12 12:15:00 | 109.90 | 110.05 | 110.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-13 09:15:00 | 112.00 | 110.42 | 110.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2023-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 09:15:00 | 112.00 | 110.42 | 110.41 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-06-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 09:15:00 | 109.90 | 111.00 | 111.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-15 11:15:00 | 109.45 | 110.47 | 110.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 10:15:00 | 109.80 | 109.40 | 109.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 10:15:00 | 109.80 | 109.40 | 109.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 10:15:00 | 109.80 | 109.40 | 109.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 10:45:00 | 110.05 | 109.40 | 109.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 11:15:00 | 112.30 | 109.98 | 110.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 12:00:00 | 112.30 | 109.98 | 110.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2023-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 12:15:00 | 111.85 | 110.35 | 110.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 15:15:00 | 114.85 | 112.98 | 112.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 09:15:00 | 118.30 | 118.97 | 117.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-22 09:45:00 | 117.50 | 118.97 | 117.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 115.35 | 118.11 | 117.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 12:00:00 | 115.35 | 118.11 | 117.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 12:15:00 | 115.00 | 117.49 | 117.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 13:00:00 | 115.00 | 117.49 | 117.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2023-06-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 14:15:00 | 114.80 | 116.57 | 116.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 113.50 | 115.73 | 116.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 10:15:00 | 115.80 | 114.37 | 115.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 10:15:00 | 115.80 | 114.37 | 115.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 115.80 | 114.37 | 115.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 11:00:00 | 115.80 | 114.37 | 115.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 11:15:00 | 114.95 | 114.49 | 115.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 12:45:00 | 114.50 | 114.47 | 114.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-27 09:15:00 | 116.40 | 115.13 | 115.15 | SL hit (close>static) qty=1.00 sl=115.85 alert=retest2 |

### Cycle 14 — BUY (started 2023-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 10:15:00 | 116.80 | 115.46 | 115.30 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 09:15:00 | 115.35 | 115.58 | 115.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-30 11:15:00 | 114.60 | 115.27 | 115.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-03 14:15:00 | 114.70 | 114.70 | 114.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-03 15:00:00 | 114.70 | 114.70 | 114.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 15:15:00 | 114.60 | 114.68 | 114.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-04 09:15:00 | 115.15 | 114.68 | 114.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 113.60 | 114.46 | 114.80 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 10:15:00 | 118.20 | 115.33 | 114.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 09:15:00 | 120.05 | 116.95 | 116.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 15:15:00 | 121.00 | 121.18 | 119.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-10 09:15:00 | 119.80 | 121.18 | 119.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 119.40 | 120.83 | 119.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:45:00 | 119.00 | 120.83 | 119.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 120.10 | 120.68 | 119.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-11 09:15:00 | 121.00 | 120.25 | 119.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-11 11:30:00 | 121.40 | 120.59 | 120.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-12 09:30:00 | 121.25 | 120.92 | 120.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-17 10:15:00 | 133.10 | 127.07 | 124.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2023-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 11:15:00 | 147.75 | 152.78 | 153.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 145.75 | 150.51 | 152.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 14:15:00 | 147.60 | 147.41 | 149.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-03 15:00:00 | 147.60 | 147.41 | 149.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 148.95 | 147.60 | 149.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 09:30:00 | 149.80 | 147.60 | 149.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 10:15:00 | 147.25 | 147.53 | 148.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 09:15:00 | 146.80 | 147.59 | 148.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 10:00:00 | 147.00 | 147.47 | 148.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 10:45:00 | 147.05 | 147.34 | 148.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 11:30:00 | 147.00 | 147.26 | 148.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 12:15:00 | 147.30 | 147.26 | 147.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-07 13:00:00 | 147.30 | 147.26 | 147.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 14:15:00 | 150.00 | 147.71 | 148.05 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-08-07 14:15:00 | 150.00 | 147.71 | 148.05 | SL hit (close>static) qty=1.00 sl=148.95 alert=retest2 |

### Cycle 18 — BUY (started 2023-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 10:15:00 | 148.45 | 148.25 | 148.25 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 12:15:00 | 148.20 | 148.24 | 148.24 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 13:15:00 | 153.90 | 149.37 | 148.76 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 10:15:00 | 150.50 | 151.42 | 151.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 09:15:00 | 147.95 | 150.56 | 151.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 14:15:00 | 149.35 | 149.28 | 150.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-14 14:15:00 | 149.35 | 149.28 | 150.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 14:15:00 | 149.35 | 149.28 | 150.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 14:30:00 | 149.80 | 149.28 | 150.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 15:15:00 | 149.50 | 149.33 | 150.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 09:45:00 | 151.15 | 149.86 | 150.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 10:15:00 | 149.80 | 149.85 | 150.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 11:30:00 | 149.65 | 149.88 | 150.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 12:15:00 | 148.80 | 149.88 | 150.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 10:45:00 | 149.75 | 149.57 | 149.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-18 09:15:00 | 152.50 | 150.24 | 150.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2023-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 09:15:00 | 152.50 | 150.24 | 150.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 11:15:00 | 159.90 | 154.16 | 152.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 10:15:00 | 162.55 | 162.74 | 159.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-24 10:30:00 | 162.70 | 162.74 | 159.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 13:15:00 | 161.00 | 162.17 | 160.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 13:45:00 | 161.05 | 162.17 | 160.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 14:15:00 | 161.10 | 161.96 | 160.40 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 12:15:00 | 157.55 | 159.51 | 159.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-29 11:15:00 | 155.65 | 157.74 | 158.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-30 09:15:00 | 157.30 | 156.97 | 157.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 09:15:00 | 157.30 | 156.97 | 157.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 09:15:00 | 157.30 | 156.97 | 157.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-30 13:30:00 | 156.20 | 157.12 | 157.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-30 15:00:00 | 155.80 | 156.85 | 157.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-31 11:45:00 | 155.45 | 156.51 | 157.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-01 11:30:00 | 156.30 | 156.12 | 156.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 12:15:00 | 156.05 | 156.10 | 156.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-09-01 15:15:00 | 158.50 | 156.86 | 156.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2023-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 15:15:00 | 158.50 | 156.86 | 156.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 09:15:00 | 162.75 | 158.03 | 157.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 12:15:00 | 161.95 | 163.22 | 161.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 12:15:00 | 161.95 | 163.22 | 161.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 12:15:00 | 161.95 | 163.22 | 161.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 12:45:00 | 162.10 | 163.22 | 161.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 15:15:00 | 162.30 | 162.99 | 161.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 09:15:00 | 158.80 | 162.99 | 161.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 09:15:00 | 159.45 | 162.28 | 161.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 09:30:00 | 159.00 | 162.28 | 161.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 10:15:00 | 159.50 | 161.72 | 161.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 10:30:00 | 159.45 | 161.72 | 161.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2023-09-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 11:15:00 | 157.75 | 160.93 | 160.98 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 15:15:00 | 161.85 | 158.96 | 158.59 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 152.60 | 157.69 | 158.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 14:15:00 | 146.75 | 152.19 | 154.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 15:15:00 | 147.70 | 147.31 | 150.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-14 09:15:00 | 148.10 | 147.31 | 150.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 148.50 | 147.55 | 150.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:30:00 | 150.30 | 147.55 | 150.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 09:15:00 | 151.90 | 148.84 | 149.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-15 09:45:00 | 152.60 | 148.84 | 149.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 10:15:00 | 152.75 | 149.62 | 149.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-15 11:00:00 | 152.75 | 149.62 | 149.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2023-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 11:15:00 | 153.10 | 150.32 | 150.09 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 10:15:00 | 148.60 | 150.73 | 150.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 12:15:00 | 146.80 | 149.71 | 150.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 12:15:00 | 145.15 | 144.61 | 146.24 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 09:15:00 | 143.30 | 144.46 | 145.76 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 10:00:00 | 143.40 | 144.25 | 145.55 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 11:00:00 | 143.80 | 144.16 | 145.39 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 14:30:00 | 143.70 | 144.07 | 144.96 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 143.30 | 142.85 | 143.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 09:45:00 | 144.05 | 142.85 | 143.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 143.95 | 143.07 | 143.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-09-27 10:15:00 | 143.95 | 143.07 | 143.67 | SL hit (close>ema400) qty=1.00 sl=143.67 alert=retest1 |

### Cycle 30 — BUY (started 2023-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 13:15:00 | 145.25 | 144.10 | 144.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 13:15:00 | 148.25 | 145.46 | 144.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 09:15:00 | 145.80 | 146.90 | 146.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 145.80 | 146.90 | 146.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 145.80 | 146.90 | 146.25 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 13:15:00 | 145.00 | 145.95 | 145.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 09:15:00 | 144.05 | 145.46 | 145.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 144.10 | 143.57 | 144.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 144.10 | 143.57 | 144.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 144.10 | 143.57 | 144.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 09:45:00 | 143.75 | 143.57 | 144.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 13:15:00 | 143.10 | 143.00 | 143.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 13:45:00 | 143.50 | 143.00 | 143.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 14:15:00 | 142.95 | 142.99 | 143.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 09:15:00 | 137.60 | 143.00 | 143.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-12 09:15:00 | 142.85 | 141.07 | 140.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2023-10-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 09:15:00 | 142.85 | 141.07 | 140.96 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 14:15:00 | 140.00 | 141.19 | 141.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-16 09:15:00 | 138.95 | 140.55 | 141.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 140.95 | 139.83 | 140.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 09:15:00 | 140.95 | 139.83 | 140.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 140.95 | 139.83 | 140.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 09:30:00 | 141.45 | 139.83 | 140.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 10:15:00 | 140.65 | 139.99 | 140.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 11:30:00 | 140.05 | 140.00 | 140.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 10:45:00 | 139.90 | 138.16 | 138.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-19 14:15:00 | 139.75 | 139.06 | 138.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2023-10-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-19 14:15:00 | 139.75 | 139.06 | 138.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-20 09:15:00 | 140.45 | 139.41 | 139.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-20 11:15:00 | 138.20 | 139.28 | 139.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 11:15:00 | 138.20 | 139.28 | 139.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 11:15:00 | 138.20 | 139.28 | 139.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 11:45:00 | 138.45 | 139.28 | 139.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2023-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 12:15:00 | 137.00 | 138.83 | 138.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 14:15:00 | 136.20 | 138.01 | 138.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 123.90 | 120.73 | 124.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 123.90 | 120.73 | 124.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 123.90 | 120.73 | 124.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:45:00 | 124.10 | 120.73 | 124.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 125.60 | 121.70 | 124.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 11:00:00 | 125.60 | 121.70 | 124.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 11:15:00 | 126.15 | 122.59 | 124.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 11:30:00 | 126.80 | 122.59 | 124.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2023-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 09:15:00 | 128.40 | 125.65 | 125.39 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 15:15:00 | 123.00 | 125.97 | 126.18 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-11-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 13:15:00 | 126.20 | 125.29 | 125.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 09:15:00 | 129.90 | 126.27 | 125.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 14:15:00 | 128.50 | 128.56 | 127.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-03 15:00:00 | 128.50 | 128.56 | 127.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 14:15:00 | 128.35 | 128.75 | 128.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 15:00:00 | 128.35 | 128.75 | 128.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 128.00 | 128.57 | 128.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 09:30:00 | 128.05 | 128.57 | 128.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 10:15:00 | 130.50 | 128.96 | 128.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 12:45:00 | 131.75 | 129.60 | 128.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 11:15:00 | 131.80 | 130.17 | 129.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-10 09:15:00 | 133.10 | 130.07 | 129.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2023-11-12 18:15:00 | 144.93 | 138.35 | 134.89 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2023-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 13:15:00 | 142.60 | 143.94 | 143.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 14:15:00 | 141.80 | 143.51 | 143.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 09:15:00 | 143.70 | 143.27 | 143.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 09:15:00 | 143.70 | 143.27 | 143.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 143.70 | 143.27 | 143.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 09:45:00 | 144.35 | 143.27 | 143.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 10:15:00 | 142.60 | 143.13 | 143.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-20 11:30:00 | 142.00 | 142.90 | 143.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-21 09:15:00 | 148.70 | 143.78 | 143.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2023-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 09:15:00 | 148.70 | 143.78 | 143.52 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 09:15:00 | 143.50 | 145.24 | 145.42 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 14:15:00 | 150.20 | 145.71 | 145.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 09:15:00 | 153.55 | 148.99 | 147.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 14:15:00 | 151.25 | 151.26 | 149.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-01 15:00:00 | 151.25 | 151.26 | 149.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 11:15:00 | 156.65 | 156.98 | 154.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 11:45:00 | 155.25 | 156.98 | 154.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 09:15:00 | 159.40 | 157.48 | 155.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 09:30:00 | 155.90 | 157.48 | 155.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 157.30 | 158.06 | 156.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-07 10:00:00 | 157.30 | 158.06 | 156.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 10:15:00 | 156.70 | 157.79 | 156.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-07 10:30:00 | 156.90 | 157.79 | 156.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 11:15:00 | 156.55 | 157.54 | 156.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-07 11:45:00 | 156.70 | 157.54 | 156.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 12:15:00 | 154.75 | 156.98 | 156.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-07 12:30:00 | 154.70 | 156.98 | 156.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2023-12-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 14:15:00 | 155.65 | 156.49 | 156.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 12:15:00 | 153.65 | 155.50 | 156.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 09:15:00 | 155.05 | 154.54 | 155.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 09:15:00 | 155.05 | 154.54 | 155.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 155.05 | 154.54 | 155.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 09:45:00 | 155.25 | 154.54 | 155.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 10:15:00 | 154.75 | 154.58 | 155.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 10:30:00 | 155.50 | 154.58 | 155.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 13:15:00 | 155.15 | 154.46 | 155.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 14:00:00 | 155.15 | 154.46 | 155.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2023-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 14:15:00 | 160.00 | 155.57 | 155.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 09:15:00 | 160.70 | 158.75 | 157.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-13 14:15:00 | 159.50 | 159.65 | 158.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-13 14:30:00 | 159.90 | 159.65 | 158.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 15:15:00 | 158.85 | 159.49 | 158.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:15:00 | 158.80 | 159.49 | 158.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 158.65 | 159.32 | 158.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:45:00 | 159.10 | 159.32 | 158.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 10:15:00 | 167.40 | 160.94 | 159.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-14 15:00:00 | 169.70 | 164.89 | 161.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-15 09:15:00 | 170.80 | 165.69 | 162.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-15 09:45:00 | 171.35 | 166.74 | 163.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-15 13:45:00 | 170.35 | 167.97 | 165.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 165.85 | 167.43 | 165.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 09:30:00 | 165.20 | 167.43 | 165.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 10:15:00 | 166.05 | 167.15 | 165.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 11:00:00 | 166.05 | 167.15 | 165.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 11:15:00 | 165.60 | 166.84 | 165.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 12:00:00 | 165.60 | 166.84 | 165.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 12:15:00 | 168.70 | 167.21 | 165.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 12:30:00 | 166.00 | 167.21 | 165.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 164.65 | 167.30 | 166.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:00:00 | 164.65 | 167.30 | 166.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 165.45 | 166.93 | 166.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 12:45:00 | 167.20 | 166.58 | 166.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 09:15:00 | 166.80 | 166.38 | 166.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 11:15:00 | 163.30 | 165.75 | 166.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2023-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 11:15:00 | 163.30 | 165.75 | 166.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 12:15:00 | 161.05 | 164.81 | 165.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 155.95 | 155.49 | 158.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 15:00:00 | 155.95 | 155.49 | 158.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 157.65 | 156.01 | 158.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:15:00 | 158.70 | 156.01 | 158.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 158.55 | 156.52 | 158.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:30:00 | 158.55 | 156.52 | 158.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 11:15:00 | 157.75 | 156.76 | 158.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 12:15:00 | 156.90 | 156.76 | 158.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-22 14:15:00 | 161.15 | 157.80 | 158.48 | SL hit (close>static) qty=1.00 sl=159.05 alert=retest2 |

### Cycle 46 — BUY (started 2023-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 09:15:00 | 161.95 | 159.05 | 158.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 09:15:00 | 165.50 | 161.62 | 160.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 12:15:00 | 162.55 | 162.76 | 161.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 12:30:00 | 162.50 | 162.76 | 161.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 09:15:00 | 167.45 | 163.54 | 162.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-28 10:15:00 | 172.10 | 163.54 | 162.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-02 13:15:00 | 189.31 | 182.99 | 178.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2024-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 11:15:00 | 196.40 | 198.46 | 198.46 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 13:15:00 | 201.65 | 199.07 | 198.74 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 10:15:00 | 197.15 | 198.52 | 198.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 12:15:00 | 195.40 | 197.59 | 198.13 | Break + close below crossover candle low |

### Cycle 50 — BUY (started 2024-01-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 09:15:00 | 207.80 | 199.33 | 198.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-11 10:15:00 | 213.10 | 202.08 | 200.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-12 14:15:00 | 212.10 | 213.50 | 209.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-12 15:00:00 | 212.10 | 213.50 | 209.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 213.30 | 213.73 | 210.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 10:00:00 | 213.30 | 213.73 | 210.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 217.35 | 222.79 | 219.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:00:00 | 217.35 | 222.79 | 219.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 13:15:00 | 220.00 | 222.23 | 219.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:30:00 | 217.90 | 222.23 | 219.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 14:15:00 | 219.70 | 221.72 | 219.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 14:30:00 | 220.00 | 221.72 | 219.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 15:15:00 | 219.50 | 221.28 | 219.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 09:15:00 | 218.35 | 221.28 | 219.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 225.90 | 222.20 | 219.87 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 09:15:00 | 213.40 | 218.37 | 218.89 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-01-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 12:15:00 | 221.70 | 219.26 | 219.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-18 13:15:00 | 223.70 | 220.15 | 219.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 09:15:00 | 231.60 | 238.69 | 233.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 231.60 | 238.69 | 233.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 231.60 | 238.69 | 233.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:00:00 | 231.60 | 238.69 | 233.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 230.95 | 237.14 | 233.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:30:00 | 229.70 | 237.14 | 233.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2024-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 13:15:00 | 223.60 | 230.82 | 231.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 14:15:00 | 221.80 | 229.02 | 230.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 09:15:00 | 228.50 | 228.02 | 229.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 09:15:00 | 228.50 | 228.02 | 229.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 228.50 | 228.02 | 229.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 09:30:00 | 228.80 | 228.02 | 229.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 10:15:00 | 239.45 | 230.31 | 230.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 11:00:00 | 239.45 | 230.31 | 230.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 11:15:00 | 230.75 | 230.40 | 230.59 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 12:15:00 | 234.75 | 231.27 | 230.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 13:15:00 | 235.75 | 232.16 | 231.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-25 10:15:00 | 232.30 | 232.80 | 232.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 10:15:00 | 232.30 | 232.80 | 232.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 232.30 | 232.80 | 232.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 09:15:00 | 236.45 | 232.66 | 232.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 15:15:00 | 235.60 | 234.93 | 233.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-30 15:15:00 | 233.20 | 233.97 | 233.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2024-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 15:15:00 | 233.20 | 233.97 | 233.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-31 11:15:00 | 231.85 | 233.12 | 233.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 15:15:00 | 236.65 | 233.30 | 233.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 15:15:00 | 236.65 | 233.30 | 233.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 15:15:00 | 236.65 | 233.30 | 233.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-01 09:15:00 | 237.30 | 233.30 | 233.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2024-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-01 09:15:00 | 237.05 | 234.05 | 233.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-02 09:15:00 | 247.75 | 237.58 | 235.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-02 15:15:00 | 251.85 | 252.55 | 245.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-05 09:15:00 | 251.00 | 252.55 | 245.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 09:15:00 | 249.85 | 252.01 | 245.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-05 13:45:00 | 255.05 | 251.15 | 247.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-05 15:15:00 | 243.85 | 248.78 | 246.75 | SL hit (close<static) qty=1.00 sl=245.40 alert=retest2 |

### Cycle 57 — SELL (started 2024-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-06 11:15:00 | 240.90 | 245.01 | 245.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-06 12:15:00 | 240.05 | 244.02 | 244.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 09:15:00 | 247.45 | 243.38 | 244.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 09:15:00 | 247.45 | 243.38 | 244.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 247.45 | 243.38 | 244.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 09:30:00 | 249.00 | 243.38 | 244.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 10:15:00 | 245.00 | 243.70 | 244.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 11:15:00 | 244.50 | 243.70 | 244.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 12:45:00 | 243.55 | 243.72 | 244.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 14:00:00 | 244.00 | 243.78 | 244.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 09:30:00 | 243.00 | 242.65 | 243.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 11:15:00 | 243.30 | 242.37 | 243.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 12:00:00 | 243.30 | 242.37 | 243.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 12:15:00 | 241.70 | 242.24 | 243.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 15:15:00 | 240.25 | 241.86 | 242.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 09:15:00 | 232.27 | 239.62 | 241.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 09:15:00 | 231.37 | 239.62 | 241.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 09:15:00 | 231.80 | 239.62 | 241.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 09:15:00 | 230.85 | 239.62 | 241.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 09:15:00 | 228.24 | 239.62 | 241.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-02-12 09:15:00 | 220.05 | 227.00 | 233.21 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 58 — BUY (started 2024-02-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 10:15:00 | 217.20 | 214.64 | 214.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 09:15:00 | 221.00 | 218.08 | 216.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 10:15:00 | 226.85 | 228.06 | 224.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-20 10:45:00 | 226.85 | 228.06 | 224.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 13:15:00 | 224.80 | 226.96 | 224.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 14:00:00 | 224.80 | 226.96 | 224.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 14:15:00 | 225.50 | 226.67 | 224.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 09:45:00 | 227.70 | 227.09 | 225.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 10:45:00 | 226.95 | 227.26 | 225.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 13:15:00 | 222.00 | 226.14 | 225.43 | SL hit (close<static) qty=1.00 sl=224.40 alert=retest2 |

### Cycle 59 — SELL (started 2024-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 14:15:00 | 219.95 | 224.90 | 224.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 09:15:00 | 217.50 | 222.76 | 223.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 15:15:00 | 219.00 | 218.73 | 220.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-23 09:15:00 | 225.20 | 218.73 | 220.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 220.65 | 219.11 | 220.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-26 09:15:00 | 217.35 | 220.34 | 220.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 11:15:00 | 206.48 | 212.40 | 215.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-29 12:15:00 | 210.10 | 207.92 | 210.58 | SL hit (close>ema200) qty=0.50 sl=207.92 alert=retest2 |

### Cycle 60 — BUY (started 2024-03-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 13:15:00 | 213.00 | 211.52 | 211.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 14:15:00 | 215.90 | 212.40 | 211.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 225.90 | 230.23 | 225.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 09:15:00 | 225.90 | 230.23 | 225.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 225.90 | 230.23 | 225.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 225.90 | 230.23 | 225.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 222.50 | 228.68 | 225.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:30:00 | 223.05 | 228.68 | 225.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 11:15:00 | 221.65 | 227.28 | 224.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 11:30:00 | 221.40 | 227.28 | 224.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 224.75 | 225.85 | 224.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 14:45:00 | 224.60 | 225.85 | 224.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 15:15:00 | 224.80 | 225.64 | 224.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-07 09:15:00 | 225.50 | 225.64 | 224.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-07 09:45:00 | 226.85 | 225.98 | 224.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-07 10:45:00 | 225.90 | 226.17 | 225.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-11 09:15:00 | 218.95 | 224.67 | 224.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2024-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 09:15:00 | 218.95 | 224.67 | 224.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 11:15:00 | 218.65 | 222.84 | 223.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-12 15:15:00 | 212.90 | 212.58 | 216.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-13 09:15:00 | 208.75 | 212.58 | 216.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 189.15 | 194.08 | 198.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 10:15:00 | 186.60 | 194.08 | 198.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 10:45:00 | 186.35 | 192.34 | 197.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 10:15:00 | 186.80 | 189.32 | 193.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 13:15:00 | 186.70 | 187.45 | 191.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 09:15:00 | 177.27 | 181.31 | 185.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 09:15:00 | 177.03 | 181.31 | 185.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 09:15:00 | 177.46 | 181.31 | 185.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 09:15:00 | 177.36 | 181.31 | 185.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 182.65 | 178.16 | 181.17 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-03-21 09:15:00 | 182.65 | 178.16 | 181.17 | SL hit (close>ema200) qty=0.50 sl=178.16 alert=retest2 |

### Cycle 62 — BUY (started 2024-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 12:15:00 | 194.40 | 184.55 | 183.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 13:15:00 | 196.10 | 186.86 | 184.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 13:15:00 | 205.00 | 205.31 | 201.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-27 14:15:00 | 203.75 | 205.31 | 201.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 200.60 | 204.37 | 201.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 15:00:00 | 200.60 | 204.37 | 201.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 200.25 | 203.55 | 201.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 09:15:00 | 204.25 | 203.55 | 201.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-04 09:15:00 | 224.68 | 219.27 | 215.65 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-04-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 09:15:00 | 220.40 | 222.80 | 222.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 14:15:00 | 218.10 | 220.49 | 221.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 09:15:00 | 214.05 | 212.51 | 214.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 09:15:00 | 214.05 | 212.51 | 214.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 214.05 | 212.51 | 214.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 12:15:00 | 212.85 | 213.15 | 214.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-19 09:15:00 | 202.21 | 209.20 | 212.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-19 13:15:00 | 206.90 | 206.54 | 209.63 | SL hit (close>ema200) qty=0.50 sl=206.54 alert=retest2 |

### Cycle 64 — BUY (started 2024-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 14:15:00 | 212.30 | 209.38 | 209.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 15:15:00 | 216.00 | 210.71 | 209.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-02 14:15:00 | 245.00 | 245.24 | 241.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-02 15:00:00 | 245.00 | 245.24 | 241.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 09:15:00 | 242.25 | 244.61 | 241.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 10:00:00 | 242.25 | 244.61 | 241.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 10:15:00 | 242.70 | 244.22 | 241.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 10:45:00 | 242.70 | 244.22 | 241.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 11:15:00 | 240.00 | 243.38 | 241.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 11:30:00 | 240.70 | 243.38 | 241.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 12:15:00 | 241.45 | 242.99 | 241.51 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 09:15:00 | 232.20 | 240.29 | 240.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 11:15:00 | 226.10 | 232.69 | 235.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 11:15:00 | 228.55 | 227.59 | 230.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 12:00:00 | 228.55 | 227.59 | 230.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 222.65 | 218.23 | 221.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 14:00:00 | 222.65 | 218.23 | 221.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 222.40 | 219.06 | 221.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 15:00:00 | 222.40 | 219.06 | 221.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 221.45 | 219.54 | 221.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:15:00 | 220.00 | 219.54 | 221.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 218.25 | 219.28 | 221.13 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 13:15:00 | 226.90 | 222.74 | 222.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 09:15:00 | 235.00 | 225.53 | 223.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 09:15:00 | 266.90 | 270.07 | 264.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 09:15:00 | 266.90 | 270.07 | 264.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 266.90 | 270.07 | 264.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 266.80 | 270.07 | 264.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 265.65 | 268.65 | 265.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:30:00 | 265.55 | 268.65 | 265.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 264.70 | 267.86 | 265.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:00:00 | 264.70 | 267.86 | 265.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 269.00 | 268.09 | 265.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 14:15:00 | 269.40 | 268.09 | 265.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 15:15:00 | 270.00 | 268.10 | 265.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 14:45:00 | 269.10 | 275.69 | 274.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 15:15:00 | 269.70 | 275.69 | 274.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 09:15:00 | 265.35 | 272.66 | 273.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 09:15:00 | 265.35 | 272.66 | 273.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 09:15:00 | 260.60 | 268.31 | 270.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 14:15:00 | 266.45 | 265.04 | 267.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 14:15:00 | 266.45 | 265.04 | 267.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 266.45 | 265.04 | 267.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 15:00:00 | 266.45 | 265.04 | 267.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 266.30 | 253.18 | 254.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:45:00 | 265.45 | 253.18 | 254.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 266.00 | 255.74 | 255.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 269.35 | 260.23 | 257.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 261.40 | 263.65 | 260.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 261.40 | 263.65 | 260.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 261.40 | 263.65 | 260.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 259.30 | 263.65 | 260.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 237.30 | 258.38 | 258.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 237.30 | 258.38 | 258.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 215.15 | 249.73 | 254.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 196.70 | 228.12 | 241.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 234.60 | 220.64 | 229.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 234.60 | 220.64 | 229.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 234.60 | 220.64 | 229.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 234.30 | 220.64 | 229.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 236.85 | 223.88 | 230.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 236.50 | 223.88 | 230.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 12:15:00 | 226.00 | 224.98 | 229.48 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-06-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 14:15:00 | 232.55 | 230.05 | 230.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 09:15:00 | 241.01 | 233.11 | 231.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 10:15:00 | 252.40 | 252.84 | 247.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 11:00:00 | 252.40 | 252.84 | 247.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 264.20 | 267.46 | 262.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 265.49 | 267.46 | 262.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 262.59 | 266.48 | 262.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 12:15:00 | 265.05 | 266.04 | 262.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 13:45:00 | 267.55 | 265.78 | 263.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 10:15:00 | 266.00 | 264.87 | 263.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 11:30:00 | 268.03 | 264.79 | 263.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 12:15:00 | 262.56 | 264.35 | 263.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 13:00:00 | 262.56 | 264.35 | 263.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 259.26 | 263.33 | 263.14 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-20 13:15:00 | 259.26 | 263.33 | 263.14 | SL hit (close<static) qty=1.00 sl=260.00 alert=retest2 |

### Cycle 71 — SELL (started 2024-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 14:15:00 | 257.74 | 262.21 | 262.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 255.70 | 258.70 | 260.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 10:15:00 | 255.48 | 253.63 | 256.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 10:15:00 | 255.48 | 253.63 | 256.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 255.48 | 253.63 | 256.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:45:00 | 256.72 | 253.63 | 256.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 254.35 | 253.74 | 255.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 13:45:00 | 255.30 | 253.74 | 255.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 250.85 | 249.98 | 252.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:30:00 | 251.98 | 249.98 | 252.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 251.49 | 249.08 | 250.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 251.49 | 249.08 | 250.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 252.00 | 249.66 | 250.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 252.20 | 249.66 | 250.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 252.27 | 250.46 | 250.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 12:00:00 | 251.25 | 250.62 | 251.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 13:15:00 | 254.05 | 251.06 | 250.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 13:15:00 | 254.05 | 251.06 | 250.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 14:15:00 | 256.20 | 252.09 | 251.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 14:15:00 | 257.55 | 257.59 | 255.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 15:00:00 | 257.55 | 257.59 | 255.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 280.40 | 271.30 | 267.76 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2024-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 14:15:00 | 263.90 | 268.06 | 268.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 258.35 | 265.54 | 267.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 15:15:00 | 261.90 | 261.58 | 264.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-11 09:15:00 | 262.80 | 261.58 | 264.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 272.40 | 263.75 | 264.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:00:00 | 272.40 | 263.75 | 264.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 10:15:00 | 280.25 | 267.05 | 266.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 12:15:00 | 293.85 | 274.26 | 269.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 14:15:00 | 288.00 | 289.71 | 282.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-12 15:00:00 | 288.00 | 289.71 | 282.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 282.50 | 288.01 | 283.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 281.00 | 288.01 | 283.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 282.65 | 286.94 | 283.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:30:00 | 282.80 | 286.94 | 283.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 282.00 | 285.95 | 283.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 12:15:00 | 281.20 | 285.95 | 283.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 281.80 | 285.12 | 282.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:30:00 | 284.15 | 284.67 | 283.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 14:45:00 | 283.30 | 284.46 | 283.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 15:15:00 | 280.20 | 283.61 | 283.49 | SL hit (close<static) qty=1.00 sl=281.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 273.50 | 281.59 | 282.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 262.90 | 270.83 | 275.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 12:15:00 | 270.35 | 269.27 | 273.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-19 12:45:00 | 270.20 | 269.27 | 273.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 268.20 | 267.18 | 270.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 269.40 | 267.18 | 270.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 271.70 | 268.30 | 270.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:00:00 | 271.70 | 268.30 | 270.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 268.80 | 268.40 | 270.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 14:45:00 | 267.25 | 268.32 | 270.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:30:00 | 266.50 | 267.68 | 269.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 253.89 | 263.67 | 267.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 253.17 | 263.67 | 267.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-25 10:15:00 | 259.90 | 257.84 | 260.81 | SL hit (close>ema200) qty=0.50 sl=257.84 alert=retest2 |

### Cycle 76 — BUY (started 2024-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 11:15:00 | 262.40 | 258.48 | 258.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 09:15:00 | 267.75 | 262.56 | 260.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 09:15:00 | 261.95 | 263.99 | 262.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 09:15:00 | 261.95 | 263.99 | 262.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 261.95 | 263.99 | 262.56 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 11:15:00 | 260.70 | 262.42 | 262.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 12:15:00 | 260.45 | 262.03 | 262.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 14:15:00 | 233.10 | 232.03 | 237.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 14:45:00 | 231.80 | 232.03 | 237.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 227.20 | 223.45 | 226.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 15:00:00 | 227.20 | 223.45 | 226.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 15:15:00 | 226.65 | 224.09 | 226.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:15:00 | 228.90 | 224.09 | 226.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 223.65 | 224.00 | 226.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 10:15:00 | 222.95 | 224.00 | 226.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 12:15:00 | 223.05 | 223.89 | 225.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 09:30:00 | 222.80 | 216.42 | 217.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 12:15:00 | 219.40 | 218.06 | 218.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 12:15:00 | 219.40 | 218.06 | 218.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 14:15:00 | 220.95 | 218.81 | 218.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 09:15:00 | 218.70 | 219.16 | 218.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 09:15:00 | 218.70 | 219.16 | 218.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 218.70 | 219.16 | 218.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:00:00 | 218.70 | 219.16 | 218.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 218.95 | 219.12 | 218.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 15:15:00 | 219.80 | 218.94 | 218.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 14:00:00 | 219.70 | 219.14 | 218.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 15:15:00 | 219.20 | 219.10 | 218.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 09:15:00 | 216.75 | 218.65 | 218.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 09:15:00 | 216.75 | 218.65 | 218.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 11:15:00 | 215.80 | 217.75 | 218.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 09:15:00 | 216.80 | 216.66 | 217.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 216.80 | 216.66 | 217.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 216.80 | 216.66 | 217.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 13:30:00 | 214.70 | 215.95 | 216.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 10:00:00 | 214.25 | 215.49 | 216.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 12:45:00 | 214.75 | 214.56 | 215.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 14:45:00 | 214.60 | 214.78 | 215.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 215.50 | 214.93 | 215.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:15:00 | 217.45 | 214.93 | 215.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-28 09:15:00 | 222.25 | 216.39 | 215.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 09:15:00 | 222.25 | 216.39 | 215.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 10:15:00 | 224.75 | 218.06 | 216.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 13:15:00 | 222.95 | 223.75 | 221.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-29 14:00:00 | 222.95 | 223.75 | 221.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 224.15 | 223.83 | 221.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 14:30:00 | 222.25 | 223.83 | 221.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 222.00 | 223.54 | 222.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:45:00 | 222.50 | 223.54 | 222.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 222.00 | 223.23 | 222.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:30:00 | 222.05 | 223.23 | 222.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 220.00 | 222.58 | 221.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 12:00:00 | 220.00 | 222.58 | 221.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 12:15:00 | 220.00 | 222.07 | 221.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 12:30:00 | 219.65 | 222.07 | 221.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 14:15:00 | 217.45 | 220.78 | 221.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 09:15:00 | 216.99 | 219.56 | 220.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 11:15:00 | 222.90 | 220.12 | 220.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 11:15:00 | 222.90 | 220.12 | 220.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 222.90 | 220.12 | 220.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:30:00 | 222.88 | 220.12 | 220.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 222.32 | 220.56 | 220.76 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 13:15:00 | 222.50 | 220.95 | 220.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 14:15:00 | 223.27 | 221.41 | 221.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 221.99 | 222.67 | 222.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 14:15:00 | 221.99 | 222.67 | 222.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 221.99 | 222.67 | 222.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 221.99 | 222.67 | 222.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 222.61 | 222.66 | 222.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 218.64 | 222.66 | 222.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 219.23 | 221.97 | 221.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:30:00 | 218.10 | 221.97 | 221.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 217.37 | 221.05 | 221.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 14:15:00 | 216.90 | 217.90 | 219.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 13:15:00 | 216.55 | 216.41 | 217.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-06 13:45:00 | 216.52 | 216.41 | 217.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 218.38 | 216.92 | 217.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 213.35 | 216.92 | 217.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 09:15:00 | 216.78 | 212.18 | 212.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 09:15:00 | 216.78 | 212.18 | 212.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 10:15:00 | 219.77 | 213.70 | 212.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 12:15:00 | 222.88 | 223.54 | 219.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 12:45:00 | 222.60 | 223.54 | 219.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 219.51 | 222.57 | 220.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:00:00 | 219.51 | 222.57 | 220.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 218.54 | 221.76 | 220.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:00:00 | 218.54 | 221.76 | 220.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 13:15:00 | 216.95 | 219.66 | 219.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 14:15:00 | 215.70 | 218.87 | 219.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 206.19 | 205.75 | 208.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 206.19 | 205.75 | 208.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 206.19 | 205.75 | 208.66 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2024-09-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 15:15:00 | 211.65 | 209.74 | 209.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 13:15:00 | 217.32 | 212.13 | 210.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 217.16 | 217.81 | 215.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 10:00:00 | 217.16 | 217.81 | 215.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 215.24 | 217.30 | 215.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:00:00 | 215.24 | 217.30 | 215.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 214.27 | 216.69 | 215.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:45:00 | 214.19 | 216.69 | 215.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 214.77 | 216.31 | 215.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 13:15:00 | 215.15 | 216.31 | 215.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 10:00:00 | 215.46 | 216.51 | 215.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 11:15:00 | 213.41 | 215.63 | 215.55 | SL hit (close<static) qty=1.00 sl=213.49 alert=retest2 |

### Cycle 87 — SELL (started 2024-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 12:15:00 | 212.36 | 214.97 | 215.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 13:15:00 | 211.77 | 214.33 | 214.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 10:15:00 | 212.34 | 211.88 | 212.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 10:15:00 | 212.34 | 211.88 | 212.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 212.34 | 211.88 | 212.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:45:00 | 212.17 | 211.88 | 212.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 212.39 | 211.98 | 212.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:00:00 | 212.39 | 211.98 | 212.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 212.05 | 212.00 | 212.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:30:00 | 212.32 | 212.00 | 212.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 212.05 | 212.01 | 212.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:00:00 | 212.05 | 212.01 | 212.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 209.30 | 211.33 | 212.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 10:15:00 | 209.03 | 211.33 | 212.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 11:00:00 | 208.90 | 210.84 | 211.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 12:00:00 | 209.00 | 210.47 | 211.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 13:30:00 | 209.11 | 209.98 | 211.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 198.58 | 205.29 | 207.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 198.45 | 205.29 | 207.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 198.55 | 205.29 | 207.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 198.65 | 205.29 | 207.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-04 10:15:00 | 206.80 | 205.59 | 207.55 | SL hit (close>ema200) qty=0.50 sl=205.59 alert=retest2 |

### Cycle 88 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 205.30 | 200.24 | 199.89 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 199.86 | 201.00 | 201.04 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 09:15:00 | 202.66 | 201.29 | 201.16 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 14:15:00 | 200.62 | 201.05 | 201.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 10:15:00 | 199.50 | 200.53 | 200.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 09:15:00 | 196.52 | 195.58 | 197.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 09:15:00 | 196.52 | 195.58 | 197.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 196.52 | 195.58 | 197.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:45:00 | 198.05 | 195.58 | 197.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 192.59 | 193.68 | 195.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 11:15:00 | 191.95 | 193.44 | 194.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 12:30:00 | 191.36 | 192.62 | 194.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 182.35 | 185.52 | 187.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 181.79 | 185.52 | 187.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-23 09:15:00 | 172.75 | 180.20 | 183.21 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 92 — BUY (started 2024-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 15:15:00 | 180.25 | 178.95 | 178.95 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 177.77 | 178.82 | 178.89 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 180.42 | 179.23 | 179.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 10:15:00 | 183.95 | 180.52 | 179.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 191.08 | 193.03 | 189.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 10:00:00 | 191.08 | 193.03 | 189.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 193.61 | 193.18 | 191.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 14:00:00 | 194.39 | 193.37 | 191.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 194.24 | 192.83 | 191.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 11:00:00 | 194.41 | 193.35 | 192.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 10:15:00 | 190.42 | 193.53 | 193.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 190.42 | 193.53 | 193.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 188.64 | 192.55 | 193.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 09:15:00 | 190.29 | 190.08 | 191.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 09:15:00 | 190.29 | 190.08 | 191.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 190.29 | 190.08 | 191.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 10:00:00 | 190.29 | 190.08 | 191.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 193.48 | 190.76 | 191.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 193.48 | 190.76 | 191.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 189.70 | 190.55 | 191.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 13:15:00 | 189.31 | 190.32 | 191.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 14:00:00 | 189.37 | 190.13 | 191.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 15:15:00 | 189.15 | 190.06 | 191.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:45:00 | 189.00 | 190.04 | 190.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 14:15:00 | 179.84 | 183.28 | 186.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 14:15:00 | 179.90 | 183.28 | 186.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 14:15:00 | 179.69 | 183.28 | 186.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 14:15:00 | 179.55 | 183.28 | 186.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 183.25 | 182.67 | 185.44 | SL hit (close>ema200) qty=0.50 sl=182.67 alert=retest2 |

### Cycle 96 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 183.87 | 182.58 | 182.55 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 181.30 | 182.42 | 182.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 180.55 | 182.05 | 182.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 179.19 | 179.03 | 180.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 179.19 | 179.03 | 180.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 179.19 | 179.03 | 180.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:30:00 | 178.71 | 179.03 | 180.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 180.92 | 179.41 | 180.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:00:00 | 180.92 | 179.41 | 180.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 180.41 | 179.61 | 180.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 12:45:00 | 179.99 | 179.59 | 180.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 15:15:00 | 180.20 | 179.86 | 180.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 185.74 | 181.09 | 180.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 185.74 | 181.09 | 180.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 09:15:00 | 193.66 | 185.49 | 183.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 09:15:00 | 200.00 | 200.28 | 197.86 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 11:00:00 | 202.75 | 200.77 | 198.31 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 198.88 | 200.13 | 198.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 14:00:00 | 198.88 | 200.13 | 198.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 198.63 | 199.83 | 198.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:15:00 | 198.24 | 199.83 | 198.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 198.24 | 199.51 | 198.58 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-02 15:15:00 | 198.24 | 199.51 | 198.58 | SL hit (close<ema400) qty=1.00 sl=198.58 alert=retest1 |

### Cycle 99 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 204.80 | 206.34 | 206.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 11:15:00 | 202.95 | 205.66 | 206.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 202.13 | 201.89 | 203.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 15:00:00 | 202.13 | 201.89 | 203.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 202.81 | 202.06 | 203.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 15:00:00 | 201.82 | 202.11 | 202.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:15:00 | 201.00 | 202.00 | 202.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 11:45:00 | 201.78 | 202.05 | 202.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 13:15:00 | 201.52 | 202.07 | 202.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 14:15:00 | 191.73 | 196.61 | 198.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 14:15:00 | 191.69 | 196.61 | 198.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 190.95 | 194.98 | 197.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 191.44 | 194.98 | 197.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-19 15:15:00 | 193.95 | 193.13 | 195.43 | SL hit (close>ema200) qty=0.50 sl=193.13 alert=retest2 |

### Cycle 100 — BUY (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 13:15:00 | 189.88 | 187.45 | 187.23 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 10:15:00 | 184.93 | 187.12 | 187.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 11:15:00 | 183.62 | 186.42 | 186.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 10:15:00 | 182.32 | 181.82 | 183.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 11:00:00 | 182.32 | 181.82 | 183.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 184.79 | 182.44 | 183.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:00:00 | 184.79 | 182.44 | 183.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 184.34 | 182.82 | 183.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:45:00 | 184.97 | 182.82 | 183.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2025-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 14:15:00 | 184.25 | 183.72 | 183.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 185.58 | 184.20 | 183.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 10:15:00 | 182.15 | 183.79 | 183.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 10:15:00 | 182.15 | 183.79 | 183.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 182.15 | 183.79 | 183.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:00:00 | 182.15 | 183.79 | 183.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 11:15:00 | 182.25 | 183.48 | 183.62 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 09:15:00 | 187.62 | 183.95 | 183.72 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 180.99 | 184.40 | 184.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 179.19 | 182.69 | 183.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 182.39 | 182.31 | 183.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 182.39 | 182.31 | 183.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 182.39 | 182.31 | 183.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 183.76 | 182.31 | 183.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 182.41 | 182.12 | 182.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:30:00 | 182.45 | 182.12 | 182.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 181.98 | 181.99 | 182.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:30:00 | 180.64 | 181.63 | 182.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 13:00:00 | 180.59 | 181.63 | 182.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:30:00 | 180.40 | 181.57 | 182.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:45:00 | 180.36 | 181.35 | 181.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 171.61 | 178.00 | 179.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 171.56 | 178.00 | 179.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 171.38 | 178.00 | 179.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 171.34 | 178.00 | 179.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 13:15:00 | 162.58 | 169.30 | 173.50 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 106 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 176.30 | 171.69 | 171.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 14:15:00 | 177.15 | 173.97 | 172.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 15:15:00 | 176.01 | 176.59 | 175.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 09:15:00 | 180.03 | 176.59 | 175.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 181.50 | 177.57 | 175.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 09:30:00 | 182.08 | 179.94 | 178.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 15:15:00 | 176.15 | 177.86 | 178.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2025-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 15:15:00 | 176.15 | 177.86 | 178.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 172.79 | 176.84 | 177.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 174.19 | 173.03 | 174.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 174.19 | 173.03 | 174.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 174.19 | 173.03 | 174.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 174.19 | 173.03 | 174.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 174.40 | 173.30 | 174.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 174.81 | 173.30 | 174.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 173.80 | 173.40 | 174.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 13:30:00 | 173.00 | 173.27 | 174.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:00:00 | 172.60 | 173.27 | 174.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 164.35 | 167.13 | 170.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 163.97 | 167.13 | 170.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 09:15:00 | 155.70 | 160.87 | 164.97 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 108 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 166.60 | 163.02 | 162.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 15:15:00 | 168.00 | 165.43 | 164.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 168.76 | 172.46 | 170.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 168.76 | 172.46 | 170.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 168.76 | 172.46 | 170.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 170.83 | 172.46 | 170.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 170.74 | 172.11 | 170.23 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 161.30 | 168.95 | 169.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 160.47 | 166.03 | 167.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 164.79 | 163.99 | 165.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 164.79 | 163.99 | 165.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 164.79 | 163.99 | 165.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 10:45:00 | 163.65 | 163.96 | 165.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 14:15:00 | 167.10 | 164.84 | 165.55 | SL hit (close>static) qty=1.00 sl=166.43 alert=retest2 |

### Cycle 110 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 169.70 | 166.09 | 166.01 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 11:15:00 | 165.55 | 167.04 | 167.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 164.51 | 166.30 | 166.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 12:15:00 | 166.89 | 164.14 | 165.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 12:15:00 | 166.89 | 164.14 | 165.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 12:15:00 | 166.89 | 164.14 | 165.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 13:00:00 | 166.89 | 164.14 | 165.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 13:15:00 | 166.30 | 164.57 | 165.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 14:15:00 | 170.00 | 164.57 | 165.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 169.90 | 165.64 | 165.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 15:00:00 | 169.90 | 165.64 | 165.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2025-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 15:15:00 | 167.70 | 166.05 | 165.96 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 163.35 | 165.51 | 165.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 11:15:00 | 162.55 | 164.63 | 165.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 160.00 | 159.18 | 161.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:00:00 | 160.00 | 159.18 | 161.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 161.82 | 159.71 | 161.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 161.82 | 159.71 | 161.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 162.21 | 160.21 | 161.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 14:00:00 | 162.21 | 160.21 | 161.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 169.91 | 162.15 | 162.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 169.91 | 162.15 | 162.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2025-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 15:15:00 | 170.00 | 163.72 | 163.24 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 160.74 | 164.37 | 164.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 12:15:00 | 157.89 | 162.42 | 163.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 11:15:00 | 156.12 | 156.04 | 158.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-18 11:45:00 | 156.00 | 156.04 | 158.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 13:15:00 | 157.73 | 156.36 | 158.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 14:00:00 | 157.73 | 156.36 | 158.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 158.14 | 156.71 | 158.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 14:30:00 | 158.32 | 156.71 | 158.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 15:15:00 | 156.60 | 156.69 | 157.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:15:00 | 160.51 | 156.69 | 157.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 162.30 | 157.81 | 158.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 162.30 | 157.81 | 158.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 160.88 | 158.43 | 158.59 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 161.90 | 159.12 | 158.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 12:15:00 | 165.41 | 160.38 | 159.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 11:15:00 | 171.86 | 171.87 | 168.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 15:15:00 | 168.50 | 170.74 | 168.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 168.50 | 170.74 | 168.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 165.92 | 170.74 | 168.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 166.23 | 169.83 | 168.69 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 12:15:00 | 165.45 | 167.81 | 167.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 14:15:00 | 164.99 | 166.85 | 167.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 169.30 | 166.93 | 167.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 169.30 | 166.93 | 167.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 169.30 | 166.93 | 167.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:45:00 | 169.10 | 166.93 | 167.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 167.82 | 167.11 | 167.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 11:15:00 | 167.01 | 167.11 | 167.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 12:15:00 | 169.40 | 167.73 | 167.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2025-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 12:15:00 | 169.40 | 167.73 | 167.65 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2025-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 14:15:00 | 165.99 | 167.42 | 167.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 15:15:00 | 165.25 | 166.99 | 167.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 148.67 | 148.21 | 152.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 09:45:00 | 149.12 | 148.21 | 152.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 150.16 | 147.99 | 149.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:30:00 | 152.52 | 147.99 | 149.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 150.45 | 148.48 | 150.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 150.45 | 148.48 | 150.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 151.09 | 149.00 | 150.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:45:00 | 150.87 | 149.00 | 150.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 155.10 | 150.22 | 150.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:45:00 | 155.07 | 150.22 | 150.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 157.51 | 151.68 | 151.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 158.60 | 153.06 | 151.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 161.33 | 161.39 | 158.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 161.33 | 161.39 | 158.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 160.14 | 161.14 | 159.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 159.80 | 161.14 | 159.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 158.68 | 160.24 | 159.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:00:00 | 158.68 | 160.24 | 159.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 157.68 | 159.73 | 159.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:00:00 | 157.68 | 159.73 | 159.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 152.89 | 157.54 | 158.15 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2025-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 14:15:00 | 161.05 | 158.61 | 158.37 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 11:15:00 | 156.73 | 158.19 | 158.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 15:15:00 | 156.00 | 157.13 | 157.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 161.68 | 158.04 | 158.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 161.68 | 158.04 | 158.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 161.68 | 158.04 | 158.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:45:00 | 163.34 | 158.04 | 158.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 10:15:00 | 160.47 | 158.53 | 158.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 14:15:00 | 161.75 | 159.46 | 158.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-17 10:15:00 | 159.28 | 159.66 | 159.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 10:15:00 | 159.28 | 159.66 | 159.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 159.28 | 159.66 | 159.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:00:00 | 159.28 | 159.66 | 159.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 160.04 | 159.74 | 159.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 12:15:00 | 159.31 | 159.74 | 159.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 158.98 | 159.58 | 159.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:00:00 | 158.98 | 159.58 | 159.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 159.20 | 159.51 | 159.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:30:00 | 159.49 | 159.51 | 159.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 159.16 | 159.44 | 159.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:30:00 | 159.00 | 159.44 | 159.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 159.00 | 159.35 | 159.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 09:15:00 | 161.10 | 159.35 | 159.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 14:15:00 | 163.13 | 164.80 | 164.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 14:15:00 | 163.13 | 164.80 | 164.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 15:15:00 | 162.70 | 164.38 | 164.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 12:15:00 | 158.34 | 158.22 | 160.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 13:00:00 | 158.34 | 158.22 | 160.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 163.40 | 159.14 | 159.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:45:00 | 163.80 | 159.14 | 159.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 162.35 | 159.78 | 160.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 11:00:00 | 162.35 | 159.78 | 160.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 12:15:00 | 161.90 | 160.53 | 160.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 12:15:00 | 162.37 | 161.09 | 160.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 09:15:00 | 161.10 | 161.61 | 161.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 161.10 | 161.61 | 161.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 161.10 | 161.61 | 161.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 163.60 | 162.05 | 161.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 10:15:00 | 163.40 | 162.25 | 161.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 10:45:00 | 164.00 | 164.28 | 163.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 12:15:00 | 163.32 | 163.99 | 163.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 161.98 | 163.59 | 163.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:00:00 | 161.98 | 163.59 | 163.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 13:15:00 | 161.90 | 163.25 | 163.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:45:00 | 162.00 | 163.25 | 163.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 163.08 | 163.22 | 163.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 14:30:00 | 162.02 | 163.22 | 163.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 15:15:00 | 162.70 | 163.11 | 163.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-07 09:15:00 | 153.86 | 163.11 | 163.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 153.13 | 161.12 | 162.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 153.13 | 161.12 | 162.18 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2025-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 13:15:00 | 164.30 | 160.67 | 160.41 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 13:15:00 | 159.06 | 160.77 | 160.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 14:15:00 | 158.99 | 160.41 | 160.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 09:15:00 | 160.85 | 160.31 | 160.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 09:15:00 | 160.85 | 160.31 | 160.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 160.85 | 160.31 | 160.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 11:15:00 | 160.40 | 160.34 | 160.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 12:30:00 | 160.40 | 160.39 | 160.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 13:15:00 | 160.38 | 160.39 | 160.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 164.16 | 160.97 | 160.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 164.16 | 160.97 | 160.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 10:15:00 | 166.49 | 162.08 | 161.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 181.14 | 181.42 | 176.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 15:00:00 | 181.14 | 181.42 | 176.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 183.19 | 180.83 | 179.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 186.95 | 183.23 | 181.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 14:45:00 | 185.15 | 183.61 | 182.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 09:30:00 | 185.12 | 184.00 | 182.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 10:30:00 | 185.77 | 184.33 | 182.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 182.80 | 183.88 | 183.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:30:00 | 182.97 | 183.88 | 183.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 182.75 | 183.65 | 183.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 15:00:00 | 182.75 | 183.65 | 183.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 183.29 | 183.58 | 183.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:15:00 | 181.40 | 183.58 | 183.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 178.29 | 182.52 | 182.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 178.29 | 182.52 | 182.63 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 12:15:00 | 184.12 | 180.45 | 180.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 15:15:00 | 185.00 | 182.45 | 181.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 11:15:00 | 181.79 | 182.48 | 181.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 11:45:00 | 181.92 | 182.48 | 181.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 181.38 | 182.26 | 181.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 13:00:00 | 181.38 | 182.26 | 181.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 13:15:00 | 180.57 | 181.92 | 181.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 13:45:00 | 180.94 | 181.92 | 181.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 178.00 | 181.14 | 181.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 177.45 | 180.40 | 180.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 178.65 | 178.12 | 179.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 10:00:00 | 178.65 | 178.12 | 179.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 179.39 | 178.37 | 179.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:30:00 | 179.20 | 178.37 | 179.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 179.29 | 178.56 | 179.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 13:00:00 | 178.65 | 178.57 | 179.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 14:15:00 | 180.78 | 179.10 | 179.33 | SL hit (close>static) qty=1.00 sl=179.55 alert=retest2 |

### Cycle 134 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 178.69 | 176.81 | 176.78 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 174.69 | 176.47 | 176.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 14:15:00 | 173.04 | 175.78 | 176.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 11:15:00 | 174.10 | 173.74 | 175.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 12:00:00 | 174.10 | 173.74 | 175.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 175.19 | 174.06 | 174.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:00:00 | 175.19 | 174.06 | 174.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 176.00 | 174.45 | 174.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 182.14 | 174.45 | 174.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 181.53 | 175.86 | 175.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 183.85 | 177.46 | 176.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 183.30 | 183.45 | 181.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 183.30 | 183.45 | 181.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 192.00 | 193.62 | 191.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 192.00 | 193.62 | 191.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 192.50 | 193.40 | 191.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:45:00 | 192.19 | 193.40 | 191.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 192.10 | 193.14 | 191.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 189.93 | 193.14 | 191.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 192.06 | 192.92 | 191.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 191.45 | 192.92 | 191.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 192.94 | 192.93 | 192.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 11:30:00 | 195.03 | 193.36 | 192.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 195.02 | 195.04 | 193.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-29 14:15:00 | 214.53 | 208.36 | 206.49 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 15:15:00 | 229.10 | 230.25 | 230.38 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 233.06 | 230.81 | 230.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 12:15:00 | 241.50 | 234.83 | 233.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 10:15:00 | 238.40 | 238.74 | 236.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 229.40 | 236.61 | 235.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 229.40 | 236.61 | 235.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 229.40 | 236.61 | 235.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 230.30 | 235.35 | 235.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:30:00 | 230.24 | 235.35 | 235.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 229.70 | 234.22 | 234.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 227.42 | 232.86 | 234.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 224.99 | 224.59 | 227.13 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:15:00 | 221.45 | 224.59 | 227.13 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 226.94 | 224.30 | 225.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-17 09:15:00 | 226.94 | 224.30 | 225.30 | SL hit (close>ema400) qty=1.00 sl=225.30 alert=retest1 |

### Cycle 140 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 222.00 | 220.39 | 220.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 224.27 | 221.17 | 220.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 10:15:00 | 232.30 | 232.38 | 229.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 10:45:00 | 232.71 | 232.38 | 229.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 234.47 | 233.32 | 232.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:45:00 | 235.00 | 233.32 | 232.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 232.36 | 233.13 | 232.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:00:00 | 232.36 | 233.13 | 232.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 233.82 | 233.27 | 232.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:30:00 | 233.31 | 233.27 | 232.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 237.11 | 238.33 | 236.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 236.91 | 238.33 | 236.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 237.80 | 238.49 | 237.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:30:00 | 235.55 | 237.71 | 237.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 234.57 | 237.08 | 236.97 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 11:15:00 | 235.77 | 236.82 | 236.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 09:15:00 | 233.31 | 235.26 | 236.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 11:15:00 | 235.07 | 234.81 | 235.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 11:15:00 | 235.07 | 234.81 | 235.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 235.07 | 234.81 | 235.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:45:00 | 235.32 | 234.81 | 235.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 242.80 | 236.44 | 236.08 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 15:15:00 | 235.95 | 238.36 | 238.67 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 10:15:00 | 241.60 | 239.34 | 239.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 250.31 | 242.66 | 240.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 15:15:00 | 244.86 | 245.72 | 243.59 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 09:15:00 | 247.98 | 245.72 | 243.59 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 09:45:00 | 246.60 | 246.12 | 243.96 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 11:00:00 | 248.16 | 246.53 | 244.35 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 250.95 | 250.40 | 248.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 248.11 | 249.77 | 249.03 | SL hit (close<ema400) qty=1.00 sl=249.03 alert=retest1 |

### Cycle 145 — SELL (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 11:15:00 | 247.57 | 248.55 | 248.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 13:15:00 | 246.40 | 247.93 | 248.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 09:15:00 | 248.31 | 247.69 | 248.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 248.31 | 247.69 | 248.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 248.31 | 247.69 | 248.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 12:15:00 | 246.30 | 247.49 | 247.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 15:15:00 | 245.90 | 247.30 | 247.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 14:15:00 | 233.99 | 237.57 | 240.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 14:15:00 | 233.60 | 237.57 | 240.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-21 11:15:00 | 238.33 | 236.73 | 239.24 | SL hit (close>ema200) qty=0.50 sl=236.73 alert=retest2 |

### Cycle 146 — BUY (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 13:15:00 | 241.99 | 239.25 | 239.00 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 236.51 | 238.89 | 239.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 235.35 | 238.18 | 238.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 10:15:00 | 217.50 | 216.94 | 220.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-30 11:00:00 | 217.50 | 216.94 | 220.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 211.69 | 210.26 | 211.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 213.67 | 210.26 | 211.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 211.90 | 210.59 | 211.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:45:00 | 210.50 | 210.58 | 211.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 11:30:00 | 210.62 | 210.51 | 211.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 12:15:00 | 199.97 | 203.19 | 206.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 12:15:00 | 200.09 | 203.19 | 206.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 203.41 | 202.79 | 205.33 | SL hit (close>ema200) qty=0.50 sl=202.79 alert=retest2 |

### Cycle 148 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 198.01 | 195.30 | 194.98 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 193.32 | 194.94 | 195.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 11:15:00 | 193.22 | 194.60 | 194.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 11:15:00 | 193.01 | 192.76 | 193.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 11:30:00 | 193.17 | 192.76 | 193.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 193.39 | 192.89 | 193.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:30:00 | 193.85 | 192.89 | 193.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 193.66 | 193.04 | 193.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:00:00 | 193.66 | 193.04 | 193.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 193.71 | 193.18 | 193.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 192.17 | 193.23 | 193.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 12:15:00 | 194.63 | 193.50 | 193.59 | SL hit (close>static) qty=1.00 sl=194.20 alert=retest2 |

### Cycle 150 — BUY (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 13:15:00 | 194.85 | 193.77 | 193.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 14:15:00 | 196.34 | 194.29 | 193.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 15:15:00 | 193.99 | 194.23 | 193.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 15:15:00 | 193.99 | 194.23 | 193.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 193.99 | 194.23 | 193.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 10:30:00 | 198.99 | 195.28 | 194.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 11:15:00 | 199.83 | 201.73 | 201.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 199.83 | 201.73 | 201.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 198.96 | 200.88 | 201.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 200.25 | 200.22 | 200.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 200.25 | 200.22 | 200.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 200.25 | 200.22 | 200.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 200.25 | 200.22 | 200.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 201.01 | 200.38 | 200.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 12:00:00 | 199.25 | 200.15 | 200.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 15:15:00 | 201.93 | 200.61 | 200.83 | SL hit (close>static) qty=1.00 sl=201.49 alert=retest2 |

### Cycle 152 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 202.46 | 200.98 | 200.98 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 15:15:00 | 200.51 | 200.98 | 201.03 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 09:15:00 | 210.84 | 202.95 | 201.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 214.65 | 207.50 | 205.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 14:15:00 | 208.60 | 210.04 | 207.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 14:15:00 | 208.60 | 210.04 | 207.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 208.60 | 210.04 | 207.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 15:00:00 | 208.60 | 210.04 | 207.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 209.14 | 209.91 | 207.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 208.60 | 209.91 | 207.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 208.14 | 209.33 | 208.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:30:00 | 207.67 | 209.33 | 208.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 208.50 | 209.17 | 208.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:45:00 | 208.10 | 209.17 | 208.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 208.47 | 209.03 | 208.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 208.47 | 209.03 | 208.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 208.80 | 208.98 | 208.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:45:00 | 208.18 | 208.98 | 208.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 209.00 | 208.99 | 208.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 208.87 | 208.99 | 208.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 209.15 | 209.02 | 208.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 214.75 | 208.69 | 208.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:45:00 | 210.65 | 210.54 | 209.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 212.00 | 210.33 | 210.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 10:15:00 | 210.43 | 210.50 | 210.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 10:15:00 | 209.60 | 210.32 | 210.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 209.60 | 210.32 | 210.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 208.62 | 209.91 | 210.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 207.91 | 207.43 | 208.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 207.91 | 207.43 | 208.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 207.91 | 207.43 | 208.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:30:00 | 208.20 | 207.43 | 208.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 209.80 | 207.90 | 208.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 211.03 | 207.90 | 208.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 208.80 | 208.08 | 208.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 12:15:00 | 208.68 | 208.08 | 208.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:45:00 | 208.65 | 208.52 | 208.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 14:30:00 | 208.38 | 208.28 | 208.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 198.25 | 199.78 | 202.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 198.22 | 199.78 | 202.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 197.96 | 199.78 | 202.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 197.93 | 197.00 | 199.14 | SL hit (close>ema200) qty=0.50 sl=197.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 198.22 | 196.16 | 195.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 201.50 | 197.23 | 196.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 199.20 | 199.98 | 198.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 199.20 | 199.98 | 198.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 199.20 | 199.98 | 198.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:00:00 | 199.20 | 199.98 | 198.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 198.40 | 199.66 | 198.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 198.14 | 199.66 | 198.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 198.20 | 199.37 | 198.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:45:00 | 198.03 | 199.37 | 198.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 200.90 | 199.15 | 198.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 202.85 | 201.02 | 200.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 15:15:00 | 199.00 | 199.68 | 199.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 15:15:00 | 199.00 | 199.68 | 199.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 196.91 | 199.13 | 199.46 | Break + close below crossover candle low |

### Cycle 158 — BUY (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 12:15:00 | 203.38 | 199.50 | 199.50 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 14:15:00 | 199.50 | 200.62 | 200.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 15:15:00 | 199.40 | 200.38 | 200.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 198.44 | 197.69 | 198.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 11:00:00 | 198.44 | 197.69 | 198.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 198.35 | 197.82 | 198.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 198.94 | 197.82 | 198.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 198.33 | 197.92 | 198.66 | EMA400 retest candle locked (from downside) |

### Cycle 160 — BUY (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 14:15:00 | 199.80 | 199.01 | 198.95 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 197.80 | 198.79 | 198.86 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 199.00 | 198.68 | 198.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 200.68 | 199.08 | 198.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 198.71 | 199.61 | 199.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 198.71 | 199.61 | 199.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 198.71 | 199.61 | 199.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 198.71 | 199.61 | 199.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 198.12 | 199.31 | 199.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 11:15:00 | 200.00 | 199.22 | 199.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 11:15:00 | 198.90 | 200.07 | 200.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 198.90 | 200.07 | 200.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 13:15:00 | 198.64 | 199.61 | 199.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 201.30 | 199.42 | 199.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 201.30 | 199.42 | 199.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 201.30 | 199.42 | 199.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 201.30 | 199.42 | 199.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 200.85 | 199.71 | 199.78 | EMA400 retest candle locked (from downside) |

### Cycle 164 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 201.19 | 200.01 | 199.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 201.54 | 200.49 | 200.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 203.17 | 205.20 | 204.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 14:15:00 | 203.17 | 205.20 | 204.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 203.17 | 205.20 | 204.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 203.17 | 205.20 | 204.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 201.40 | 204.44 | 203.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 204.48 | 204.44 | 203.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 200.33 | 203.10 | 203.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 200.33 | 203.10 | 203.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 199.87 | 201.55 | 202.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 195.48 | 194.83 | 196.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:00:00 | 195.48 | 194.83 | 196.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 196.24 | 195.11 | 196.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:45:00 | 196.51 | 195.11 | 196.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 196.25 | 195.34 | 196.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 196.19 | 195.34 | 196.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 196.48 | 195.57 | 196.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 196.41 | 195.57 | 196.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 196.60 | 195.77 | 196.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 196.75 | 195.77 | 196.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 196.23 | 195.87 | 196.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:00:00 | 196.11 | 196.16 | 196.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 10:45:00 | 195.47 | 194.86 | 195.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 14:15:00 | 198.15 | 193.34 | 193.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — BUY (started 2025-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 14:15:00 | 198.15 | 193.34 | 193.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 15:15:00 | 202.40 | 195.15 | 194.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 202.78 | 203.53 | 200.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 10:00:00 | 202.78 | 203.53 | 200.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 200.80 | 202.52 | 201.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 200.80 | 202.52 | 201.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 201.70 | 202.35 | 201.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 201.70 | 202.35 | 201.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 200.00 | 201.88 | 200.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 200.00 | 201.88 | 200.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 199.60 | 201.43 | 200.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:30:00 | 199.95 | 201.43 | 200.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 202.26 | 201.37 | 200.92 | EMA400 retest candle locked (from upside) |

### Cycle 167 — SELL (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 13:15:00 | 198.85 | 200.65 | 200.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 15:15:00 | 197.95 | 199.69 | 200.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 200.51 | 197.31 | 198.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 200.51 | 197.31 | 198.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 200.51 | 197.31 | 198.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 200.51 | 197.31 | 198.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 199.03 | 197.65 | 198.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 12:00:00 | 198.61 | 197.84 | 198.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 13:15:00 | 198.30 | 198.14 | 198.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 196.62 | 198.52 | 198.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 09:30:00 | 198.64 | 196.80 | 197.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 199.50 | 197.34 | 197.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 11:30:00 | 198.66 | 197.52 | 197.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 12:15:00 | 198.70 | 197.76 | 197.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 198.70 | 197.76 | 197.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 199.66 | 198.33 | 197.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 199.60 | 199.64 | 198.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 199.31 | 199.68 | 199.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 199.31 | 199.68 | 199.13 | EMA400 retest candle locked (from upside) |

### Cycle 169 — SELL (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 13:15:00 | 197.91 | 198.69 | 198.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 197.45 | 198.44 | 198.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 198.78 | 198.25 | 198.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 198.78 | 198.25 | 198.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 198.78 | 198.25 | 198.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:15:00 | 197.34 | 198.09 | 198.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 13:15:00 | 197.44 | 197.97 | 198.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:00:00 | 196.75 | 197.73 | 198.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 11:15:00 | 197.00 | 197.44 | 197.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 197.18 | 197.39 | 197.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:45:00 | 196.67 | 197.38 | 197.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 11:15:00 | 196.59 | 197.26 | 197.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 12:15:00 | 198.12 | 197.51 | 197.63 | SL hit (close>static) qty=1.00 sl=198.07 alert=retest2 |

### Cycle 170 — BUY (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 14:15:00 | 198.86 | 197.85 | 197.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 14:15:00 | 202.00 | 199.04 | 198.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 197.64 | 200.79 | 200.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 197.64 | 200.79 | 200.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 197.64 | 200.79 | 200.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 197.64 | 200.79 | 200.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 197.53 | 200.14 | 199.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:30:00 | 197.89 | 200.14 | 199.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 194.81 | 199.07 | 199.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 192.63 | 197.78 | 198.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 09:15:00 | 194.90 | 194.10 | 196.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 10:00:00 | 194.90 | 194.10 | 196.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 196.44 | 194.75 | 196.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:00:00 | 194.00 | 195.68 | 196.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:30:00 | 194.62 | 193.72 | 193.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 11:15:00 | 194.66 | 193.97 | 193.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 11:15:00 | 195.01 | 194.18 | 194.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 11:15:00 | 195.01 | 194.18 | 194.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 13:15:00 | 195.34 | 194.54 | 194.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 15:15:00 | 193.00 | 194.27 | 194.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 15:15:00 | 193.00 | 194.27 | 194.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 193.00 | 194.27 | 194.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 193.26 | 194.27 | 194.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 192.60 | 193.94 | 194.05 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 13:15:00 | 194.27 | 194.12 | 194.11 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 193.30 | 194.00 | 194.06 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 10:15:00 | 194.75 | 194.20 | 194.14 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 192.50 | 193.87 | 194.00 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 15:15:00 | 194.35 | 194.04 | 194.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 195.69 | 194.37 | 194.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 14:15:00 | 198.74 | 198.75 | 197.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 15:00:00 | 198.74 | 198.75 | 197.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 200.03 | 199.70 | 198.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 199.20 | 199.70 | 198.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 207.99 | 202.33 | 200.70 | EMA400 retest candle locked (from upside) |

### Cycle 179 — SELL (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 15:15:00 | 201.99 | 202.54 | 202.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 200.36 | 202.10 | 202.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 202.00 | 200.67 | 201.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 202.00 | 200.67 | 201.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 202.00 | 200.67 | 201.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 201.84 | 200.67 | 201.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 201.92 | 200.92 | 201.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:30:00 | 202.15 | 200.92 | 201.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 202.44 | 201.39 | 201.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:30:00 | 202.15 | 201.39 | 201.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 201.22 | 201.41 | 201.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:45:00 | 202.40 | 201.41 | 201.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 201.36 | 201.40 | 201.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 203.17 | 201.40 | 201.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 203.10 | 201.74 | 201.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 10:15:00 | 204.25 | 202.24 | 201.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 09:15:00 | 200.12 | 202.15 | 202.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 200.12 | 202.15 | 202.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 200.12 | 202.15 | 202.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 200.12 | 202.15 | 202.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 202.07 | 202.13 | 202.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 11:15:00 | 203.32 | 202.13 | 202.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 202.74 | 204.80 | 204.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 202.74 | 204.80 | 204.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 201.51 | 204.14 | 204.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 194.07 | 193.76 | 196.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 13:00:00 | 194.07 | 193.76 | 196.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 196.60 | 194.67 | 196.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:00:00 | 196.60 | 194.67 | 196.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 195.37 | 194.81 | 196.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 194.28 | 194.56 | 195.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 197.40 | 195.06 | 195.37 | SL hit (close>static) qty=1.00 sl=196.98 alert=retest2 |

### Cycle 182 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 196.76 | 195.66 | 195.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 14:15:00 | 196.91 | 195.91 | 195.72 | Break + close above crossover candle high |

### Cycle 183 — SELL (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 10:15:00 | 192.88 | 195.48 | 195.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 11:15:00 | 191.94 | 194.77 | 195.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 167.70 | 166.45 | 169.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 09:15:00 | 167.70 | 166.45 | 169.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 167.70 | 166.45 | 169.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 168.84 | 166.45 | 169.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 170.69 | 167.71 | 169.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:00:00 | 170.69 | 167.71 | 169.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 170.65 | 168.30 | 169.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:00:00 | 169.70 | 170.20 | 170.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 15:15:00 | 169.60 | 169.83 | 170.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 171.54 | 170.13 | 170.17 | SL hit (close>static) qty=1.00 sl=171.04 alert=retest2 |

### Cycle 184 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 171.85 | 170.48 | 170.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 14:15:00 | 172.58 | 171.16 | 170.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 172.25 | 172.54 | 171.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 172.25 | 172.54 | 171.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 172.25 | 172.54 | 171.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 172.25 | 172.54 | 171.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 170.17 | 172.06 | 171.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 170.20 | 172.06 | 171.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 169.27 | 171.50 | 171.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:45:00 | 169.01 | 171.50 | 171.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 168.20 | 170.84 | 171.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 167.28 | 169.39 | 170.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 171.24 | 169.17 | 169.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 171.24 | 169.17 | 169.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 171.24 | 169.17 | 169.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 171.24 | 169.17 | 169.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 172.00 | 169.73 | 169.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 175.63 | 169.73 | 169.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 173.02 | 170.39 | 170.26 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 171.60 | 173.31 | 173.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 12:15:00 | 170.52 | 172.51 | 173.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 175.35 | 172.63 | 172.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 175.35 | 172.63 | 172.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 175.35 | 172.63 | 172.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 175.82 | 172.63 | 172.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 176.92 | 173.49 | 173.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 178.70 | 175.50 | 174.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 10:15:00 | 181.99 | 182.00 | 179.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 11:00:00 | 181.99 | 182.00 | 179.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 180.63 | 182.17 | 181.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 180.63 | 182.17 | 181.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 181.07 | 181.95 | 181.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:45:00 | 182.35 | 182.01 | 181.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 204.05 | 181.51 | 181.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-13 09:15:00 | 200.59 | 186.15 | 183.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 212.89 | 215.27 | 215.53 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 217.00 | 215.71 | 215.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 223.03 | 219.27 | 217.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 222.08 | 222.73 | 220.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 10:00:00 | 222.08 | 222.73 | 220.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 221.11 | 222.41 | 220.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:45:00 | 220.75 | 222.41 | 220.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 219.23 | 221.77 | 220.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:00:00 | 219.23 | 221.77 | 220.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 219.82 | 221.38 | 220.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:30:00 | 219.55 | 221.38 | 220.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 221.78 | 221.26 | 220.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:30:00 | 219.72 | 221.26 | 220.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 221.00 | 221.21 | 220.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 214.92 | 221.21 | 220.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 216.60 | 220.29 | 220.25 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 213.90 | 219.01 | 219.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 212.10 | 217.63 | 218.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 204.83 | 203.78 | 207.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 204.83 | 203.78 | 207.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 207.62 | 204.76 | 207.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 207.93 | 204.76 | 207.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 206.65 | 205.14 | 207.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 12:00:00 | 205.67 | 205.24 | 206.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:45:00 | 205.50 | 205.89 | 206.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 195.39 | 203.72 | 205.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 195.22 | 203.72 | 205.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 198.44 | 197.46 | 200.73 | SL hit (close>ema200) qty=0.50 sl=197.46 alert=retest2 |

### Cycle 192 — BUY (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 11:15:00 | 202.39 | 200.71 | 200.70 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 199.26 | 200.45 | 200.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 198.57 | 200.08 | 200.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 201.70 | 199.80 | 200.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 11:15:00 | 201.70 | 199.80 | 200.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 201.70 | 199.80 | 200.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:30:00 | 202.21 | 199.80 | 200.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 203.24 | 200.49 | 200.41 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 193.21 | 199.38 | 200.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 192.25 | 197.95 | 199.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 14:15:00 | 189.49 | 188.45 | 190.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 14:15:00 | 189.49 | 188.45 | 190.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 189.49 | 188.45 | 190.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:45:00 | 189.88 | 188.45 | 190.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 191.21 | 189.19 | 190.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 191.94 | 189.19 | 190.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 194.32 | 190.21 | 190.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 194.32 | 190.21 | 190.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 195.45 | 191.98 | 191.51 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 189.00 | 191.34 | 191.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 187.92 | 190.66 | 191.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 194.05 | 190.97 | 191.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 194.05 | 190.97 | 191.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 194.05 | 190.97 | 191.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 194.91 | 190.97 | 191.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 192.80 | 191.33 | 191.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:00:00 | 192.09 | 191.48 | 191.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 12:15:00 | 192.06 | 191.60 | 191.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 192.06 | 191.60 | 191.55 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 13:15:00 | 191.00 | 191.48 | 191.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 183.42 | 189.89 | 190.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 184.54 | 183.76 | 186.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 184.54 | 183.76 | 186.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 184.54 | 183.76 | 186.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 183.66 | 183.86 | 186.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 188.35 | 185.00 | 186.39 | SL hit (close>static) qty=1.00 sl=188.00 alert=retest2 |

### Cycle 200 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 195.71 | 188.11 | 187.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 196.00 | 189.69 | 188.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 190.59 | 193.24 | 191.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 190.59 | 193.24 | 191.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 190.59 | 193.24 | 191.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 190.59 | 193.24 | 191.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 188.83 | 192.36 | 190.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 188.70 | 192.36 | 190.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 190.00 | 190.74 | 190.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:30:00 | 189.57 | 190.74 | 190.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 189.45 | 190.48 | 190.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 186.99 | 190.48 | 190.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 187.65 | 189.92 | 190.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 182.31 | 186.58 | 188.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 192.58 | 187.13 | 188.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 192.58 | 187.13 | 188.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 192.58 | 187.13 | 188.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 193.18 | 187.13 | 188.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 199.80 | 190.40 | 189.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 202.00 | 198.95 | 196.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 200.01 | 200.16 | 198.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 200.01 | 200.16 | 198.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 198.77 | 199.39 | 198.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 204.38 | 199.39 | 198.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 10:15:00 | 224.82 | 221.28 | 218.14 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2026-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 14:15:00 | 242.73 | 243.55 | 243.66 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 252.50 | 245.12 | 244.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 261.48 | 248.40 | 245.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 255.56 | 256.21 | 253.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 14:00:00 | 255.56 | 256.21 | 253.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 253.30 | 255.61 | 253.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 11:00:00 | 253.30 | 255.61 | 253.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 252.29 | 254.95 | 253.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:00:00 | 252.29 | 254.95 | 253.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 252.66 | 254.49 | 253.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:15:00 | 252.47 | 254.49 | 253.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 250.98 | 252.81 | 252.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 248.98 | 252.04 | 252.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 13:15:00 | 252.32 | 250.87 | 251.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 13:15:00 | 252.32 | 250.87 | 251.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 252.32 | 250.87 | 251.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 13:45:00 | 253.30 | 250.87 | 251.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 251.87 | 251.07 | 251.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 253.17 | 251.07 | 251.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 253.20 | 251.50 | 251.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 253.35 | 251.50 | 251.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 252.45 | 251.69 | 251.91 | EMA400 retest candle locked (from downside) |

### Cycle 206 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 254.75 | 252.30 | 252.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 12:15:00 | 256.90 | 253.66 | 252.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 10:15:00 | 256.70 | 257.18 | 255.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 10:45:00 | 256.35 | 257.18 | 255.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 258.70 | 260.48 | 259.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:45:00 | 258.05 | 260.48 | 259.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 259.40 | 260.26 | 259.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 11:15:00 | 261.85 | 260.26 | 259.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 12:15:00 | 257.75 | 259.86 | 259.83 | SL hit (close<static) qty=1.00 sl=258.50 alert=retest2 |

### Cycle 207 — SELL (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 13:15:00 | 256.75 | 259.24 | 259.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 14:15:00 | 256.20 | 258.63 | 259.24 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-31 14:00:00 | 109.80 | 2023-06-02 12:15:00 | 108.95 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2023-05-31 14:30:00 | 109.80 | 2023-06-02 12:15:00 | 108.95 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2023-05-31 15:00:00 | 110.75 | 2023-06-02 12:15:00 | 108.95 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2023-06-02 09:45:00 | 109.85 | 2023-06-02 12:15:00 | 108.95 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2023-06-12 12:15:00 | 109.90 | 2023-06-13 09:15:00 | 112.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2023-06-26 12:45:00 | 114.50 | 2023-06-27 09:15:00 | 116.40 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2023-07-11 09:15:00 | 121.00 | 2023-07-17 10:15:00 | 133.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-11 11:30:00 | 121.40 | 2023-07-21 13:15:00 | 133.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-12 09:30:00 | 121.25 | 2023-07-21 13:15:00 | 133.38 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-08-07 09:15:00 | 146.80 | 2023-08-07 14:15:00 | 150.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2023-08-07 10:00:00 | 147.00 | 2023-08-07 14:15:00 | 150.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2023-08-07 10:45:00 | 147.05 | 2023-08-07 14:15:00 | 150.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2023-08-07 11:30:00 | 147.00 | 2023-08-07 14:15:00 | 150.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2023-08-16 11:30:00 | 149.65 | 2023-08-18 09:15:00 | 152.50 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2023-08-16 12:15:00 | 148.80 | 2023-08-18 09:15:00 | 152.50 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2023-08-17 10:45:00 | 149.75 | 2023-08-18 09:15:00 | 152.50 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2023-08-30 13:30:00 | 156.20 | 2023-09-01 15:15:00 | 158.50 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2023-08-30 15:00:00 | 155.80 | 2023-09-01 15:15:00 | 158.50 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2023-08-31 11:45:00 | 155.45 | 2023-09-01 15:15:00 | 158.50 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2023-09-01 11:30:00 | 156.30 | 2023-09-01 15:15:00 | 158.50 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest1 | 2023-09-25 09:15:00 | 143.30 | 2023-09-27 10:15:00 | 143.95 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2023-09-25 10:00:00 | 143.40 | 2023-09-27 10:15:00 | 143.95 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-09-25 11:00:00 | 143.80 | 2023-09-27 10:15:00 | 143.95 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest1 | 2023-09-25 14:30:00 | 143.70 | 2023-09-27 10:15:00 | 143.95 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2023-10-09 09:15:00 | 137.60 | 2023-10-12 09:15:00 | 142.85 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2023-10-17 11:30:00 | 140.05 | 2023-10-19 14:15:00 | 139.75 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2023-10-19 10:45:00 | 139.90 | 2023-10-19 14:15:00 | 139.75 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2023-11-07 12:45:00 | 131.75 | 2023-11-12 18:15:00 | 144.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-09 11:15:00 | 131.80 | 2023-11-12 18:15:00 | 144.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-10 09:15:00 | 133.10 | 2023-11-12 18:15:00 | 146.41 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-11-20 11:30:00 | 142.00 | 2023-11-21 09:15:00 | 148.70 | STOP_HIT | 1.00 | -4.72% |
| BUY | retest2 | 2023-12-14 15:00:00 | 169.70 | 2023-12-20 11:15:00 | 163.30 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2023-12-15 09:15:00 | 170.80 | 2023-12-20 11:15:00 | 163.30 | STOP_HIT | 1.00 | -4.39% |
| BUY | retest2 | 2023-12-15 09:45:00 | 171.35 | 2023-12-20 11:15:00 | 163.30 | STOP_HIT | 1.00 | -4.70% |
| BUY | retest2 | 2023-12-15 13:45:00 | 170.35 | 2023-12-20 11:15:00 | 163.30 | STOP_HIT | 1.00 | -4.14% |
| BUY | retest2 | 2023-12-19 12:45:00 | 167.20 | 2023-12-20 11:15:00 | 163.30 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2023-12-20 09:15:00 | 166.80 | 2023-12-20 11:15:00 | 163.30 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2023-12-22 12:15:00 | 156.90 | 2023-12-22 14:15:00 | 161.15 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2023-12-28 10:15:00 | 172.10 | 2024-01-02 13:15:00 | 189.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-29 09:15:00 | 236.45 | 2024-01-30 15:15:00 | 233.20 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-01-29 15:15:00 | 235.60 | 2024-01-30 15:15:00 | 233.20 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-02-05 13:45:00 | 255.05 | 2024-02-05 15:15:00 | 243.85 | STOP_HIT | 1.00 | -4.39% |
| SELL | retest2 | 2024-02-07 11:15:00 | 244.50 | 2024-02-09 09:15:00 | 232.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-07 12:45:00 | 243.55 | 2024-02-09 09:15:00 | 231.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-07 14:00:00 | 244.00 | 2024-02-09 09:15:00 | 231.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-08 09:30:00 | 243.00 | 2024-02-09 09:15:00 | 230.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-08 15:15:00 | 240.25 | 2024-02-09 09:15:00 | 228.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-07 11:15:00 | 244.50 | 2024-02-12 09:15:00 | 220.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-02-07 12:45:00 | 243.55 | 2024-02-12 09:15:00 | 219.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-02-07 14:00:00 | 244.00 | 2024-02-12 09:15:00 | 219.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-02-08 09:30:00 | 243.00 | 2024-02-12 09:15:00 | 218.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-02-08 15:15:00 | 240.25 | 2024-02-12 09:15:00 | 216.22 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-02-21 09:45:00 | 227.70 | 2024-02-21 13:15:00 | 222.00 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2024-02-21 10:45:00 | 226.95 | 2024-02-21 13:15:00 | 222.00 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2024-02-21 13:30:00 | 226.70 | 2024-02-21 14:15:00 | 219.95 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2024-02-26 09:15:00 | 217.35 | 2024-02-28 11:15:00 | 206.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-26 09:15:00 | 217.35 | 2024-02-29 12:15:00 | 210.10 | STOP_HIT | 0.50 | 3.34% |
| BUY | retest2 | 2024-03-07 09:15:00 | 225.50 | 2024-03-11 09:15:00 | 218.95 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2024-03-07 09:45:00 | 226.85 | 2024-03-11 09:15:00 | 218.95 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2024-03-07 10:45:00 | 225.90 | 2024-03-11 09:15:00 | 218.95 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2024-03-15 10:15:00 | 186.60 | 2024-03-20 09:15:00 | 177.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-15 10:45:00 | 186.35 | 2024-03-20 09:15:00 | 177.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-18 10:15:00 | 186.80 | 2024-03-20 09:15:00 | 177.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-18 13:15:00 | 186.70 | 2024-03-20 09:15:00 | 177.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-15 10:15:00 | 186.60 | 2024-03-21 09:15:00 | 182.65 | STOP_HIT | 0.50 | 2.12% |
| SELL | retest2 | 2024-03-15 10:45:00 | 186.35 | 2024-03-21 09:15:00 | 182.65 | STOP_HIT | 0.50 | 1.99% |
| SELL | retest2 | 2024-03-18 10:15:00 | 186.80 | 2024-03-21 09:15:00 | 182.65 | STOP_HIT | 0.50 | 2.22% |
| SELL | retest2 | 2024-03-18 13:15:00 | 186.70 | 2024-03-21 09:15:00 | 182.65 | STOP_HIT | 0.50 | 2.17% |
| BUY | retest2 | 2024-03-28 09:15:00 | 204.25 | 2024-04-04 09:15:00 | 224.68 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-04-18 12:15:00 | 212.85 | 2024-04-19 09:15:00 | 202.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-18 12:15:00 | 212.85 | 2024-04-19 13:15:00 | 206.90 | STOP_HIT | 0.50 | 2.80% |
| BUY | retest2 | 2024-05-22 14:15:00 | 269.40 | 2024-05-27 09:15:00 | 265.35 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-05-22 15:15:00 | 270.00 | 2024-05-27 09:15:00 | 265.35 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-05-24 14:45:00 | 269.10 | 2024-05-27 09:15:00 | 265.35 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-05-24 15:15:00 | 269.70 | 2024-05-27 09:15:00 | 265.35 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-06-19 12:15:00 | 265.05 | 2024-06-20 13:15:00 | 259.26 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2024-06-19 13:45:00 | 267.55 | 2024-06-20 13:15:00 | 259.26 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2024-06-20 10:15:00 | 266.00 | 2024-06-20 13:15:00 | 259.26 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2024-06-20 11:30:00 | 268.03 | 2024-06-20 13:15:00 | 259.26 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2024-06-28 12:00:00 | 251.25 | 2024-07-01 13:15:00 | 254.05 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-07-16 09:30:00 | 284.15 | 2024-07-16 15:15:00 | 280.20 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-07-16 14:45:00 | 283.30 | 2024-07-16 15:15:00 | 280.20 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-07-22 14:45:00 | 267.25 | 2024-07-23 12:15:00 | 253.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-23 09:30:00 | 266.50 | 2024-07-23 12:15:00 | 253.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 14:45:00 | 267.25 | 2024-07-25 10:15:00 | 259.90 | STOP_HIT | 0.50 | 2.75% |
| SELL | retest2 | 2024-07-23 09:30:00 | 266.50 | 2024-07-25 10:15:00 | 259.90 | STOP_HIT | 0.50 | 2.48% |
| SELL | retest2 | 2024-08-13 10:15:00 | 222.95 | 2024-08-19 12:15:00 | 219.40 | STOP_HIT | 1.00 | 1.59% |
| SELL | retest2 | 2024-08-13 12:15:00 | 223.05 | 2024-08-19 12:15:00 | 219.40 | STOP_HIT | 1.00 | 1.64% |
| SELL | retest2 | 2024-08-19 09:30:00 | 222.80 | 2024-08-19 12:15:00 | 219.40 | STOP_HIT | 1.00 | 1.53% |
| BUY | retest2 | 2024-08-20 15:15:00 | 219.80 | 2024-08-22 09:15:00 | 216.75 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-08-21 14:00:00 | 219.70 | 2024-08-22 09:15:00 | 216.75 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-08-21 15:15:00 | 219.20 | 2024-08-22 09:15:00 | 216.75 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-08-23 13:30:00 | 214.70 | 2024-08-28 09:15:00 | 222.25 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2024-08-26 10:00:00 | 214.25 | 2024-08-28 09:15:00 | 222.25 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2024-08-27 12:45:00 | 214.75 | 2024-08-28 09:15:00 | 222.25 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2024-08-27 14:45:00 | 214.60 | 2024-08-28 09:15:00 | 222.25 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2024-09-09 09:15:00 | 213.35 | 2024-09-12 09:15:00 | 216.78 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-09-25 13:15:00 | 215.15 | 2024-09-26 11:15:00 | 213.41 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-09-26 10:00:00 | 215.46 | 2024-09-26 11:15:00 | 213.41 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-10-01 10:15:00 | 209.03 | 2024-10-04 09:15:00 | 198.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 11:00:00 | 208.90 | 2024-10-04 09:15:00 | 198.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 12:00:00 | 209.00 | 2024-10-04 09:15:00 | 198.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 13:30:00 | 209.11 | 2024-10-04 09:15:00 | 198.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 10:15:00 | 209.03 | 2024-10-04 10:15:00 | 206.80 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2024-10-01 11:00:00 | 208.90 | 2024-10-04 10:15:00 | 206.80 | STOP_HIT | 0.50 | 1.01% |
| SELL | retest2 | 2024-10-01 12:00:00 | 209.00 | 2024-10-04 10:15:00 | 206.80 | STOP_HIT | 0.50 | 1.05% |
| SELL | retest2 | 2024-10-01 13:30:00 | 209.11 | 2024-10-04 10:15:00 | 206.80 | STOP_HIT | 0.50 | 1.10% |
| SELL | retest2 | 2024-10-04 12:30:00 | 204.54 | 2024-10-07 10:15:00 | 194.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 12:30:00 | 204.54 | 2024-10-08 09:15:00 | 195.40 | STOP_HIT | 0.50 | 4.47% |
| SELL | retest2 | 2024-10-17 11:15:00 | 191.95 | 2024-10-22 09:15:00 | 182.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 12:30:00 | 191.36 | 2024-10-22 09:15:00 | 181.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 11:15:00 | 191.95 | 2024-10-23 09:15:00 | 172.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-17 12:30:00 | 191.36 | 2024-10-23 09:15:00 | 181.48 | STOP_HIT | 0.50 | 5.16% |
| BUY | retest2 | 2024-11-05 14:00:00 | 194.39 | 2024-11-08 10:15:00 | 190.42 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-11-06 09:15:00 | 194.24 | 2024-11-08 10:15:00 | 190.42 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-11-06 11:00:00 | 194.41 | 2024-11-08 10:15:00 | 190.42 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-11-11 13:15:00 | 189.31 | 2024-11-13 14:15:00 | 179.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 14:00:00 | 189.37 | 2024-11-13 14:15:00 | 179.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 15:15:00 | 189.15 | 2024-11-13 14:15:00 | 179.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:45:00 | 189.00 | 2024-11-13 14:15:00 | 179.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 13:15:00 | 189.31 | 2024-11-14 09:15:00 | 183.25 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2024-11-11 14:00:00 | 189.37 | 2024-11-14 09:15:00 | 183.25 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2024-11-11 15:15:00 | 189.15 | 2024-11-14 09:15:00 | 183.25 | STOP_HIT | 0.50 | 3.12% |
| SELL | retest2 | 2024-11-12 12:45:00 | 189.00 | 2024-11-14 09:15:00 | 183.25 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2024-11-22 12:45:00 | 179.99 | 2024-11-25 09:15:00 | 185.74 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2024-11-22 15:15:00 | 180.20 | 2024-11-25 09:15:00 | 185.74 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest1 | 2024-12-02 11:00:00 | 202.75 | 2024-12-02 15:15:00 | 198.24 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-12-03 09:15:00 | 202.88 | 2024-12-12 10:15:00 | 204.80 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2024-12-16 15:00:00 | 201.82 | 2024-12-18 14:15:00 | 191.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 10:15:00 | 201.00 | 2024-12-18 14:15:00 | 191.69 | PARTIAL | 0.50 | 4.63% |
| SELL | retest2 | 2024-12-17 11:45:00 | 201.78 | 2024-12-19 09:15:00 | 190.95 | PARTIAL | 0.50 | 5.37% |
| SELL | retest2 | 2024-12-17 13:15:00 | 201.52 | 2024-12-19 09:15:00 | 191.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 15:00:00 | 201.82 | 2024-12-19 15:15:00 | 193.95 | STOP_HIT | 0.50 | 3.90% |
| SELL | retest2 | 2024-12-17 10:15:00 | 201.00 | 2024-12-19 15:15:00 | 193.95 | STOP_HIT | 0.50 | 3.51% |
| SELL | retest2 | 2024-12-17 11:45:00 | 201.78 | 2024-12-19 15:15:00 | 193.95 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2024-12-17 13:15:00 | 201.52 | 2024-12-19 15:15:00 | 193.95 | STOP_HIT | 0.50 | 3.76% |
| SELL | retest2 | 2025-01-08 12:30:00 | 180.64 | 2025-01-10 09:15:00 | 171.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 13:00:00 | 180.59 | 2025-01-10 09:15:00 | 171.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 09:30:00 | 180.40 | 2025-01-10 09:15:00 | 171.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 10:45:00 | 180.36 | 2025-01-10 09:15:00 | 171.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 12:30:00 | 180.64 | 2025-01-13 13:15:00 | 162.58 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-08 13:00:00 | 180.59 | 2025-01-13 13:15:00 | 162.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 09:30:00 | 180.40 | 2025-01-13 14:15:00 | 162.36 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 10:45:00 | 180.36 | 2025-01-13 14:15:00 | 162.32 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-14 12:15:00 | 167.50 | 2025-01-15 09:15:00 | 174.38 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2025-01-14 13:45:00 | 167.45 | 2025-01-15 09:15:00 | 174.38 | STOP_HIT | 1.00 | -4.14% |
| BUY | retest2 | 2025-01-21 09:30:00 | 182.08 | 2025-01-21 15:15:00 | 176.15 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2025-01-23 13:30:00 | 173.00 | 2025-01-27 09:15:00 | 164.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 14:00:00 | 172.60 | 2025-01-27 09:15:00 | 163.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 13:30:00 | 173.00 | 2025-01-28 09:15:00 | 155.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-23 14:00:00 | 172.60 | 2025-01-28 09:15:00 | 155.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-04 10:45:00 | 163.65 | 2025-02-04 14:15:00 | 167.10 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-02-25 11:15:00 | 167.01 | 2025-02-25 12:15:00 | 169.40 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-03-18 09:15:00 | 161.10 | 2025-03-25 14:15:00 | 163.13 | STOP_HIT | 1.00 | 1.26% |
| BUY | retest2 | 2025-04-03 09:15:00 | 163.60 | 2025-04-07 09:15:00 | 153.13 | STOP_HIT | 1.00 | -6.40% |
| BUY | retest2 | 2025-04-03 10:15:00 | 163.40 | 2025-04-07 09:15:00 | 153.13 | STOP_HIT | 1.00 | -6.29% |
| BUY | retest2 | 2025-04-04 10:45:00 | 164.00 | 2025-04-07 09:15:00 | 153.13 | STOP_HIT | 1.00 | -6.63% |
| BUY | retest2 | 2025-04-04 12:15:00 | 163.32 | 2025-04-07 09:15:00 | 153.13 | STOP_HIT | 1.00 | -6.24% |
| SELL | retest2 | 2025-04-11 11:15:00 | 160.40 | 2025-04-15 09:15:00 | 164.16 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-04-11 12:30:00 | 160.40 | 2025-04-15 09:15:00 | 164.16 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-04-11 13:15:00 | 160.38 | 2025-04-15 09:15:00 | 164.16 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-04-23 09:15:00 | 186.95 | 2025-04-25 09:15:00 | 178.29 | STOP_HIT | 1.00 | -4.63% |
| BUY | retest2 | 2025-04-23 14:45:00 | 185.15 | 2025-04-25 09:15:00 | 178.29 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2025-04-24 09:30:00 | 185.12 | 2025-04-25 09:15:00 | 178.29 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest2 | 2025-04-24 10:30:00 | 185.77 | 2025-04-25 09:15:00 | 178.29 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2025-05-05 13:00:00 | 178.65 | 2025-05-05 14:15:00 | 180.78 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-05-06 09:30:00 | 178.68 | 2025-05-08 11:15:00 | 178.69 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-05-08 10:30:00 | 178.70 | 2025-05-08 11:15:00 | 178.69 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-05-20 11:30:00 | 195.03 | 2025-05-29 14:15:00 | 214.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-21 09:30:00 | 195.02 | 2025-05-29 14:15:00 | 214.52 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2025-06-16 09:15:00 | 221.45 | 2025-06-17 09:15:00 | 226.94 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-06-17 11:30:00 | 222.76 | 2025-06-23 11:15:00 | 222.00 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-06-18 12:30:00 | 222.50 | 2025-06-23 11:15:00 | 222.00 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2025-06-18 14:15:00 | 222.74 | 2025-06-23 11:15:00 | 222.00 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-06-18 15:15:00 | 222.00 | 2025-06-23 11:15:00 | 222.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-06-19 10:15:00 | 219.77 | 2025-06-23 11:15:00 | 222.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-06-23 09:15:00 | 220.31 | 2025-06-23 11:15:00 | 222.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-06-23 10:15:00 | 219.91 | 2025-06-23 11:15:00 | 222.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest1 | 2025-07-10 09:15:00 | 247.98 | 2025-07-14 14:15:00 | 248.11 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest1 | 2025-07-10 09:45:00 | 246.60 | 2025-07-14 14:15:00 | 248.11 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest1 | 2025-07-10 11:00:00 | 248.16 | 2025-07-14 14:15:00 | 248.11 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-07-16 12:15:00 | 246.30 | 2025-07-18 14:15:00 | 233.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-16 15:15:00 | 245.90 | 2025-07-18 14:15:00 | 233.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-16 12:15:00 | 246.30 | 2025-07-21 11:15:00 | 238.33 | STOP_HIT | 0.50 | 3.24% |
| SELL | retest2 | 2025-07-16 15:15:00 | 245.90 | 2025-07-21 11:15:00 | 238.33 | STOP_HIT | 0.50 | 3.08% |
| SELL | retest2 | 2025-08-05 10:45:00 | 210.50 | 2025-08-07 12:15:00 | 199.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 11:30:00 | 210.62 | 2025-08-07 12:15:00 | 200.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 10:45:00 | 210.50 | 2025-08-07 14:15:00 | 203.41 | STOP_HIT | 0.50 | 3.37% |
| SELL | retest2 | 2025-08-05 11:30:00 | 210.62 | 2025-08-07 14:15:00 | 203.41 | STOP_HIT | 0.50 | 3.42% |
| SELL | retest2 | 2025-08-26 09:15:00 | 192.17 | 2025-08-26 12:15:00 | 194.63 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-08-28 10:30:00 | 198.99 | 2025-09-04 11:15:00 | 199.83 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-09-05 12:00:00 | 199.25 | 2025-09-05 15:15:00 | 201.93 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-09-15 09:15:00 | 214.75 | 2025-09-18 10:15:00 | 209.60 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-09-16 09:45:00 | 210.65 | 2025-09-18 10:15:00 | 209.60 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-09-17 09:15:00 | 212.00 | 2025-09-18 10:15:00 | 209.60 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-09-18 10:15:00 | 210.43 | 2025-09-18 10:15:00 | 209.60 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-09-22 12:15:00 | 208.68 | 2025-09-26 09:15:00 | 198.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 13:45:00 | 208.65 | 2025-09-26 09:15:00 | 198.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 14:30:00 | 208.38 | 2025-09-26 09:15:00 | 197.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 12:15:00 | 208.68 | 2025-09-29 09:15:00 | 197.93 | STOP_HIT | 0.50 | 5.15% |
| SELL | retest2 | 2025-09-22 13:45:00 | 208.65 | 2025-09-29 09:15:00 | 197.93 | STOP_HIT | 0.50 | 5.14% |
| SELL | retest2 | 2025-09-22 14:30:00 | 208.38 | 2025-09-29 09:15:00 | 197.93 | STOP_HIT | 0.50 | 5.01% |
| BUY | retest2 | 2025-10-08 09:15:00 | 202.85 | 2025-10-08 15:15:00 | 199.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-10-24 11:15:00 | 200.00 | 2025-10-28 11:15:00 | 198.90 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-11-03 09:15:00 | 204.48 | 2025-11-03 11:15:00 | 200.33 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-11-10 12:00:00 | 196.11 | 2025-11-14 14:15:00 | 198.15 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-11-12 10:45:00 | 195.47 | 2025-11-14 14:15:00 | 198.15 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-11-24 12:00:00 | 198.61 | 2025-11-26 12:15:00 | 198.70 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-11-24 13:15:00 | 198.30 | 2025-11-26 12:15:00 | 198.70 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-11-25 09:15:00 | 196.62 | 2025-11-26 12:15:00 | 198.70 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-11-26 09:30:00 | 198.64 | 2025-11-26 12:15:00 | 198.70 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-11-26 11:30:00 | 198.66 | 2025-11-26 12:15:00 | 198.70 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-12-01 12:15:00 | 197.34 | 2025-12-03 12:15:00 | 198.12 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-12-01 13:15:00 | 197.44 | 2025-12-03 12:15:00 | 198.12 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-12-01 14:00:00 | 196.75 | 2025-12-03 14:15:00 | 198.86 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-12-02 11:15:00 | 197.00 | 2025-12-03 14:15:00 | 198.86 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-12-03 09:45:00 | 196.67 | 2025-12-03 14:15:00 | 198.86 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-12-03 11:15:00 | 196.59 | 2025-12-03 14:15:00 | 198.86 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-12-10 11:00:00 | 194.00 | 2025-12-15 11:15:00 | 195.01 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-12-15 09:30:00 | 194.62 | 2025-12-15 11:15:00 | 195.01 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-12-15 11:15:00 | 194.66 | 2025-12-15 11:15:00 | 195.01 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2026-01-02 11:15:00 | 203.32 | 2026-01-08 09:15:00 | 202.74 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2026-01-13 11:30:00 | 194.28 | 2026-01-14 11:15:00 | 197.40 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-01-29 10:00:00 | 169.70 | 2026-01-30 09:15:00 | 171.54 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-01-29 15:15:00 | 169.60 | 2026-01-30 09:15:00 | 171.54 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-02-12 13:45:00 | 182.35 | 2026-02-13 09:15:00 | 200.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-13 09:15:00 | 204.05 | 2026-02-16 14:15:00 | 224.46 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-06 12:00:00 | 205.67 | 2026-03-09 09:15:00 | 195.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:45:00 | 205.50 | 2026-03-09 09:15:00 | 195.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 12:00:00 | 205.67 | 2026-03-10 09:15:00 | 198.44 | STOP_HIT | 0.50 | 3.52% |
| SELL | retest2 | 2026-03-06 14:45:00 | 205.50 | 2026-03-10 09:15:00 | 198.44 | STOP_HIT | 0.50 | 3.44% |
| SELL | retest2 | 2026-03-20 12:00:00 | 192.09 | 2026-03-20 12:15:00 | 192.06 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2026-03-24 10:30:00 | 183.66 | 2026-03-24 12:15:00 | 188.35 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2026-04-08 09:15:00 | 204.38 | 2026-04-16 10:15:00 | 224.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-07 11:15:00 | 261.85 | 2026-05-08 12:15:00 | 257.75 | STOP_HIT | 1.00 | -1.57% |
