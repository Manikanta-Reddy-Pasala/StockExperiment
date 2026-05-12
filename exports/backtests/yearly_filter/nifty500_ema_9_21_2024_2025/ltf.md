# L&T Finance Ltd. (LTF)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 302.85
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 167 |
| ALERT1 | 110 |
| ALERT2 | 107 |
| ALERT2_SKIP | 57 |
| ALERT3 | 297 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 112 |
| PARTIAL | 9 |
| TARGET_HIT | 2 |
| STOP_HIT | 114 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 123 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 93
- **Target hits / Stop hits / Partials:** 2 / 113 / 8
- **Avg / median % per leg:** -0.08% / -0.88%
- **Sum % (uncompounded):** -10.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 60 | 15 | 25.0% | 2 | 57 | 1 | 0.16% | 9.8% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 5.03% | 10.1% |
| BUY @ 3rd Alert (retest2) | 58 | 13 | 22.4% | 2 | 56 | 0 | -0.00% | -0.2% |
| SELL (all) | 63 | 15 | 23.8% | 0 | 56 | 7 | -0.32% | -20.2% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.02% | -4.0% |
| SELL @ 3rd Alert (retest2) | 61 | 15 | 24.6% | 0 | 54 | 7 | -0.26% | -16.1% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | 1.51% | 6.0% |
| retest2 (combined) | 119 | 28 | 23.5% | 2 | 110 | 7 | -0.14% | -16.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 160.25 | 157.48 | 157.20 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 15:15:00 | 158.65 | 159.18 | 159.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 157.00 | 158.74 | 159.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 11:15:00 | 156.80 | 156.75 | 157.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-23 12:00:00 | 156.80 | 156.75 | 157.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 159.50 | 157.39 | 157.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 15:00:00 | 159.50 | 157.39 | 157.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 159.00 | 157.71 | 157.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 10:00:00 | 158.00 | 157.77 | 157.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 10:15:00 | 158.75 | 157.97 | 157.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 10:15:00 | 158.75 | 157.97 | 157.87 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 09:15:00 | 157.20 | 157.82 | 157.86 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 12:15:00 | 158.20 | 157.87 | 157.87 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 10:15:00 | 157.25 | 157.79 | 157.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 11:15:00 | 155.85 | 157.40 | 157.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 158.50 | 157.20 | 157.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 158.50 | 157.20 | 157.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 158.50 | 157.20 | 157.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:00:00 | 158.50 | 157.20 | 157.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 158.10 | 157.38 | 157.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:45:00 | 158.65 | 157.38 | 157.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 12:15:00 | 157.80 | 157.58 | 157.55 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 155.40 | 157.12 | 157.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 11:15:00 | 154.50 | 156.29 | 156.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 09:15:00 | 155.70 | 154.78 | 155.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 09:15:00 | 155.70 | 154.78 | 155.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 155.70 | 154.78 | 155.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 14:45:00 | 152.80 | 154.32 | 155.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 159.35 | 155.26 | 155.46 | SL hit (close>static) qty=1.00 sl=156.75 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 158.55 | 155.92 | 155.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 13:15:00 | 159.70 | 157.61 | 156.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 154.30 | 157.53 | 156.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 154.30 | 157.53 | 156.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 154.30 | 157.53 | 156.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 152.45 | 157.53 | 156.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 147.75 | 155.57 | 156.06 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 15:15:00 | 156.70 | 154.03 | 153.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 159.15 | 155.05 | 154.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 13:15:00 | 165.62 | 165.75 | 163.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 13:45:00 | 165.90 | 165.75 | 163.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 176.63 | 179.29 | 178.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 175.73 | 179.29 | 178.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 176.04 | 178.64 | 177.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 174.85 | 178.64 | 177.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 13:15:00 | 177.00 | 178.11 | 177.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 13:45:00 | 177.25 | 178.11 | 177.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 14:15:00 | 176.14 | 177.72 | 177.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 15:00:00 | 176.14 | 177.72 | 177.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 15:15:00 | 176.00 | 177.38 | 177.52 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 10:15:00 | 178.40 | 177.67 | 177.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 12:15:00 | 181.01 | 178.58 | 178.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 15:15:00 | 182.94 | 182.95 | 181.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-24 09:15:00 | 179.55 | 182.95 | 181.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 178.97 | 182.15 | 181.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 177.87 | 182.15 | 181.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 179.23 | 181.57 | 181.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:15:00 | 178.78 | 181.57 | 181.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 181.00 | 181.02 | 180.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 09:15:00 | 181.48 | 180.84 | 180.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 12:15:00 | 180.76 | 180.84 | 180.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 12:15:00 | 180.76 | 180.84 | 180.84 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 13:15:00 | 182.25 | 181.12 | 180.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 14:15:00 | 185.57 | 182.01 | 181.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 12:15:00 | 182.55 | 183.45 | 182.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 12:15:00 | 182.55 | 183.45 | 182.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 182.55 | 183.45 | 182.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:00:00 | 182.55 | 183.45 | 182.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 183.77 | 183.51 | 182.60 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2024-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 12:15:00 | 179.90 | 182.10 | 182.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 179.61 | 181.60 | 181.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 182.37 | 181.07 | 181.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 09:15:00 | 182.37 | 181.07 | 181.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 182.37 | 181.07 | 181.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:00:00 | 182.37 | 181.07 | 181.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 182.02 | 181.26 | 181.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:00:00 | 182.02 | 181.26 | 181.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 181.35 | 181.28 | 181.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 12:15:00 | 180.60 | 181.28 | 181.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 09:15:00 | 185.03 | 181.97 | 181.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 185.03 | 181.97 | 181.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 10:15:00 | 186.80 | 182.94 | 182.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 187.47 | 188.11 | 186.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 13:00:00 | 187.47 | 188.11 | 186.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 188.55 | 188.92 | 188.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 11:30:00 | 188.28 | 188.92 | 188.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 187.99 | 188.72 | 188.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 187.99 | 188.72 | 188.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 188.30 | 188.63 | 188.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 186.86 | 188.63 | 188.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 186.46 | 188.20 | 188.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:45:00 | 185.15 | 188.20 | 188.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 187.05 | 187.97 | 187.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 11:15:00 | 188.27 | 187.97 | 187.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 11:15:00 | 186.09 | 188.25 | 188.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 186.09 | 188.25 | 188.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 14:15:00 | 185.89 | 187.20 | 187.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 181.44 | 181.26 | 182.99 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:15:00 | 180.31 | 181.26 | 182.99 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 11:15:00 | 179.96 | 181.12 | 182.77 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 183.77 | 180.81 | 181.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-12 09:15:00 | 183.77 | 180.81 | 181.76 | SL hit (close>ema400) qty=1.00 sl=181.76 alert=retest1 |

### Cycle 19 — BUY (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 10:15:00 | 183.74 | 182.16 | 182.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 12:15:00 | 184.33 | 182.87 | 182.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 09:15:00 | 183.00 | 183.96 | 183.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 183.00 | 183.96 | 183.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 183.00 | 183.96 | 183.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:00:00 | 183.00 | 183.96 | 183.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 182.75 | 183.72 | 183.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 11:15:00 | 182.46 | 183.72 | 183.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 182.93 | 183.56 | 183.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 14:30:00 | 184.80 | 183.63 | 183.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 186.92 | 183.76 | 183.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 11:45:00 | 185.22 | 184.65 | 183.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 15:15:00 | 184.72 | 184.86 | 184.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 184.72 | 184.84 | 184.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:15:00 | 183.81 | 184.84 | 184.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 180.23 | 183.91 | 183.88 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 180.23 | 183.91 | 183.88 | SL hit (close<static) qty=1.00 sl=182.25 alert=retest2 |

### Cycle 20 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 179.42 | 183.02 | 183.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 12:15:00 | 178.02 | 181.51 | 182.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 13:15:00 | 173.30 | 173.25 | 175.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 14:00:00 | 173.30 | 173.25 | 175.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 177.10 | 174.30 | 175.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 177.10 | 174.30 | 175.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 176.65 | 174.77 | 175.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 11:15:00 | 175.74 | 174.77 | 175.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 10:15:00 | 178.40 | 175.38 | 175.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 178.40 | 175.38 | 175.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 180.01 | 177.36 | 176.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 179.25 | 179.56 | 178.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:00:00 | 179.25 | 179.56 | 178.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 12:15:00 | 180.25 | 180.64 | 179.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 12:30:00 | 179.89 | 180.64 | 179.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 179.99 | 180.51 | 179.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:00:00 | 179.99 | 180.51 | 179.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 179.69 | 180.34 | 179.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 179.69 | 180.34 | 179.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 180.30 | 180.33 | 179.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 181.00 | 180.33 | 179.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 11:00:00 | 180.67 | 180.55 | 180.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 11:15:00 | 178.66 | 180.17 | 180.00 | SL hit (close<static) qty=1.00 sl=179.70 alert=retest2 |

### Cycle 22 — SELL (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 12:15:00 | 177.92 | 179.72 | 179.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 13:15:00 | 177.15 | 179.21 | 179.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 12:15:00 | 177.14 | 176.97 | 178.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-02 12:45:00 | 177.32 | 176.97 | 178.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 177.37 | 177.15 | 177.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 14:30:00 | 178.17 | 177.15 | 177.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 172.64 | 171.04 | 173.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 168.85 | 170.75 | 172.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 09:30:00 | 169.05 | 169.28 | 171.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:15:00 | 160.41 | 163.91 | 165.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:15:00 | 160.60 | 163.91 | 165.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-16 09:15:00 | 162.73 | 161.90 | 163.42 | SL hit (close>ema200) qty=0.50 sl=161.90 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 164.95 | 163.97 | 163.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 15:15:00 | 165.50 | 164.53 | 164.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 14:15:00 | 169.10 | 169.49 | 168.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 15:00:00 | 169.10 | 169.49 | 168.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 168.06 | 169.09 | 168.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:45:00 | 168.34 | 169.09 | 168.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 168.12 | 168.90 | 168.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 11:15:00 | 168.50 | 168.90 | 168.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 14:15:00 | 167.53 | 168.31 | 168.16 | SL hit (close<static) qty=1.00 sl=167.75 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 10:15:00 | 167.15 | 167.89 | 167.98 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 14:15:00 | 169.14 | 168.21 | 168.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 09:15:00 | 170.29 | 168.73 | 168.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 14:15:00 | 170.60 | 171.18 | 170.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 15:00:00 | 170.60 | 171.18 | 170.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 170.50 | 171.04 | 170.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 170.27 | 171.04 | 170.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 171.23 | 171.08 | 170.45 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 14:15:00 | 169.45 | 170.08 | 170.16 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 12:15:00 | 171.09 | 170.34 | 170.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 09:15:00 | 173.00 | 171.13 | 170.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 09:15:00 | 171.76 | 172.23 | 171.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 171.76 | 172.23 | 171.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 171.76 | 172.23 | 171.63 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 12:15:00 | 168.93 | 170.81 | 171.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 166.85 | 169.81 | 170.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 166.32 | 165.57 | 166.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 15:00:00 | 166.32 | 165.57 | 166.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 168.50 | 166.31 | 167.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 169.60 | 166.31 | 167.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 167.65 | 166.57 | 167.08 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2024-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 13:15:00 | 171.83 | 168.19 | 167.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 15:15:00 | 172.40 | 169.61 | 168.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 11:15:00 | 169.97 | 170.10 | 169.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 11:45:00 | 170.10 | 170.10 | 169.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 166.80 | 169.42 | 168.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 166.80 | 169.42 | 168.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 166.19 | 168.77 | 168.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 166.19 | 168.77 | 168.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 15:15:00 | 167.00 | 168.42 | 168.51 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 09:15:00 | 171.17 | 168.97 | 168.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 12:15:00 | 172.66 | 170.42 | 169.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 10:15:00 | 175.75 | 176.43 | 174.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 11:00:00 | 175.75 | 176.43 | 174.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 175.40 | 175.63 | 174.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 11:30:00 | 176.72 | 175.75 | 175.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 12:30:00 | 176.97 | 175.92 | 175.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 09:15:00 | 177.40 | 175.86 | 175.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 10:45:00 | 176.71 | 176.25 | 175.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 176.57 | 176.41 | 175.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:30:00 | 176.43 | 176.41 | 175.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 174.58 | 176.12 | 175.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:00:00 | 174.58 | 176.12 | 175.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-19 11:15:00 | 173.97 | 175.69 | 175.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 11:15:00 | 173.97 | 175.69 | 175.73 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 14:15:00 | 178.25 | 175.89 | 175.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 09:15:00 | 178.79 | 176.79 | 176.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 14:15:00 | 183.81 | 184.40 | 182.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 15:00:00 | 183.81 | 184.40 | 182.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 184.50 | 184.41 | 183.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:30:00 | 184.04 | 184.41 | 183.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 184.27 | 186.09 | 184.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 184.37 | 186.09 | 184.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 184.11 | 185.69 | 184.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:30:00 | 184.28 | 185.69 | 184.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 182.90 | 185.13 | 184.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:45:00 | 182.83 | 185.13 | 184.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 185.74 | 186.31 | 185.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 13:00:00 | 185.74 | 186.31 | 185.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 186.29 | 186.31 | 185.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 14:45:00 | 188.66 | 186.65 | 185.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 11:15:00 | 187.02 | 186.70 | 186.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 13:45:00 | 186.99 | 186.74 | 186.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 15:15:00 | 185.50 | 186.36 | 186.19 | SL hit (close<static) qty=1.00 sl=185.51 alert=retest2 |

### Cycle 34 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 182.69 | 186.00 | 186.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 181.94 | 184.65 | 185.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 171.76 | 170.80 | 174.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 171.76 | 170.80 | 174.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 175.14 | 172.08 | 173.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 175.14 | 172.08 | 173.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 174.91 | 172.65 | 173.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:45:00 | 172.90 | 172.75 | 173.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 12:30:00 | 173.17 | 172.62 | 173.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 12:15:00 | 164.25 | 166.71 | 168.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 12:15:00 | 164.51 | 166.71 | 168.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-14 09:15:00 | 165.88 | 165.42 | 167.49 | SL hit (close>ema200) qty=0.50 sl=165.42 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 15:15:00 | 168.00 | 167.07 | 166.95 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 15:15:00 | 166.55 | 166.88 | 166.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 164.91 | 166.49 | 166.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 12:15:00 | 165.91 | 165.86 | 166.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-17 13:00:00 | 165.91 | 165.86 | 166.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 166.50 | 165.73 | 166.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 14:45:00 | 166.13 | 165.73 | 166.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 165.60 | 165.71 | 166.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 09:15:00 | 162.18 | 165.71 | 166.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 11:15:00 | 167.60 | 166.09 | 166.18 | SL hit (close>static) qty=1.00 sl=166.93 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 12:15:00 | 166.94 | 166.26 | 166.25 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 09:15:00 | 164.75 | 166.19 | 166.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 12:15:00 | 159.98 | 164.10 | 165.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 151.74 | 151.57 | 156.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:00:00 | 151.74 | 151.57 | 156.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 141.11 | 141.66 | 144.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 09:30:00 | 140.79 | 142.07 | 143.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:00:00 | 140.98 | 142.07 | 143.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 10:15:00 | 146.44 | 143.98 | 143.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 146.44 | 143.98 | 143.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 11:15:00 | 147.59 | 144.70 | 144.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 11:15:00 | 145.87 | 146.20 | 145.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 12:00:00 | 145.87 | 146.20 | 145.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 145.48 | 146.06 | 145.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 13:00:00 | 145.48 | 146.06 | 145.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 144.85 | 145.82 | 145.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:00:00 | 144.85 | 145.82 | 145.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 145.80 | 145.81 | 145.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:30:00 | 145.15 | 145.81 | 145.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 144.25 | 146.45 | 145.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 144.25 | 146.45 | 145.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 143.01 | 145.76 | 145.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 143.01 | 145.76 | 145.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 143.07 | 145.22 | 145.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 11:15:00 | 141.80 | 143.18 | 144.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 145.90 | 143.71 | 144.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 145.90 | 143.71 | 144.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 145.90 | 143.71 | 144.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 145.90 | 143.71 | 144.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 145.34 | 144.04 | 144.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 15:15:00 | 145.01 | 144.04 | 144.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 09:15:00 | 147.02 | 144.79 | 144.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 147.02 | 144.79 | 144.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 148.23 | 146.28 | 145.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 145.55 | 146.65 | 145.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 145.55 | 146.65 | 145.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 145.55 | 146.65 | 145.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 145.55 | 146.65 | 145.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 145.00 | 146.32 | 145.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 144.38 | 146.32 | 145.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 144.28 | 145.91 | 145.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 144.28 | 145.91 | 145.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 13:15:00 | 144.60 | 145.45 | 145.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 143.47 | 144.69 | 145.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 141.75 | 141.22 | 142.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 11:00:00 | 141.75 | 141.22 | 142.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 137.20 | 137.31 | 138.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 138.05 | 137.31 | 138.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 138.40 | 137.65 | 138.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 138.40 | 137.65 | 138.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 137.89 | 137.70 | 138.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 136.56 | 137.75 | 138.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 11:15:00 | 138.80 | 137.86 | 138.18 | SL hit (close>static) qty=1.00 sl=138.60 alert=retest2 |

### Cycle 43 — BUY (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 13:15:00 | 140.05 | 138.64 | 138.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 142.31 | 139.35 | 138.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 139.53 | 140.37 | 139.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 14:15:00 | 139.53 | 140.37 | 139.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 139.53 | 140.37 | 139.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:00:00 | 139.53 | 140.37 | 139.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 138.98 | 140.09 | 139.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 136.63 | 140.09 | 139.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 136.30 | 139.33 | 139.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 136.45 | 139.33 | 139.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 10:15:00 | 137.43 | 138.95 | 139.14 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 12:15:00 | 139.78 | 138.94 | 138.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 142.78 | 139.97 | 139.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 140.78 | 141.06 | 140.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 15:00:00 | 140.78 | 141.06 | 140.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 140.30 | 140.95 | 140.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:45:00 | 139.73 | 140.95 | 140.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 140.39 | 140.84 | 140.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 14:30:00 | 140.52 | 140.46 | 140.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:15:00 | 140.60 | 140.32 | 140.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 11:15:00 | 147.59 | 149.04 | 149.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 11:15:00 | 147.59 | 149.04 | 149.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 143.88 | 147.32 | 148.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 146.50 | 146.23 | 147.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 14:00:00 | 146.50 | 146.23 | 147.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 147.63 | 146.63 | 147.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 147.73 | 146.63 | 147.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 146.58 | 146.62 | 147.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:30:00 | 147.19 | 146.62 | 147.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 147.33 | 146.74 | 147.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:45:00 | 147.32 | 146.74 | 147.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 147.20 | 146.83 | 147.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 15:00:00 | 147.20 | 146.83 | 147.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 147.15 | 146.90 | 147.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:15:00 | 145.83 | 146.90 | 147.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 138.54 | 141.33 | 142.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-19 10:15:00 | 141.54 | 141.37 | 142.72 | SL hit (close>ema200) qty=0.50 sl=141.37 alert=retest2 |

### Cycle 47 — BUY (started 2024-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 12:15:00 | 138.96 | 138.12 | 138.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 13:15:00 | 139.08 | 138.31 | 138.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 14:15:00 | 138.05 | 138.26 | 138.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 14:15:00 | 138.05 | 138.26 | 138.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 138.05 | 138.26 | 138.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 15:00:00 | 138.05 | 138.26 | 138.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 137.99 | 138.21 | 138.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 137.65 | 138.21 | 138.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 09:15:00 | 137.72 | 138.11 | 138.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 12:15:00 | 137.05 | 137.71 | 137.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 09:15:00 | 136.67 | 136.09 | 136.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-01 09:15:00 | 136.67 | 136.09 | 136.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 136.67 | 136.09 | 136.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:30:00 | 137.38 | 136.09 | 136.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 137.12 | 136.30 | 136.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 11:00:00 | 137.12 | 136.30 | 136.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 136.83 | 136.40 | 136.67 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 14:15:00 | 137.83 | 136.82 | 136.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 139.82 | 137.56 | 137.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 140.87 | 142.42 | 141.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 140.87 | 142.42 | 141.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 140.87 | 142.42 | 141.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 140.33 | 142.42 | 141.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 138.54 | 141.64 | 140.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 138.54 | 141.64 | 140.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 138.67 | 141.05 | 140.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 137.55 | 141.05 | 140.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 136.95 | 139.85 | 140.21 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 12:15:00 | 141.65 | 139.99 | 139.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 13:15:00 | 144.44 | 140.88 | 140.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 09:15:00 | 141.74 | 141.87 | 140.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 12:15:00 | 141.79 | 141.80 | 141.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 141.79 | 141.80 | 141.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 15:15:00 | 144.00 | 141.87 | 141.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-10 09:15:00 | 139.00 | 141.64 | 141.32 | SL hit (close<static) qty=1.00 sl=140.83 alert=retest2 |

### Cycle 52 — SELL (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 11:15:00 | 138.80 | 140.84 | 140.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 137.30 | 139.73 | 140.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 133.94 | 133.64 | 135.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 133.94 | 133.64 | 135.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 134.00 | 133.73 | 134.96 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2025-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 14:15:00 | 139.77 | 135.66 | 135.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 15:15:00 | 141.50 | 136.83 | 136.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 15:15:00 | 142.44 | 142.56 | 140.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-20 09:15:00 | 141.83 | 142.56 | 140.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 147.13 | 144.31 | 142.75 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 139.64 | 143.31 | 143.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 12:15:00 | 139.05 | 142.46 | 143.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 13:15:00 | 140.51 | 140.22 | 141.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 13:15:00 | 140.51 | 140.22 | 141.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 140.51 | 140.22 | 141.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 14:00:00 | 140.51 | 140.22 | 141.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 138.98 | 139.87 | 140.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 135.27 | 140.18 | 140.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 14:15:00 | 141.67 | 139.65 | 139.92 | SL hit (close>static) qty=1.00 sl=141.43 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 12:15:00 | 143.36 | 140.33 | 140.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 09:15:00 | 145.95 | 142.10 | 141.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 14:15:00 | 143.99 | 145.99 | 144.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 14:15:00 | 143.99 | 145.99 | 144.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 143.99 | 145.99 | 144.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 15:00:00 | 143.99 | 145.99 | 144.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 144.75 | 145.74 | 144.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:15:00 | 145.11 | 145.74 | 144.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 13:15:00 | 144.82 | 145.10 | 145.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:15:00 | 144.99 | 145.42 | 145.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 13:15:00 | 148.75 | 150.74 | 150.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 148.75 | 150.74 | 150.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 147.55 | 149.62 | 150.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 141.97 | 141.44 | 143.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:30:00 | 141.29 | 141.44 | 143.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 142.12 | 141.30 | 142.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 142.12 | 141.30 | 142.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 139.81 | 140.73 | 141.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:15:00 | 138.47 | 140.73 | 141.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 12:15:00 | 131.55 | 134.35 | 136.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 134.33 | 133.42 | 135.13 | SL hit (close>ema200) qty=0.50 sl=133.42 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 12:15:00 | 135.95 | 135.22 | 135.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 13:15:00 | 137.98 | 135.77 | 135.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 136.44 | 136.49 | 135.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 136.44 | 136.49 | 135.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 136.00 | 136.39 | 135.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 136.00 | 136.39 | 135.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 136.53 | 136.42 | 135.95 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 133.80 | 135.61 | 135.71 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 09:15:00 | 140.49 | 135.25 | 135.09 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-02-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 12:15:00 | 133.73 | 136.06 | 136.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 09:15:00 | 132.74 | 134.77 | 135.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 10:15:00 | 135.38 | 134.89 | 135.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 11:00:00 | 135.38 | 134.89 | 135.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 136.44 | 135.24 | 135.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:00:00 | 136.44 | 135.24 | 135.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 136.72 | 135.54 | 135.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:45:00 | 136.00 | 135.54 | 135.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 14:15:00 | 138.25 | 136.08 | 135.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 10:15:00 | 138.78 | 137.04 | 136.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 141.80 | 141.85 | 140.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 141.80 | 141.85 | 140.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 144.45 | 142.28 | 141.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 10:15:00 | 144.80 | 142.28 | 141.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 11:15:00 | 144.84 | 142.71 | 141.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 14:15:00 | 139.93 | 142.28 | 141.74 | SL hit (close<static) qty=1.00 sl=141.11 alert=retest2 |

### Cycle 62 — SELL (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 10:15:00 | 140.82 | 141.43 | 141.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 10:15:00 | 139.20 | 140.41 | 140.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 14:15:00 | 140.23 | 139.72 | 140.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 14:15:00 | 140.23 | 139.72 | 140.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 140.23 | 139.72 | 140.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 140.23 | 139.72 | 140.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 140.30 | 139.83 | 140.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 139.69 | 139.83 | 140.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 140.39 | 139.95 | 140.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:45:00 | 139.10 | 139.88 | 140.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:15:00 | 139.30 | 139.88 | 140.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 12:15:00 | 139.24 | 139.79 | 140.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 13:00:00 | 139.00 | 139.63 | 140.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 140.11 | 139.39 | 139.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:45:00 | 140.64 | 139.39 | 139.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 139.79 | 139.47 | 139.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:45:00 | 139.82 | 139.47 | 139.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 140.57 | 139.69 | 139.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 12:00:00 | 140.57 | 139.69 | 139.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 139.99 | 139.75 | 139.87 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-17 14:15:00 | 140.02 | 139.94 | 139.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 14:15:00 | 140.02 | 139.94 | 139.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 141.60 | 140.36 | 140.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 154.90 | 156.98 | 154.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:00:00 | 154.90 | 156.98 | 154.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 154.44 | 156.13 | 154.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:00:00 | 154.44 | 156.13 | 154.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 155.17 | 155.94 | 154.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:30:00 | 154.70 | 155.94 | 154.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 155.11 | 155.77 | 154.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 14:45:00 | 154.94 | 155.77 | 154.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 154.25 | 155.47 | 154.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 156.18 | 155.47 | 154.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 155.13 | 155.40 | 154.71 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 153.04 | 154.41 | 154.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 152.64 | 154.06 | 154.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 154.86 | 154.14 | 154.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 154.86 | 154.14 | 154.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 154.86 | 154.14 | 154.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:00:00 | 154.86 | 154.14 | 154.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 152.93 | 153.90 | 154.15 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 155.71 | 154.54 | 154.39 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 10:15:00 | 153.74 | 154.18 | 154.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 12:15:00 | 152.78 | 153.85 | 154.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 151.96 | 151.44 | 152.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 10:15:00 | 151.96 | 151.44 | 152.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 151.96 | 151.44 | 152.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:00:00 | 151.96 | 151.44 | 152.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 151.88 | 151.65 | 152.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:30:00 | 152.14 | 151.65 | 152.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 152.04 | 151.65 | 152.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 152.04 | 151.65 | 152.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 152.25 | 151.77 | 152.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 152.76 | 151.77 | 152.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 152.28 | 151.87 | 152.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:15:00 | 153.31 | 151.87 | 152.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 153.35 | 152.17 | 152.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:30:00 | 153.25 | 152.17 | 152.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 153.67 | 152.47 | 152.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 15:15:00 | 154.20 | 153.21 | 152.78 | Break + close above crossover candle high |

### Cycle 68 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 149.54 | 152.47 | 152.49 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 14:15:00 | 153.67 | 152.63 | 152.52 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 146.63 | 151.60 | 152.08 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 154.46 | 151.29 | 151.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 154.93 | 153.31 | 152.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 173.50 | 177.92 | 176.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 173.50 | 177.92 | 176.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 173.50 | 177.92 | 176.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 173.50 | 177.92 | 176.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 172.39 | 176.81 | 175.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:45:00 | 172.25 | 176.81 | 175.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 14:15:00 | 172.91 | 174.76 | 175.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 171.95 | 174.19 | 174.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 14:15:00 | 167.50 | 167.03 | 168.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-29 15:00:00 | 167.50 | 167.03 | 168.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 167.20 | 167.06 | 168.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:15:00 | 165.57 | 167.06 | 168.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 11:00:00 | 166.32 | 166.88 | 168.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 13:45:00 | 166.81 | 166.81 | 167.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 14:15:00 | 166.75 | 166.21 | 166.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 166.28 | 166.22 | 166.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 15:15:00 | 165.91 | 166.22 | 166.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 15:15:00 | 167.40 | 167.04 | 167.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 167.40 | 167.04 | 167.03 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 165.32 | 166.70 | 166.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 164.70 | 165.96 | 166.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 09:15:00 | 166.09 | 165.65 | 166.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 166.09 | 165.65 | 166.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 166.09 | 165.65 | 166.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:30:00 | 166.30 | 165.65 | 166.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 168.20 | 166.16 | 166.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 168.20 | 166.16 | 166.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 11:15:00 | 168.35 | 166.60 | 166.55 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 165.51 | 166.95 | 166.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 14:15:00 | 163.91 | 166.34 | 166.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 167.70 | 163.51 | 164.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 167.70 | 163.51 | 164.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 167.70 | 163.51 | 164.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 167.70 | 163.51 | 164.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 169.27 | 165.40 | 165.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 169.74 | 166.86 | 165.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 170.70 | 170.71 | 168.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 170.70 | 170.71 | 168.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 175.65 | 177.04 | 175.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:45:00 | 175.43 | 177.04 | 175.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 12:15:00 | 175.46 | 176.72 | 175.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 12:45:00 | 175.00 | 176.72 | 175.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 13:15:00 | 175.36 | 176.45 | 175.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 13:30:00 | 175.06 | 176.45 | 175.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 174.77 | 176.12 | 175.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 15:00:00 | 174.77 | 176.12 | 175.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 175.00 | 175.89 | 175.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:15:00 | 174.90 | 175.89 | 175.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 174.72 | 175.35 | 175.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 11:15:00 | 175.85 | 175.35 | 175.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 11:45:00 | 175.37 | 175.32 | 175.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 12:15:00 | 175.81 | 175.32 | 175.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 15:15:00 | 175.58 | 175.28 | 175.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 175.58 | 175.34 | 175.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 174.84 | 175.34 | 175.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 175.11 | 175.29 | 175.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 175.20 | 175.29 | 175.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 175.16 | 175.27 | 175.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:15:00 | 173.97 | 175.27 | 175.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-20 11:15:00 | 173.70 | 174.95 | 175.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 11:15:00 | 173.70 | 174.95 | 175.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 172.82 | 174.53 | 174.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 173.14 | 173.06 | 173.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:00:00 | 173.14 | 173.06 | 173.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 173.87 | 173.22 | 173.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 173.87 | 173.22 | 173.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 171.30 | 172.84 | 173.70 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 175.07 | 173.51 | 173.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 176.49 | 174.33 | 173.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 11:15:00 | 174.29 | 174.65 | 174.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 11:15:00 | 174.29 | 174.65 | 174.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 174.29 | 174.65 | 174.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:00:00 | 174.29 | 174.65 | 174.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 174.73 | 174.67 | 174.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:45:00 | 174.50 | 174.67 | 174.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 174.50 | 174.63 | 174.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:00:00 | 174.50 | 174.63 | 174.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 174.44 | 174.59 | 174.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:30:00 | 174.21 | 174.59 | 174.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 174.25 | 174.53 | 174.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 171.78 | 174.53 | 174.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 09:15:00 | 171.12 | 173.84 | 173.92 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 14:15:00 | 174.21 | 173.32 | 173.32 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 13:15:00 | 172.71 | 173.25 | 173.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 14:15:00 | 171.87 | 172.97 | 173.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 173.25 | 171.77 | 172.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 173.25 | 171.77 | 172.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 173.25 | 171.77 | 172.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 173.25 | 171.77 | 172.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 174.09 | 172.23 | 172.33 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 11:15:00 | 174.52 | 172.69 | 172.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 12:15:00 | 175.99 | 173.35 | 172.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 10:15:00 | 174.02 | 174.43 | 173.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 11:00:00 | 174.02 | 174.43 | 173.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 172.80 | 173.99 | 173.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 172.80 | 173.99 | 173.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 172.90 | 173.78 | 173.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:45:00 | 172.67 | 173.78 | 173.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 173.30 | 173.65 | 173.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 175.41 | 173.65 | 173.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-09 09:15:00 | 192.95 | 189.17 | 184.59 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 188.43 | 191.19 | 191.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 14:15:00 | 187.87 | 190.52 | 191.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 188.22 | 186.60 | 187.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 188.22 | 186.60 | 187.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 188.22 | 186.60 | 187.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:45:00 | 188.38 | 186.60 | 187.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 190.48 | 187.38 | 188.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 190.48 | 187.38 | 188.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 190.34 | 187.97 | 188.36 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 191.68 | 188.71 | 188.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 193.52 | 190.53 | 189.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 12:15:00 | 189.90 | 191.05 | 190.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 12:15:00 | 189.90 | 191.05 | 190.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 189.90 | 191.05 | 190.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:00:00 | 189.90 | 191.05 | 190.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 190.08 | 190.85 | 190.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 14:00:00 | 190.08 | 190.85 | 190.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 190.22 | 190.73 | 190.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 14:30:00 | 189.49 | 190.73 | 190.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 189.71 | 190.52 | 190.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 191.00 | 190.52 | 190.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 192.21 | 190.86 | 190.28 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 14:15:00 | 188.64 | 189.86 | 189.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 11:15:00 | 187.38 | 188.96 | 189.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 189.90 | 188.42 | 188.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 189.90 | 188.42 | 188.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 189.90 | 188.42 | 188.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 189.60 | 188.42 | 188.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 191.60 | 189.06 | 189.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 191.60 | 189.06 | 189.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 189.42 | 189.23 | 189.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:15:00 | 189.62 | 189.23 | 189.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 13:15:00 | 189.50 | 189.29 | 189.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 190.00 | 189.43 | 189.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 189.32 | 189.52 | 189.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 189.32 | 189.52 | 189.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 189.32 | 189.52 | 189.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 10:15:00 | 190.54 | 189.52 | 189.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-02 09:15:00 | 209.59 | 207.30 | 205.86 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 11:15:00 | 204.25 | 205.80 | 205.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 12:15:00 | 203.64 | 205.37 | 205.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 209.00 | 205.33 | 205.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 209.00 | 205.33 | 205.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 209.00 | 205.33 | 205.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:45:00 | 209.25 | 205.33 | 205.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 10:15:00 | 206.93 | 205.65 | 205.58 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 204.47 | 205.42 | 205.48 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 13:15:00 | 205.89 | 205.54 | 205.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 15:15:00 | 206.10 | 205.64 | 205.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 207.16 | 207.63 | 206.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 09:15:00 | 207.16 | 207.63 | 206.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 207.16 | 207.63 | 206.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 207.16 | 207.63 | 206.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 206.27 | 207.36 | 206.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 206.27 | 207.36 | 206.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 206.11 | 207.11 | 206.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:45:00 | 206.22 | 207.11 | 206.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 206.75 | 207.09 | 206.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:00:00 | 206.75 | 207.09 | 206.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 208.18 | 207.31 | 207.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:30:00 | 208.60 | 207.82 | 207.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 12:15:00 | 206.33 | 208.24 | 208.18 | SL hit (close<static) qty=1.00 sl=206.48 alert=retest2 |

### Cycle 92 — SELL (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 13:15:00 | 206.80 | 207.95 | 208.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 205.01 | 207.06 | 207.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 207.41 | 205.76 | 206.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 207.41 | 205.76 | 206.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 207.41 | 205.76 | 206.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 207.41 | 205.76 | 206.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 207.50 | 206.11 | 206.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 206.95 | 206.11 | 206.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 205.85 | 203.36 | 203.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 205.85 | 203.36 | 203.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 10:15:00 | 208.70 | 204.43 | 203.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 205.91 | 208.13 | 206.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 205.91 | 208.13 | 206.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 205.91 | 208.13 | 206.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 205.91 | 208.13 | 206.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 207.05 | 207.91 | 206.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:30:00 | 206.89 | 207.91 | 206.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 210.41 | 211.78 | 210.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:00:00 | 210.41 | 211.78 | 210.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 210.30 | 211.48 | 210.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 12:30:00 | 211.43 | 211.32 | 210.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 204.90 | 209.78 | 209.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 204.90 | 209.78 | 209.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 14:15:00 | 202.00 | 204.41 | 206.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 204.49 | 203.99 | 205.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 10:00:00 | 204.49 | 203.99 | 205.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 204.01 | 204.01 | 205.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:30:00 | 205.05 | 204.01 | 205.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 205.40 | 204.29 | 205.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 205.40 | 204.29 | 205.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 206.00 | 204.63 | 205.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 205.91 | 204.63 | 205.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 205.90 | 204.88 | 205.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 205.59 | 204.88 | 205.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 202.90 | 203.43 | 204.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:45:00 | 203.68 | 203.43 | 204.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 203.95 | 203.53 | 204.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:00:00 | 203.95 | 203.53 | 204.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 204.00 | 203.63 | 204.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:45:00 | 204.14 | 203.63 | 204.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 203.98 | 203.70 | 204.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:30:00 | 204.09 | 203.70 | 204.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 203.79 | 203.72 | 204.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:30:00 | 204.18 | 203.72 | 204.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 203.15 | 203.08 | 203.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 203.15 | 203.08 | 203.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 200.64 | 201.69 | 202.63 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 205.07 | 202.81 | 202.59 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 201.77 | 202.69 | 202.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 200.15 | 201.81 | 202.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 200.38 | 199.86 | 200.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 200.38 | 199.86 | 200.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 201.16 | 200.12 | 201.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 201.30 | 200.12 | 201.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 198.23 | 199.74 | 200.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:30:00 | 197.90 | 199.42 | 200.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 11:45:00 | 198.10 | 199.22 | 200.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:30:00 | 197.84 | 198.84 | 200.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 15:00:00 | 198.10 | 196.77 | 197.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 197.75 | 196.96 | 197.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 197.29 | 196.96 | 197.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 197.52 | 197.08 | 197.87 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 202.35 | 198.83 | 198.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 202.35 | 198.83 | 198.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 210.35 | 203.19 | 201.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 14:15:00 | 222.77 | 222.78 | 220.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 14:30:00 | 222.80 | 222.78 | 220.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 219.44 | 222.11 | 220.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:00:00 | 219.44 | 222.11 | 220.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 219.94 | 221.67 | 220.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:30:00 | 219.10 | 221.67 | 220.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 15:15:00 | 217.89 | 219.37 | 219.50 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 10:15:00 | 220.13 | 219.61 | 219.59 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 218.26 | 219.57 | 219.63 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 10:15:00 | 220.08 | 219.67 | 219.67 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 11:15:00 | 219.54 | 219.65 | 219.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 12:15:00 | 218.50 | 219.42 | 219.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 218.75 | 218.31 | 218.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 218.75 | 218.31 | 218.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 218.75 | 218.31 | 218.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 10:30:00 | 218.40 | 218.31 | 218.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 219.03 | 218.45 | 218.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 219.70 | 218.45 | 218.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 218.00 | 218.36 | 218.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:30:00 | 217.40 | 218.27 | 218.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:15:00 | 217.55 | 218.27 | 218.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:00:00 | 217.21 | 218.06 | 218.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 220.40 | 218.36 | 218.61 | SL hit (close>static) qty=1.00 sl=219.10 alert=retest2 |

### Cycle 103 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 222.38 | 219.16 | 218.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 222.85 | 219.90 | 219.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 10:15:00 | 220.69 | 221.48 | 220.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 10:15:00 | 220.69 | 221.48 | 220.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 220.69 | 221.48 | 220.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:00:00 | 220.69 | 221.48 | 220.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 222.16 | 221.61 | 220.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:30:00 | 222.14 | 221.61 | 220.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 232.33 | 233.40 | 232.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 232.33 | 233.40 | 232.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 233.67 | 233.46 | 232.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 12:15:00 | 234.15 | 233.36 | 232.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:30:00 | 233.88 | 233.33 | 232.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 235.05 | 233.68 | 233.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 12:15:00 | 246.77 | 247.15 | 247.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 12:15:00 | 246.77 | 247.15 | 247.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 244.03 | 246.39 | 246.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 242.00 | 239.66 | 242.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 242.00 | 239.66 | 242.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 242.00 | 239.66 | 242.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 242.00 | 239.66 | 242.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 240.74 | 239.87 | 242.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:45:00 | 241.71 | 239.87 | 242.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 243.84 | 240.63 | 242.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:00:00 | 243.84 | 240.63 | 242.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 247.25 | 241.95 | 242.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 247.25 | 241.95 | 242.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 15:15:00 | 244.50 | 242.92 | 242.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 09:15:00 | 246.22 | 243.58 | 243.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 257.05 | 259.29 | 255.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 257.05 | 259.29 | 255.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 257.05 | 259.29 | 255.39 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 255.32 | 257.69 | 257.74 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 260.77 | 258.11 | 257.76 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 12:15:00 | 256.80 | 257.85 | 257.99 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 10:15:00 | 259.72 | 257.87 | 257.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 11:15:00 | 263.50 | 259.00 | 258.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 11:15:00 | 263.16 | 263.93 | 261.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 12:00:00 | 263.16 | 263.93 | 261.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 262.99 | 263.56 | 261.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 267.56 | 263.29 | 262.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 14:15:00 | 266.13 | 267.37 | 267.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 266.13 | 267.37 | 267.37 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 268.15 | 267.39 | 267.38 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 267.15 | 267.34 | 267.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 11:15:00 | 266.82 | 267.24 | 267.31 | Break + close below crossover candle low |

### Cycle 113 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 268.56 | 267.50 | 267.42 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 265.50 | 267.21 | 267.35 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 13:15:00 | 268.54 | 267.42 | 267.39 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 266.25 | 267.18 | 267.28 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 268.38 | 267.39 | 267.36 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 266.71 | 267.25 | 267.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 09:15:00 | 265.24 | 266.72 | 267.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 11:15:00 | 268.06 | 266.72 | 266.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 11:15:00 | 268.06 | 266.72 | 266.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 268.06 | 266.72 | 266.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:30:00 | 268.05 | 266.72 | 266.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 268.19 | 267.01 | 267.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:30:00 | 269.23 | 267.01 | 267.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 267.82 | 267.17 | 267.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 14:15:00 | 270.34 | 268.09 | 267.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 10:15:00 | 266.57 | 268.04 | 267.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 10:15:00 | 266.57 | 268.04 | 267.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 266.57 | 268.04 | 267.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:45:00 | 266.59 | 268.04 | 267.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 267.45 | 267.92 | 267.73 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 13:15:00 | 267.16 | 267.60 | 267.61 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 268.47 | 267.77 | 267.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 09:15:00 | 269.89 | 268.23 | 267.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 13:15:00 | 268.29 | 268.57 | 268.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 13:15:00 | 268.29 | 268.57 | 268.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 268.29 | 268.57 | 268.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 267.92 | 268.57 | 268.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 268.05 | 268.47 | 268.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 15:00:00 | 268.05 | 268.47 | 268.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 268.30 | 268.43 | 268.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 270.13 | 268.43 | 268.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 15:15:00 | 275.20 | 276.21 | 276.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 15:15:00 | 275.20 | 276.21 | 276.32 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 09:15:00 | 288.80 | 278.73 | 277.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 10:15:00 | 294.65 | 281.91 | 279.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 293.25 | 298.79 | 294.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 293.25 | 298.79 | 294.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 293.25 | 298.79 | 294.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 293.25 | 298.79 | 294.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 293.25 | 297.68 | 294.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 293.25 | 297.68 | 294.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 293.75 | 296.89 | 294.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 295.95 | 295.86 | 294.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 12:30:00 | 295.95 | 295.76 | 294.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 13:15:00 | 296.20 | 295.76 | 294.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 14:00:00 | 296.00 | 295.81 | 294.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 295.00 | 295.65 | 294.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:45:00 | 295.20 | 295.65 | 294.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 294.10 | 295.34 | 294.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 292.75 | 295.34 | 294.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 292.45 | 294.76 | 294.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:30:00 | 293.15 | 294.76 | 294.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-13 10:15:00 | 293.60 | 294.53 | 294.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 293.60 | 294.53 | 294.59 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 296.90 | 294.64 | 294.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 11:15:00 | 297.70 | 295.48 | 294.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 298.05 | 298.18 | 296.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 298.05 | 298.18 | 296.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 298.05 | 298.18 | 296.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:00:00 | 301.10 | 298.08 | 297.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 294.85 | 297.44 | 297.36 | SL hit (close<static) qty=1.00 sl=295.80 alert=retest2 |

### Cycle 126 — SELL (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 10:15:00 | 293.40 | 296.63 | 297.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 292.25 | 294.41 | 295.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 293.10 | 291.32 | 292.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 293.10 | 291.32 | 292.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 293.10 | 291.32 | 292.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 293.10 | 291.32 | 292.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 292.35 | 291.53 | 292.86 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 09:15:00 | 297.00 | 293.41 | 293.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 13:15:00 | 297.60 | 295.08 | 294.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 12:15:00 | 309.35 | 309.62 | 306.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 12:30:00 | 309.60 | 309.62 | 306.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 309.70 | 309.81 | 308.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:45:00 | 308.30 | 309.81 | 308.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 308.70 | 309.58 | 308.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 310.05 | 309.58 | 308.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 308.90 | 309.45 | 308.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 308.90 | 309.45 | 308.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 307.95 | 309.14 | 308.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:45:00 | 308.45 | 309.14 | 308.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 307.70 | 308.85 | 308.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:00:00 | 307.70 | 308.85 | 308.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 305.85 | 307.88 | 308.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 303.30 | 306.66 | 307.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 303.60 | 301.67 | 302.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 303.60 | 301.67 | 302.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 303.60 | 301.67 | 302.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 303.60 | 301.67 | 302.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 305.05 | 302.34 | 303.16 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 308.20 | 304.05 | 303.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 308.80 | 305.00 | 304.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 304.40 | 305.69 | 304.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 304.40 | 305.69 | 304.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 304.40 | 305.69 | 304.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 304.40 | 305.69 | 304.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 303.30 | 305.21 | 304.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 303.30 | 305.21 | 304.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 301.75 | 304.52 | 304.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 301.75 | 304.52 | 304.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 299.20 | 303.45 | 303.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 298.05 | 302.37 | 303.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 306.05 | 302.00 | 302.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 10:15:00 | 306.05 | 302.00 | 302.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 306.05 | 302.00 | 302.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 306.05 | 302.00 | 302.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 309.40 | 303.48 | 303.34 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 10:15:00 | 303.90 | 305.04 | 305.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 12:15:00 | 302.60 | 304.36 | 304.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 306.25 | 303.60 | 304.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 306.25 | 303.60 | 304.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 306.25 | 303.60 | 304.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 306.30 | 303.60 | 304.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 305.20 | 303.92 | 304.24 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 308.70 | 305.16 | 304.77 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 302.05 | 304.62 | 304.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 301.20 | 303.46 | 304.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 302.35 | 301.77 | 303.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 302.35 | 301.77 | 303.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 302.35 | 301.77 | 303.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 13:45:00 | 300.50 | 301.72 | 302.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 14:15:00 | 300.75 | 301.72 | 302.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:30:00 | 300.30 | 301.68 | 302.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:00:00 | 299.85 | 301.67 | 302.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 299.00 | 300.63 | 301.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 13:00:00 | 297.15 | 299.71 | 300.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 14:15:00 | 298.70 | 299.54 | 300.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 13:15:00 | 302.20 | 300.96 | 300.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 302.20 | 300.96 | 300.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 303.55 | 301.48 | 301.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 12:15:00 | 304.75 | 305.56 | 304.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 13:00:00 | 304.75 | 305.56 | 304.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 303.45 | 305.14 | 303.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 303.45 | 305.14 | 303.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 303.65 | 304.84 | 303.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:45:00 | 303.35 | 304.84 | 303.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 302.65 | 304.40 | 303.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 304.50 | 304.40 | 303.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 300.20 | 303.22 | 303.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 299.85 | 301.40 | 302.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 301.25 | 300.78 | 301.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 10:00:00 | 301.25 | 300.78 | 301.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 301.50 | 300.93 | 301.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:15:00 | 301.95 | 300.93 | 301.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 300.25 | 300.79 | 301.41 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 306.70 | 302.50 | 302.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 313.00 | 304.92 | 303.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 318.35 | 318.81 | 316.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 09:45:00 | 318.85 | 318.81 | 316.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 319.50 | 318.95 | 316.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 317.80 | 318.95 | 316.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 317.85 | 320.49 | 319.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:00:00 | 317.85 | 320.49 | 319.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 319.60 | 320.31 | 319.23 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 313.15 | 317.78 | 318.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 13:15:00 | 312.50 | 315.61 | 317.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 10:15:00 | 309.10 | 308.85 | 311.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 11:00:00 | 309.10 | 308.85 | 311.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 293.30 | 296.87 | 301.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 291.55 | 295.13 | 299.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 292.30 | 291.67 | 294.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 13:15:00 | 298.10 | 295.18 | 295.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 13:15:00 | 298.10 | 295.18 | 295.18 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 285.95 | 293.62 | 294.49 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-01-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 13:15:00 | 298.90 | 294.62 | 294.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-20 09:15:00 | 304.55 | 298.20 | 296.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 12:15:00 | 296.30 | 298.59 | 297.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 12:15:00 | 296.30 | 298.59 | 297.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 296.30 | 298.59 | 297.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 13:00:00 | 296.30 | 298.59 | 297.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 296.85 | 298.24 | 297.07 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2026-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 09:15:00 | 281.00 | 293.57 | 295.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 13:15:00 | 279.80 | 287.09 | 291.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 286.55 | 285.50 | 289.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 12:15:00 | 285.50 | 285.63 | 288.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 285.50 | 285.63 | 288.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:45:00 | 287.05 | 285.63 | 288.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 288.85 | 286.57 | 288.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:15:00 | 289.90 | 286.57 | 288.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 293.35 | 287.93 | 288.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:00:00 | 293.35 | 287.93 | 288.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 292.95 | 288.93 | 289.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:30:00 | 293.60 | 288.93 | 289.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 11:15:00 | 293.55 | 289.86 | 289.44 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 09:15:00 | 287.45 | 289.14 | 289.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 10:15:00 | 284.40 | 288.19 | 288.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 286.05 | 285.67 | 287.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 15:00:00 | 286.05 | 285.67 | 287.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 289.80 | 286.50 | 287.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 289.20 | 286.50 | 287.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 287.15 | 286.63 | 287.47 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 289.50 | 288.07 | 287.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 291.65 | 288.78 | 288.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 11:15:00 | 286.60 | 288.65 | 288.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 11:15:00 | 286.60 | 288.65 | 288.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 286.60 | 288.65 | 288.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 286.60 | 288.65 | 288.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 288.05 | 288.53 | 288.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:45:00 | 288.90 | 288.38 | 288.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 14:30:00 | 288.75 | 288.50 | 288.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 288.95 | 288.80 | 288.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:00:00 | 289.60 | 288.80 | 288.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 291.95 | 289.43 | 288.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:30:00 | 288.70 | 289.43 | 288.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 289.10 | 289.90 | 289.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 289.10 | 289.90 | 289.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 289.85 | 289.89 | 289.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:15:00 | 287.65 | 289.89 | 289.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 286.25 | 289.16 | 288.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 15:00:00 | 286.25 | 289.16 | 288.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-30 15:15:00 | 286.00 | 288.53 | 288.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2026-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 15:15:00 | 286.00 | 288.53 | 288.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 13:15:00 | 281.65 | 285.83 | 287.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 277.50 | 276.44 | 280.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 277.50 | 276.44 | 280.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 284.00 | 278.22 | 280.66 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 284.95 | 281.97 | 281.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 289.40 | 284.28 | 283.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 15:15:00 | 286.60 | 286.67 | 285.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:15:00 | 282.00 | 286.67 | 285.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 280.85 | 285.50 | 284.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 279.35 | 285.50 | 284.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 283.20 | 285.04 | 284.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 280.80 | 285.04 | 284.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 283.00 | 284.09 | 284.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 279.05 | 282.82 | 283.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 285.60 | 281.32 | 282.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 14:15:00 | 285.60 | 281.32 | 282.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 285.60 | 281.32 | 282.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 285.60 | 281.32 | 282.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 285.00 | 282.06 | 282.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 290.40 | 282.06 | 282.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 293.80 | 284.41 | 283.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 296.35 | 286.80 | 284.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 294.50 | 294.78 | 291.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 13:30:00 | 294.50 | 294.78 | 291.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 288.40 | 293.25 | 291.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 286.95 | 293.25 | 291.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 287.80 | 292.16 | 291.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:45:00 | 287.10 | 292.16 | 291.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 13:15:00 | 288.55 | 290.33 | 290.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 15:15:00 | 287.75 | 289.49 | 290.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 09:15:00 | 290.60 | 289.71 | 290.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 290.60 | 289.71 | 290.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 290.60 | 289.71 | 290.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 289.85 | 289.71 | 290.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 288.85 | 289.54 | 290.03 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 291.80 | 290.58 | 290.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 15:15:00 | 292.40 | 290.94 | 290.59 | Break + close above crossover candle high |

### Cycle 152 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 284.90 | 289.73 | 290.07 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 12:15:00 | 291.70 | 289.38 | 289.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 15:15:00 | 295.10 | 291.50 | 290.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 10:15:00 | 291.10 | 292.23 | 290.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-17 11:00:00 | 291.10 | 292.23 | 290.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 297.10 | 293.20 | 291.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 12:15:00 | 298.30 | 293.20 | 291.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 14:30:00 | 297.75 | 295.51 | 293.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 10:00:00 | 297.50 | 298.82 | 296.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 14:45:00 | 297.55 | 297.63 | 296.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 294.60 | 297.03 | 296.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 293.30 | 297.03 | 296.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 297.20 | 297.02 | 296.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 12:00:00 | 298.30 | 297.27 | 296.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 12:15:00 | 296.70 | 297.16 | 296.88 | SL hit (close<static) qty=1.00 sl=296.75 alert=retest2 |

### Cycle 154 — SELL (started 2026-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 14:15:00 | 297.10 | 297.90 | 297.95 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 299.00 | 298.12 | 298.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 308.00 | 300.10 | 298.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 302.50 | 302.59 | 300.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 13:00:00 | 302.50 | 302.59 | 300.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 302.20 | 302.84 | 301.36 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 13:15:00 | 297.95 | 300.09 | 300.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 289.10 | 297.66 | 299.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 10:15:00 | 272.75 | 272.28 | 277.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 11:00:00 | 272.75 | 272.28 | 277.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 275.00 | 272.80 | 276.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 276.40 | 272.80 | 276.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 276.00 | 273.44 | 276.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 276.85 | 273.44 | 276.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 274.90 | 273.73 | 275.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 276.00 | 273.73 | 275.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 275.80 | 274.09 | 275.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:30:00 | 273.20 | 273.93 | 275.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 13:45:00 | 272.70 | 270.39 | 270.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 14:15:00 | 273.55 | 271.03 | 270.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 273.55 | 271.03 | 270.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 274.90 | 271.80 | 271.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 11:15:00 | 271.10 | 271.79 | 271.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 11:15:00 | 271.10 | 271.79 | 271.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 271.10 | 271.79 | 271.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 12:00:00 | 271.10 | 271.79 | 271.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 269.20 | 271.27 | 271.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 269.20 | 271.27 | 271.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 268.60 | 270.74 | 271.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 266.20 | 269.83 | 270.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 269.00 | 268.04 | 269.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 12:00:00 | 269.00 | 268.04 | 269.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 267.35 | 267.90 | 269.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 14:30:00 | 265.30 | 267.20 | 268.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 271.25 | 262.81 | 261.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 271.25 | 262.81 | 261.82 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 258.60 | 263.58 | 263.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 257.25 | 262.31 | 263.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 261.80 | 260.64 | 261.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 10:15:00 | 261.80 | 260.64 | 261.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 261.80 | 260.64 | 261.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 260.90 | 260.64 | 261.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 261.75 | 260.86 | 261.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:00:00 | 261.75 | 260.86 | 261.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 263.35 | 261.36 | 262.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:45:00 | 263.25 | 261.36 | 262.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 261.90 | 261.47 | 262.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:15:00 | 261.50 | 261.47 | 262.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 248.42 | 258.50 | 260.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 253.10 | 251.26 | 254.94 | SL hit (close>ema200) qty=0.50 sl=251.26 alert=retest2 |

### Cycle 161 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 265.60 | 257.06 | 255.96 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 251.10 | 256.33 | 256.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 243.75 | 251.80 | 254.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 245.39 | 244.29 | 248.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 245.39 | 244.29 | 248.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 245.39 | 244.29 | 248.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 244.54 | 244.28 | 247.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:30:00 | 244.67 | 245.26 | 247.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:00:00 | 245.18 | 242.61 | 243.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 251.44 | 245.30 | 244.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 251.44 | 245.30 | 244.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 254.04 | 247.05 | 245.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 272.24 | 273.29 | 267.39 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 275.46 | 272.84 | 268.20 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 272.82 | 275.85 | 272.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:45:00 | 274.86 | 275.70 | 273.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 280.67 | 275.28 | 273.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 09:15:00 | 289.23 | 285.59 | 282.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 289.40 | 289.72 | 286.62 | SL hit (close<ema200) qty=0.50 sl=289.72 alert=retest1 |

### Cycle 164 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 289.25 | 290.94 | 291.11 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 292.48 | 291.20 | 291.17 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 10:15:00 | 290.30 | 291.02 | 291.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 15:15:00 | 287.45 | 289.33 | 290.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 288.54 | 287.27 | 288.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 288.54 | 287.27 | 288.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 288.54 | 287.27 | 288.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:00:00 | 288.54 | 287.27 | 288.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 288.24 | 287.47 | 288.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:45:00 | 286.15 | 287.54 | 288.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:30:00 | 286.35 | 284.07 | 284.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 15:15:00 | 286.95 | 284.98 | 284.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 286.95 | 284.98 | 284.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 287.45 | 285.47 | 285.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 10:15:00 | 285.05 | 285.39 | 285.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 10:15:00 | 285.05 | 285.39 | 285.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 285.05 | 285.39 | 285.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:00:00 | 285.05 | 285.39 | 285.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 285.80 | 285.47 | 285.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 285.25 | 285.47 | 285.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-24 10:00:00 | 158.00 | 2024-05-24 10:15:00 | 158.75 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-05-31 14:45:00 | 152.80 | 2024-06-03 09:15:00 | 159.35 | STOP_HIT | 1.00 | -4.29% |
| BUY | retest2 | 2024-06-25 09:15:00 | 181.48 | 2024-06-25 12:15:00 | 180.76 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2024-06-28 12:15:00 | 180.60 | 2024-07-01 09:15:00 | 185.03 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2024-07-05 11:15:00 | 188.27 | 2024-07-08 11:15:00 | 186.09 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest1 | 2024-07-11 10:15:00 | 180.31 | 2024-07-12 09:15:00 | 183.77 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest1 | 2024-07-11 11:15:00 | 179.96 | 2024-07-12 09:15:00 | 183.77 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2024-07-12 11:15:00 | 180.81 | 2024-07-15 10:15:00 | 183.74 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-07-12 12:30:00 | 181.03 | 2024-07-15 10:15:00 | 183.74 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-07-15 09:30:00 | 180.80 | 2024-07-15 10:15:00 | 183.74 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-07-16 14:30:00 | 184.80 | 2024-07-19 09:15:00 | 180.23 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2024-07-18 09:15:00 | 186.92 | 2024-07-19 09:15:00 | 180.23 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2024-07-18 11:45:00 | 185.22 | 2024-07-19 09:15:00 | 180.23 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2024-07-18 15:15:00 | 184.72 | 2024-07-19 09:15:00 | 180.23 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-07-24 11:15:00 | 175.74 | 2024-07-26 10:15:00 | 178.40 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-08-01 09:15:00 | 181.00 | 2024-08-01 11:15:00 | 178.66 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-08-01 11:00:00 | 180.67 | 2024-08-01 11:15:00 | 178.66 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-08-06 14:00:00 | 168.85 | 2024-08-14 09:15:00 | 160.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-07 09:30:00 | 169.05 | 2024-08-14 09:15:00 | 160.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-06 14:00:00 | 168.85 | 2024-08-16 09:15:00 | 162.73 | STOP_HIT | 0.50 | 3.62% |
| SELL | retest2 | 2024-08-07 09:30:00 | 169.05 | 2024-08-16 09:15:00 | 162.73 | STOP_HIT | 0.50 | 3.74% |
| BUY | retest2 | 2024-08-23 11:15:00 | 168.50 | 2024-08-23 14:15:00 | 167.53 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-09-17 11:30:00 | 176.72 | 2024-09-19 11:15:00 | 173.97 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-09-17 12:30:00 | 176.97 | 2024-09-19 11:15:00 | 173.97 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-09-18 09:15:00 | 177.40 | 2024-09-19 11:15:00 | 173.97 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-09-18 10:45:00 | 176.71 | 2024-09-19 11:15:00 | 173.97 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-09-27 14:45:00 | 188.66 | 2024-09-30 15:15:00 | 185.50 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-09-30 11:15:00 | 187.02 | 2024-09-30 15:15:00 | 185.50 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-09-30 13:45:00 | 186.99 | 2024-09-30 15:15:00 | 185.50 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-10-01 09:30:00 | 186.87 | 2024-10-03 09:15:00 | 183.70 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-10-01 12:15:00 | 188.91 | 2024-10-03 09:15:00 | 183.70 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2024-10-09 11:45:00 | 172.90 | 2024-10-11 12:15:00 | 164.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-09 12:30:00 | 173.17 | 2024-10-11 12:15:00 | 164.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-09 11:45:00 | 172.90 | 2024-10-14 09:15:00 | 165.88 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2024-10-09 12:30:00 | 173.17 | 2024-10-14 09:15:00 | 165.88 | STOP_HIT | 0.50 | 4.21% |
| SELL | retest2 | 2024-10-18 09:15:00 | 162.18 | 2024-10-18 11:15:00 | 167.60 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2024-10-29 09:30:00 | 140.79 | 2024-10-30 10:15:00 | 146.44 | STOP_HIT | 1.00 | -4.01% |
| SELL | retest2 | 2024-10-29 10:00:00 | 140.98 | 2024-10-30 10:15:00 | 146.44 | STOP_HIT | 1.00 | -3.87% |
| SELL | retest2 | 2024-11-05 15:15:00 | 145.01 | 2024-11-06 09:15:00 | 147.02 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-11-18 09:15:00 | 136.56 | 2024-11-18 11:15:00 | 138.80 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-11-26 14:30:00 | 140.52 | 2024-12-12 11:15:00 | 147.59 | STOP_HIT | 1.00 | 5.03% |
| BUY | retest2 | 2024-11-27 12:15:00 | 140.60 | 2024-12-12 11:15:00 | 147.59 | STOP_HIT | 1.00 | 4.97% |
| SELL | retest2 | 2024-12-17 09:15:00 | 145.83 | 2024-12-19 09:15:00 | 138.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 09:15:00 | 145.83 | 2024-12-19 10:15:00 | 141.54 | STOP_HIT | 0.50 | 2.94% |
| BUY | retest2 | 2025-01-09 15:15:00 | 144.00 | 2025-01-10 09:15:00 | 139.00 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2025-01-27 09:15:00 | 135.27 | 2025-01-27 14:15:00 | 141.67 | STOP_HIT | 1.00 | -4.73% |
| SELL | retest2 | 2025-01-28 10:45:00 | 137.50 | 2025-01-28 12:15:00 | 143.36 | STOP_HIT | 1.00 | -4.26% |
| BUY | retest2 | 2025-01-31 09:15:00 | 145.11 | 2025-02-07 13:15:00 | 148.75 | STOP_HIT | 1.00 | 2.51% |
| BUY | retest2 | 2025-02-01 13:15:00 | 144.82 | 2025-02-07 13:15:00 | 148.75 | STOP_HIT | 1.00 | 2.71% |
| BUY | retest2 | 2025-02-03 10:15:00 | 144.99 | 2025-02-07 13:15:00 | 148.75 | STOP_HIT | 1.00 | 2.59% |
| SELL | retest2 | 2025-02-14 10:15:00 | 138.47 | 2025-02-18 12:15:00 | 131.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 10:15:00 | 138.47 | 2025-02-19 09:15:00 | 134.33 | STOP_HIT | 0.50 | 2.99% |
| BUY | retest2 | 2025-03-10 10:15:00 | 144.80 | 2025-03-10 14:15:00 | 139.93 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2025-03-10 11:15:00 | 144.84 | 2025-03-10 14:15:00 | 139.93 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2025-03-13 10:45:00 | 139.10 | 2025-03-17 14:15:00 | 140.02 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-03-13 11:15:00 | 139.30 | 2025-03-17 14:15:00 | 140.02 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-03-13 12:15:00 | 139.24 | 2025-03-17 14:15:00 | 140.02 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-03-13 13:00:00 | 139.00 | 2025-03-17 14:15:00 | 140.02 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-04-30 09:15:00 | 165.57 | 2025-05-05 15:15:00 | 167.40 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-04-30 11:00:00 | 166.32 | 2025-05-05 15:15:00 | 167.40 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-04-30 13:45:00 | 166.81 | 2025-05-05 15:15:00 | 167.40 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-05-02 14:15:00 | 166.75 | 2025-05-05 15:15:00 | 167.40 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-05-02 15:15:00 | 165.91 | 2025-05-05 15:15:00 | 167.40 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-05-19 11:15:00 | 175.85 | 2025-05-20 11:15:00 | 173.70 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-05-19 11:45:00 | 175.37 | 2025-05-20 11:15:00 | 173.70 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-05-19 12:15:00 | 175.81 | 2025-05-20 11:15:00 | 173.70 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-05-19 15:15:00 | 175.58 | 2025-05-20 11:15:00 | 173.70 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-06-04 09:15:00 | 175.41 | 2025-06-09 09:15:00 | 192.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-23 10:15:00 | 190.54 | 2025-07-02 09:15:00 | 209.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-09 09:30:00 | 208.60 | 2025-07-10 12:15:00 | 206.33 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-07-14 11:15:00 | 206.95 | 2025-07-21 09:15:00 | 205.85 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2025-07-24 12:30:00 | 211.43 | 2025-07-25 09:15:00 | 204.90 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-08-08 10:30:00 | 197.90 | 2025-08-13 09:15:00 | 202.35 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-08-08 11:45:00 | 198.10 | 2025-08-13 09:15:00 | 202.35 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-08-08 12:30:00 | 197.84 | 2025-08-13 09:15:00 | 202.35 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-08-11 15:00:00 | 198.10 | 2025-08-13 09:15:00 | 202.35 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-08-29 13:30:00 | 217.40 | 2025-09-01 09:15:00 | 220.40 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-08-29 14:15:00 | 217.55 | 2025-09-01 09:15:00 | 220.40 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-08-29 15:00:00 | 217.21 | 2025-09-01 09:15:00 | 220.40 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-09-10 12:15:00 | 234.15 | 2025-09-25 12:15:00 | 246.77 | STOP_HIT | 1.00 | 5.39% |
| BUY | retest2 | 2025-09-11 09:30:00 | 233.88 | 2025-09-25 12:15:00 | 246.77 | STOP_HIT | 1.00 | 5.51% |
| BUY | retest2 | 2025-09-12 09:15:00 | 235.05 | 2025-09-25 12:15:00 | 246.77 | STOP_HIT | 1.00 | 4.99% |
| BUY | retest2 | 2025-10-15 09:15:00 | 267.56 | 2025-10-17 14:15:00 | 266.13 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-10-31 09:15:00 | 270.13 | 2025-11-06 15:15:00 | 275.20 | STOP_HIT | 1.00 | 1.88% |
| BUY | retest2 | 2025-11-11 15:15:00 | 295.95 | 2025-11-13 10:15:00 | 293.60 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-11-12 12:30:00 | 295.95 | 2025-11-13 10:15:00 | 293.60 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-11-12 13:15:00 | 296.20 | 2025-11-13 10:15:00 | 293.60 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-11-12 14:00:00 | 296.00 | 2025-11-13 10:15:00 | 293.60 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-11-19 10:00:00 | 301.10 | 2025-11-20 09:15:00 | 294.85 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-12-17 13:45:00 | 300.50 | 2025-12-22 13:15:00 | 302.20 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-12-17 14:15:00 | 300.75 | 2025-12-22 13:15:00 | 302.20 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-12-18 09:30:00 | 300.30 | 2025-12-22 13:15:00 | 302.20 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-12-18 14:00:00 | 299.85 | 2025-12-22 13:15:00 | 302.20 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-12-19 13:00:00 | 297.15 | 2025-12-22 13:15:00 | 302.20 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-12-19 14:15:00 | 298.70 | 2025-12-22 13:15:00 | 302.20 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-01-13 11:30:00 | 291.55 | 2026-01-16 13:15:00 | 298.10 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2026-01-14 14:15:00 | 292.30 | 2026-01-16 13:15:00 | 298.10 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-01-29 13:45:00 | 288.90 | 2026-01-30 15:15:00 | 286.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-01-29 14:30:00 | 288.75 | 2026-01-30 15:15:00 | 286.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2026-01-30 09:30:00 | 288.95 | 2026-01-30 15:15:00 | 286.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2026-01-30 10:00:00 | 289.60 | 2026-01-30 15:15:00 | 286.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-02-17 12:15:00 | 298.30 | 2026-02-20 12:15:00 | 296.70 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2026-02-17 14:30:00 | 297.75 | 2026-02-24 09:15:00 | 295.80 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2026-02-19 10:00:00 | 297.50 | 2026-02-24 09:15:00 | 295.80 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-02-19 14:45:00 | 297.55 | 2026-02-24 14:15:00 | 297.10 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2026-02-20 12:00:00 | 298.30 | 2026-02-24 14:15:00 | 297.10 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2026-02-20 13:45:00 | 298.20 | 2026-02-24 14:15:00 | 297.10 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2026-02-23 09:15:00 | 301.95 | 2026-02-24 14:15:00 | 297.10 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2026-02-24 11:00:00 | 299.15 | 2026-02-24 14:15:00 | 297.10 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-03-06 14:30:00 | 273.20 | 2026-03-10 14:15:00 | 273.55 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2026-03-10 13:45:00 | 272.70 | 2026-03-10 14:15:00 | 273.55 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2026-03-12 14:30:00 | 265.30 | 2026-03-18 09:15:00 | 271.25 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2026-03-20 14:15:00 | 261.50 | 2026-03-23 09:15:00 | 248.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 14:15:00 | 261.50 | 2026-03-24 09:15:00 | 253.10 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2026-04-01 10:45:00 | 244.54 | 2026-04-06 12:15:00 | 251.44 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2026-04-01 14:30:00 | 244.67 | 2026-04-06 12:15:00 | 251.44 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2026-04-06 11:00:00 | 245.18 | 2026-04-06 12:15:00 | 251.44 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest1 | 2026-04-10 09:15:00 | 275.46 | 2026-04-20 09:15:00 | 289.23 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-10 09:15:00 | 275.46 | 2026-04-20 15:15:00 | 289.40 | STOP_HIT | 0.50 | 5.06% |
| BUY | retest2 | 2026-04-13 11:45:00 | 274.86 | 2026-04-24 13:15:00 | 289.25 | STOP_HIT | 1.00 | 5.24% |
| BUY | retest2 | 2026-04-15 09:15:00 | 280.67 | 2026-04-24 13:15:00 | 289.25 | STOP_HIT | 1.00 | 3.06% |
| SELL | retest2 | 2026-04-29 13:45:00 | 286.15 | 2026-05-04 15:15:00 | 286.95 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2026-05-04 12:30:00 | 286.35 | 2026-05-04 15:15:00 | 286.95 | STOP_HIT | 1.00 | -0.21% |
