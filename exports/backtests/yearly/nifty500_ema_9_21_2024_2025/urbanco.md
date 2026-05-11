# Urban Company Ltd. (URBANCO)

## Backtest Summary

- **Window:** 2025-09-17 09:15:00 → 2026-05-08 15:15:00 (1094 bars)
- **Last close:** 137.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 51 |
| ALERT1 | 32 |
| ALERT2 | 32 |
| ALERT2_SKIP | 17 |
| ALERT3 | 90 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 34 |
| PARTIAL | 10 |
| TARGET_HIT | 9 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 46 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 30 / 16
- **Target hits / Stop hits / Partials:** 9 / 27 / 10
- **Avg / median % per leg:** 3.67% / 4.95%
- **Sum % (uncompounded):** 168.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 10 | 58.8% | 9 | 8 | 0 | 4.74% | 80.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 17 | 10 | 58.8% | 9 | 8 | 0 | 4.74% | 80.5% |
| SELL (all) | 29 | 20 | 69.0% | 0 | 19 | 10 | 3.04% | 88.2% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.45% | -7.3% |
| SELL @ 3rd Alert (retest2) | 26 | 20 | 76.9% | 0 | 16 | 10 | 3.68% | 95.6% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.45% | -7.3% |
| retest2 (combined) | 43 | 30 | 69.8% | 9 | 24 | 10 | 4.10% | 176.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 13:15:00 | 178.10 | 180.12 | 180.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 177.01 | 179.50 | 179.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 09:15:00 | 175.30 | 173.59 | 175.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 175.30 | 173.59 | 175.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 175.30 | 173.59 | 175.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:00:00 | 175.30 | 173.59 | 175.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 171.40 | 171.91 | 173.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 172.90 | 171.91 | 173.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 168.06 | 167.94 | 169.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:45:00 | 166.46 | 167.51 | 169.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 171.55 | 168.55 | 168.99 | SL hit (close>static) qty=1.00 sl=170.75 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 13:15:00 | 172.47 | 169.82 | 169.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 174.40 | 170.74 | 169.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 171.19 | 171.58 | 170.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:45:00 | 171.25 | 171.58 | 170.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 170.15 | 171.29 | 170.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:15:00 | 169.95 | 171.29 | 170.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 170.05 | 171.04 | 170.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 13:15:00 | 170.89 | 171.04 | 170.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 13:15:00 | 169.36 | 170.71 | 170.40 | SL hit (close<static) qty=1.00 sl=169.51 alert=retest2 |

### Cycle 3 — SELL (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 15:15:00 | 167.24 | 169.62 | 169.93 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 12:15:00 | 171.25 | 170.08 | 170.03 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 169.09 | 169.92 | 169.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 168.28 | 169.59 | 169.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 157.41 | 153.05 | 155.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 157.41 | 153.05 | 155.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 157.41 | 153.05 | 155.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 157.41 | 153.05 | 155.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 158.09 | 154.05 | 155.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 161.77 | 154.05 | 155.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 160.15 | 156.56 | 156.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 14:15:00 | 164.51 | 158.15 | 157.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 158.50 | 159.65 | 158.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 158.50 | 159.65 | 158.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 158.50 | 159.65 | 158.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 158.77 | 159.65 | 158.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 156.86 | 159.09 | 157.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:45:00 | 156.50 | 159.09 | 157.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 159.50 | 159.17 | 158.06 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 10:15:00 | 154.05 | 157.22 | 157.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 153.56 | 156.03 | 156.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 13:15:00 | 154.33 | 152.89 | 154.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 13:15:00 | 154.33 | 152.89 | 154.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 154.33 | 152.89 | 154.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:00:00 | 154.33 | 152.89 | 154.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 154.39 | 153.19 | 154.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 15:00:00 | 154.39 | 153.19 | 154.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 154.00 | 153.35 | 154.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 157.00 | 154.23 | 154.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 158.25 | 155.03 | 155.00 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 153.60 | 154.75 | 154.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 10:15:00 | 152.73 | 154.34 | 154.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 148.60 | 148.26 | 150.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 09:45:00 | 148.61 | 148.26 | 150.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 149.50 | 148.51 | 150.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:00:00 | 149.50 | 148.51 | 150.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 153.66 | 149.54 | 150.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:00:00 | 153.66 | 149.54 | 150.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 151.10 | 149.85 | 150.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 13:30:00 | 150.65 | 149.79 | 150.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 11:30:00 | 150.71 | 149.42 | 149.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:15:00 | 149.49 | 149.42 | 149.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 151.80 | 150.32 | 150.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 14:15:00 | 151.80 | 150.32 | 150.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 152.86 | 150.97 | 150.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 11:15:00 | 153.58 | 153.93 | 152.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 12:00:00 | 153.58 | 153.93 | 152.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 153.30 | 154.01 | 153.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 153.30 | 154.01 | 153.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 152.39 | 153.69 | 153.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 152.39 | 153.69 | 153.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 153.62 | 153.68 | 153.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 12:45:00 | 154.03 | 153.77 | 153.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 152.42 | 155.07 | 154.23 | SL hit (close<static) qty=1.00 sl=152.50 alert=retest2 |

### Cycle 11 — SELL (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 12:15:00 | 151.17 | 153.36 | 153.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 149.50 | 152.04 | 152.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 09:15:00 | 143.15 | 137.98 | 140.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 143.15 | 137.98 | 140.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 143.15 | 137.98 | 140.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 143.15 | 137.98 | 140.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 143.35 | 139.05 | 140.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:15:00 | 144.90 | 139.05 | 140.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 145.84 | 142.24 | 142.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 15:15:00 | 146.85 | 143.16 | 142.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 13:15:00 | 144.21 | 144.31 | 143.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 14:15:00 | 143.01 | 144.05 | 143.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 143.01 | 144.05 | 143.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 143.01 | 144.05 | 143.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 142.39 | 143.72 | 143.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 142.95 | 143.72 | 143.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 141.65 | 143.06 | 143.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:45:00 | 141.81 | 143.06 | 143.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 11:15:00 | 142.50 | 142.95 | 143.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 12:15:00 | 141.50 | 142.66 | 142.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 10:15:00 | 142.17 | 141.38 | 141.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 10:15:00 | 142.17 | 141.38 | 141.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 142.17 | 141.38 | 141.77 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 13:15:00 | 145.42 | 142.49 | 142.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 10:15:00 | 145.88 | 144.24 | 143.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 10:15:00 | 146.28 | 146.79 | 145.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 11:00:00 | 146.28 | 146.79 | 145.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 145.17 | 146.32 | 145.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:00:00 | 145.17 | 146.32 | 145.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 144.00 | 145.86 | 145.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 143.58 | 145.86 | 145.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 144.77 | 145.39 | 145.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 11:45:00 | 145.81 | 145.43 | 145.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 14:15:00 | 143.90 | 145.18 | 145.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 14:15:00 | 143.90 | 145.18 | 145.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 142.31 | 144.45 | 144.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 138.73 | 136.94 | 138.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 138.73 | 136.94 | 138.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 138.73 | 136.94 | 138.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 12:30:00 | 136.14 | 137.13 | 137.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 14:30:00 | 136.04 | 136.80 | 137.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 10:45:00 | 135.75 | 136.16 | 137.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 10:30:00 | 135.90 | 135.66 | 136.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 135.39 | 135.37 | 135.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 135.39 | 135.37 | 135.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 135.11 | 135.32 | 135.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 134.94 | 135.25 | 135.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 129.33 | 130.15 | 131.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 129.24 | 130.15 | 131.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 128.96 | 130.15 | 131.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 129.10 | 130.15 | 131.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 128.19 | 130.15 | 131.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 130.87 | 129.97 | 131.25 | SL hit (close>ema200) qty=0.50 sl=129.97 alert=retest2 |

### Cycle 16 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 131.56 | 129.35 | 129.20 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 12:15:00 | 128.03 | 129.04 | 129.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 127.35 | 128.34 | 128.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 12:15:00 | 127.95 | 127.20 | 127.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 12:15:00 | 127.95 | 127.20 | 127.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 127.95 | 127.20 | 127.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:30:00 | 127.85 | 127.20 | 127.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 130.29 | 127.82 | 127.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:00:00 | 130.29 | 127.82 | 127.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 14:15:00 | 128.72 | 128.00 | 127.98 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 126.50 | 127.90 | 127.95 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 125.80 | 124.01 | 123.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 128.19 | 125.16 | 124.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 11:15:00 | 126.10 | 126.20 | 125.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 12:00:00 | 126.10 | 126.20 | 125.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 124.95 | 125.81 | 125.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 14:45:00 | 125.07 | 125.81 | 125.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 124.85 | 125.62 | 125.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 124.20 | 125.62 | 125.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 125.10 | 125.47 | 125.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:45:00 | 126.25 | 125.62 | 125.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:30:00 | 126.00 | 125.65 | 125.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 13:30:00 | 126.11 | 125.64 | 125.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 14:15:00 | 126.31 | 125.64 | 125.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 124.81 | 127.57 | 127.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 124.81 | 127.57 | 127.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 125.53 | 127.17 | 126.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 11:45:00 | 126.55 | 127.08 | 126.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-05 11:15:00 | 138.88 | 134.65 | 133.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 11:15:00 | 133.32 | 134.37 | 134.48 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 14:15:00 | 136.94 | 134.64 | 134.55 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 133.43 | 134.54 | 134.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 132.08 | 133.50 | 134.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 135.34 | 133.86 | 134.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 135.34 | 133.86 | 134.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 135.34 | 133.86 | 134.15 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 11:15:00 | 135.32 | 134.40 | 134.36 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2026-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 11:15:00 | 133.80 | 134.39 | 134.43 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 136.50 | 134.83 | 134.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 137.21 | 135.44 | 134.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 12:15:00 | 135.65 | 135.72 | 135.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 13:00:00 | 135.65 | 135.72 | 135.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 135.18 | 135.61 | 135.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:00:00 | 135.18 | 135.61 | 135.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 135.51 | 135.59 | 135.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 135.95 | 135.59 | 135.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 09:15:00 | 134.30 | 135.39 | 135.23 | SL hit (close<static) qty=1.00 sl=135.00 alert=retest2 |

### Cycle 27 — SELL (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 11:15:00 | 132.53 | 135.09 | 135.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 12:15:00 | 131.91 | 134.46 | 135.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 127.37 | 127.11 | 129.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 10:00:00 | 127.37 | 127.11 | 129.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 128.56 | 127.50 | 128.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:45:00 | 128.69 | 127.50 | 128.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 127.30 | 127.46 | 128.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:30:00 | 130.30 | 127.46 | 128.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 128.40 | 127.77 | 128.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 128.40 | 127.77 | 128.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 128.05 | 127.82 | 128.47 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 129.67 | 128.93 | 128.84 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 128.01 | 128.75 | 128.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 10:15:00 | 125.86 | 128.17 | 128.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 123.00 | 122.41 | 124.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 15:00:00 | 123.00 | 122.41 | 124.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 123.60 | 122.23 | 123.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 13:00:00 | 123.60 | 122.23 | 123.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 13:15:00 | 124.89 | 122.77 | 123.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 13:45:00 | 125.22 | 122.77 | 123.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 122.89 | 122.79 | 123.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 15:15:00 | 124.52 | 122.79 | 123.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 124.52 | 123.14 | 123.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:30:00 | 122.27 | 123.15 | 123.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 11:45:00 | 122.53 | 123.10 | 123.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 13:15:00 | 125.60 | 123.78 | 123.80 | SL hit (close>static) qty=1.00 sl=125.28 alert=retest2 |

### Cycle 30 — BUY (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 14:15:00 | 124.68 | 123.96 | 123.88 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 122.59 | 123.98 | 124.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 122.01 | 123.35 | 123.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 11:15:00 | 122.89 | 122.06 | 122.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 11:15:00 | 122.89 | 122.06 | 122.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 122.89 | 122.06 | 122.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:00:00 | 122.89 | 122.06 | 122.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 121.15 | 121.88 | 122.54 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 124.39 | 122.85 | 122.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 125.47 | 123.37 | 123.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 13:15:00 | 123.36 | 123.60 | 123.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-03 14:00:00 | 123.36 | 123.60 | 123.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 124.14 | 123.71 | 123.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-03 14:30:00 | 123.20 | 123.71 | 123.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 122.40 | 123.49 | 123.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:30:00 | 121.61 | 123.49 | 123.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 122.65 | 123.32 | 123.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:30:00 | 122.41 | 123.32 | 123.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 123.08 | 123.30 | 123.21 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2026-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 15:15:00 | 122.70 | 123.09 | 123.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 122.00 | 122.75 | 122.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 122.83 | 122.26 | 122.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 14:15:00 | 122.83 | 122.26 | 122.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 122.83 | 122.26 | 122.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 122.83 | 122.26 | 122.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 121.60 | 122.13 | 122.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 123.74 | 122.13 | 122.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 125.99 | 122.90 | 122.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 126.06 | 124.07 | 123.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 125.30 | 126.02 | 125.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 10:00:00 | 125.30 | 126.02 | 125.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 124.35 | 125.69 | 124.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 124.35 | 125.69 | 124.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 125.08 | 125.57 | 125.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:30:00 | 125.00 | 125.57 | 125.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 123.71 | 125.19 | 124.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:30:00 | 123.40 | 125.19 | 124.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 123.55 | 124.87 | 124.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:00:00 | 123.55 | 124.87 | 124.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 14:15:00 | 123.77 | 124.65 | 124.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 122.67 | 124.10 | 124.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 118.64 | 118.58 | 119.99 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 11:15:00 | 117.30 | 118.37 | 119.76 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 12:00:00 | 117.25 | 118.14 | 119.53 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 120.40 | 118.75 | 119.38 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-17 15:15:00 | 120.40 | 118.75 | 119.38 | SL hit (close>ema400) qty=1.00 sl=119.38 alert=retest1 |

### Cycle 36 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 120.45 | 119.65 | 119.61 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 118.66 | 119.45 | 119.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 10:15:00 | 117.86 | 119.13 | 119.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 12:15:00 | 119.00 | 118.86 | 119.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 12:15:00 | 119.00 | 118.86 | 119.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 119.00 | 118.86 | 119.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:45:00 | 118.97 | 118.86 | 119.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 118.15 | 118.72 | 119.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:30:00 | 118.85 | 118.72 | 119.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 118.90 | 118.64 | 118.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 117.24 | 118.64 | 118.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 11:00:00 | 117.75 | 118.29 | 118.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 11:45:00 | 117.81 | 118.20 | 118.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 15:00:00 | 117.84 | 118.23 | 118.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 116.10 | 117.70 | 118.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:30:00 | 115.51 | 117.16 | 117.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 15:15:00 | 111.86 | 114.44 | 116.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 15:15:00 | 111.92 | 114.44 | 116.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 15:15:00 | 111.95 | 114.44 | 116.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 111.38 | 113.33 | 115.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 109.73 | 113.33 | 115.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 108.10 | 108.02 | 109.26 | SL hit (close>ema200) qty=0.50 sl=108.02 alert=retest2 |

### Cycle 38 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 108.32 | 105.87 | 105.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 15:15:00 | 108.90 | 106.47 | 106.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 10:15:00 | 106.44 | 106.60 | 106.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 10:15:00 | 106.44 | 106.60 | 106.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 106.44 | 106.60 | 106.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 106.47 | 106.60 | 106.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 106.13 | 106.42 | 106.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:45:00 | 105.59 | 106.42 | 106.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 106.10 | 106.36 | 106.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 104.39 | 106.36 | 106.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 104.71 | 106.03 | 106.09 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 14:15:00 | 107.08 | 106.21 | 106.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 09:15:00 | 110.24 | 107.27 | 106.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 110.29 | 110.39 | 109.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 110.29 | 110.39 | 109.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 110.29 | 110.39 | 109.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 108.55 | 110.39 | 109.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 110.82 | 112.54 | 111.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:00:00 | 110.82 | 112.54 | 111.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 111.75 | 112.38 | 111.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 09:15:00 | 112.50 | 111.10 | 111.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 109.04 | 110.63 | 110.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2026-03-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 10:15:00 | 109.04 | 110.63 | 110.82 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2026-03-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 14:15:00 | 113.60 | 111.33 | 111.10 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 12:15:00 | 109.92 | 111.10 | 111.15 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 124.67 | 113.56 | 112.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 127.98 | 119.87 | 115.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 14:15:00 | 121.12 | 121.37 | 117.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 15:00:00 | 121.12 | 121.37 | 117.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 116.25 | 120.39 | 117.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 116.25 | 120.39 | 117.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 116.75 | 119.67 | 117.42 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 113.32 | 116.16 | 116.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 110.58 | 114.33 | 115.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 111.82 | 111.40 | 113.00 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:15:00 | 111.33 | 111.40 | 113.00 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 113.57 | 111.73 | 112.87 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 113.57 | 111.73 | 112.87 | SL hit (close>ema400) qty=1.00 sl=112.87 alert=retest1 |

### Cycle 46 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 114.35 | 113.54 | 113.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 09:15:00 | 119.44 | 115.37 | 114.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 120.32 | 122.47 | 120.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 120.32 | 122.47 | 120.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 120.32 | 122.47 | 120.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 14:00:00 | 121.60 | 121.56 | 120.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 14:45:00 | 121.60 | 121.70 | 120.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 123.79 | 121.60 | 120.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-10 10:15:00 | 133.76 | 128.06 | 126.12 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 13:15:00 | 133.32 | 133.97 | 133.97 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 09:15:00 | 134.33 | 134.00 | 133.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 11:15:00 | 137.20 | 134.69 | 134.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 14:15:00 | 134.67 | 135.07 | 134.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 14:15:00 | 134.67 | 135.07 | 134.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 134.67 | 135.07 | 134.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 14:45:00 | 134.65 | 135.07 | 134.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 134.25 | 134.91 | 134.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:30:00 | 134.92 | 134.88 | 134.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 11:00:00 | 137.22 | 135.34 | 134.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 14:15:00 | 148.41 | 143.98 | 140.40 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 12:15:00 | 142.33 | 143.26 | 143.29 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 14:15:00 | 147.49 | 143.94 | 143.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 15:15:00 | 148.30 | 144.81 | 143.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 144.78 | 144.81 | 144.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:00:00 | 144.78 | 144.81 | 144.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 144.75 | 144.79 | 144.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:45:00 | 143.71 | 144.79 | 144.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 150.75 | 151.66 | 149.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:30:00 | 150.20 | 151.66 | 149.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 149.79 | 151.28 | 149.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 149.21 | 151.28 | 149.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 150.49 | 151.13 | 149.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:15:00 | 149.22 | 151.13 | 149.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 148.13 | 150.53 | 149.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:00:00 | 148.13 | 150.53 | 149.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 147.26 | 149.87 | 149.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:45:00 | 146.62 | 149.87 | 149.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 09:15:00 | 145.60 | 148.56 | 148.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 10:15:00 | 143.63 | 147.57 | 148.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 15:15:00 | 146.50 | 146.16 | 147.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-07 09:15:00 | 147.76 | 146.16 | 147.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 144.85 | 145.90 | 147.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 10:45:00 | 143.85 | 145.77 | 146.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-10-01 11:45:00 | 166.46 | 2025-10-01 15:15:00 | 171.55 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2025-10-06 13:15:00 | 170.89 | 2025-10-06 13:15:00 | 169.36 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-10-27 13:30:00 | 150.65 | 2025-10-28 14:15:00 | 151.80 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-10-28 11:30:00 | 150.71 | 2025-10-28 14:15:00 | 151.80 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-10-28 12:15:00 | 149.49 | 2025-10-28 14:15:00 | 151.80 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-10-31 12:45:00 | 154.03 | 2025-11-03 09:15:00 | 152.42 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-11-20 11:45:00 | 145.81 | 2025-11-20 14:15:00 | 143.90 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-11-27 12:30:00 | 136.14 | 2025-12-05 09:15:00 | 129.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 14:30:00 | 136.04 | 2025-12-05 09:15:00 | 129.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 10:45:00 | 135.75 | 2025-12-05 09:15:00 | 128.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 10:30:00 | 135.90 | 2025-12-05 09:15:00 | 129.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 09:15:00 | 134.94 | 2025-12-05 09:15:00 | 128.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 12:30:00 | 136.14 | 2025-12-05 11:15:00 | 130.87 | STOP_HIT | 0.50 | 3.87% |
| SELL | retest2 | 2025-11-27 14:30:00 | 136.04 | 2025-12-05 11:15:00 | 130.87 | STOP_HIT | 0.50 | 3.80% |
| SELL | retest2 | 2025-11-28 10:45:00 | 135.75 | 2025-12-05 11:15:00 | 130.87 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2025-12-01 10:30:00 | 135.90 | 2025-12-05 11:15:00 | 130.87 | STOP_HIT | 0.50 | 3.70% |
| SELL | retest2 | 2025-12-03 09:15:00 | 134.94 | 2025-12-05 11:15:00 | 130.87 | STOP_HIT | 0.50 | 3.02% |
| BUY | retest2 | 2025-12-24 09:45:00 | 126.25 | 2026-01-05 11:15:00 | 138.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-24 11:30:00 | 126.00 | 2026-01-05 11:15:00 | 138.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-24 13:30:00 | 126.11 | 2026-01-05 11:15:00 | 138.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-24 14:15:00 | 126.31 | 2026-01-05 11:15:00 | 138.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-29 11:45:00 | 126.55 | 2026-01-05 11:15:00 | 139.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-13 15:15:00 | 135.95 | 2026-01-14 09:15:00 | 134.30 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2026-01-14 10:30:00 | 136.67 | 2026-01-16 11:15:00 | 132.53 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2026-01-14 14:15:00 | 136.13 | 2026-01-16 11:15:00 | 132.53 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2026-01-29 09:30:00 | 122.27 | 2026-01-29 13:15:00 | 125.60 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2026-01-29 11:45:00 | 122.53 | 2026-01-29 13:15:00 | 125.60 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest1 | 2026-02-17 11:15:00 | 117.30 | 2026-02-17 15:15:00 | 120.40 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest1 | 2026-02-17 12:00:00 | 117.25 | 2026-02-17 15:15:00 | 120.40 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2026-02-20 09:15:00 | 117.24 | 2026-02-23 15:15:00 | 111.86 | PARTIAL | 0.50 | 4.59% |
| SELL | retest2 | 2026-02-20 11:00:00 | 117.75 | 2026-02-23 15:15:00 | 111.92 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2026-02-20 11:45:00 | 117.81 | 2026-02-23 15:15:00 | 111.95 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2026-02-20 15:00:00 | 117.84 | 2026-02-24 09:15:00 | 111.38 | PARTIAL | 0.50 | 5.48% |
| SELL | retest2 | 2026-02-23 10:30:00 | 115.51 | 2026-02-24 09:15:00 | 109.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 09:15:00 | 117.24 | 2026-02-27 09:15:00 | 108.10 | STOP_HIT | 0.50 | 7.80% |
| SELL | retest2 | 2026-02-20 11:00:00 | 117.75 | 2026-02-27 09:15:00 | 108.10 | STOP_HIT | 0.50 | 8.20% |
| SELL | retest2 | 2026-02-20 11:45:00 | 117.81 | 2026-02-27 09:15:00 | 108.10 | STOP_HIT | 0.50 | 8.24% |
| SELL | retest2 | 2026-02-20 15:00:00 | 117.84 | 2026-02-27 09:15:00 | 108.10 | STOP_HIT | 0.50 | 8.27% |
| SELL | retest2 | 2026-02-23 10:30:00 | 115.51 | 2026-02-27 09:15:00 | 108.10 | STOP_HIT | 0.50 | 6.42% |
| BUY | retest2 | 2026-03-16 09:15:00 | 112.50 | 2026-03-16 10:15:00 | 109.04 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest1 | 2026-03-24 10:15:00 | 111.33 | 2026-03-24 11:15:00 | 113.57 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-04-02 14:00:00 | 121.60 | 2026-04-10 10:15:00 | 133.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 14:45:00 | 121.60 | 2026-04-10 10:15:00 | 133.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 09:15:00 | 123.79 | 2026-04-10 12:15:00 | 136.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-21 09:30:00 | 134.92 | 2026-04-22 14:15:00 | 148.41 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-21 11:00:00 | 137.22 | 2026-04-28 12:15:00 | 142.33 | STOP_HIT | 1.00 | 3.72% |
