# Angel One Ltd. (ANGELONE)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 326.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 190 |
| ALERT1 | 144 |
| ALERT2 | 142 |
| ALERT2_SKIP | 76 |
| ALERT3 | 356 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 161 |
| PARTIAL | 27 |
| TARGET_HIT | 26 |
| STOP_HIT | 145 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 197 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 85 / 112
- **Target hits / Stop hits / Partials:** 26 / 144 / 27
- **Avg / median % per leg:** 1.03% / -0.98%
- **Sum % (uncompounded):** 203.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 87 | 26 | 29.9% | 18 | 68 | 1 | 0.38% | 33.1% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 5 | 1 | 1.05% | 7.3% |
| BUY @ 3rd Alert (retest2) | 80 | 23 | 28.7% | 17 | 63 | 0 | 0.32% | 25.7% |
| SELL (all) | 110 | 59 | 53.6% | 8 | 76 | 26 | 1.55% | 170.5% |
| SELL @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 3 | 0 | 3 | 7.50% | 45.0% |
| SELL @ 3rd Alert (retest2) | 104 | 53 | 51.0% | 5 | 76 | 23 | 1.21% | 125.5% |
| retest1 (combined) | 13 | 9 | 69.2% | 4 | 5 | 4 | 4.03% | 52.3% |
| retest2 (combined) | 184 | 76 | 41.3% | 22 | 139 | 23 | 0.82% | 151.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 12:15:00 | 130.54 | 130.77 | 130.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 14:15:00 | 129.36 | 130.44 | 130.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 09:15:00 | 123.18 | 120.95 | 122.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 09:15:00 | 123.18 | 120.95 | 122.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 123.18 | 120.95 | 122.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:30:00 | 123.21 | 120.95 | 122.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 10:15:00 | 125.60 | 121.88 | 123.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 11:00:00 | 125.60 | 121.88 | 123.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 11:15:00 | 125.33 | 122.57 | 123.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 13:00:00 | 124.98 | 123.05 | 123.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-22 15:15:00 | 125.20 | 123.95 | 123.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 15:15:00 | 125.20 | 123.95 | 123.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 09:15:00 | 125.28 | 124.22 | 123.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 10:15:00 | 123.70 | 124.11 | 123.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 10:15:00 | 123.70 | 124.11 | 123.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 10:15:00 | 123.70 | 124.11 | 123.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 11:00:00 | 123.70 | 124.11 | 123.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2023-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 11:15:00 | 122.46 | 123.78 | 123.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-23 13:15:00 | 121.26 | 122.99 | 123.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-24 14:15:00 | 121.61 | 121.59 | 122.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-24 15:15:00 | 121.72 | 121.59 | 122.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 122.60 | 121.82 | 122.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 09:45:00 | 122.61 | 121.82 | 122.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 10:15:00 | 123.22 | 122.10 | 122.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 10:30:00 | 123.34 | 122.10 | 122.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2023-05-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 12:15:00 | 123.19 | 122.49 | 122.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 15:15:00 | 123.50 | 122.89 | 122.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 13:15:00 | 129.35 | 130.37 | 128.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-30 14:00:00 | 129.35 | 130.37 | 128.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 131.43 | 130.56 | 129.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 09:30:00 | 128.60 | 130.56 | 129.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 130.83 | 133.12 | 132.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 10:00:00 | 130.83 | 133.12 | 132.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 10:15:00 | 130.89 | 132.67 | 132.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 12:30:00 | 131.21 | 132.07 | 131.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-02 14:15:00 | 131.39 | 131.80 | 131.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 14:15:00 | 131.39 | 131.80 | 131.85 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 09:15:00 | 133.10 | 132.00 | 131.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 13:15:00 | 135.09 | 133.09 | 132.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 11:15:00 | 144.20 | 144.73 | 142.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 11:15:00 | 144.20 | 144.73 | 142.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 144.20 | 144.73 | 142.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 11:45:00 | 143.29 | 144.73 | 142.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 09:15:00 | 144.85 | 145.46 | 143.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 09:30:00 | 144.64 | 145.46 | 143.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 09:15:00 | 154.44 | 154.99 | 154.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 10:00:00 | 154.44 | 154.99 | 154.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 10:15:00 | 154.62 | 154.91 | 154.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 11:00:00 | 154.62 | 154.91 | 154.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 11:15:00 | 154.73 | 154.88 | 154.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 11:30:00 | 154.53 | 154.88 | 154.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 12:15:00 | 153.72 | 154.65 | 154.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 13:00:00 | 153.72 | 154.65 | 154.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 13:15:00 | 152.19 | 154.15 | 154.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 13:30:00 | 151.56 | 154.15 | 154.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 157.66 | 160.32 | 159.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 12:00:00 | 157.66 | 160.32 | 159.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 12:15:00 | 158.20 | 159.90 | 159.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-22 13:30:00 | 158.99 | 159.59 | 159.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-22 15:15:00 | 156.80 | 158.71 | 158.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2023-06-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 15:15:00 | 156.80 | 158.71 | 158.83 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-23 09:15:00 | 160.21 | 159.01 | 158.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-23 11:15:00 | 160.79 | 159.47 | 159.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 15:15:00 | 169.71 | 171.16 | 169.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 14:15:00 | 172.21 | 171.29 | 170.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 14:15:00 | 172.21 | 171.29 | 170.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-30 14:30:00 | 169.98 | 171.29 | 170.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 09:15:00 | 175.71 | 178.28 | 176.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-06 09:45:00 | 175.69 | 178.28 | 176.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 10:15:00 | 176.50 | 177.93 | 176.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-07 09:15:00 | 176.88 | 176.92 | 176.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-07 11:15:00 | 172.92 | 175.98 | 176.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2023-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 11:15:00 | 172.92 | 175.98 | 176.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 12:15:00 | 170.94 | 174.97 | 175.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 13:15:00 | 170.90 | 170.53 | 172.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-10 14:00:00 | 170.90 | 170.53 | 172.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 172.32 | 170.94 | 172.21 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 15:15:00 | 173.83 | 172.67 | 172.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 09:15:00 | 176.60 | 173.46 | 172.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 14:15:00 | 174.95 | 175.07 | 174.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-12 14:30:00 | 174.69 | 175.07 | 174.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 15:15:00 | 174.70 | 175.00 | 174.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 09:15:00 | 178.00 | 175.00 | 174.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 15:15:00 | 175.73 | 177.05 | 175.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 10:00:00 | 175.36 | 176.50 | 175.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-14 10:15:00 | 173.55 | 175.91 | 175.59 | SL hit (close<static) qty=1.00 sl=174.01 alert=retest2 |

### Cycle 11 — SELL (started 2023-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 11:15:00 | 172.03 | 175.13 | 175.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 12:15:00 | 171.20 | 174.35 | 174.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-19 09:15:00 | 150.71 | 150.70 | 157.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-19 09:45:00 | 152.07 | 150.70 | 157.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 10:15:00 | 154.67 | 151.68 | 154.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 11:00:00 | 154.67 | 151.68 | 154.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 11:15:00 | 156.53 | 152.65 | 154.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 12:00:00 | 156.53 | 152.65 | 154.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 12:15:00 | 156.18 | 153.36 | 154.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 12:45:00 | 156.60 | 153.36 | 154.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2023-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 09:15:00 | 159.00 | 155.85 | 155.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 10:15:00 | 161.90 | 157.06 | 156.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 11:15:00 | 160.35 | 161.34 | 159.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 11:15:00 | 160.35 | 161.34 | 159.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 11:15:00 | 160.35 | 161.34 | 159.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 11:30:00 | 159.33 | 161.34 | 159.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 15:15:00 | 158.70 | 160.74 | 159.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 09:45:00 | 158.16 | 160.36 | 159.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 157.90 | 159.87 | 159.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 10:45:00 | 157.54 | 159.87 | 159.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 13:15:00 | 160.07 | 159.66 | 159.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 09:15:00 | 160.72 | 159.75 | 159.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 09:45:00 | 160.60 | 159.94 | 159.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-27 09:15:00 | 160.78 | 159.60 | 159.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-27 13:15:00 | 158.60 | 159.51 | 159.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-07-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 13:15:00 | 158.60 | 159.51 | 159.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-28 09:15:00 | 157.90 | 159.19 | 159.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 14:15:00 | 149.10 | 148.71 | 150.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-03 15:00:00 | 149.10 | 148.71 | 150.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 151.34 | 149.24 | 150.15 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 10:15:00 | 150.88 | 150.48 | 150.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 14:15:00 | 151.49 | 150.82 | 150.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 09:15:00 | 172.20 | 173.09 | 169.63 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-14 11:15:00 | 174.85 | 173.24 | 171.53 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 15:15:00 | 177.06 | 178.00 | 176.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 09:15:00 | 178.18 | 178.00 | 176.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 11:00:00 | 178.27 | 178.10 | 176.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-18 09:15:00 | 174.90 | 177.30 | 176.82 | SL hit (close<ema400) qty=1.00 sl=176.82 alert=retest1 |

### Cycle 15 — SELL (started 2023-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 14:15:00 | 176.00 | 176.62 | 176.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-21 09:15:00 | 174.78 | 176.17 | 176.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 10:15:00 | 176.53 | 176.24 | 176.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 10:15:00 | 176.53 | 176.24 | 176.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 176.53 | 176.24 | 176.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 11:00:00 | 176.53 | 176.24 | 176.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 11:15:00 | 176.59 | 176.31 | 176.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 12:45:00 | 176.22 | 176.34 | 176.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 13:15:00 | 176.13 | 176.34 | 176.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 14:30:00 | 176.20 | 176.30 | 176.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 15:00:00 | 176.07 | 176.30 | 176.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 15:15:00 | 176.00 | 176.24 | 176.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 09:15:00 | 178.38 | 176.24 | 176.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-08-22 09:15:00 | 178.90 | 176.77 | 176.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2023-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 09:15:00 | 178.90 | 176.77 | 176.60 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 09:15:00 | 173.70 | 176.21 | 176.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 09:15:00 | 171.64 | 174.67 | 175.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-25 11:15:00 | 171.69 | 171.48 | 172.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-25 12:00:00 | 171.69 | 171.48 | 172.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 174.81 | 172.15 | 172.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 10:00:00 | 174.81 | 172.15 | 172.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 10:15:00 | 175.09 | 172.73 | 172.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 10:30:00 | 175.20 | 172.73 | 172.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2023-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 11:15:00 | 174.45 | 173.08 | 173.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-28 12:15:00 | 175.77 | 173.62 | 173.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-29 13:15:00 | 176.12 | 177.01 | 175.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 13:15:00 | 176.12 | 177.01 | 175.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 13:15:00 | 176.12 | 177.01 | 175.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-29 13:30:00 | 175.81 | 177.01 | 175.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 09:15:00 | 187.06 | 190.70 | 189.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 09:30:00 | 185.95 | 190.70 | 189.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 10:15:00 | 186.30 | 189.82 | 188.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-05 11:45:00 | 187.33 | 189.26 | 188.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-05 13:15:00 | 185.50 | 188.04 | 188.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2023-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-05 13:15:00 | 185.50 | 188.04 | 188.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-06 12:15:00 | 185.10 | 186.61 | 187.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-07 10:15:00 | 184.98 | 184.77 | 185.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-07 10:30:00 | 184.70 | 184.77 | 185.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 11:15:00 | 183.59 | 184.54 | 185.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 11:30:00 | 184.56 | 184.54 | 185.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 184.73 | 183.99 | 184.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 10:15:00 | 186.40 | 183.99 | 184.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 10:15:00 | 185.95 | 184.38 | 185.03 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 13:15:00 | 186.60 | 185.40 | 185.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 14:15:00 | 187.69 | 185.86 | 185.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-11 14:15:00 | 188.18 | 188.85 | 187.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-11 15:00:00 | 188.18 | 188.85 | 187.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 15:15:00 | 187.03 | 188.48 | 187.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:30:00 | 184.30 | 187.49 | 187.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2023-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 10:15:00 | 183.88 | 186.77 | 186.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 11:15:00 | 181.24 | 185.66 | 186.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 10:15:00 | 176.10 | 176.08 | 178.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-14 11:00:00 | 176.10 | 176.08 | 178.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 12:15:00 | 178.90 | 176.82 | 178.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 13:00:00 | 178.90 | 176.82 | 178.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 13:15:00 | 178.38 | 177.13 | 178.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 13:30:00 | 178.59 | 177.13 | 178.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 14:15:00 | 178.80 | 177.46 | 178.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 15:00:00 | 178.80 | 177.46 | 178.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 15:15:00 | 179.10 | 177.79 | 178.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-15 09:15:00 | 183.29 | 177.79 | 178.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2023-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 09:15:00 | 185.40 | 179.31 | 179.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 14:15:00 | 189.83 | 183.99 | 181.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-20 09:15:00 | 187.08 | 188.14 | 185.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 09:15:00 | 187.08 | 188.14 | 185.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 187.08 | 188.14 | 185.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 09:30:00 | 186.94 | 188.14 | 185.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 10:15:00 | 185.46 | 187.61 | 185.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 11:00:00 | 185.46 | 187.61 | 185.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 11:15:00 | 185.85 | 187.26 | 185.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 11:45:00 | 185.43 | 187.26 | 185.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 12:15:00 | 185.34 | 186.87 | 185.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 13:00:00 | 185.34 | 186.87 | 185.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 13:15:00 | 183.98 | 186.29 | 185.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 13:45:00 | 183.56 | 186.29 | 185.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 14:15:00 | 183.70 | 185.77 | 185.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 15:00:00 | 183.70 | 185.77 | 185.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2023-09-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 15:15:00 | 183.00 | 185.22 | 185.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 09:15:00 | 181.76 | 184.53 | 184.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 13:15:00 | 183.70 | 183.21 | 184.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 13:15:00 | 183.70 | 183.21 | 184.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 13:15:00 | 183.70 | 183.21 | 184.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 14:00:00 | 183.70 | 183.21 | 184.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 14:15:00 | 186.79 | 183.93 | 184.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 14:30:00 | 186.76 | 183.93 | 184.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 15:15:00 | 186.66 | 184.47 | 184.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 09:15:00 | 182.93 | 184.47 | 184.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-22 12:15:00 | 187.90 | 184.94 | 184.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2023-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 12:15:00 | 187.90 | 184.94 | 184.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 14:15:00 | 193.33 | 187.86 | 186.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 12:15:00 | 187.92 | 189.09 | 187.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 12:15:00 | 187.92 | 189.09 | 187.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 12:15:00 | 187.92 | 189.09 | 187.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 13:00:00 | 187.92 | 189.09 | 187.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 188.00 | 188.87 | 187.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 13:30:00 | 187.90 | 188.87 | 187.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 187.75 | 188.65 | 187.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 15:00:00 | 187.75 | 188.65 | 187.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 187.40 | 188.40 | 187.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 09:15:00 | 187.81 | 188.40 | 187.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-29 11:15:00 | 184.59 | 187.26 | 187.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2023-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 11:15:00 | 184.59 | 187.26 | 187.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-03 09:15:00 | 183.49 | 185.50 | 186.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-03 13:15:00 | 185.50 | 184.91 | 185.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 13:15:00 | 185.50 | 184.91 | 185.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 13:15:00 | 185.50 | 184.91 | 185.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-03 14:00:00 | 185.50 | 184.91 | 185.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 14:15:00 | 183.72 | 184.67 | 185.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-04 12:30:00 | 183.20 | 184.52 | 185.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-04 13:15:00 | 183.48 | 184.52 | 185.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-04 14:15:00 | 186.05 | 184.71 | 185.15 | SL hit (close>static) qty=1.00 sl=185.60 alert=retest2 |

### Cycle 26 — BUY (started 2023-10-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 10:15:00 | 186.37 | 185.45 | 185.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 13:15:00 | 192.60 | 187.09 | 186.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 14:15:00 | 210.03 | 210.70 | 208.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 14:15:00 | 210.03 | 210.70 | 208.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 14:15:00 | 210.03 | 210.70 | 208.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 15:00:00 | 210.03 | 210.70 | 208.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 13:15:00 | 210.48 | 214.59 | 211.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 14:00:00 | 210.48 | 214.59 | 211.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 14:15:00 | 207.25 | 213.12 | 211.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 15:00:00 | 207.25 | 213.12 | 211.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2023-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 10:15:00 | 207.54 | 210.51 | 210.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-16 11:15:00 | 206.91 | 209.79 | 210.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 209.80 | 208.01 | 208.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 09:15:00 | 209.80 | 208.01 | 208.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 209.80 | 208.01 | 208.99 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 14:15:00 | 210.70 | 209.57 | 209.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-18 09:15:00 | 215.66 | 210.94 | 210.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-19 09:15:00 | 214.78 | 214.84 | 212.96 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 11:45:00 | 217.72 | 215.67 | 213.68 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-20 10:15:00 | 228.61 | 221.50 | 217.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2023-10-23 09:15:00 | 239.49 | 229.92 | 224.03 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 29 — SELL (started 2023-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 09:15:00 | 253.00 | 265.93 | 267.26 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 14:15:00 | 267.25 | 264.13 | 263.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 15:15:00 | 267.90 | 264.88 | 264.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 15:15:00 | 269.49 | 269.94 | 267.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-10 09:15:00 | 269.19 | 269.94 | 267.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 272.24 | 270.40 | 268.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-13 14:15:00 | 275.99 | 272.70 | 271.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-11-15 14:15:00 | 303.59 | 291.02 | 282.80 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2023-11-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 15:15:00 | 294.40 | 296.19 | 296.36 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 09:15:00 | 304.10 | 297.77 | 297.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 09:15:00 | 314.90 | 304.78 | 301.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 09:15:00 | 304.23 | 306.23 | 304.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 09:15:00 | 304.23 | 306.23 | 304.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 304.23 | 306.23 | 304.07 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-04 09:15:00 | 297.83 | 304.78 | 305.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-04 13:15:00 | 297.42 | 300.94 | 303.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-06 09:15:00 | 315.35 | 299.56 | 300.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-06 09:15:00 | 315.35 | 299.56 | 300.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 09:15:00 | 315.35 | 299.56 | 300.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-06 10:00:00 | 315.35 | 299.56 | 300.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2023-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 10:15:00 | 315.67 | 302.78 | 301.45 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 10:15:00 | 301.29 | 306.78 | 306.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 13:15:00 | 297.28 | 303.36 | 305.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 09:15:00 | 301.26 | 300.85 | 303.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 09:15:00 | 301.26 | 300.85 | 303.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 301.26 | 300.85 | 303.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 09:30:00 | 300.58 | 300.85 | 303.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 303.88 | 300.10 | 301.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 10:00:00 | 303.88 | 300.10 | 301.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 10:15:00 | 307.10 | 301.50 | 302.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 11:00:00 | 307.10 | 301.50 | 302.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2023-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 12:15:00 | 303.50 | 302.59 | 302.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 13:15:00 | 305.96 | 303.26 | 302.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-13 09:15:00 | 303.67 | 304.15 | 303.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 09:15:00 | 303.67 | 304.15 | 303.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 303.67 | 304.15 | 303.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 09:45:00 | 304.05 | 304.15 | 303.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 10:15:00 | 302.15 | 303.75 | 303.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 10:45:00 | 302.45 | 303.75 | 303.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2023-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 11:15:00 | 298.49 | 302.70 | 302.86 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-12-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 09:15:00 | 313.82 | 304.98 | 303.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 12:15:00 | 317.40 | 309.96 | 306.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 09:15:00 | 328.77 | 329.55 | 324.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-19 10:00:00 | 328.77 | 329.55 | 324.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 09:15:00 | 323.90 | 328.05 | 326.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 09:30:00 | 325.30 | 328.05 | 326.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 328.78 | 328.20 | 326.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 11:45:00 | 330.74 | 329.24 | 327.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 13:15:00 | 317.53 | 327.51 | 326.65 | SL hit (close<static) qty=1.00 sl=322.86 alert=retest2 |

### Cycle 39 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 314.38 | 324.88 | 325.53 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 09:15:00 | 331.43 | 326.25 | 325.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 13:15:00 | 337.20 | 332.99 | 331.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 13:15:00 | 339.52 | 340.59 | 336.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-29 14:00:00 | 339.52 | 340.59 | 336.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 14:15:00 | 349.16 | 342.30 | 337.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 14:00:00 | 353.00 | 347.64 | 342.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 10:00:00 | 351.40 | 353.33 | 350.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 11:00:00 | 350.73 | 352.81 | 350.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 09:15:00 | 372.50 | 350.35 | 349.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-08 09:15:00 | 388.30 | 371.53 | 364.16 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 09:15:00 | 342.42 | 374.46 | 376.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 12:15:00 | 338.60 | 359.12 | 368.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 334.50 | 332.45 | 338.81 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-19 10:30:00 | 331.90 | 332.36 | 338.19 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-19 11:00:00 | 332.00 | 332.36 | 338.19 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 09:15:00 | 331.44 | 332.92 | 336.19 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 09:15:00 | 315.30 | 325.24 | 330.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 09:15:00 | 315.40 | 325.24 | 330.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 09:15:00 | 314.87 | 325.24 | 330.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-01-23 14:15:00 | 298.71 | 311.49 | 320.92 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 42 — BUY (started 2024-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 10:15:00 | 307.60 | 296.78 | 296.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-30 14:15:00 | 309.85 | 303.06 | 299.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-02 12:15:00 | 334.77 | 335.75 | 328.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-02 13:00:00 | 334.77 | 335.75 | 328.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 15:15:00 | 333.40 | 335.43 | 329.82 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 14:15:00 | 321.50 | 328.24 | 328.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-06 09:15:00 | 316.50 | 324.98 | 326.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 09:15:00 | 335.06 | 319.15 | 321.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 09:15:00 | 335.06 | 319.15 | 321.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 335.06 | 319.15 | 321.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 10:00:00 | 335.06 | 319.15 | 321.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 10:15:00 | 331.80 | 321.68 | 322.40 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 11:15:00 | 331.40 | 323.62 | 323.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 14:15:00 | 333.27 | 327.99 | 325.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 09:15:00 | 338.40 | 338.62 | 334.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 338.40 | 338.62 | 334.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 338.40 | 338.62 | 334.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 09:45:00 | 334.40 | 338.62 | 334.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 335.13 | 337.92 | 334.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 11:00:00 | 335.13 | 337.92 | 334.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 11:15:00 | 338.77 | 338.09 | 334.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 12:15:00 | 339.45 | 338.09 | 334.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 13:15:00 | 339.00 | 338.19 | 334.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 13:45:00 | 339.77 | 338.51 | 335.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 15:15:00 | 339.30 | 338.55 | 335.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 323.73 | 335.71 | 334.89 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-12 09:15:00 | 323.73 | 335.71 | 334.89 | SL hit (close<static) qty=1.00 sl=333.00 alert=retest2 |

### Cycle 45 — SELL (started 2024-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 10:15:00 | 327.04 | 333.97 | 334.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 12:15:00 | 321.96 | 329.94 | 332.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 14:15:00 | 326.72 | 323.63 | 326.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 14:15:00 | 326.72 | 323.63 | 326.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 326.72 | 323.63 | 326.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 15:00:00 | 326.72 | 323.63 | 326.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 15:15:00 | 327.30 | 324.36 | 326.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 09:30:00 | 325.52 | 324.29 | 326.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 12:15:00 | 325.78 | 324.62 | 325.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 13:00:00 | 325.78 | 324.62 | 325.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 13:15:00 | 324.78 | 324.65 | 325.86 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2024-02-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 10:15:00 | 328.56 | 326.56 | 326.47 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 10:15:00 | 325.90 | 326.69 | 326.79 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 11:15:00 | 327.84 | 326.92 | 326.89 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 14:15:00 | 323.00 | 326.16 | 326.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 15:15:00 | 321.50 | 325.23 | 326.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 09:15:00 | 308.90 | 308.34 | 312.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 09:15:00 | 308.90 | 308.34 | 312.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 308.90 | 308.34 | 312.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 10:15:00 | 305.33 | 308.34 | 312.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 12:15:00 | 305.50 | 307.53 | 311.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-26 09:45:00 | 304.56 | 305.06 | 308.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 10:15:00 | 290.06 | 293.14 | 297.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 10:15:00 | 290.22 | 293.14 | 297.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 10:15:00 | 289.33 | 293.14 | 297.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-01 09:15:00 | 284.95 | 281.46 | 285.99 | SL hit (close>ema200) qty=0.50 sl=281.46 alert=retest2 |

### Cycle 50 — BUY (started 2024-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-04 12:15:00 | 291.14 | 285.15 | 285.00 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 281.00 | 286.43 | 286.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 12:15:00 | 277.99 | 282.59 | 284.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 264.12 | 248.26 | 253.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 09:15:00 | 264.12 | 248.26 | 253.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 264.12 | 248.26 | 253.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 10:00:00 | 264.12 | 248.26 | 253.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 264.96 | 251.60 | 254.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-14 13:00:00 | 262.71 | 255.86 | 256.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-14 13:30:00 | 259.55 | 256.16 | 256.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-14 14:15:00 | 264.52 | 257.83 | 257.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2024-03-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 14:15:00 | 264.52 | 257.83 | 257.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-14 15:15:00 | 266.30 | 259.53 | 258.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-15 10:15:00 | 254.00 | 258.90 | 258.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 10:15:00 | 254.00 | 258.90 | 258.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 10:15:00 | 254.00 | 258.90 | 258.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-15 11:00:00 | 254.00 | 258.90 | 258.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2024-03-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 11:15:00 | 251.58 | 257.44 | 257.49 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-03-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 13:15:00 | 260.70 | 257.80 | 257.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 14:15:00 | 263.01 | 258.84 | 258.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-18 09:15:00 | 258.31 | 259.23 | 258.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 09:15:00 | 258.31 | 259.23 | 258.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 258.31 | 259.23 | 258.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-18 09:45:00 | 260.70 | 259.23 | 258.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 10:15:00 | 257.01 | 258.78 | 258.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-18 11:00:00 | 257.01 | 258.78 | 258.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 11:15:00 | 257.08 | 258.44 | 258.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-18 11:30:00 | 255.90 | 258.44 | 258.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2024-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 12:15:00 | 255.10 | 257.77 | 257.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-18 13:15:00 | 254.42 | 257.10 | 257.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 10:15:00 | 251.75 | 250.65 | 252.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-20 11:00:00 | 251.75 | 250.65 | 252.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 11:15:00 | 252.79 | 251.08 | 252.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 12:00:00 | 252.79 | 251.08 | 252.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 260.19 | 252.90 | 253.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 13:00:00 | 260.19 | 252.90 | 253.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2024-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 13:15:00 | 262.46 | 254.81 | 254.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 09:15:00 | 270.40 | 262.99 | 259.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 265.32 | 266.24 | 262.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-22 15:00:00 | 265.32 | 266.24 | 262.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 09:15:00 | 287.90 | 273.64 | 268.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 12:30:00 | 290.80 | 281.31 | 273.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-03 14:15:00 | 319.88 | 310.03 | 305.36 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2024-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 14:15:00 | 301.65 | 304.82 | 304.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-05 09:15:00 | 295.90 | 302.87 | 303.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 15:15:00 | 299.06 | 298.59 | 300.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-08 09:15:00 | 302.36 | 298.59 | 300.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 300.00 | 298.87 | 300.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-08 12:30:00 | 297.94 | 299.10 | 300.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-10 09:15:00 | 283.04 | 288.10 | 292.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-10 10:15:00 | 291.81 | 288.84 | 292.41 | SL hit (close>ema200) qty=0.50 sl=288.84 alert=retest2 |

### Cycle 58 — BUY (started 2024-04-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 09:15:00 | 293.70 | 286.83 | 286.65 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-04-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 13:15:00 | 283.70 | 286.37 | 286.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 14:15:00 | 279.22 | 284.94 | 285.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-22 09:15:00 | 274.19 | 272.76 | 277.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-22 09:15:00 | 274.19 | 272.76 | 277.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 274.19 | 272.76 | 277.25 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 09:15:00 | 287.27 | 279.64 | 278.83 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 09:15:00 | 281.58 | 283.44 | 283.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-29 14:15:00 | 276.43 | 280.11 | 281.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 09:15:00 | 280.73 | 279.90 | 281.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 09:15:00 | 280.73 | 279.90 | 281.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 280.73 | 279.90 | 281.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 10:00:00 | 280.73 | 279.90 | 281.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 282.30 | 280.38 | 281.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 10:45:00 | 282.81 | 280.38 | 281.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 11:15:00 | 281.90 | 280.68 | 281.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 12:30:00 | 281.01 | 280.53 | 281.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:15:00 | 266.96 | 272.13 | 275.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-07 09:15:00 | 268.31 | 266.12 | 270.01 | SL hit (close>ema200) qty=0.50 sl=266.12 alert=retest2 |

### Cycle 62 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 255.19 | 250.07 | 249.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 262.90 | 254.19 | 251.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 269.00 | 273.19 | 270.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 269.00 | 273.19 | 270.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 269.00 | 273.19 | 270.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:00:00 | 269.00 | 273.19 | 270.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 265.81 | 271.71 | 270.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 11:00:00 | 265.81 | 271.71 | 270.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2024-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 12:15:00 | 265.30 | 269.20 | 269.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 262.50 | 266.61 | 268.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 09:15:00 | 263.65 | 262.44 | 264.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-23 10:00:00 | 263.65 | 262.44 | 264.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 263.79 | 262.93 | 264.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 13:00:00 | 263.38 | 263.02 | 264.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 14:15:00 | 262.94 | 263.18 | 264.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 09:45:00 | 261.80 | 261.25 | 262.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 10:15:00 | 250.21 | 253.92 | 256.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 14:15:00 | 249.79 | 251.71 | 254.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 14:15:00 | 248.71 | 251.71 | 254.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-31 13:15:00 | 246.93 | 246.54 | 248.92 | SL hit (close>ema200) qty=0.50 sl=246.54 alert=retest2 |

### Cycle 64 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 256.73 | 250.91 | 250.32 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 235.83 | 250.70 | 251.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 218.43 | 244.24 | 248.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 14:15:00 | 235.25 | 234.70 | 238.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 15:00:00 | 235.25 | 234.70 | 238.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 245.67 | 236.94 | 239.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 245.67 | 236.94 | 239.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 248.39 | 239.23 | 240.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 248.26 | 239.23 | 240.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 249.99 | 242.41 | 241.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 15:15:00 | 252.40 | 246.85 | 243.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 10:15:00 | 264.38 | 264.94 | 261.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 10:45:00 | 264.10 | 264.94 | 261.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 262.40 | 264.07 | 261.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 13:30:00 | 263.20 | 263.90 | 261.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 14:30:00 | 264.01 | 263.90 | 261.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:45:00 | 263.04 | 264.02 | 262.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 15:15:00 | 263.40 | 263.54 | 262.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 263.40 | 263.51 | 262.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 263.51 | 263.51 | 262.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 13:15:00 | 260.81 | 262.83 | 262.75 | SL hit (close<static) qty=1.00 sl=260.90 alert=retest2 |

### Cycle 67 — SELL (started 2024-06-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 14:15:00 | 260.30 | 262.33 | 262.53 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 11:15:00 | 263.98 | 262.69 | 262.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 15:15:00 | 264.90 | 263.65 | 263.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 261.80 | 263.28 | 263.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 261.80 | 263.28 | 263.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 261.80 | 263.28 | 263.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:45:00 | 262.92 | 263.28 | 263.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 261.50 | 262.92 | 262.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:45:00 | 261.30 | 262.92 | 262.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2024-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 11:15:00 | 261.44 | 262.63 | 262.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 09:15:00 | 260.30 | 261.39 | 262.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 259.82 | 259.71 | 260.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 259.82 | 259.71 | 260.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 259.82 | 259.71 | 260.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 13:30:00 | 258.11 | 259.50 | 260.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 09:30:00 | 258.19 | 258.43 | 259.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 09:15:00 | 259.81 | 257.12 | 257.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 259.81 | 257.12 | 257.07 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 10:15:00 | 255.17 | 258.34 | 258.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 09:15:00 | 235.67 | 253.66 | 256.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 09:15:00 | 234.61 | 234.56 | 239.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 09:15:00 | 239.00 | 235.42 | 237.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 239.00 | 235.42 | 237.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 13:15:00 | 235.05 | 235.86 | 237.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 14:30:00 | 233.71 | 235.59 | 236.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 10:15:00 | 234.60 | 235.08 | 236.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 14:15:00 | 223.30 | 230.43 | 233.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 14:15:00 | 222.87 | 230.43 | 233.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 09:15:00 | 222.02 | 228.55 | 231.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-12 09:15:00 | 224.30 | 222.84 | 225.06 | SL hit (close>ema200) qty=0.50 sl=222.84 alert=retest2 |

### Cycle 72 — BUY (started 2024-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 13:15:00 | 226.80 | 225.31 | 225.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 15:15:00 | 228.40 | 226.31 | 225.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 11:15:00 | 226.05 | 226.44 | 225.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 11:15:00 | 226.05 | 226.44 | 225.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 226.05 | 226.44 | 225.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 11:45:00 | 225.98 | 226.44 | 225.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 226.00 | 226.35 | 225.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:45:00 | 225.85 | 226.35 | 225.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 222.99 | 225.68 | 225.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 13:45:00 | 222.70 | 225.68 | 225.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2024-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 14:15:00 | 221.90 | 224.92 | 225.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 217.23 | 222.99 | 224.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 14:15:00 | 213.22 | 210.21 | 212.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 14:15:00 | 213.22 | 210.21 | 212.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 213.22 | 210.21 | 212.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 213.22 | 210.21 | 212.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 210.00 | 210.17 | 212.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:30:00 | 208.63 | 210.01 | 212.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 10:15:00 | 208.98 | 210.01 | 212.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:15:00 | 209.15 | 210.03 | 212.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 12:15:00 | 225.00 | 212.74 | 213.00 | SL hit (close>static) qty=1.00 sl=214.00 alert=retest2 |

### Cycle 74 — BUY (started 2024-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 13:15:00 | 225.55 | 215.31 | 214.14 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 12:15:00 | 213.63 | 215.91 | 216.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 14:15:00 | 213.15 | 215.06 | 215.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 11:15:00 | 215.90 | 214.58 | 215.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 11:15:00 | 215.90 | 214.58 | 215.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 11:15:00 | 215.90 | 214.58 | 215.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 12:00:00 | 215.90 | 214.58 | 215.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 12:15:00 | 215.27 | 214.72 | 215.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 13:30:00 | 214.00 | 214.38 | 214.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 09:30:00 | 214.47 | 214.07 | 214.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 10:15:00 | 214.49 | 214.07 | 214.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 11:15:00 | 214.00 | 214.22 | 214.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 11:15:00 | 213.90 | 214.15 | 214.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 13:45:00 | 213.45 | 213.94 | 214.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 09:30:00 | 213.58 | 213.86 | 214.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 12:30:00 | 213.57 | 213.72 | 214.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 15:00:00 | 213.50 | 213.69 | 214.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 09:15:00 | 222.22 | 215.43 | 214.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 09:15:00 | 222.22 | 215.43 | 214.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-01 10:15:00 | 223.80 | 219.33 | 217.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 14:15:00 | 224.29 | 225.43 | 222.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-02 15:00:00 | 224.29 | 225.43 | 222.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 15:15:00 | 223.50 | 225.05 | 222.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 09:15:00 | 215.86 | 225.05 | 222.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 217.34 | 223.51 | 222.30 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 11:15:00 | 215.00 | 220.25 | 220.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 10:15:00 | 209.74 | 213.95 | 217.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 210.30 | 209.46 | 212.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 10:45:00 | 210.37 | 209.46 | 212.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 213.00 | 210.74 | 212.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:00:00 | 213.00 | 210.74 | 212.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 215.40 | 211.67 | 212.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 215.40 | 211.67 | 212.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 215.40 | 212.42 | 213.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:15:00 | 215.95 | 212.42 | 213.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 216.79 | 214.07 | 213.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 13:15:00 | 218.31 | 216.60 | 215.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 215.49 | 216.56 | 215.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 215.49 | 216.56 | 215.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 215.49 | 216.56 | 215.77 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2024-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 14:15:00 | 209.87 | 214.74 | 215.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 09:15:00 | 209.60 | 213.08 | 214.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 11:15:00 | 210.84 | 210.32 | 211.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 12:00:00 | 210.84 | 210.32 | 211.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 210.93 | 210.45 | 211.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 13:45:00 | 210.48 | 210.50 | 211.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 14:30:00 | 210.40 | 210.67 | 211.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 11:15:00 | 212.89 | 211.49 | 211.72 | SL hit (close>static) qty=1.00 sl=212.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 214.71 | 212.43 | 212.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 216.26 | 213.20 | 212.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 15:15:00 | 268.00 | 268.45 | 256.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 09:15:00 | 263.95 | 268.45 | 256.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 257.43 | 262.17 | 258.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 257.43 | 262.17 | 258.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 256.70 | 261.08 | 258.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 11:30:00 | 258.60 | 260.44 | 258.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 12:00:00 | 257.90 | 260.44 | 258.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 268.00 | 258.34 | 258.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 11:15:00 | 260.16 | 261.08 | 261.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 11:15:00 | 260.16 | 261.08 | 261.15 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 13:15:00 | 262.50 | 261.21 | 261.18 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 09:15:00 | 259.70 | 261.13 | 261.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 10:15:00 | 257.95 | 260.50 | 260.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 260.40 | 258.96 | 259.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 260.40 | 258.96 | 259.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 260.40 | 258.96 | 259.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:00:00 | 260.40 | 258.96 | 259.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 259.54 | 259.08 | 259.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 11:15:00 | 259.01 | 259.08 | 259.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 10:00:00 | 259.20 | 258.88 | 259.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 10:00:00 | 258.83 | 257.55 | 258.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:15:00 | 246.06 | 251.17 | 253.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:15:00 | 246.24 | 251.17 | 253.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:15:00 | 245.89 | 251.17 | 253.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-09-09 09:15:00 | 233.11 | 243.89 | 248.17 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 84 — BUY (started 2024-09-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 12:15:00 | 245.82 | 242.71 | 242.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 09:15:00 | 250.12 | 245.27 | 243.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 09:15:00 | 257.11 | 260.58 | 257.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 257.11 | 260.58 | 257.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 257.11 | 260.58 | 257.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 257.11 | 260.58 | 257.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 255.35 | 259.54 | 257.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:00:00 | 255.35 | 259.54 | 257.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 254.53 | 258.54 | 256.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:45:00 | 251.28 | 258.54 | 256.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 14:15:00 | 251.69 | 255.53 | 255.82 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 259.36 | 256.21 | 256.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 12:15:00 | 260.40 | 257.62 | 256.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 15:15:00 | 257.60 | 258.19 | 257.28 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 09:15:00 | 263.50 | 258.19 | 257.28 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 262.16 | 260.93 | 259.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 09:30:00 | 260.49 | 260.93 | 259.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 259.50 | 260.53 | 259.54 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-24 11:15:00 | 259.50 | 260.53 | 259.54 | SL hit (close<ema400) qty=1.00 sl=259.54 alert=retest1 |

### Cycle 87 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 253.94 | 259.01 | 259.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 15:15:00 | 253.70 | 255.62 | 257.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 250.50 | 249.10 | 251.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-27 09:45:00 | 250.30 | 249.10 | 251.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 254.61 | 250.03 | 251.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 14:00:00 | 254.61 | 250.03 | 251.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 252.51 | 250.53 | 251.58 | EMA400 retest candle locked (from downside) |

### Cycle 88 — BUY (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 10:15:00 | 256.13 | 252.87 | 252.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 11:15:00 | 259.05 | 254.11 | 253.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 13:15:00 | 263.20 | 263.33 | 259.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 13:45:00 | 261.39 | 263.33 | 259.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 259.49 | 262.11 | 259.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:15:00 | 261.24 | 262.11 | 259.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 274.05 | 264.50 | 260.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 10:15:00 | 277.40 | 264.50 | 260.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 11:00:00 | 277.51 | 267.10 | 262.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 11:45:00 | 277.40 | 268.80 | 263.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 09:15:00 | 257.83 | 266.02 | 266.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 257.83 | 266.02 | 266.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 14:15:00 | 250.94 | 257.88 | 261.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 261.19 | 257.46 | 260.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 10:15:00 | 261.19 | 257.46 | 260.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 261.19 | 257.46 | 260.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 261.19 | 257.46 | 260.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 258.42 | 257.65 | 260.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 12:00:00 | 256.97 | 259.18 | 259.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 09:15:00 | 264.40 | 260.36 | 260.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 09:15:00 | 264.40 | 260.36 | 260.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 14:15:00 | 270.06 | 264.51 | 262.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 14:15:00 | 271.32 | 273.67 | 269.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 14:15:00 | 271.32 | 273.67 | 269.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 271.32 | 273.67 | 269.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:15:00 | 290.40 | 273.54 | 269.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-15 13:15:00 | 319.44 | 295.89 | 283.21 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-10-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 12:15:00 | 305.94 | 310.19 | 310.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 14:15:00 | 301.58 | 307.91 | 309.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 297.95 | 295.78 | 300.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:00:00 | 297.95 | 295.78 | 300.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 302.35 | 297.09 | 300.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:00:00 | 302.35 | 297.09 | 300.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 299.72 | 297.62 | 300.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 12:15:00 | 297.60 | 297.62 | 300.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 14:00:00 | 297.98 | 298.19 | 300.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 14:30:00 | 296.10 | 297.61 | 300.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 15:00:00 | 295.27 | 297.61 | 300.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 292.82 | 296.47 | 299.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:00:00 | 289.70 | 295.12 | 298.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:45:00 | 289.64 | 294.02 | 297.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 12:45:00 | 288.90 | 293.08 | 296.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 282.72 | 288.88 | 293.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 283.08 | 288.88 | 293.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 281.30 | 288.88 | 293.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 280.51 | 288.88 | 293.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 09:15:00 | 284.01 | 282.88 | 287.42 | SL hit (close>ema200) qty=0.50 sl=282.88 alert=retest2 |

### Cycle 92 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 297.00 | 289.81 | 288.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 307.62 | 294.31 | 291.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 10:15:00 | 299.97 | 300.60 | 297.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 10:30:00 | 299.89 | 300.60 | 297.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 295.45 | 302.26 | 300.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 295.45 | 302.26 | 300.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 293.46 | 300.50 | 299.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 293.46 | 300.50 | 299.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 292.27 | 297.72 | 298.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 14:15:00 | 288.30 | 295.03 | 296.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 291.68 | 289.92 | 292.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 14:00:00 | 291.68 | 289.92 | 292.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 297.28 | 291.83 | 293.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:30:00 | 297.80 | 291.83 | 293.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 294.48 | 292.36 | 293.24 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2024-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 14:15:00 | 294.53 | 293.76 | 293.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 09:15:00 | 296.67 | 294.37 | 294.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 10:15:00 | 294.00 | 294.29 | 294.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 10:15:00 | 294.00 | 294.29 | 294.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 294.00 | 294.29 | 294.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 291.74 | 294.29 | 294.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 11:15:00 | 291.40 | 293.71 | 293.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 14:15:00 | 288.61 | 291.90 | 292.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 09:15:00 | 293.49 | 291.72 | 292.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 09:15:00 | 293.49 | 291.72 | 292.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 293.49 | 291.72 | 292.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:00:00 | 293.49 | 291.72 | 292.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 292.32 | 291.84 | 292.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 11:15:00 | 290.50 | 291.84 | 292.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 13:15:00 | 275.97 | 278.73 | 282.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-13 14:15:00 | 261.45 | 266.88 | 273.33 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 96 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 273.74 | 270.62 | 270.38 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 265.30 | 269.39 | 269.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 264.81 | 268.48 | 269.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-21 11:15:00 | 269.15 | 268.34 | 269.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 11:15:00 | 269.15 | 268.34 | 269.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 269.15 | 268.34 | 269.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 12:00:00 | 269.15 | 268.34 | 269.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 12:15:00 | 268.57 | 268.39 | 269.15 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2024-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 15:15:00 | 273.30 | 269.86 | 269.66 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 14:15:00 | 269.10 | 269.49 | 269.53 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 284.01 | 272.43 | 270.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 14:15:00 | 292.60 | 282.26 | 278.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 290.80 | 291.23 | 287.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 13:00:00 | 290.80 | 291.23 | 287.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 288.27 | 290.66 | 288.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:45:00 | 289.38 | 290.66 | 288.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 289.40 | 290.41 | 288.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 293.55 | 290.41 | 288.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 09:45:00 | 292.15 | 290.79 | 289.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-05 13:15:00 | 322.91 | 313.13 | 307.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 331.44 | 339.22 | 339.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 09:15:00 | 323.50 | 335.09 | 337.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 291.62 | 288.63 | 294.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 10:00:00 | 291.62 | 288.63 | 294.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 291.46 | 289.20 | 294.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:30:00 | 292.13 | 289.20 | 294.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 290.56 | 289.66 | 292.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 14:30:00 | 287.44 | 290.46 | 291.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 12:15:00 | 288.02 | 290.41 | 291.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 12:45:00 | 286.81 | 289.99 | 291.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 15:00:00 | 288.26 | 289.53 | 290.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 289.80 | 289.48 | 290.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:45:00 | 289.38 | 289.48 | 290.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 291.60 | 289.91 | 290.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:00:00 | 291.60 | 289.91 | 290.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 293.29 | 290.58 | 290.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 12:00:00 | 293.29 | 290.58 | 290.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 288.91 | 290.22 | 290.57 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-30 15:15:00 | 292.70 | 291.04 | 290.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 15:15:00 | 292.70 | 291.04 | 290.91 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 09:15:00 | 286.20 | 290.07 | 290.48 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 14:15:00 | 293.10 | 290.96 | 290.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 09:15:00 | 302.19 | 293.53 | 291.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 11:15:00 | 297.60 | 298.81 | 296.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 11:15:00 | 297.60 | 298.81 | 296.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 297.60 | 298.81 | 296.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:45:00 | 297.15 | 298.81 | 296.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 14:15:00 | 297.10 | 299.06 | 297.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 15:00:00 | 297.10 | 299.06 | 297.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 15:15:00 | 297.90 | 298.83 | 297.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:15:00 | 298.73 | 298.83 | 297.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 294.12 | 297.89 | 297.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:00:00 | 294.12 | 297.89 | 297.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2025-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 10:15:00 | 288.52 | 296.01 | 296.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 14:15:00 | 285.95 | 290.59 | 293.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 09:15:00 | 291.40 | 289.91 | 292.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 291.40 | 289.91 | 292.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 291.40 | 289.91 | 292.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 10:30:00 | 286.22 | 289.00 | 291.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 09:15:00 | 271.91 | 279.83 | 285.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-07 14:15:00 | 276.46 | 275.74 | 280.88 | SL hit (close>ema200) qty=0.50 sl=275.74 alert=retest2 |

### Cycle 106 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 252.29 | 248.25 | 247.84 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 12:15:00 | 246.52 | 247.59 | 247.60 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2025-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 14:15:00 | 248.52 | 247.76 | 247.67 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2025-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 11:15:00 | 246.20 | 247.46 | 247.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 13:15:00 | 244.62 | 246.62 | 247.17 | Break + close below crossover candle low |

### Cycle 110 — BUY (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 11:15:00 | 253.27 | 247.14 | 247.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 12:15:00 | 257.80 | 249.27 | 248.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 251.40 | 253.22 | 250.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 251.40 | 253.22 | 250.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 249.12 | 252.40 | 250.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 249.12 | 252.40 | 250.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 250.11 | 251.94 | 250.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 250.80 | 251.94 | 250.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 250.35 | 251.62 | 250.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:15:00 | 250.75 | 251.62 | 250.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 250.95 | 251.49 | 250.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 15:00:00 | 252.89 | 251.77 | 250.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 09:15:00 | 242.83 | 250.10 | 250.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 242.83 | 250.10 | 250.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 13:15:00 | 241.52 | 245.25 | 247.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 245.82 | 245.21 | 246.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 245.82 | 245.21 | 246.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 245.82 | 245.21 | 246.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 247.22 | 245.21 | 246.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 246.06 | 245.38 | 246.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 244.90 | 245.38 | 246.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 246.26 | 245.31 | 246.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 14:00:00 | 246.26 | 245.31 | 246.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 244.02 | 245.06 | 246.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:15:00 | 242.10 | 245.06 | 246.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 229.99 | 234.15 | 238.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 09:15:00 | 217.89 | 224.35 | 230.95 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 112 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 231.46 | 227.49 | 227.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 11:15:00 | 233.62 | 229.97 | 228.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 231.60 | 233.28 | 231.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 231.60 | 233.28 | 231.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 231.60 | 233.28 | 231.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 231.60 | 233.28 | 231.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 238.00 | 234.23 | 232.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:45:00 | 244.38 | 237.62 | 233.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 10:15:00 | 231.90 | 234.37 | 234.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 10:15:00 | 231.90 | 234.37 | 234.41 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 246.50 | 236.56 | 235.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 11:15:00 | 251.05 | 241.31 | 237.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 11:15:00 | 245.60 | 246.79 | 243.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 12:00:00 | 245.60 | 246.79 | 243.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 243.46 | 245.53 | 243.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:45:00 | 243.67 | 245.53 | 243.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 244.25 | 245.27 | 243.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 244.84 | 245.27 | 243.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 245.18 | 245.25 | 243.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:00:00 | 246.75 | 245.55 | 243.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 15:15:00 | 241.20 | 243.02 | 243.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2025-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 15:15:00 | 241.20 | 243.02 | 243.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 235.74 | 241.57 | 242.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 15:15:00 | 227.47 | 227.46 | 232.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 09:15:00 | 222.47 | 227.46 | 232.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 234.30 | 227.76 | 231.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:00:00 | 234.30 | 227.76 | 231.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 234.11 | 229.03 | 231.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 234.11 | 229.03 | 231.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 235.21 | 231.92 | 232.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 235.21 | 231.92 | 232.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 235.50 | 232.64 | 232.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:15:00 | 232.85 | 232.64 | 232.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 09:15:00 | 236.05 | 233.32 | 233.10 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2025-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 13:15:00 | 231.40 | 233.04 | 233.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 226.70 | 231.28 | 232.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 14:15:00 | 221.41 | 220.39 | 222.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-18 15:00:00 | 221.41 | 220.39 | 222.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 226.48 | 221.59 | 222.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:45:00 | 227.30 | 221.59 | 222.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 225.50 | 222.37 | 222.96 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 230.20 | 223.94 | 223.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 12:15:00 | 231.10 | 225.37 | 224.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 12:15:00 | 236.28 | 236.30 | 233.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 13:00:00 | 236.28 | 236.30 | 233.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 229.31 | 234.50 | 233.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:00:00 | 229.31 | 234.50 | 233.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 226.68 | 232.94 | 232.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:00:00 | 226.68 | 232.94 | 232.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 11:15:00 | 228.83 | 232.11 | 232.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 09:15:00 | 223.83 | 228.28 | 230.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 218.98 | 218.74 | 222.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 15:00:00 | 218.98 | 218.74 | 222.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 220.00 | 215.84 | 218.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 209.78 | 215.84 | 218.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 10:15:00 | 199.29 | 210.31 | 215.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-04 09:15:00 | 203.30 | 202.89 | 208.81 | SL hit (close>ema200) qty=0.50 sl=202.89 alert=retest2 |

### Cycle 120 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 209.85 | 207.45 | 207.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 10:15:00 | 211.80 | 209.09 | 208.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 211.15 | 211.87 | 210.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 211.15 | 211.87 | 210.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 211.29 | 211.75 | 210.42 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2025-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 10:15:00 | 207.78 | 209.80 | 209.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 11:15:00 | 205.41 | 208.92 | 209.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 198.82 | 198.40 | 200.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 10:15:00 | 199.28 | 198.40 | 200.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 198.70 | 196.87 | 198.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 198.41 | 196.87 | 198.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 198.20 | 197.14 | 198.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:00:00 | 198.20 | 197.14 | 198.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 198.63 | 197.44 | 198.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:30:00 | 199.40 | 197.44 | 198.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 197.58 | 197.47 | 198.56 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 204.66 | 199.08 | 198.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 206.11 | 201.38 | 200.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 15:15:00 | 236.90 | 237.32 | 231.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 09:15:00 | 234.18 | 237.32 | 231.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 233.01 | 235.72 | 231.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:30:00 | 233.05 | 235.72 | 231.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 232.63 | 235.10 | 232.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 232.59 | 235.10 | 232.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 232.36 | 234.55 | 232.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:30:00 | 232.14 | 234.55 | 232.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 233.59 | 234.36 | 232.22 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 11:15:00 | 229.61 | 231.01 | 231.16 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 12:15:00 | 232.63 | 231.34 | 231.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 10:15:00 | 233.00 | 232.09 | 231.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-27 11:15:00 | 232.00 | 232.07 | 231.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 11:15:00 | 232.00 | 232.07 | 231.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 232.00 | 232.07 | 231.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:00:00 | 232.00 | 232.07 | 231.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 231.88 | 232.03 | 231.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:30:00 | 231.64 | 232.03 | 231.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 232.00 | 232.03 | 231.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:15:00 | 231.90 | 232.03 | 231.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 231.81 | 231.98 | 231.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 231.81 | 231.98 | 231.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 231.60 | 231.91 | 231.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 09:15:00 | 238.53 | 231.91 | 231.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 14:00:00 | 232.21 | 232.77 | 232.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 15:15:00 | 230.80 | 232.20 | 232.19 | SL hit (close<static) qty=1.00 sl=231.18 alert=retest2 |

### Cycle 125 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 231.17 | 236.27 | 236.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 229.00 | 234.82 | 235.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 218.75 | 216.68 | 222.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 218.75 | 216.68 | 222.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 218.75 | 216.68 | 222.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 11:00:00 | 216.45 | 219.77 | 221.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 227.89 | 222.44 | 222.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 227.89 | 222.44 | 222.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 231.21 | 226.73 | 224.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 226.40 | 232.20 | 230.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 226.40 | 232.20 | 230.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 226.40 | 232.20 | 230.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:00:00 | 229.71 | 231.70 | 230.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-22 09:15:00 | 252.68 | 245.98 | 240.69 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 237.70 | 246.21 | 246.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 14:15:00 | 232.01 | 240.30 | 243.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 238.00 | 236.38 | 238.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 238.00 | 236.38 | 238.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 238.00 | 236.38 | 238.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:30:00 | 238.00 | 236.38 | 238.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 233.81 | 232.57 | 234.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 233.81 | 232.57 | 234.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 232.54 | 232.57 | 234.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:30:00 | 233.99 | 232.57 | 234.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 234.47 | 232.44 | 233.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:45:00 | 234.70 | 232.44 | 233.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 235.15 | 232.98 | 233.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:00:00 | 235.15 | 232.98 | 233.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 234.60 | 233.59 | 233.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:30:00 | 234.47 | 233.59 | 233.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 13:15:00 | 236.37 | 234.15 | 233.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 14:15:00 | 238.95 | 235.11 | 234.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 233.84 | 235.45 | 234.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 233.84 | 235.45 | 234.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 233.84 | 235.45 | 234.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:45:00 | 233.55 | 235.45 | 234.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 233.01 | 234.96 | 234.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:30:00 | 233.50 | 234.96 | 234.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 231.40 | 233.89 | 234.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 15:15:00 | 230.00 | 232.26 | 233.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 234.66 | 232.45 | 233.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 10:15:00 | 234.66 | 232.45 | 233.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 234.66 | 232.45 | 233.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 10:30:00 | 233.70 | 232.45 | 233.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 234.69 | 232.90 | 233.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 13:00:00 | 233.72 | 233.06 | 233.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 14:15:00 | 236.23 | 233.96 | 233.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 14:15:00 | 236.23 | 233.96 | 233.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 15:15:00 | 238.20 | 234.81 | 234.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 235.98 | 237.96 | 236.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 235.98 | 237.96 | 236.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 235.98 | 237.96 | 236.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 235.98 | 237.96 | 236.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 232.81 | 236.93 | 235.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 232.81 | 236.93 | 235.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 231.75 | 235.89 | 235.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 227.51 | 235.89 | 235.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 228.40 | 234.39 | 234.93 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 243.47 | 234.43 | 234.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 247.58 | 240.96 | 237.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 13:15:00 | 280.14 | 280.87 | 275.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 14:00:00 | 280.14 | 280.87 | 275.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 277.26 | 280.15 | 275.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:45:00 | 280.84 | 280.28 | 276.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-23 13:15:00 | 308.92 | 298.67 | 291.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 13:15:00 | 299.13 | 300.18 | 300.31 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 304.72 | 301.05 | 300.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 11:15:00 | 307.34 | 302.80 | 301.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 15:15:00 | 303.50 | 303.74 | 302.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 09:15:00 | 303.49 | 303.74 | 302.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 304.77 | 303.94 | 302.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 11:15:00 | 306.80 | 303.98 | 302.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 10:15:00 | 315.41 | 319.50 | 319.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 10:15:00 | 315.41 | 319.50 | 319.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 11:15:00 | 313.91 | 318.38 | 319.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 291.59 | 291.51 | 296.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 291.59 | 291.51 | 296.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 287.24 | 287.55 | 289.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:30:00 | 287.60 | 287.55 | 289.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 282.96 | 281.78 | 283.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 282.96 | 281.78 | 283.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 284.20 | 282.38 | 283.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 284.20 | 282.38 | 283.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 285.24 | 282.95 | 283.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:00:00 | 285.24 | 282.95 | 283.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 287.85 | 283.93 | 284.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:00:00 | 287.85 | 283.93 | 284.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 292.40 | 285.62 | 285.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 14:15:00 | 293.60 | 288.37 | 286.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 13:15:00 | 287.19 | 294.96 | 293.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 13:15:00 | 287.19 | 294.96 | 293.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 287.19 | 294.96 | 293.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:00:00 | 287.19 | 294.96 | 293.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 287.10 | 293.39 | 292.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:30:00 | 285.94 | 293.39 | 292.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 15:15:00 | 284.80 | 291.67 | 291.93 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 296.95 | 290.80 | 290.73 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 290.49 | 292.35 | 292.52 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 294.98 | 293.01 | 292.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 14:15:00 | 295.19 | 293.66 | 293.16 | Break + close above crossover candle high |

### Cycle 141 — SELL (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 09:15:00 | 278.95 | 290.90 | 292.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 270.90 | 277.64 | 281.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 273.00 | 270.84 | 274.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 09:45:00 | 273.30 | 270.84 | 274.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 274.62 | 273.17 | 274.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 11:45:00 | 272.25 | 273.15 | 274.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 14:00:00 | 272.66 | 273.08 | 273.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 15:15:00 | 272.50 | 273.09 | 273.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:45:00 | 272.30 | 272.57 | 273.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 270.60 | 269.70 | 271.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 10:45:00 | 268.49 | 269.24 | 270.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 09:30:00 | 264.28 | 267.33 | 269.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 13:45:00 | 268.18 | 267.45 | 268.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 11:00:00 | 267.80 | 268.11 | 268.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 270.39 | 268.57 | 268.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:00:00 | 270.39 | 268.57 | 268.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-16 12:15:00 | 272.20 | 269.29 | 269.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 12:15:00 | 272.20 | 269.29 | 269.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 277.40 | 271.91 | 270.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 269.19 | 272.42 | 271.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 09:15:00 | 269.19 | 272.42 | 271.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 269.19 | 272.42 | 271.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 269.19 | 272.42 | 271.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 269.02 | 271.74 | 271.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 268.59 | 271.74 | 271.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 268.69 | 270.69 | 270.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 13:15:00 | 267.81 | 270.12 | 270.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 272.07 | 269.84 | 270.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 272.07 | 269.84 | 270.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 272.07 | 269.84 | 270.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:00:00 | 272.07 | 269.84 | 270.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 271.43 | 270.16 | 270.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 271.06 | 270.16 | 270.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 270.50 | 270.55 | 270.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:45:00 | 270.91 | 270.55 | 270.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 270.70 | 270.58 | 270.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 280.23 | 270.58 | 270.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 280.68 | 272.60 | 271.50 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 273.99 | 277.73 | 278.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 272.48 | 275.56 | 276.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 15:15:00 | 264.62 | 263.35 | 266.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-30 09:15:00 | 263.36 | 263.35 | 266.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 264.50 | 263.24 | 265.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:45:00 | 264.43 | 263.24 | 265.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 257.92 | 258.82 | 260.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 257.75 | 258.82 | 260.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 261.36 | 259.36 | 260.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 261.36 | 259.36 | 260.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 262.27 | 259.94 | 260.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 262.27 | 259.94 | 260.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 263.18 | 261.32 | 261.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 09:15:00 | 263.28 | 261.72 | 261.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 10:15:00 | 259.52 | 261.28 | 261.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 10:15:00 | 259.52 | 261.28 | 261.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 259.52 | 261.28 | 261.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 11:00:00 | 259.52 | 261.28 | 261.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 262.00 | 261.42 | 261.28 | EMA400 retest candle locked (from upside) |

### Cycle 147 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 259.80 | 261.10 | 261.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 256.80 | 259.96 | 260.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 11:15:00 | 262.00 | 259.62 | 260.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 11:15:00 | 262.00 | 259.62 | 260.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 262.00 | 259.62 | 260.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:00:00 | 262.00 | 259.62 | 260.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 261.08 | 259.91 | 260.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 09:15:00 | 260.00 | 260.53 | 260.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 10:15:00 | 258.22 | 260.49 | 260.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 12:30:00 | 260.07 | 260.10 | 260.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 263.55 | 261.00 | 260.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 263.55 | 261.00 | 260.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 15:15:00 | 264.60 | 261.72 | 261.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 260.88 | 261.55 | 261.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 260.88 | 261.55 | 261.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 260.88 | 261.55 | 261.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:00:00 | 260.88 | 261.55 | 261.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 260.02 | 261.24 | 260.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:00:00 | 260.02 | 261.24 | 260.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 258.81 | 260.76 | 260.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 12:15:00 | 258.29 | 260.26 | 260.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 258.24 | 253.84 | 255.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 258.24 | 253.84 | 255.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 258.24 | 253.84 | 255.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:00:00 | 258.24 | 253.84 | 255.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 257.37 | 254.55 | 255.25 | EMA400 retest candle locked (from downside) |

### Cycle 150 — BUY (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 12:15:00 | 261.66 | 256.77 | 256.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 14:15:00 | 263.00 | 258.90 | 257.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 266.60 | 266.64 | 264.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 266.60 | 266.64 | 264.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 266.60 | 266.64 | 264.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:45:00 | 265.36 | 266.64 | 264.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 267.58 | 267.54 | 265.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:45:00 | 267.18 | 267.54 | 265.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 265.68 | 267.33 | 266.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:00:00 | 265.68 | 267.33 | 266.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 265.46 | 266.95 | 266.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:45:00 | 265.09 | 266.95 | 266.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 260.73 | 268.50 | 267.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 260.73 | 268.50 | 267.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2025-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 12:15:00 | 257.83 | 266.36 | 266.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 14:15:00 | 253.66 | 262.15 | 264.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 225.47 | 223.66 | 228.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 09:45:00 | 226.93 | 223.66 | 228.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 226.51 | 225.06 | 227.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 227.00 | 225.06 | 227.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 230.67 | 226.49 | 227.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:00:00 | 230.67 | 226.49 | 227.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 230.89 | 227.37 | 228.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:00:00 | 229.51 | 228.44 | 228.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 10:00:00 | 229.35 | 227.70 | 227.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 14:15:00 | 229.31 | 227.54 | 227.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 14:15:00 | 229.31 | 227.54 | 227.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 234.49 | 229.27 | 228.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 15:15:00 | 233.66 | 233.71 | 232.33 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:15:00 | 235.50 | 233.71 | 232.33 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 231.39 | 233.65 | 232.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 231.39 | 233.65 | 232.70 | SL hit (close<ema400) qty=1.00 sl=232.70 alert=retest1 |

### Cycle 153 — SELL (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 11:15:00 | 223.55 | 230.92 | 231.82 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 15:15:00 | 224.95 | 224.55 | 224.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 227.19 | 225.08 | 224.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 225.17 | 226.03 | 225.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 225.17 | 226.03 | 225.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 225.17 | 226.03 | 225.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 13:45:00 | 227.31 | 225.95 | 225.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 12:15:00 | 224.10 | 225.52 | 225.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 12:15:00 | 224.10 | 225.52 | 225.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 223.40 | 225.10 | 225.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 14:15:00 | 225.36 | 225.15 | 225.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 14:15:00 | 225.36 | 225.15 | 225.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 225.36 | 225.15 | 225.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 15:00:00 | 225.36 | 225.15 | 225.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 224.30 | 224.98 | 225.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 222.81 | 224.98 | 225.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 220.09 | 224.00 | 224.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 217.66 | 220.12 | 221.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:00:00 | 217.18 | 219.53 | 221.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 217.27 | 214.70 | 214.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 217.27 | 214.70 | 214.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 221.48 | 216.73 | 215.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 223.93 | 224.59 | 222.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 11:30:00 | 223.75 | 224.59 | 222.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 223.66 | 224.98 | 223.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 223.66 | 224.98 | 223.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 222.24 | 224.43 | 223.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 222.24 | 224.43 | 223.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 221.78 | 223.90 | 223.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:00:00 | 221.78 | 223.90 | 223.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 222.80 | 223.68 | 223.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 13:15:00 | 222.97 | 223.68 | 223.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 221.20 | 222.96 | 222.89 | SL hit (close<static) qty=1.00 sl=221.75 alert=retest2 |

### Cycle 157 — SELL (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 15:15:00 | 221.49 | 222.67 | 222.76 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 223.68 | 222.97 | 222.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 11:15:00 | 226.80 | 223.74 | 223.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 226.87 | 228.98 | 227.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 226.87 | 228.98 | 227.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 226.87 | 228.98 | 227.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 226.97 | 228.98 | 227.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 225.93 | 228.37 | 227.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:45:00 | 225.91 | 228.37 | 227.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 226.59 | 227.80 | 227.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:00:00 | 226.59 | 227.80 | 227.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 226.50 | 227.54 | 227.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:00:00 | 226.50 | 227.54 | 227.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 234.98 | 229.03 | 227.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 242.05 | 230.34 | 228.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 12:30:00 | 235.74 | 234.10 | 231.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 247.80 | 252.68 | 252.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 247.80 | 252.68 | 252.81 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 254.14 | 251.49 | 251.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 254.98 | 252.64 | 251.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 12:15:00 | 254.82 | 254.91 | 253.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 13:00:00 | 254.82 | 254.91 | 253.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 252.54 | 254.44 | 253.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:00:00 | 252.54 | 254.44 | 253.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 252.53 | 254.05 | 253.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:45:00 | 251.32 | 254.05 | 253.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 253.07 | 253.10 | 253.03 | EMA400 retest candle locked (from upside) |

### Cycle 161 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 251.27 | 252.73 | 252.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 249.01 | 251.73 | 252.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 254.93 | 250.93 | 251.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 12:15:00 | 254.93 | 250.93 | 251.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 254.93 | 250.93 | 251.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:45:00 | 254.86 | 250.93 | 251.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 257.50 | 252.24 | 252.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 14:15:00 | 261.70 | 254.14 | 252.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 13:15:00 | 272.10 | 272.51 | 270.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 14:00:00 | 272.10 | 272.51 | 270.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 280.19 | 281.94 | 279.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 280.19 | 281.94 | 279.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 279.09 | 281.37 | 279.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 280.20 | 281.37 | 279.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 281.04 | 281.30 | 279.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:00:00 | 282.50 | 281.18 | 279.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 09:45:00 | 282.99 | 281.89 | 280.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 11:15:00 | 282.95 | 281.86 | 280.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 11:15:00 | 278.05 | 280.40 | 280.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 278.05 | 280.40 | 280.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 12:15:00 | 276.15 | 279.55 | 280.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 15:15:00 | 270.08 | 268.94 | 271.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 09:15:00 | 273.87 | 268.94 | 271.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 273.17 | 269.79 | 271.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:30:00 | 272.55 | 269.79 | 271.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 274.90 | 270.81 | 271.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 274.90 | 270.81 | 271.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 274.60 | 272.74 | 272.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 15:15:00 | 275.02 | 273.48 | 272.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 270.77 | 274.21 | 273.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 270.77 | 274.21 | 273.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 270.77 | 274.21 | 273.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 270.77 | 274.21 | 273.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — SELL (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 10:15:00 | 269.00 | 273.17 | 273.38 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 13:15:00 | 274.14 | 272.61 | 272.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 14:15:00 | 276.97 | 273.48 | 272.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 265.97 | 276.73 | 275.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 265.97 | 276.73 | 275.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 265.97 | 276.73 | 275.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:30:00 | 267.17 | 276.73 | 275.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 267.82 | 274.95 | 275.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 12:15:00 | 263.36 | 266.87 | 270.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 13:15:00 | 263.71 | 263.63 | 266.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 14:00:00 | 263.71 | 263.63 | 266.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 252.00 | 250.32 | 252.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:45:00 | 252.69 | 250.32 | 252.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 256.47 | 251.55 | 252.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 256.47 | 251.55 | 252.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 256.10 | 252.46 | 253.27 | EMA400 retest candle locked (from downside) |

### Cycle 168 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 257.85 | 254.23 | 253.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 259.10 | 255.69 | 254.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 257.78 | 257.88 | 256.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 257.78 | 257.88 | 256.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 257.78 | 257.88 | 256.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 256.99 | 257.88 | 256.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 257.98 | 257.81 | 256.68 | EMA400 retest candle locked (from upside) |

### Cycle 169 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 255.19 | 256.63 | 256.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 253.30 | 255.77 | 256.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 09:15:00 | 254.27 | 251.90 | 253.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 09:15:00 | 254.27 | 251.90 | 253.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 254.27 | 251.90 | 253.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 11:30:00 | 250.35 | 251.39 | 252.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 253.29 | 250.87 | 250.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 253.29 | 250.87 | 250.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 255.67 | 251.83 | 251.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 256.18 | 256.41 | 254.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 14:15:00 | 256.18 | 256.41 | 254.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 256.18 | 256.41 | 254.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 256.49 | 256.41 | 254.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 254.70 | 256.07 | 254.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 256.14 | 256.07 | 254.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 256.17 | 256.09 | 254.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:30:00 | 257.20 | 256.25 | 255.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:30:00 | 257.20 | 256.41 | 255.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 13:15:00 | 253.48 | 255.63 | 255.09 | SL hit (close<static) qty=1.00 sl=253.60 alert=retest2 |

### Cycle 171 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 252.40 | 254.48 | 254.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 10:15:00 | 251.65 | 253.68 | 254.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 236.90 | 236.78 | 241.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 235.32 | 236.78 | 241.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 237.10 | 235.21 | 236.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:15:00 | 238.89 | 235.21 | 236.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 237.10 | 235.59 | 236.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 10:15:00 | 236.03 | 235.59 | 236.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 11:15:00 | 236.00 | 235.84 | 236.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 14:15:00 | 238.45 | 237.23 | 237.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 14:15:00 | 238.45 | 237.23 | 237.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 11:15:00 | 241.60 | 238.63 | 237.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 238.91 | 239.26 | 238.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 238.91 | 239.26 | 238.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 239.66 | 240.66 | 239.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:00:00 | 239.66 | 240.66 | 239.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 241.20 | 240.77 | 239.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 240.16 | 240.77 | 239.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 243.45 | 241.29 | 240.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 12:15:00 | 244.97 | 242.10 | 240.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 12:45:00 | 246.13 | 242.90 | 241.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 237.33 | 241.40 | 241.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 237.33 | 241.40 | 241.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 10:15:00 | 236.15 | 240.35 | 241.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 235.11 | 234.64 | 236.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:00:00 | 235.11 | 234.64 | 236.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 236.80 | 235.07 | 236.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 237.17 | 235.07 | 236.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 237.19 | 235.49 | 236.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 243.35 | 235.49 | 236.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 242.92 | 236.98 | 237.42 | EMA400 retest candle locked (from downside) |

### Cycle 174 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 242.11 | 238.00 | 237.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 12:15:00 | 244.40 | 240.16 | 238.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 11:15:00 | 269.22 | 269.62 | 261.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 12:00:00 | 269.22 | 269.62 | 261.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 267.20 | 269.00 | 264.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 264.07 | 269.00 | 264.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 263.64 | 266.81 | 264.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:00:00 | 263.64 | 266.81 | 264.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 263.56 | 266.16 | 264.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 15:15:00 | 263.30 | 266.16 | 264.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 263.30 | 265.58 | 264.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 259.38 | 265.58 | 264.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2026-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 09:15:00 | 255.74 | 263.62 | 263.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 10:15:00 | 253.41 | 261.57 | 262.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 258.19 | 256.61 | 259.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 258.19 | 256.61 | 259.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 258.19 | 256.61 | 259.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 258.33 | 256.61 | 259.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 256.11 | 256.51 | 259.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 255.71 | 256.51 | 259.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:30:00 | 255.43 | 256.26 | 258.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:45:00 | 255.61 | 256.43 | 257.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 260.46 | 254.44 | 254.78 | SL hit (close>static) qty=1.00 sl=259.20 alert=retest2 |

### Cycle 176 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 259.97 | 255.55 | 255.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 12:15:00 | 261.34 | 257.54 | 256.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 258.27 | 259.19 | 257.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 10:15:00 | 258.27 | 259.19 | 257.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 258.27 | 259.19 | 257.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 257.72 | 259.19 | 257.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 257.10 | 258.77 | 257.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 257.10 | 258.77 | 257.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 258.34 | 258.68 | 257.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 14:30:00 | 258.69 | 258.56 | 257.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 15:00:00 | 258.78 | 258.56 | 257.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 11:30:00 | 258.69 | 258.75 | 258.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 12:00:00 | 259.76 | 258.75 | 258.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 257.50 | 258.50 | 258.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 257.50 | 258.50 | 258.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 255.98 | 258.00 | 257.91 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 255.98 | 258.00 | 257.91 | SL hit (close<static) qty=1.00 sl=256.39 alert=retest2 |

### Cycle 177 — SELL (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 14:15:00 | 255.45 | 257.49 | 257.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 15:15:00 | 252.00 | 256.39 | 257.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 235.71 | 235.28 | 241.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:45:00 | 235.43 | 235.28 | 241.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 240.00 | 236.95 | 241.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 256.04 | 236.95 | 241.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 256.93 | 240.95 | 242.66 | EMA400 retest candle locked (from downside) |

### Cycle 178 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 260.30 | 244.82 | 244.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 264.44 | 248.74 | 246.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 11:15:00 | 264.20 | 264.29 | 259.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 11:45:00 | 264.24 | 264.29 | 259.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 258.59 | 263.05 | 260.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:45:00 | 257.98 | 263.05 | 260.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 258.97 | 262.23 | 260.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 257.91 | 262.23 | 260.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 258.85 | 261.08 | 260.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 258.85 | 261.08 | 260.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 272.53 | 276.50 | 274.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 272.50 | 276.50 | 274.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 273.23 | 275.84 | 274.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:45:00 | 272.71 | 275.84 | 274.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 274.89 | 274.59 | 274.19 | EMA400 retest candle locked (from upside) |

### Cycle 179 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 268.19 | 273.44 | 273.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 258.88 | 268.28 | 270.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 15:15:00 | 257.50 | 256.97 | 260.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 09:15:00 | 265.07 | 256.97 | 260.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 258.95 | 257.37 | 260.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 261.67 | 257.37 | 260.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 256.09 | 258.06 | 259.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:30:00 | 255.56 | 257.26 | 258.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:15:00 | 242.78 | 246.36 | 248.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 248.76 | 246.15 | 247.63 | SL hit (close>ema200) qty=0.50 sl=246.15 alert=retest2 |

### Cycle 180 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 225.17 | 221.11 | 220.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 226.00 | 222.73 | 221.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 221.60 | 222.63 | 221.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 10:15:00 | 221.60 | 222.63 | 221.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 221.60 | 222.63 | 221.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:00:00 | 221.60 | 222.63 | 221.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 221.54 | 222.41 | 221.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:30:00 | 221.33 | 222.41 | 221.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 219.91 | 221.91 | 221.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 219.91 | 221.91 | 221.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 219.40 | 221.41 | 221.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 219.40 | 221.41 | 221.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 218.27 | 220.78 | 221.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 214.90 | 219.31 | 220.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 11:15:00 | 218.00 | 215.14 | 216.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 11:15:00 | 218.00 | 215.14 | 216.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 218.00 | 215.14 | 216.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:00:00 | 218.00 | 215.14 | 216.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 214.31 | 214.97 | 216.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 10:15:00 | 213.41 | 215.28 | 216.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 220.11 | 215.57 | 215.68 | SL hit (close>static) qty=1.00 sl=218.20 alert=retest2 |

### Cycle 182 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 217.21 | 215.90 | 215.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 227.78 | 218.85 | 217.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 15:15:00 | 232.10 | 232.16 | 228.56 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:15:00 | 236.01 | 232.16 | 228.56 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:30:00 | 234.80 | 233.05 | 229.61 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 230.20 | 232.08 | 230.43 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-20 15:15:00 | 230.20 | 232.08 | 230.43 | SL hit (close<ema400) qty=1.00 sl=230.43 alert=retest1 |

### Cycle 183 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 219.85 | 227.99 | 228.76 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 230.48 | 226.67 | 226.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 237.40 | 229.35 | 227.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 237.06 | 238.09 | 234.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:00:00 | 237.06 | 238.09 | 234.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 234.44 | 236.97 | 234.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 234.44 | 236.97 | 234.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 236.85 | 236.94 | 234.44 | EMA400 retest candle locked (from upside) |

### Cycle 185 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 230.37 | 233.27 | 233.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 227.50 | 231.50 | 232.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 240.40 | 232.83 | 233.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 240.40 | 232.83 | 233.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 240.40 | 232.83 | 233.09 | EMA400 retest candle locked (from downside) |

### Cycle 186 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 240.30 | 234.32 | 233.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 243.26 | 236.11 | 234.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 229.34 | 237.25 | 236.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 229.34 | 237.25 | 236.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 229.34 | 237.25 | 236.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 229.34 | 237.25 | 236.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 242.87 | 239.36 | 237.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:30:00 | 244.75 | 239.91 | 238.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:45:00 | 244.95 | 242.22 | 239.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 09:15:00 | 269.23 | 262.83 | 254.90 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 187 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 310.45 | 319.96 | 320.97 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 319.31 | 318.51 | 318.42 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 314.60 | 317.68 | 318.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 15:15:00 | 313.00 | 316.10 | 317.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 10:15:00 | 311.50 | 308.63 | 310.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 10:15:00 | 311.50 | 308.63 | 310.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 311.50 | 308.63 | 310.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:00:00 | 311.50 | 308.63 | 310.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 312.75 | 309.45 | 310.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:30:00 | 315.50 | 309.45 | 310.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 14:15:00 | 314.70 | 311.59 | 311.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 317.00 | 313.06 | 312.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 13:15:00 | 323.55 | 324.82 | 321.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 14:00:00 | 323.55 | 324.82 | 321.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-15 14:30:00 | 130.84 | 2023-05-16 12:15:00 | 130.54 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2023-05-16 09:15:00 | 131.70 | 2023-05-16 12:15:00 | 130.54 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2023-05-22 13:00:00 | 124.98 | 2023-05-22 15:15:00 | 125.20 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2023-06-02 12:30:00 | 131.21 | 2023-06-02 14:15:00 | 131.39 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2023-06-22 13:30:00 | 158.99 | 2023-06-22 15:15:00 | 156.80 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2023-07-07 09:15:00 | 176.88 | 2023-07-07 11:15:00 | 172.92 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2023-07-13 09:15:00 | 178.00 | 2023-07-14 10:15:00 | 173.55 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2023-07-13 15:15:00 | 175.73 | 2023-07-14 10:15:00 | 173.55 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2023-07-14 10:00:00 | 175.36 | 2023-07-14 10:15:00 | 173.55 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2023-07-26 09:15:00 | 160.72 | 2023-07-27 13:15:00 | 158.60 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2023-07-26 09:45:00 | 160.60 | 2023-07-27 13:15:00 | 158.60 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2023-07-27 09:15:00 | 160.78 | 2023-07-27 13:15:00 | 158.60 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest1 | 2023-08-14 11:15:00 | 174.85 | 2023-08-18 09:15:00 | 174.90 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2023-08-17 09:15:00 | 178.18 | 2023-08-18 09:15:00 | 174.90 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2023-08-17 11:00:00 | 178.27 | 2023-08-18 09:15:00 | 174.90 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2023-08-21 12:45:00 | 176.22 | 2023-08-22 09:15:00 | 178.90 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2023-08-21 13:15:00 | 176.13 | 2023-08-22 09:15:00 | 178.90 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2023-08-21 14:30:00 | 176.20 | 2023-08-22 09:15:00 | 178.90 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2023-08-21 15:00:00 | 176.07 | 2023-08-22 09:15:00 | 178.90 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2023-09-05 11:45:00 | 187.33 | 2023-09-05 13:15:00 | 185.50 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2023-09-22 09:15:00 | 182.93 | 2023-09-22 12:15:00 | 187.90 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2023-09-29 09:15:00 | 187.81 | 2023-09-29 11:15:00 | 184.59 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2023-10-04 12:30:00 | 183.20 | 2023-10-04 14:15:00 | 186.05 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2023-10-04 13:15:00 | 183.48 | 2023-10-04 14:15:00 | 186.05 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest1 | 2023-10-19 11:45:00 | 217.72 | 2023-10-20 10:15:00 | 228.61 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-10-19 11:45:00 | 217.72 | 2023-10-23 09:15:00 | 239.49 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2023-10-25 10:15:00 | 227.67 | 2023-10-27 11:15:00 | 250.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-25 11:15:00 | 227.87 | 2023-10-27 13:15:00 | 250.66 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-25 15:00:00 | 231.28 | 2023-10-27 14:15:00 | 254.41 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-26 10:00:00 | 228.40 | 2023-10-27 14:15:00 | 251.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-26 11:45:00 | 234.18 | 2023-10-30 10:15:00 | 257.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-13 14:15:00 | 275.99 | 2023-11-15 14:15:00 | 303.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-20 11:45:00 | 330.74 | 2023-12-20 13:15:00 | 317.53 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2024-01-01 14:00:00 | 353.00 | 2024-01-08 09:15:00 | 388.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-03 10:00:00 | 351.40 | 2024-01-08 09:15:00 | 386.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-03 11:00:00 | 350.73 | 2024-01-08 09:15:00 | 385.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-04 09:15:00 | 372.50 | 2024-01-16 09:15:00 | 342.42 | STOP_HIT | 1.00 | -8.08% |
| BUY | retest2 | 2024-01-09 15:15:00 | 379.90 | 2024-01-16 09:15:00 | 342.42 | STOP_HIT | 1.00 | -9.87% |
| BUY | retest2 | 2024-01-10 09:30:00 | 378.82 | 2024-01-16 09:15:00 | 342.42 | STOP_HIT | 1.00 | -9.61% |
| BUY | retest2 | 2024-01-10 14:45:00 | 375.90 | 2024-01-16 09:15:00 | 342.42 | STOP_HIT | 1.00 | -8.91% |
| SELL | retest1 | 2024-01-19 10:30:00 | 331.90 | 2024-01-23 09:15:00 | 315.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-01-19 11:00:00 | 332.00 | 2024-01-23 09:15:00 | 315.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-01-20 09:15:00 | 331.44 | 2024-01-23 09:15:00 | 314.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-01-19 10:30:00 | 331.90 | 2024-01-23 14:15:00 | 298.71 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2024-01-19 11:00:00 | 332.00 | 2024-01-23 14:15:00 | 298.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2024-01-20 09:15:00 | 331.44 | 2024-01-23 14:15:00 | 298.30 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-02-09 12:15:00 | 339.45 | 2024-02-12 09:15:00 | 323.73 | STOP_HIT | 1.00 | -4.63% |
| BUY | retest2 | 2024-02-09 13:15:00 | 339.00 | 2024-02-12 09:15:00 | 323.73 | STOP_HIT | 1.00 | -4.50% |
| BUY | retest2 | 2024-02-09 13:45:00 | 339.77 | 2024-02-12 09:15:00 | 323.73 | STOP_HIT | 1.00 | -4.72% |
| BUY | retest2 | 2024-02-09 15:15:00 | 339.30 | 2024-02-12 09:15:00 | 323.73 | STOP_HIT | 1.00 | -4.59% |
| SELL | retest2 | 2024-02-23 10:15:00 | 305.33 | 2024-02-28 10:15:00 | 290.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-23 12:15:00 | 305.50 | 2024-02-28 10:15:00 | 290.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-26 09:45:00 | 304.56 | 2024-02-28 10:15:00 | 289.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-23 10:15:00 | 305.33 | 2024-03-01 09:15:00 | 284.95 | STOP_HIT | 0.50 | 6.67% |
| SELL | retest2 | 2024-02-23 12:15:00 | 305.50 | 2024-03-01 09:15:00 | 284.95 | STOP_HIT | 0.50 | 6.73% |
| SELL | retest2 | 2024-02-26 09:45:00 | 304.56 | 2024-03-01 09:15:00 | 284.95 | STOP_HIT | 0.50 | 6.44% |
| SELL | retest2 | 2024-03-14 13:00:00 | 262.71 | 2024-03-14 14:15:00 | 264.52 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-03-14 13:30:00 | 259.55 | 2024-03-14 14:15:00 | 264.52 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-03-27 12:30:00 | 290.80 | 2024-04-03 14:15:00 | 319.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-04-08 12:30:00 | 297.94 | 2024-04-10 09:15:00 | 283.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-08 12:30:00 | 297.94 | 2024-04-10 10:15:00 | 291.81 | STOP_HIT | 0.50 | 2.06% |
| SELL | retest2 | 2024-04-30 12:30:00 | 281.01 | 2024-05-06 09:15:00 | 266.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-30 12:30:00 | 281.01 | 2024-05-07 09:15:00 | 268.31 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2024-05-23 13:00:00 | 263.38 | 2024-05-29 10:15:00 | 250.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-23 14:15:00 | 262.94 | 2024-05-29 14:15:00 | 249.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 09:45:00 | 261.80 | 2024-05-29 14:15:00 | 248.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-23 13:00:00 | 263.38 | 2024-05-31 13:15:00 | 246.93 | STOP_HIT | 0.50 | 6.25% |
| SELL | retest2 | 2024-05-23 14:15:00 | 262.94 | 2024-05-31 13:15:00 | 246.93 | STOP_HIT | 0.50 | 6.09% |
| SELL | retest2 | 2024-05-27 09:45:00 | 261.80 | 2024-05-31 13:15:00 | 246.93 | STOP_HIT | 0.50 | 5.68% |
| BUY | retest2 | 2024-06-12 13:30:00 | 263.20 | 2024-06-14 13:15:00 | 260.81 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-06-12 14:30:00 | 264.01 | 2024-06-14 13:15:00 | 260.81 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-06-13 12:45:00 | 263.04 | 2024-06-14 13:15:00 | 260.81 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-06-13 15:15:00 | 263.40 | 2024-06-14 13:15:00 | 260.81 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-06-14 09:15:00 | 263.51 | 2024-06-14 13:15:00 | 260.81 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-06-24 13:30:00 | 258.11 | 2024-06-27 09:15:00 | 259.81 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-06-25 09:30:00 | 258.19 | 2024-06-27 09:15:00 | 259.81 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-07-05 13:15:00 | 235.05 | 2024-07-09 14:15:00 | 223.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-08 14:30:00 | 233.71 | 2024-07-09 14:15:00 | 222.87 | PARTIAL | 0.50 | 4.64% |
| SELL | retest2 | 2024-07-09 10:15:00 | 234.60 | 2024-07-10 09:15:00 | 222.02 | PARTIAL | 0.50 | 5.36% |
| SELL | retest2 | 2024-07-05 13:15:00 | 235.05 | 2024-07-12 09:15:00 | 224.30 | STOP_HIT | 0.50 | 4.57% |
| SELL | retest2 | 2024-07-08 14:30:00 | 233.71 | 2024-07-12 09:15:00 | 224.30 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2024-07-09 10:15:00 | 234.60 | 2024-07-12 09:15:00 | 224.30 | STOP_HIT | 0.50 | 4.39% |
| SELL | retest2 | 2024-07-23 09:30:00 | 208.63 | 2024-07-23 12:15:00 | 225.00 | STOP_HIT | 1.00 | -7.85% |
| SELL | retest2 | 2024-07-23 10:15:00 | 208.98 | 2024-07-23 12:15:00 | 225.00 | STOP_HIT | 1.00 | -7.67% |
| SELL | retest2 | 2024-07-23 11:15:00 | 209.15 | 2024-07-23 12:15:00 | 225.00 | STOP_HIT | 1.00 | -7.58% |
| SELL | retest2 | 2024-07-26 13:30:00 | 214.00 | 2024-07-31 09:15:00 | 222.22 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2024-07-29 09:30:00 | 214.47 | 2024-07-31 09:15:00 | 222.22 | STOP_HIT | 1.00 | -3.61% |
| SELL | retest2 | 2024-07-29 10:15:00 | 214.49 | 2024-07-31 09:15:00 | 222.22 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2024-07-29 11:15:00 | 214.00 | 2024-07-31 09:15:00 | 222.22 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2024-07-29 13:45:00 | 213.45 | 2024-07-31 09:15:00 | 222.22 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2024-07-30 09:30:00 | 213.58 | 2024-07-31 09:15:00 | 222.22 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2024-07-30 12:30:00 | 213.57 | 2024-07-31 09:15:00 | 222.22 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2024-07-30 15:00:00 | 213.50 | 2024-07-31 09:15:00 | 222.22 | STOP_HIT | 1.00 | -4.08% |
| SELL | retest2 | 2024-08-14 13:45:00 | 210.48 | 2024-08-16 11:15:00 | 212.89 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-08-14 14:30:00 | 210.40 | 2024-08-16 11:15:00 | 212.89 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-08-23 11:30:00 | 258.60 | 2024-08-28 11:15:00 | 260.16 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2024-08-23 12:00:00 | 257.90 | 2024-08-28 11:15:00 | 260.16 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest2 | 2024-08-26 09:15:00 | 268.00 | 2024-08-28 11:15:00 | 260.16 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2024-08-30 11:15:00 | 259.01 | 2024-09-06 09:15:00 | 246.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-02 10:00:00 | 259.20 | 2024-09-06 09:15:00 | 246.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-03 10:00:00 | 258.83 | 2024-09-06 09:15:00 | 245.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-30 11:15:00 | 259.01 | 2024-09-09 09:15:00 | 233.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-02 10:00:00 | 259.20 | 2024-09-09 09:15:00 | 233.28 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-03 10:00:00 | 258.83 | 2024-09-09 09:15:00 | 232.95 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2024-09-23 09:15:00 | 263.50 | 2024-09-24 11:15:00 | 259.50 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-09-24 13:15:00 | 260.42 | 2024-09-25 09:15:00 | 253.94 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2024-09-24 15:00:00 | 260.30 | 2024-09-25 09:15:00 | 253.94 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2024-10-03 10:15:00 | 277.40 | 2024-10-07 09:15:00 | 257.83 | STOP_HIT | 1.00 | -7.05% |
| BUY | retest2 | 2024-10-03 11:00:00 | 277.51 | 2024-10-07 09:15:00 | 257.83 | STOP_HIT | 1.00 | -7.09% |
| BUY | retest2 | 2024-10-03 11:45:00 | 277.40 | 2024-10-07 09:15:00 | 257.83 | STOP_HIT | 1.00 | -7.05% |
| SELL | retest2 | 2024-10-10 12:00:00 | 256.97 | 2024-10-11 09:15:00 | 264.40 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2024-10-15 09:15:00 | 290.40 | 2024-10-15 13:15:00 | 319.44 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-23 12:15:00 | 297.60 | 2024-10-25 09:15:00 | 282.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 14:00:00 | 297.98 | 2024-10-25 09:15:00 | 283.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 14:30:00 | 296.10 | 2024-10-25 09:15:00 | 281.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 15:00:00 | 295.27 | 2024-10-25 09:15:00 | 280.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 12:15:00 | 297.60 | 2024-10-28 09:15:00 | 284.01 | STOP_HIT | 0.50 | 4.57% |
| SELL | retest2 | 2024-10-23 14:00:00 | 297.98 | 2024-10-28 09:15:00 | 284.01 | STOP_HIT | 0.50 | 4.69% |
| SELL | retest2 | 2024-10-23 14:30:00 | 296.10 | 2024-10-28 09:15:00 | 284.01 | STOP_HIT | 0.50 | 4.08% |
| SELL | retest2 | 2024-10-23 15:00:00 | 295.27 | 2024-10-28 09:15:00 | 284.01 | STOP_HIT | 0.50 | 3.81% |
| SELL | retest2 | 2024-10-24 11:00:00 | 289.70 | 2024-10-29 14:15:00 | 297.00 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2024-10-24 11:45:00 | 289.64 | 2024-10-29 14:15:00 | 297.00 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2024-10-24 12:45:00 | 288.90 | 2024-10-29 14:15:00 | 297.00 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2024-10-28 13:30:00 | 289.33 | 2024-10-29 14:15:00 | 297.00 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2024-11-08 11:15:00 | 290.50 | 2024-11-12 13:15:00 | 275.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 11:15:00 | 290.50 | 2024-11-13 14:15:00 | 261.45 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-11-29 09:15:00 | 293.55 | 2024-12-05 13:15:00 | 322.91 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-02 09:45:00 | 292.15 | 2024-12-05 13:15:00 | 321.37 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-26 14:30:00 | 287.44 | 2024-12-30 15:15:00 | 292.70 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-12-27 12:15:00 | 288.02 | 2024-12-30 15:15:00 | 292.70 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-12-27 12:45:00 | 286.81 | 2024-12-30 15:15:00 | 292.70 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-12-27 15:00:00 | 288.26 | 2024-12-30 15:15:00 | 292.70 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-01-06 10:30:00 | 286.22 | 2025-01-07 09:15:00 | 271.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 10:30:00 | 286.22 | 2025-01-07 14:15:00 | 276.46 | STOP_HIT | 0.50 | 3.41% |
| BUY | retest2 | 2025-01-21 15:00:00 | 252.89 | 2025-01-22 09:15:00 | 242.83 | STOP_HIT | 1.00 | -3.98% |
| SELL | retest2 | 2025-01-23 15:15:00 | 242.10 | 2025-01-27 09:15:00 | 229.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 15:15:00 | 242.10 | 2025-01-28 09:15:00 | 217.89 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-01 14:45:00 | 244.38 | 2025-02-04 10:15:00 | 231.90 | STOP_HIT | 1.00 | -5.11% |
| BUY | retest2 | 2025-02-07 11:00:00 | 246.75 | 2025-02-07 15:15:00 | 241.20 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-03-03 09:15:00 | 209.78 | 2025-03-03 10:15:00 | 199.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-03 09:15:00 | 209.78 | 2025-03-04 09:15:00 | 203.30 | STOP_HIT | 0.50 | 3.09% |
| BUY | retest2 | 2025-03-28 09:15:00 | 238.53 | 2025-03-28 15:15:00 | 230.80 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2025-03-28 14:00:00 | 232.21 | 2025-03-28 15:15:00 | 230.80 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-04-01 09:15:00 | 232.37 | 2025-04-01 14:15:00 | 230.13 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-04-01 15:15:00 | 232.20 | 2025-04-04 12:15:00 | 231.17 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-04-02 10:45:00 | 235.47 | 2025-04-04 12:15:00 | 231.17 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-04-02 14:45:00 | 235.17 | 2025-04-04 12:15:00 | 231.17 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-04-02 15:15:00 | 235.50 | 2025-04-04 12:15:00 | 231.17 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-04-09 11:00:00 | 216.45 | 2025-04-11 09:15:00 | 227.89 | STOP_HIT | 1.00 | -5.29% |
| BUY | retest2 | 2025-04-17 11:00:00 | 229.71 | 2025-04-22 09:15:00 | 252.68 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-07 13:00:00 | 233.72 | 2025-05-07 14:15:00 | 236.23 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-05-21 09:45:00 | 280.84 | 2025-05-23 13:15:00 | 308.92 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-02 11:15:00 | 306.80 | 2025-06-10 10:15:00 | 315.41 | STOP_HIT | 1.00 | 2.81% |
| SELL | retest2 | 2025-07-10 11:45:00 | 272.25 | 2025-07-16 12:15:00 | 272.20 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2025-07-10 14:00:00 | 272.66 | 2025-07-16 12:15:00 | 272.20 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2025-07-10 15:15:00 | 272.50 | 2025-07-16 12:15:00 | 272.20 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-07-11 09:45:00 | 272.30 | 2025-07-16 12:15:00 | 272.20 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-07-14 10:45:00 | 268.49 | 2025-07-16 12:15:00 | 272.20 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-07-15 09:30:00 | 264.28 | 2025-07-16 12:15:00 | 272.20 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-07-15 13:45:00 | 268.18 | 2025-07-16 12:15:00 | 272.20 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-07-16 11:00:00 | 267.80 | 2025-07-16 12:15:00 | 272.20 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-08-07 09:15:00 | 260.00 | 2025-08-07 14:15:00 | 263.55 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-08-07 10:15:00 | 258.22 | 2025-08-07 14:15:00 | 263.55 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-08-07 12:30:00 | 260.07 | 2025-08-07 14:15:00 | 263.55 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-09-02 13:00:00 | 229.51 | 2025-09-05 14:15:00 | 229.31 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-09-04 10:00:00 | 229.35 | 2025-09-05 14:15:00 | 229.31 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest1 | 2025-09-10 09:15:00 | 235.50 | 2025-09-10 11:15:00 | 231.39 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-09-19 13:45:00 | 227.31 | 2025-09-22 12:15:00 | 224.10 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-09-26 09:15:00 | 217.66 | 2025-10-01 13:15:00 | 217.27 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-09-26 10:00:00 | 217.18 | 2025-10-01 13:15:00 | 217.27 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-10-08 13:15:00 | 222.97 | 2025-10-08 14:15:00 | 221.20 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-10-14 09:15:00 | 242.05 | 2025-10-29 09:15:00 | 247.80 | STOP_HIT | 1.00 | 2.38% |
| BUY | retest2 | 2025-10-14 12:30:00 | 235.74 | 2025-10-29 09:15:00 | 247.80 | STOP_HIT | 1.00 | 5.12% |
| BUY | retest2 | 2025-11-19 12:00:00 | 282.50 | 2025-11-21 11:15:00 | 278.05 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-11-20 09:45:00 | 282.99 | 2025-11-21 11:15:00 | 278.05 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-11-20 11:15:00 | 282.95 | 2025-11-21 11:15:00 | 278.05 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-12-18 11:30:00 | 250.35 | 2025-12-22 10:15:00 | 253.29 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-12-24 10:30:00 | 257.20 | 2025-12-24 13:15:00 | 253.48 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-12-24 11:30:00 | 257.20 | 2025-12-24 13:15:00 | 253.48 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-01-02 10:15:00 | 236.03 | 2026-01-02 14:15:00 | 238.45 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-01-02 11:15:00 | 236.00 | 2026-01-02 14:15:00 | 238.45 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2026-01-07 12:15:00 | 244.97 | 2026-01-09 09:15:00 | 237.33 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2026-01-07 12:45:00 | 246.13 | 2026-01-09 09:15:00 | 237.33 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2026-01-22 11:15:00 | 255.71 | 2026-01-28 09:15:00 | 260.46 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2026-01-22 12:30:00 | 255.43 | 2026-01-28 09:15:00 | 260.46 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-01-23 11:45:00 | 255.61 | 2026-01-28 09:15:00 | 260.46 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-01-29 14:30:00 | 258.69 | 2026-01-30 13:15:00 | 255.98 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-01-29 15:00:00 | 258.78 | 2026-01-30 13:15:00 | 255.98 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-01-30 11:30:00 | 258.69 | 2026-01-30 13:15:00 | 255.98 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-01-30 12:00:00 | 259.76 | 2026-01-30 13:15:00 | 255.98 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-02-19 10:30:00 | 255.56 | 2026-02-25 11:15:00 | 242.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 10:30:00 | 255.56 | 2026-02-25 14:15:00 | 248.76 | STOP_HIT | 0.50 | 2.66% |
| SELL | retest2 | 2026-03-16 10:15:00 | 213.41 | 2026-03-17 09:15:00 | 220.11 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest1 | 2026-03-20 09:15:00 | 236.01 | 2026-03-20 15:15:00 | 230.20 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest1 | 2026-03-20 10:30:00 | 234.80 | 2026-03-20 15:15:00 | 230.20 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2026-04-06 10:30:00 | 244.75 | 2026-04-09 09:15:00 | 269.23 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 13:45:00 | 244.95 | 2026-04-09 09:15:00 | 269.44 | TARGET_HIT | 1.00 | 10.00% |
