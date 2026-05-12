# BSE Ltd. (BSE)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 3905.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 186 |
| ALERT1 | 145 |
| ALERT2 | 142 |
| ALERT2_SKIP | 73 |
| ALERT3 | 389 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 175 |
| PARTIAL | 23 |
| TARGET_HIT | 16 |
| STOP_HIT | 161 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 197 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 82 / 115
- **Target hits / Stop hits / Partials:** 16 / 158 / 23
- **Avg / median % per leg:** 0.80% / -0.47%
- **Sum % (uncompounded):** 157.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 89 | 31 | 34.8% | 13 | 76 | 0 | 0.62% | 55.0% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.06% | -1.1% |
| BUY @ 3rd Alert (retest2) | 88 | 31 | 35.2% | 13 | 75 | 0 | 0.64% | 56.1% |
| SELL (all) | 108 | 51 | 47.2% | 3 | 82 | 23 | 0.94% | 102.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 108 | 51 | 47.2% | 3 | 82 | 23 | 0.94% | 102.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.06% | -1.1% |
| retest2 (combined) | 196 | 82 | 41.8% | 16 | 157 | 23 | 0.81% | 158.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 15:15:00 | 179.43 | 181.37 | 181.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 11:15:00 | 177.82 | 180.20 | 180.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-18 09:15:00 | 179.55 | 179.31 | 180.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 09:15:00 | 179.55 | 179.31 | 180.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 179.55 | 179.31 | 180.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-18 09:45:00 | 180.00 | 179.31 | 180.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 09:15:00 | 178.73 | 176.20 | 176.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 09:45:00 | 179.60 | 176.20 | 176.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 10:15:00 | 179.10 | 176.78 | 176.74 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 15:15:00 | 176.67 | 177.21 | 177.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-25 09:15:00 | 176.20 | 177.01 | 177.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 11:15:00 | 177.55 | 177.00 | 177.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 11:15:00 | 177.55 | 177.00 | 177.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 11:15:00 | 177.55 | 177.00 | 177.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 11:45:00 | 178.80 | 177.00 | 177.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 12:15:00 | 177.23 | 177.05 | 177.12 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-05-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 14:15:00 | 178.00 | 177.29 | 177.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 10:15:00 | 179.17 | 177.92 | 177.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 09:15:00 | 180.00 | 180.06 | 179.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 10:15:00 | 179.22 | 179.89 | 179.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 10:15:00 | 179.22 | 179.89 | 179.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 10:30:00 | 179.45 | 179.89 | 179.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 11:15:00 | 179.35 | 179.78 | 179.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 14:00:00 | 179.75 | 179.71 | 179.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 14:30:00 | 179.90 | 179.85 | 179.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-06 12:15:00 | 187.67 | 189.39 | 189.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 12:15:00 | 187.67 | 189.39 | 189.52 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 11:15:00 | 189.87 | 189.51 | 189.47 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-07 14:15:00 | 188.63 | 189.42 | 189.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-08 09:15:00 | 187.95 | 188.96 | 189.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-08 10:15:00 | 189.47 | 189.06 | 189.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 10:15:00 | 189.47 | 189.06 | 189.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 10:15:00 | 189.47 | 189.06 | 189.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-08 11:00:00 | 189.47 | 189.06 | 189.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 186.67 | 188.58 | 189.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-09 09:15:00 | 185.45 | 187.48 | 188.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-12 11:15:00 | 190.32 | 187.00 | 187.08 | SL hit (close>static) qty=1.00 sl=189.57 alert=retest2 |

### Cycle 8 — BUY (started 2023-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 12:15:00 | 190.40 | 187.68 | 187.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-15 09:15:00 | 192.97 | 190.01 | 189.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 15:15:00 | 191.17 | 191.23 | 190.26 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 09:15:00 | 192.20 | 191.23 | 190.26 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 13:15:00 | 190.17 | 191.28 | 190.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-06-16 13:15:00 | 190.17 | 191.28 | 190.70 | SL hit (close<ema400) qty=1.00 sl=190.70 alert=retest1 |

### Cycle 9 — SELL (started 2023-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 13:15:00 | 189.37 | 190.45 | 190.54 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 09:15:00 | 193.15 | 190.77 | 190.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-22 09:15:00 | 199.93 | 192.98 | 191.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-26 09:15:00 | 197.17 | 201.53 | 199.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 09:15:00 | 197.17 | 201.53 | 199.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 197.17 | 201.53 | 199.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-26 10:00:00 | 197.17 | 201.53 | 199.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 198.37 | 200.90 | 199.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-26 11:45:00 | 199.20 | 200.49 | 199.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-26 12:15:00 | 199.23 | 200.49 | 199.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-26 14:45:00 | 199.47 | 199.45 | 198.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-26 15:15:00 | 199.05 | 199.45 | 198.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 15:15:00 | 199.05 | 199.37 | 198.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:15:00 | 198.78 | 199.37 | 198.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 200.33 | 199.56 | 199.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-27 12:15:00 | 200.57 | 199.71 | 199.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-27 14:00:00 | 200.80 | 199.95 | 199.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-28 09:15:00 | 205.92 | 200.08 | 199.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-03 13:15:00 | 219.12 | 211.06 | 207.08 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 11:15:00 | 245.32 | 248.66 | 248.73 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 11:15:00 | 247.65 | 246.57 | 246.47 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 14:15:00 | 245.62 | 246.31 | 246.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 10:15:00 | 244.82 | 245.93 | 246.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 12:15:00 | 245.67 | 245.63 | 245.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-24 13:00:00 | 245.67 | 245.63 | 245.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 245.97 | 244.78 | 245.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 10:00:00 | 245.97 | 244.78 | 245.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 247.22 | 245.27 | 245.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 11:00:00 | 247.22 | 245.27 | 245.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 11:15:00 | 246.18 | 245.45 | 245.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-25 12:15:00 | 245.60 | 245.45 | 245.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-25 15:15:00 | 246.10 | 245.64 | 245.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2023-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 15:15:00 | 246.10 | 245.64 | 245.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-26 09:15:00 | 256.83 | 247.88 | 246.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-28 09:15:00 | 258.90 | 259.70 | 256.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-28 09:15:00 | 258.90 | 259.70 | 256.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 09:15:00 | 258.90 | 259.70 | 256.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-28 12:00:00 | 263.27 | 260.64 | 257.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-08-02 10:15:00 | 289.60 | 279.55 | 273.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 09:15:00 | 293.95 | 298.56 | 298.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-16 09:15:00 | 286.98 | 293.11 | 295.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 09:15:00 | 288.30 | 284.89 | 289.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 09:15:00 | 288.30 | 284.89 | 289.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 288.30 | 284.89 | 289.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 09:30:00 | 288.52 | 284.89 | 289.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 10:15:00 | 288.62 | 285.64 | 289.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 13:45:00 | 287.10 | 286.85 | 288.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 10:00:00 | 287.00 | 286.85 | 288.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 10:30:00 | 288.00 | 286.78 | 288.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 15:00:00 | 287.83 | 286.69 | 287.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 15:15:00 | 287.82 | 286.92 | 287.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 09:15:00 | 288.33 | 286.92 | 287.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 289.87 | 287.51 | 287.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 10:15:00 | 291.67 | 287.51 | 287.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-08-21 10:15:00 | 291.68 | 288.34 | 288.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2023-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 10:15:00 | 291.68 | 288.34 | 288.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 14:15:00 | 296.72 | 291.44 | 289.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 12:15:00 | 305.48 | 305.89 | 302.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-24 12:30:00 | 305.23 | 305.89 | 302.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 13:15:00 | 302.75 | 305.26 | 302.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 14:00:00 | 302.75 | 305.26 | 302.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 299.33 | 303.82 | 302.12 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 12:15:00 | 300.33 | 301.05 | 301.14 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 12:15:00 | 302.92 | 301.40 | 301.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 09:15:00 | 306.28 | 303.31 | 302.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 12:15:00 | 390.33 | 391.53 | 378.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-05 12:30:00 | 390.07 | 391.53 | 378.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 09:15:00 | 384.28 | 391.07 | 382.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 14:15:00 | 406.18 | 388.07 | 383.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-08 15:15:00 | 446.80 | 431.94 | 418.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2023-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-13 09:15:00 | 419.05 | 434.15 | 434.27 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-09-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 12:15:00 | 434.22 | 432.00 | 431.94 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-09-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-14 13:15:00 | 430.85 | 431.77 | 431.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-14 14:15:00 | 428.70 | 431.16 | 431.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-15 09:15:00 | 431.68 | 430.44 | 431.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 09:15:00 | 431.68 | 430.44 | 431.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 09:15:00 | 431.68 | 430.44 | 431.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 10:45:00 | 428.85 | 430.27 | 430.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 11:30:00 | 428.90 | 430.25 | 430.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 12:30:00 | 429.00 | 430.75 | 431.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 15:00:00 | 426.17 | 429.92 | 430.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 428.98 | 428.41 | 429.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-18 15:15:00 | 413.33 | 423.46 | 426.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 09:15:00 | 407.41 | 417.68 | 423.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 09:15:00 | 407.45 | 417.68 | 423.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 09:15:00 | 407.55 | 417.68 | 423.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 09:15:00 | 404.86 | 417.68 | 423.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 10:15:00 | 392.66 | 414.41 | 421.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2023-09-22 09:15:00 | 385.97 | 394.36 | 402.72 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 22 — BUY (started 2023-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 09:15:00 | 416.07 | 402.63 | 401.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 10:15:00 | 417.45 | 405.59 | 402.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 14:15:00 | 430.60 | 437.67 | 430.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 14:15:00 | 430.60 | 437.67 | 430.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 430.60 | 437.67 | 430.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 15:00:00 | 430.60 | 437.67 | 430.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 432.65 | 436.66 | 430.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 09:15:00 | 439.97 | 436.66 | 430.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-29 10:15:00 | 428.03 | 434.19 | 430.43 | SL hit (close<static) qty=1.00 sl=430.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 12:15:00 | 424.40 | 431.76 | 431.82 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-10-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 12:15:00 | 432.92 | 431.64 | 431.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 14:15:00 | 441.00 | 433.75 | 432.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 14:15:00 | 460.55 | 464.29 | 456.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-09 15:00:00 | 460.55 | 464.29 | 456.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 11:15:00 | 484.27 | 490.64 | 483.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 12:00:00 | 484.27 | 490.64 | 483.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 12:15:00 | 485.52 | 489.61 | 483.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 12:45:00 | 484.35 | 489.61 | 483.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 13:15:00 | 483.67 | 488.43 | 483.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 14:00:00 | 483.67 | 488.43 | 483.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 14:15:00 | 490.33 | 488.81 | 484.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 15:15:00 | 492.67 | 488.81 | 484.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 09:45:00 | 491.83 | 490.02 | 485.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 10:30:00 | 492.02 | 490.41 | 486.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 11:00:00 | 492.47 | 495.01 | 492.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 11:15:00 | 491.98 | 494.40 | 492.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-17 12:00:00 | 491.98 | 494.40 | 492.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 12:15:00 | 492.07 | 493.94 | 492.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-17 13:15:00 | 491.68 | 493.94 | 492.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 14:15:00 | 494.18 | 493.36 | 492.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-10-18 12:15:00 | 491.70 | 492.41 | 492.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2023-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 12:15:00 | 491.70 | 492.41 | 492.48 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 13:15:00 | 493.32 | 492.59 | 492.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-18 14:15:00 | 494.98 | 493.07 | 492.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-19 09:15:00 | 493.12 | 493.67 | 493.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 09:15:00 | 493.12 | 493.67 | 493.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 09:15:00 | 493.12 | 493.67 | 493.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-19 09:30:00 | 492.00 | 493.67 | 493.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 10:15:00 | 498.93 | 494.72 | 493.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-19 14:15:00 | 504.75 | 496.90 | 495.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-10-23 09:15:00 | 555.23 | 531.67 | 517.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2023-11-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 12:15:00 | 606.00 | 618.50 | 619.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 13:15:00 | 601.98 | 615.19 | 617.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 09:15:00 | 609.67 | 609.58 | 614.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 09:15:00 | 609.67 | 609.58 | 614.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 609.67 | 609.58 | 614.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 09:45:00 | 616.97 | 609.58 | 614.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 10:15:00 | 613.05 | 610.28 | 614.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 10:45:00 | 611.03 | 610.28 | 614.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 599.17 | 608.05 | 612.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-02 14:00:00 | 586.57 | 602.44 | 609.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-06 11:00:00 | 594.33 | 598.25 | 601.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-06 12:00:00 | 594.95 | 597.59 | 601.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-06 14:15:00 | 622.20 | 606.99 | 605.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2023-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 14:15:00 | 622.20 | 606.99 | 605.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 15:15:00 | 623.20 | 610.23 | 606.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 11:15:00 | 674.77 | 677.21 | 661.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-09 11:45:00 | 674.33 | 677.21 | 661.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 13:15:00 | 660.30 | 673.58 | 662.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 14:00:00 | 660.30 | 673.58 | 662.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 654.38 | 669.74 | 662.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 15:00:00 | 654.38 | 669.74 | 662.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 14:15:00 | 666.10 | 672.31 | 667.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 15:00:00 | 666.10 | 672.31 | 667.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 15:15:00 | 676.33 | 673.12 | 668.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-12 18:15:00 | 705.03 | 673.12 | 668.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-11-13 14:15:00 | 775.53 | 738.86 | 709.11 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2023-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 14:15:00 | 755.25 | 778.59 | 778.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-21 09:15:00 | 728.33 | 765.33 | 772.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 12:15:00 | 712.17 | 708.71 | 722.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-23 12:45:00 | 715.02 | 708.71 | 722.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 13:15:00 | 712.75 | 705.77 | 713.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 14:00:00 | 712.75 | 705.77 | 713.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 14:15:00 | 722.68 | 709.15 | 713.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 15:00:00 | 722.68 | 709.15 | 713.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 15:15:00 | 726.67 | 712.65 | 715.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 09:15:00 | 756.10 | 712.65 | 715.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 09:15:00 | 769.00 | 723.92 | 720.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 13:15:00 | 774.87 | 750.68 | 735.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 13:15:00 | 832.08 | 835.49 | 817.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-01 14:00:00 | 832.08 | 835.49 | 817.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 830.78 | 835.18 | 828.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 09:30:00 | 830.02 | 835.18 | 828.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 10:15:00 | 812.67 | 830.68 | 827.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 11:00:00 | 812.67 | 830.68 | 827.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 11:15:00 | 814.65 | 827.47 | 825.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 11:45:00 | 811.02 | 827.47 | 825.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2023-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 13:15:00 | 817.87 | 824.21 | 824.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-05 14:15:00 | 814.68 | 822.30 | 823.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-06 10:15:00 | 822.98 | 821.38 | 822.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-06 10:15:00 | 822.98 | 821.38 | 822.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 10:15:00 | 822.98 | 821.38 | 822.85 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 12:15:00 | 836.53 | 825.64 | 824.61 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-12-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 15:15:00 | 820.67 | 825.05 | 825.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 10:15:00 | 816.80 | 822.70 | 824.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 09:15:00 | 791.63 | 779.71 | 789.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 09:15:00 | 791.63 | 779.71 | 789.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 791.63 | 779.71 | 789.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 09:30:00 | 799.00 | 779.71 | 789.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 10:15:00 | 778.93 | 779.55 | 788.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 10:30:00 | 788.65 | 779.55 | 788.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 12:15:00 | 791.03 | 782.51 | 788.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 12:45:00 | 791.67 | 782.51 | 788.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 13:15:00 | 790.82 | 784.17 | 788.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 13:30:00 | 792.80 | 784.17 | 788.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 15:15:00 | 793.02 | 785.99 | 788.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:15:00 | 800.62 | 785.99 | 788.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 794.67 | 787.73 | 789.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-14 10:15:00 | 790.68 | 787.73 | 789.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-14 11:30:00 | 792.00 | 789.76 | 790.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-14 12:45:00 | 792.45 | 790.07 | 790.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-18 09:45:00 | 791.33 | 779.39 | 782.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 10:15:00 | 786.93 | 780.90 | 783.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-18 11:00:00 | 786.93 | 780.90 | 783.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-12-18 13:15:00 | 790.00 | 785.33 | 784.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2023-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 13:15:00 | 790.00 | 785.33 | 784.86 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 11:15:00 | 781.03 | 784.73 | 784.96 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 14:15:00 | 800.85 | 787.05 | 785.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-19 15:15:00 | 803.67 | 790.37 | 787.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 13:15:00 | 790.00 | 799.79 | 794.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 13:15:00 | 790.00 | 799.79 | 794.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 790.00 | 799.79 | 794.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:45:00 | 787.67 | 799.79 | 794.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 14:15:00 | 782.08 | 796.25 | 793.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 15:15:00 | 765.00 | 796.25 | 793.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2023-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 15:15:00 | 765.00 | 790.00 | 790.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-27 12:15:00 | 754.67 | 766.16 | 772.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-02 09:15:00 | 746.67 | 737.56 | 742.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-02 09:15:00 | 746.67 | 737.56 | 742.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 09:15:00 | 746.67 | 737.56 | 742.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-02 10:30:00 | 734.38 | 738.04 | 741.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-03 10:15:00 | 754.67 | 745.10 | 743.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2024-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 10:15:00 | 754.67 | 745.10 | 743.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 11:15:00 | 759.00 | 753.26 | 749.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 10:15:00 | 762.33 | 763.16 | 759.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-08 11:00:00 | 762.33 | 763.16 | 759.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 11:15:00 | 758.02 | 762.14 | 758.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:45:00 | 756.77 | 762.14 | 758.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 12:15:00 | 757.53 | 761.21 | 758.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 12:30:00 | 759.38 | 761.21 | 758.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 13:15:00 | 760.32 | 761.04 | 758.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 09:15:00 | 766.65 | 759.84 | 758.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 10:15:00 | 761.67 | 759.85 | 758.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 11:15:00 | 760.82 | 762.42 | 761.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-10 11:15:00 | 756.65 | 761.27 | 760.85 | SL hit (close<static) qty=1.00 sl=757.05 alert=retest2 |

### Cycle 39 — SELL (started 2024-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 12:15:00 | 754.00 | 759.82 | 760.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 13:15:00 | 749.85 | 757.82 | 759.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 09:15:00 | 756.67 | 756.27 | 758.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 09:15:00 | 756.67 | 756.27 | 758.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 756.67 | 756.27 | 758.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 09:45:00 | 756.67 | 756.27 | 758.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 745.70 | 751.94 | 754.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-12 11:15:00 | 744.47 | 750.71 | 754.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-12 15:15:00 | 744.33 | 748.21 | 751.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-15 13:30:00 | 744.72 | 748.07 | 750.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-16 09:15:00 | 766.27 | 751.60 | 751.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2024-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 09:15:00 | 766.27 | 751.60 | 751.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 15:15:00 | 773.33 | 760.57 | 756.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 11:15:00 | 762.33 | 763.23 | 758.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-17 12:00:00 | 762.33 | 763.23 | 758.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 14:15:00 | 763.25 | 763.79 | 760.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 15:00:00 | 763.25 | 763.79 | 760.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 764.37 | 763.72 | 760.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 09:45:00 | 756.65 | 763.72 | 760.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 13:15:00 | 765.78 | 764.57 | 762.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 09:15:00 | 771.33 | 765.01 | 762.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-19 10:15:00 | 757.00 | 763.51 | 762.49 | SL hit (close<static) qty=1.00 sl=762.02 alert=retest2 |

### Cycle 41 — SELL (started 2024-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-19 11:15:00 | 749.83 | 760.77 | 761.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-20 14:15:00 | 745.67 | 752.83 | 756.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 14:15:00 | 702.33 | 694.78 | 713.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 14:15:00 | 702.33 | 694.78 | 713.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 702.33 | 694.78 | 713.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 15:00:00 | 702.33 | 694.78 | 713.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 706.60 | 697.97 | 711.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 10:00:00 | 706.60 | 697.97 | 711.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 15:15:00 | 714.00 | 705.69 | 710.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 09:15:00 | 711.38 | 705.69 | 710.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 717.33 | 708.01 | 710.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 10:15:00 | 716.67 | 708.01 | 710.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 12:15:00 | 712.48 | 709.35 | 710.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 12:30:00 | 713.33 | 709.35 | 710.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 13:15:00 | 709.68 | 709.42 | 710.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 13:30:00 | 712.52 | 709.42 | 710.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 14:15:00 | 709.03 | 709.34 | 710.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 15:00:00 | 709.03 | 709.34 | 710.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 15:15:00 | 709.98 | 709.47 | 710.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-30 09:15:00 | 720.02 | 709.47 | 710.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2024-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 09:15:00 | 734.35 | 714.44 | 712.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 10:15:00 | 743.73 | 731.56 | 723.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-06 09:15:00 | 821.72 | 842.55 | 826.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 09:15:00 | 821.72 | 842.55 | 826.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 821.72 | 842.55 | 826.64 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 12:15:00 | 824.37 | 829.50 | 829.68 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2024-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 14:15:00 | 830.53 | 829.82 | 829.80 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 15:15:00 | 826.67 | 829.19 | 829.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 09:15:00 | 803.00 | 823.96 | 827.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 15:15:00 | 783.33 | 781.97 | 790.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-15 09:15:00 | 789.10 | 781.97 | 790.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 09:15:00 | 784.00 | 782.37 | 789.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-15 10:45:00 | 779.52 | 781.76 | 788.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-19 13:15:00 | 778.67 | 775.81 | 777.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-21 14:15:00 | 740.54 | 755.73 | 763.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-21 15:15:00 | 739.74 | 751.10 | 760.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-23 09:15:00 | 745.40 | 738.79 | 747.09 | SL hit (close>ema200) qty=0.50 sl=738.79 alert=retest2 |

### Cycle 46 — BUY (started 2024-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 14:15:00 | 760.00 | 750.51 | 749.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 12:15:00 | 791.00 | 768.18 | 760.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 10:15:00 | 781.78 | 783.64 | 772.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-28 11:00:00 | 781.78 | 783.64 | 772.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 11:15:00 | 771.30 | 781.17 | 772.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 12:00:00 | 771.30 | 781.17 | 772.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 12:15:00 | 761.87 | 777.31 | 771.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 13:00:00 | 761.87 | 777.31 | 771.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 13:15:00 | 775.83 | 777.01 | 771.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-28 15:00:00 | 783.33 | 778.28 | 773.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-29 12:30:00 | 779.73 | 777.29 | 774.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-04 10:00:00 | 777.40 | 781.88 | 780.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-04 11:30:00 | 776.02 | 779.62 | 779.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-04 12:15:00 | 771.32 | 777.96 | 778.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2024-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 12:15:00 | 771.32 | 777.96 | 778.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 09:15:00 | 761.60 | 772.55 | 775.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 13:15:00 | 764.10 | 759.49 | 764.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 13:15:00 | 764.10 | 759.49 | 764.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 13:15:00 | 764.10 | 759.49 | 764.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 13:45:00 | 764.67 | 759.49 | 764.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 766.63 | 760.92 | 765.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 14:30:00 | 769.67 | 760.92 | 765.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 15:15:00 | 764.50 | 761.63 | 765.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 09:15:00 | 761.22 | 761.63 | 765.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 09:45:00 | 761.18 | 761.71 | 764.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 723.16 | 737.10 | 745.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 723.12 | 737.10 | 745.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-12 15:15:00 | 728.33 | 720.58 | 732.11 | SL hit (close>ema200) qty=0.50 sl=720.58 alert=retest2 |

### Cycle 48 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 718.93 | 681.83 | 677.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 12:15:00 | 732.82 | 704.69 | 690.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 732.80 | 733.40 | 718.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-22 14:30:00 | 733.67 | 733.40 | 718.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 13:15:00 | 935.88 | 953.33 | 949.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 13:30:00 | 936.25 | 953.33 | 949.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 14:15:00 | 931.33 | 948.93 | 947.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 14:30:00 | 926.90 | 948.93 | 947.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2024-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 15:15:00 | 933.67 | 945.88 | 946.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 09:15:00 | 926.32 | 941.97 | 944.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 09:15:00 | 938.58 | 926.51 | 933.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 09:15:00 | 938.58 | 926.51 | 933.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 938.58 | 926.51 | 933.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 09:45:00 | 937.18 | 926.51 | 933.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 10:15:00 | 957.07 | 932.62 | 935.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 11:00:00 | 957.07 | 932.62 | 935.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2024-04-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 12:15:00 | 954.98 | 940.41 | 938.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 13:15:00 | 959.10 | 944.14 | 940.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 12:15:00 | 946.50 | 954.15 | 948.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 12:15:00 | 946.50 | 954.15 | 948.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 12:15:00 | 946.50 | 954.15 | 948.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 12:45:00 | 947.22 | 954.15 | 948.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 13:15:00 | 949.33 | 953.18 | 948.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 14:15:00 | 943.17 | 953.18 | 948.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 14:15:00 | 941.53 | 950.85 | 947.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 14:30:00 | 941.77 | 950.85 | 947.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 15:15:00 | 945.33 | 949.75 | 947.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 09:15:00 | 934.97 | 949.75 | 947.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 10:15:00 | 943.70 | 948.15 | 947.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 11:30:00 | 953.60 | 948.78 | 947.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 13:00:00 | 951.05 | 949.23 | 947.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 13:15:00 | 939.82 | 947.35 | 947.14 | SL hit (close<static) qty=1.00 sl=940.33 alert=retest2 |

### Cycle 51 — SELL (started 2024-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 14:15:00 | 923.33 | 942.55 | 944.97 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-04-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 11:15:00 | 947.70 | 944.82 | 944.77 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-04-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 13:15:00 | 938.07 | 944.22 | 944.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 14:15:00 | 925.00 | 940.37 | 942.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-22 09:15:00 | 944.58 | 931.05 | 934.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-22 09:15:00 | 944.58 | 931.05 | 934.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 944.58 | 931.05 | 934.68 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 11:15:00 | 953.33 | 938.32 | 937.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 13:15:00 | 953.97 | 943.74 | 940.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 15:15:00 | 1048.33 | 1048.84 | 1032.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 09:15:00 | 1048.30 | 1048.84 | 1032.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 55 — SELL (started 2024-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 09:15:00 | 899.07 | 1027.73 | 1032.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 09:15:00 | 861.63 | 886.58 | 903.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 09:15:00 | 883.67 | 875.41 | 887.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 09:15:00 | 883.67 | 875.41 | 887.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 883.67 | 875.41 | 887.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:45:00 | 883.33 | 875.41 | 887.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 887.50 | 877.83 | 887.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 11:15:00 | 887.67 | 877.83 | 887.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 11:15:00 | 888.23 | 879.91 | 887.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 13:45:00 | 884.15 | 882.02 | 887.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 15:15:00 | 901.67 | 887.22 | 889.01 | SL hit (close>static) qty=1.00 sl=893.33 alert=retest2 |

### Cycle 56 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 932.35 | 896.25 | 892.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 14:15:00 | 945.53 | 923.35 | 909.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 11:15:00 | 935.03 | 935.35 | 927.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 12:00:00 | 935.03 | 935.35 | 927.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 13:15:00 | 924.33 | 932.32 | 927.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 14:00:00 | 924.33 | 932.32 | 927.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 928.20 | 931.50 | 927.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 14:30:00 | 925.50 | 931.50 | 927.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 926.67 | 930.53 | 927.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:15:00 | 934.00 | 930.53 | 927.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 09:15:00 | 915.67 | 928.07 | 927.23 | SL hit (close<static) qty=1.00 sl=926.33 alert=retest2 |

### Cycle 57 — SELL (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 10:15:00 | 911.70 | 924.80 | 925.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 12:15:00 | 907.68 | 919.18 | 922.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 09:15:00 | 881.67 | 878.74 | 889.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 881.67 | 878.74 | 889.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 881.67 | 878.74 | 889.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:45:00 | 882.78 | 878.74 | 889.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 908.45 | 885.85 | 889.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 13:30:00 | 914.30 | 885.85 | 889.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 910.97 | 890.87 | 891.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 14:45:00 | 912.63 | 890.87 | 891.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2024-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 15:15:00 | 908.75 | 894.45 | 892.79 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 10:15:00 | 885.97 | 894.59 | 895.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 11:15:00 | 882.50 | 892.17 | 894.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 11:15:00 | 866.27 | 861.73 | 871.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 11:15:00 | 866.27 | 861.73 | 871.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 866.27 | 861.73 | 871.42 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 15:15:00 | 896.67 | 877.46 | 876.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 909.57 | 883.88 | 879.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 884.58 | 898.52 | 891.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 884.58 | 898.52 | 891.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 884.58 | 898.52 | 891.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 873.33 | 898.52 | 891.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 822.40 | 883.30 | 885.43 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 888.33 | 872.49 | 872.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 894.38 | 878.81 | 875.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 09:15:00 | 892.02 | 894.24 | 886.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-07 10:15:00 | 888.50 | 893.09 | 887.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 888.50 | 893.09 | 887.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 10:45:00 | 888.00 | 893.09 | 887.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 11:15:00 | 885.80 | 891.63 | 886.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 11:45:00 | 884.55 | 891.63 | 886.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 12:15:00 | 886.62 | 890.63 | 886.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 13:30:00 | 894.42 | 891.60 | 887.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 11:15:00 | 890.47 | 891.35 | 888.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 893.67 | 888.65 | 888.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 892.72 | 890.90 | 890.04 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 893.40 | 891.40 | 890.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:45:00 | 892.12 | 891.40 | 890.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 912.93 | 910.09 | 904.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 14:30:00 | 923.97 | 911.54 | 907.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 11:15:00 | 903.68 | 908.39 | 908.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 11:15:00 | 903.68 | 908.39 | 908.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 09:15:00 | 879.25 | 899.77 | 904.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 11:15:00 | 840.93 | 840.54 | 854.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-25 12:00:00 | 840.93 | 840.54 | 854.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 849.25 | 840.58 | 845.39 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 15:15:00 | 850.00 | 847.13 | 846.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 09:15:00 | 869.67 | 851.64 | 848.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 12:15:00 | 860.20 | 860.87 | 857.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 12:15:00 | 860.20 | 860.87 | 857.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 12:15:00 | 860.20 | 860.87 | 857.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 13:00:00 | 860.20 | 860.87 | 857.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 858.57 | 860.18 | 857.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 15:00:00 | 858.57 | 860.18 | 857.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 15:15:00 | 859.00 | 859.94 | 857.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:15:00 | 839.67 | 859.94 | 857.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 09:15:00 | 835.35 | 855.02 | 855.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 12:15:00 | 829.77 | 844.28 | 849.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 09:15:00 | 829.58 | 825.18 | 832.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 09:15:00 | 829.58 | 825.18 | 832.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 829.58 | 825.18 | 832.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:45:00 | 831.02 | 825.18 | 832.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 807.93 | 799.18 | 805.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 12:30:00 | 789.67 | 797.02 | 803.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 12:15:00 | 750.19 | 772.56 | 786.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-11 09:15:00 | 780.00 | 766.20 | 778.12 | SL hit (close>ema200) qty=0.50 sl=766.20 alert=retest2 |

### Cycle 66 — BUY (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 11:15:00 | 790.32 | 781.60 | 780.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 798.52 | 789.72 | 786.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 12:15:00 | 790.47 | 791.09 | 787.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 12:45:00 | 791.33 | 791.09 | 787.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 789.33 | 790.50 | 788.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 778.37 | 790.50 | 788.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 773.72 | 787.15 | 787.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 10:00:00 | 773.72 | 787.15 | 787.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 764.67 | 782.65 | 785.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 753.47 | 768.36 | 775.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 757.13 | 755.90 | 763.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 10:15:00 | 757.13 | 755.90 | 763.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 757.13 | 755.90 | 763.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 760.67 | 755.90 | 763.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 740.00 | 746.08 | 755.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:30:00 | 748.33 | 746.08 | 755.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 752.00 | 746.27 | 752.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:00:00 | 752.00 | 746.27 | 752.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 741.67 | 745.35 | 751.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:30:00 | 747.75 | 745.35 | 751.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 741.50 | 744.58 | 750.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:30:00 | 744.33 | 744.58 | 750.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 776.00 | 751.07 | 752.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 776.00 | 751.07 | 752.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 799.40 | 760.74 | 757.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 09:15:00 | 814.67 | 795.19 | 778.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 09:15:00 | 815.97 | 816.63 | 800.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 09:45:00 | 817.33 | 816.63 | 800.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 811.33 | 815.29 | 808.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 10:30:00 | 810.02 | 815.29 | 808.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 11:15:00 | 809.18 | 814.06 | 808.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 12:00:00 | 809.18 | 814.06 | 808.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 12:15:00 | 816.00 | 814.45 | 809.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 13:15:00 | 817.12 | 814.45 | 809.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 14:15:00 | 804.57 | 810.83 | 810.21 | SL hit (close<static) qty=1.00 sl=808.37 alert=retest2 |

### Cycle 69 — SELL (started 2024-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 15:15:00 | 804.18 | 809.50 | 809.66 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 09:15:00 | 862.67 | 820.13 | 814.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-01 10:15:00 | 880.02 | 857.76 | 840.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 864.00 | 871.33 | 856.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 09:15:00 | 864.00 | 871.33 | 856.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 864.00 | 871.33 | 856.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:45:00 | 857.33 | 871.33 | 856.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 12:15:00 | 867.18 | 869.35 | 859.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 12:30:00 | 860.10 | 869.35 | 859.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 854.48 | 865.04 | 859.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 15:00:00 | 854.48 | 865.04 | 859.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 15:15:00 | 855.67 | 863.17 | 858.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 09:15:00 | 807.33 | 863.17 | 858.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 816.00 | 853.73 | 854.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 794.68 | 841.92 | 849.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 797.37 | 789.45 | 804.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 10:15:00 | 797.37 | 789.45 | 804.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 797.37 | 789.45 | 804.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:30:00 | 795.80 | 789.45 | 804.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 802.15 | 791.99 | 804.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:00:00 | 802.15 | 791.99 | 804.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 800.77 | 794.07 | 803.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:00:00 | 800.77 | 794.07 | 803.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 861.67 | 809.20 | 807.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 10:15:00 | 892.43 | 864.84 | 843.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 860.67 | 873.64 | 858.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 860.67 | 873.64 | 858.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 860.67 | 873.64 | 858.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:45:00 | 857.03 | 873.64 | 858.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 855.67 | 866.06 | 860.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 15:00:00 | 855.67 | 866.06 | 860.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 15:15:00 | 857.33 | 864.32 | 859.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 09:30:00 | 860.58 | 862.11 | 859.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 10:15:00 | 855.30 | 860.75 | 859.00 | SL hit (close<static) qty=1.00 sl=855.33 alert=retest2 |

### Cycle 73 — SELL (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 12:15:00 | 852.55 | 857.79 | 857.87 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 14:15:00 | 863.15 | 858.80 | 858.31 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 848.42 | 857.50 | 857.86 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 863.60 | 857.12 | 856.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 12:15:00 | 874.53 | 861.49 | 858.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 09:15:00 | 897.57 | 899.51 | 891.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 897.57 | 899.51 | 891.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 897.57 | 899.51 | 891.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:30:00 | 889.18 | 899.51 | 891.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 895.75 | 896.96 | 892.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 14:30:00 | 891.35 | 896.96 | 892.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 902.00 | 897.72 | 893.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 09:45:00 | 893.33 | 897.72 | 893.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 902.42 | 905.73 | 900.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 11:45:00 | 916.68 | 909.48 | 903.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 15:15:00 | 915.28 | 914.03 | 907.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 09:15:00 | 896.55 | 907.12 | 907.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 09:15:00 | 896.55 | 907.12 | 907.16 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 09:15:00 | 928.73 | 907.62 | 906.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 15:15:00 | 946.67 | 940.08 | 932.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 09:15:00 | 924.25 | 936.91 | 931.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 09:15:00 | 924.25 | 936.91 | 931.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 924.25 | 936.91 | 931.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:30:00 | 924.65 | 936.91 | 931.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 926.23 | 934.78 | 931.43 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2024-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 13:15:00 | 922.52 | 928.58 | 929.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 15:15:00 | 919.67 | 925.69 | 927.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 09:15:00 | 917.83 | 915.49 | 920.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 917.83 | 915.49 | 920.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 917.83 | 915.49 | 920.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:00:00 | 917.83 | 915.49 | 920.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 11:15:00 | 937.20 | 920.22 | 921.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 12:00:00 | 937.20 | 920.22 | 921.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 916.83 | 919.54 | 921.00 | EMA400 retest candle locked (from downside) |

### Cycle 80 — BUY (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 09:15:00 | 953.33 | 927.07 | 924.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 13:15:00 | 958.67 | 943.09 | 933.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 10:15:00 | 948.02 | 950.46 | 940.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 11:00:00 | 948.02 | 950.46 | 940.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 939.37 | 948.30 | 942.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 15:00:00 | 939.37 | 948.30 | 942.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 944.67 | 947.58 | 943.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 915.42 | 947.58 | 943.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 926.90 | 943.44 | 941.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:30:00 | 923.72 | 943.44 | 941.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 11:15:00 | 930.05 | 939.01 | 939.81 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 15:15:00 | 945.67 | 940.61 | 940.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 953.67 | 943.23 | 941.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 15:15:00 | 948.33 | 948.69 | 945.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 09:15:00 | 965.05 | 948.69 | 945.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 965.00 | 951.95 | 947.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:15:00 | 1017.07 | 964.56 | 960.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-16 12:15:00 | 1118.78 | 1019.99 | 989.72 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 13:15:00 | 1283.30 | 1302.37 | 1303.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 15:15:00 | 1278.33 | 1294.25 | 1299.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 10:15:00 | 1236.38 | 1229.10 | 1246.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 10:15:00 | 1236.38 | 1229.10 | 1246.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 1236.38 | 1229.10 | 1246.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:45:00 | 1243.33 | 1229.10 | 1246.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1255.22 | 1232.08 | 1239.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:00:00 | 1255.22 | 1232.08 | 1239.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 1291.60 | 1243.98 | 1244.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 11:00:00 | 1291.60 | 1243.98 | 1244.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 11:15:00 | 1276.00 | 1250.39 | 1247.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-03 09:15:00 | 1395.00 | 1294.12 | 1270.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 13:15:00 | 1308.82 | 1323.80 | 1295.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 14:00:00 | 1308.82 | 1323.80 | 1295.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1339.82 | 1326.89 | 1303.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 11:00:00 | 1365.48 | 1334.61 | 1309.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 14:15:00 | 1270.33 | 1319.28 | 1323.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 14:15:00 | 1270.33 | 1319.28 | 1323.54 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 11:15:00 | 1347.33 | 1325.32 | 1324.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 13:15:00 | 1393.25 | 1344.30 | 1333.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 13:15:00 | 1402.35 | 1407.00 | 1378.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-09 14:00:00 | 1402.35 | 1407.00 | 1378.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 1402.07 | 1412.43 | 1397.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:30:00 | 1399.33 | 1412.43 | 1397.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 1490.72 | 1561.90 | 1542.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:00:00 | 1490.72 | 1561.90 | 1542.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 13:15:00 | 1532.20 | 1539.03 | 1535.76 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 14:15:00 | 1508.42 | 1532.91 | 1533.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 15:15:00 | 1498.33 | 1525.99 | 1530.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 12:15:00 | 1477.95 | 1451.95 | 1474.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 12:15:00 | 1477.95 | 1451.95 | 1474.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 1477.95 | 1451.95 | 1474.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:00:00 | 1477.95 | 1451.95 | 1474.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 1473.28 | 1456.22 | 1474.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 14:30:00 | 1449.32 | 1450.80 | 1470.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 10:30:00 | 1449.05 | 1451.01 | 1465.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:45:00 | 1446.67 | 1450.09 | 1462.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 13:30:00 | 1450.55 | 1451.33 | 1461.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 11:15:00 | 1376.85 | 1420.23 | 1442.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 11:15:00 | 1376.60 | 1420.23 | 1442.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 11:15:00 | 1374.34 | 1420.23 | 1442.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 11:15:00 | 1378.02 | 1420.23 | 1442.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 1422.02 | 1398.55 | 1420.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-23 09:15:00 | 1422.02 | 1398.55 | 1420.76 | SL hit (close>ema200) qty=0.50 sl=1398.55 alert=retest2 |

### Cycle 88 — BUY (started 2024-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 11:15:00 | 1442.00 | 1429.40 | 1427.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 12:15:00 | 1453.70 | 1434.26 | 1430.08 | Break + close above crossover candle high |

### Cycle 89 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 1384.20 | 1426.84 | 1428.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 1368.00 | 1415.07 | 1422.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 09:15:00 | 1393.35 | 1380.17 | 1398.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-28 09:15:00 | 1393.35 | 1380.17 | 1398.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 1393.35 | 1380.17 | 1398.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:00:00 | 1393.35 | 1380.17 | 1398.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 1406.18 | 1385.37 | 1398.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:45:00 | 1406.87 | 1385.37 | 1398.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 1400.00 | 1388.29 | 1398.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 13:00:00 | 1389.72 | 1388.58 | 1398.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 14:00:00 | 1389.47 | 1388.76 | 1397.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:30:00 | 1389.60 | 1392.42 | 1396.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-29 14:15:00 | 1430.78 | 1404.86 | 1401.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 1430.78 | 1404.86 | 1401.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 1480.67 | 1423.78 | 1410.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 1428.73 | 1442.99 | 1426.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 14:00:00 | 1428.73 | 1442.99 | 1426.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1477.27 | 1477.34 | 1460.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 13:30:00 | 1535.28 | 1490.00 | 1477.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 14:30:00 | 1522.55 | 1500.00 | 1483.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 12:15:00 | 1570.77 | 1578.17 | 1578.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 12:15:00 | 1570.77 | 1578.17 | 1578.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 1551.67 | 1572.87 | 1576.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 1578.33 | 1564.95 | 1570.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 1578.33 | 1564.95 | 1570.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1578.33 | 1564.95 | 1570.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:30:00 | 1581.90 | 1564.95 | 1570.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 1564.15 | 1564.79 | 1570.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 13:15:00 | 1549.12 | 1563.35 | 1568.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 09:15:00 | 1545.00 | 1561.33 | 1566.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 12:15:00 | 1565.20 | 1542.86 | 1540.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 12:15:00 | 1565.20 | 1542.86 | 1540.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 1613.57 | 1565.97 | 1552.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 1573.67 | 1581.23 | 1567.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 15:00:00 | 1573.67 | 1581.23 | 1567.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 1572.33 | 1579.45 | 1567.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 09:15:00 | 1599.92 | 1579.45 | 1567.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-21 10:15:00 | 1563.00 | 1579.09 | 1569.69 | SL hit (close<static) qty=1.00 sl=1564.53 alert=retest2 |

### Cycle 93 — SELL (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 10:15:00 | 1560.73 | 1567.94 | 1567.95 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 11:15:00 | 1574.68 | 1569.29 | 1568.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 13:15:00 | 1579.27 | 1571.99 | 1569.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 1549.40 | 1595.31 | 1586.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 14:15:00 | 1549.40 | 1595.31 | 1586.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 1549.40 | 1595.31 | 1586.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 1549.40 | 1595.31 | 1586.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 1533.33 | 1582.91 | 1581.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:15:00 | 1534.77 | 1582.91 | 1581.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 09:15:00 | 1498.72 | 1566.07 | 1573.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 12:15:00 | 1481.62 | 1529.41 | 1553.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 13:15:00 | 1491.73 | 1491.48 | 1515.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-27 13:30:00 | 1477.28 | 1491.48 | 1515.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1520.85 | 1498.35 | 1513.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:45:00 | 1525.68 | 1498.35 | 1513.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1511.75 | 1501.03 | 1512.89 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 09:15:00 | 1552.70 | 1521.99 | 1518.80 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 15:15:00 | 1516.67 | 1530.48 | 1531.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-03 11:15:00 | 1503.17 | 1523.99 | 1528.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 14:15:00 | 1526.33 | 1509.60 | 1515.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 14:15:00 | 1526.33 | 1509.60 | 1515.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 1526.33 | 1509.60 | 1515.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 15:00:00 | 1526.33 | 1509.60 | 1515.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 1528.67 | 1513.41 | 1516.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:15:00 | 1565.30 | 1513.41 | 1516.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 09:15:00 | 1594.57 | 1529.64 | 1523.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 10:15:00 | 1673.97 | 1558.51 | 1537.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 09:15:00 | 1806.30 | 1812.67 | 1763.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-10 10:00:00 | 1806.30 | 1812.67 | 1763.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 1851.77 | 1863.14 | 1843.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 13:00:00 | 1874.50 | 1858.59 | 1845.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 14:00:00 | 1873.43 | 1861.56 | 1848.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 13:00:00 | 1872.50 | 1887.53 | 1885.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 14:15:00 | 1874.17 | 1883.65 | 1884.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 14:15:00 | 1874.17 | 1883.65 | 1884.29 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 09:15:00 | 1894.97 | 1885.01 | 1884.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 11:15:00 | 1912.83 | 1892.54 | 1888.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 09:15:00 | 1913.00 | 1913.15 | 1901.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 10:00:00 | 1913.00 | 1913.15 | 1901.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 1888.03 | 1907.78 | 1901.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:00:00 | 1888.03 | 1907.78 | 1901.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 1863.80 | 1898.98 | 1897.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:00:00 | 1863.80 | 1898.98 | 1897.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 1866.67 | 1892.52 | 1894.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 1841.78 | 1882.37 | 1890.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 14:15:00 | 1817.00 | 1809.52 | 1826.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-26 15:00:00 | 1817.00 | 1809.52 | 1826.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1777.10 | 1803.53 | 1820.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 12:15:00 | 1769.62 | 1793.62 | 1812.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 12:45:00 | 1768.95 | 1789.49 | 1809.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 14:45:00 | 1769.97 | 1781.60 | 1802.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 10:15:00 | 1768.10 | 1777.49 | 1796.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 1771.42 | 1768.36 | 1784.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:45:00 | 1785.28 | 1768.36 | 1784.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 1775.18 | 1769.72 | 1783.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:30:00 | 1758.22 | 1766.63 | 1780.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 11:00:00 | 1762.20 | 1765.74 | 1778.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 09:15:00 | 1789.65 | 1770.71 | 1774.75 | SL hit (close>static) qty=1.00 sl=1789.33 alert=retest2 |

### Cycle 102 — BUY (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 11:15:00 | 1796.87 | 1779.82 | 1778.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 12:15:00 | 1807.75 | 1785.41 | 1781.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 1796.50 | 1796.59 | 1788.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 09:45:00 | 1796.65 | 1796.59 | 1788.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 1806.00 | 1798.47 | 1790.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:30:00 | 1791.67 | 1798.47 | 1790.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 1796.35 | 1813.71 | 1806.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:00:00 | 1796.35 | 1813.71 | 1806.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 1781.02 | 1807.17 | 1804.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 1781.02 | 1807.17 | 1804.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2025-01-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 15:15:00 | 1778.00 | 1801.34 | 1801.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 1757.27 | 1792.52 | 1797.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1735.00 | 1731.25 | 1757.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:30:00 | 1732.08 | 1731.25 | 1757.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 1782.05 | 1743.10 | 1756.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:00:00 | 1782.05 | 1743.10 | 1756.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 1808.32 | 1756.14 | 1761.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:00:00 | 1808.32 | 1756.14 | 1761.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2025-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 15:15:00 | 1788.27 | 1769.26 | 1766.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 13:15:00 | 1804.63 | 1782.66 | 1774.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 09:15:00 | 1767.13 | 1785.59 | 1778.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 09:15:00 | 1767.13 | 1785.59 | 1778.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1767.13 | 1785.59 | 1778.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 1767.13 | 1785.59 | 1778.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 1750.60 | 1778.59 | 1775.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:45:00 | 1749.33 | 1778.59 | 1775.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2025-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 11:15:00 | 1754.33 | 1773.74 | 1774.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 14:15:00 | 1738.67 | 1758.94 | 1766.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-13 09:15:00 | 1731.07 | 1719.71 | 1736.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-13 09:15:00 | 1731.07 | 1719.71 | 1736.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 1731.07 | 1719.71 | 1736.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-13 09:30:00 | 1720.08 | 1719.71 | 1736.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 10:15:00 | 1735.68 | 1722.90 | 1736.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-13 10:30:00 | 1741.67 | 1722.90 | 1736.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 11:15:00 | 1742.00 | 1726.72 | 1736.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-13 12:00:00 | 1742.00 | 1726.72 | 1736.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 12:15:00 | 1746.33 | 1730.64 | 1737.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-13 13:00:00 | 1746.33 | 1730.64 | 1737.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 13:15:00 | 1727.92 | 1730.10 | 1736.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 14:15:00 | 1709.67 | 1730.10 | 1736.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 15:00:00 | 1721.25 | 1728.33 | 1735.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 1786.77 | 1737.88 | 1738.31 | SL hit (close>static) qty=1.00 sl=1749.97 alert=retest2 |

### Cycle 106 — BUY (started 2025-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 10:15:00 | 1807.33 | 1751.77 | 1744.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 11:15:00 | 1821.00 | 1765.62 | 1751.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 11:15:00 | 1971.53 | 1975.87 | 1932.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 12:00:00 | 1971.53 | 1975.87 | 1932.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 1943.85 | 1977.83 | 1965.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 1943.85 | 1977.83 | 1965.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1925.97 | 1967.45 | 1962.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:45:00 | 1936.58 | 1967.45 | 1962.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 1955.67 | 1964.28 | 1961.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:30:00 | 1954.17 | 1964.28 | 1961.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 1960.33 | 1963.49 | 1961.60 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 1927.73 | 1956.34 | 1958.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 15:15:00 | 1927.33 | 1950.54 | 1955.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 1922.63 | 1916.73 | 1933.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 1922.63 | 1916.73 | 1933.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1936.67 | 1920.92 | 1932.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 1948.02 | 1920.92 | 1932.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1954.12 | 1927.56 | 1934.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 1946.67 | 1927.56 | 1934.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 1976.75 | 1944.99 | 1941.77 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 1865.72 | 1936.49 | 1943.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 12:15:00 | 1852.33 | 1898.14 | 1922.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 1765.33 | 1761.88 | 1814.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 09:30:00 | 1774.03 | 1761.88 | 1814.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 1767.80 | 1770.33 | 1792.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 10:45:00 | 1758.10 | 1770.03 | 1790.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 11:30:00 | 1751.65 | 1763.77 | 1785.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 11:45:00 | 1759.45 | 1759.20 | 1771.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 12:15:00 | 1759.23 | 1759.20 | 1771.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 1768.50 | 1757.58 | 1767.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 15:00:00 | 1768.50 | 1757.58 | 1767.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 15:15:00 | 1771.67 | 1760.39 | 1767.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:15:00 | 1791.63 | 1760.39 | 1767.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 1802.02 | 1768.72 | 1770.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:30:00 | 1809.33 | 1768.72 | 1770.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-01 10:15:00 | 1793.67 | 1773.71 | 1772.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2025-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 10:15:00 | 1793.67 | 1773.71 | 1772.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 09:15:00 | 1824.22 | 1799.07 | 1789.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 1884.00 | 1901.95 | 1867.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 09:45:00 | 1884.33 | 1901.95 | 1867.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 1893.33 | 1900.23 | 1869.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:30:00 | 1875.00 | 1900.23 | 1869.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 1867.87 | 1891.08 | 1872.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:00:00 | 1867.87 | 1891.08 | 1872.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 1859.62 | 1884.79 | 1871.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 1859.62 | 1884.79 | 1871.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 1854.00 | 1878.63 | 1870.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 1875.98 | 1878.63 | 1870.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1886.97 | 1882.29 | 1873.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 1895.32 | 1882.29 | 1873.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 12:30:00 | 1897.43 | 1887.33 | 1877.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 14:30:00 | 1902.72 | 1892.97 | 1881.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 1799.52 | 1867.35 | 1875.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 1799.52 | 1867.35 | 1875.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 10:15:00 | 1764.10 | 1846.70 | 1865.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 1773.13 | 1761.43 | 1799.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:45:00 | 1782.67 | 1761.43 | 1799.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1810.00 | 1771.15 | 1800.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 1810.00 | 1771.15 | 1800.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 1775.48 | 1772.01 | 1798.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 14:15:00 | 1766.70 | 1772.01 | 1798.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 09:15:00 | 1752.37 | 1775.62 | 1795.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 10:15:00 | 1769.08 | 1776.50 | 1794.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:00:00 | 1769.33 | 1776.24 | 1789.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 1756.22 | 1762.60 | 1778.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:15:00 | 1750.45 | 1762.60 | 1778.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 1678.37 | 1712.48 | 1741.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 1664.75 | 1712.48 | 1741.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 1680.63 | 1712.48 | 1741.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 1680.86 | 1712.48 | 1741.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 1662.93 | 1712.48 | 1741.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 12:15:00 | 1725.10 | 1714.54 | 1735.30 | SL hit (close>ema200) qty=0.50 sl=1714.54 alert=retest2 |

### Cycle 112 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 1831.33 | 1741.91 | 1735.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 11:15:00 | 1861.73 | 1780.48 | 1755.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1935.95 | 1937.93 | 1880.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 1935.95 | 1937.93 | 1880.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1874.23 | 1916.62 | 1897.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:00:00 | 1874.23 | 1916.62 | 1897.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 1876.67 | 1908.63 | 1895.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:30:00 | 1889.98 | 1902.41 | 1893.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 13:15:00 | 1866.63 | 1890.29 | 1889.54 | SL hit (close<static) qty=1.00 sl=1870.33 alert=retest2 |

### Cycle 113 — SELL (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 14:15:00 | 1870.88 | 1886.41 | 1887.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 15:15:00 | 1863.38 | 1881.80 | 1885.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 1498.75 | 1495.02 | 1572.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 09:45:00 | 1515.52 | 1495.02 | 1572.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 1343.33 | 1303.90 | 1331.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:30:00 | 1345.12 | 1303.90 | 1331.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 1329.22 | 1308.96 | 1331.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 1311.80 | 1323.56 | 1332.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 13:00:00 | 1318.98 | 1325.74 | 1330.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 15:00:00 | 1308.78 | 1322.50 | 1328.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 09:30:00 | 1314.95 | 1322.31 | 1327.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 1322.33 | 1322.32 | 1326.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:15:00 | 1318.63 | 1322.32 | 1326.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:45:00 | 1317.08 | 1321.44 | 1325.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 14:15:00 | 1319.67 | 1320.43 | 1324.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 1341.73 | 1324.56 | 1325.41 | SL hit (close>static) qty=1.00 sl=1341.67 alert=retest2 |

### Cycle 114 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 1342.98 | 1328.24 | 1327.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 13:15:00 | 1360.90 | 1339.51 | 1332.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 13:15:00 | 1576.00 | 1576.26 | 1535.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 13:30:00 | 1573.50 | 1576.26 | 1535.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1561.67 | 1563.39 | 1541.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:30:00 | 1556.97 | 1563.39 | 1541.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1550.33 | 1560.78 | 1542.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 1542.67 | 1560.78 | 1542.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 1546.85 | 1557.99 | 1542.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:15:00 | 1541.77 | 1557.99 | 1542.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 1562.93 | 1558.98 | 1544.58 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 1502.92 | 1533.51 | 1537.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 1499.33 | 1526.67 | 1534.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 1563.63 | 1524.95 | 1530.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 09:15:00 | 1563.63 | 1524.95 | 1530.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 1563.63 | 1524.95 | 1530.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:45:00 | 1573.55 | 1524.95 | 1530.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 1564.18 | 1532.79 | 1533.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 11:15:00 | 1544.55 | 1532.79 | 1533.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 11:15:00 | 1552.30 | 1536.69 | 1535.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 11:15:00 | 1552.30 | 1536.69 | 1535.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 1779.15 | 1596.02 | 1564.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 12:15:00 | 1868.93 | 1869.56 | 1828.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-03 12:45:00 | 1869.42 | 1869.56 | 1828.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1828.00 | 1863.07 | 1838.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 1828.00 | 1863.07 | 1838.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 1849.72 | 1860.40 | 1839.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 11:15:00 | 1857.63 | 1860.40 | 1839.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 1671.67 | 1813.51 | 1825.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1671.67 | 1813.51 | 1825.96 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2025-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 14:15:00 | 1846.00 | 1801.49 | 1795.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 12:15:00 | 1868.67 | 1825.39 | 1810.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1963.00 | 1969.30 | 1944.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 1963.00 | 1969.30 | 1944.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 2119.50 | 2151.82 | 2128.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 2119.50 | 2151.82 | 2128.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 2088.67 | 2139.19 | 2125.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 2088.67 | 2139.19 | 2125.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 2103.67 | 2118.12 | 2117.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 14:45:00 | 2108.17 | 2118.12 | 2117.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 15:15:00 | 2098.67 | 2114.23 | 2116.14 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 2141.33 | 2119.65 | 2118.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 11:15:00 | 2156.67 | 2131.11 | 2124.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 2160.17 | 2200.68 | 2180.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 2160.17 | 2200.68 | 2180.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 2160.17 | 2200.68 | 2180.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:45:00 | 2154.67 | 2200.68 | 2180.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 2145.67 | 2189.68 | 2177.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:45:00 | 2150.33 | 2189.68 | 2177.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2025-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 12:15:00 | 2128.50 | 2169.22 | 2169.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 14:15:00 | 2120.33 | 2153.49 | 2162.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 10:15:00 | 2130.67 | 2121.61 | 2134.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 10:30:00 | 2129.83 | 2121.61 | 2134.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 2130.00 | 2123.28 | 2134.44 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 09:15:00 | 2157.00 | 2139.28 | 2138.49 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 2121.33 | 2135.19 | 2136.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 2104.50 | 2129.05 | 2133.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 09:15:00 | 2205.50 | 2130.59 | 2132.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 2205.50 | 2130.59 | 2132.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 2205.50 | 2130.59 | 2132.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:30:00 | 2230.83 | 2130.59 | 2132.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 10:15:00 | 2254.00 | 2155.27 | 2143.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 11:15:00 | 2258.67 | 2175.95 | 2153.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 2239.67 | 2250.22 | 2216.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 15:00:00 | 2239.67 | 2250.22 | 2216.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 2214.83 | 2243.14 | 2216.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 2174.17 | 2243.14 | 2216.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 2183.67 | 2231.25 | 2213.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:00:00 | 2183.67 | 2231.25 | 2213.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 2171.67 | 2219.33 | 2209.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:30:00 | 2163.67 | 2219.33 | 2209.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2025-05-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 13:15:00 | 2175.00 | 2199.69 | 2202.17 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 2319.83 | 2221.91 | 2211.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 2386.67 | 2308.25 | 2266.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 2399.33 | 2427.82 | 2388.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 09:30:00 | 2406.33 | 2427.82 | 2388.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 2393.00 | 2420.85 | 2389.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:00:00 | 2393.00 | 2420.85 | 2389.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 2385.50 | 2413.78 | 2389.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:30:00 | 2392.50 | 2413.78 | 2389.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 2375.67 | 2406.16 | 2387.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:00:00 | 2375.67 | 2406.16 | 2387.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 2401.00 | 2405.13 | 2388.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 2422.33 | 2397.18 | 2387.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 12:15:00 | 2424.33 | 2453.89 | 2454.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 12:15:00 | 2424.33 | 2453.89 | 2454.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 10:15:00 | 2304.67 | 2410.24 | 2432.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 2449.50 | 2379.95 | 2402.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 2449.50 | 2379.95 | 2402.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 2449.50 | 2379.95 | 2402.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 2449.50 | 2379.95 | 2402.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 2405.50 | 2385.06 | 2402.63 | EMA400 retest candle locked (from downside) |

### Cycle 128 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 2457.50 | 2412.92 | 2412.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 2493.50 | 2438.53 | 2424.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 14:15:00 | 2452.50 | 2455.04 | 2440.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 14:15:00 | 2452.50 | 2455.04 | 2440.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 2452.50 | 2455.04 | 2440.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:30:00 | 2450.00 | 2455.04 | 2440.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 2418.00 | 2447.15 | 2439.03 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 2403.50 | 2430.01 | 2432.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 2391.00 | 2413.08 | 2422.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 10:15:00 | 2414.00 | 2413.27 | 2422.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-28 11:00:00 | 2414.00 | 2413.27 | 2422.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 2420.00 | 2414.61 | 2421.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:45:00 | 2423.00 | 2414.61 | 2421.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 2418.00 | 2415.29 | 2421.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:45:00 | 2421.50 | 2415.29 | 2421.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 2406.00 | 2413.47 | 2419.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 10:00:00 | 2396.00 | 2408.46 | 2416.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 13:15:00 | 2443.00 | 2416.30 | 2417.29 | SL hit (close>static) qty=1.00 sl=2421.50 alert=retest2 |

### Cycle 130 — BUY (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 14:15:00 | 2465.50 | 2426.14 | 2421.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 2613.00 | 2470.21 | 2442.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 2718.00 | 2732.48 | 2682.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 09:45:00 | 2721.60 | 2732.48 | 2682.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 2928.00 | 2986.47 | 2963.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 2928.00 | 2986.47 | 2963.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 2917.30 | 2972.64 | 2959.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 2924.80 | 2972.64 | 2959.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 12:15:00 | 2877.50 | 2941.20 | 2946.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 2812.00 | 2890.09 | 2918.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 09:15:00 | 2712.00 | 2706.97 | 2740.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-17 09:45:00 | 2718.80 | 2706.97 | 2740.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 2655.30 | 2619.62 | 2637.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 2655.30 | 2619.62 | 2637.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 2657.10 | 2627.12 | 2638.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:00:00 | 2657.10 | 2627.12 | 2638.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 2655.00 | 2632.69 | 2640.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:45:00 | 2663.50 | 2632.69 | 2640.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 15:15:00 | 2691.00 | 2653.03 | 2648.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 10:15:00 | 2723.50 | 2671.18 | 2658.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 13:15:00 | 2787.80 | 2792.81 | 2766.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 14:00:00 | 2787.80 | 2792.81 | 2766.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 2756.00 | 2784.61 | 2769.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:45:00 | 2793.70 | 2779.04 | 2769.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:30:00 | 2791.40 | 2782.63 | 2772.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 11:15:00 | 2778.00 | 2784.67 | 2785.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 11:15:00 | 2778.00 | 2784.67 | 2785.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 12:15:00 | 2759.90 | 2773.13 | 2778.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 14:15:00 | 2777.20 | 2769.41 | 2775.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 14:15:00 | 2777.20 | 2769.41 | 2775.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 2777.20 | 2769.41 | 2775.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 15:00:00 | 2777.20 | 2769.41 | 2775.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 2773.80 | 2770.29 | 2775.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 2793.40 | 2770.29 | 2775.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 2778.00 | 2771.83 | 2775.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 10:30:00 | 2757.40 | 2769.79 | 2774.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 12:45:00 | 2760.40 | 2767.50 | 2772.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 11:15:00 | 2787.90 | 2774.48 | 2773.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 2787.90 | 2774.48 | 2773.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 14:15:00 | 2814.50 | 2787.46 | 2779.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 09:15:00 | 2707.00 | 2777.38 | 2776.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 2707.00 | 2777.38 | 2776.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 2707.00 | 2777.38 | 2776.88 | EMA400 retest candle locked (from upside) |

### Cycle 135 — SELL (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 10:15:00 | 2670.40 | 2755.98 | 2767.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 2654.00 | 2735.58 | 2756.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 2530.60 | 2517.42 | 2578.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 09:30:00 | 2537.60 | 2517.42 | 2578.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 2439.90 | 2412.32 | 2452.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:15:00 | 2456.00 | 2412.32 | 2452.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 2447.00 | 2419.26 | 2452.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 2465.90 | 2419.26 | 2452.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 2434.70 | 2422.35 | 2450.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:15:00 | 2446.50 | 2422.35 | 2450.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 2437.10 | 2425.30 | 2449.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:45:00 | 2440.50 | 2425.30 | 2449.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 2450.00 | 2430.24 | 2449.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:30:00 | 2448.10 | 2430.24 | 2449.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 2460.00 | 2436.19 | 2450.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:45:00 | 2460.40 | 2436.19 | 2450.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 2467.00 | 2442.35 | 2452.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 2476.30 | 2442.35 | 2452.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 2467.30 | 2450.49 | 2454.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 2467.30 | 2450.49 | 2454.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 2486.50 | 2457.69 | 2457.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 2532.60 | 2472.67 | 2464.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 14:15:00 | 2528.50 | 2534.30 | 2510.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 15:00:00 | 2528.50 | 2534.30 | 2510.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 2512.00 | 2528.83 | 2512.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 2507.50 | 2528.83 | 2512.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 2525.50 | 2528.16 | 2513.55 | EMA400 retest candle locked (from upside) |

### Cycle 137 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 2464.00 | 2499.10 | 2502.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 2454.00 | 2483.57 | 2493.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 2506.00 | 2466.99 | 2476.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 2506.00 | 2466.99 | 2476.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 2506.00 | 2466.99 | 2476.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 2502.20 | 2466.99 | 2476.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 2523.00 | 2478.19 | 2480.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:00:00 | 2523.00 | 2478.19 | 2480.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 11:15:00 | 2528.50 | 2488.25 | 2485.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 09:15:00 | 2553.20 | 2516.46 | 2501.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 2520.60 | 2537.15 | 2522.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 2520.60 | 2537.15 | 2522.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 2520.60 | 2537.15 | 2522.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:30:00 | 2528.70 | 2537.15 | 2522.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 2522.60 | 2534.24 | 2522.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:30:00 | 2522.30 | 2534.24 | 2522.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 2523.80 | 2532.15 | 2522.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:30:00 | 2517.80 | 2532.15 | 2522.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 2536.90 | 2532.28 | 2524.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 14:45:00 | 2549.10 | 2536.62 | 2527.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 09:30:00 | 2539.70 | 2538.64 | 2529.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 12:00:00 | 2545.10 | 2538.55 | 2531.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 2479.70 | 2531.01 | 2531.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 2479.70 | 2531.01 | 2531.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 11:15:00 | 2469.40 | 2509.65 | 2520.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 2520.60 | 2488.11 | 2503.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 2520.60 | 2488.11 | 2503.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 2520.60 | 2488.11 | 2503.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 2520.60 | 2488.11 | 2503.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 2500.80 | 2490.64 | 2502.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:30:00 | 2476.50 | 2486.84 | 2500.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 15:15:00 | 2485.00 | 2468.74 | 2476.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 14:15:00 | 2485.00 | 2449.99 | 2445.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 2485.00 | 2449.99 | 2445.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 15:15:00 | 2493.20 | 2458.63 | 2450.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 10:15:00 | 2396.70 | 2447.91 | 2446.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 10:15:00 | 2396.70 | 2447.91 | 2446.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 2396.70 | 2447.91 | 2446.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 11:00:00 | 2396.70 | 2447.91 | 2446.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 11:15:00 | 2412.00 | 2440.72 | 2443.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 13:15:00 | 2369.30 | 2418.68 | 2432.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 11:15:00 | 2407.40 | 2373.03 | 2401.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 11:15:00 | 2407.40 | 2373.03 | 2401.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 2407.40 | 2373.03 | 2401.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:00:00 | 2407.40 | 2373.03 | 2401.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 2400.00 | 2378.42 | 2401.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 15:00:00 | 2382.20 | 2382.71 | 2399.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 13:15:00 | 2378.40 | 2389.50 | 2397.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 2432.20 | 2400.84 | 2401.16 | SL hit (close>static) qty=1.00 sl=2428.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 2450.20 | 2410.71 | 2405.62 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 13:15:00 | 2394.70 | 2403.33 | 2403.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 2389.40 | 2399.11 | 2401.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 10:15:00 | 2397.40 | 2396.80 | 2400.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 10:15:00 | 2397.40 | 2396.80 | 2400.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 2397.40 | 2396.80 | 2400.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:30:00 | 2403.60 | 2396.80 | 2400.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 2395.00 | 2396.44 | 2399.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:30:00 | 2393.20 | 2396.44 | 2399.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 2410.70 | 2399.29 | 2400.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:45:00 | 2413.90 | 2399.29 | 2400.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 2407.40 | 2400.91 | 2401.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 14:15:00 | 2402.20 | 2400.91 | 2401.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 2456.00 | 2393.78 | 2393.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 2456.00 | 2393.78 | 2393.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 12:15:00 | 2478.00 | 2429.78 | 2411.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 13:15:00 | 2494.20 | 2497.21 | 2475.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 14:00:00 | 2494.20 | 2497.21 | 2475.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 2488.60 | 2495.29 | 2480.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 2471.50 | 2495.29 | 2480.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 2496.00 | 2502.62 | 2490.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 2496.00 | 2502.62 | 2490.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 2495.80 | 2501.26 | 2490.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 2496.80 | 2501.26 | 2490.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 2503.50 | 2501.71 | 2491.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 12:15:00 | 2518.80 | 2504.95 | 2495.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:15:00 | 2520.00 | 2516.58 | 2505.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 11:15:00 | 2399.90 | 2493.46 | 2497.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 11:15:00 | 2399.90 | 2493.46 | 2497.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 2379.30 | 2470.63 | 2486.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 2185.70 | 2140.81 | 2180.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 2185.70 | 2140.81 | 2180.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 2185.70 | 2140.81 | 2180.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 2185.70 | 2140.81 | 2180.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 2175.00 | 2147.65 | 2179.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:45:00 | 2165.50 | 2150.70 | 2178.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 12:45:00 | 2168.20 | 2157.22 | 2178.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 09:15:00 | 2169.00 | 2168.87 | 2179.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 2235.80 | 2182.26 | 2184.40 | SL hit (close>static) qty=1.00 sl=2193.70 alert=retest2 |

### Cycle 146 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 2228.70 | 2191.55 | 2188.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 09:15:00 | 2305.00 | 2236.17 | 2219.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 2337.40 | 2340.58 | 2310.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:30:00 | 2332.10 | 2340.58 | 2310.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 2262.30 | 2329.81 | 2324.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:45:00 | 2267.30 | 2329.81 | 2324.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 2278.60 | 2319.57 | 2320.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 09:15:00 | 2241.10 | 2285.05 | 2301.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 2210.00 | 2203.90 | 2244.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 10:00:00 | 2210.00 | 2203.90 | 2244.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 2227.00 | 2205.91 | 2225.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 2193.00 | 2225.02 | 2228.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:30:00 | 2213.00 | 2210.63 | 2214.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 13:00:00 | 2215.50 | 2211.60 | 2214.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 13:45:00 | 2214.60 | 2211.28 | 2214.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 2229.30 | 2214.58 | 2215.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:45:00 | 2229.50 | 2214.58 | 2215.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 2202.00 | 2203.72 | 2208.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:30:00 | 2201.00 | 2203.72 | 2208.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 09:15:00 | 2102.35 | 2158.09 | 2176.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 09:15:00 | 2104.72 | 2158.09 | 2176.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 09:15:00 | 2103.87 | 2158.09 | 2176.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 15:15:00 | 2083.35 | 2101.83 | 2126.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 2064.80 | 2061.10 | 2086.78 | SL hit (close>ema200) qty=0.50 sl=2061.10 alert=retest2 |

### Cycle 148 — BUY (started 2025-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 13:15:00 | 2096.00 | 2078.17 | 2078.10 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 10:15:00 | 2058.00 | 2076.96 | 2078.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 12:15:00 | 2046.40 | 2066.89 | 2073.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 2063.00 | 2056.83 | 2065.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 10:00:00 | 2063.00 | 2056.83 | 2065.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 2046.40 | 2054.74 | 2063.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:15:00 | 2041.00 | 2054.74 | 2063.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 2074.40 | 2059.30 | 2063.69 | SL hit (close>static) qty=1.00 sl=2071.40 alert=retest2 |

### Cycle 150 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 2080.50 | 2067.00 | 2066.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 2085.20 | 2070.64 | 2068.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 2212.00 | 2221.68 | 2189.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 11:00:00 | 2212.00 | 2221.68 | 2189.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 2489.90 | 2509.66 | 2489.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 2489.90 | 2509.66 | 2489.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 2486.30 | 2504.98 | 2488.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 2477.60 | 2504.98 | 2488.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 2478.90 | 2499.77 | 2487.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 2478.90 | 2499.77 | 2487.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 2489.00 | 2497.61 | 2488.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 2506.60 | 2497.61 | 2488.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 09:15:00 | 2497.10 | 2495.60 | 2493.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:30:00 | 2500.50 | 2500.34 | 2499.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 11:00:00 | 2493.40 | 2498.95 | 2498.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 11:15:00 | 2495.00 | 2498.16 | 2498.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 2495.00 | 2498.16 | 2498.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 2473.60 | 2491.62 | 2495.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 2499.60 | 2488.32 | 2492.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 2499.60 | 2488.32 | 2492.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 2499.60 | 2488.32 | 2492.38 | EMA400 retest candle locked (from downside) |

### Cycle 152 — BUY (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 14:15:00 | 2510.50 | 2495.46 | 2494.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 15:15:00 | 2512.00 | 2498.77 | 2496.06 | Break + close above crossover candle high |

### Cycle 153 — SELL (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 09:15:00 | 2463.00 | 2491.61 | 2493.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 10:15:00 | 2426.60 | 2478.61 | 2487.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 2441.00 | 2440.39 | 2460.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 09:30:00 | 2436.00 | 2440.39 | 2460.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 2456.40 | 2446.29 | 2453.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:30:00 | 2464.40 | 2446.29 | 2453.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 2464.60 | 2449.95 | 2454.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 2461.30 | 2449.95 | 2454.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 2456.90 | 2451.34 | 2454.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 12:15:00 | 2447.30 | 2451.34 | 2454.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 12:15:00 | 2471.80 | 2434.85 | 2442.53 | SL hit (close>static) qty=1.00 sl=2465.80 alert=retest2 |

### Cycle 154 — BUY (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 14:15:00 | 2478.00 | 2451.76 | 2449.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 09:15:00 | 2530.50 | 2472.04 | 2459.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 2522.40 | 2527.89 | 2501.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 11:15:00 | 2516.80 | 2523.51 | 2503.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 2516.80 | 2523.51 | 2503.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:45:00 | 2505.60 | 2523.51 | 2503.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 2503.80 | 2519.57 | 2503.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:00:00 | 2503.80 | 2519.57 | 2503.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 2494.30 | 2514.52 | 2502.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:00:00 | 2494.30 | 2514.52 | 2502.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 2499.10 | 2511.43 | 2502.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:45:00 | 2493.50 | 2511.43 | 2502.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 2445.10 | 2495.54 | 2496.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 2437.50 | 2483.93 | 2491.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 09:15:00 | 2502.10 | 2469.32 | 2477.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 09:15:00 | 2502.10 | 2469.32 | 2477.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 2502.10 | 2469.32 | 2477.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:45:00 | 2486.10 | 2469.32 | 2477.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — BUY (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 10:15:00 | 2539.80 | 2483.41 | 2483.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 12:15:00 | 2603.60 | 2517.54 | 2499.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 14:15:00 | 2626.90 | 2630.32 | 2587.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 15:00:00 | 2626.90 | 2630.32 | 2587.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 2798.10 | 2775.72 | 2741.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:00:00 | 2816.00 | 2787.81 | 2753.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 14:45:00 | 2825.20 | 2800.84 | 2768.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 12:45:00 | 2817.70 | 2808.14 | 2784.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 2841.00 | 2811.56 | 2792.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 2839.60 | 2817.17 | 2796.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:00:00 | 2880.00 | 2848.83 | 2828.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 2847.10 | 2873.91 | 2877.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 09:15:00 | 2847.10 | 2873.91 | 2877.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 10:15:00 | 2825.80 | 2864.28 | 2872.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 11:15:00 | 2843.90 | 2828.36 | 2844.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 11:15:00 | 2843.90 | 2828.36 | 2844.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 2843.90 | 2828.36 | 2844.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:00:00 | 2843.90 | 2828.36 | 2844.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 2829.90 | 2828.67 | 2842.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:30:00 | 2842.10 | 2828.67 | 2842.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 2842.90 | 2833.01 | 2842.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 2842.90 | 2833.01 | 2842.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 2835.10 | 2833.42 | 2841.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 2873.10 | 2833.42 | 2841.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 2881.00 | 2842.94 | 2845.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:30:00 | 2877.10 | 2842.94 | 2845.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 2898.00 | 2853.95 | 2850.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 2903.80 | 2880.26 | 2866.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 13:15:00 | 2910.50 | 2910.50 | 2897.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 14:15:00 | 2902.20 | 2910.50 | 2897.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 2900.00 | 2908.40 | 2897.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:45:00 | 2902.90 | 2908.40 | 2897.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 2910.70 | 2912.68 | 2903.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:30:00 | 2906.60 | 2912.68 | 2903.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 2905.80 | 2911.31 | 2904.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:30:00 | 2908.00 | 2911.31 | 2904.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 2889.00 | 2906.84 | 2902.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:45:00 | 2877.40 | 2906.84 | 2902.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 2892.00 | 2903.88 | 2901.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 2880.10 | 2903.88 | 2901.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 2866.60 | 2896.42 | 2898.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 10:15:00 | 2849.90 | 2887.12 | 2894.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 10:15:00 | 2792.30 | 2773.83 | 2795.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 11:00:00 | 2792.30 | 2773.83 | 2795.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 2794.50 | 2777.96 | 2795.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:45:00 | 2799.90 | 2777.96 | 2795.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 2808.70 | 2784.11 | 2796.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:00:00 | 2808.70 | 2784.11 | 2796.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 2816.80 | 2790.65 | 2798.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 2816.80 | 2790.65 | 2798.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 2815.60 | 2799.04 | 2800.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 2896.80 | 2799.04 | 2800.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 09:15:00 | 2883.00 | 2815.83 | 2808.32 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 2686.50 | 2788.25 | 2800.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 2579.90 | 2661.85 | 2712.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 2657.00 | 2648.12 | 2696.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 10:00:00 | 2657.00 | 2648.12 | 2696.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 2679.00 | 2662.79 | 2688.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:45:00 | 2684.20 | 2662.79 | 2688.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 2701.50 | 2670.53 | 2690.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 2701.50 | 2670.53 | 2690.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 2697.90 | 2676.00 | 2690.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 2705.60 | 2676.00 | 2690.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 2709.90 | 2682.78 | 2692.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 2718.70 | 2682.78 | 2692.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 2714.80 | 2690.50 | 2694.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:45:00 | 2717.00 | 2690.50 | 2694.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 2726.70 | 2697.74 | 2697.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 2731.00 | 2704.39 | 2700.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 2694.10 | 2710.51 | 2705.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 2694.10 | 2710.51 | 2705.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 2694.10 | 2710.51 | 2705.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 2697.70 | 2710.51 | 2705.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 2685.10 | 2705.43 | 2703.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 2685.10 | 2705.43 | 2703.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — SELL (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 12:15:00 | 2673.10 | 2697.10 | 2699.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 13:15:00 | 2653.10 | 2688.30 | 2695.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 2650.40 | 2621.36 | 2643.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 2650.40 | 2621.36 | 2643.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 2650.40 | 2621.36 | 2643.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:45:00 | 2670.70 | 2621.36 | 2643.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 2649.10 | 2626.91 | 2644.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:15:00 | 2653.30 | 2626.91 | 2644.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 2654.80 | 2632.49 | 2645.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:45:00 | 2659.00 | 2632.49 | 2645.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 2642.40 | 2634.47 | 2644.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 13:30:00 | 2635.00 | 2634.58 | 2643.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 2663.70 | 2638.84 | 2643.45 | SL hit (close>static) qty=1.00 sl=2656.80 alert=retest2 |

### Cycle 164 — BUY (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 10:15:00 | 2682.10 | 2647.49 | 2646.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 11:15:00 | 2690.50 | 2656.09 | 2650.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 14:15:00 | 2686.20 | 2690.41 | 2677.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-19 15:00:00 | 2686.20 | 2690.41 | 2677.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 2680.00 | 2686.66 | 2677.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:30:00 | 2682.90 | 2686.66 | 2677.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 2697.00 | 2688.73 | 2679.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 11:15:00 | 2703.50 | 2688.73 | 2679.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:00:00 | 2703.50 | 2729.40 | 2724.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 13:00:00 | 2707.70 | 2725.06 | 2722.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 13:15:00 | 2691.30 | 2718.31 | 2719.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 2691.30 | 2718.31 | 2719.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 2674.40 | 2709.52 | 2715.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 2631.60 | 2608.64 | 2629.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 2631.60 | 2608.64 | 2629.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2631.60 | 2608.64 | 2629.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:15:00 | 2642.90 | 2608.64 | 2629.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 2634.10 | 2613.73 | 2629.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 2634.10 | 2613.73 | 2629.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 2639.50 | 2618.89 | 2630.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 2639.50 | 2618.89 | 2630.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 2641.60 | 2623.43 | 2631.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:45:00 | 2644.00 | 2623.43 | 2631.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 2636.10 | 2625.96 | 2632.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:30:00 | 2638.20 | 2625.96 | 2632.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 2637.20 | 2629.18 | 2632.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 2627.00 | 2629.18 | 2632.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 2618.40 | 2627.02 | 2631.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:00:00 | 2611.80 | 2622.65 | 2628.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 2657.70 | 2629.87 | 2629.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 2657.70 | 2629.87 | 2629.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 2665.90 | 2640.91 | 2634.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 2633.60 | 2651.47 | 2643.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 2633.60 | 2651.47 | 2643.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 2633.60 | 2651.47 | 2643.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 2620.00 | 2651.47 | 2643.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 2689.70 | 2659.12 | 2647.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 13:00:00 | 2700.00 | 2671.27 | 2655.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 2700.80 | 2672.33 | 2659.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:45:00 | 2694.00 | 2686.98 | 2672.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 14:30:00 | 2693.80 | 2690.30 | 2676.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 2720.30 | 2730.84 | 2714.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 2720.30 | 2730.84 | 2714.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 2719.20 | 2728.51 | 2715.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:45:00 | 2708.10 | 2728.51 | 2715.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 2703.50 | 2723.51 | 2714.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:30:00 | 2709.30 | 2723.51 | 2714.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 2696.00 | 2718.01 | 2712.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 2696.00 | 2718.01 | 2712.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 2697.00 | 2713.80 | 2711.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 2676.80 | 2713.80 | 2711.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 2701.90 | 2711.12 | 2710.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:00:00 | 2701.90 | 2711.12 | 2710.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-09 11:15:00 | 2687.20 | 2706.34 | 2708.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 2687.20 | 2706.34 | 2708.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 2672.50 | 2699.57 | 2705.04 | Break + close below crossover candle low |

### Cycle 168 — BUY (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 09:15:00 | 2777.20 | 2704.00 | 2703.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 15:15:00 | 2796.10 | 2758.33 | 2734.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 09:15:00 | 2811.00 | 2816.48 | 2786.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 10:00:00 | 2811.00 | 2816.48 | 2786.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 2823.60 | 2843.53 | 2822.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:00:00 | 2823.60 | 2843.53 | 2822.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 2815.10 | 2837.85 | 2821.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:15:00 | 2805.30 | 2837.85 | 2821.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 2812.70 | 2832.82 | 2820.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 2810.30 | 2832.82 | 2820.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 2747.70 | 2811.15 | 2812.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 2734.00 | 2795.72 | 2805.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 2672.20 | 2663.23 | 2700.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 13:00:00 | 2672.20 | 2663.23 | 2700.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 2713.90 | 2659.89 | 2685.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 2711.80 | 2659.89 | 2685.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 2690.00 | 2665.91 | 2685.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 2685.00 | 2673.51 | 2687.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 15:15:00 | 2709.50 | 2694.15 | 2694.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 2709.50 | 2694.15 | 2694.06 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 2685.10 | 2694.47 | 2694.74 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 2729.90 | 2698.35 | 2696.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 10:15:00 | 2736.50 | 2705.98 | 2699.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 13:15:00 | 2815.20 | 2841.15 | 2819.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 13:15:00 | 2815.20 | 2841.15 | 2819.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 2815.20 | 2841.15 | 2819.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:00:00 | 2815.20 | 2841.15 | 2819.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 2811.60 | 2835.24 | 2818.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:30:00 | 2806.60 | 2835.24 | 2818.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 2775.80 | 2823.35 | 2814.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 2669.00 | 2823.35 | 2814.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 2718.00 | 2802.28 | 2805.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 2550.00 | 2743.30 | 2777.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 2653.80 | 2647.60 | 2709.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:45:00 | 2671.10 | 2647.60 | 2709.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 2696.70 | 2660.99 | 2692.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 2696.70 | 2660.99 | 2692.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 2706.00 | 2669.99 | 2693.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 2818.00 | 2669.99 | 2693.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 2848.30 | 2728.06 | 2717.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 14:15:00 | 2867.90 | 2806.41 | 2762.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 2805.50 | 2821.25 | 2781.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 11:00:00 | 2805.50 | 2821.25 | 2781.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 2855.10 | 2874.75 | 2851.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 2855.10 | 2874.75 | 2851.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 2862.90 | 2872.38 | 2852.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 2842.50 | 2872.38 | 2852.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 3038.90 | 3127.61 | 3115.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:45:00 | 3041.90 | 3127.61 | 3115.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 3040.60 | 3110.21 | 3108.82 | EMA400 retest candle locked (from upside) |

### Cycle 175 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 11:15:00 | 3051.00 | 3098.37 | 3103.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 3028.40 | 3069.62 | 3087.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 09:15:00 | 2798.40 | 2773.06 | 2843.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 09:30:00 | 2803.00 | 2773.06 | 2843.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 2787.00 | 2802.18 | 2827.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:15:00 | 2784.40 | 2802.18 | 2827.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 2760.90 | 2749.51 | 2749.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 2760.90 | 2749.51 | 2749.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 2806.80 | 2769.62 | 2759.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 15:15:00 | 2774.00 | 2790.43 | 2777.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 15:15:00 | 2774.00 | 2790.43 | 2777.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 2774.00 | 2790.43 | 2777.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 2769.70 | 2790.43 | 2777.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 2733.10 | 2778.96 | 2773.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 2733.10 | 2778.96 | 2773.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 2755.50 | 2774.27 | 2771.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 12:00:00 | 2757.00 | 2770.81 | 2770.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 12:15:00 | 2717.90 | 2760.23 | 2765.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 12:15:00 | 2717.90 | 2760.23 | 2765.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 2705.60 | 2742.16 | 2755.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 13:15:00 | 2617.10 | 2607.51 | 2652.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 14:00:00 | 2617.10 | 2607.51 | 2652.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 2720.10 | 2636.59 | 2655.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:00:00 | 2720.10 | 2636.59 | 2655.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 11:15:00 | 2728.30 | 2669.77 | 2668.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 12:15:00 | 2742.80 | 2684.37 | 2674.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 15:15:00 | 2750.90 | 2751.33 | 2727.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 09:15:00 | 2669.60 | 2751.33 | 2727.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 2724.20 | 2745.91 | 2726.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 13:30:00 | 2737.00 | 2734.33 | 2726.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 14:15:00 | 2740.40 | 2734.33 | 2726.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 14:15:00 | 2797.90 | 2824.36 | 2826.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — SELL (started 2026-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 14:15:00 | 2797.90 | 2824.36 | 2826.24 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 11:15:00 | 2852.00 | 2827.65 | 2826.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 09:15:00 | 2926.00 | 2859.23 | 2842.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 2966.10 | 2976.89 | 2942.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 10:15:00 | 2942.70 | 2970.05 | 2942.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 2942.70 | 2970.05 | 2942.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 2942.70 | 2970.05 | 2942.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 2933.20 | 2962.68 | 2941.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:45:00 | 2929.60 | 2962.68 | 2941.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 2916.40 | 2953.42 | 2939.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:00:00 | 2916.40 | 2953.42 | 2939.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 2908.10 | 2944.36 | 2936.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 14:00:00 | 2908.10 | 2944.36 | 2936.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 2899.00 | 2926.73 | 2929.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 09:15:00 | 2836.00 | 2908.59 | 2920.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 2752.20 | 2742.24 | 2791.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 2752.20 | 2742.24 | 2791.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 2752.20 | 2742.24 | 2791.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 2747.40 | 2742.24 | 2791.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 13:15:00 | 2802.10 | 2763.37 | 2786.25 | SL hit (close>static) qty=1.00 sl=2800.00 alert=retest2 |

### Cycle 182 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 2833.60 | 2799.37 | 2797.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 2864.00 | 2812.30 | 2803.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 10:15:00 | 2829.00 | 2851.90 | 2832.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 10:15:00 | 2829.00 | 2851.90 | 2832.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 2829.00 | 2851.90 | 2832.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 2829.00 | 2851.90 | 2832.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 2806.70 | 2842.86 | 2830.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 2806.70 | 2842.86 | 2830.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 2823.60 | 2839.01 | 2829.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:30:00 | 2824.30 | 2836.49 | 2829.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 14:15:00 | 2778.00 | 2824.79 | 2824.78 | SL hit (close<static) qty=1.00 sl=2796.80 alert=retest2 |

### Cycle 183 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 2775.00 | 2814.83 | 2820.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 2731.80 | 2798.22 | 2812.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 2859.00 | 2752.72 | 2773.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 2859.00 | 2752.72 | 2773.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 2859.00 | 2752.72 | 2773.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 2856.20 | 2752.72 | 2773.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 2876.10 | 2793.17 | 2789.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 2893.90 | 2813.31 | 2798.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 2785.90 | 2830.29 | 2813.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 2785.90 | 2830.29 | 2813.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 2785.90 | 2830.29 | 2813.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:45:00 | 2779.80 | 2830.29 | 2813.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 2787.00 | 2821.63 | 2811.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:45:00 | 2786.00 | 2821.63 | 2811.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 12:15:00 | 2811.90 | 2817.90 | 2811.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:15:00 | 2829.90 | 2817.90 | 2811.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 09:15:00 | 3112.89 | 2989.58 | 2938.94 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 12:15:00 | 3480.10 | 3489.18 | 3490.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 13:15:00 | 3461.70 | 3483.69 | 3487.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 3447.00 | 3443.27 | 3458.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 3476.20 | 3443.27 | 3458.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 3512.30 | 3457.07 | 3463.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 3512.30 | 3457.07 | 3463.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 3517.00 | 3469.06 | 3468.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 3525.30 | 3487.64 | 3477.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 15:15:00 | 3629.00 | 3633.49 | 3596.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 09:15:00 | 3619.60 | 3633.49 | 3596.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 3577.30 | 3622.25 | 3594.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 14:30:00 | 3654.70 | 3622.29 | 3604.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 3717.10 | 3624.50 | 3606.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-30 14:00:00 | 179.75 | 2023-06-06 12:15:00 | 187.67 | STOP_HIT | 1.00 | 4.41% |
| BUY | retest2 | 2023-05-30 14:30:00 | 179.90 | 2023-06-06 12:15:00 | 187.67 | STOP_HIT | 1.00 | 4.32% |
| SELL | retest2 | 2023-06-09 09:15:00 | 185.45 | 2023-06-12 11:15:00 | 190.32 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest1 | 2023-06-16 09:15:00 | 192.20 | 2023-06-16 13:15:00 | 190.17 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2023-06-19 09:30:00 | 191.48 | 2023-06-19 10:15:00 | 190.02 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2023-06-19 10:00:00 | 191.88 | 2023-06-19 10:15:00 | 190.02 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2023-06-26 11:45:00 | 199.20 | 2023-07-03 13:15:00 | 219.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-26 12:15:00 | 199.23 | 2023-07-03 13:15:00 | 219.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-26 14:45:00 | 199.47 | 2023-07-03 13:15:00 | 219.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-26 15:15:00 | 199.05 | 2023-07-03 13:15:00 | 218.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-27 12:15:00 | 200.57 | 2023-07-03 13:15:00 | 220.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-27 14:00:00 | 200.80 | 2023-07-03 13:15:00 | 220.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-28 09:15:00 | 205.92 | 2023-07-04 09:15:00 | 226.51 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-07-25 12:15:00 | 245.60 | 2023-07-25 15:15:00 | 246.10 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2023-07-28 12:00:00 | 263.27 | 2023-08-02 10:15:00 | 289.60 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-08-17 13:45:00 | 287.10 | 2023-08-21 10:15:00 | 291.68 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2023-08-18 10:00:00 | 287.00 | 2023-08-21 10:15:00 | 291.68 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2023-08-18 10:30:00 | 288.00 | 2023-08-21 10:15:00 | 291.68 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2023-08-18 15:00:00 | 287.83 | 2023-08-21 10:15:00 | 291.68 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2023-09-06 14:15:00 | 406.18 | 2023-09-08 15:15:00 | 446.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-09-15 10:45:00 | 428.85 | 2023-09-20 09:15:00 | 407.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-15 11:30:00 | 428.90 | 2023-09-20 09:15:00 | 407.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-15 12:30:00 | 429.00 | 2023-09-20 09:15:00 | 407.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-15 15:00:00 | 426.17 | 2023-09-20 09:15:00 | 404.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-18 15:15:00 | 413.33 | 2023-09-20 10:15:00 | 392.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-15 10:45:00 | 428.85 | 2023-09-22 09:15:00 | 385.97 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-09-15 11:30:00 | 428.90 | 2023-09-22 09:15:00 | 386.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-09-15 12:30:00 | 429.00 | 2023-09-22 09:15:00 | 386.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-09-15 15:00:00 | 426.17 | 2023-09-22 10:15:00 | 394.53 | STOP_HIT | 0.50 | 7.42% |
| SELL | retest2 | 2023-09-18 15:15:00 | 413.33 | 2023-09-22 10:15:00 | 394.53 | STOP_HIT | 0.50 | 4.55% |
| BUY | retest2 | 2023-09-29 09:15:00 | 439.97 | 2023-09-29 10:15:00 | 428.03 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2023-09-29 12:00:00 | 435.03 | 2023-10-03 09:15:00 | 428.97 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2023-09-29 12:30:00 | 435.65 | 2023-10-03 09:15:00 | 428.97 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2023-09-29 13:15:00 | 434.60 | 2023-10-03 09:15:00 | 428.97 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2023-10-04 09:30:00 | 437.82 | 2023-10-04 12:15:00 | 424.40 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2023-10-04 10:15:00 | 436.93 | 2023-10-04 12:15:00 | 424.40 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2023-10-12 15:15:00 | 492.67 | 2023-10-18 12:15:00 | 491.70 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2023-10-13 09:45:00 | 491.83 | 2023-10-18 12:15:00 | 491.70 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2023-10-13 10:30:00 | 492.02 | 2023-10-18 12:15:00 | 491.70 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2023-10-17 11:00:00 | 492.47 | 2023-10-18 12:15:00 | 491.70 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2023-10-19 14:15:00 | 504.75 | 2023-10-23 09:15:00 | 555.23 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-11-02 14:00:00 | 586.57 | 2023-11-06 14:15:00 | 622.20 | STOP_HIT | 1.00 | -6.07% |
| SELL | retest2 | 2023-11-06 11:00:00 | 594.33 | 2023-11-06 14:15:00 | 622.20 | STOP_HIT | 1.00 | -4.69% |
| SELL | retest2 | 2023-11-06 12:00:00 | 594.95 | 2023-11-06 14:15:00 | 622.20 | STOP_HIT | 1.00 | -4.58% |
| BUY | retest2 | 2023-11-12 18:15:00 | 705.03 | 2023-11-13 14:15:00 | 775.53 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-12-14 10:15:00 | 790.68 | 2023-12-18 13:15:00 | 790.00 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2023-12-14 11:30:00 | 792.00 | 2023-12-18 13:15:00 | 790.00 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2023-12-14 12:45:00 | 792.45 | 2023-12-18 13:15:00 | 790.00 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2023-12-18 09:45:00 | 791.33 | 2023-12-18 13:15:00 | 790.00 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2024-01-02 10:30:00 | 734.38 | 2024-01-03 10:15:00 | 754.67 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2024-01-09 09:15:00 | 766.65 | 2024-01-10 11:15:00 | 756.65 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-01-09 10:15:00 | 761.67 | 2024-01-10 11:15:00 | 756.65 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-01-10 11:15:00 | 760.82 | 2024-01-10 11:15:00 | 756.65 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-01-12 11:15:00 | 744.47 | 2024-01-16 09:15:00 | 766.27 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2024-01-12 15:15:00 | 744.33 | 2024-01-16 09:15:00 | 766.27 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2024-01-15 13:30:00 | 744.72 | 2024-01-16 09:15:00 | 766.27 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2024-01-19 09:15:00 | 771.33 | 2024-01-19 10:15:00 | 757.00 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-02-15 10:45:00 | 779.52 | 2024-02-21 14:15:00 | 740.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-19 13:15:00 | 778.67 | 2024-02-21 15:15:00 | 739.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-15 10:45:00 | 779.52 | 2024-02-23 09:15:00 | 745.40 | STOP_HIT | 0.50 | 4.38% |
| SELL | retest2 | 2024-02-19 13:15:00 | 778.67 | 2024-02-23 09:15:00 | 745.40 | STOP_HIT | 0.50 | 4.27% |
| BUY | retest2 | 2024-02-28 15:00:00 | 783.33 | 2024-03-04 12:15:00 | 771.32 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-02-29 12:30:00 | 779.73 | 2024-03-04 12:15:00 | 771.32 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-03-04 10:00:00 | 777.40 | 2024-03-04 12:15:00 | 771.32 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-03-04 11:30:00 | 776.02 | 2024-03-04 12:15:00 | 771.32 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-03-07 09:15:00 | 761.22 | 2024-03-12 09:15:00 | 723.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 09:45:00 | 761.18 | 2024-03-12 09:15:00 | 723.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 09:15:00 | 761.22 | 2024-03-12 15:15:00 | 728.33 | STOP_HIT | 0.50 | 4.32% |
| SELL | retest2 | 2024-03-07 09:45:00 | 761.18 | 2024-03-12 15:15:00 | 728.33 | STOP_HIT | 0.50 | 4.32% |
| BUY | retest2 | 2024-04-15 11:30:00 | 953.60 | 2024-04-15 13:15:00 | 939.82 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-04-15 13:00:00 | 951.05 | 2024-04-15 13:15:00 | 939.82 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-05-14 13:45:00 | 884.15 | 2024-05-14 15:15:00 | 901.67 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-05-18 09:15:00 | 934.00 | 2024-05-21 09:15:00 | 915.67 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-06-07 13:30:00 | 894.42 | 2024-06-19 11:15:00 | 903.68 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2024-06-10 11:15:00 | 890.47 | 2024-06-19 11:15:00 | 903.68 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest2 | 2024-06-11 09:15:00 | 893.67 | 2024-06-19 11:15:00 | 903.68 | STOP_HIT | 1.00 | 1.12% |
| BUY | retest2 | 2024-06-12 09:15:00 | 892.72 | 2024-06-19 11:15:00 | 903.68 | STOP_HIT | 1.00 | 1.23% |
| BUY | retest2 | 2024-06-14 14:30:00 | 923.97 | 2024-06-19 11:15:00 | 903.68 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2024-07-09 12:30:00 | 789.67 | 2024-07-10 12:15:00 | 750.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-09 12:30:00 | 789.67 | 2024-07-11 09:15:00 | 780.00 | STOP_HIT | 0.50 | 1.22% |
| BUY | retest2 | 2024-07-29 13:15:00 | 817.12 | 2024-07-30 14:15:00 | 804.57 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-08-13 09:30:00 | 860.58 | 2024-08-13 10:15:00 | 855.30 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-08-23 11:45:00 | 916.68 | 2024-08-27 09:15:00 | 896.55 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-08-23 15:15:00 | 915.28 | 2024-08-27 09:15:00 | 896.55 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2024-09-16 09:15:00 | 1017.07 | 2024-09-16 12:15:00 | 1118.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-04 11:00:00 | 1365.48 | 2024-10-07 14:15:00 | 1270.33 | STOP_HIT | 1.00 | -6.97% |
| SELL | retest2 | 2024-10-18 14:30:00 | 1449.32 | 2024-10-22 11:15:00 | 1376.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 10:30:00 | 1449.05 | 2024-10-22 11:15:00 | 1376.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:45:00 | 1446.67 | 2024-10-22 11:15:00 | 1374.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 13:30:00 | 1450.55 | 2024-10-22 11:15:00 | 1378.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 14:30:00 | 1449.32 | 2024-10-23 09:15:00 | 1422.02 | STOP_HIT | 0.50 | 1.88% |
| SELL | retest2 | 2024-10-21 10:30:00 | 1449.05 | 2024-10-23 09:15:00 | 1422.02 | STOP_HIT | 0.50 | 1.87% |
| SELL | retest2 | 2024-10-21 12:45:00 | 1446.67 | 2024-10-23 09:15:00 | 1422.02 | STOP_HIT | 0.50 | 1.70% |
| SELL | retest2 | 2024-10-21 13:30:00 | 1450.55 | 2024-10-23 09:15:00 | 1422.02 | STOP_HIT | 0.50 | 1.97% |
| SELL | retest2 | 2024-10-28 13:00:00 | 1389.72 | 2024-10-29 14:15:00 | 1430.78 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2024-10-28 14:00:00 | 1389.47 | 2024-10-29 14:15:00 | 1430.78 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2024-10-29 10:30:00 | 1389.60 | 2024-10-29 14:15:00 | 1430.78 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2024-11-05 13:30:00 | 1535.28 | 2024-11-11 12:15:00 | 1570.77 | STOP_HIT | 1.00 | 2.31% |
| BUY | retest2 | 2024-11-05 14:30:00 | 1522.55 | 2024-11-11 12:15:00 | 1570.77 | STOP_HIT | 1.00 | 3.17% |
| SELL | retest2 | 2024-11-12 13:15:00 | 1549.12 | 2024-11-18 12:15:00 | 1565.20 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-11-13 09:15:00 | 1545.00 | 2024-11-18 12:15:00 | 1565.20 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-11-21 09:15:00 | 1599.92 | 2024-11-21 10:15:00 | 1563.00 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2024-11-22 09:15:00 | 1586.87 | 2024-11-22 09:15:00 | 1558.15 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-12-13 13:00:00 | 1874.50 | 2024-12-18 14:15:00 | 1874.17 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2024-12-13 14:00:00 | 1873.43 | 2024-12-18 14:15:00 | 1874.17 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2024-12-18 13:00:00 | 1872.50 | 2024-12-18 14:15:00 | 1874.17 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2024-12-27 12:15:00 | 1769.62 | 2025-01-01 09:15:00 | 1789.65 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-12-27 12:45:00 | 1768.95 | 2025-01-01 09:15:00 | 1789.65 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-12-27 14:45:00 | 1769.97 | 2025-01-01 11:15:00 | 1796.87 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-12-30 10:15:00 | 1768.10 | 2025-01-01 11:15:00 | 1796.87 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-12-31 09:30:00 | 1758.22 | 2025-01-01 11:15:00 | 1796.87 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2024-12-31 11:00:00 | 1762.20 | 2025-01-01 11:15:00 | 1796.87 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-01-13 14:15:00 | 1709.67 | 2025-01-14 09:15:00 | 1786.77 | STOP_HIT | 1.00 | -4.51% |
| SELL | retest2 | 2025-01-13 15:00:00 | 1721.25 | 2025-01-14 09:15:00 | 1786.77 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2025-01-30 10:45:00 | 1758.10 | 2025-02-01 10:15:00 | 1793.67 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-01-30 11:30:00 | 1751.65 | 2025-02-01 10:15:00 | 1793.67 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-01-31 11:45:00 | 1759.45 | 2025-02-01 10:15:00 | 1793.67 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-01-31 12:15:00 | 1759.23 | 2025-02-01 10:15:00 | 1793.67 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-02-07 11:15:00 | 1895.32 | 2025-02-11 09:15:00 | 1799.52 | STOP_HIT | 1.00 | -5.05% |
| BUY | retest2 | 2025-02-07 12:30:00 | 1897.43 | 2025-02-11 09:15:00 | 1799.52 | STOP_HIT | 1.00 | -5.16% |
| BUY | retest2 | 2025-02-07 14:30:00 | 1902.72 | 2025-02-11 09:15:00 | 1799.52 | STOP_HIT | 1.00 | -5.42% |
| SELL | retest2 | 2025-02-12 14:15:00 | 1766.70 | 2025-02-17 09:15:00 | 1678.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 09:15:00 | 1752.37 | 2025-02-17 09:15:00 | 1664.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 10:15:00 | 1769.08 | 2025-02-17 09:15:00 | 1680.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:00:00 | 1769.33 | 2025-02-17 09:15:00 | 1680.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 10:15:00 | 1750.45 | 2025-02-17 09:15:00 | 1662.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 14:15:00 | 1766.70 | 2025-02-17 12:15:00 | 1725.10 | STOP_HIT | 0.50 | 2.35% |
| SELL | retest2 | 2025-02-13 09:15:00 | 1752.37 | 2025-02-17 12:15:00 | 1725.10 | STOP_HIT | 0.50 | 1.56% |
| SELL | retest2 | 2025-02-13 10:15:00 | 1769.08 | 2025-02-17 12:15:00 | 1725.10 | STOP_HIT | 0.50 | 2.49% |
| SELL | retest2 | 2025-02-13 13:00:00 | 1769.33 | 2025-02-17 12:15:00 | 1725.10 | STOP_HIT | 0.50 | 2.50% |
| SELL | retest2 | 2025-02-14 10:15:00 | 1750.45 | 2025-02-17 12:15:00 | 1725.10 | STOP_HIT | 0.50 | 1.45% |
| BUY | retest2 | 2025-02-24 11:30:00 | 1889.98 | 2025-02-24 13:15:00 | 1866.63 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-03-13 09:15:00 | 1311.80 | 2025-03-18 09:15:00 | 1341.73 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-03-13 13:00:00 | 1318.98 | 2025-03-18 09:15:00 | 1341.73 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-03-13 15:00:00 | 1308.78 | 2025-03-18 09:15:00 | 1341.73 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-03-17 09:30:00 | 1314.95 | 2025-03-18 10:15:00 | 1342.98 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-03-17 11:15:00 | 1318.63 | 2025-03-18 10:15:00 | 1342.98 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-03-17 11:45:00 | 1317.08 | 2025-03-18 10:15:00 | 1342.98 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-03-17 14:15:00 | 1319.67 | 2025-03-18 10:15:00 | 1342.98 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-03-27 11:15:00 | 1544.55 | 2025-03-27 11:15:00 | 1552.30 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-04-04 11:15:00 | 1857.63 | 2025-04-07 09:15:00 | 1671.67 | STOP_HIT | 1.00 | -10.01% |
| BUY | retest2 | 2025-05-16 09:15:00 | 2422.33 | 2025-05-21 12:15:00 | 2424.33 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-05-29 10:00:00 | 2396.00 | 2025-05-29 13:15:00 | 2443.00 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-06-26 12:45:00 | 2793.70 | 2025-06-30 11:15:00 | 2778.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-06-26 13:30:00 | 2791.40 | 2025-06-30 11:15:00 | 2778.00 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-07-02 10:30:00 | 2757.40 | 2025-07-03 11:15:00 | 2787.90 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-07-02 12:45:00 | 2760.40 | 2025-07-03 11:15:00 | 2787.90 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-07-23 14:45:00 | 2549.10 | 2025-07-25 09:15:00 | 2479.70 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2025-07-24 09:30:00 | 2539.70 | 2025-07-25 09:15:00 | 2479.70 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-07-24 12:00:00 | 2545.10 | 2025-07-25 09:15:00 | 2479.70 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-07-28 11:30:00 | 2476.50 | 2025-08-04 14:15:00 | 2485.00 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-07-29 15:15:00 | 2485.00 | 2025-08-04 14:15:00 | 2485.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-08-06 15:00:00 | 2382.20 | 2025-08-07 14:15:00 | 2432.20 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-08-07 13:15:00 | 2378.40 | 2025-08-07 14:15:00 | 2432.20 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-08-11 14:15:00 | 2402.20 | 2025-08-13 09:15:00 | 2456.00 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-08-20 12:15:00 | 2518.80 | 2025-08-21 11:15:00 | 2399.90 | STOP_HIT | 1.00 | -4.72% |
| BUY | retest2 | 2025-08-21 10:15:00 | 2520.00 | 2025-08-21 11:15:00 | 2399.90 | STOP_HIT | 1.00 | -4.77% |
| SELL | retest2 | 2025-09-01 11:45:00 | 2165.50 | 2025-09-02 09:15:00 | 2235.80 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2025-09-01 12:45:00 | 2168.20 | 2025-09-02 09:15:00 | 2235.80 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-09-02 09:15:00 | 2169.00 | 2025-09-02 09:15:00 | 2235.80 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-09-16 09:15:00 | 2193.00 | 2025-09-23 09:15:00 | 2102.35 | PARTIAL | 0.50 | 4.13% |
| SELL | retest2 | 2025-09-17 11:30:00 | 2213.00 | 2025-09-23 09:15:00 | 2104.72 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2025-09-17 13:00:00 | 2215.50 | 2025-09-23 09:15:00 | 2103.87 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2025-09-17 13:45:00 | 2214.60 | 2025-09-24 15:15:00 | 2083.35 | PARTIAL | 0.50 | 5.93% |
| SELL | retest2 | 2025-09-16 09:15:00 | 2193.00 | 2025-09-26 09:15:00 | 2064.80 | STOP_HIT | 0.50 | 5.85% |
| SELL | retest2 | 2025-09-17 11:30:00 | 2213.00 | 2025-09-26 09:15:00 | 2064.80 | STOP_HIT | 0.50 | 6.70% |
| SELL | retest2 | 2025-09-17 13:00:00 | 2215.50 | 2025-09-26 09:15:00 | 2064.80 | STOP_HIT | 0.50 | 6.80% |
| SELL | retest2 | 2025-09-17 13:45:00 | 2214.60 | 2025-09-26 09:15:00 | 2064.80 | STOP_HIT | 0.50 | 6.76% |
| SELL | retest2 | 2025-09-29 11:30:00 | 2087.10 | 2025-09-29 13:15:00 | 2096.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-10-01 11:15:00 | 2041.00 | 2025-10-01 13:15:00 | 2074.40 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-10-20 09:15:00 | 2506.60 | 2025-10-24 11:15:00 | 2495.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-10-23 09:15:00 | 2497.10 | 2025-10-24 11:15:00 | 2495.00 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-10-24 09:30:00 | 2500.50 | 2025-10-24 11:15:00 | 2495.00 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-10-24 11:00:00 | 2493.40 | 2025-10-24 11:15:00 | 2495.00 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-10-30 12:15:00 | 2447.30 | 2025-10-31 12:15:00 | 2471.80 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-11-14 12:00:00 | 2816.00 | 2025-11-24 09:15:00 | 2847.10 | STOP_HIT | 1.00 | 1.10% |
| BUY | retest2 | 2025-11-14 14:45:00 | 2825.20 | 2025-11-24 09:15:00 | 2847.10 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2025-11-17 12:45:00 | 2817.70 | 2025-11-24 09:15:00 | 2847.10 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2025-11-18 09:15:00 | 2841.00 | 2025-11-24 09:15:00 | 2847.10 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest2 | 2025-11-19 12:00:00 | 2880.00 | 2025-11-24 09:15:00 | 2847.10 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-12-17 13:30:00 | 2635.00 | 2025-12-18 09:15:00 | 2663.70 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-12-22 11:15:00 | 2703.50 | 2025-12-24 13:15:00 | 2691.30 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-12-24 12:00:00 | 2703.50 | 2025-12-24 13:15:00 | 2691.30 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-12-24 13:00:00 | 2707.70 | 2025-12-24 13:15:00 | 2691.30 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-01-01 12:00:00 | 2611.80 | 2026-01-02 09:15:00 | 2657.70 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-01-05 13:00:00 | 2700.00 | 2026-01-09 11:15:00 | 2687.20 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2026-01-06 09:15:00 | 2700.80 | 2026-01-09 11:15:00 | 2687.20 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2026-01-06 12:45:00 | 2694.00 | 2026-01-09 11:15:00 | 2687.20 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2026-01-06 14:30:00 | 2693.80 | 2026-01-09 11:15:00 | 2687.20 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2026-01-22 11:30:00 | 2685.00 | 2026-01-22 15:15:00 | 2709.50 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-02-19 10:15:00 | 2784.40 | 2026-02-25 10:15:00 | 2760.90 | STOP_HIT | 1.00 | 0.84% |
| BUY | retest2 | 2026-02-27 12:00:00 | 2757.00 | 2026-02-27 12:15:00 | 2717.90 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-03-09 13:30:00 | 2737.00 | 2026-03-13 14:15:00 | 2797.90 | STOP_HIT | 1.00 | 2.23% |
| BUY | retest2 | 2026-03-09 14:15:00 | 2740.40 | 2026-03-13 14:15:00 | 2797.90 | STOP_HIT | 1.00 | 2.10% |
| SELL | retest2 | 2026-03-24 10:15:00 | 2747.40 | 2026-03-24 13:15:00 | 2802.10 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2026-03-27 13:30:00 | 2824.30 | 2026-03-27 14:15:00 | 2778.00 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2026-04-02 13:15:00 | 2829.90 | 2026-04-08 09:15:00 | 3112.89 | TARGET_HIT | 1.00 | 10.00% |
