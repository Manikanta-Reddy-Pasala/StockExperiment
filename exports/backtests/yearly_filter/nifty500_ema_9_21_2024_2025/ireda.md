# Indian Renewable Energy Development Agency Ltd. (IREDA)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-11 15:15:00 (3717 bars)
- **Last close:** 130.92
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 142 |
| ALERT1 | 100 |
| ALERT2 | 97 |
| ALERT2_SKIP | 57 |
| ALERT3 | 259 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 110 |
| PARTIAL | 17 |
| TARGET_HIT | 14 |
| STOP_HIT | 96 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 127 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 53 / 74
- **Target hits / Stop hits / Partials:** 14 / 96 / 17
- **Avg / median % per leg:** 1.23% / -0.29%
- **Sum % (uncompounded):** 156.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 55 | 12 | 21.8% | 10 | 45 | 0 | 0.59% | 32.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 55 | 12 | 21.8% | 10 | 45 | 0 | 0.59% | 32.7% |
| SELL (all) | 72 | 41 | 56.9% | 4 | 51 | 17 | 1.71% | 123.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 72 | 41 | 56.9% | 4 | 51 | 17 | 1.71% | 123.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 127 | 53 | 41.7% | 14 | 96 | 17 | 1.23% | 156.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 171.95 | 166.71 | 166.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 15:15:00 | 174.50 | 170.18 | 168.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 10:15:00 | 171.80 | 171.95 | 170.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 11:00:00 | 171.80 | 171.95 | 170.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 170.70 | 171.49 | 170.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:45:00 | 171.40 | 171.49 | 170.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 171.30 | 171.46 | 170.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 174.70 | 171.40 | 170.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-22 09:15:00 | 192.17 | 182.64 | 178.99 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 184.30 | 188.65 | 189.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 182.10 | 184.52 | 185.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 183.90 | 181.47 | 182.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 13:15:00 | 183.90 | 181.47 | 182.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 183.90 | 181.47 | 182.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 183.90 | 181.47 | 182.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 184.65 | 182.10 | 182.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 184.65 | 182.10 | 182.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 185.40 | 182.76 | 183.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 191.25 | 182.76 | 183.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 192.25 | 184.66 | 184.03 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 166.85 | 184.44 | 185.27 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 13:15:00 | 178.00 | 176.18 | 176.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 13:15:00 | 180.75 | 178.07 | 177.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 11:15:00 | 179.23 | 179.44 | 178.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 12:00:00 | 179.23 | 179.44 | 178.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 178.69 | 179.29 | 178.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:30:00 | 179.08 | 179.29 | 178.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 178.55 | 179.15 | 178.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 181.40 | 179.15 | 178.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 15:15:00 | 180.24 | 180.75 | 180.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 15:15:00 | 180.24 | 180.75 | 180.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 09:15:00 | 179.75 | 180.55 | 180.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 178.70 | 177.05 | 177.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 178.70 | 177.05 | 177.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 178.70 | 177.05 | 177.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:30:00 | 180.70 | 177.05 | 177.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 178.08 | 177.26 | 177.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 11:15:00 | 177.59 | 177.26 | 177.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 13:15:00 | 177.77 | 177.52 | 177.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 15:15:00 | 177.50 | 177.52 | 177.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 09:15:00 | 186.80 | 179.37 | 178.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 186.80 | 179.37 | 178.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 14:15:00 | 187.80 | 183.63 | 181.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 15:15:00 | 189.20 | 189.34 | 186.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-25 09:15:00 | 186.84 | 189.34 | 186.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 186.09 | 188.69 | 186.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:00:00 | 186.09 | 188.69 | 186.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 185.40 | 188.03 | 186.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:00:00 | 185.40 | 188.03 | 186.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 185.47 | 187.52 | 186.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:30:00 | 185.32 | 187.52 | 186.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 185.99 | 186.18 | 185.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:00:00 | 185.99 | 186.18 | 185.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 194.61 | 187.86 | 186.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 12:30:00 | 195.98 | 189.81 | 187.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 15:15:00 | 195.00 | 191.99 | 189.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 14:15:00 | 190.27 | 191.69 | 191.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 14:15:00 | 190.27 | 191.69 | 191.75 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 197.18 | 192.68 | 192.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 14:15:00 | 204.02 | 196.52 | 194.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 09:15:00 | 238.00 | 239.26 | 231.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-09 10:00:00 | 238.00 | 239.26 | 231.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 236.75 | 238.76 | 232.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 13:15:00 | 242.35 | 238.24 | 232.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 14:30:00 | 240.98 | 239.26 | 234.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 15:00:00 | 241.13 | 239.26 | 234.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 09:30:00 | 245.40 | 239.28 | 235.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 237.38 | 238.90 | 235.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 234.34 | 238.90 | 235.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Target hit | 2024-07-11 13:15:00 | 266.59 | 253.51 | 245.69 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 14:15:00 | 271.85 | 279.33 | 279.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 263.49 | 275.03 | 277.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 09:15:00 | 266.65 | 263.59 | 268.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 09:15:00 | 266.65 | 263.59 | 268.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 266.65 | 263.59 | 268.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:45:00 | 270.56 | 263.59 | 268.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 268.20 | 264.52 | 268.87 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2024-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 09:15:00 | 273.57 | 270.45 | 270.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 10:15:00 | 279.95 | 273.38 | 271.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 11:15:00 | 271.97 | 273.10 | 271.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 11:15:00 | 271.97 | 273.10 | 271.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 271.97 | 273.10 | 271.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:00:00 | 271.97 | 273.10 | 271.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 261.72 | 270.82 | 270.97 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 15:15:00 | 266.00 | 263.56 | 263.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 09:15:00 | 267.00 | 264.25 | 263.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 13:15:00 | 264.70 | 264.94 | 264.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 13:15:00 | 264.70 | 264.94 | 264.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 264.70 | 264.94 | 264.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 13:30:00 | 264.84 | 264.94 | 264.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 263.50 | 264.65 | 264.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 263.50 | 264.65 | 264.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 264.45 | 264.61 | 264.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 266.22 | 264.61 | 264.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:45:00 | 265.60 | 264.68 | 264.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 13:15:00 | 263.80 | 264.23 | 264.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 13:15:00 | 263.80 | 264.23 | 264.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 14:15:00 | 262.51 | 263.89 | 264.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 245.40 | 243.39 | 248.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 245.40 | 243.39 | 248.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 245.40 | 243.39 | 248.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:15:00 | 243.10 | 243.39 | 248.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 09:15:00 | 250.00 | 245.14 | 244.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 250.00 | 245.14 | 244.89 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 13:15:00 | 245.40 | 246.03 | 246.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 14:15:00 | 243.50 | 245.52 | 245.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 13:15:00 | 246.45 | 243.47 | 244.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 13:15:00 | 246.45 | 243.47 | 244.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 246.45 | 243.47 | 244.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:00:00 | 246.45 | 243.47 | 244.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 247.40 | 244.26 | 244.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:30:00 | 249.35 | 244.26 | 244.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 15:15:00 | 248.30 | 245.07 | 244.96 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 10:15:00 | 244.45 | 244.86 | 244.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 243.15 | 244.32 | 244.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 240.00 | 238.33 | 240.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 240.00 | 238.33 | 240.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 240.00 | 238.33 | 240.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:45:00 | 239.80 | 238.33 | 240.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 239.55 | 238.58 | 240.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:45:00 | 240.15 | 238.58 | 240.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 241.10 | 239.08 | 240.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:00:00 | 241.10 | 239.08 | 240.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 240.30 | 239.32 | 240.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 14:45:00 | 239.90 | 239.86 | 240.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 15:15:00 | 239.80 | 239.86 | 240.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 09:15:00 | 247.00 | 241.28 | 240.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 247.00 | 241.28 | 240.93 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 13:15:00 | 241.10 | 241.84 | 241.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-21 09:15:00 | 239.95 | 241.14 | 241.54 | Break + close below crossover candle low |

### Cycle 21 — BUY (started 2024-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 09:15:00 | 254.40 | 242.80 | 241.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 11:15:00 | 261.55 | 248.78 | 244.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 09:15:00 | 254.35 | 257.73 | 254.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 254.35 | 257.73 | 254.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 254.35 | 257.73 | 254.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:45:00 | 254.40 | 257.73 | 254.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 253.85 | 256.96 | 254.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:45:00 | 253.90 | 256.96 | 254.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 254.20 | 256.41 | 254.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 10:00:00 | 254.75 | 254.76 | 254.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 10:30:00 | 255.50 | 254.84 | 254.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 13:45:00 | 255.45 | 254.72 | 254.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 15:00:00 | 255.00 | 254.77 | 254.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 256.50 | 255.24 | 254.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 12:30:00 | 257.55 | 255.35 | 254.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 15:15:00 | 254.00 | 254.88 | 254.73 | SL hit (close<static) qty=1.00 sl=254.40 alert=retest2 |

### Cycle 22 — SELL (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 15:15:00 | 254.50 | 254.77 | 254.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 09:15:00 | 252.20 | 254.25 | 254.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 14:15:00 | 239.07 | 238.65 | 241.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-03 15:00:00 | 239.07 | 238.65 | 241.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 235.87 | 236.22 | 238.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 10:15:00 | 234.62 | 236.22 | 238.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 11:15:00 | 235.05 | 236.14 | 238.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 14:15:00 | 235.51 | 236.14 | 237.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 222.89 | 228.35 | 231.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 223.30 | 228.35 | 231.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 223.73 | 228.35 | 231.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-09 15:15:00 | 224.78 | 224.23 | 227.84 | SL hit (close>ema200) qty=0.50 sl=224.23 alert=retest2 |

### Cycle 23 — BUY (started 2024-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 13:15:00 | 233.64 | 230.45 | 230.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 15:15:00 | 235.00 | 231.76 | 230.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 12:15:00 | 231.68 | 231.83 | 231.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 13:00:00 | 231.68 | 231.83 | 231.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 230.15 | 231.50 | 231.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 230.15 | 231.50 | 231.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 229.40 | 231.08 | 230.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 229.40 | 231.08 | 230.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 15:15:00 | 228.20 | 230.50 | 230.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 09:15:00 | 225.55 | 229.51 | 230.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 13:15:00 | 228.09 | 228.00 | 229.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-12 14:00:00 | 228.09 | 228.00 | 229.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 232.64 | 228.93 | 229.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 14:45:00 | 232.88 | 228.93 | 229.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 233.20 | 229.78 | 229.77 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 10:15:00 | 228.26 | 230.22 | 230.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 09:15:00 | 226.11 | 228.48 | 229.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 13:15:00 | 232.46 | 228.49 | 228.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 13:15:00 | 232.46 | 228.49 | 228.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 232.46 | 228.49 | 228.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:00:00 | 232.46 | 228.49 | 228.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 228.16 | 228.43 | 228.87 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 11:15:00 | 230.34 | 229.31 | 229.18 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 14:15:00 | 227.40 | 228.83 | 228.98 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 09:15:00 | 232.29 | 229.29 | 229.15 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 12:15:00 | 228.38 | 228.99 | 229.04 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 13:15:00 | 229.76 | 229.14 | 229.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 10:15:00 | 231.36 | 229.88 | 229.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 09:15:00 | 230.44 | 231.21 | 230.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 09:15:00 | 230.44 | 231.21 | 230.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 230.44 | 231.21 | 230.47 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 09:15:00 | 228.02 | 230.09 | 230.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 14:15:00 | 227.16 | 228.53 | 229.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 10:15:00 | 226.01 | 224.03 | 225.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 10:15:00 | 226.01 | 224.03 | 225.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 226.01 | 224.03 | 225.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:45:00 | 227.04 | 224.03 | 225.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 224.26 | 224.08 | 225.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:30:00 | 225.80 | 224.08 | 225.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 13:15:00 | 225.47 | 224.36 | 225.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:45:00 | 226.19 | 224.36 | 225.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 224.90 | 224.47 | 225.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 14:30:00 | 225.50 | 224.47 | 225.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 225.00 | 224.53 | 225.34 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 14:15:00 | 233.74 | 226.51 | 225.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 14:15:00 | 234.90 | 230.16 | 228.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 14:15:00 | 230.65 | 231.36 | 230.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 15:00:00 | 230.65 | 231.36 | 230.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 230.60 | 231.21 | 230.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:15:00 | 227.26 | 231.21 | 230.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 226.28 | 230.22 | 229.71 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 225.30 | 228.60 | 229.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 223.25 | 226.91 | 228.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 218.29 | 215.09 | 218.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 10:15:00 | 218.29 | 215.09 | 218.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 218.29 | 215.09 | 218.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:45:00 | 216.19 | 215.09 | 218.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 220.60 | 216.19 | 218.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:30:00 | 222.39 | 216.19 | 218.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 221.16 | 217.18 | 218.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 221.16 | 217.18 | 218.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 225.11 | 220.64 | 220.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 228.48 | 222.21 | 220.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 11:15:00 | 232.88 | 233.05 | 230.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 11:30:00 | 232.80 | 233.05 | 230.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 230.09 | 232.48 | 230.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 15:00:00 | 230.09 | 232.48 | 230.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 227.40 | 231.46 | 230.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:30:00 | 224.96 | 230.15 | 229.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 10:15:00 | 223.89 | 228.89 | 229.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 12:15:00 | 221.93 | 226.73 | 228.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 12:15:00 | 223.80 | 222.66 | 224.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 12:15:00 | 223.80 | 222.66 | 224.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 223.80 | 222.66 | 224.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:45:00 | 224.11 | 222.66 | 224.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 222.29 | 222.70 | 224.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:45:00 | 223.45 | 222.70 | 224.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 222.99 | 222.77 | 224.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 14:45:00 | 221.40 | 222.14 | 223.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 10:15:00 | 221.00 | 221.88 | 223.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 10:45:00 | 220.95 | 221.53 | 222.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 14:15:00 | 210.33 | 213.13 | 215.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 15:15:00 | 209.95 | 212.50 | 215.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 15:15:00 | 209.90 | 212.50 | 215.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-22 14:15:00 | 199.26 | 204.67 | 209.60 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 09:15:00 | 201.40 | 197.90 | 197.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 10:15:00 | 203.32 | 198.99 | 198.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 12:15:00 | 208.50 | 209.46 | 206.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 13:00:00 | 208.50 | 209.46 | 206.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 203.69 | 209.12 | 207.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 203.69 | 209.12 | 207.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 204.04 | 208.11 | 207.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:45:00 | 205.31 | 207.55 | 207.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 12:15:00 | 205.30 | 207.10 | 207.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 205.30 | 207.10 | 207.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 11:15:00 | 203.39 | 205.32 | 206.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 205.88 | 205.22 | 205.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 205.88 | 205.22 | 205.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 205.88 | 205.22 | 205.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 13:45:00 | 205.62 | 205.22 | 205.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 205.14 | 205.20 | 205.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:30:00 | 205.54 | 205.20 | 205.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 207.79 | 205.53 | 205.89 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 207.88 | 206.27 | 206.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 13:15:00 | 208.40 | 207.00 | 206.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 10:15:00 | 207.95 | 208.13 | 207.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 10:15:00 | 207.95 | 208.13 | 207.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 207.95 | 208.13 | 207.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 207.29 | 208.13 | 207.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 207.40 | 207.98 | 207.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 207.40 | 207.98 | 207.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 207.26 | 207.84 | 207.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:00:00 | 207.26 | 207.84 | 207.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 206.83 | 207.64 | 207.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:15:00 | 206.70 | 207.64 | 207.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 206.17 | 207.34 | 207.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:30:00 | 206.10 | 207.34 | 207.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 15:15:00 | 206.00 | 207.07 | 207.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 203.89 | 206.44 | 206.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 194.51 | 191.60 | 194.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 194.51 | 191.60 | 194.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 194.51 | 191.60 | 194.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 194.51 | 191.60 | 194.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 191.83 | 191.65 | 194.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 11:15:00 | 191.39 | 191.65 | 194.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 12:30:00 | 191.42 | 191.80 | 193.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 13:30:00 | 191.28 | 191.56 | 193.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 09:30:00 | 191.32 | 189.71 | 190.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 192.52 | 190.27 | 191.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 192.86 | 190.27 | 191.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 09:15:00 | 181.82 | 185.80 | 187.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 09:15:00 | 181.85 | 185.80 | 187.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 09:15:00 | 181.72 | 185.80 | 187.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 09:15:00 | 181.75 | 185.80 | 187.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 185.30 | 184.68 | 186.56 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-22 13:15:00 | 185.30 | 184.68 | 186.56 | SL hit (close>ema200) qty=0.50 sl=184.68 alert=retest2 |

### Cycle 41 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 192.23 | 187.72 | 187.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 10:15:00 | 194.69 | 191.54 | 190.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 202.24 | 204.26 | 199.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 10:00:00 | 202.24 | 204.26 | 199.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 204.55 | 204.39 | 202.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 208.74 | 206.04 | 204.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 09:15:00 | 215.70 | 220.39 | 220.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 215.70 | 220.39 | 220.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 09:15:00 | 215.22 | 216.59 | 217.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 14:15:00 | 208.22 | 206.27 | 208.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 14:15:00 | 208.22 | 206.27 | 208.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 208.22 | 206.27 | 208.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:45:00 | 209.30 | 206.27 | 208.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 209.29 | 206.88 | 208.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 207.20 | 206.88 | 208.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 205.89 | 206.68 | 208.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 13:00:00 | 205.18 | 206.61 | 208.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 13:30:00 | 204.16 | 206.04 | 207.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 09:15:00 | 202.40 | 206.17 | 207.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 13:15:00 | 202.71 | 200.78 | 200.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 13:15:00 | 202.71 | 200.78 | 200.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 14:15:00 | 218.87 | 204.40 | 202.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 15:15:00 | 221.57 | 221.79 | 218.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 09:15:00 | 223.71 | 221.79 | 218.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 219.95 | 225.12 | 222.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 219.95 | 225.12 | 222.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 219.60 | 224.01 | 222.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 218.85 | 224.01 | 222.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 216.00 | 221.62 | 221.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 214.64 | 220.23 | 221.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 219.25 | 219.11 | 220.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 10:15:00 | 219.25 | 219.11 | 220.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 219.25 | 219.11 | 220.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 219.25 | 219.11 | 220.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 220.36 | 219.36 | 220.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:30:00 | 221.12 | 219.36 | 220.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 220.58 | 219.60 | 220.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 12:30:00 | 219.65 | 219.60 | 220.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 219.66 | 219.61 | 220.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:30:00 | 219.90 | 219.61 | 220.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 218.72 | 219.43 | 220.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 216.41 | 219.31 | 219.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 09:15:00 | 222.33 | 219.92 | 220.16 | SL hit (close>static) qty=1.00 sl=220.59 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 13:15:00 | 222.82 | 220.61 | 220.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 14:15:00 | 223.81 | 221.25 | 220.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 10:15:00 | 222.05 | 222.25 | 221.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 10:15:00 | 222.05 | 222.25 | 221.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 222.05 | 222.25 | 221.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 11:00:00 | 222.05 | 222.25 | 221.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 221.83 | 222.22 | 221.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 12:45:00 | 221.33 | 222.22 | 221.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 221.35 | 222.05 | 221.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 13:30:00 | 221.01 | 222.05 | 221.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 14:15:00 | 216.54 | 220.94 | 221.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 15:15:00 | 215.10 | 219.78 | 220.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-13 09:15:00 | 211.25 | 208.75 | 213.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-13 10:00:00 | 211.25 | 208.75 | 213.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 207.14 | 205.09 | 207.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:15:00 | 206.24 | 205.09 | 207.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 14:15:00 | 206.29 | 205.41 | 207.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 10:00:00 | 205.39 | 205.70 | 207.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 11:30:00 | 206.21 | 205.96 | 207.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 12:15:00 | 205.78 | 205.92 | 206.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:45:00 | 206.89 | 205.92 | 206.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 206.00 | 205.07 | 206.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 11:15:00 | 205.00 | 205.18 | 206.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 11:45:00 | 204.82 | 205.05 | 205.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 12:45:00 | 205.11 | 205.10 | 205.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 13:15:00 | 205.02 | 205.10 | 205.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 210.56 | 206.06 | 206.08 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-17 09:15:00 | 210.56 | 206.06 | 206.08 | SL hit (close>static) qty=1.00 sl=209.80 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 10:15:00 | 208.67 | 206.58 | 206.31 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 203.40 | 206.27 | 206.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 199.95 | 204.31 | 205.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 187.71 | 185.60 | 189.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 187.71 | 185.60 | 189.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 193.07 | 187.68 | 189.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 193.39 | 187.68 | 189.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 192.45 | 188.64 | 189.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 11:15:00 | 191.95 | 188.64 | 189.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 14:15:00 | 191.80 | 190.31 | 190.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 191.80 | 190.31 | 190.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 196.55 | 191.99 | 190.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 202.07 | 202.19 | 198.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 202.07 | 202.19 | 198.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 202.07 | 202.19 | 198.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 201.00 | 202.19 | 198.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 194.95 | 200.74 | 198.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 194.95 | 200.74 | 198.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 196.90 | 199.97 | 198.39 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 186.80 | 195.73 | 196.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 12:15:00 | 184.35 | 190.58 | 193.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 192.35 | 189.24 | 191.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 192.35 | 189.24 | 191.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 192.35 | 189.24 | 191.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 193.80 | 189.24 | 191.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 191.51 | 189.69 | 191.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:00:00 | 191.51 | 189.69 | 191.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 191.52 | 190.06 | 191.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:30:00 | 191.58 | 190.06 | 191.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 191.88 | 190.42 | 191.90 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 11:15:00 | 193.32 | 192.38 | 192.35 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 09:15:00 | 189.91 | 192.21 | 192.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 188.02 | 189.42 | 190.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 175.30 | 174.79 | 178.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:45:00 | 174.87 | 174.79 | 178.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 180.10 | 175.75 | 177.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 179.83 | 175.75 | 177.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 179.30 | 176.46 | 178.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:30:00 | 177.80 | 177.34 | 178.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 12:15:00 | 168.91 | 172.95 | 175.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 15:15:00 | 166.60 | 166.47 | 169.57 | SL hit (close>ema200) qty=0.50 sl=166.47 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 170.17 | 167.27 | 166.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 172.20 | 168.69 | 167.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 15:15:00 | 176.00 | 177.51 | 174.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 09:15:00 | 173.03 | 177.51 | 174.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 173.78 | 176.77 | 174.56 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2025-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 15:15:00 | 171.80 | 173.54 | 173.67 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 10:15:00 | 174.70 | 173.86 | 173.80 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 169.94 | 173.03 | 173.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 168.52 | 172.13 | 173.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 149.50 | 149.17 | 154.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 10:00:00 | 149.50 | 149.17 | 154.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 150.95 | 147.98 | 149.25 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 09:15:00 | 153.26 | 150.33 | 150.03 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 15:15:00 | 149.40 | 149.94 | 149.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 09:15:00 | 148.00 | 149.55 | 149.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 141.69 | 141.55 | 142.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 140.66 | 140.60 | 141.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 140.66 | 140.60 | 141.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 140.95 | 140.60 | 141.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 144.38 | 140.30 | 140.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 144.38 | 140.30 | 140.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 143.56 | 140.95 | 141.13 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 144.23 | 141.61 | 141.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 13:15:00 | 144.65 | 142.57 | 141.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 149.55 | 149.82 | 147.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 149.55 | 149.82 | 147.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 164.60 | 167.61 | 165.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 14:00:00 | 164.60 | 167.61 | 165.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 163.14 | 166.71 | 165.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 14:45:00 | 163.18 | 166.71 | 165.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 165.36 | 165.70 | 165.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:45:00 | 164.85 | 165.70 | 165.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 163.96 | 165.35 | 165.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:00:00 | 163.96 | 165.35 | 165.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 163.45 | 164.97 | 164.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:30:00 | 164.91 | 164.97 | 164.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 165.54 | 165.09 | 165.00 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 14:15:00 | 161.64 | 164.40 | 164.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 14:15:00 | 160.90 | 162.42 | 163.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 12:15:00 | 160.85 | 160.83 | 162.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-01 12:45:00 | 160.65 | 160.83 | 162.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 158.40 | 160.28 | 161.45 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 162.18 | 161.42 | 161.39 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 157.10 | 160.76 | 161.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 156.31 | 158.87 | 160.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 151.47 | 150.41 | 153.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 155.80 | 150.41 | 153.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 153.84 | 151.10 | 153.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 152.87 | 151.70 | 153.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 152.81 | 153.49 | 154.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 12:15:00 | 154.35 | 153.39 | 153.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 154.35 | 153.39 | 153.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 161.49 | 155.26 | 154.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 173.40 | 173.60 | 167.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 173.40 | 173.60 | 167.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 175.28 | 177.00 | 175.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 175.28 | 177.00 | 175.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 175.15 | 176.63 | 175.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 175.29 | 176.63 | 175.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 175.46 | 176.40 | 175.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:00:00 | 175.46 | 176.40 | 175.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 176.69 | 176.46 | 175.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 14:00:00 | 177.60 | 176.69 | 175.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 14:45:00 | 177.50 | 176.92 | 175.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 169.19 | 175.60 | 176.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 169.19 | 175.60 | 176.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 166.74 | 170.09 | 172.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 169.30 | 168.67 | 170.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 169.30 | 168.67 | 170.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 169.30 | 168.67 | 170.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 12:15:00 | 168.67 | 168.95 | 170.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 14:00:00 | 168.67 | 168.88 | 169.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 15:15:00 | 168.59 | 167.23 | 167.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 168.59 | 167.23 | 167.19 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 163.76 | 166.53 | 166.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 163.32 | 165.44 | 166.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 14:15:00 | 162.55 | 161.61 | 163.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 15:00:00 | 162.55 | 161.61 | 163.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 164.79 | 162.39 | 163.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:00:00 | 164.79 | 162.39 | 163.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 163.90 | 162.69 | 163.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:00:00 | 163.19 | 163.01 | 163.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 155.03 | 159.75 | 161.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 157.90 | 156.89 | 158.99 | SL hit (close>ema200) qty=0.50 sl=156.89 alert=retest2 |

### Cycle 67 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 165.45 | 161.08 | 160.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 166.30 | 162.85 | 161.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 12:15:00 | 167.64 | 167.65 | 166.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 12:45:00 | 167.11 | 167.65 | 166.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 166.60 | 167.44 | 166.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:00:00 | 166.60 | 167.44 | 166.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 166.65 | 167.47 | 166.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 170.18 | 166.76 | 166.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 15:15:00 | 169.70 | 171.46 | 171.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 169.70 | 171.46 | 171.68 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 172.25 | 171.76 | 171.72 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 170.97 | 171.60 | 171.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 169.65 | 170.80 | 171.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 170.64 | 170.50 | 171.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 15:00:00 | 170.64 | 170.50 | 171.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 170.16 | 170.46 | 170.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 170.89 | 170.46 | 170.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 170.98 | 170.56 | 170.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:30:00 | 170.45 | 170.56 | 170.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 171.35 | 170.72 | 170.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:45:00 | 171.64 | 170.72 | 170.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 171.80 | 170.94 | 171.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:45:00 | 171.90 | 170.94 | 171.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 171.84 | 171.12 | 171.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 172.88 | 171.57 | 171.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 172.47 | 172.82 | 172.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 172.47 | 172.82 | 172.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 172.47 | 172.82 | 172.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 172.47 | 172.82 | 172.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 173.58 | 172.97 | 172.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:30:00 | 173.75 | 173.17 | 172.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 14:15:00 | 173.68 | 173.39 | 172.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 174.32 | 174.08 | 173.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 173.10 | 174.76 | 174.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 173.10 | 174.76 | 174.81 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 175.40 | 174.24 | 174.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 10:15:00 | 177.19 | 174.83 | 174.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 173.98 | 175.64 | 175.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 173.98 | 175.64 | 175.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 173.98 | 175.64 | 175.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:30:00 | 176.40 | 176.39 | 175.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 178.56 | 181.20 | 181.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 178.56 | 181.20 | 181.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 175.98 | 179.47 | 180.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 163.57 | 161.24 | 163.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 163.57 | 161.24 | 163.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 163.57 | 161.24 | 163.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 162.76 | 161.24 | 163.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 163.50 | 161.69 | 163.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:15:00 | 164.16 | 161.69 | 163.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 164.45 | 162.24 | 163.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 164.15 | 162.24 | 163.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 165.28 | 162.85 | 163.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:00:00 | 165.28 | 162.85 | 163.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 165.75 | 164.29 | 164.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 167.71 | 165.71 | 165.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 166.15 | 166.59 | 165.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 13:15:00 | 166.15 | 166.59 | 165.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 166.15 | 166.59 | 165.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 166.15 | 166.59 | 165.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 169.91 | 170.64 | 169.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:15:00 | 169.90 | 170.64 | 169.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 169.38 | 170.39 | 169.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 169.38 | 170.39 | 169.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 169.75 | 170.26 | 169.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 171.70 | 170.26 | 169.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 13:00:00 | 170.19 | 170.48 | 170.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 14:30:00 | 170.40 | 170.37 | 170.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 11:00:00 | 170.25 | 170.27 | 170.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 170.15 | 170.24 | 170.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 169.81 | 170.24 | 170.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 170.04 | 170.20 | 170.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 170.04 | 170.20 | 170.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 170.14 | 170.19 | 170.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:30:00 | 169.95 | 170.19 | 170.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 170.15 | 170.18 | 170.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:30:00 | 170.00 | 170.18 | 170.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 170.60 | 170.27 | 170.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 170.65 | 170.27 | 170.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 168.98 | 170.01 | 170.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 168.98 | 170.01 | 170.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 168.75 | 169.76 | 169.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 10:15:00 | 167.00 | 166.95 | 167.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 11:00:00 | 167.00 | 166.95 | 167.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 166.70 | 166.91 | 167.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:30:00 | 166.98 | 166.91 | 167.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 166.50 | 166.87 | 167.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:30:00 | 167.30 | 166.87 | 167.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 165.41 | 166.22 | 166.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:45:00 | 164.30 | 165.77 | 166.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 168.53 | 166.23 | 166.32 | SL hit (close>static) qty=1.00 sl=167.27 alert=retest2 |

### Cycle 77 — BUY (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 10:15:00 | 168.64 | 166.71 | 166.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 12:15:00 | 169.69 | 167.60 | 166.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 161.07 | 167.02 | 167.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 161.07 | 167.02 | 167.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 161.07 | 167.02 | 167.00 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 161.45 | 165.91 | 166.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 09:15:00 | 158.59 | 161.53 | 163.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 10:15:00 | 160.33 | 159.80 | 161.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 11:00:00 | 160.33 | 159.80 | 161.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 160.31 | 159.97 | 160.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:45:00 | 160.57 | 159.97 | 160.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 159.57 | 159.98 | 160.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 12:00:00 | 159.47 | 159.86 | 160.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 14:45:00 | 159.46 | 159.69 | 160.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 151.50 | 153.13 | 154.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 151.49 | 153.13 | 154.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 150.43 | 150.26 | 151.86 | SL hit (close>ema200) qty=0.50 sl=150.26 alert=retest2 |

### Cycle 79 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 145.00 | 144.18 | 144.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 145.25 | 144.40 | 144.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 13:15:00 | 144.25 | 144.61 | 144.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 13:15:00 | 144.25 | 144.61 | 144.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 144.25 | 144.61 | 144.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 144.25 | 144.61 | 144.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 143.54 | 144.40 | 144.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 143.54 | 144.40 | 144.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 143.50 | 144.22 | 144.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 12:15:00 | 143.41 | 143.80 | 144.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 14:15:00 | 143.75 | 143.74 | 143.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 14:15:00 | 143.75 | 143.74 | 143.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 143.75 | 143.74 | 143.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:45:00 | 143.90 | 143.74 | 143.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 143.90 | 143.77 | 143.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 145.10 | 143.77 | 143.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 147.00 | 144.42 | 144.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 15:15:00 | 147.65 | 146.46 | 145.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 13:15:00 | 148.50 | 149.30 | 148.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 13:15:00 | 148.50 | 149.30 | 148.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 148.50 | 149.30 | 148.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:00:00 | 148.50 | 149.30 | 148.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 147.45 | 148.93 | 148.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 147.45 | 148.93 | 148.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 147.19 | 148.58 | 148.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 147.59 | 148.63 | 148.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 148.64 | 148.63 | 148.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 148.50 | 148.63 | 148.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 148.43 | 148.59 | 148.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:00:00 | 148.43 | 148.59 | 148.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 148.23 | 148.52 | 148.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:00:00 | 148.23 | 148.52 | 148.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 148.10 | 148.43 | 148.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:45:00 | 148.21 | 148.43 | 148.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 147.78 | 148.30 | 148.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 15:15:00 | 147.21 | 148.08 | 148.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 148.00 | 147.93 | 148.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 10:15:00 | 148.00 | 147.93 | 148.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 148.00 | 147.93 | 148.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:45:00 | 148.08 | 147.93 | 148.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 147.84 | 147.92 | 148.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 146.04 | 148.04 | 148.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 145.08 | 143.00 | 142.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 145.08 | 143.00 | 142.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 145.48 | 143.50 | 143.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 13:15:00 | 144.86 | 145.39 | 144.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 13:15:00 | 144.86 | 145.39 | 144.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 144.86 | 145.39 | 144.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:45:00 | 144.76 | 145.39 | 144.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 144.05 | 145.12 | 144.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 144.05 | 145.12 | 144.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 144.10 | 144.92 | 144.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 145.21 | 144.92 | 144.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 11:15:00 | 144.34 | 144.68 | 144.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 143.44 | 144.43 | 144.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 143.44 | 144.43 | 144.56 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 14:15:00 | 145.42 | 144.76 | 144.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 146.66 | 145.32 | 144.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 14:15:00 | 145.55 | 145.93 | 145.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 14:15:00 | 145.55 | 145.93 | 145.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 145.55 | 145.93 | 145.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:00:00 | 145.55 | 145.93 | 145.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 145.80 | 145.90 | 145.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 144.87 | 145.90 | 145.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 145.35 | 145.79 | 145.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:30:00 | 145.79 | 145.79 | 145.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 145.58 | 145.75 | 145.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:45:00 | 145.43 | 145.75 | 145.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 145.10 | 145.62 | 145.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 145.10 | 145.62 | 145.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 145.13 | 145.52 | 145.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:30:00 | 145.10 | 145.52 | 145.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 145.32 | 145.48 | 145.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:45:00 | 145.28 | 145.48 | 145.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 147.14 | 145.78 | 145.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 10:15:00 | 147.34 | 145.78 | 145.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 10:45:00 | 147.55 | 146.25 | 145.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 11:45:00 | 147.36 | 147.55 | 146.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 147.80 | 147.09 | 146.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 147.42 | 147.20 | 146.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:45:00 | 146.96 | 147.20 | 146.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 147.11 | 147.19 | 146.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:45:00 | 147.08 | 147.19 | 146.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 147.10 | 147.17 | 146.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:45:00 | 146.90 | 147.17 | 146.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 147.18 | 147.17 | 147.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:45:00 | 147.12 | 147.17 | 147.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 147.26 | 147.19 | 147.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:45:00 | 146.88 | 147.19 | 147.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 155.98 | 155.33 | 154.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 156.38 | 155.33 | 154.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-22 09:15:00 | 162.07 | 159.64 | 157.37 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 12:15:00 | 156.23 | 157.53 | 157.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 14:15:00 | 155.65 | 156.96 | 157.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 149.51 | 148.48 | 150.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 149.51 | 148.48 | 150.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 149.82 | 148.75 | 150.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 149.82 | 148.75 | 150.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 150.06 | 149.24 | 150.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 14:30:00 | 149.69 | 149.13 | 150.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:00:00 | 149.76 | 149.26 | 150.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 10:15:00 | 153.38 | 149.91 | 149.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 10:15:00 | 153.38 | 149.91 | 149.80 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 14:15:00 | 151.18 | 151.69 | 151.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 10:15:00 | 150.89 | 151.49 | 151.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 11:15:00 | 151.67 | 151.52 | 151.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 11:15:00 | 151.67 | 151.52 | 151.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 151.67 | 151.52 | 151.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:00:00 | 151.67 | 151.52 | 151.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 151.80 | 151.58 | 151.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:00:00 | 151.80 | 151.58 | 151.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 13:15:00 | 153.36 | 151.94 | 151.80 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 150.10 | 151.65 | 151.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 149.56 | 151.23 | 151.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 15:15:00 | 149.60 | 149.01 | 149.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 15:15:00 | 149.60 | 149.01 | 149.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 149.60 | 149.01 | 149.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 150.16 | 149.01 | 149.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 151.32 | 149.47 | 149.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 151.32 | 149.47 | 149.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 151.29 | 149.83 | 149.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:30:00 | 151.40 | 149.83 | 149.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 151.44 | 150.34 | 150.20 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 149.30 | 150.20 | 150.23 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 09:15:00 | 151.51 | 150.21 | 150.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 10:15:00 | 154.54 | 151.07 | 150.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 14:15:00 | 154.71 | 155.40 | 153.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 15:00:00 | 154.71 | 155.40 | 153.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 154.08 | 154.85 | 154.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:30:00 | 153.70 | 154.85 | 154.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 153.64 | 154.61 | 154.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:00:00 | 153.64 | 154.61 | 154.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 153.53 | 154.39 | 154.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:15:00 | 153.51 | 154.39 | 154.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 152.05 | 153.52 | 153.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 150.97 | 152.64 | 153.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 152.82 | 152.07 | 152.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 152.82 | 152.07 | 152.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 152.82 | 152.07 | 152.68 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 15:15:00 | 153.42 | 152.94 | 152.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 153.50 | 153.05 | 152.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 153.00 | 153.04 | 152.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 14:15:00 | 153.00 | 153.04 | 152.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 153.00 | 153.04 | 152.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 153.00 | 153.04 | 152.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 153.10 | 153.83 | 153.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 153.10 | 153.83 | 153.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 153.31 | 153.72 | 153.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 154.10 | 153.72 | 153.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 154.80 | 153.94 | 153.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 11:45:00 | 155.31 | 154.25 | 153.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 152.59 | 153.64 | 153.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 09:15:00 | 152.59 | 153.64 | 153.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 10:15:00 | 151.89 | 153.29 | 153.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 152.95 | 152.39 | 152.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 152.95 | 152.39 | 152.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 152.95 | 152.39 | 152.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 152.95 | 152.39 | 152.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 153.81 | 152.67 | 152.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 153.81 | 152.67 | 152.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 154.75 | 153.09 | 153.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:00:00 | 154.75 | 153.09 | 153.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 154.13 | 153.30 | 153.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 14:15:00 | 154.95 | 153.81 | 153.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 154.18 | 154.30 | 153.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 154.18 | 154.30 | 153.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 154.18 | 154.30 | 153.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 154.18 | 154.30 | 153.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 154.07 | 154.26 | 153.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:00:00 | 154.07 | 154.26 | 153.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 153.85 | 154.18 | 153.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:00:00 | 153.85 | 154.18 | 153.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 153.50 | 154.04 | 153.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:45:00 | 153.44 | 154.04 | 153.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 153.37 | 153.91 | 153.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 153.42 | 153.91 | 153.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 153.94 | 153.87 | 153.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:15:00 | 153.84 | 153.87 | 153.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 152.82 | 153.66 | 153.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 151.73 | 153.06 | 153.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 153.15 | 152.93 | 153.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 153.15 | 152.93 | 153.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 153.15 | 152.93 | 153.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 153.31 | 152.93 | 153.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 152.70 | 152.88 | 153.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:00:00 | 151.99 | 152.82 | 153.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:30:00 | 151.98 | 152.61 | 152.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 152.68 | 149.55 | 149.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 152.68 | 149.55 | 149.31 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 149.81 | 150.01 | 150.04 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 151.84 | 150.12 | 150.02 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 148.50 | 150.15 | 150.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 10:15:00 | 148.34 | 149.79 | 149.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 11:15:00 | 147.45 | 147.39 | 148.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 11:30:00 | 147.58 | 147.39 | 148.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 144.44 | 142.53 | 143.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 144.44 | 142.53 | 143.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 144.07 | 142.84 | 143.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 11:15:00 | 143.70 | 142.84 | 143.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 13:30:00 | 143.89 | 143.38 | 143.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 14:15:00 | 143.98 | 143.38 | 143.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 144.31 | 143.83 | 143.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 144.31 | 143.83 | 143.77 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 143.36 | 143.70 | 143.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 142.75 | 143.50 | 143.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 15:15:00 | 142.99 | 142.98 | 143.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 09:15:00 | 144.10 | 142.98 | 143.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 143.24 | 143.03 | 143.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:30:00 | 142.81 | 142.90 | 143.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 135.67 | 136.55 | 138.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 10:15:00 | 132.43 | 131.79 | 133.37 | SL hit (close>ema200) qty=0.50 sl=131.79 alert=retest2 |

### Cycle 105 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 134.00 | 133.86 | 133.84 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 12:15:00 | 133.54 | 133.81 | 133.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 13:15:00 | 133.26 | 133.70 | 133.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 134.25 | 133.58 | 133.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 134.25 | 133.58 | 133.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 134.25 | 133.58 | 133.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:00:00 | 134.25 | 133.58 | 133.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 133.74 | 133.61 | 133.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:45:00 | 134.15 | 133.61 | 133.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 134.50 | 133.79 | 133.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 135.04 | 134.33 | 134.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 135.17 | 135.29 | 134.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 135.17 | 135.29 | 134.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 135.17 | 135.29 | 134.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 134.91 | 135.29 | 134.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 134.27 | 135.17 | 134.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 134.27 | 135.17 | 134.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 134.36 | 135.00 | 134.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 134.36 | 135.00 | 134.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 134.00 | 134.80 | 134.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 133.81 | 134.48 | 134.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 132.45 | 132.04 | 132.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:45:00 | 132.40 | 132.04 | 132.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 131.16 | 131.58 | 132.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 132.03 | 131.58 | 132.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 132.28 | 131.65 | 132.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 132.28 | 131.65 | 132.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 133.15 | 131.95 | 132.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 133.15 | 131.95 | 132.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 133.00 | 132.16 | 132.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 134.61 | 132.16 | 132.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 135.90 | 132.91 | 132.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 137.16 | 135.43 | 134.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 138.20 | 138.36 | 137.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 13:30:00 | 138.24 | 138.36 | 137.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 141.87 | 138.91 | 137.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 142.68 | 138.91 | 137.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 12:15:00 | 142.60 | 140.10 | 138.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:00:00 | 142.20 | 140.52 | 138.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 136.96 | 139.02 | 139.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 136.96 | 139.02 | 139.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 136.68 | 138.56 | 138.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 140.29 | 137.93 | 138.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 140.29 | 137.93 | 138.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 140.29 | 137.93 | 138.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 140.29 | 137.93 | 138.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 140.53 | 138.45 | 138.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:15:00 | 140.19 | 138.45 | 138.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 11:15:00 | 140.45 | 138.85 | 138.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 140.45 | 138.85 | 138.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 142.04 | 139.90 | 139.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 12:15:00 | 144.80 | 144.92 | 143.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 13:00:00 | 144.80 | 144.92 | 143.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 144.30 | 144.58 | 143.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 143.96 | 144.58 | 143.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 143.80 | 144.42 | 143.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:45:00 | 143.70 | 144.42 | 143.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 143.47 | 144.23 | 143.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:00:00 | 144.30 | 143.85 | 143.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 142.56 | 144.51 | 144.28 | SL hit (close<static) qty=1.00 sl=142.69 alert=retest2 |

### Cycle 112 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 142.44 | 144.10 | 144.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 141.23 | 142.91 | 143.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 140.34 | 138.94 | 140.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 140.34 | 138.94 | 140.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 140.34 | 138.94 | 140.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 137.32 | 138.55 | 139.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 130.45 | 132.42 | 134.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 131.10 | 129.08 | 130.91 | SL hit (close>ema200) qty=0.50 sl=129.08 alert=retest2 |

### Cycle 113 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 131.03 | 129.35 | 129.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 13:15:00 | 131.63 | 130.08 | 129.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 11:15:00 | 131.85 | 131.86 | 130.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 11:45:00 | 131.84 | 131.86 | 130.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 132.92 | 132.61 | 131.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 134.90 | 132.11 | 131.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 129.86 | 132.11 | 131.96 | SL hit (close<static) qty=1.00 sl=130.50 alert=retest2 |

### Cycle 114 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 128.89 | 131.46 | 131.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 127.89 | 130.75 | 131.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 127.32 | 127.02 | 128.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 127.32 | 127.02 | 128.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 130.30 | 127.88 | 128.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 13:15:00 | 129.55 | 129.17 | 129.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 14:15:00 | 129.63 | 129.41 | 129.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 129.63 | 129.41 | 129.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 130.50 | 129.64 | 129.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 130.00 | 130.74 | 130.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 130.00 | 130.74 | 130.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 130.00 | 130.74 | 130.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 130.00 | 130.74 | 130.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 129.64 | 130.52 | 130.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 129.64 | 130.52 | 130.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 129.51 | 130.32 | 130.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 12:45:00 | 129.92 | 130.26 | 130.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 14:15:00 | 129.98 | 130.16 | 130.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 14:45:00 | 130.32 | 130.15 | 130.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 126.65 | 129.41 | 129.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 126.65 | 129.41 | 129.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 12:15:00 | 126.45 | 128.13 | 129.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 128.39 | 127.94 | 128.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 15:00:00 | 128.39 | 127.94 | 128.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 128.74 | 128.22 | 128.76 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 129.50 | 128.93 | 128.92 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 126.12 | 128.53 | 128.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 11:15:00 | 125.69 | 127.62 | 128.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 14:15:00 | 125.28 | 125.23 | 126.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 15:00:00 | 125.28 | 125.23 | 126.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 126.89 | 124.35 | 124.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 126.89 | 124.35 | 124.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 126.48 | 124.77 | 125.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:45:00 | 126.55 | 124.77 | 125.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 12:15:00 | 127.22 | 125.52 | 125.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 13:15:00 | 127.48 | 125.91 | 125.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 126.46 | 126.77 | 126.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 10:00:00 | 126.46 | 126.77 | 126.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 126.79 | 126.77 | 126.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 12:00:00 | 127.17 | 126.85 | 126.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 10:45:00 | 127.01 | 127.13 | 126.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 13:15:00 | 127.06 | 126.92 | 126.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 14:15:00 | 126.02 | 126.70 | 126.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 126.02 | 126.70 | 126.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 125.10 | 126.38 | 126.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 127.10 | 126.40 | 126.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 11:15:00 | 127.10 | 126.40 | 126.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 127.10 | 126.40 | 126.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:30:00 | 127.03 | 126.40 | 126.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 126.80 | 126.48 | 126.54 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 127.44 | 126.67 | 126.62 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 09:15:00 | 126.30 | 126.55 | 126.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 10:15:00 | 125.49 | 126.34 | 126.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 13:15:00 | 126.85 | 126.41 | 126.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 13:15:00 | 126.85 | 126.41 | 126.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 126.85 | 126.41 | 126.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:45:00 | 126.60 | 126.41 | 126.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 127.02 | 126.53 | 126.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 15:15:00 | 127.32 | 126.69 | 126.59 | Break + close above crossover candle high |

### Cycle 124 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 125.79 | 126.51 | 126.52 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 10:15:00 | 126.96 | 126.60 | 126.56 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 125.62 | 126.37 | 126.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 12:15:00 | 124.38 | 125.50 | 125.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 14:15:00 | 125.39 | 125.28 | 125.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 14:15:00 | 125.39 | 125.28 | 125.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 125.39 | 125.28 | 125.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 15:00:00 | 125.39 | 125.28 | 125.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 125.93 | 125.34 | 125.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 125.11 | 125.24 | 125.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 118.85 | 122.10 | 123.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 112.60 | 117.11 | 119.76 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 127 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 117.95 | 116.48 | 116.43 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 113.83 | 116.44 | 116.52 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 116.57 | 115.57 | 115.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 117.31 | 115.92 | 115.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 116.30 | 116.53 | 116.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:00:00 | 116.30 | 116.53 | 116.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 116.00 | 116.42 | 116.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 116.00 | 116.42 | 116.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 115.09 | 116.15 | 116.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 115.09 | 116.15 | 116.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 114.69 | 115.86 | 115.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 113.10 | 115.31 | 115.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 115.55 | 115.36 | 115.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 115.55 | 115.36 | 115.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 115.55 | 115.36 | 115.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 115.55 | 115.36 | 115.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 116.81 | 115.65 | 115.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 116.81 | 115.65 | 115.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 117.91 | 116.10 | 115.94 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 114.82 | 116.12 | 116.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 113.70 | 115.24 | 115.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 115.29 | 114.44 | 115.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 115.29 | 114.44 | 115.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 115.29 | 114.44 | 115.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 115.75 | 114.44 | 115.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 114.96 | 114.55 | 115.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 114.36 | 114.55 | 115.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 115.29 | 114.70 | 115.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 115.25 | 114.70 | 115.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 115.31 | 114.82 | 115.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 114.58 | 114.85 | 115.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 115.99 | 115.19 | 115.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 115.99 | 115.19 | 115.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 116.29 | 115.41 | 115.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 116.61 | 117.24 | 116.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 116.61 | 117.24 | 116.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 116.61 | 117.24 | 116.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:00:00 | 117.28 | 117.18 | 116.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 15:15:00 | 117.45 | 116.78 | 116.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 14:45:00 | 117.39 | 117.16 | 116.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 112.47 | 116.08 | 116.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 112.47 | 116.08 | 116.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 111.50 | 115.17 | 116.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 112.82 | 112.32 | 113.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 112.82 | 112.32 | 113.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 112.82 | 112.32 | 113.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:30:00 | 113.67 | 112.32 | 113.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 114.69 | 112.90 | 113.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 114.88 | 112.90 | 113.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 114.90 | 113.30 | 113.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 114.95 | 113.30 | 113.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 119.05 | 114.80 | 114.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 120.16 | 115.87 | 114.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 116.40 | 117.87 | 116.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 116.40 | 117.87 | 116.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 116.40 | 117.87 | 116.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 116.40 | 117.87 | 116.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 115.18 | 117.33 | 116.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 115.18 | 117.33 | 116.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 114.48 | 116.21 | 116.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 114.48 | 116.21 | 116.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 114.15 | 115.80 | 116.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 112.94 | 115.23 | 115.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 114.37 | 112.32 | 113.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 114.37 | 112.32 | 113.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 114.37 | 112.32 | 113.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 114.37 | 112.32 | 113.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 113.16 | 112.49 | 113.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 110.42 | 113.64 | 113.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 114.39 | 112.68 | 113.16 | SL hit (close>static) qty=1.00 sl=114.37 alert=retest2 |

### Cycle 137 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 114.90 | 113.50 | 113.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 115.80 | 114.61 | 114.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 114.59 | 114.78 | 114.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 114.59 | 114.78 | 114.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 114.59 | 114.78 | 114.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 119.77 | 115.19 | 114.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 09:15:00 | 131.75 | 129.25 | 127.20 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 14:15:00 | 128.97 | 129.89 | 129.98 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 09:15:00 | 131.60 | 130.09 | 130.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 10:15:00 | 134.03 | 130.88 | 130.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 15:15:00 | 137.29 | 137.61 | 135.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 09:15:00 | 137.29 | 137.61 | 135.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 136.60 | 137.41 | 135.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:45:00 | 136.25 | 137.41 | 135.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 134.33 | 136.79 | 135.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 134.33 | 136.79 | 135.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 135.25 | 136.49 | 135.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 12:45:00 | 135.33 | 136.16 | 135.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 14:00:00 | 135.47 | 136.02 | 135.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 134.02 | 137.02 | 137.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 134.02 | 137.02 | 137.33 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 136.70 | 135.78 | 135.70 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 134.80 | 135.71 | 135.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-11 09:15:00 | 131.39 | 134.10 | 134.83 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-17 09:15:00 | 174.70 | 2024-05-22 09:15:00 | 192.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-11 09:15:00 | 181.40 | 2024-06-13 15:15:00 | 180.24 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-06-20 11:15:00 | 177.59 | 2024-06-21 09:15:00 | 186.80 | STOP_HIT | 1.00 | -5.19% |
| SELL | retest2 | 2024-06-20 13:15:00 | 177.77 | 2024-06-21 09:15:00 | 186.80 | STOP_HIT | 1.00 | -5.08% |
| SELL | retest2 | 2024-06-20 15:15:00 | 177.50 | 2024-06-21 09:15:00 | 186.80 | STOP_HIT | 1.00 | -5.24% |
| BUY | retest2 | 2024-06-26 12:30:00 | 195.98 | 2024-06-28 14:15:00 | 190.27 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest2 | 2024-06-26 15:15:00 | 195.00 | 2024-06-28 14:15:00 | 190.27 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-07-09 13:15:00 | 242.35 | 2024-07-11 13:15:00 | 266.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-09 14:30:00 | 240.98 | 2024-07-11 13:15:00 | 265.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-09 15:00:00 | 241.13 | 2024-07-11 13:15:00 | 265.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-10 09:30:00 | 245.40 | 2024-07-11 13:15:00 | 269.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-31 09:15:00 | 266.22 | 2024-07-31 13:15:00 | 263.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-07-31 09:45:00 | 265.60 | 2024-07-31 13:15:00 | 263.80 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-08-06 10:15:00 | 243.10 | 2024-08-08 09:15:00 | 250.00 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2024-08-16 14:45:00 | 239.90 | 2024-08-19 09:15:00 | 247.00 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2024-08-16 15:15:00 | 239.80 | 2024-08-19 09:15:00 | 247.00 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2024-08-27 10:00:00 | 254.75 | 2024-08-28 15:15:00 | 254.00 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-08-27 10:30:00 | 255.50 | 2024-08-29 11:15:00 | 254.35 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-08-27 13:45:00 | 255.45 | 2024-08-29 15:15:00 | 254.50 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-08-27 15:00:00 | 255.00 | 2024-08-29 15:15:00 | 254.50 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2024-08-28 12:30:00 | 257.55 | 2024-08-29 15:15:00 | 254.50 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-08-29 09:15:00 | 258.30 | 2024-08-29 15:15:00 | 254.50 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-09-05 10:15:00 | 234.62 | 2024-09-09 09:15:00 | 222.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 11:15:00 | 235.05 | 2024-09-09 09:15:00 | 223.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 14:15:00 | 235.51 | 2024-09-09 09:15:00 | 223.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 10:15:00 | 234.62 | 2024-09-09 15:15:00 | 224.78 | STOP_HIT | 0.50 | 4.19% |
| SELL | retest2 | 2024-09-05 11:15:00 | 235.05 | 2024-09-09 15:15:00 | 224.78 | STOP_HIT | 0.50 | 4.37% |
| SELL | retest2 | 2024-09-05 14:15:00 | 235.51 | 2024-09-09 15:15:00 | 224.78 | STOP_HIT | 0.50 | 4.56% |
| SELL | retest2 | 2024-09-10 09:30:00 | 235.40 | 2024-09-10 13:15:00 | 233.64 | STOP_HIT | 1.00 | 0.75% |
| SELL | retest2 | 2024-09-10 11:15:00 | 232.99 | 2024-09-10 13:15:00 | 233.64 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-10-16 14:45:00 | 221.40 | 2024-10-21 14:15:00 | 210.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 10:15:00 | 221.00 | 2024-10-21 15:15:00 | 209.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 10:45:00 | 220.95 | 2024-10-21 15:15:00 | 209.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-16 14:45:00 | 221.40 | 2024-10-22 14:15:00 | 199.26 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-17 10:15:00 | 221.00 | 2024-10-22 14:15:00 | 198.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-17 10:45:00 | 220.95 | 2024-10-22 14:15:00 | 198.85 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-11-04 11:45:00 | 205.31 | 2024-11-04 12:15:00 | 205.30 | STOP_HIT | 1.00 | -0.00% |
| SELL | retest2 | 2024-11-14 11:15:00 | 191.39 | 2024-11-22 09:15:00 | 181.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-14 12:30:00 | 191.42 | 2024-11-22 09:15:00 | 181.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-14 13:30:00 | 191.28 | 2024-11-22 09:15:00 | 181.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-19 09:30:00 | 191.32 | 2024-11-22 09:15:00 | 181.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-14 11:15:00 | 191.39 | 2024-11-22 13:15:00 | 185.30 | STOP_HIT | 0.50 | 3.18% |
| SELL | retest2 | 2024-11-14 12:30:00 | 191.42 | 2024-11-22 13:15:00 | 185.30 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2024-11-14 13:30:00 | 191.28 | 2024-11-22 13:15:00 | 185.30 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2024-11-19 09:30:00 | 191.32 | 2024-11-22 13:15:00 | 185.30 | STOP_HIT | 0.50 | 3.15% |
| BUY | retest2 | 2024-12-04 09:15:00 | 208.74 | 2024-12-13 09:15:00 | 215.70 | STOP_HIT | 1.00 | 3.33% |
| SELL | retest2 | 2024-12-20 13:00:00 | 205.18 | 2024-12-30 13:15:00 | 202.71 | STOP_HIT | 1.00 | 1.20% |
| SELL | retest2 | 2024-12-20 13:30:00 | 204.16 | 2024-12-30 13:15:00 | 202.71 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2024-12-23 09:15:00 | 202.40 | 2024-12-30 13:15:00 | 202.71 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-01-08 09:15:00 | 216.41 | 2025-01-08 09:15:00 | 222.33 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2025-01-14 12:15:00 | 206.24 | 2025-01-17 09:15:00 | 210.56 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-01-14 14:15:00 | 206.29 | 2025-01-17 09:15:00 | 210.56 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-01-15 10:00:00 | 205.39 | 2025-01-17 09:15:00 | 210.56 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-01-15 11:30:00 | 206.21 | 2025-01-17 09:15:00 | 210.56 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-01-16 11:15:00 | 205.00 | 2025-01-17 09:15:00 | 210.56 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-01-16 11:45:00 | 204.82 | 2025-01-17 09:15:00 | 210.56 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-01-16 12:45:00 | 205.11 | 2025-01-17 09:15:00 | 210.56 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-01-16 13:15:00 | 205.02 | 2025-01-17 09:15:00 | 210.56 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-01-29 11:15:00 | 191.95 | 2025-01-29 14:15:00 | 191.80 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-02-13 13:30:00 | 177.80 | 2025-02-14 12:15:00 | 168.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:30:00 | 177.80 | 2025-02-17 15:15:00 | 166.60 | STOP_HIT | 0.50 | 6.30% |
| SELL | retest2 | 2025-04-08 10:30:00 | 152.87 | 2025-04-11 12:15:00 | 154.35 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-04-09 09:15:00 | 152.81 | 2025-04-11 12:15:00 | 154.35 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-04-23 14:00:00 | 177.60 | 2025-04-25 09:15:00 | 169.19 | STOP_HIT | 1.00 | -4.74% |
| BUY | retest2 | 2025-04-23 14:45:00 | 177.50 | 2025-04-25 09:15:00 | 169.19 | STOP_HIT | 1.00 | -4.68% |
| SELL | retest2 | 2025-04-29 12:15:00 | 168.67 | 2025-05-05 15:15:00 | 168.59 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-04-29 14:00:00 | 168.67 | 2025-05-05 15:15:00 | 168.59 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-05-08 13:00:00 | 163.19 | 2025-05-09 09:15:00 | 155.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 13:00:00 | 163.19 | 2025-05-09 15:15:00 | 157.90 | STOP_HIT | 0.50 | 3.24% |
| BUY | retest2 | 2025-05-16 09:15:00 | 170.18 | 2025-05-20 15:15:00 | 169.70 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-05-27 11:30:00 | 173.75 | 2025-06-03 09:15:00 | 173.10 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-05-27 14:15:00 | 173.68 | 2025-06-03 09:15:00 | 173.10 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-05-29 09:15:00 | 174.32 | 2025-06-03 09:15:00 | 173.10 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-06-09 09:30:00 | 176.40 | 2025-06-12 10:15:00 | 178.56 | STOP_HIT | 1.00 | 1.22% |
| BUY | retest2 | 2025-06-30 09:15:00 | 171.70 | 2025-07-02 09:15:00 | 168.98 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-06-30 13:00:00 | 170.19 | 2025-07-02 09:15:00 | 168.98 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-06-30 14:30:00 | 170.40 | 2025-07-02 09:15:00 | 168.98 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-07-01 11:00:00 | 170.25 | 2025-07-02 09:15:00 | 168.98 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-07-09 11:45:00 | 164.30 | 2025-07-10 09:15:00 | 168.53 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-07-17 12:00:00 | 159.47 | 2025-07-28 13:15:00 | 151.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 14:45:00 | 159.46 | 2025-07-28 13:15:00 | 151.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 12:00:00 | 159.47 | 2025-07-29 14:15:00 | 150.43 | STOP_HIT | 0.50 | 5.67% |
| SELL | retest2 | 2025-07-17 14:45:00 | 159.46 | 2025-07-29 14:15:00 | 150.43 | STOP_HIT | 0.50 | 5.66% |
| SELL | retest2 | 2025-08-26 09:15:00 | 146.04 | 2025-09-02 10:15:00 | 145.08 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2025-09-05 09:15:00 | 145.21 | 2025-09-05 11:15:00 | 143.44 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-09-05 11:15:00 | 144.34 | 2025-09-05 11:15:00 | 143.44 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-09-10 10:15:00 | 147.34 | 2025-09-22 09:15:00 | 162.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-10 10:45:00 | 147.55 | 2025-09-22 09:15:00 | 162.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-11 11:45:00 | 147.36 | 2025-09-22 09:15:00 | 162.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-12 09:15:00 | 147.80 | 2025-09-22 09:15:00 | 162.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-19 10:15:00 | 156.38 | 2025-09-23 12:15:00 | 156.23 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-09-23 12:15:00 | 156.38 | 2025-09-23 12:15:00 | 156.23 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-09-29 14:30:00 | 149.69 | 2025-10-01 10:15:00 | 153.38 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-09-30 10:00:00 | 149.76 | 2025-10-01 10:15:00 | 153.38 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-10-27 11:45:00 | 155.31 | 2025-10-28 09:15:00 | 152.59 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-11-04 10:00:00 | 151.99 | 2025-11-12 09:15:00 | 152.68 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-11-04 10:30:00 | 151.98 | 2025-11-12 09:15:00 | 152.68 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-11-26 11:15:00 | 143.70 | 2025-11-27 09:15:00 | 144.31 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-11-26 13:30:00 | 143.89 | 2025-11-27 09:15:00 | 144.31 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-11-26 14:15:00 | 143.98 | 2025-11-27 09:15:00 | 144.31 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-12-01 11:30:00 | 142.81 | 2025-12-05 09:15:00 | 135.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 11:30:00 | 142.81 | 2025-12-09 10:15:00 | 132.43 | STOP_HIT | 0.50 | 7.27% |
| BUY | retest2 | 2025-12-26 10:15:00 | 142.68 | 2025-12-30 09:15:00 | 136.96 | STOP_HIT | 1.00 | -4.01% |
| BUY | retest2 | 2025-12-26 12:15:00 | 142.60 | 2025-12-30 09:15:00 | 136.96 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2025-12-26 13:00:00 | 142.20 | 2025-12-30 09:15:00 | 136.96 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2025-12-31 11:15:00 | 140.19 | 2025-12-31 11:15:00 | 140.45 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2026-01-07 10:00:00 | 144.30 | 2026-01-08 10:15:00 | 142.56 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-01-16 12:15:00 | 137.32 | 2026-01-20 13:15:00 | 130.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 12:15:00 | 137.32 | 2026-01-22 09:15:00 | 131.10 | STOP_HIT | 0.50 | 4.53% |
| BUY | retest2 | 2026-02-01 09:15:00 | 134.90 | 2026-02-01 12:15:00 | 129.86 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2026-02-03 13:15:00 | 129.55 | 2026-02-03 14:15:00 | 129.63 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2026-02-05 12:45:00 | 129.92 | 2026-02-06 09:15:00 | 126.65 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2026-02-05 14:15:00 | 129.98 | 2026-02-06 09:15:00 | 126.65 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2026-02-05 14:45:00 | 130.32 | 2026-02-06 09:15:00 | 126.65 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2026-02-18 12:00:00 | 127.17 | 2026-02-19 14:15:00 | 126.02 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2026-02-19 10:45:00 | 127.01 | 2026-02-19 14:15:00 | 126.02 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-02-19 13:15:00 | 127.06 | 2026-02-19 14:15:00 | 126.02 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-02-26 11:30:00 | 125.11 | 2026-03-02 09:15:00 | 118.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 11:30:00 | 125.11 | 2026-03-04 09:15:00 | 112.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-17 12:15:00 | 114.58 | 2026-03-17 13:15:00 | 115.99 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2026-03-19 12:00:00 | 117.28 | 2026-03-23 09:15:00 | 112.47 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2026-03-19 15:15:00 | 117.45 | 2026-03-23 09:15:00 | 112.47 | STOP_HIT | 1.00 | -4.24% |
| BUY | retest2 | 2026-03-20 14:45:00 | 117.39 | 2026-03-23 09:15:00 | 112.47 | STOP_HIT | 1.00 | -4.19% |
| SELL | retest2 | 2026-04-02 09:15:00 | 110.42 | 2026-04-02 13:15:00 | 114.39 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2026-04-08 09:15:00 | 119.77 | 2026-04-17 09:15:00 | 131.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-24 12:45:00 | 135.33 | 2026-04-30 09:15:00 | 134.02 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-04-24 14:00:00 | 135.47 | 2026-04-30 09:15:00 | 134.02 | STOP_HIT | 1.00 | -1.07% |
