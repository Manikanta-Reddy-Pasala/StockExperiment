# Firstsource Solutions Ltd. (FSL)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 272.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 145 |
| ALERT1 | 103 |
| ALERT2 | 102 |
| ALERT2_SKIP | 51 |
| ALERT3 | 271 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 121 |
| PARTIAL | 18 |
| TARGET_HIT | 2 |
| STOP_HIT | 121 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 141 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 36 / 105
- **Target hits / Stop hits / Partials:** 2 / 121 / 18
- **Avg / median % per leg:** -0.24% / -1.17%
- **Sum % (uncompounded):** -33.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 52 | 5 | 9.6% | 1 | 51 | 0 | -1.47% | -76.4% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.47% | -0.9% |
| BUY @ 3rd Alert (retest2) | 50 | 4 | 8.0% | 1 | 49 | 0 | -1.51% | -75.5% |
| SELL (all) | 89 | 31 | 34.8% | 1 | 70 | 18 | 0.48% | 42.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 89 | 31 | 34.8% | 1 | 70 | 18 | 0.48% | 42.9% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.47% | -0.9% |
| retest2 (combined) | 139 | 35 | 25.2% | 2 | 119 | 18 | -0.23% | -32.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 189.75 | 186.54 | 186.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 191.20 | 188.08 | 187.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 12:15:00 | 198.70 | 198.80 | 196.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 13:00:00 | 198.70 | 198.80 | 196.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 198.45 | 198.51 | 196.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:45:00 | 198.45 | 198.51 | 196.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 11:15:00 | 197.20 | 198.25 | 196.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 12:00:00 | 197.20 | 198.25 | 196.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 196.30 | 197.86 | 196.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:15:00 | 198.35 | 197.86 | 196.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 198.70 | 198.03 | 197.04 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 10:15:00 | 195.25 | 196.67 | 196.84 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 14:15:00 | 199.95 | 197.20 | 197.01 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 15:15:00 | 196.05 | 197.18 | 197.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 13:15:00 | 195.95 | 196.85 | 197.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 14:15:00 | 194.05 | 194.04 | 194.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 14:15:00 | 194.05 | 194.04 | 194.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 194.05 | 194.04 | 194.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:45:00 | 194.25 | 194.04 | 194.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 193.60 | 193.76 | 194.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:00:00 | 193.60 | 193.76 | 194.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 192.25 | 187.22 | 188.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:15:00 | 191.15 | 187.22 | 188.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 190.15 | 187.80 | 189.03 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 13:15:00 | 191.00 | 189.76 | 189.74 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 187.10 | 189.77 | 189.80 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 190.35 | 186.77 | 186.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 199.00 | 190.50 | 188.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 197.15 | 197.28 | 195.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:15:00 | 197.60 | 197.28 | 195.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 13:15:00 | 203.60 | 204.27 | 202.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 13:45:00 | 202.76 | 204.27 | 202.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 203.65 | 204.14 | 202.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 14:30:00 | 202.78 | 204.14 | 202.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 198.96 | 202.92 | 202.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:00:00 | 198.96 | 202.92 | 202.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 198.05 | 201.95 | 201.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:30:00 | 198.69 | 201.95 | 201.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 11:15:00 | 198.15 | 201.19 | 201.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 13:15:00 | 197.66 | 199.97 | 200.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 14:15:00 | 200.16 | 200.01 | 200.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-18 14:30:00 | 200.05 | 200.01 | 200.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 198.27 | 199.67 | 200.57 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 12:15:00 | 202.60 | 200.78 | 200.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 09:15:00 | 210.00 | 203.09 | 201.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 14:15:00 | 205.85 | 206.23 | 204.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 15:00:00 | 205.85 | 206.23 | 204.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 204.88 | 205.95 | 204.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 204.43 | 205.95 | 204.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 206.50 | 206.06 | 204.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:30:00 | 204.87 | 206.06 | 204.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 210.20 | 212.12 | 210.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:00:00 | 210.20 | 212.12 | 210.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 210.83 | 211.86 | 210.33 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 09:15:00 | 207.60 | 209.66 | 209.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 10:15:00 | 205.63 | 208.86 | 209.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 13:15:00 | 209.65 | 208.13 | 208.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 13:15:00 | 209.65 | 208.13 | 208.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 209.65 | 208.13 | 208.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 209.65 | 208.13 | 208.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 211.80 | 208.87 | 209.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 211.83 | 208.87 | 209.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 09:15:00 | 211.25 | 209.54 | 209.40 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 13:15:00 | 208.58 | 209.31 | 209.34 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 14:15:00 | 210.00 | 209.45 | 209.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 212.42 | 210.13 | 209.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 14:15:00 | 224.29 | 224.31 | 220.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 14:45:00 | 224.30 | 224.31 | 220.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 247.45 | 250.06 | 245.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:15:00 | 244.10 | 250.06 | 245.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 240.26 | 248.10 | 245.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 240.26 | 248.10 | 245.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 240.82 | 246.64 | 244.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 237.87 | 246.64 | 244.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 14:15:00 | 239.97 | 243.24 | 243.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 13:15:00 | 239.48 | 240.70 | 241.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 09:15:00 | 245.92 | 241.59 | 242.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 09:15:00 | 245.92 | 241.59 | 242.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 245.92 | 241.59 | 242.07 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 251.25 | 243.52 | 242.90 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 14:15:00 | 245.64 | 249.27 | 249.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 15:15:00 | 243.49 | 247.22 | 248.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 249.00 | 247.58 | 248.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 249.00 | 247.58 | 248.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 249.00 | 247.58 | 248.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 249.00 | 247.58 | 248.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 245.45 | 247.15 | 248.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 12:15:00 | 243.75 | 246.71 | 247.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 13:00:00 | 244.41 | 246.25 | 247.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 14:00:00 | 244.37 | 245.87 | 247.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 14:45:00 | 244.00 | 245.66 | 247.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 242.61 | 244.91 | 246.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 10:30:00 | 241.05 | 243.94 | 245.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 231.56 | 241.89 | 244.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 232.19 | 241.89 | 244.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 232.15 | 241.89 | 244.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 231.80 | 241.89 | 244.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 229.00 | 241.89 | 244.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:00:00 | 238.04 | 241.89 | 244.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 246.09 | 242.31 | 243.83 | SL hit (close>ema200) qty=0.50 sl=242.31 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 249.00 | 245.07 | 244.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 259.40 | 249.49 | 247.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 286.85 | 294.45 | 289.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 09:15:00 | 286.85 | 294.45 | 289.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 286.85 | 294.45 | 289.85 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2024-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 12:15:00 | 276.55 | 286.08 | 286.74 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 10:15:00 | 310.00 | 289.42 | 287.45 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 12:15:00 | 287.65 | 289.37 | 289.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 14:15:00 | 283.55 | 287.91 | 288.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 10:15:00 | 286.80 | 286.58 | 287.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 10:15:00 | 286.80 | 286.58 | 287.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 286.80 | 286.58 | 287.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 10:45:00 | 287.10 | 286.58 | 287.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 281.00 | 284.56 | 286.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 14:30:00 | 279.35 | 282.40 | 284.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 10:45:00 | 278.75 | 280.56 | 282.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 12:15:00 | 288.65 | 282.85 | 283.46 | SL hit (close>static) qty=1.00 sl=287.40 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 13:15:00 | 288.90 | 284.06 | 283.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-14 15:15:00 | 290.70 | 286.16 | 284.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 11:15:00 | 323.60 | 326.12 | 316.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 11:45:00 | 322.60 | 326.12 | 316.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 322.35 | 322.80 | 318.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 14:45:00 | 323.95 | 323.24 | 320.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 10:30:00 | 323.80 | 323.46 | 321.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 15:15:00 | 316.20 | 320.13 | 320.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 15:15:00 | 316.20 | 320.13 | 320.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 09:15:00 | 310.35 | 318.18 | 319.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 13:15:00 | 323.15 | 316.65 | 317.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 13:15:00 | 323.15 | 316.65 | 317.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 323.15 | 316.65 | 317.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:00:00 | 323.15 | 316.65 | 317.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 324.05 | 318.13 | 318.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:45:00 | 324.25 | 318.13 | 318.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 15:15:00 | 322.30 | 318.96 | 318.80 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 09:15:00 | 317.35 | 318.64 | 318.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 11:15:00 | 315.75 | 317.78 | 318.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 09:15:00 | 318.05 | 310.26 | 311.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 09:15:00 | 318.05 | 310.26 | 311.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 318.05 | 310.26 | 311.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:45:00 | 316.35 | 310.26 | 311.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 315.40 | 311.29 | 311.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 11:15:00 | 312.80 | 311.29 | 311.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 12:30:00 | 311.70 | 311.61 | 311.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 11:45:00 | 312.50 | 311.01 | 311.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 14:00:00 | 312.25 | 311.42 | 311.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 301.70 | 309.47 | 310.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 14:30:00 | 311.80 | 309.47 | 310.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 313.70 | 308.44 | 309.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:00:00 | 313.70 | 308.44 | 309.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-02 11:15:00 | 318.80 | 310.51 | 310.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 11:15:00 | 318.80 | 310.51 | 310.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 13:15:00 | 322.30 | 318.23 | 316.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 315.75 | 318.32 | 316.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 315.75 | 318.32 | 316.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 315.75 | 318.32 | 316.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 315.75 | 318.32 | 316.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 315.35 | 317.73 | 316.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:45:00 | 316.25 | 317.73 | 316.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 317.65 | 317.71 | 316.67 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 309.35 | 315.15 | 315.78 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 11:15:00 | 320.00 | 315.47 | 314.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 12:15:00 | 323.00 | 316.98 | 315.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 320.40 | 323.38 | 320.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 13:15:00 | 320.40 | 323.38 | 320.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 320.40 | 323.38 | 320.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 320.40 | 323.38 | 320.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 319.80 | 322.66 | 320.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 319.80 | 322.66 | 320.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 318.50 | 321.83 | 320.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 321.55 | 321.83 | 320.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 09:15:00 | 317.15 | 320.89 | 320.13 | SL hit (close<static) qty=1.00 sl=318.40 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 11:15:00 | 316.20 | 319.26 | 319.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-13 09:15:00 | 314.50 | 317.75 | 318.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 14:15:00 | 308.85 | 307.98 | 311.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-16 15:00:00 | 308.85 | 307.98 | 311.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 310.90 | 308.32 | 311.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:30:00 | 314.85 | 308.32 | 311.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 311.45 | 308.95 | 311.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:45:00 | 312.15 | 308.95 | 311.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 309.70 | 309.10 | 310.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 13:45:00 | 309.20 | 309.59 | 310.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 15:15:00 | 308.00 | 309.60 | 310.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 10:15:00 | 309.30 | 305.96 | 307.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 09:15:00 | 308.70 | 307.05 | 307.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 307.65 | 307.17 | 307.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-20 10:15:00 | 312.80 | 308.30 | 307.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 312.80 | 308.30 | 307.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 11:15:00 | 315.10 | 309.66 | 308.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 13:15:00 | 310.65 | 310.68 | 309.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-20 13:30:00 | 311.00 | 310.68 | 309.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 330.80 | 335.24 | 329.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:00:00 | 330.80 | 335.24 | 329.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 330.70 | 334.33 | 329.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:00:00 | 330.70 | 334.33 | 329.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 329.50 | 333.37 | 329.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:00:00 | 329.50 | 333.37 | 329.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 328.60 | 332.41 | 329.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:00:00 | 328.60 | 332.41 | 329.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 13:15:00 | 328.25 | 331.58 | 329.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 14:45:00 | 330.80 | 331.31 | 329.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 09:15:00 | 333.45 | 330.72 | 329.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 10:15:00 | 326.80 | 330.45 | 329.63 | SL hit (close<static) qty=1.00 sl=328.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 12:15:00 | 323.25 | 328.30 | 328.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 13:15:00 | 321.55 | 326.95 | 328.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 314.55 | 314.16 | 318.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 10:15:00 | 312.75 | 314.16 | 318.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 310.65 | 309.00 | 311.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:45:00 | 310.85 | 309.00 | 311.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 303.25 | 306.17 | 308.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:45:00 | 300.35 | 305.29 | 308.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 11:45:00 | 301.35 | 305.01 | 307.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 14:15:00 | 299.30 | 304.93 | 307.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 13:15:00 | 311.50 | 307.50 | 307.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 311.50 | 307.50 | 307.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 14:15:00 | 316.25 | 309.25 | 308.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 14:15:00 | 314.40 | 314.59 | 312.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-09 14:30:00 | 313.20 | 314.59 | 312.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 311.30 | 313.64 | 312.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:00:00 | 311.30 | 313.64 | 312.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 311.00 | 313.12 | 312.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:45:00 | 311.40 | 313.12 | 312.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 310.40 | 312.05 | 311.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 15:00:00 | 310.40 | 312.05 | 311.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 15:15:00 | 309.25 | 311.49 | 311.63 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 09:15:00 | 316.75 | 312.54 | 312.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 09:15:00 | 330.50 | 319.06 | 316.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 10:15:00 | 329.50 | 329.72 | 324.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 11:00:00 | 329.50 | 329.72 | 324.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 331.00 | 333.57 | 331.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 12:30:00 | 334.90 | 332.76 | 331.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 10:00:00 | 338.00 | 333.59 | 332.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 15:15:00 | 325.30 | 330.78 | 331.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-10-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 15:15:00 | 325.30 | 330.78 | 331.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 321.70 | 328.97 | 330.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 328.60 | 320.84 | 324.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 09:15:00 | 328.60 | 320.84 | 324.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 328.60 | 320.84 | 324.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:00:00 | 328.60 | 320.84 | 324.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 336.15 | 323.90 | 325.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:00:00 | 336.15 | 323.90 | 325.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 11:15:00 | 340.30 | 327.18 | 326.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-23 12:15:00 | 355.70 | 332.89 | 329.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-24 12:15:00 | 340.50 | 342.01 | 337.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-24 13:00:00 | 340.50 | 342.01 | 337.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 13:15:00 | 339.00 | 341.41 | 337.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-24 14:15:00 | 336.85 | 341.41 | 337.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 332.45 | 339.62 | 336.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-24 15:00:00 | 332.45 | 339.62 | 336.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 331.30 | 337.95 | 336.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 09:15:00 | 335.80 | 337.95 | 336.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 332.65 | 336.34 | 335.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 330.10 | 336.34 | 335.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 12:15:00 | 335.75 | 335.93 | 335.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 12:30:00 | 334.70 | 335.93 | 335.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 13:15:00 | 330.00 | 334.74 | 335.25 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 355.55 | 339.11 | 336.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 14:15:00 | 375.35 | 346.36 | 340.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 10:15:00 | 350.70 | 353.30 | 345.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-29 10:30:00 | 347.20 | 353.30 | 345.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 12:15:00 | 343.20 | 350.41 | 345.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 13:00:00 | 343.20 | 350.41 | 345.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 347.10 | 349.75 | 345.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:00:00 | 347.10 | 349.75 | 345.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 343.80 | 348.56 | 345.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:30:00 | 342.45 | 348.56 | 345.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 342.00 | 347.25 | 345.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:15:00 | 339.00 | 347.25 | 345.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 332.60 | 344.32 | 344.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 10:00:00 | 332.60 | 344.32 | 344.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 10:15:00 | 331.00 | 341.65 | 342.92 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 09:15:00 | 346.75 | 341.93 | 341.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-04 14:15:00 | 360.00 | 348.90 | 345.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-06 09:15:00 | 363.30 | 364.05 | 357.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-06 10:00:00 | 363.30 | 364.05 | 357.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 374.25 | 378.07 | 374.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 374.25 | 378.07 | 374.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 375.10 | 377.47 | 374.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:30:00 | 372.80 | 377.47 | 374.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 372.15 | 376.41 | 374.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 13:45:00 | 372.60 | 376.41 | 374.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 368.80 | 374.89 | 373.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 15:00:00 | 368.80 | 374.89 | 373.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 362.30 | 371.09 | 372.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 358.05 | 364.92 | 368.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 379.80 | 366.01 | 367.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 379.80 | 366.01 | 367.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 379.80 | 366.01 | 367.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:45:00 | 381.20 | 366.01 | 367.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 376.45 | 368.10 | 368.71 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 11:15:00 | 375.25 | 369.53 | 369.30 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 12:15:00 | 366.55 | 368.93 | 369.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 13:15:00 | 363.85 | 367.92 | 368.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 349.00 | 342.13 | 345.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 09:15:00 | 349.00 | 342.13 | 345.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 349.00 | 342.13 | 345.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 349.00 | 342.13 | 345.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 348.90 | 343.49 | 346.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 349.25 | 343.49 | 346.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 12:15:00 | 346.70 | 344.60 | 346.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:15:00 | 344.65 | 344.95 | 346.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 15:15:00 | 347.00 | 343.13 | 342.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 15:15:00 | 347.00 | 343.13 | 342.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 355.70 | 345.65 | 344.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 13:15:00 | 359.90 | 360.14 | 354.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 14:00:00 | 359.90 | 360.14 | 354.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 362.10 | 361.25 | 359.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 10:45:00 | 366.00 | 361.50 | 360.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 10:15:00 | 370.05 | 378.50 | 379.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 370.05 | 378.50 | 379.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 09:15:00 | 363.45 | 368.52 | 371.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 11:15:00 | 368.30 | 367.98 | 370.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-18 12:00:00 | 368.30 | 367.98 | 370.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 370.55 | 368.70 | 370.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 15:00:00 | 370.55 | 368.70 | 370.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 15:15:00 | 370.00 | 368.96 | 370.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 361.50 | 368.96 | 370.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:45:00 | 369.55 | 366.28 | 367.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 372.00 | 367.42 | 368.28 | SL hit (close>static) qty=1.00 sl=371.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 13:15:00 | 363.05 | 360.08 | 359.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 10:15:00 | 363.90 | 360.97 | 360.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 15:15:00 | 399.45 | 400.31 | 394.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 09:15:00 | 392.60 | 400.31 | 394.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 399.40 | 400.13 | 394.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 09:30:00 | 407.50 | 397.29 | 395.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 10:45:00 | 409.50 | 400.32 | 397.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 15:15:00 | 406.70 | 405.62 | 401.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 13:15:00 | 397.50 | 399.37 | 399.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 13:15:00 | 397.50 | 399.37 | 399.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 09:15:00 | 392.85 | 397.24 | 398.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 12:15:00 | 396.95 | 396.93 | 398.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 14:15:00 | 395.80 | 396.79 | 397.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 395.80 | 396.79 | 397.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 15:15:00 | 392.10 | 396.79 | 397.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:45:00 | 381.85 | 392.97 | 395.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 372.50 | 380.62 | 387.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 12:15:00 | 362.76 | 373.21 | 381.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 12:15:00 | 366.20 | 363.11 | 371.20 | SL hit (close>ema200) qty=0.50 sl=363.11 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 14:15:00 | 380.05 | 372.77 | 372.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 383.25 | 376.17 | 374.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 379.35 | 381.47 | 378.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 379.35 | 381.47 | 378.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 379.35 | 381.47 | 378.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:30:00 | 377.75 | 381.47 | 378.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 379.20 | 381.01 | 378.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:00:00 | 379.20 | 381.01 | 378.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 382.05 | 381.22 | 378.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:30:00 | 382.10 | 381.22 | 378.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 382.70 | 387.80 | 385.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 382.70 | 387.80 | 385.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 375.55 | 385.35 | 384.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 375.55 | 385.35 | 384.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 11:15:00 | 376.00 | 383.48 | 383.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 12:15:00 | 372.40 | 381.26 | 382.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 10:15:00 | 355.25 | 353.36 | 361.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:45:00 | 355.20 | 353.36 | 361.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 360.45 | 356.02 | 361.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:15:00 | 356.60 | 356.48 | 361.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 11:15:00 | 370.50 | 357.26 | 359.53 | SL hit (close>static) qty=1.00 sl=361.75 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 12:15:00 | 377.25 | 361.26 | 361.14 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 350.25 | 360.72 | 361.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 334.70 | 350.22 | 355.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 10:15:00 | 333.95 | 333.24 | 341.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 10:45:00 | 334.10 | 333.24 | 341.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 340.90 | 335.37 | 340.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:45:00 | 341.75 | 335.37 | 340.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 348.65 | 338.03 | 341.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:00:00 | 348.65 | 338.03 | 341.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 347.70 | 339.96 | 342.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 09:15:00 | 342.95 | 341.17 | 342.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 10:30:00 | 343.80 | 342.72 | 343.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 12:15:00 | 345.25 | 343.57 | 343.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 12:15:00 | 345.25 | 343.57 | 343.39 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 341.60 | 343.17 | 343.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 15:15:00 | 339.75 | 342.41 | 342.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 09:15:00 | 342.00 | 337.58 | 339.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 09:15:00 | 342.00 | 337.58 | 339.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 342.00 | 337.58 | 339.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 11:45:00 | 337.05 | 338.18 | 339.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 13:00:00 | 338.70 | 338.29 | 339.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 14:15:00 | 343.80 | 340.47 | 340.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 14:15:00 | 343.80 | 340.47 | 340.15 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 330.70 | 338.63 | 339.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 11:15:00 | 323.90 | 330.46 | 334.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 12:15:00 | 333.90 | 331.15 | 334.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 12:15:00 | 333.90 | 331.15 | 334.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 333.90 | 331.15 | 334.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 13:00:00 | 333.90 | 331.15 | 334.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 334.15 | 331.75 | 334.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 13:45:00 | 336.05 | 331.75 | 334.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 335.00 | 332.40 | 334.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 15:00:00 | 335.00 | 332.40 | 334.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 335.00 | 332.92 | 334.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 344.20 | 332.92 | 334.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 351.75 | 336.69 | 335.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 12:15:00 | 352.25 | 343.54 | 339.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 13:15:00 | 349.70 | 353.19 | 349.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 13:15:00 | 349.70 | 353.19 | 349.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 349.70 | 353.19 | 349.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:00:00 | 349.70 | 353.19 | 349.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 354.40 | 353.43 | 350.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 361.95 | 353.53 | 350.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 10:15:00 | 346.00 | 351.44 | 350.09 | SL hit (close<static) qty=1.00 sl=349.50 alert=retest2 |

### Cycle 56 — SELL (started 2025-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 12:15:00 | 348.50 | 355.50 | 356.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 13:15:00 | 347.40 | 353.88 | 355.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 13:15:00 | 345.60 | 343.51 | 346.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 13:15:00 | 345.60 | 343.51 | 346.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 13:15:00 | 345.60 | 343.51 | 346.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 13:45:00 | 345.50 | 343.51 | 346.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 349.65 | 344.74 | 346.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 15:00:00 | 349.65 | 344.74 | 346.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 15:15:00 | 348.75 | 345.54 | 346.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:15:00 | 354.40 | 345.54 | 346.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 357.95 | 348.02 | 347.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 11:15:00 | 360.50 | 352.47 | 349.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 14:15:00 | 353.50 | 353.60 | 351.20 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 09:15:00 | 356.70 | 353.56 | 351.40 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 357.05 | 360.02 | 356.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 357.05 | 360.02 | 356.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 359.80 | 359.98 | 357.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 11:30:00 | 362.15 | 360.45 | 357.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 357.20 | 362.45 | 359.90 | SL hit (close<ema400) qty=1.00 sl=359.90 alert=retest1 |

### Cycle 58 — SELL (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 11:15:00 | 356.70 | 359.33 | 359.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 14:15:00 | 354.60 | 357.61 | 358.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 13:15:00 | 353.50 | 352.95 | 355.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 14:00:00 | 353.50 | 352.95 | 355.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 357.40 | 353.84 | 355.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 15:00:00 | 357.40 | 353.84 | 355.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 354.00 | 353.87 | 355.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 349.90 | 353.87 | 355.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 10:15:00 | 332.40 | 341.85 | 347.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-03 12:15:00 | 347.50 | 342.45 | 346.62 | SL hit (close>ema200) qty=0.50 sl=342.45 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 10:15:00 | 355.40 | 349.83 | 349.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 11:15:00 | 357.00 | 351.27 | 349.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 15:15:00 | 352.80 | 353.14 | 351.35 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 09:15:00 | 358.50 | 353.14 | 351.35 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 10:15:00 | 354.60 | 357.93 | 355.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-06 10:15:00 | 354.60 | 357.93 | 355.70 | SL hit (close<ema400) qty=1.00 sl=355.70 alert=retest1 |

### Cycle 60 — SELL (started 2025-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 09:15:00 | 350.40 | 355.06 | 355.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 09:15:00 | 346.40 | 352.08 | 353.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 12:15:00 | 285.55 | 282.04 | 292.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 13:00:00 | 285.55 | 282.04 | 292.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 291.75 | 284.96 | 290.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 09:45:00 | 292.30 | 284.96 | 290.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 10:15:00 | 290.70 | 286.11 | 290.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-19 13:15:00 | 287.20 | 288.34 | 290.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 14:15:00 | 294.40 | 290.11 | 291.16 | SL hit (close>static) qty=1.00 sl=292.65 alert=retest2 |

### Cycle 61 — BUY (started 2025-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 09:15:00 | 300.15 | 292.90 | 292.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 14:15:00 | 306.90 | 301.27 | 297.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-26 14:15:00 | 325.00 | 326.20 | 322.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-26 15:00:00 | 325.00 | 326.20 | 322.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 323.85 | 325.73 | 322.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 325.25 | 325.73 | 322.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 323.05 | 325.19 | 322.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 12:15:00 | 329.40 | 325.67 | 323.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-27 14:15:00 | 362.34 | 334.84 | 328.40 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 15:15:00 | 332.80 | 336.47 | 336.64 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 341.60 | 336.66 | 336.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 345.95 | 340.17 | 338.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 330.35 | 338.97 | 338.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 330.35 | 338.97 | 338.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 330.35 | 338.97 | 338.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 330.35 | 338.97 | 338.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 330.80 | 337.34 | 337.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 322.85 | 332.31 | 334.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 314.55 | 311.44 | 318.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 314.55 | 311.44 | 318.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 314.55 | 311.44 | 318.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:15:00 | 312.50 | 311.44 | 318.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-08 13:15:00 | 324.15 | 315.77 | 318.58 | SL hit (close>static) qty=1.00 sl=323.35 alert=retest2 |

### Cycle 65 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 322.65 | 316.74 | 316.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 10:15:00 | 327.35 | 321.64 | 319.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 11:15:00 | 335.40 | 335.94 | 331.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 11:45:00 | 334.00 | 335.94 | 331.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 346.55 | 353.18 | 351.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 346.55 | 353.18 | 351.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 339.45 | 350.43 | 350.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 339.45 | 350.43 | 350.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 344.30 | 349.20 | 349.48 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 353.55 | 349.87 | 349.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 14:15:00 | 354.65 | 352.03 | 350.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 09:15:00 | 351.55 | 352.25 | 351.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 351.55 | 352.25 | 351.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 351.55 | 352.25 | 351.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:45:00 | 350.15 | 352.25 | 351.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 349.25 | 351.65 | 351.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 11:00:00 | 349.25 | 351.65 | 351.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 350.30 | 351.38 | 350.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 11:30:00 | 349.80 | 351.38 | 350.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 349.90 | 351.08 | 350.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 13:00:00 | 349.90 | 351.08 | 350.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 13:15:00 | 351.30 | 351.13 | 350.89 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 348.15 | 350.33 | 350.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 10:15:00 | 345.25 | 349.31 | 350.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 343.95 | 342.70 | 345.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 343.95 | 342.70 | 345.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 343.95 | 342.70 | 345.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 343.95 | 342.70 | 345.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 339.40 | 336.12 | 338.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 335.25 | 336.12 | 338.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 328.50 | 334.59 | 337.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 10:30:00 | 325.30 | 333.37 | 336.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 12:15:00 | 327.30 | 332.49 | 336.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 309.03 | 321.77 | 328.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 310.94 | 321.77 | 328.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 321.30 | 316.92 | 322.16 | SL hit (close>ema200) qty=0.50 sl=316.92 alert=retest2 |

### Cycle 69 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 338.40 | 325.74 | 324.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 340.00 | 328.59 | 325.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 15:15:00 | 372.20 | 373.01 | 365.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 09:15:00 | 372.50 | 373.01 | 365.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 367.80 | 369.94 | 367.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:15:00 | 371.75 | 369.94 | 367.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 374.40 | 370.83 | 368.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:15:00 | 375.30 | 370.83 | 368.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 12:45:00 | 375.05 | 373.59 | 370.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 14:15:00 | 375.45 | 373.61 | 370.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 11:15:00 | 375.25 | 373.72 | 371.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 367.65 | 372.58 | 371.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 367.65 | 372.58 | 371.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 368.70 | 371.81 | 371.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 372.50 | 371.74 | 371.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 11:30:00 | 374.40 | 372.15 | 371.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 10:45:00 | 371.15 | 372.49 | 372.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 12:15:00 | 371.05 | 371.96 | 372.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 371.05 | 371.96 | 372.04 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 14:15:00 | 374.30 | 372.28 | 372.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 396.90 | 377.32 | 374.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 387.40 | 387.50 | 383.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:15:00 | 383.50 | 387.50 | 383.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 381.95 | 386.39 | 383.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 381.95 | 386.39 | 383.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 381.40 | 385.39 | 383.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:30:00 | 381.45 | 385.39 | 383.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 385.60 | 386.92 | 385.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 390.05 | 386.92 | 385.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 10:15:00 | 382.30 | 389.71 | 388.93 | SL hit (close<static) qty=1.00 sl=384.20 alert=retest2 |

### Cycle 72 — SELL (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 11:15:00 | 376.85 | 387.13 | 387.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 370.30 | 380.99 | 384.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 380.60 | 379.43 | 383.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 380.60 | 379.43 | 383.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 380.60 | 379.43 | 383.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:30:00 | 383.35 | 379.43 | 383.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 382.25 | 380.19 | 382.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:45:00 | 382.90 | 380.19 | 382.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 381.30 | 380.42 | 382.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 14:30:00 | 379.25 | 380.25 | 382.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 15:00:00 | 378.50 | 380.25 | 382.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:30:00 | 379.05 | 379.69 | 381.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 09:45:00 | 379.35 | 379.11 | 380.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 378.75 | 379.04 | 380.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:45:00 | 379.90 | 379.04 | 380.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 381.55 | 378.72 | 379.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 381.55 | 378.72 | 379.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 381.05 | 379.19 | 379.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:15:00 | 382.55 | 379.19 | 379.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 374.00 | 375.53 | 377.20 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-10 11:15:00 | 382.50 | 376.72 | 376.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 11:15:00 | 382.50 | 376.72 | 376.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 13:15:00 | 386.05 | 379.68 | 377.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 386.65 | 386.92 | 383.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:45:00 | 387.20 | 386.92 | 383.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 382.70 | 386.31 | 384.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 382.70 | 386.31 | 384.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 381.25 | 385.29 | 384.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 381.25 | 385.29 | 384.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 381.30 | 384.50 | 384.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 377.85 | 384.50 | 384.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 384.05 | 384.16 | 384.03 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2025-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 12:15:00 | 381.95 | 383.64 | 383.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 14:15:00 | 379.95 | 382.63 | 383.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 383.00 | 380.94 | 382.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 12:15:00 | 383.00 | 380.94 | 382.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 383.00 | 380.94 | 382.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 383.00 | 380.94 | 382.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 384.65 | 381.68 | 382.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 384.65 | 381.68 | 382.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 387.50 | 382.85 | 382.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 11:15:00 | 388.15 | 385.54 | 384.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 15:15:00 | 386.00 | 386.28 | 385.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 15:15:00 | 386.00 | 386.28 | 385.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 386.00 | 386.28 | 385.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 388.40 | 386.28 | 385.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 392.15 | 387.46 | 385.70 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 382.15 | 384.82 | 385.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 378.80 | 383.62 | 384.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 14:15:00 | 375.50 | 374.07 | 377.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 15:00:00 | 375.50 | 374.07 | 377.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 376.00 | 374.46 | 377.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 372.60 | 374.46 | 377.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 11:30:00 | 374.60 | 374.53 | 376.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 14:15:00 | 377.50 | 375.24 | 376.29 | SL hit (close>static) qty=1.00 sl=377.20 alert=retest2 |

### Cycle 77 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 384.80 | 377.36 | 377.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 14:15:00 | 387.05 | 382.58 | 380.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 14:15:00 | 389.50 | 390.93 | 386.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 14:45:00 | 389.00 | 390.93 | 386.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 383.90 | 389.25 | 386.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 383.90 | 389.25 | 386.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 382.05 | 387.81 | 386.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:30:00 | 382.60 | 387.81 | 386.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 383.80 | 385.46 | 385.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 15:00:00 | 383.80 | 385.46 | 385.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 15:15:00 | 382.00 | 384.77 | 385.01 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 13:15:00 | 387.15 | 385.15 | 385.10 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 14:15:00 | 382.10 | 384.54 | 384.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 09:15:00 | 377.20 | 382.98 | 384.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 09:15:00 | 366.20 | 364.30 | 370.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 09:15:00 | 366.20 | 364.30 | 370.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 366.20 | 364.30 | 370.52 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 10:15:00 | 371.50 | 369.39 | 369.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 14:15:00 | 372.50 | 370.39 | 369.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 09:15:00 | 367.40 | 369.86 | 369.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 367.40 | 369.86 | 369.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 367.40 | 369.86 | 369.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 367.40 | 369.86 | 369.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 10:15:00 | 368.60 | 369.61 | 369.62 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 11:15:00 | 369.90 | 369.66 | 369.64 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 365.40 | 368.95 | 369.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 361.80 | 367.52 | 368.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 13:15:00 | 366.05 | 366.04 | 367.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 13:30:00 | 366.10 | 366.04 | 367.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 368.85 | 366.43 | 367.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:15:00 | 364.55 | 366.77 | 367.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 12:15:00 | 346.32 | 352.76 | 357.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 353.15 | 351.19 | 354.99 | SL hit (close>ema200) qty=0.50 sl=351.19 alert=retest2 |

### Cycle 85 — BUY (started 2025-07-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 15:15:00 | 358.90 | 356.30 | 356.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 360.85 | 357.21 | 356.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 11:15:00 | 355.85 | 357.23 | 356.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 11:15:00 | 355.85 | 357.23 | 356.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 355.85 | 357.23 | 356.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:00:00 | 355.85 | 357.23 | 356.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 356.40 | 357.07 | 356.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:45:00 | 355.90 | 357.07 | 356.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 356.55 | 356.96 | 356.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:30:00 | 356.45 | 356.96 | 356.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 15:15:00 | 354.95 | 356.28 | 356.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 351.85 | 355.24 | 355.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 352.90 | 352.86 | 354.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-18 14:45:00 | 353.45 | 352.86 | 354.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 352.80 | 352.85 | 354.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 350.00 | 352.85 | 354.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 352.55 | 352.79 | 353.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:30:00 | 348.00 | 350.12 | 351.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 15:00:00 | 348.00 | 348.37 | 350.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:30:00 | 347.70 | 347.83 | 349.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 12:15:00 | 330.60 | 336.16 | 340.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 12:15:00 | 330.60 | 336.16 | 340.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 12:15:00 | 330.31 | 336.16 | 340.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 15:15:00 | 332.45 | 331.05 | 334.30 | SL hit (close>ema200) qty=0.50 sl=331.05 alert=retest2 |

### Cycle 87 — BUY (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 10:15:00 | 341.65 | 334.13 | 334.00 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 10:15:00 | 331.75 | 336.34 | 336.54 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 342.45 | 337.77 | 337.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 14:15:00 | 343.70 | 338.96 | 337.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 12:15:00 | 350.35 | 352.01 | 347.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 13:00:00 | 350.35 | 352.01 | 347.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 349.80 | 351.29 | 348.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 10:45:00 | 353.45 | 352.58 | 349.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 14:15:00 | 360.30 | 365.04 | 365.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 14:15:00 | 360.30 | 365.04 | 365.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 15:15:00 | 358.30 | 363.69 | 364.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 13:15:00 | 362.00 | 361.57 | 363.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-18 14:00:00 | 362.00 | 361.57 | 363.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 363.40 | 360.93 | 362.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:00:00 | 363.40 | 360.93 | 362.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 367.40 | 362.23 | 362.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:00:00 | 367.40 | 362.23 | 362.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 13:15:00 | 366.65 | 363.11 | 362.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 12:15:00 | 374.20 | 367.68 | 365.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 370.00 | 370.44 | 368.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:00:00 | 370.00 | 370.44 | 368.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 368.50 | 370.05 | 368.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:30:00 | 369.85 | 370.05 | 368.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 370.45 | 370.13 | 368.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 368.40 | 370.13 | 368.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 370.40 | 371.00 | 369.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:45:00 | 370.05 | 371.00 | 369.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 367.90 | 370.38 | 369.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 367.90 | 370.38 | 369.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 369.00 | 370.10 | 369.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 372.85 | 370.10 | 369.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 366.00 | 370.99 | 370.79 | SL hit (close<static) qty=1.00 sl=367.05 alert=retest2 |

### Cycle 92 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 362.60 | 369.31 | 370.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 12:15:00 | 361.20 | 366.56 | 368.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 12:15:00 | 350.50 | 350.12 | 355.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 13:00:00 | 350.50 | 350.12 | 355.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 356.05 | 351.68 | 354.21 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 359.35 | 355.87 | 355.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 15:15:00 | 360.00 | 356.69 | 355.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 15:15:00 | 365.80 | 366.48 | 363.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:00:00 | 364.65 | 366.12 | 363.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 362.95 | 365.48 | 363.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:45:00 | 363.90 | 365.48 | 363.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 367.15 | 365.82 | 364.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:30:00 | 361.65 | 365.82 | 364.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 363.25 | 365.30 | 363.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 363.25 | 365.30 | 363.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 362.40 | 364.72 | 363.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 362.40 | 364.72 | 363.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 361.20 | 364.02 | 363.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 361.20 | 364.02 | 363.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 361.50 | 363.51 | 363.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 363.40 | 363.51 | 363.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 355.00 | 361.73 | 362.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 355.00 | 361.73 | 362.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 354.00 | 360.18 | 361.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 12:15:00 | 349.50 | 345.83 | 349.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 13:00:00 | 349.50 | 345.83 | 349.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 351.30 | 346.93 | 350.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:30:00 | 351.85 | 346.93 | 350.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 351.70 | 347.88 | 350.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:45:00 | 350.85 | 347.88 | 350.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 350.05 | 348.32 | 350.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 353.05 | 348.32 | 350.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 354.15 | 349.48 | 350.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:30:00 | 357.75 | 349.48 | 350.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 11:15:00 | 356.05 | 351.69 | 351.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 12:15:00 | 358.00 | 352.96 | 351.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 09:15:00 | 366.35 | 366.60 | 363.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:00:00 | 366.35 | 366.60 | 363.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 362.95 | 365.36 | 363.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:00:00 | 362.95 | 365.36 | 363.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 367.00 | 365.69 | 363.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 364.35 | 365.69 | 363.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 362.65 | 365.47 | 364.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 372.50 | 365.11 | 364.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 15:15:00 | 373.10 | 367.29 | 366.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 10:00:00 | 369.45 | 372.92 | 371.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 11:30:00 | 368.85 | 371.55 | 371.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 12:15:00 | 368.10 | 370.86 | 371.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 12:15:00 | 368.10 | 370.86 | 371.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 365.50 | 369.79 | 370.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 359.35 | 358.38 | 361.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 10:00:00 | 359.35 | 358.38 | 361.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 326.30 | 324.08 | 326.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:00:00 | 326.30 | 324.08 | 326.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 326.50 | 324.57 | 326.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:45:00 | 326.50 | 324.57 | 326.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 328.60 | 325.37 | 326.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 15:00:00 | 328.60 | 325.37 | 326.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 328.50 | 326.00 | 326.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:15:00 | 329.70 | 326.00 | 326.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 325.30 | 325.94 | 326.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:30:00 | 326.15 | 325.94 | 326.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 326.25 | 325.34 | 326.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:00:00 | 326.25 | 325.34 | 326.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 327.10 | 325.69 | 326.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 15:15:00 | 324.35 | 325.69 | 326.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 11:00:00 | 325.70 | 325.80 | 326.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 15:15:00 | 324.10 | 325.81 | 325.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 10:15:00 | 328.20 | 326.26 | 326.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 328.20 | 326.26 | 326.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 11:15:00 | 329.50 | 326.91 | 326.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 325.10 | 331.15 | 329.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 325.10 | 331.15 | 329.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 325.10 | 331.15 | 329.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:00:00 | 325.10 | 331.15 | 329.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 322.80 | 329.48 | 329.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 322.80 | 329.48 | 329.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 324.00 | 328.38 | 328.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 322.00 | 324.94 | 326.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 325.10 | 324.58 | 326.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 14:45:00 | 325.05 | 324.58 | 326.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 325.05 | 324.48 | 325.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 11:45:00 | 323.20 | 324.17 | 325.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 13:30:00 | 323.35 | 323.82 | 325.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:00:00 | 323.45 | 323.61 | 324.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:30:00 | 323.30 | 323.57 | 324.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 325.15 | 323.88 | 324.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:00:00 | 325.15 | 323.88 | 324.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 323.55 | 323.82 | 324.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:45:00 | 323.15 | 323.88 | 324.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 327.30 | 324.31 | 324.43 | SL hit (close>static) qty=1.00 sl=326.40 alert=retest2 |

### Cycle 99 — BUY (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 12:15:00 | 327.55 | 324.96 | 324.72 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 322.55 | 324.74 | 324.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 11:15:00 | 321.10 | 324.01 | 324.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 322.90 | 321.23 | 322.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 322.90 | 321.23 | 322.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 322.90 | 321.23 | 322.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:00:00 | 322.90 | 321.23 | 322.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 321.00 | 321.19 | 322.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 321.00 | 321.19 | 322.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 325.95 | 322.14 | 322.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 325.95 | 322.14 | 322.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 323.80 | 322.47 | 322.92 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 11:15:00 | 326.55 | 323.29 | 323.25 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 321.30 | 323.07 | 323.25 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 12:15:00 | 326.65 | 323.39 | 323.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 343.00 | 328.34 | 325.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 12:15:00 | 345.70 | 346.42 | 340.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 10:15:00 | 345.15 | 344.59 | 341.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 345.15 | 344.59 | 341.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 14:45:00 | 348.10 | 345.93 | 343.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 13:15:00 | 348.80 | 350.89 | 351.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 348.80 | 350.89 | 351.07 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 14:15:00 | 352.70 | 351.25 | 351.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 09:15:00 | 357.00 | 352.52 | 351.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 13:15:00 | 354.40 | 354.53 | 353.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 13:30:00 | 354.35 | 354.53 | 353.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 351.35 | 353.90 | 353.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 351.35 | 353.90 | 353.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 350.05 | 353.13 | 352.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 343.00 | 353.13 | 352.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 340.50 | 350.60 | 351.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 15:15:00 | 340.00 | 343.90 | 347.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 10:15:00 | 349.45 | 345.01 | 347.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 10:15:00 | 349.45 | 345.01 | 347.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 349.45 | 345.01 | 347.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 347.50 | 345.01 | 347.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 349.60 | 345.93 | 347.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:30:00 | 349.55 | 345.93 | 347.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 350.45 | 348.35 | 348.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 353.25 | 349.66 | 348.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 11:15:00 | 364.10 | 364.57 | 360.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:00:00 | 364.10 | 364.57 | 360.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 362.10 | 363.91 | 361.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 364.60 | 362.93 | 361.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 11:15:00 | 364.15 | 363.11 | 362.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 13:30:00 | 363.70 | 363.53 | 362.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 14:00:00 | 364.15 | 363.53 | 362.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 363.50 | 363.47 | 362.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:15:00 | 360.75 | 363.47 | 362.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 359.80 | 362.74 | 362.46 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 359.80 | 362.74 | 362.46 | SL hit (close<static) qty=1.00 sl=360.15 alert=retest2 |

### Cycle 108 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 358.85 | 361.96 | 362.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 358.75 | 360.32 | 361.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 10:15:00 | 366.80 | 361.05 | 361.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 10:15:00 | 366.80 | 361.05 | 361.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 366.80 | 361.05 | 361.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 366.80 | 361.05 | 361.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 370.05 | 362.85 | 362.06 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 12:15:00 | 359.20 | 362.33 | 362.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 357.00 | 360.76 | 361.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 12:15:00 | 359.60 | 357.02 | 359.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 12:15:00 | 359.60 | 357.02 | 359.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 359.60 | 357.02 | 359.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:45:00 | 360.00 | 357.02 | 359.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 358.95 | 357.41 | 359.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:15:00 | 359.50 | 357.41 | 359.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 359.60 | 357.84 | 359.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 359.60 | 357.84 | 359.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 358.70 | 358.02 | 359.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 357.25 | 358.02 | 359.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 15:15:00 | 339.39 | 350.69 | 354.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 11:15:00 | 350.55 | 349.99 | 353.26 | SL hit (close>ema200) qty=0.50 sl=349.99 alert=retest2 |

### Cycle 111 — BUY (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 14:15:00 | 348.30 | 347.19 | 347.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 09:15:00 | 351.00 | 347.92 | 347.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 15:15:00 | 349.70 | 349.80 | 348.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 15:15:00 | 349.70 | 349.80 | 348.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 349.70 | 349.80 | 348.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 350.10 | 349.80 | 348.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 351.15 | 350.07 | 349.02 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 14:15:00 | 347.15 | 348.43 | 348.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 341.15 | 346.74 | 347.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 340.00 | 339.85 | 342.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 340.00 | 339.85 | 342.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 342.50 | 340.68 | 342.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 343.65 | 341.19 | 342.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 339.00 | 340.75 | 342.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:30:00 | 338.15 | 339.92 | 341.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 344.10 | 339.83 | 339.86 | SL hit (close>static) qty=1.00 sl=343.30 alert=retest2 |

### Cycle 113 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 344.00 | 340.66 | 340.24 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 339.35 | 341.50 | 341.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 339.00 | 340.96 | 341.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 340.00 | 339.68 | 340.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 10:15:00 | 340.00 | 339.68 | 340.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 340.00 | 339.68 | 340.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:45:00 | 340.05 | 339.68 | 340.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 341.05 | 339.96 | 340.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:30:00 | 340.75 | 339.96 | 340.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 340.50 | 340.07 | 340.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 341.25 | 340.07 | 340.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 340.15 | 340.08 | 340.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:45:00 | 340.70 | 340.08 | 340.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 342.85 | 340.64 | 340.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 342.85 | 340.64 | 340.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 15:15:00 | 342.80 | 341.07 | 340.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 346.80 | 343.48 | 342.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 352.35 | 353.36 | 349.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 10:00:00 | 352.35 | 353.36 | 349.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 355.00 | 352.79 | 350.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:30:00 | 351.20 | 352.79 | 350.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 351.15 | 352.67 | 351.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 351.15 | 352.67 | 351.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 350.70 | 352.28 | 351.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:45:00 | 350.80 | 352.28 | 351.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 349.05 | 350.88 | 350.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 11:15:00 | 348.60 | 350.43 | 350.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 12:15:00 | 333.60 | 332.97 | 336.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 13:00:00 | 333.60 | 332.97 | 336.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 335.50 | 333.47 | 336.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:30:00 | 335.20 | 333.47 | 336.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 336.25 | 334.03 | 336.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 336.25 | 334.03 | 336.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 333.80 | 334.22 | 336.11 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 341.75 | 336.28 | 336.07 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 336.70 | 337.25 | 337.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 336.20 | 337.01 | 337.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 336.85 | 336.60 | 336.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 336.85 | 336.60 | 336.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 336.85 | 336.60 | 336.90 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 12:15:00 | 339.70 | 337.55 | 337.29 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 334.55 | 337.08 | 337.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 333.15 | 336.30 | 336.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 10:15:00 | 315.90 | 313.90 | 318.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 11:00:00 | 315.90 | 313.90 | 318.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 318.40 | 314.80 | 318.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:00:00 | 318.40 | 314.80 | 318.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 318.70 | 315.58 | 318.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:30:00 | 316.50 | 317.80 | 319.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 320.25 | 318.29 | 319.13 | SL hit (close>static) qty=1.00 sl=320.00 alert=retest2 |

### Cycle 121 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 327.80 | 320.93 | 320.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 10:15:00 | 333.70 | 323.49 | 321.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 15:15:00 | 325.30 | 326.22 | 323.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 15:15:00 | 325.30 | 326.22 | 323.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 325.30 | 326.22 | 323.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 320.70 | 326.22 | 323.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 322.50 | 325.47 | 323.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 320.80 | 325.47 | 323.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 320.40 | 324.46 | 323.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 320.55 | 324.46 | 323.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 317.30 | 322.16 | 322.49 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 10:15:00 | 325.15 | 322.38 | 322.34 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 321.40 | 322.19 | 322.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 320.05 | 321.60 | 321.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 11:15:00 | 311.05 | 310.92 | 314.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 11:30:00 | 310.40 | 310.92 | 314.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 312.30 | 311.24 | 313.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:45:00 | 313.10 | 311.24 | 313.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 314.00 | 311.80 | 313.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:45:00 | 311.95 | 311.74 | 313.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 311.20 | 313.20 | 313.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 10:15:00 | 310.60 | 313.43 | 313.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 11:15:00 | 314.40 | 312.93 | 312.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 314.40 | 312.93 | 312.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 13:15:00 | 316.10 | 313.82 | 313.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 15:15:00 | 317.45 | 318.31 | 316.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 15:15:00 | 317.45 | 318.31 | 316.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 317.45 | 318.31 | 316.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:15:00 | 317.05 | 318.31 | 316.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 320.00 | 318.64 | 316.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 11:30:00 | 322.00 | 319.67 | 317.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 12:00:00 | 321.60 | 319.67 | 317.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 12:30:00 | 321.55 | 320.12 | 318.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 13:00:00 | 321.95 | 320.12 | 318.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 319.50 | 320.26 | 318.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 318.35 | 320.26 | 318.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 317.95 | 319.79 | 318.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:30:00 | 317.95 | 319.79 | 318.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 319.10 | 319.66 | 318.67 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 312.05 | 318.13 | 318.07 | SL hit (close<static) qty=1.00 sl=315.25 alert=retest2 |

### Cycle 126 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 312.15 | 316.94 | 317.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 305.00 | 312.36 | 315.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 314.80 | 308.34 | 310.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 314.80 | 308.34 | 310.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 314.80 | 308.34 | 310.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 12:45:00 | 309.95 | 310.58 | 311.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-06 09:15:00 | 294.45 | 298.92 | 302.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 293.90 | 292.90 | 296.67 | SL hit (close>ema200) qty=0.50 sl=292.90 alert=retest2 |

### Cycle 127 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 260.50 | 256.86 | 256.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 262.50 | 257.99 | 257.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 257.65 | 258.38 | 257.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 257.65 | 258.38 | 257.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 257.65 | 258.38 | 257.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 257.65 | 258.38 | 257.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 257.50 | 258.20 | 257.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 248.50 | 258.20 | 257.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 247.95 | 256.15 | 256.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 10:15:00 | 244.05 | 253.73 | 255.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 246.20 | 245.76 | 249.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 246.20 | 245.76 | 249.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 246.20 | 245.76 | 249.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:15:00 | 244.05 | 245.61 | 249.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:15:00 | 231.85 | 238.99 | 243.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-25 13:15:00 | 219.65 | 225.60 | 232.39 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 129 — BUY (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 12:15:00 | 218.25 | 216.00 | 215.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 13:15:00 | 219.76 | 216.75 | 216.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 11:15:00 | 218.39 | 219.54 | 218.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 11:15:00 | 218.39 | 219.54 | 218.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 218.39 | 219.54 | 218.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:00:00 | 218.39 | 219.54 | 218.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 221.15 | 219.86 | 218.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 14:00:00 | 222.00 | 220.29 | 218.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 215.79 | 219.25 | 218.56 | SL hit (close<static) qty=1.00 sl=218.21 alert=retest2 |

### Cycle 130 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 215.62 | 217.81 | 217.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 210.45 | 215.98 | 217.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 214.95 | 214.67 | 216.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 214.95 | 214.67 | 216.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 214.95 | 214.67 | 216.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 11:15:00 | 211.51 | 214.34 | 215.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:30:00 | 211.37 | 213.15 | 214.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 218.01 | 215.08 | 214.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 14:15:00 | 218.01 | 215.08 | 214.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 09:15:00 | 220.20 | 216.57 | 215.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 220.90 | 220.97 | 218.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 220.90 | 220.97 | 218.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 220.90 | 220.97 | 218.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 10:45:00 | 223.00 | 221.33 | 219.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 13:30:00 | 222.80 | 222.06 | 220.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 09:15:00 | 215.68 | 221.66 | 220.50 | SL hit (close<static) qty=1.00 sl=217.55 alert=retest2 |

### Cycle 132 — SELL (started 2026-03-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 10:15:00 | 211.78 | 219.68 | 219.71 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 224.57 | 218.14 | 217.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 226.40 | 219.80 | 218.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 222.80 | 224.14 | 221.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 13:15:00 | 222.00 | 223.26 | 222.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 222.00 | 223.26 | 222.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 14:00:00 | 222.00 | 223.26 | 222.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 220.40 | 222.68 | 221.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 14:45:00 | 219.42 | 222.68 | 221.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 219.00 | 221.95 | 221.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 221.91 | 221.95 | 221.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 10:15:00 | 219.31 | 221.22 | 221.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 10:15:00 | 219.31 | 221.22 | 221.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 11:15:00 | 218.24 | 220.62 | 221.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 214.62 | 210.71 | 213.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 214.62 | 210.71 | 213.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 214.62 | 210.71 | 213.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 214.62 | 210.71 | 213.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 215.72 | 211.72 | 213.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 215.72 | 211.72 | 213.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 223.98 | 216.10 | 215.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 14:15:00 | 224.59 | 221.23 | 218.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 214.00 | 220.23 | 218.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 214.00 | 220.23 | 218.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 214.00 | 220.23 | 218.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 214.00 | 220.23 | 218.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 213.62 | 218.91 | 218.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 213.46 | 218.91 | 218.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 213.75 | 216.91 | 217.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 207.16 | 213.41 | 215.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 217.21 | 209.20 | 211.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 217.21 | 209.20 | 211.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 217.21 | 209.20 | 211.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 217.21 | 209.20 | 211.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 216.63 | 210.69 | 211.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:15:00 | 217.61 | 210.69 | 211.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 218.69 | 213.75 | 213.21 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 210.28 | 212.69 | 212.96 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 216.51 | 213.50 | 213.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 218.14 | 214.42 | 213.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 10:15:00 | 225.13 | 226.40 | 223.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-08 11:00:00 | 225.13 | 226.40 | 223.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 223.57 | 225.84 | 223.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:45:00 | 223.90 | 225.84 | 223.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 12:15:00 | 223.69 | 225.41 | 223.31 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2026-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 10:15:00 | 220.01 | 222.44 | 222.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-09 15:15:00 | 218.99 | 220.48 | 221.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 10:15:00 | 220.76 | 220.53 | 221.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 10:15:00 | 220.76 | 220.53 | 221.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 220.76 | 220.53 | 221.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:00:00 | 220.76 | 220.53 | 221.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 223.00 | 218.25 | 219.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:15:00 | 223.10 | 218.25 | 219.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 14:15:00 | 220.30 | 219.49 | 219.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 251.10 | 226.05 | 222.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 12:15:00 | 240.80 | 241.02 | 235.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 12:45:00 | 240.51 | 241.02 | 235.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 233.37 | 238.85 | 236.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:30:00 | 232.47 | 238.85 | 236.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 235.10 | 238.10 | 236.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 11:15:00 | 236.01 | 238.10 | 236.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 232.30 | 235.43 | 235.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 232.30 | 235.43 | 235.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 14:15:00 | 230.21 | 232.58 | 233.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 15:15:00 | 224.60 | 224.47 | 228.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-23 09:15:00 | 221.76 | 224.47 | 228.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 215.71 | 213.28 | 214.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 215.98 | 213.28 | 214.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 215.80 | 213.79 | 214.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 11:00:00 | 215.80 | 213.79 | 214.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 216.29 | 214.29 | 215.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:00:00 | 216.29 | 214.29 | 215.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 216.66 | 214.76 | 215.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:30:00 | 215.30 | 214.80 | 215.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:45:00 | 215.70 | 214.65 | 214.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:45:00 | 214.88 | 214.71 | 214.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 11:45:00 | 215.63 | 214.94 | 215.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 214.84 | 214.92 | 215.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 15:00:00 | 214.00 | 214.83 | 214.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 216.29 | 214.91 | 214.98 | SL hit (close>static) qty=1.00 sl=215.95 alert=retest2 |

### Cycle 143 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 217.10 | 215.35 | 215.17 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-05-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 13:15:00 | 214.69 | 215.09 | 215.09 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 216.11 | 215.27 | 215.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 217.82 | 216.02 | 215.56 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-07-22 12:15:00 | 243.75 | 2024-07-23 12:15:00 | 231.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 13:00:00 | 244.41 | 2024-07-23 12:15:00 | 232.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 14:00:00 | 244.37 | 2024-07-23 12:15:00 | 232.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 14:45:00 | 244.00 | 2024-07-23 12:15:00 | 231.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-23 10:30:00 | 241.05 | 2024-07-23 12:15:00 | 229.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 12:15:00 | 243.75 | 2024-07-24 09:15:00 | 246.09 | STOP_HIT | 0.50 | -0.96% |
| SELL | retest2 | 2024-07-22 13:00:00 | 244.41 | 2024-07-24 09:15:00 | 246.09 | STOP_HIT | 0.50 | -0.69% |
| SELL | retest2 | 2024-07-22 14:00:00 | 244.37 | 2024-07-24 09:15:00 | 246.09 | STOP_HIT | 0.50 | -0.70% |
| SELL | retest2 | 2024-07-22 14:45:00 | 244.00 | 2024-07-24 09:15:00 | 246.09 | STOP_HIT | 0.50 | -0.86% |
| SELL | retest2 | 2024-07-23 10:30:00 | 241.05 | 2024-07-24 09:15:00 | 246.09 | STOP_HIT | 0.50 | -2.09% |
| SELL | retest2 | 2024-07-23 13:00:00 | 238.04 | 2024-07-24 11:15:00 | 248.34 | STOP_HIT | 1.00 | -4.33% |
| SELL | retest2 | 2024-08-13 14:30:00 | 279.35 | 2024-08-14 12:15:00 | 288.65 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2024-08-14 10:45:00 | 278.75 | 2024-08-14 12:15:00 | 288.65 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2024-08-21 14:45:00 | 323.95 | 2024-08-22 15:15:00 | 316.20 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2024-08-22 10:30:00 | 323.80 | 2024-08-22 15:15:00 | 316.20 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2024-08-29 11:15:00 | 312.80 | 2024-09-02 11:15:00 | 318.80 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-08-29 12:30:00 | 311.70 | 2024-09-02 11:15:00 | 318.80 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-08-30 11:45:00 | 312.50 | 2024-09-02 11:15:00 | 318.80 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-08-30 14:00:00 | 312.25 | 2024-09-02 11:15:00 | 318.80 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-09-12 09:15:00 | 321.55 | 2024-09-12 09:15:00 | 317.15 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-09-17 13:45:00 | 309.20 | 2024-09-20 10:15:00 | 312.80 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-09-17 15:15:00 | 308.00 | 2024-09-20 10:15:00 | 312.80 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-09-19 10:15:00 | 309.30 | 2024-09-20 10:15:00 | 312.80 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-09-20 09:15:00 | 308.70 | 2024-09-20 10:15:00 | 312.80 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-09-26 14:45:00 | 330.80 | 2024-09-27 10:15:00 | 326.80 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-09-27 09:15:00 | 333.45 | 2024-09-27 10:15:00 | 326.80 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-10-07 10:45:00 | 300.35 | 2024-10-08 13:15:00 | 311.50 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2024-10-07 11:45:00 | 301.35 | 2024-10-08 13:15:00 | 311.50 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2024-10-07 14:15:00 | 299.30 | 2024-10-08 13:15:00 | 311.50 | STOP_HIT | 1.00 | -4.08% |
| BUY | retest2 | 2024-10-18 12:30:00 | 334.90 | 2024-10-21 15:15:00 | 325.30 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2024-10-21 10:00:00 | 338.00 | 2024-10-21 15:15:00 | 325.30 | STOP_HIT | 1.00 | -3.76% |
| SELL | retest2 | 2024-11-19 14:15:00 | 344.65 | 2024-11-22 15:15:00 | 347.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-11-29 10:45:00 | 366.00 | 2024-12-13 10:15:00 | 370.05 | STOP_HIT | 1.00 | 1.11% |
| SELL | retest2 | 2024-12-19 09:15:00 | 361.50 | 2024-12-20 10:15:00 | 372.00 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2024-12-20 09:45:00 | 369.55 | 2024-12-20 10:15:00 | 372.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-12-20 12:45:00 | 367.55 | 2024-12-23 11:15:00 | 349.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 12:45:00 | 367.55 | 2024-12-23 14:15:00 | 356.15 | STOP_HIT | 0.50 | 3.10% |
| BUY | retest2 | 2025-01-07 09:30:00 | 407.50 | 2025-01-08 13:15:00 | 397.50 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-01-07 10:45:00 | 409.50 | 2025-01-08 13:15:00 | 397.50 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-01-07 15:15:00 | 406.70 | 2025-01-08 13:15:00 | 397.50 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-01-09 15:15:00 | 392.10 | 2025-01-13 09:15:00 | 372.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 09:45:00 | 381.85 | 2025-01-13 12:15:00 | 362.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 15:15:00 | 392.10 | 2025-01-14 12:15:00 | 366.20 | STOP_HIT | 0.50 | 6.61% |
| SELL | retest2 | 2025-01-10 09:45:00 | 381.85 | 2025-01-14 12:15:00 | 366.20 | STOP_HIT | 0.50 | 4.10% |
| SELL | retest2 | 2025-01-23 14:15:00 | 356.60 | 2025-01-24 11:15:00 | 370.50 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2025-01-30 09:15:00 | 342.95 | 2025-01-30 12:15:00 | 345.25 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-01-30 10:30:00 | 343.80 | 2025-01-30 12:15:00 | 345.25 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-02-01 11:45:00 | 337.05 | 2025-02-01 14:15:00 | 343.80 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-02-01 13:00:00 | 338.70 | 2025-02-01 14:15:00 | 343.80 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-02-10 09:15:00 | 361.95 | 2025-02-10 10:15:00 | 346.00 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2025-02-11 13:45:00 | 359.15 | 2025-02-12 09:15:00 | 345.00 | STOP_HIT | 1.00 | -3.94% |
| BUY | retest2 | 2025-02-11 15:00:00 | 359.00 | 2025-02-12 09:15:00 | 345.00 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2025-02-12 11:45:00 | 359.00 | 2025-02-14 12:15:00 | 348.50 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest1 | 2025-02-20 09:15:00 | 356.70 | 2025-02-24 09:15:00 | 357.20 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-02-21 11:30:00 | 362.15 | 2025-02-25 11:15:00 | 356.70 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-02-24 11:30:00 | 361.40 | 2025-02-25 11:15:00 | 356.70 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-02-24 12:45:00 | 361.30 | 2025-02-25 11:15:00 | 356.70 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-02-25 09:30:00 | 361.35 | 2025-02-25 11:15:00 | 356.70 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-02-28 09:15:00 | 349.90 | 2025-03-03 10:15:00 | 332.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-28 09:15:00 | 349.90 | 2025-03-03 12:15:00 | 347.50 | STOP_HIT | 0.50 | 0.69% |
| SELL | retest2 | 2025-03-04 09:15:00 | 350.60 | 2025-03-04 10:15:00 | 355.40 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-03-04 10:15:00 | 351.20 | 2025-03-04 10:15:00 | 355.40 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest1 | 2025-03-05 09:15:00 | 358.50 | 2025-03-06 10:15:00 | 354.60 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-03-06 13:45:00 | 356.35 | 2025-03-06 15:15:00 | 353.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-03-19 13:15:00 | 287.20 | 2025-03-19 14:15:00 | 294.40 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-03-27 12:15:00 | 329.40 | 2025-03-27 14:15:00 | 362.34 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-08 10:15:00 | 312.50 | 2025-04-08 13:15:00 | 324.15 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2025-04-09 09:15:00 | 310.90 | 2025-04-11 11:15:00 | 322.65 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2025-05-06 10:30:00 | 325.30 | 2025-05-07 09:15:00 | 309.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 12:15:00 | 327.30 | 2025-05-07 09:15:00 | 310.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 10:30:00 | 325.30 | 2025-05-08 09:15:00 | 321.30 | STOP_HIT | 0.50 | 1.23% |
| SELL | retest2 | 2025-05-06 12:15:00 | 327.30 | 2025-05-08 09:15:00 | 321.30 | STOP_HIT | 0.50 | 1.83% |
| BUY | retest2 | 2025-05-19 10:15:00 | 375.30 | 2025-05-22 12:15:00 | 371.05 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-05-19 12:45:00 | 375.05 | 2025-05-22 12:15:00 | 371.05 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-05-19 14:15:00 | 375.45 | 2025-05-22 12:15:00 | 371.05 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-05-20 11:15:00 | 375.25 | 2025-05-22 12:15:00 | 371.05 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-05-21 09:30:00 | 372.50 | 2025-05-22 12:15:00 | 371.05 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-05-21 11:30:00 | 374.40 | 2025-05-22 12:15:00 | 371.05 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-05-22 10:45:00 | 371.15 | 2025-05-22 12:15:00 | 371.05 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-05-29 09:15:00 | 390.05 | 2025-05-30 10:15:00 | 382.30 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-06-02 14:30:00 | 379.25 | 2025-06-10 11:15:00 | 382.50 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-06-02 15:00:00 | 378.50 | 2025-06-10 11:15:00 | 382.50 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-06-03 09:30:00 | 379.05 | 2025-06-10 11:15:00 | 382.50 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-06-04 09:45:00 | 379.35 | 2025-06-10 11:15:00 | 382.50 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-06-23 09:15:00 | 372.60 | 2025-06-23 14:15:00 | 377.50 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-06-23 11:30:00 | 374.60 | 2025-06-23 14:15:00 | 377.50 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-07-10 10:15:00 | 364.55 | 2025-07-14 12:15:00 | 346.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-10 10:15:00 | 364.55 | 2025-07-15 09:15:00 | 353.15 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2025-07-23 10:30:00 | 348.00 | 2025-07-28 12:15:00 | 330.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 15:00:00 | 348.00 | 2025-07-28 12:15:00 | 330.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 10:30:00 | 347.70 | 2025-07-28 12:15:00 | 330.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 10:30:00 | 348.00 | 2025-07-29 15:15:00 | 332.45 | STOP_HIT | 0.50 | 4.47% |
| SELL | retest2 | 2025-07-23 15:00:00 | 348.00 | 2025-07-29 15:15:00 | 332.45 | STOP_HIT | 0.50 | 4.47% |
| SELL | retest2 | 2025-07-24 10:30:00 | 347.70 | 2025-07-29 15:15:00 | 332.45 | STOP_HIT | 0.50 | 4.39% |
| BUY | retest2 | 2025-08-07 10:45:00 | 353.45 | 2025-08-14 14:15:00 | 360.30 | STOP_HIT | 1.00 | 1.94% |
| BUY | retest2 | 2025-08-25 09:15:00 | 372.85 | 2025-08-26 09:15:00 | 366.00 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-09-05 09:15:00 | 363.40 | 2025-09-05 10:15:00 | 355.00 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-09-17 09:15:00 | 372.50 | 2025-09-22 12:15:00 | 368.10 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-09-17 15:15:00 | 373.10 | 2025-09-22 12:15:00 | 368.10 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-09-22 10:00:00 | 369.45 | 2025-09-22 12:15:00 | 368.10 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-09-22 11:30:00 | 368.85 | 2025-09-22 12:15:00 | 368.10 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-10-07 15:15:00 | 324.35 | 2025-10-09 10:15:00 | 328.20 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-10-08 11:00:00 | 325.70 | 2025-10-09 10:15:00 | 328.20 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-10-08 15:15:00 | 324.10 | 2025-10-09 10:15:00 | 328.20 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-10-15 11:45:00 | 323.20 | 2025-10-17 11:15:00 | 327.30 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-10-15 13:30:00 | 323.35 | 2025-10-17 11:15:00 | 327.30 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-10-16 11:00:00 | 323.45 | 2025-10-17 11:15:00 | 327.30 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-10-16 11:30:00 | 323.30 | 2025-10-17 11:15:00 | 327.30 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-10-17 09:45:00 | 323.15 | 2025-10-17 12:15:00 | 327.55 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-10-29 14:45:00 | 348.10 | 2025-11-04 13:15:00 | 348.80 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-11-17 09:15:00 | 364.60 | 2025-11-18 09:15:00 | 359.80 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-11-17 11:15:00 | 364.15 | 2025-11-18 09:15:00 | 359.80 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-11-17 13:30:00 | 363.70 | 2025-11-18 09:15:00 | 359.80 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-11-17 14:00:00 | 364.15 | 2025-11-18 09:15:00 | 359.80 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-11-24 09:15:00 | 357.25 | 2025-11-24 15:15:00 | 339.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 09:15:00 | 357.25 | 2025-11-25 11:15:00 | 350.55 | STOP_HIT | 0.50 | 1.88% |
| SELL | retest2 | 2025-12-10 12:30:00 | 338.15 | 2025-12-12 09:15:00 | 344.10 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-01-14 10:30:00 | 316.50 | 2026-01-14 11:15:00 | 320.25 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-01-23 09:45:00 | 311.95 | 2026-01-28 11:15:00 | 314.40 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-01-27 09:15:00 | 311.20 | 2026-01-28 11:15:00 | 314.40 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-01-27 10:15:00 | 310.60 | 2026-01-28 11:15:00 | 314.40 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-01-30 11:30:00 | 322.00 | 2026-02-01 11:15:00 | 312.05 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2026-01-30 12:00:00 | 321.60 | 2026-02-01 11:15:00 | 312.05 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2026-01-30 12:30:00 | 321.55 | 2026-02-01 11:15:00 | 312.05 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2026-01-30 13:00:00 | 321.95 | 2026-02-01 11:15:00 | 312.05 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2026-02-03 12:45:00 | 309.95 | 2026-02-06 09:15:00 | 294.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-03 12:45:00 | 309.95 | 2026-02-09 10:15:00 | 293.90 | STOP_HIT | 0.50 | 5.18% |
| SELL | retest2 | 2026-02-23 12:15:00 | 244.05 | 2026-02-24 10:15:00 | 231.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 12:15:00 | 244.05 | 2026-02-25 13:15:00 | 219.65 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-06 14:00:00 | 222.00 | 2026-03-09 09:15:00 | 215.79 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2026-03-10 11:15:00 | 211.51 | 2026-03-11 14:15:00 | 218.01 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2026-03-11 10:30:00 | 211.37 | 2026-03-11 14:15:00 | 218.01 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2026-03-13 10:45:00 | 223.00 | 2026-03-16 09:15:00 | 215.68 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2026-03-13 13:30:00 | 222.80 | 2026-03-16 09:15:00 | 215.68 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2026-03-20 09:15:00 | 221.91 | 2026-03-20 10:15:00 | 219.31 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-04-20 11:15:00 | 236.01 | 2026-04-20 15:15:00 | 232.30 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-04-29 13:30:00 | 215.30 | 2026-05-04 09:15:00 | 216.29 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2026-04-30 09:45:00 | 215.70 | 2026-05-04 10:15:00 | 217.10 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2026-04-30 10:45:00 | 214.88 | 2026-05-04 10:15:00 | 217.10 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-04-30 11:45:00 | 215.63 | 2026-05-04 10:15:00 | 217.10 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2026-04-30 15:00:00 | 214.00 | 2026-05-04 10:15:00 | 217.10 | STOP_HIT | 1.00 | -1.45% |
