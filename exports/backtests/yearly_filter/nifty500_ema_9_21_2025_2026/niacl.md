# The New India Assurance Company Ltd. (NIACL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 163.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 87 |
| ALERT1 | 53 |
| ALERT2 | 53 |
| ALERT2_SKIP | 32 |
| ALERT3 | 121 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 47 |
| PARTIAL | 10 |
| TARGET_HIT | 1 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 28 / 27
- **Target hits / Stop hits / Partials:** 1 / 44 / 10
- **Avg / median % per leg:** 1.23% / 0.02%
- **Sum % (uncompounded):** 67.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 1 | 11.1% | 0 | 9 | 0 | -1.73% | -15.6% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.18% | -3.2% |
| BUY @ 3rd Alert (retest2) | 8 | 1 | 12.5% | 0 | 8 | 0 | -1.55% | -12.4% |
| SELL (all) | 46 | 27 | 58.7% | 1 | 35 | 10 | 1.80% | 83.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 46 | 27 | 58.7% | 1 | 35 | 10 | 1.80% | 83.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.18% | -3.2% |
| retest2 (combined) | 54 | 28 | 51.9% | 1 | 43 | 10 | 1.31% | 70.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 168.58 | 165.48 | 165.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 170.84 | 167.85 | 166.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 13:15:00 | 171.75 | 172.12 | 170.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 13:30:00 | 171.44 | 172.12 | 170.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 178.49 | 180.36 | 178.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 178.49 | 180.36 | 178.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 182.81 | 180.85 | 178.95 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 174.77 | 178.04 | 178.23 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 179.00 | 178.27 | 178.21 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 177.75 | 178.16 | 178.17 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 179.00 | 178.33 | 178.25 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 177.00 | 178.03 | 178.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 13:15:00 | 176.70 | 177.76 | 177.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 180.61 | 177.92 | 177.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 180.61 | 177.92 | 177.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 180.61 | 177.92 | 177.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 181.04 | 177.92 | 177.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 186.69 | 179.67 | 178.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 188.20 | 183.69 | 182.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 10:15:00 | 185.78 | 185.87 | 184.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 10:30:00 | 185.68 | 185.87 | 184.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 184.00 | 185.94 | 185.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:45:00 | 183.85 | 185.94 | 185.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 183.72 | 185.50 | 185.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 183.03 | 185.50 | 185.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 187.16 | 185.84 | 185.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 15:15:00 | 188.15 | 186.76 | 186.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 13:15:00 | 184.72 | 185.97 | 186.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 13:15:00 | 184.72 | 185.97 | 186.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 15:15:00 | 184.00 | 185.42 | 185.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 12:15:00 | 186.21 | 184.96 | 185.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 12:15:00 | 186.21 | 184.96 | 185.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 186.21 | 184.96 | 185.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:45:00 | 186.50 | 184.96 | 185.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 185.30 | 185.03 | 185.35 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 190.50 | 186.33 | 185.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 193.99 | 192.12 | 190.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 197.72 | 197.90 | 195.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 09:45:00 | 197.85 | 197.90 | 195.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 194.06 | 196.92 | 195.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 194.06 | 196.92 | 195.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 194.38 | 196.41 | 195.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 194.38 | 196.41 | 195.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 191.22 | 194.51 | 194.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 188.50 | 192.86 | 193.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 187.07 | 186.98 | 188.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 15:15:00 | 186.94 | 186.98 | 188.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 187.79 | 187.13 | 188.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:30:00 | 186.03 | 186.57 | 188.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 13:15:00 | 176.73 | 180.26 | 182.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 180.48 | 179.39 | 181.47 | SL hit (close>ema200) qty=0.50 sl=179.39 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 185.92 | 182.19 | 182.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 15:15:00 | 186.13 | 182.98 | 182.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 186.79 | 187.43 | 185.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 14:00:00 | 186.79 | 187.43 | 185.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 187.10 | 188.65 | 187.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 187.10 | 188.65 | 187.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 186.87 | 188.29 | 187.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:00:00 | 186.87 | 188.29 | 187.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 185.69 | 187.77 | 187.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:45:00 | 185.83 | 187.77 | 187.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 13:15:00 | 185.26 | 187.27 | 187.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 10:15:00 | 183.71 | 185.84 | 186.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 09:15:00 | 186.18 | 184.70 | 185.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 09:15:00 | 186.18 | 184.70 | 185.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 186.18 | 184.70 | 185.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:00:00 | 186.18 | 184.70 | 185.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 189.17 | 185.60 | 185.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:45:00 | 191.44 | 185.60 | 185.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 11:15:00 | 190.08 | 186.49 | 186.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 14:15:00 | 192.47 | 188.59 | 187.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 11:15:00 | 188.94 | 189.41 | 188.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 12:00:00 | 188.94 | 189.41 | 188.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 187.55 | 189.82 | 188.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 187.55 | 189.82 | 188.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 188.32 | 189.52 | 188.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:15:00 | 188.78 | 189.52 | 188.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 14:45:00 | 189.15 | 189.18 | 188.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 187.73 | 188.64 | 188.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 10:15:00 | 187.73 | 188.64 | 188.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 11:15:00 | 187.52 | 188.41 | 188.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 186.34 | 185.80 | 186.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 14:45:00 | 186.34 | 185.80 | 186.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 184.88 | 185.61 | 186.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:30:00 | 184.65 | 185.22 | 186.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 15:15:00 | 188.00 | 185.81 | 186.12 | SL hit (close>static) qty=1.00 sl=187.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 14:15:00 | 188.79 | 186.65 | 186.42 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 184.98 | 186.80 | 187.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 183.59 | 186.15 | 186.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 13:15:00 | 184.10 | 183.68 | 184.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 14:00:00 | 184.10 | 183.68 | 184.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 183.37 | 183.62 | 184.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:30:00 | 184.10 | 183.62 | 184.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 184.69 | 183.75 | 184.46 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-07-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 15:15:00 | 184.90 | 184.73 | 184.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 186.99 | 185.18 | 184.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 15:15:00 | 185.70 | 186.08 | 185.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 15:15:00 | 185.70 | 186.08 | 185.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 185.70 | 186.08 | 185.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 14:30:00 | 186.53 | 185.73 | 185.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 184.05 | 185.33 | 185.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 184.05 | 185.33 | 185.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 15:15:00 | 183.70 | 184.29 | 184.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 11:15:00 | 183.97 | 183.81 | 184.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 11:15:00 | 183.97 | 183.81 | 184.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 183.97 | 183.81 | 184.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:00:00 | 183.97 | 183.81 | 184.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 183.50 | 183.75 | 184.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 13:15:00 | 183.00 | 183.75 | 184.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 14:15:00 | 184.45 | 183.62 | 184.08 | SL hit (close>static) qty=1.00 sl=184.25 alert=retest2 |

### Cycle 19 — BUY (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 14:15:00 | 185.35 | 184.20 | 184.17 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 14:15:00 | 183.95 | 184.21 | 184.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 15:15:00 | 183.47 | 184.07 | 184.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 14:15:00 | 173.93 | 173.67 | 176.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:30:00 | 173.10 | 173.67 | 176.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 186.80 | 176.24 | 177.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:30:00 | 186.00 | 176.24 | 177.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 189.79 | 178.95 | 178.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 12:15:00 | 193.30 | 183.43 | 180.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 15:15:00 | 196.45 | 201.18 | 195.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 15:15:00 | 196.45 | 201.18 | 195.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 196.45 | 201.18 | 195.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:45:00 | 195.25 | 200.20 | 195.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 195.63 | 199.28 | 195.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:45:00 | 195.45 | 199.28 | 195.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 191.80 | 197.79 | 194.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:00:00 | 191.80 | 197.79 | 194.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 193.00 | 196.83 | 194.69 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 188.49 | 193.20 | 193.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 184.97 | 191.55 | 192.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 09:15:00 | 200.78 | 190.53 | 190.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 09:15:00 | 200.78 | 190.53 | 190.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 200.78 | 190.53 | 190.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:30:00 | 202.69 | 190.53 | 190.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 201.82 | 192.79 | 191.97 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 13:15:00 | 191.58 | 193.83 | 193.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 189.67 | 192.58 | 193.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 193.00 | 190.15 | 191.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 193.00 | 190.15 | 191.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 193.00 | 190.15 | 191.40 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 13:15:00 | 193.50 | 192.20 | 192.09 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 189.90 | 191.64 | 191.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 10:15:00 | 187.51 | 190.49 | 191.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 15:15:00 | 187.00 | 185.09 | 186.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 15:15:00 | 187.00 | 185.09 | 186.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 187.00 | 185.09 | 186.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 190.88 | 185.09 | 186.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 197.17 | 187.50 | 187.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 204.53 | 195.32 | 193.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 15:15:00 | 198.40 | 199.23 | 196.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 09:15:00 | 195.40 | 199.23 | 196.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 196.73 | 198.73 | 196.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 195.60 | 198.73 | 196.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 195.80 | 198.14 | 196.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 195.81 | 198.14 | 196.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 196.22 | 197.76 | 196.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:45:00 | 196.50 | 197.76 | 196.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 196.00 | 197.41 | 196.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:30:00 | 196.15 | 197.41 | 196.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 196.30 | 196.81 | 196.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 195.00 | 196.81 | 196.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 193.91 | 196.23 | 196.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 12:15:00 | 193.73 | 195.07 | 195.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 10:15:00 | 191.10 | 190.91 | 192.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 10:45:00 | 191.11 | 190.91 | 192.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 191.81 | 191.06 | 191.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:30:00 | 192.50 | 191.06 | 191.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 192.82 | 191.41 | 192.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:15:00 | 189.25 | 191.41 | 192.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 15:00:00 | 190.86 | 189.66 | 190.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 193.30 | 190.63 | 190.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 193.30 | 190.63 | 190.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 195.54 | 192.70 | 191.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 13:15:00 | 199.10 | 199.35 | 196.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 14:00:00 | 199.10 | 199.35 | 196.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 196.00 | 198.71 | 197.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 196.00 | 198.71 | 197.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 196.17 | 198.21 | 197.00 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 15:15:00 | 194.95 | 196.22 | 196.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 10:15:00 | 193.32 | 195.27 | 195.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 193.46 | 192.04 | 193.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 193.46 | 192.04 | 193.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 193.46 | 192.04 | 193.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:00:00 | 193.06 | 192.24 | 193.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:45:00 | 192.85 | 192.66 | 192.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 193.60 | 192.71 | 192.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 193.60 | 192.71 | 192.65 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 10:15:00 | 192.20 | 192.78 | 192.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 11:15:00 | 192.08 | 192.64 | 192.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 12:15:00 | 192.80 | 192.68 | 192.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 12:15:00 | 192.80 | 192.68 | 192.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 192.80 | 192.68 | 192.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 13:00:00 | 192.80 | 192.68 | 192.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 192.74 | 192.69 | 192.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 13:30:00 | 193.21 | 192.69 | 192.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 192.61 | 192.67 | 192.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:45:00 | 192.86 | 192.67 | 192.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 192.46 | 192.63 | 192.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 192.88 | 192.63 | 192.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 192.03 | 192.51 | 192.68 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 194.56 | 192.92 | 192.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 14:15:00 | 201.05 | 196.27 | 194.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 198.40 | 198.76 | 197.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 15:00:00 | 198.40 | 198.76 | 197.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 198.15 | 198.64 | 197.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 196.67 | 198.45 | 197.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 196.40 | 198.04 | 197.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:45:00 | 196.80 | 198.04 | 197.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 197.05 | 197.84 | 197.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 196.45 | 197.84 | 197.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 197.30 | 197.73 | 197.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:45:00 | 196.92 | 197.73 | 197.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 196.75 | 197.54 | 197.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:00:00 | 196.75 | 197.54 | 197.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 195.69 | 197.17 | 197.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:45:00 | 196.22 | 197.17 | 197.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 15:15:00 | 196.00 | 196.93 | 196.95 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 09:15:00 | 197.80 | 197.11 | 197.02 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 195.54 | 196.79 | 196.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 11:15:00 | 195.00 | 196.43 | 196.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 196.73 | 196.08 | 196.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 196.73 | 196.08 | 196.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 196.73 | 196.08 | 196.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 10:30:00 | 195.20 | 195.98 | 196.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:45:00 | 195.61 | 195.80 | 196.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 12:15:00 | 185.83 | 188.42 | 190.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 188.78 | 188.22 | 189.99 | SL hit (close>ema200) qty=0.50 sl=188.22 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 191.04 | 189.54 | 189.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 11:15:00 | 192.11 | 190.40 | 190.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 191.86 | 191.97 | 191.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 11:00:00 | 191.86 | 191.97 | 191.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 191.13 | 191.81 | 191.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:45:00 | 191.05 | 191.81 | 191.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 190.70 | 191.58 | 191.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:30:00 | 190.93 | 191.58 | 191.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 191.43 | 191.55 | 191.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:45:00 | 193.10 | 191.76 | 191.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 190.33 | 191.43 | 191.20 | SL hit (close<static) qty=1.00 sl=190.70 alert=retest2 |

### Cycle 38 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 189.54 | 190.76 | 190.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 15:15:00 | 189.10 | 190.20 | 190.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 15:15:00 | 189.45 | 189.43 | 189.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 09:15:00 | 190.32 | 189.43 | 189.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 191.00 | 189.75 | 190.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 190.71 | 189.75 | 190.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 189.87 | 189.77 | 189.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:30:00 | 190.14 | 189.77 | 189.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 189.37 | 189.69 | 189.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:30:00 | 189.99 | 189.69 | 189.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 14:15:00 | 192.71 | 190.13 | 190.06 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 188.78 | 190.14 | 190.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 187.75 | 189.66 | 190.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 188.76 | 188.70 | 189.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 188.76 | 188.70 | 189.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 188.76 | 188.70 | 189.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:15:00 | 188.00 | 188.54 | 188.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 188.78 | 187.59 | 187.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 188.78 | 187.59 | 187.53 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 187.25 | 187.61 | 187.65 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 190.11 | 187.85 | 187.70 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 15:15:00 | 189.05 | 189.66 | 189.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 188.96 | 189.52 | 189.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 09:15:00 | 184.80 | 183.97 | 185.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 184.80 | 183.97 | 185.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 184.80 | 183.97 | 185.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 14:15:00 | 183.48 | 184.19 | 185.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 14:45:00 | 183.38 | 184.06 | 185.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 15:15:00 | 182.20 | 184.06 | 185.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:30:00 | 183.43 | 183.41 | 184.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 182.21 | 181.54 | 182.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 182.21 | 181.54 | 182.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 182.87 | 181.80 | 182.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 183.00 | 181.80 | 182.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 182.24 | 181.89 | 182.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-10 14:15:00 | 183.21 | 182.86 | 182.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 183.21 | 182.86 | 182.81 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 181.05 | 182.58 | 182.70 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 184.90 | 182.81 | 182.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 14:15:00 | 185.60 | 184.33 | 183.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 184.39 | 184.70 | 184.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 12:15:00 | 184.39 | 184.70 | 184.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 184.39 | 184.70 | 184.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:30:00 | 183.73 | 184.70 | 184.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 184.15 | 184.59 | 184.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 184.15 | 184.59 | 184.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 184.69 | 184.61 | 184.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 179.75 | 184.61 | 184.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 179.86 | 183.66 | 183.79 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 13:15:00 | 182.29 | 182.02 | 182.00 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 181.75 | 181.98 | 181.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 179.62 | 181.51 | 181.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 176.51 | 175.74 | 177.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 176.51 | 175.74 | 177.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 176.51 | 175.74 | 177.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:15:00 | 175.03 | 175.80 | 176.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 09:15:00 | 166.28 | 169.54 | 170.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-08 09:15:00 | 157.53 | 160.11 | 162.51 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 161.30 | 159.63 | 159.62 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 11:15:00 | 159.51 | 159.82 | 159.86 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 164.06 | 160.57 | 160.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 171.00 | 164.31 | 162.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 15:15:00 | 166.99 | 167.04 | 165.25 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 09:15:00 | 169.71 | 167.04 | 165.25 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 166.37 | 167.30 | 166.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 166.37 | 167.30 | 166.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 166.90 | 167.22 | 166.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 165.80 | 167.22 | 166.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 164.31 | 166.64 | 166.13 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 164.31 | 166.64 | 166.13 | SL hit (close<ema400) qty=1.00 sl=166.13 alert=retest1 |

### Cycle 54 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 163.21 | 165.47 | 165.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 163.00 | 164.97 | 165.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 155.11 | 154.64 | 156.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:45:00 | 155.22 | 154.64 | 156.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 156.29 | 153.22 | 154.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:45:00 | 156.89 | 153.22 | 154.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 154.29 | 153.43 | 154.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:30:00 | 156.87 | 153.43 | 154.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 154.03 | 153.55 | 154.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 13:30:00 | 153.85 | 153.60 | 154.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:30:00 | 153.90 | 153.72 | 153.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 156.67 | 153.34 | 153.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 156.67 | 153.34 | 153.28 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 154.10 | 155.32 | 155.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 153.64 | 154.75 | 155.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 154.05 | 153.41 | 153.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 14:15:00 | 154.05 | 153.41 | 153.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 154.05 | 153.41 | 153.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 154.40 | 153.41 | 153.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 154.00 | 153.53 | 153.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 153.29 | 153.53 | 153.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 153.13 | 153.45 | 153.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:45:00 | 151.84 | 153.06 | 153.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 12:15:00 | 151.08 | 150.39 | 150.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 12:15:00 | 151.08 | 150.39 | 150.37 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 14:15:00 | 150.09 | 150.31 | 150.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 10:15:00 | 149.19 | 149.86 | 150.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 11:15:00 | 149.92 | 149.88 | 150.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 12:00:00 | 149.92 | 149.88 | 150.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 150.30 | 149.96 | 150.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:00:00 | 150.30 | 149.96 | 150.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 150.15 | 150.00 | 150.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 149.82 | 150.00 | 150.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 149.60 | 150.00 | 150.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 11:15:00 | 150.86 | 149.99 | 150.05 | SL hit (close>static) qty=1.00 sl=150.60 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 12:15:00 | 150.53 | 150.10 | 150.09 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 149.98 | 150.07 | 150.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 148.37 | 149.71 | 149.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 145.98 | 143.84 | 144.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 145.98 | 143.84 | 144.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 145.98 | 143.84 | 144.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 146.21 | 143.84 | 144.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 145.19 | 144.11 | 144.93 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 146.58 | 145.35 | 145.33 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 144.31 | 145.26 | 145.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 143.25 | 144.86 | 145.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 144.85 | 143.73 | 144.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 144.85 | 143.73 | 144.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 144.85 | 143.73 | 144.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 145.52 | 143.73 | 144.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 144.62 | 143.91 | 144.22 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 146.80 | 144.83 | 144.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 147.45 | 145.87 | 145.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 146.40 | 146.44 | 145.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 11:00:00 | 146.40 | 146.44 | 145.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 147.11 | 146.75 | 146.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:15:00 | 145.39 | 146.75 | 146.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 147.80 | 146.96 | 146.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 149.77 | 147.76 | 146.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 142.94 | 147.33 | 147.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 142.94 | 147.33 | 147.89 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 147.65 | 147.18 | 147.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 148.79 | 147.62 | 147.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 147.80 | 148.39 | 147.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 147.80 | 148.39 | 147.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 147.80 | 148.39 | 147.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 148.17 | 148.39 | 147.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 146.80 | 148.07 | 147.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 146.80 | 148.07 | 147.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 146.72 | 147.80 | 147.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 12:30:00 | 147.15 | 147.73 | 147.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 13:15:00 | 147.18 | 147.62 | 147.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 147.18 | 147.62 | 147.65 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 15:15:00 | 148.00 | 147.71 | 147.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 10:15:00 | 150.95 | 148.56 | 148.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 156.84 | 157.21 | 154.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 155.12 | 156.34 | 155.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 155.12 | 156.34 | 155.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 153.91 | 156.34 | 155.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 154.83 | 156.04 | 155.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 154.83 | 156.04 | 155.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 153.70 | 155.57 | 155.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 153.70 | 155.57 | 155.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 15:15:00 | 154.63 | 154.97 | 154.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 151.06 | 154.19 | 154.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 152.45 | 151.21 | 151.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 152.45 | 151.21 | 151.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 152.45 | 151.21 | 151.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 152.15 | 151.21 | 151.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 152.15 | 151.40 | 151.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 152.89 | 151.40 | 151.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 152.00 | 151.96 | 152.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 152.07 | 151.96 | 152.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 151.75 | 151.91 | 152.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:15:00 | 151.17 | 151.91 | 152.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 13:30:00 | 151.50 | 151.63 | 151.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 14:00:00 | 151.54 | 151.63 | 151.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 15:15:00 | 154.50 | 152.33 | 152.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 154.50 | 152.33 | 152.13 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 150.74 | 151.90 | 151.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 150.61 | 151.42 | 151.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 152.45 | 150.81 | 151.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 152.45 | 150.81 | 151.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 152.45 | 150.81 | 151.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 152.06 | 150.81 | 151.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 151.70 | 150.99 | 151.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:30:00 | 151.12 | 151.13 | 151.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 152.08 | 151.32 | 151.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 152.08 | 151.32 | 151.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 15:15:00 | 152.24 | 151.50 | 151.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 151.13 | 151.43 | 151.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 151.13 | 151.43 | 151.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 151.13 | 151.43 | 151.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 150.70 | 151.43 | 151.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 151.25 | 151.39 | 151.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 10:45:00 | 150.76 | 151.39 | 151.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 150.50 | 151.21 | 151.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 149.36 | 150.84 | 151.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 150.99 | 150.35 | 150.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 150.99 | 150.35 | 150.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 150.99 | 150.35 | 150.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 150.99 | 150.35 | 150.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 149.80 | 150.24 | 150.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:15:00 | 149.67 | 150.20 | 150.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 10:45:00 | 149.70 | 149.53 | 149.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 149.65 | 149.37 | 149.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 142.19 | 146.56 | 147.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 142.21 | 146.56 | 147.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 142.17 | 146.56 | 147.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 139.75 | 138.81 | 140.49 | SL hit (close>ema200) qty=0.50 sl=138.81 alert=retest2 |

### Cycle 73 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 138.29 | 137.42 | 137.36 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 136.74 | 137.42 | 137.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 136.55 | 137.25 | 137.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 137.10 | 136.82 | 137.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 137.10 | 136.82 | 137.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 137.10 | 136.82 | 137.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 137.10 | 136.82 | 137.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 138.20 | 137.09 | 137.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 138.20 | 137.09 | 137.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 138.48 | 137.37 | 137.32 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 136.77 | 137.31 | 137.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 135.26 | 136.81 | 137.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 133.07 | 132.88 | 134.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 09:30:00 | 132.85 | 132.88 | 134.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 134.29 | 133.38 | 133.85 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 135.45 | 134.12 | 134.10 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 132.05 | 133.91 | 134.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 130.50 | 132.48 | 133.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 132.29 | 131.77 | 132.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 132.29 | 131.77 | 132.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 132.29 | 131.77 | 132.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:45:00 | 132.68 | 131.77 | 132.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 132.63 | 131.94 | 132.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:30:00 | 132.56 | 131.94 | 132.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 132.41 | 132.04 | 132.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 131.90 | 132.04 | 132.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:00:00 | 131.90 | 132.01 | 132.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:45:00 | 131.71 | 131.93 | 132.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 128.44 | 132.29 | 132.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 125.30 | 129.79 | 131.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 125.30 | 129.79 | 131.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 125.12 | 129.79 | 131.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-24 10:15:00 | 122.02 | 125.26 | 127.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 13:15:00 | 124.78 | 124.76 | 126.97 | SL hit (close>ema200) qty=0.50 sl=124.76 alert=retest2 |

### Cycle 79 — BUY (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 14:15:00 | 128.03 | 127.41 | 127.36 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 124.51 | 126.93 | 127.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 124.00 | 126.34 | 126.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 123.03 | 120.07 | 122.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 123.03 | 120.07 | 122.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 123.03 | 120.07 | 122.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 123.54 | 120.07 | 122.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 122.15 | 120.49 | 122.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 119.11 | 122.64 | 122.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 123.38 | 121.90 | 122.05 | SL hit (close>static) qty=1.00 sl=123.14 alert=retest2 |

### Cycle 81 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 123.50 | 122.22 | 122.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 11:15:00 | 123.68 | 122.74 | 122.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 123.19 | 123.46 | 122.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 123.19 | 123.46 | 122.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 123.19 | 123.46 | 122.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:45:00 | 123.28 | 123.46 | 122.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 165.86 | 167.79 | 166.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 10:30:00 | 167.30 | 167.61 | 166.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 164.00 | 165.82 | 165.77 | SL hit (close<static) qty=1.00 sl=164.13 alert=retest2 |

### Cycle 82 — SELL (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 09:15:00 | 165.30 | 165.71 | 165.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 14:15:00 | 163.96 | 165.01 | 165.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 166.89 | 164.23 | 164.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 166.89 | 164.23 | 164.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 166.89 | 164.23 | 164.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:00:00 | 166.89 | 164.23 | 164.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 10:15:00 | 170.48 | 165.48 | 165.04 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 163.13 | 165.17 | 165.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 161.89 | 163.57 | 164.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 163.93 | 163.64 | 164.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 15:00:00 | 163.93 | 163.64 | 164.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 164.10 | 163.73 | 164.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 163.45 | 163.73 | 164.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 164.52 | 163.89 | 164.32 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 165.36 | 164.65 | 164.59 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 12:15:00 | 164.00 | 164.62 | 164.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 13:15:00 | 163.70 | 164.44 | 164.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 164.12 | 164.05 | 164.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 164.12 | 164.05 | 164.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 164.12 | 164.05 | 164.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:45:00 | 162.39 | 163.61 | 164.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:00:00 | 162.30 | 161.28 | 161.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 162.17 | 161.53 | 161.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 11:15:00 | 162.17 | 161.53 | 161.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 12:15:00 | 162.64 | 161.76 | 161.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 14:15:00 | 161.22 | 161.72 | 161.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 14:15:00 | 161.22 | 161.72 | 161.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 161.22 | 161.72 | 161.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 15:00:00 | 161.22 | 161.72 | 161.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 162.20 | 161.82 | 161.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:30:00 | 162.95 | 162.01 | 161.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 12:30:00 | 163.17 | 162.29 | 161.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:00:00 | 163.26 | 162.29 | 161.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 09:45:00 | 167.41 | 2025-05-12 12:15:00 | 168.58 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-06-02 15:15:00 | 188.15 | 2025-06-03 13:15:00 | 184.72 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-06-17 11:30:00 | 186.03 | 2025-06-19 13:15:00 | 176.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:30:00 | 186.03 | 2025-06-20 09:15:00 | 180.48 | STOP_HIT | 0.50 | 2.98% |
| BUY | retest2 | 2025-07-02 11:15:00 | 188.78 | 2025-07-03 10:15:00 | 187.73 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-07-02 14:45:00 | 189.15 | 2025-07-03 10:15:00 | 187.73 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-07-07 11:30:00 | 184.65 | 2025-07-07 15:15:00 | 188.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-07-17 14:30:00 | 186.53 | 2025-07-18 09:15:00 | 184.05 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-07-22 13:15:00 | 183.00 | 2025-07-22 14:15:00 | 184.45 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-08-29 09:15:00 | 189.25 | 2025-09-02 09:15:00 | 193.30 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-09-01 15:00:00 | 190.86 | 2025-09-02 09:15:00 | 193.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-09-10 11:00:00 | 193.06 | 2025-09-15 09:15:00 | 193.60 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-09-11 10:45:00 | 192.85 | 2025-09-15 09:15:00 | 193.60 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-09-25 10:30:00 | 195.20 | 2025-09-29 12:15:00 | 185.83 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2025-09-25 10:30:00 | 195.20 | 2025-09-30 09:15:00 | 188.78 | STOP_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2025-09-25 12:45:00 | 195.61 | 2025-10-03 09:15:00 | 191.04 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest2 | 2025-10-07 14:45:00 | 193.10 | 2025-10-08 09:15:00 | 190.33 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-10-17 11:15:00 | 188.00 | 2025-10-23 10:15:00 | 188.78 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-11-04 14:15:00 | 183.48 | 2025-11-10 14:15:00 | 183.21 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-11-04 14:45:00 | 183.38 | 2025-11-10 14:15:00 | 183.21 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-11-04 15:15:00 | 182.20 | 2025-11-10 14:15:00 | 183.21 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-11-06 09:30:00 | 183.43 | 2025-11-10 14:15:00 | 183.21 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2025-11-24 14:15:00 | 175.03 | 2025-12-02 09:15:00 | 166.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 14:15:00 | 175.03 | 2025-12-08 09:15:00 | 157.53 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-12-16 09:15:00 | 169.71 | 2025-12-17 09:15:00 | 164.31 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-12-26 13:30:00 | 153.85 | 2025-12-31 09:15:00 | 156.67 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-12-29 10:30:00 | 153.90 | 2025-12-31 09:15:00 | 156.67 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-01-08 10:45:00 | 151.84 | 2026-01-13 12:15:00 | 151.08 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2026-01-14 14:15:00 | 149.82 | 2026-01-16 11:15:00 | 150.86 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-01-16 09:15:00 | 149.60 | 2026-01-16 11:15:00 | 150.86 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2026-01-30 10:30:00 | 149.77 | 2026-02-02 10:15:00 | 142.94 | STOP_HIT | 1.00 | -4.56% |
| BUY | retest2 | 2026-02-05 12:30:00 | 147.15 | 2026-02-05 13:15:00 | 147.18 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2026-02-18 10:15:00 | 151.17 | 2026-02-18 15:15:00 | 154.50 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2026-02-18 13:30:00 | 151.50 | 2026-02-18 15:15:00 | 154.50 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2026-02-18 14:00:00 | 151.54 | 2026-02-18 15:15:00 | 154.50 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2026-02-23 13:30:00 | 151.12 | 2026-02-23 14:15:00 | 152.08 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-02-25 12:15:00 | 149.67 | 2026-03-02 09:15:00 | 142.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 10:45:00 | 149.70 | 2026-03-02 09:15:00 | 142.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 11:30:00 | 149.65 | 2026-03-02 09:15:00 | 142.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 12:15:00 | 149.67 | 2026-03-05 14:15:00 | 139.75 | STOP_HIT | 0.50 | 6.63% |
| SELL | retest2 | 2026-02-26 10:45:00 | 149.70 | 2026-03-05 14:15:00 | 139.75 | STOP_HIT | 0.50 | 6.65% |
| SELL | retest2 | 2026-02-26 11:30:00 | 149.65 | 2026-03-05 14:15:00 | 139.75 | STOP_HIT | 0.50 | 6.62% |
| SELL | retest2 | 2026-03-20 12:15:00 | 131.90 | 2026-03-23 10:15:00 | 125.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:00:00 | 131.90 | 2026-03-23 10:15:00 | 125.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:45:00 | 131.71 | 2026-03-23 10:15:00 | 125.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-23 09:15:00 | 128.44 | 2026-03-24 10:15:00 | 122.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 131.90 | 2026-03-24 13:15:00 | 124.78 | STOP_HIT | 0.50 | 5.40% |
| SELL | retest2 | 2026-03-20 13:00:00 | 131.90 | 2026-03-24 13:15:00 | 124.78 | STOP_HIT | 0.50 | 5.40% |
| SELL | retest2 | 2026-03-20 13:45:00 | 131.71 | 2026-03-24 13:15:00 | 124.78 | STOP_HIT | 0.50 | 5.26% |
| SELL | retest2 | 2026-03-23 09:15:00 | 128.44 | 2026-03-24 13:15:00 | 124.78 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest2 | 2026-04-02 09:15:00 | 119.11 | 2026-04-02 14:15:00 | 123.38 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2026-04-02 14:30:00 | 121.79 | 2026-04-02 15:15:00 | 123.50 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-04-20 10:30:00 | 167.30 | 2026-04-20 15:15:00 | 164.00 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-04-29 13:45:00 | 162.39 | 2026-05-05 11:15:00 | 162.17 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2026-05-05 10:00:00 | 162.30 | 2026-05-05 11:15:00 | 162.17 | STOP_HIT | 1.00 | 0.08% |
