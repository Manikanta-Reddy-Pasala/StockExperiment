# BPCL (BPCL)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 303.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 226 |
| ALERT1 | 151 |
| ALERT2 | 151 |
| ALERT2_SKIP | 76 |
| ALERT3 | 402 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 177 |
| PARTIAL | 17 |
| TARGET_HIT | 4 |
| STOP_HIT | 182 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 203 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 69 / 134
- **Target hits / Stop hits / Partials:** 4 / 182 / 17
- **Avg / median % per leg:** 0.17% / -0.80%
- **Sum % (uncompounded):** 34.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 72 | 22 | 30.6% | 1 | 71 | 0 | -0.61% | -44.2% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 0 | 8 | 0 | -0.52% | -4.2% |
| BUY @ 3rd Alert (retest2) | 64 | 17 | 26.6% | 1 | 63 | 0 | -0.63% | -40.0% |
| SELL (all) | 131 | 47 | 35.9% | 3 | 111 | 17 | 0.60% | 78.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.99% | -1.0% |
| SELL @ 3rd Alert (retest2) | 130 | 47 | 36.2% | 3 | 110 | 17 | 0.61% | 79.5% |
| retest1 (combined) | 9 | 5 | 55.6% | 0 | 9 | 0 | -0.58% | -5.2% |
| retest2 (combined) | 194 | 64 | 33.0% | 4 | 173 | 17 | 0.20% | 39.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 09:15:00 | 183.73 | 181.45 | 181.43 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 14:15:00 | 180.53 | 181.70 | 181.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 15:15:00 | 180.20 | 181.40 | 181.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 15:15:00 | 180.48 | 180.31 | 180.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-22 09:15:00 | 179.60 | 180.31 | 180.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 180.35 | 180.32 | 180.78 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 09:15:00 | 184.90 | 181.55 | 181.18 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 09:15:00 | 181.23 | 182.31 | 182.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-25 12:15:00 | 180.58 | 181.64 | 182.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 14:15:00 | 181.70 | 181.53 | 181.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 14:15:00 | 181.70 | 181.53 | 181.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 14:15:00 | 181.70 | 181.53 | 181.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 15:00:00 | 181.70 | 181.53 | 181.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 15:15:00 | 182.00 | 181.63 | 181.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 09:15:00 | 182.08 | 181.63 | 181.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 182.05 | 181.71 | 181.91 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 13:15:00 | 182.35 | 182.03 | 182.02 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 09:15:00 | 181.58 | 181.96 | 181.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-29 11:15:00 | 181.25 | 181.77 | 181.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-30 09:15:00 | 181.55 | 181.34 | 181.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 09:15:00 | 181.55 | 181.34 | 181.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 09:15:00 | 181.55 | 181.34 | 181.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-30 10:00:00 | 181.55 | 181.34 | 181.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 10:15:00 | 180.50 | 181.17 | 181.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-30 14:00:00 | 180.20 | 180.86 | 181.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-30 15:15:00 | 180.10 | 180.77 | 181.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-31 09:15:00 | 182.20 | 180.95 | 181.19 | SL hit (close>static) qty=1.00 sl=181.50 alert=retest2 |

### Cycle 7 — BUY (started 2023-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 11:15:00 | 182.35 | 181.48 | 181.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 09:15:00 | 183.45 | 182.14 | 181.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 14:15:00 | 182.35 | 182.59 | 182.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 14:15:00 | 182.35 | 182.59 | 182.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 14:15:00 | 182.35 | 182.59 | 182.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-01 15:00:00 | 182.35 | 182.59 | 182.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 15:15:00 | 182.25 | 182.52 | 182.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 09:15:00 | 182.68 | 182.52 | 182.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 182.88 | 182.60 | 182.26 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 14:15:00 | 180.10 | 181.81 | 182.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-05 09:15:00 | 178.55 | 180.85 | 181.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-06 15:15:00 | 178.38 | 178.25 | 179.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-07 09:15:00 | 178.85 | 178.25 | 179.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 09:15:00 | 181.23 | 178.85 | 179.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 10:00:00 | 181.23 | 178.85 | 179.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2023-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 10:15:00 | 183.13 | 179.70 | 179.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 11:15:00 | 183.50 | 180.46 | 180.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 09:15:00 | 181.88 | 182.47 | 181.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 09:15:00 | 181.88 | 182.47 | 181.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 09:15:00 | 181.88 | 182.47 | 181.37 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 13:15:00 | 179.88 | 180.93 | 181.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 14:15:00 | 179.40 | 180.62 | 180.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 09:15:00 | 181.88 | 180.79 | 180.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 09:15:00 | 181.88 | 180.79 | 180.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 181.88 | 180.79 | 180.93 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 10:15:00 | 182.73 | 181.18 | 181.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-12 12:15:00 | 183.65 | 181.92 | 181.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 14:15:00 | 186.63 | 187.55 | 186.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-14 15:00:00 | 186.63 | 187.55 | 186.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 10:15:00 | 187.28 | 188.31 | 187.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-16 11:00:00 | 187.28 | 188.31 | 187.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 11:15:00 | 187.80 | 188.20 | 187.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 12:15:00 | 187.98 | 188.20 | 187.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 13:00:00 | 188.03 | 188.17 | 187.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 14:15:00 | 188.20 | 188.08 | 187.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 15:00:00 | 188.43 | 188.15 | 187.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 09:15:00 | 189.00 | 188.30 | 187.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-06-20 09:15:00 | 186.68 | 187.66 | 187.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 09:15:00 | 186.68 | 187.66 | 187.73 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-06-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-22 10:15:00 | 188.78 | 187.05 | 187.03 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-06-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 13:15:00 | 186.63 | 187.01 | 187.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 182.15 | 185.99 | 186.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 09:15:00 | 181.03 | 179.78 | 181.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-27 10:00:00 | 181.03 | 179.78 | 181.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 180.95 | 180.01 | 181.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 11:00:00 | 180.95 | 180.01 | 181.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 11:15:00 | 180.48 | 180.11 | 181.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 13:15:00 | 179.85 | 180.17 | 181.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 14:30:00 | 179.93 | 180.25 | 181.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 15:15:00 | 179.93 | 180.25 | 181.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-28 09:45:00 | 179.98 | 180.15 | 180.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 10:15:00 | 181.28 | 180.37 | 180.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-28 11:00:00 | 181.28 | 180.37 | 180.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 11:15:00 | 182.25 | 180.75 | 181.05 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-06-28 11:15:00 | 182.25 | 180.75 | 181.05 | SL hit (close>static) qty=1.00 sl=181.48 alert=retest2 |

### Cycle 15 — BUY (started 2023-06-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 13:15:00 | 182.10 | 181.41 | 181.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-03 09:15:00 | 184.15 | 182.55 | 182.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 11:15:00 | 194.78 | 195.48 | 193.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-07 11:45:00 | 195.05 | 195.48 | 193.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 193.75 | 195.34 | 194.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:30:00 | 193.68 | 195.34 | 194.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 192.58 | 194.79 | 194.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 11:00:00 | 192.58 | 194.79 | 194.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2023-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 15:15:00 | 193.25 | 193.67 | 193.69 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 11:15:00 | 193.85 | 193.72 | 193.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 12:15:00 | 194.85 | 193.94 | 193.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 10:15:00 | 194.60 | 194.61 | 194.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 10:15:00 | 194.60 | 194.61 | 194.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 10:15:00 | 194.60 | 194.61 | 194.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 10:30:00 | 194.73 | 194.61 | 194.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 11:15:00 | 193.33 | 194.35 | 194.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 12:00:00 | 193.33 | 194.35 | 194.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 12:15:00 | 193.68 | 194.22 | 194.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 12:30:00 | 193.43 | 194.22 | 194.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2023-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 14:15:00 | 192.95 | 193.85 | 193.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 11:15:00 | 192.03 | 193.20 | 193.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-17 11:15:00 | 190.50 | 190.19 | 191.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-17 12:00:00 | 190.50 | 190.19 | 191.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 12:15:00 | 191.18 | 190.39 | 191.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 12:30:00 | 191.43 | 190.39 | 191.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 13:15:00 | 191.23 | 190.56 | 191.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 13:30:00 | 191.23 | 190.56 | 191.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 14:15:00 | 190.90 | 190.63 | 191.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 14:30:00 | 191.55 | 190.63 | 191.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 15:15:00 | 190.93 | 190.69 | 191.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-18 09:15:00 | 192.33 | 190.69 | 191.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 192.35 | 191.02 | 191.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-18 09:45:00 | 192.83 | 191.02 | 191.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 10:15:00 | 191.50 | 191.12 | 191.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-18 11:15:00 | 191.30 | 191.12 | 191.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-19 09:45:00 | 190.83 | 190.35 | 190.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-19 13:00:00 | 191.18 | 190.71 | 190.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-19 13:15:00 | 191.75 | 190.92 | 190.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2023-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 13:15:00 | 191.75 | 190.92 | 190.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-20 09:15:00 | 192.25 | 191.55 | 191.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 09:15:00 | 194.88 | 194.90 | 193.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 09:15:00 | 194.88 | 194.90 | 193.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 194.88 | 194.90 | 193.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 09:30:00 | 195.00 | 194.90 | 193.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 194.58 | 194.82 | 194.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-25 11:00:00 | 195.20 | 194.89 | 194.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-25 13:15:00 | 192.75 | 194.02 | 194.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2023-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 13:15:00 | 192.75 | 194.02 | 194.09 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 10:15:00 | 196.40 | 194.40 | 194.22 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-07-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 09:15:00 | 193.43 | 194.30 | 194.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-27 10:15:00 | 192.73 | 193.98 | 194.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 10:15:00 | 188.53 | 187.94 | 189.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-31 11:00:00 | 188.53 | 187.94 | 189.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 09:15:00 | 188.63 | 188.44 | 189.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 09:30:00 | 189.08 | 188.44 | 189.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 189.25 | 188.60 | 189.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-01 12:15:00 | 187.70 | 188.54 | 189.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-01 13:45:00 | 187.93 | 188.34 | 188.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-02 09:15:00 | 187.75 | 188.51 | 188.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-08 13:15:00 | 178.31 | 179.72 | 180.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-08 13:15:00 | 178.53 | 179.72 | 180.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-08 13:15:00 | 178.36 | 179.72 | 180.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-08-09 09:15:00 | 180.00 | 179.55 | 180.48 | SL hit (close>ema200) qty=0.50 sl=179.55 alert=retest2 |

### Cycle 23 — BUY (started 2023-08-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 10:15:00 | 181.90 | 180.61 | 180.56 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 10:15:00 | 179.80 | 180.50 | 180.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 11:15:00 | 179.33 | 180.26 | 180.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 13:15:00 | 178.28 | 178.12 | 178.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-14 13:45:00 | 178.18 | 178.12 | 178.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 14:15:00 | 178.15 | 178.12 | 178.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 14:45:00 | 178.53 | 178.12 | 178.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 179.68 | 178.43 | 178.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 09:30:00 | 179.50 | 178.43 | 178.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 10:15:00 | 179.08 | 178.56 | 178.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 11:15:00 | 178.80 | 178.56 | 178.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 12:45:00 | 178.80 | 178.76 | 178.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 13:30:00 | 178.65 | 178.76 | 178.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 09:45:00 | 178.95 | 178.66 | 178.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 10:15:00 | 178.60 | 178.65 | 178.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 10:30:00 | 178.40 | 178.65 | 178.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 14:15:00 | 178.23 | 178.15 | 178.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 15:00:00 | 178.23 | 178.15 | 178.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 15:15:00 | 178.83 | 178.28 | 178.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 10:30:00 | 177.63 | 178.01 | 178.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 14:15:00 | 177.70 | 177.74 | 178.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-24 15:15:00 | 175.73 | 174.84 | 174.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2023-08-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 15:15:00 | 175.73 | 174.84 | 174.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-25 09:15:00 | 175.98 | 175.07 | 174.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-25 13:15:00 | 175.25 | 175.49 | 175.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-25 14:00:00 | 175.25 | 175.49 | 175.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 14:15:00 | 175.10 | 175.41 | 175.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 14:45:00 | 175.20 | 175.41 | 175.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 15:15:00 | 174.63 | 175.25 | 175.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-28 09:15:00 | 176.55 | 175.25 | 175.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-30 13:15:00 | 176.13 | 176.89 | 176.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2023-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-30 13:15:00 | 176.13 | 176.89 | 176.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-30 14:15:00 | 176.03 | 176.72 | 176.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 13:15:00 | 171.85 | 171.78 | 173.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-01 14:00:00 | 171.85 | 171.78 | 173.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 173.05 | 172.15 | 173.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 10:00:00 | 173.05 | 172.15 | 173.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 10:15:00 | 173.78 | 172.48 | 173.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 11:00:00 | 173.78 | 172.48 | 173.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 11:15:00 | 174.03 | 172.79 | 173.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 11:30:00 | 174.05 | 172.79 | 173.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 13:15:00 | 172.93 | 172.85 | 173.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 14:00:00 | 172.93 | 172.85 | 173.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2023-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 09:15:00 | 175.53 | 173.27 | 173.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 12:15:00 | 176.00 | 175.17 | 174.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 15:15:00 | 176.88 | 176.99 | 176.12 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 09:45:00 | 177.65 | 177.16 | 176.27 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 10:30:00 | 177.83 | 177.40 | 176.46 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 177.95 | 180.65 | 179.67 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-09-12 09:15:00 | 177.95 | 180.65 | 179.67 | SL hit (close<ema400) qty=1.00 sl=179.67 alert=retest1 |

### Cycle 28 — SELL (started 2023-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 12:15:00 | 174.98 | 178.45 | 178.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 14:15:00 | 174.00 | 176.98 | 178.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 10:15:00 | 177.13 | 176.60 | 177.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 10:45:00 | 177.40 | 176.60 | 177.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 12:15:00 | 177.60 | 176.88 | 177.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 13:00:00 | 177.60 | 176.88 | 177.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 13:15:00 | 177.28 | 176.96 | 177.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 13:45:00 | 177.48 | 176.96 | 177.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 14:15:00 | 177.70 | 177.11 | 177.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 15:00:00 | 177.70 | 177.11 | 177.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 15:15:00 | 177.60 | 177.21 | 177.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:15:00 | 178.35 | 177.21 | 177.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 179.28 | 177.62 | 177.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:45:00 | 180.00 | 177.62 | 177.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2023-09-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 10:15:00 | 180.93 | 178.28 | 177.99 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 11:15:00 | 176.70 | 178.28 | 178.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-15 12:15:00 | 176.25 | 177.87 | 178.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-18 09:15:00 | 178.50 | 177.37 | 177.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 09:15:00 | 178.50 | 177.37 | 177.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 178.50 | 177.37 | 177.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-18 10:00:00 | 178.50 | 177.37 | 177.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 10:15:00 | 178.08 | 177.51 | 177.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-18 10:45:00 | 178.95 | 177.51 | 177.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2023-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 12:15:00 | 179.58 | 178.23 | 178.05 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 11:15:00 | 178.00 | 178.19 | 178.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 13:15:00 | 176.45 | 177.84 | 178.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 10:15:00 | 177.90 | 177.41 | 177.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 10:15:00 | 177.90 | 177.41 | 177.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 10:15:00 | 177.90 | 177.41 | 177.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 10:30:00 | 178.08 | 177.41 | 177.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 11:15:00 | 177.58 | 177.44 | 177.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 12:00:00 | 177.58 | 177.44 | 177.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 12:15:00 | 177.10 | 177.37 | 177.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 13:15:00 | 176.48 | 177.32 | 177.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 09:30:00 | 176.48 | 176.05 | 176.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 11:15:00 | 176.38 | 176.18 | 176.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 12:30:00 | 176.43 | 176.33 | 176.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 13:15:00 | 176.88 | 176.44 | 176.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 14:00:00 | 176.88 | 176.44 | 176.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 14:15:00 | 175.80 | 176.31 | 176.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 09:30:00 | 175.23 | 175.89 | 176.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-06 10:15:00 | 172.95 | 172.19 | 172.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2023-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 10:15:00 | 172.95 | 172.19 | 172.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 12:15:00 | 173.40 | 172.53 | 172.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 169.70 | 172.38 | 172.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 169.70 | 172.38 | 172.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 169.70 | 172.38 | 172.34 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 10:15:00 | 169.73 | 171.85 | 172.11 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 11:15:00 | 172.00 | 171.18 | 171.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 09:15:00 | 174.08 | 171.84 | 171.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 13:15:00 | 173.90 | 174.22 | 173.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-13 14:00:00 | 173.90 | 174.22 | 173.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 172.60 | 173.78 | 173.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 10:15:00 | 173.00 | 173.78 | 173.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-20 11:15:00 | 173.58 | 175.58 | 175.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2023-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 11:15:00 | 173.58 | 175.58 | 175.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 12:15:00 | 172.63 | 174.99 | 175.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 172.58 | 172.17 | 173.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 09:15:00 | 172.58 | 172.17 | 173.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 172.58 | 172.17 | 173.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 09:45:00 | 172.88 | 172.17 | 173.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 10:15:00 | 171.98 | 172.13 | 173.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 12:30:00 | 170.93 | 171.62 | 172.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-30 10:45:00 | 171.38 | 168.83 | 169.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-30 11:15:00 | 173.33 | 169.73 | 169.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2023-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 11:15:00 | 173.33 | 169.73 | 169.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 13:15:00 | 174.03 | 171.14 | 170.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 14:15:00 | 180.80 | 180.82 | 179.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-03 14:45:00 | 180.55 | 180.82 | 179.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 12:15:00 | 197.15 | 198.59 | 197.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 12:45:00 | 196.85 | 198.59 | 197.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 13:15:00 | 196.88 | 198.25 | 197.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 14:00:00 | 196.88 | 198.25 | 197.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 14:15:00 | 194.90 | 197.58 | 197.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 15:00:00 | 194.90 | 197.58 | 197.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2023-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 10:15:00 | 195.03 | 196.44 | 196.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-21 09:15:00 | 194.18 | 195.72 | 196.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-22 09:15:00 | 199.50 | 195.59 | 195.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 09:15:00 | 199.50 | 195.59 | 195.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 199.50 | 195.59 | 195.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-22 09:45:00 | 200.20 | 195.59 | 195.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2023-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 10:15:00 | 199.38 | 196.35 | 196.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 14:15:00 | 201.03 | 198.60 | 197.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 09:15:00 | 235.10 | 236.50 | 234.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-08 10:00:00 | 235.10 | 236.50 | 234.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 10:15:00 | 235.13 | 236.23 | 234.73 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 12:15:00 | 230.30 | 234.55 | 234.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 09:15:00 | 224.68 | 231.88 | 233.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-15 09:15:00 | 225.30 | 223.48 | 224.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 09:15:00 | 225.30 | 223.48 | 224.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 09:15:00 | 225.30 | 223.48 | 224.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-15 10:00:00 | 225.30 | 223.48 | 224.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 10:15:00 | 224.38 | 223.66 | 224.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-18 09:15:00 | 223.45 | 224.21 | 224.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-18 09:45:00 | 223.50 | 223.96 | 224.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-18 15:15:00 | 223.33 | 223.59 | 224.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-19 10:00:00 | 223.23 | 223.48 | 223.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 223.70 | 223.52 | 223.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:45:00 | 223.85 | 223.52 | 223.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 11:15:00 | 224.65 | 223.75 | 223.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 12:00:00 | 224.65 | 223.75 | 223.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 12:15:00 | 224.30 | 223.86 | 224.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 12:30:00 | 224.48 | 223.86 | 224.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-12-19 14:15:00 | 224.83 | 224.20 | 224.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2023-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 14:15:00 | 224.83 | 224.20 | 224.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-20 09:15:00 | 226.25 | 224.71 | 224.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 12:15:00 | 224.68 | 225.21 | 224.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 12:15:00 | 224.68 | 225.21 | 224.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 224.68 | 225.21 | 224.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:00:00 | 224.68 | 225.21 | 224.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 221.38 | 224.44 | 224.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 219.58 | 223.47 | 224.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 11:15:00 | 222.23 | 222.05 | 223.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 12:00:00 | 222.23 | 222.05 | 223.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 12:15:00 | 223.33 | 222.30 | 223.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 12:30:00 | 223.45 | 222.30 | 223.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 13:15:00 | 223.03 | 222.45 | 223.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 13:30:00 | 223.50 | 222.45 | 223.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 14:15:00 | 224.65 | 222.89 | 223.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 15:00:00 | 224.65 | 222.89 | 223.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 15:15:00 | 224.63 | 223.24 | 223.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:15:00 | 225.18 | 223.24 | 223.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2023-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 09:15:00 | 224.80 | 223.55 | 223.46 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2023-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-22 12:15:00 | 222.88 | 223.41 | 223.42 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 09:15:00 | 225.33 | 223.78 | 223.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 10:15:00 | 227.75 | 224.57 | 223.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 09:15:00 | 226.40 | 229.98 | 228.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 09:15:00 | 226.40 | 229.98 | 228.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 226.40 | 229.98 | 228.43 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2023-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 13:15:00 | 224.88 | 227.60 | 227.68 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 14:15:00 | 228.23 | 227.04 | 227.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-03 10:15:00 | 230.53 | 228.26 | 227.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-04 09:15:00 | 226.58 | 229.38 | 228.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 09:15:00 | 226.58 | 229.38 | 228.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 09:15:00 | 226.58 | 229.38 | 228.69 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-01-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 12:15:00 | 226.58 | 228.15 | 228.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-04 13:15:00 | 226.13 | 227.75 | 228.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-05 09:15:00 | 228.75 | 227.65 | 227.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 09:15:00 | 228.75 | 227.65 | 227.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 09:15:00 | 228.75 | 227.65 | 227.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-05 09:30:00 | 227.68 | 227.65 | 227.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 10:15:00 | 227.45 | 227.61 | 227.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-05 14:15:00 | 226.10 | 227.57 | 227.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-05 15:15:00 | 226.25 | 227.38 | 227.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-08 09:15:00 | 229.00 | 227.52 | 227.69 | SL hit (close>static) qty=1.00 sl=228.85 alert=retest2 |

### Cycle 49 — BUY (started 2024-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-08 13:15:00 | 228.55 | 227.84 | 227.79 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 15:15:00 | 227.40 | 227.70 | 227.73 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 09:15:00 | 228.90 | 227.94 | 227.84 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 09:15:00 | 223.40 | 228.18 | 228.24 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-01-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 12:15:00 | 228.90 | 227.33 | 227.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-11 14:15:00 | 229.53 | 228.01 | 227.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-12 09:15:00 | 227.88 | 228.19 | 227.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 09:15:00 | 227.88 | 228.19 | 227.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 227.88 | 228.19 | 227.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-12 09:30:00 | 227.85 | 228.19 | 227.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 10:15:00 | 229.25 | 228.40 | 227.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 09:15:00 | 230.05 | 228.64 | 228.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 11:30:00 | 231.10 | 228.84 | 228.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 13:00:00 | 229.88 | 229.05 | 228.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 13:45:00 | 230.30 | 229.36 | 228.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 235.95 | 236.38 | 234.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 09:45:00 | 232.13 | 236.38 | 234.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 236.05 | 239.97 | 239.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:00:00 | 236.05 | 239.97 | 239.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 237.18 | 239.41 | 238.98 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-23 11:15:00 | 234.58 | 238.44 | 238.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 11:15:00 | 234.58 | 238.44 | 238.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 12:15:00 | 233.20 | 237.39 | 238.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 13:15:00 | 235.45 | 234.37 | 235.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 13:15:00 | 235.45 | 234.37 | 235.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 235.45 | 234.37 | 235.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:00:00 | 235.45 | 234.37 | 235.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 238.48 | 235.19 | 235.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:45:00 | 238.63 | 235.19 | 235.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 238.93 | 235.94 | 236.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:15:00 | 241.45 | 235.94 | 236.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 09:15:00 | 241.73 | 237.10 | 236.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 11:15:00 | 243.98 | 240.09 | 238.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 252.13 | 252.29 | 247.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-30 15:00:00 | 252.13 | 252.29 | 247.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 252.60 | 252.27 | 250.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:00:00 | 252.60 | 252.27 | 250.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 304.25 | 308.80 | 303.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 10:00:00 | 304.25 | 308.80 | 303.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 299.83 | 307.01 | 303.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 11:00:00 | 299.83 | 307.01 | 303.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 11:15:00 | 305.52 | 306.71 | 303.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 12:15:00 | 307.18 | 306.71 | 303.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 13:30:00 | 306.35 | 306.89 | 304.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 14:00:00 | 307.25 | 306.89 | 304.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 15:00:00 | 307.30 | 306.97 | 304.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 298.23 | 305.27 | 304.13 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-12 09:15:00 | 298.23 | 305.27 | 304.13 | SL hit (close<static) qty=1.00 sl=298.83 alert=retest2 |

### Cycle 56 — SELL (started 2024-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 10:15:00 | 294.77 | 303.17 | 303.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-13 11:15:00 | 291.27 | 296.30 | 298.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 09:15:00 | 295.65 | 294.21 | 296.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 09:15:00 | 295.65 | 294.21 | 296.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 295.65 | 294.21 | 296.69 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 12:15:00 | 308.10 | 299.18 | 298.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 14:15:00 | 311.43 | 303.20 | 300.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 14:15:00 | 327.73 | 328.39 | 321.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-16 14:30:00 | 328.78 | 328.39 | 321.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 321.05 | 326.12 | 324.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 09:45:00 | 321.48 | 326.12 | 324.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 319.80 | 324.86 | 323.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 10:45:00 | 320.40 | 324.86 | 323.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 13:15:00 | 326.88 | 324.51 | 323.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 15:15:00 | 331.80 | 325.06 | 324.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 13:15:00 | 319.93 | 323.74 | 324.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 13:15:00 | 319.93 | 323.74 | 324.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 14:15:00 | 316.90 | 322.37 | 323.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-26 13:15:00 | 311.20 | 307.95 | 310.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 13:15:00 | 311.20 | 307.95 | 310.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 13:15:00 | 311.20 | 307.95 | 310.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 13:45:00 | 310.35 | 307.95 | 310.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 14:15:00 | 314.08 | 309.17 | 310.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 14:30:00 | 316.10 | 309.17 | 310.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 15:15:00 | 313.00 | 309.94 | 311.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 09:15:00 | 315.05 | 309.94 | 311.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 10:15:00 | 314.00 | 311.23 | 311.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 10:45:00 | 313.93 | 311.23 | 311.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 12:15:00 | 310.50 | 310.97 | 311.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 12:30:00 | 310.50 | 310.97 | 311.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 13:15:00 | 308.35 | 310.45 | 311.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 10:45:00 | 306.90 | 310.34 | 310.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 14:15:00 | 307.80 | 308.93 | 310.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-01 11:15:00 | 311.18 | 307.21 | 307.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2024-03-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 11:15:00 | 311.18 | 307.21 | 307.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 12:15:00 | 312.50 | 310.86 | 309.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 09:15:00 | 317.60 | 318.09 | 314.79 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 10:45:00 | 321.58 | 318.73 | 315.39 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 313.27 | 319.66 | 317.63 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-03-06 09:15:00 | 313.27 | 319.66 | 317.63 | SL hit (close<ema400) qty=1.00 sl=317.63 alert=retest1 |

### Cycle 60 — SELL (started 2024-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-07 10:15:00 | 313.08 | 316.53 | 316.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-07 15:15:00 | 311.50 | 314.01 | 315.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-11 09:15:00 | 314.75 | 314.16 | 315.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 09:15:00 | 314.75 | 314.16 | 315.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 314.75 | 314.16 | 315.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-11 09:30:00 | 316.40 | 314.16 | 315.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 10:15:00 | 315.77 | 314.48 | 315.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-11 10:45:00 | 318.33 | 314.48 | 315.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 11:15:00 | 313.00 | 314.19 | 315.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-11 11:30:00 | 315.98 | 314.19 | 315.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 13:15:00 | 313.88 | 314.09 | 314.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-11 14:00:00 | 313.88 | 314.09 | 314.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 14:15:00 | 312.85 | 313.85 | 314.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-11 14:30:00 | 314.83 | 313.85 | 314.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 314.85 | 313.83 | 314.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-12 11:15:00 | 310.93 | 313.49 | 314.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-12 12:30:00 | 311.10 | 312.35 | 313.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 09:15:00 | 311.05 | 312.53 | 313.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 09:15:00 | 295.38 | 300.52 | 303.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 09:15:00 | 295.55 | 300.52 | 303.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 09:15:00 | 295.50 | 300.52 | 303.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-15 12:15:00 | 279.84 | 292.06 | 298.83 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 61 — BUY (started 2024-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 11:15:00 | 291.75 | 287.29 | 286.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 14:15:00 | 294.02 | 290.13 | 288.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 295.95 | 296.33 | 293.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-22 15:00:00 | 295.95 | 296.33 | 293.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 297.65 | 296.58 | 293.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 10:30:00 | 298.65 | 297.47 | 294.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 15:15:00 | 298.43 | 301.08 | 299.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-04 11:15:00 | 298.30 | 304.03 | 304.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2024-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 11:15:00 | 298.30 | 304.03 | 304.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-05 09:15:00 | 292.00 | 299.67 | 302.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-08 09:15:00 | 299.43 | 296.66 | 298.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-08 09:15:00 | 299.43 | 296.66 | 298.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 299.43 | 296.66 | 298.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-08 12:00:00 | 296.25 | 296.89 | 298.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-08 12:45:00 | 296.95 | 297.00 | 298.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-08 13:15:00 | 296.08 | 297.00 | 298.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-10 10:15:00 | 302.85 | 296.77 | 296.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2024-04-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 10:15:00 | 302.85 | 296.77 | 296.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-12 10:15:00 | 304.68 | 301.64 | 299.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 12:15:00 | 302.13 | 302.21 | 300.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-12 13:00:00 | 302.13 | 302.21 | 300.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 14:15:00 | 300.68 | 301.67 | 300.34 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2024-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 10:15:00 | 297.58 | 299.64 | 299.66 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-04-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 09:15:00 | 302.95 | 298.25 | 298.22 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-04-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 15:15:00 | 296.50 | 298.58 | 298.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 09:15:00 | 285.98 | 296.06 | 297.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 13:15:00 | 293.98 | 293.46 | 295.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-19 14:00:00 | 293.98 | 293.46 | 295.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 303.65 | 295.28 | 295.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 10:00:00 | 303.65 | 295.28 | 295.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2024-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 10:15:00 | 303.55 | 296.93 | 296.59 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 09:15:00 | 296.98 | 298.22 | 298.35 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 13:15:00 | 301.02 | 297.93 | 297.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 14:15:00 | 301.83 | 298.71 | 298.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 12:15:00 | 308.23 | 309.13 | 306.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-30 13:00:00 | 308.23 | 309.13 | 306.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 303.70 | 308.05 | 306.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 14:45:00 | 303.88 | 308.05 | 306.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 305.08 | 307.46 | 306.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 09:15:00 | 309.70 | 307.46 | 306.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 10:15:00 | 306.45 | 312.81 | 312.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-06 10:15:00 | 307.95 | 311.84 | 311.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 10:15:00 | 307.95 | 311.84 | 311.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 11:15:00 | 302.90 | 306.35 | 308.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 308.33 | 304.90 | 306.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 09:15:00 | 308.33 | 304.90 | 306.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 308.33 | 304.90 | 306.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 09:45:00 | 308.25 | 304.90 | 306.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 309.08 | 305.74 | 306.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 11:00:00 | 309.08 | 305.74 | 306.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 310.45 | 306.68 | 307.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:00:00 | 310.45 | 306.68 | 307.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2024-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 13:15:00 | 309.88 | 308.01 | 307.82 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 10:15:00 | 304.23 | 307.43 | 307.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 12:15:00 | 303.27 | 306.36 | 307.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 09:15:00 | 306.38 | 302.84 | 304.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 09:15:00 | 306.38 | 302.84 | 304.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 306.38 | 302.84 | 304.90 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-05-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 14:15:00 | 308.90 | 306.45 | 306.13 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 09:15:00 | 300.27 | 305.68 | 305.87 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 12:15:00 | 304.75 | 304.16 | 304.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 13:15:00 | 307.08 | 304.74 | 304.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 10:15:00 | 308.20 | 308.45 | 306.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 11:00:00 | 308.20 | 308.45 | 306.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 309.58 | 308.68 | 306.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:45:00 | 307.93 | 308.68 | 306.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 305.88 | 308.01 | 306.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:30:00 | 306.63 | 308.01 | 306.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 309.40 | 308.29 | 307.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 311.65 | 308.59 | 307.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 10:15:00 | 310.15 | 308.86 | 307.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 14:15:00 | 323.80 | 325.17 | 325.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 14:15:00 | 323.80 | 325.17 | 325.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 09:15:00 | 320.05 | 323.77 | 324.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 315.10 | 313.46 | 315.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 13:45:00 | 315.48 | 313.46 | 315.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 77 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 336.33 | 318.19 | 317.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 10:15:00 | 337.18 | 321.99 | 319.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 326.20 | 329.35 | 325.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 326.20 | 329.35 | 325.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 326.20 | 329.35 | 325.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 318.08 | 329.35 | 325.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 307.43 | 324.97 | 323.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 307.43 | 324.97 | 323.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 286.23 | 317.22 | 320.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 274.27 | 297.40 | 308.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 15:15:00 | 290.00 | 289.77 | 298.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-06 09:15:00 | 296.52 | 289.77 | 298.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 303.40 | 293.67 | 299.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:45:00 | 301.75 | 293.67 | 299.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 11:15:00 | 296.90 | 294.31 | 298.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 12:30:00 | 295.10 | 294.24 | 298.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 13:45:00 | 294.43 | 294.12 | 297.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-10 09:15:00 | 302.73 | 297.22 | 296.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2024-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 09:15:00 | 302.73 | 297.22 | 296.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 13:15:00 | 305.73 | 301.35 | 299.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 310.00 | 313.82 | 312.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 310.00 | 313.82 | 312.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 310.00 | 313.82 | 312.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 310.00 | 313.82 | 312.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 309.88 | 313.03 | 312.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:45:00 | 309.80 | 313.03 | 312.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 13:15:00 | 309.48 | 311.58 | 311.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 14:15:00 | 307.73 | 310.81 | 311.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 310.88 | 310.39 | 310.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 310.88 | 310.39 | 310.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 310.88 | 310.39 | 310.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:00:00 | 310.88 | 310.39 | 310.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 313.30 | 310.98 | 311.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:45:00 | 313.45 | 310.98 | 311.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 11:15:00 | 312.88 | 311.36 | 311.32 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 11:15:00 | 309.45 | 311.36 | 311.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 305.95 | 310.28 | 310.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 10:15:00 | 299.70 | 299.59 | 302.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-26 11:00:00 | 299.70 | 299.59 | 302.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 301.20 | 299.40 | 300.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:45:00 | 300.75 | 299.40 | 300.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 300.20 | 299.56 | 300.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:15:00 | 301.60 | 299.56 | 300.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 305.45 | 300.74 | 301.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 305.45 | 300.74 | 301.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2024-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 15:15:00 | 305.05 | 301.60 | 301.49 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 14:15:00 | 303.20 | 304.78 | 304.88 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 09:15:00 | 306.60 | 305.02 | 304.97 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 302.50 | 304.75 | 304.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 11:15:00 | 301.65 | 304.13 | 304.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 14:15:00 | 301.25 | 299.56 | 301.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 14:15:00 | 301.25 | 299.56 | 301.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 301.25 | 299.56 | 301.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 15:00:00 | 301.25 | 299.56 | 301.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 300.20 | 299.69 | 301.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:15:00 | 299.20 | 299.69 | 301.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 296.50 | 299.05 | 300.68 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2024-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 10:15:00 | 305.20 | 301.01 | 300.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 12:15:00 | 307.70 | 302.89 | 301.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 12:15:00 | 305.20 | 305.99 | 304.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 12:15:00 | 305.20 | 305.99 | 304.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 305.20 | 305.99 | 304.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:30:00 | 305.05 | 305.99 | 304.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 304.65 | 305.64 | 304.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 15:00:00 | 304.65 | 305.64 | 304.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 305.25 | 305.57 | 304.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 09:15:00 | 305.90 | 305.57 | 304.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 11:00:00 | 305.30 | 312.75 | 312.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 11:15:00 | 308.00 | 311.80 | 312.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 11:15:00 | 308.00 | 311.80 | 312.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 09:15:00 | 303.65 | 308.57 | 309.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 310.90 | 307.28 | 308.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 310.90 | 307.28 | 308.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 310.90 | 307.28 | 308.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 310.90 | 307.28 | 308.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 312.40 | 308.31 | 308.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:00:00 | 312.40 | 308.31 | 308.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 312.70 | 309.18 | 308.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 13:15:00 | 313.45 | 310.26 | 309.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 11:15:00 | 348.85 | 348.93 | 342.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 11:30:00 | 349.00 | 348.93 | 342.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 349.15 | 349.38 | 345.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:30:00 | 346.75 | 349.38 | 345.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 347.00 | 348.75 | 345.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:45:00 | 348.00 | 348.75 | 345.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 350.10 | 348.52 | 346.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:30:00 | 347.60 | 348.52 | 346.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 344.50 | 347.73 | 346.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:45:00 | 342.60 | 347.73 | 346.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 346.85 | 347.55 | 346.41 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 339.25 | 345.82 | 346.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 13:15:00 | 338.60 | 342.36 | 343.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 343.20 | 340.37 | 342.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 343.20 | 340.37 | 342.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 343.20 | 340.37 | 342.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 09:45:00 | 343.45 | 340.37 | 342.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 344.95 | 341.28 | 342.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:30:00 | 346.35 | 341.28 | 342.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 344.00 | 343.00 | 342.95 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 09:15:00 | 338.25 | 342.05 | 342.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 10:15:00 | 334.85 | 340.61 | 341.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 13:15:00 | 335.45 | 335.27 | 337.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-09 14:00:00 | 335.45 | 335.27 | 337.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 333.75 | 333.02 | 334.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:30:00 | 336.00 | 333.02 | 334.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 329.80 | 326.14 | 327.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:45:00 | 330.90 | 326.14 | 327.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 328.60 | 326.63 | 327.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:15:00 | 330.30 | 326.63 | 327.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 332.80 | 329.26 | 328.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 341.70 | 332.83 | 330.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 10:15:00 | 350.05 | 350.84 | 347.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 11:00:00 | 350.05 | 350.84 | 347.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 353.00 | 353.28 | 351.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 350.75 | 353.28 | 351.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 351.90 | 353.00 | 351.44 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 09:15:00 | 344.45 | 350.19 | 350.70 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 10:15:00 | 351.80 | 350.47 | 350.43 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 14:15:00 | 347.95 | 350.06 | 350.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 15:15:00 | 347.00 | 349.45 | 349.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 11:15:00 | 350.30 | 349.26 | 349.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 11:15:00 | 350.30 | 349.26 | 349.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 350.30 | 349.26 | 349.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 11:45:00 | 350.15 | 349.26 | 349.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 12:15:00 | 353.40 | 350.09 | 350.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-29 14:15:00 | 355.70 | 351.54 | 350.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 15:15:00 | 357.00 | 357.00 | 354.67 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 09:30:00 | 359.85 | 357.74 | 355.22 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 354.85 | 358.28 | 357.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-03 09:15:00 | 354.85 | 358.28 | 357.08 | SL hit (close<ema400) qty=1.00 sl=357.08 alert=retest1 |

### Cycle 98 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 351.55 | 357.55 | 357.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 350.80 | 353.41 | 355.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 345.00 | 342.65 | 345.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 345.00 | 342.65 | 345.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 345.00 | 342.65 | 345.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:00:00 | 345.00 | 342.65 | 345.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 344.90 | 343.10 | 345.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 11:45:00 | 342.80 | 342.91 | 344.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 15:15:00 | 342.40 | 343.12 | 344.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 09:15:00 | 347.15 | 343.81 | 344.62 | SL hit (close>static) qty=1.00 sl=346.20 alert=retest2 |

### Cycle 99 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 338.20 | 332.59 | 332.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 340.65 | 337.09 | 335.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 14:15:00 | 339.15 | 339.49 | 337.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 15:00:00 | 339.15 | 339.49 | 337.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 335.80 | 338.72 | 337.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 335.80 | 338.72 | 337.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 339.80 | 338.94 | 337.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 13:00:00 | 341.65 | 338.63 | 338.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-30 09:15:00 | 375.81 | 360.63 | 352.40 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 351.65 | 360.45 | 360.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 350.60 | 358.48 | 359.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 14:15:00 | 337.15 | 335.78 | 339.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 15:00:00 | 337.15 | 335.78 | 339.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 341.75 | 337.46 | 339.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:45:00 | 338.75 | 338.62 | 339.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 12:30:00 | 339.55 | 338.90 | 339.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:00:00 | 340.05 | 338.90 | 339.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 14:15:00 | 339.05 | 339.17 | 339.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 338.60 | 339.06 | 339.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:30:00 | 336.55 | 339.02 | 339.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 09:45:00 | 337.65 | 337.66 | 338.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 10:45:00 | 337.70 | 337.53 | 338.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 09:15:00 | 342.45 | 338.39 | 338.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 09:15:00 | 342.45 | 338.39 | 338.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 09:15:00 | 349.20 | 341.44 | 339.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 346.20 | 348.66 | 346.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 346.20 | 348.66 | 346.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 346.20 | 348.66 | 346.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:45:00 | 345.65 | 348.66 | 346.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 343.35 | 347.60 | 346.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 343.35 | 347.60 | 346.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 344.30 | 346.94 | 346.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:45:00 | 344.30 | 346.94 | 346.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2024-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 14:15:00 | 342.90 | 345.10 | 345.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 336.75 | 342.93 | 344.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 14:15:00 | 342.45 | 341.24 | 342.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 14:15:00 | 342.45 | 341.24 | 342.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 342.45 | 341.24 | 342.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 15:00:00 | 342.45 | 341.24 | 342.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 342.60 | 341.51 | 342.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 336.85 | 341.51 | 342.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 332.95 | 339.80 | 341.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:45:00 | 331.30 | 336.82 | 340.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:30:00 | 331.55 | 335.74 | 339.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 331.20 | 334.84 | 338.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 15:00:00 | 331.55 | 334.18 | 337.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 314.74 | 320.47 | 323.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 314.97 | 320.47 | 323.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 314.64 | 320.47 | 323.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 314.97 | 320.47 | 323.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 310.85 | 310.53 | 315.68 | SL hit (close>ema200) qty=0.50 sl=310.53 alert=retest2 |

### Cycle 103 — BUY (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 10:15:00 | 314.70 | 312.28 | 312.04 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 12:15:00 | 308.55 | 311.52 | 311.74 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-11-01 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 18:15:00 | 312.85 | 311.78 | 311.72 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 302.20 | 309.87 | 310.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 301.10 | 308.11 | 309.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 12:15:00 | 304.05 | 303.77 | 305.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 13:00:00 | 304.05 | 303.77 | 305.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 307.05 | 304.43 | 306.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 13:45:00 | 307.30 | 304.43 | 306.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 307.55 | 305.05 | 306.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 307.55 | 305.05 | 306.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 312.60 | 307.03 | 306.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 10:15:00 | 314.85 | 308.60 | 307.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 313.15 | 314.06 | 311.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 10:15:00 | 315.05 | 314.26 | 311.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 315.05 | 314.26 | 311.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 311.65 | 314.26 | 311.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 309.00 | 313.62 | 312.61 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 307.30 | 311.63 | 311.84 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 12:15:00 | 313.30 | 311.54 | 311.44 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 310.75 | 311.59 | 311.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 304.45 | 309.62 | 310.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 296.55 | 293.99 | 297.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 09:30:00 | 295.45 | 293.99 | 297.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 286.15 | 284.64 | 286.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:45:00 | 286.35 | 284.64 | 286.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 300.60 | 288.05 | 288.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 10:00:00 | 300.60 | 288.05 | 288.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 298.55 | 290.15 | 289.04 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 13:15:00 | 291.60 | 293.11 | 293.27 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2024-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 13:15:00 | 293.80 | 292.59 | 292.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 15:15:00 | 295.75 | 293.33 | 292.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 11:15:00 | 292.95 | 293.52 | 293.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 11:15:00 | 292.95 | 293.52 | 293.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 292.95 | 293.52 | 293.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 11:45:00 | 293.20 | 293.52 | 293.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 12:15:00 | 293.70 | 293.56 | 293.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 13:30:00 | 294.70 | 293.81 | 293.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 15:15:00 | 294.50 | 293.88 | 293.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 09:15:00 | 295.20 | 294.66 | 294.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 09:15:00 | 291.85 | 294.10 | 294.07 | SL hit (close<static) qty=1.00 sl=292.05 alert=retest2 |

### Cycle 114 — SELL (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 10:15:00 | 292.00 | 293.68 | 293.88 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 11:15:00 | 295.60 | 294.06 | 294.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 12:15:00 | 296.15 | 294.48 | 294.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 09:15:00 | 299.80 | 300.03 | 298.19 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 11:45:00 | 302.50 | 300.63 | 298.78 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 13:30:00 | 302.45 | 301.18 | 299.36 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 12:45:00 | 302.95 | 302.80 | 301.16 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 13:15:00 | 301.65 | 302.57 | 301.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 14:00:00 | 301.65 | 302.57 | 301.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 303.55 | 302.77 | 301.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 09:15:00 | 306.35 | 302.82 | 301.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 10:15:00 | 304.05 | 305.53 | 304.23 | SL hit (close<ema400) qty=1.00 sl=304.23 alert=retest1 |

### Cycle 116 — SELL (started 2024-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 15:15:00 | 301.75 | 303.79 | 303.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 301.00 | 303.23 | 303.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 302.15 | 301.58 | 302.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 13:15:00 | 302.15 | 301.58 | 302.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 302.15 | 301.58 | 302.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:45:00 | 302.20 | 301.58 | 302.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 301.80 | 301.62 | 302.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:30:00 | 301.45 | 301.62 | 302.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 295.15 | 297.81 | 299.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:15:00 | 294.15 | 297.81 | 299.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 11:00:00 | 294.00 | 297.05 | 299.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 13:00:00 | 294.40 | 296.29 | 298.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 12:30:00 | 294.45 | 292.12 | 293.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 294.05 | 293.54 | 293.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:45:00 | 294.85 | 293.54 | 293.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 297.20 | 294.27 | 293.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 10:15:00 | 297.20 | 294.27 | 293.98 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 290.75 | 293.56 | 293.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 288.70 | 292.59 | 293.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 10:15:00 | 293.40 | 292.03 | 292.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 10:15:00 | 293.40 | 292.03 | 292.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 293.40 | 292.03 | 292.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:00:00 | 293.40 | 292.03 | 292.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 291.45 | 291.92 | 292.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 12:15:00 | 290.35 | 291.92 | 292.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 09:15:00 | 297.60 | 292.91 | 292.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2024-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 09:15:00 | 297.60 | 292.91 | 292.40 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 10:15:00 | 292.30 | 293.57 | 293.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 290.60 | 292.97 | 293.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 14:15:00 | 293.05 | 292.98 | 293.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 14:15:00 | 293.05 | 292.98 | 293.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 293.05 | 292.98 | 293.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 293.05 | 292.98 | 293.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 292.50 | 292.89 | 293.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 292.10 | 292.89 | 293.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 10:00:00 | 292.25 | 292.76 | 293.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 12:45:00 | 292.20 | 292.66 | 293.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 14:15:00 | 292.35 | 292.86 | 293.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 292.70 | 292.83 | 293.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 15:15:00 | 292.10 | 292.83 | 293.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 09:45:00 | 291.35 | 292.34 | 292.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 12:15:00 | 294.30 | 292.97 | 292.98 | SL hit (close>static) qty=1.00 sl=294.15 alert=retest2 |

### Cycle 121 — BUY (started 2025-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 13:15:00 | 295.45 | 293.47 | 293.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 296.00 | 294.38 | 293.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 292.00 | 295.34 | 295.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 292.00 | 295.34 | 295.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 292.00 | 295.34 | 295.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 290.20 | 295.34 | 295.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 286.85 | 293.64 | 294.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 286.40 | 290.45 | 292.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 09:15:00 | 285.70 | 284.51 | 287.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 10:00:00 | 285.70 | 284.51 | 287.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 287.15 | 285.45 | 286.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 284.20 | 285.63 | 286.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:45:00 | 284.40 | 285.48 | 286.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 269.99 | 278.28 | 281.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 270.18 | 278.28 | 281.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 270.70 | 270.02 | 274.32 | SL hit (close>ema200) qty=0.50 sl=270.02 alert=retest2 |

### Cycle 123 — BUY (started 2025-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 11:15:00 | 274.15 | 270.04 | 269.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 12:15:00 | 274.50 | 270.93 | 270.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 14:15:00 | 280.25 | 280.45 | 277.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 15:00:00 | 280.25 | 280.45 | 277.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 278.00 | 279.87 | 277.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 278.15 | 279.87 | 277.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 277.15 | 279.33 | 277.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:00:00 | 277.15 | 279.33 | 277.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 274.85 | 278.43 | 277.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:00:00 | 274.85 | 278.43 | 277.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 12:15:00 | 276.55 | 278.06 | 277.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 14:45:00 | 276.90 | 277.72 | 277.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 10:15:00 | 274.20 | 276.59 | 276.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 10:15:00 | 274.20 | 276.59 | 276.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 11:15:00 | 272.10 | 275.69 | 276.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 262.10 | 260.78 | 263.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:45:00 | 261.90 | 260.78 | 263.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 259.95 | 257.41 | 259.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 14:30:00 | 257.65 | 258.21 | 259.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 15:00:00 | 257.05 | 258.21 | 259.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 10:00:00 | 257.65 | 257.94 | 258.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 14:15:00 | 261.05 | 259.53 | 259.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 14:15:00 | 261.05 | 259.53 | 259.40 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 11:15:00 | 257.90 | 259.30 | 259.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 12:15:00 | 250.40 | 257.52 | 258.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 15:15:00 | 250.00 | 249.78 | 252.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 09:15:00 | 251.95 | 249.78 | 252.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 254.05 | 250.64 | 252.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:15:00 | 254.55 | 250.64 | 252.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 253.90 | 251.29 | 253.02 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 14:15:00 | 255.90 | 253.85 | 253.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 263.75 | 256.22 | 254.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 263.10 | 263.17 | 260.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 13:45:00 | 263.35 | 263.17 | 260.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 261.70 | 262.77 | 261.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:15:00 | 260.75 | 262.77 | 261.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 263.70 | 262.96 | 261.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 15:00:00 | 264.45 | 263.03 | 261.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 258.60 | 262.34 | 261.72 | SL hit (close<static) qty=1.00 sl=260.05 alert=retest2 |

### Cycle 128 — SELL (started 2025-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 12:15:00 | 259.60 | 261.03 | 261.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 256.85 | 259.65 | 260.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 255.80 | 255.06 | 257.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:00:00 | 255.80 | 255.06 | 257.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 257.35 | 255.67 | 256.65 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2025-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 12:15:00 | 258.80 | 257.21 | 257.18 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 13:15:00 | 256.00 | 256.96 | 257.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-13 14:15:00 | 255.75 | 256.72 | 256.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 252.80 | 250.65 | 252.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 14:15:00 | 252.80 | 250.65 | 252.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 252.80 | 250.65 | 252.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 252.80 | 250.65 | 252.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 252.90 | 251.10 | 252.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 247.95 | 251.10 | 252.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 253.95 | 251.97 | 252.01 | SL hit (close>static) qty=1.00 sl=253.55 alert=retest2 |

### Cycle 131 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 253.80 | 252.31 | 252.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 12:15:00 | 253.95 | 252.64 | 252.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 253.35 | 256.59 | 255.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 253.35 | 256.59 | 255.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 253.35 | 256.59 | 255.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 253.35 | 256.59 | 255.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 251.90 | 255.66 | 255.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 251.90 | 255.66 | 255.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2025-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 12:15:00 | 252.80 | 254.54 | 254.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 13:15:00 | 252.00 | 254.03 | 254.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 11:15:00 | 238.97 | 238.68 | 241.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 11:45:00 | 238.65 | 238.68 | 241.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 243.16 | 239.58 | 241.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:00:00 | 243.16 | 239.58 | 241.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 242.83 | 240.23 | 241.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 239.97 | 241.10 | 242.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 10:15:00 | 246.05 | 242.33 | 242.49 | SL hit (close>static) qty=1.00 sl=243.39 alert=retest2 |

### Cycle 133 — BUY (started 2025-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 11:15:00 | 245.97 | 243.06 | 242.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 13:15:00 | 249.62 | 244.82 | 243.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 260.83 | 262.09 | 258.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 260.83 | 262.09 | 258.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 259.19 | 261.10 | 259.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 259.19 | 261.10 | 259.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 258.86 | 260.65 | 259.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:45:00 | 259.09 | 260.65 | 259.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 262.15 | 259.67 | 259.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 13:45:00 | 262.75 | 260.95 | 259.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 11:45:00 | 262.60 | 264.63 | 263.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 10:15:00 | 259.98 | 262.91 | 263.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 10:15:00 | 259.98 | 262.91 | 263.10 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 265.00 | 262.44 | 262.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 10:15:00 | 270.85 | 265.92 | 264.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 278.27 | 279.98 | 277.01 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 13:45:00 | 282.96 | 280.41 | 278.12 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 278.60 | 279.82 | 278.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 282.35 | 279.82 | 278.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 279.86 | 279.83 | 278.39 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-26 12:15:00 | 277.70 | 279.40 | 278.56 | SL hit (close<ema400) qty=1.00 sl=278.56 alert=retest1 |

### Cycle 136 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 273.26 | 277.49 | 277.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 09:15:00 | 272.86 | 275.91 | 276.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 11:15:00 | 276.73 | 275.89 | 276.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 11:15:00 | 276.73 | 275.89 | 276.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 276.73 | 275.89 | 276.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:00:00 | 276.73 | 275.89 | 276.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 276.62 | 276.04 | 276.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:30:00 | 277.15 | 276.04 | 276.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 276.95 | 276.22 | 276.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:45:00 | 277.27 | 276.22 | 276.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 276.36 | 276.25 | 276.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:45:00 | 278.05 | 276.25 | 276.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 279.00 | 276.80 | 276.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 279.90 | 276.80 | 276.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 280.85 | 277.61 | 277.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 10:15:00 | 283.47 | 278.78 | 277.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 12:15:00 | 278.60 | 279.36 | 278.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 12:15:00 | 278.60 | 279.36 | 278.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 278.60 | 279.36 | 278.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 13:00:00 | 278.60 | 279.36 | 278.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 278.25 | 279.14 | 278.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 13:45:00 | 277.70 | 279.14 | 278.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 278.50 | 279.01 | 278.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:15:00 | 278.35 | 279.01 | 278.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 278.35 | 278.88 | 278.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 279.40 | 278.88 | 278.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 282.25 | 279.55 | 278.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 12:15:00 | 284.80 | 280.66 | 279.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:30:00 | 284.60 | 283.08 | 281.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 09:45:00 | 285.75 | 285.32 | 284.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 11:15:00 | 279.00 | 283.56 | 283.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 279.00 | 283.56 | 283.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 272.10 | 279.17 | 281.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 280.30 | 277.23 | 278.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 10:15:00 | 280.30 | 277.23 | 278.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 280.30 | 277.23 | 278.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 11:00:00 | 280.30 | 277.23 | 278.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 281.40 | 278.07 | 278.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:15:00 | 281.35 | 278.07 | 278.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 283.00 | 279.05 | 279.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:45:00 | 283.45 | 279.05 | 279.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2025-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 13:15:00 | 285.35 | 280.31 | 279.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 14:15:00 | 286.10 | 281.47 | 280.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 09:15:00 | 291.50 | 291.66 | 288.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 09:45:00 | 292.00 | 291.66 | 288.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 301.20 | 304.08 | 302.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 301.20 | 304.08 | 302.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 299.45 | 303.15 | 302.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 299.25 | 303.15 | 302.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 14:15:00 | 300.45 | 301.60 | 301.60 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 09:15:00 | 303.30 | 301.70 | 301.64 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 13:15:00 | 300.80 | 301.54 | 301.59 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 15:15:00 | 302.20 | 301.71 | 301.66 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 296.50 | 300.67 | 301.19 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 11:15:00 | 307.85 | 301.30 | 300.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 13:15:00 | 308.40 | 303.78 | 301.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 12:15:00 | 312.65 | 312.93 | 309.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 13:00:00 | 312.65 | 312.93 | 309.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 13:15:00 | 310.55 | 312.45 | 309.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:00:00 | 310.55 | 312.45 | 309.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 309.95 | 311.95 | 309.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 315.95 | 311.55 | 309.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 13:15:00 | 312.05 | 312.62 | 311.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 09:15:00 | 321.70 | 311.97 | 311.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 14:15:00 | 311.65 | 315.16 | 315.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 311.65 | 315.16 | 315.43 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 316.80 | 315.11 | 314.98 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 11:15:00 | 312.75 | 314.57 | 314.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 308.95 | 313.32 | 314.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 14:15:00 | 307.20 | 306.87 | 309.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 14:30:00 | 306.60 | 306.87 | 309.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 313.55 | 308.24 | 309.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 10:30:00 | 311.20 | 308.61 | 309.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 11:15:00 | 310.35 | 308.51 | 308.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 11:15:00 | 310.35 | 308.51 | 308.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 14:15:00 | 312.50 | 309.99 | 309.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 10:15:00 | 316.85 | 318.23 | 316.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 11:00:00 | 316.85 | 318.23 | 316.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 317.35 | 317.95 | 316.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:45:00 | 316.85 | 317.95 | 316.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 317.10 | 317.78 | 316.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 316.30 | 317.78 | 316.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 318.15 | 317.86 | 316.95 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 312.85 | 315.97 | 316.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 13:15:00 | 310.90 | 314.41 | 314.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 317.95 | 314.65 | 314.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 317.95 | 314.65 | 314.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 317.95 | 314.65 | 314.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:45:00 | 319.35 | 314.65 | 314.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 317.75 | 315.27 | 315.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 318.50 | 316.22 | 315.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 318.50 | 320.92 | 319.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 318.50 | 320.92 | 319.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 318.50 | 320.92 | 319.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 318.30 | 320.92 | 319.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 321.20 | 320.98 | 319.45 | EMA400 retest candle locked (from upside) |

### Cycle 152 — SELL (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 15:15:00 | 317.00 | 318.62 | 318.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 315.35 | 317.96 | 318.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 10:15:00 | 319.25 | 318.22 | 318.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 10:15:00 | 319.25 | 318.22 | 318.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 319.25 | 318.22 | 318.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:45:00 | 319.75 | 318.22 | 318.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 319.35 | 318.45 | 318.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:30:00 | 319.90 | 318.45 | 318.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 319.30 | 318.62 | 318.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:30:00 | 319.40 | 318.62 | 318.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 13:15:00 | 320.50 | 318.99 | 318.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 14:15:00 | 321.65 | 319.52 | 319.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 10:15:00 | 319.30 | 319.76 | 319.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 10:15:00 | 319.30 | 319.76 | 319.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 319.30 | 319.76 | 319.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:00:00 | 319.30 | 319.76 | 319.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 319.70 | 319.75 | 319.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:00:00 | 319.70 | 319.75 | 319.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 318.60 | 319.52 | 319.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:00:00 | 318.60 | 319.52 | 319.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 318.00 | 319.22 | 319.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:00:00 | 318.00 | 319.22 | 319.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 319.40 | 319.36 | 319.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 321.05 | 319.36 | 319.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 318.20 | 319.13 | 319.16 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 11:15:00 | 320.20 | 319.32 | 319.24 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 317.50 | 318.92 | 319.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 14:15:00 | 316.60 | 317.59 | 318.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 311.75 | 311.46 | 313.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 12:00:00 | 311.75 | 311.46 | 313.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 313.70 | 312.05 | 313.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:00:00 | 313.70 | 312.05 | 313.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 311.70 | 311.98 | 313.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:30:00 | 313.45 | 311.98 | 313.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 313.65 | 312.32 | 313.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:00:00 | 313.65 | 312.32 | 313.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 311.55 | 312.16 | 312.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:15:00 | 310.40 | 312.16 | 312.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 10:15:00 | 315.75 | 312.35 | 312.59 | SL hit (close>static) qty=1.00 sl=314.15 alert=retest2 |

### Cycle 157 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 316.25 | 313.13 | 312.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 317.55 | 314.02 | 313.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 14:15:00 | 320.10 | 320.70 | 318.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 15:00:00 | 320.10 | 320.70 | 318.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 321.85 | 328.91 | 325.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 321.85 | 328.91 | 325.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 322.55 | 327.63 | 325.23 | EMA400 retest candle locked (from upside) |

### Cycle 158 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 319.00 | 323.23 | 323.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 309.45 | 319.84 | 322.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 316.25 | 312.97 | 315.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 13:15:00 | 316.25 | 312.97 | 315.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 316.25 | 312.97 | 315.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 316.25 | 312.97 | 315.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 316.65 | 313.70 | 315.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 316.35 | 313.70 | 315.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 315.85 | 314.13 | 315.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 318.60 | 314.13 | 315.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 318.05 | 315.59 | 316.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:15:00 | 319.85 | 315.59 | 316.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 315.50 | 315.86 | 316.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:30:00 | 316.45 | 315.86 | 316.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 314.45 | 313.38 | 314.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:30:00 | 314.55 | 313.38 | 314.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 316.30 | 313.96 | 314.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:30:00 | 316.70 | 313.96 | 314.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 316.00 | 314.37 | 314.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 314.65 | 314.37 | 314.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 314.00 | 313.20 | 313.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 14:00:00 | 314.00 | 313.20 | 313.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 313.90 | 313.34 | 313.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 15:15:00 | 311.60 | 313.34 | 313.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 09:45:00 | 312.40 | 312.88 | 313.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:30:00 | 312.70 | 313.52 | 313.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 14:15:00 | 312.85 | 313.51 | 313.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 313.25 | 313.45 | 313.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:45:00 | 313.45 | 313.45 | 313.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 313.00 | 313.36 | 313.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 312.30 | 313.36 | 313.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 12:45:00 | 312.35 | 312.08 | 312.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 13:15:00 | 314.10 | 312.48 | 312.90 | SL hit (close>static) qty=1.00 sl=314.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 323.20 | 314.91 | 313.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 14:15:00 | 329.80 | 325.02 | 321.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 11:15:00 | 332.15 | 332.38 | 328.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 12:00:00 | 332.15 | 332.38 | 328.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 332.80 | 332.40 | 331.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:45:00 | 332.20 | 332.40 | 331.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 330.20 | 331.92 | 331.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 330.70 | 331.92 | 331.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 330.75 | 331.68 | 331.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:45:00 | 329.00 | 331.68 | 331.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 331.80 | 332.42 | 331.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:30:00 | 331.85 | 332.42 | 331.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 331.45 | 332.23 | 331.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:30:00 | 331.15 | 332.23 | 331.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 331.75 | 332.13 | 331.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:45:00 | 331.60 | 332.13 | 331.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 331.75 | 332.06 | 331.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:30:00 | 331.00 | 332.06 | 331.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 331.15 | 331.87 | 331.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 334.10 | 331.87 | 331.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 345.90 | 349.79 | 349.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 345.90 | 349.79 | 349.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 345.20 | 348.87 | 349.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 346.80 | 346.53 | 347.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 09:30:00 | 346.15 | 346.53 | 347.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 346.80 | 346.58 | 347.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 347.05 | 346.58 | 347.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 348.75 | 345.80 | 346.69 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 348.50 | 347.37 | 347.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 15:15:00 | 349.00 | 347.69 | 347.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 12:15:00 | 347.85 | 348.62 | 348.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 12:15:00 | 347.85 | 348.62 | 348.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 347.85 | 348.62 | 348.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:00:00 | 347.85 | 348.62 | 348.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 348.10 | 348.52 | 348.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:45:00 | 348.45 | 348.52 | 348.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 347.85 | 348.38 | 348.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 15:00:00 | 347.85 | 348.38 | 348.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 348.50 | 348.41 | 348.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 345.45 | 348.41 | 348.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — SELL (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 09:15:00 | 345.50 | 347.83 | 347.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 13:15:00 | 343.95 | 346.00 | 346.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 343.45 | 343.17 | 344.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 343.45 | 343.17 | 344.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 343.45 | 343.17 | 344.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:00:00 | 340.90 | 342.72 | 344.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 14:15:00 | 344.75 | 341.27 | 341.87 | SL hit (close>static) qty=1.00 sl=344.35 alert=retest2 |

### Cycle 163 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 343.25 | 342.27 | 342.24 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 340.90 | 341.99 | 342.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 336.40 | 340.68 | 341.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 10:15:00 | 335.95 | 335.83 | 337.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 10:15:00 | 335.95 | 335.83 | 337.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 335.95 | 335.83 | 337.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:30:00 | 336.30 | 335.83 | 337.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 334.55 | 334.41 | 336.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:45:00 | 335.55 | 334.41 | 336.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 335.70 | 334.67 | 336.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:00:00 | 335.70 | 334.67 | 336.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 334.75 | 334.69 | 335.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:15:00 | 332.35 | 334.89 | 335.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 11:15:00 | 338.90 | 335.39 | 335.66 | SL hit (close>static) qty=1.00 sl=336.75 alert=retest2 |

### Cycle 165 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 340.75 | 336.46 | 336.12 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 326.90 | 334.99 | 335.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 10:15:00 | 321.90 | 327.82 | 331.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 15:15:00 | 318.05 | 317.99 | 321.81 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 09:15:00 | 313.70 | 317.99 | 321.81 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 317.10 | 315.90 | 318.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:30:00 | 317.05 | 315.90 | 318.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 316.80 | 312.30 | 313.98 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 316.80 | 312.30 | 313.98 | SL hit (close>ema400) qty=1.00 sl=313.98 alert=retest1 |

### Cycle 167 — BUY (started 2025-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 12:15:00 | 319.45 | 315.42 | 315.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 11:15:00 | 320.35 | 318.57 | 317.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 10:15:00 | 322.65 | 323.31 | 321.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 10:45:00 | 322.80 | 323.31 | 321.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 321.85 | 323.01 | 321.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 321.85 | 323.01 | 321.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 322.15 | 322.84 | 321.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:30:00 | 321.10 | 322.84 | 321.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 323.70 | 322.88 | 321.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 321.70 | 322.88 | 321.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 320.25 | 322.36 | 321.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 320.25 | 322.36 | 321.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 319.35 | 321.76 | 321.56 | EMA400 retest candle locked (from upside) |

### Cycle 168 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 319.30 | 321.26 | 321.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 13:15:00 | 318.20 | 320.31 | 320.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 317.10 | 315.57 | 317.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 317.10 | 315.57 | 317.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 317.10 | 315.57 | 317.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 317.10 | 315.57 | 317.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 317.55 | 315.97 | 317.33 | EMA400 retest candle locked (from downside) |

### Cycle 169 — BUY (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 15:15:00 | 321.90 | 318.66 | 318.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 10:15:00 | 322.95 | 320.91 | 319.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 320.85 | 320.90 | 319.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:00:00 | 320.85 | 320.90 | 319.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 319.60 | 320.64 | 319.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:30:00 | 319.70 | 320.64 | 319.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 320.15 | 320.54 | 319.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 319.15 | 320.54 | 319.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 320.15 | 320.46 | 319.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:15:00 | 320.05 | 320.46 | 319.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 320.05 | 320.38 | 319.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 317.70 | 320.38 | 319.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 318.30 | 319.96 | 319.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:45:00 | 317.70 | 319.96 | 319.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 317.40 | 319.45 | 319.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 316.65 | 318.61 | 319.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 12:15:00 | 316.45 | 316.35 | 317.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 12:45:00 | 316.20 | 316.35 | 317.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 316.55 | 316.39 | 317.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:45:00 | 317.20 | 316.39 | 317.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 314.55 | 315.86 | 316.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:15:00 | 313.60 | 315.86 | 316.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 12:15:00 | 313.35 | 315.17 | 316.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 14:00:00 | 313.70 | 314.60 | 315.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 14:30:00 | 313.95 | 314.09 | 315.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 311.80 | 313.44 | 314.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:15:00 | 309.60 | 312.04 | 313.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 10:45:00 | 310.10 | 311.12 | 312.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 314.65 | 312.19 | 311.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 314.65 | 312.19 | 311.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 316.40 | 313.35 | 312.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 314.40 | 314.67 | 313.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 314.40 | 314.67 | 313.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 314.95 | 316.04 | 315.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 314.95 | 316.04 | 315.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 315.00 | 315.83 | 315.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 315.25 | 315.83 | 315.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 312.70 | 314.93 | 314.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 312.70 | 314.93 | 314.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 312.00 | 314.00 | 314.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 314.95 | 313.69 | 314.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 314.95 | 313.69 | 314.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 314.95 | 313.69 | 314.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 314.95 | 313.69 | 314.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 312.65 | 313.48 | 314.03 | EMA400 retest candle locked (from downside) |

### Cycle 173 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 315.90 | 314.52 | 314.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 12:15:00 | 317.80 | 315.18 | 314.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 12:15:00 | 315.30 | 316.47 | 315.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 12:15:00 | 315.30 | 316.47 | 315.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 315.30 | 316.47 | 315.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:00:00 | 315.30 | 316.47 | 315.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 314.95 | 316.17 | 315.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:30:00 | 314.25 | 316.17 | 315.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 315.70 | 315.90 | 315.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:45:00 | 315.00 | 315.90 | 315.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 316.00 | 315.92 | 315.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 13:30:00 | 316.60 | 316.09 | 315.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 11:15:00 | 317.45 | 318.58 | 318.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 11:15:00 | 317.45 | 318.58 | 318.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 13:15:00 | 315.85 | 317.93 | 318.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 15:15:00 | 318.85 | 318.11 | 318.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 15:15:00 | 318.85 | 318.11 | 318.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 318.85 | 318.11 | 318.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 319.65 | 318.11 | 318.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 320.70 | 318.63 | 318.60 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 11:15:00 | 317.45 | 318.47 | 318.54 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 323.10 | 319.29 | 318.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 10:15:00 | 323.80 | 320.20 | 319.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 11:15:00 | 322.25 | 322.43 | 321.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 11:15:00 | 322.25 | 322.43 | 321.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 322.25 | 322.43 | 321.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:45:00 | 321.55 | 322.43 | 321.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 326.80 | 328.86 | 327.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:45:00 | 326.20 | 328.86 | 327.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 325.45 | 328.18 | 327.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 323.70 | 328.18 | 327.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 331.45 | 330.56 | 329.18 | EMA400 retest candle locked (from upside) |

### Cycle 178 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 325.50 | 328.74 | 329.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 10:15:00 | 324.05 | 327.80 | 328.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 335.30 | 327.64 | 327.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 335.30 | 327.64 | 327.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 335.30 | 327.64 | 327.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 334.75 | 327.64 | 327.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — BUY (started 2025-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 10:15:00 | 335.45 | 329.20 | 328.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 09:15:00 | 340.40 | 335.79 | 332.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 13:15:00 | 339.00 | 340.36 | 337.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 13:30:00 | 339.10 | 340.36 | 337.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 339.95 | 339.87 | 338.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 12:45:00 | 341.20 | 339.83 | 338.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 13:15:00 | 341.00 | 339.83 | 338.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 14:30:00 | 340.80 | 340.71 | 339.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 12:15:00 | 341.50 | 343.92 | 344.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — SELL (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 12:15:00 | 341.50 | 343.92 | 344.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 13:15:00 | 340.40 | 343.22 | 343.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 338.95 | 334.76 | 336.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 10:15:00 | 338.95 | 334.76 | 336.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 338.95 | 334.76 | 336.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 338.95 | 334.76 | 336.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 338.40 | 335.49 | 336.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 338.40 | 335.49 | 336.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 338.95 | 336.18 | 336.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 13:45:00 | 337.40 | 336.18 | 336.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 335.40 | 337.01 | 337.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:30:00 | 336.65 | 336.33 | 336.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 11:15:00 | 337.10 | 336.03 | 336.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 337.10 | 336.03 | 336.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 13:15:00 | 339.50 | 337.06 | 336.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 333.90 | 337.19 | 336.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 333.90 | 337.19 | 336.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 333.90 | 337.19 | 336.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 333.90 | 337.19 | 336.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 333.65 | 336.48 | 336.58 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 335.95 | 334.60 | 334.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 12:15:00 | 341.10 | 335.90 | 335.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 357.25 | 357.68 | 353.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 15:00:00 | 357.25 | 357.68 | 353.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 368.05 | 369.58 | 367.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 368.05 | 369.58 | 367.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 363.85 | 368.26 | 367.19 | EMA400 retest candle locked (from upside) |

### Cycle 184 — SELL (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 12:15:00 | 364.35 | 366.16 | 366.39 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 368.90 | 366.68 | 366.54 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 12:15:00 | 364.25 | 366.12 | 366.31 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 11:15:00 | 369.65 | 366.38 | 366.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 12:15:00 | 371.25 | 367.35 | 366.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 374.05 | 374.73 | 373.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 14:00:00 | 374.05 | 374.73 | 373.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 375.20 | 374.79 | 373.49 | EMA400 retest candle locked (from upside) |

### Cycle 188 — SELL (started 2025-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 13:15:00 | 370.20 | 372.92 | 372.96 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 379.95 | 373.91 | 373.36 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 372.05 | 373.78 | 373.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 10:15:00 | 368.30 | 371.95 | 372.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 11:15:00 | 366.85 | 366.18 | 367.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 12:00:00 | 366.85 | 366.18 | 367.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 365.80 | 365.67 | 366.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:30:00 | 365.20 | 365.67 | 366.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 361.70 | 357.82 | 360.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:15:00 | 363.20 | 357.82 | 360.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 365.25 | 359.31 | 360.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 365.25 | 359.31 | 360.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 365.95 | 361.74 | 361.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 367.90 | 363.76 | 362.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 363.80 | 364.40 | 363.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 363.80 | 364.40 | 363.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 363.80 | 364.40 | 363.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:00:00 | 363.80 | 364.40 | 363.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 363.75 | 364.27 | 363.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 363.75 | 364.27 | 363.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 363.70 | 364.15 | 363.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:45:00 | 363.85 | 364.15 | 363.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 364.05 | 364.04 | 363.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:45:00 | 361.75 | 364.04 | 363.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 364.75 | 364.19 | 363.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:45:00 | 363.55 | 364.19 | 363.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 361.85 | 363.82 | 363.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 361.85 | 363.82 | 363.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 363.00 | 363.65 | 363.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:15:00 | 359.85 | 363.65 | 363.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 359.60 | 362.84 | 363.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 13:15:00 | 359.20 | 361.68 | 362.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 09:15:00 | 357.80 | 356.94 | 358.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 357.80 | 356.94 | 358.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 357.80 | 356.94 | 358.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:30:00 | 359.90 | 356.94 | 358.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 358.20 | 357.19 | 358.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 13:45:00 | 356.20 | 357.10 | 358.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:15:00 | 356.05 | 357.56 | 358.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 11:15:00 | 356.35 | 357.38 | 358.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 11:45:00 | 356.40 | 357.24 | 358.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 357.20 | 357.23 | 357.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:00:00 | 357.20 | 357.23 | 357.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 359.55 | 357.70 | 358.12 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-03 13:15:00 | 359.55 | 357.70 | 358.12 | SL hit (close>static) qty=1.00 sl=358.90 alert=retest2 |

### Cycle 193 — BUY (started 2025-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 12:15:00 | 360.90 | 357.57 | 357.36 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 355.80 | 357.82 | 358.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 350.90 | 354.74 | 355.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 356.80 | 353.30 | 354.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 356.80 | 353.30 | 354.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 356.80 | 353.30 | 354.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 357.30 | 353.30 | 354.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 358.70 | 354.38 | 354.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:00:00 | 358.70 | 354.38 | 354.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 359.40 | 355.38 | 355.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 361.80 | 356.67 | 355.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 363.90 | 364.89 | 362.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 363.90 | 364.89 | 362.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 362.15 | 363.99 | 362.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:00:00 | 362.15 | 363.99 | 362.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 363.00 | 363.79 | 362.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:45:00 | 362.50 | 363.79 | 362.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 365.00 | 364.04 | 362.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:30:00 | 366.40 | 364.95 | 363.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 13:15:00 | 361.75 | 364.51 | 364.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — SELL (started 2025-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 13:15:00 | 361.75 | 364.51 | 364.81 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 367.25 | 365.00 | 364.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 369.65 | 366.19 | 365.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 369.70 | 370.16 | 368.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 370.75 | 370.16 | 368.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 369.20 | 369.97 | 368.79 | EMA400 retest candle locked (from upside) |

### Cycle 198 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 366.35 | 368.00 | 368.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 362.65 | 366.46 | 367.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 14:15:00 | 365.90 | 365.39 | 366.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 14:15:00 | 365.90 | 365.39 | 366.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 365.90 | 365.39 | 366.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 15:00:00 | 365.90 | 365.39 | 366.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 367.85 | 365.98 | 366.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:45:00 | 368.20 | 365.98 | 366.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 369.60 | 366.70 | 366.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:30:00 | 369.75 | 366.70 | 366.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — BUY (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 11:15:00 | 368.65 | 367.09 | 366.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 13:15:00 | 370.40 | 367.88 | 367.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 14:15:00 | 370.25 | 370.78 | 369.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 14:15:00 | 370.25 | 370.78 | 369.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 370.25 | 370.78 | 369.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 370.25 | 370.78 | 369.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 369.20 | 370.46 | 369.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 09:15:00 | 373.65 | 370.46 | 369.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 15:15:00 | 377.60 | 379.26 | 379.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — SELL (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 15:15:00 | 377.60 | 379.26 | 379.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 368.20 | 377.05 | 378.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 355.65 | 354.36 | 357.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 355.65 | 354.36 | 357.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 357.00 | 354.89 | 357.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 357.00 | 354.89 | 357.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 358.65 | 355.64 | 357.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 359.00 | 355.64 | 357.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 359.00 | 356.31 | 357.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 356.20 | 356.31 | 357.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 357.45 | 356.59 | 357.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 355.60 | 356.14 | 357.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 354.95 | 355.72 | 356.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 365.85 | 358.11 | 357.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 365.85 | 358.11 | 357.15 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 355.40 | 359.22 | 359.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 353.50 | 357.01 | 358.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 14:15:00 | 352.35 | 352.14 | 354.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 15:00:00 | 352.35 | 352.14 | 354.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 357.00 | 353.10 | 354.67 | EMA400 retest candle locked (from downside) |

### Cycle 203 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 361.45 | 355.65 | 355.29 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 352.30 | 354.68 | 354.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 348.85 | 353.51 | 354.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 357.20 | 353.64 | 354.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 357.20 | 353.64 | 354.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 357.20 | 353.64 | 354.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:15:00 | 358.85 | 353.64 | 354.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 355.60 | 354.03 | 354.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 12:00:00 | 354.15 | 354.06 | 354.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 13:30:00 | 354.05 | 354.32 | 354.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 14:15:00 | 356.20 | 354.70 | 354.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — BUY (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 14:15:00 | 356.20 | 354.70 | 354.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 15:15:00 | 358.25 | 355.41 | 354.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 10:15:00 | 364.25 | 364.73 | 362.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 11:00:00 | 364.25 | 364.73 | 362.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 362.95 | 364.38 | 362.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 362.95 | 364.38 | 362.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 360.00 | 363.50 | 362.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 360.00 | 363.50 | 362.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 363.20 | 363.44 | 362.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:15:00 | 363.45 | 363.44 | 362.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:45:00 | 364.60 | 363.53 | 362.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 10:00:00 | 363.95 | 363.77 | 362.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 10:30:00 | 363.30 | 363.71 | 362.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 361.85 | 363.34 | 362.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 361.85 | 363.34 | 362.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 357.30 | 362.13 | 362.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 357.30 | 362.13 | 362.23 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 367.10 | 362.33 | 361.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 368.60 | 364.14 | 363.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 12:15:00 | 379.00 | 380.19 | 375.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 13:00:00 | 379.00 | 380.19 | 375.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 382.35 | 385.77 | 383.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 388.50 | 385.45 | 384.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 15:00:00 | 387.25 | 386.73 | 385.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:30:00 | 387.25 | 386.68 | 385.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 379.85 | 385.31 | 385.30 | SL hit (close<static) qty=1.00 sl=381.90 alert=retest2 |

### Cycle 208 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 377.40 | 383.73 | 384.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 374.25 | 378.93 | 381.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 373.80 | 373.00 | 375.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 373.80 | 373.00 | 375.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 374.75 | 372.26 | 373.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 374.75 | 372.26 | 373.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 375.00 | 372.81 | 373.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 377.00 | 372.81 | 373.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 377.95 | 374.59 | 374.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 12:15:00 | 379.35 | 376.22 | 375.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 372.65 | 377.31 | 376.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 372.65 | 377.31 | 376.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 372.65 | 377.31 | 376.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 372.65 | 377.31 | 376.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 373.75 | 376.60 | 376.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 371.85 | 376.60 | 376.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 372.65 | 375.15 | 375.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 370.10 | 374.14 | 374.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 369.75 | 366.98 | 369.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 369.75 | 366.98 | 369.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 369.75 | 366.98 | 369.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 369.70 | 366.98 | 369.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 375.05 | 368.59 | 369.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 375.05 | 368.59 | 369.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 372.20 | 369.31 | 370.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:15:00 | 371.50 | 369.31 | 370.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:45:00 | 371.70 | 370.12 | 370.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 372.00 | 370.50 | 370.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 372.00 | 370.50 | 370.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 15:15:00 | 373.50 | 371.10 | 370.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 12:15:00 | 371.80 | 372.08 | 371.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 12:15:00 | 371.80 | 372.08 | 371.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 371.80 | 372.08 | 371.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:00:00 | 371.80 | 372.08 | 371.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 372.80 | 372.22 | 371.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:30:00 | 371.10 | 372.22 | 371.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 375.00 | 372.78 | 371.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 14:45:00 | 372.40 | 372.78 | 371.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 371.80 | 382.39 | 381.39 | EMA400 retest candle locked (from upside) |

### Cycle 212 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 374.70 | 379.80 | 380.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 372.20 | 378.28 | 379.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 358.60 | 357.61 | 363.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 358.60 | 357.61 | 363.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 327.80 | 326.72 | 329.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 329.60 | 326.72 | 329.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 279.50 | 275.77 | 280.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 281.75 | 275.77 | 280.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 280.80 | 276.78 | 280.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 281.15 | 276.78 | 280.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 282.05 | 277.83 | 280.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 282.05 | 277.83 | 280.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 282.45 | 278.75 | 280.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 285.55 | 278.75 | 280.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 12:15:00 | 283.10 | 281.86 | 281.96 | EMA400 retest candle locked (from downside) |

### Cycle 213 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 285.10 | 282.51 | 282.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 12:15:00 | 286.30 | 283.89 | 283.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 14:15:00 | 283.15 | 283.99 | 283.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 14:15:00 | 283.15 | 283.99 | 283.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 283.15 | 283.99 | 283.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 283.15 | 283.99 | 283.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 281.80 | 283.55 | 283.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 280.70 | 283.55 | 283.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 281.55 | 283.15 | 283.03 | EMA400 retest candle locked (from upside) |

### Cycle 214 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 280.20 | 282.56 | 282.78 | EMA200 below EMA400 |

### Cycle 215 — BUY (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 12:15:00 | 284.55 | 282.98 | 282.93 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-03-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 14:15:00 | 281.25 | 282.71 | 282.82 | EMA200 below EMA400 |

### Cycle 217 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 284.50 | 283.06 | 282.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 286.40 | 283.73 | 283.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-01 13:15:00 | 283.40 | 283.66 | 283.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 13:15:00 | 283.40 | 283.66 | 283.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 283.40 | 283.66 | 283.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:30:00 | 283.20 | 283.66 | 283.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 281.25 | 283.18 | 283.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 15:00:00 | 281.25 | 283.18 | 283.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 218 — SELL (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 15:15:00 | 280.60 | 282.66 | 282.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 267.55 | 279.64 | 281.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 277.05 | 276.37 | 279.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 14:00:00 | 277.05 | 276.37 | 279.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 279.00 | 277.22 | 278.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 273.70 | 277.22 | 278.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:15:00 | 273.80 | 276.91 | 277.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 295.25 | 279.43 | 277.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 219 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 295.25 | 279.43 | 277.87 | EMA200 above EMA400 |

### Cycle 220 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 293.05 | 294.02 | 294.14 | EMA200 below EMA400 |

### Cycle 221 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 305.60 | 296.32 | 295.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 10:15:00 | 307.30 | 298.51 | 296.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 14:15:00 | 307.65 | 307.74 | 304.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 15:00:00 | 307.65 | 307.74 | 304.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 308.55 | 310.45 | 308.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 307.85 | 310.45 | 308.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 315.90 | 316.87 | 314.51 | EMA400 retest candle locked (from upside) |

### Cycle 222 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 310.30 | 313.70 | 314.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 301.90 | 309.69 | 311.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 308.50 | 308.15 | 309.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 309.00 | 308.15 | 309.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 312.50 | 309.02 | 310.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 312.50 | 309.02 | 310.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 312.95 | 309.81 | 310.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 312.90 | 309.81 | 310.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 223 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 312.55 | 311.05 | 310.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 313.70 | 311.76 | 311.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 311.20 | 311.65 | 311.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 09:15:00 | 311.20 | 311.65 | 311.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 311.20 | 311.65 | 311.24 | EMA400 retest candle locked (from upside) |

### Cycle 224 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 11:15:00 | 308.20 | 310.88 | 310.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 12:15:00 | 306.35 | 309.97 | 310.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 12:15:00 | 309.20 | 308.42 | 309.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 12:15:00 | 309.20 | 308.42 | 309.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 309.20 | 308.42 | 309.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:45:00 | 309.20 | 308.42 | 309.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 306.00 | 307.93 | 308.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:15:00 | 305.15 | 307.93 | 308.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 306.35 | 301.98 | 301.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 225 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 306.35 | 301.98 | 301.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 312.00 | 304.61 | 303.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 307.75 | 307.99 | 305.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 10:00:00 | 307.75 | 307.99 | 305.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 305.65 | 307.37 | 305.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:45:00 | 305.80 | 307.37 | 305.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 308.45 | 307.58 | 305.70 | EMA400 retest candle locked (from upside) |

### Cycle 226 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 302.55 | 305.26 | 305.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 13:15:00 | 301.85 | 304.57 | 305.11 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-30 14:00:00 | 180.20 | 2023-05-31 09:15:00 | 182.20 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2023-05-30 15:15:00 | 180.10 | 2023-05-31 09:15:00 | 182.20 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2023-06-16 12:15:00 | 187.98 | 2023-06-20 09:15:00 | 186.68 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2023-06-16 13:00:00 | 188.03 | 2023-06-20 09:15:00 | 186.68 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2023-06-16 14:15:00 | 188.20 | 2023-06-20 09:15:00 | 186.68 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2023-06-16 15:00:00 | 188.43 | 2023-06-20 09:15:00 | 186.68 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2023-06-27 13:15:00 | 179.85 | 2023-06-28 11:15:00 | 182.25 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2023-06-27 14:30:00 | 179.93 | 2023-06-28 11:15:00 | 182.25 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2023-06-27 15:15:00 | 179.93 | 2023-06-28 11:15:00 | 182.25 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2023-06-28 09:45:00 | 179.98 | 2023-06-28 11:15:00 | 182.25 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2023-07-18 11:15:00 | 191.30 | 2023-07-19 13:15:00 | 191.75 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2023-07-19 09:45:00 | 190.83 | 2023-07-19 13:15:00 | 191.75 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2023-07-19 13:00:00 | 191.18 | 2023-07-19 13:15:00 | 191.75 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2023-07-25 11:00:00 | 195.20 | 2023-07-25 13:15:00 | 192.75 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2023-08-01 12:15:00 | 187.70 | 2023-08-08 13:15:00 | 178.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-01 13:45:00 | 187.93 | 2023-08-08 13:15:00 | 178.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-02 09:15:00 | 187.75 | 2023-08-08 13:15:00 | 178.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-01 12:15:00 | 187.70 | 2023-08-09 09:15:00 | 180.00 | STOP_HIT | 0.50 | 4.10% |
| SELL | retest2 | 2023-08-01 13:45:00 | 187.93 | 2023-08-09 09:15:00 | 180.00 | STOP_HIT | 0.50 | 4.22% |
| SELL | retest2 | 2023-08-02 09:15:00 | 187.75 | 2023-08-09 09:15:00 | 180.00 | STOP_HIT | 0.50 | 4.13% |
| SELL | retest2 | 2023-08-16 11:15:00 | 178.80 | 2023-08-24 15:15:00 | 175.73 | STOP_HIT | 1.00 | 1.72% |
| SELL | retest2 | 2023-08-16 12:45:00 | 178.80 | 2023-08-24 15:15:00 | 175.73 | STOP_HIT | 1.00 | 1.72% |
| SELL | retest2 | 2023-08-16 13:30:00 | 178.65 | 2023-08-24 15:15:00 | 175.73 | STOP_HIT | 1.00 | 1.63% |
| SELL | retest2 | 2023-08-17 09:45:00 | 178.95 | 2023-08-24 15:15:00 | 175.73 | STOP_HIT | 1.00 | 1.80% |
| SELL | retest2 | 2023-08-18 10:30:00 | 177.63 | 2023-08-24 15:15:00 | 175.73 | STOP_HIT | 1.00 | 1.07% |
| SELL | retest2 | 2023-08-18 14:15:00 | 177.70 | 2023-08-24 15:15:00 | 175.73 | STOP_HIT | 1.00 | 1.11% |
| BUY | retest2 | 2023-08-28 09:15:00 | 176.55 | 2023-08-30 13:15:00 | 176.13 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-09-08 09:45:00 | 177.65 | 2023-09-12 09:15:00 | 177.95 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest1 | 2023-09-08 10:30:00 | 177.83 | 2023-09-12 09:15:00 | 177.95 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2023-09-22 13:15:00 | 176.48 | 2023-10-06 10:15:00 | 172.95 | STOP_HIT | 1.00 | 2.00% |
| SELL | retest2 | 2023-09-26 09:30:00 | 176.48 | 2023-10-06 10:15:00 | 172.95 | STOP_HIT | 1.00 | 2.00% |
| SELL | retest2 | 2023-09-26 11:15:00 | 176.38 | 2023-10-06 10:15:00 | 172.95 | STOP_HIT | 1.00 | 1.94% |
| SELL | retest2 | 2023-09-26 12:30:00 | 176.43 | 2023-10-06 10:15:00 | 172.95 | STOP_HIT | 1.00 | 1.97% |
| SELL | retest2 | 2023-09-27 09:30:00 | 175.23 | 2023-10-06 10:15:00 | 172.95 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2023-10-16 10:15:00 | 173.00 | 2023-10-20 11:15:00 | 173.58 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2023-10-25 12:30:00 | 170.93 | 2023-10-30 11:15:00 | 173.33 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2023-10-30 10:45:00 | 171.38 | 2023-10-30 11:15:00 | 173.33 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2023-12-18 09:15:00 | 223.45 | 2023-12-19 14:15:00 | 224.83 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2023-12-18 09:45:00 | 223.50 | 2023-12-19 14:15:00 | 224.83 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2023-12-18 15:15:00 | 223.33 | 2023-12-19 14:15:00 | 224.83 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2023-12-19 10:00:00 | 223.23 | 2023-12-19 14:15:00 | 224.83 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-01-05 14:15:00 | 226.10 | 2024-01-08 09:15:00 | 229.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-01-05 15:15:00 | 226.25 | 2024-01-08 09:15:00 | 229.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-01-15 09:15:00 | 230.05 | 2024-01-23 11:15:00 | 234.58 | STOP_HIT | 1.00 | 1.97% |
| BUY | retest2 | 2024-01-15 11:30:00 | 231.10 | 2024-01-23 11:15:00 | 234.58 | STOP_HIT | 1.00 | 1.51% |
| BUY | retest2 | 2024-01-15 13:00:00 | 229.88 | 2024-01-23 11:15:00 | 234.58 | STOP_HIT | 1.00 | 2.04% |
| BUY | retest2 | 2024-01-15 13:45:00 | 230.30 | 2024-01-23 11:15:00 | 234.58 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2024-02-09 12:15:00 | 307.18 | 2024-02-12 09:15:00 | 298.23 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest2 | 2024-02-09 13:30:00 | 306.35 | 2024-02-12 09:15:00 | 298.23 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2024-02-09 14:00:00 | 307.25 | 2024-02-12 09:15:00 | 298.23 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2024-02-09 15:00:00 | 307.30 | 2024-02-12 09:15:00 | 298.23 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2024-02-20 15:15:00 | 331.80 | 2024-02-21 13:15:00 | 319.93 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2024-02-28 10:45:00 | 306.90 | 2024-03-01 11:15:00 | 311.18 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-02-28 14:15:00 | 307.80 | 2024-03-01 11:15:00 | 311.18 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest1 | 2024-03-05 10:45:00 | 321.58 | 2024-03-06 09:15:00 | 313.27 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2024-03-06 15:00:00 | 319.60 | 2024-03-07 10:15:00 | 313.08 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-03-12 11:15:00 | 310.93 | 2024-03-15 09:15:00 | 295.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-12 12:30:00 | 311.10 | 2024-03-15 09:15:00 | 295.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-13 09:15:00 | 311.05 | 2024-03-15 09:15:00 | 295.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-12 11:15:00 | 310.93 | 2024-03-15 12:15:00 | 279.84 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-12 12:30:00 | 311.10 | 2024-03-15 12:15:00 | 279.99 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-13 09:15:00 | 311.05 | 2024-03-15 12:15:00 | 279.94 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-03-26 10:30:00 | 298.65 | 2024-04-04 11:15:00 | 298.30 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2024-03-27 15:15:00 | 298.43 | 2024-04-04 11:15:00 | 298.30 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2024-04-08 12:00:00 | 296.25 | 2024-04-10 10:15:00 | 302.85 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2024-04-08 12:45:00 | 296.95 | 2024-04-10 10:15:00 | 302.85 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-04-08 13:15:00 | 296.08 | 2024-04-10 10:15:00 | 302.85 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2024-05-02 09:15:00 | 309.70 | 2024-05-06 10:15:00 | 307.95 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-05-06 10:15:00 | 306.45 | 2024-05-06 10:15:00 | 307.95 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest2 | 2024-05-17 09:15:00 | 311.65 | 2024-05-28 14:15:00 | 323.80 | STOP_HIT | 1.00 | 3.90% |
| BUY | retest2 | 2024-05-17 10:15:00 | 310.15 | 2024-05-28 14:15:00 | 323.80 | STOP_HIT | 1.00 | 4.40% |
| SELL | retest2 | 2024-06-06 12:30:00 | 295.10 | 2024-06-10 09:15:00 | 302.73 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-06-06 13:45:00 | 294.43 | 2024-06-10 09:15:00 | 302.73 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2024-07-15 09:15:00 | 305.90 | 2024-07-19 11:15:00 | 308.00 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2024-07-19 11:00:00 | 305.30 | 2024-07-19 11:15:00 | 308.00 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest1 | 2024-09-02 09:30:00 | 359.85 | 2024-09-03 09:15:00 | 354.85 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-09-03 12:00:00 | 357.70 | 2024-09-06 10:15:00 | 351.55 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-09-04 09:15:00 | 358.90 | 2024-09-06 10:15:00 | 351.55 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2024-09-04 11:00:00 | 356.90 | 2024-09-06 10:15:00 | 351.55 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-09-04 12:15:00 | 357.70 | 2024-09-06 10:15:00 | 351.55 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-09-04 14:30:00 | 358.50 | 2024-09-06 10:15:00 | 351.55 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2024-09-05 09:30:00 | 358.40 | 2024-09-06 10:15:00 | 351.55 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-09-05 10:30:00 | 360.20 | 2024-09-06 10:15:00 | 351.55 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2024-09-12 11:45:00 | 342.80 | 2024-09-13 09:15:00 | 347.15 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-09-12 15:15:00 | 342.40 | 2024-09-13 09:15:00 | 347.15 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-09-13 13:15:00 | 342.45 | 2024-09-19 11:15:00 | 325.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-13 14:30:00 | 342.40 | 2024-09-19 11:15:00 | 325.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-16 11:30:00 | 341.35 | 2024-09-19 11:15:00 | 324.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-16 12:00:00 | 340.70 | 2024-09-19 11:15:00 | 323.85 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2024-09-16 14:15:00 | 340.90 | 2024-09-19 15:15:00 | 323.66 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2024-09-13 13:15:00 | 342.45 | 2024-09-20 09:15:00 | 329.80 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2024-09-13 14:30:00 | 342.40 | 2024-09-20 09:15:00 | 329.80 | STOP_HIT | 0.50 | 3.68% |
| SELL | retest2 | 2024-09-16 11:30:00 | 341.35 | 2024-09-20 09:15:00 | 329.80 | STOP_HIT | 0.50 | 3.38% |
| SELL | retest2 | 2024-09-16 12:00:00 | 340.70 | 2024-09-20 09:15:00 | 329.80 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2024-09-16 14:15:00 | 340.90 | 2024-09-20 09:15:00 | 329.80 | STOP_HIT | 0.50 | 3.26% |
| BUY | retest2 | 2024-09-26 13:00:00 | 341.65 | 2024-09-30 09:15:00 | 375.81 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-09 11:45:00 | 338.75 | 2024-10-14 09:15:00 | 342.45 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-10-09 12:30:00 | 339.55 | 2024-10-14 09:15:00 | 342.45 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-10-09 13:00:00 | 340.05 | 2024-10-14 09:15:00 | 342.45 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-10-09 14:15:00 | 339.05 | 2024-10-14 09:15:00 | 342.45 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-10-10 11:30:00 | 336.55 | 2024-10-14 09:15:00 | 342.45 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-10-11 09:45:00 | 337.65 | 2024-10-14 09:15:00 | 342.45 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-10-11 10:45:00 | 337.70 | 2024-10-14 09:15:00 | 342.45 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-10-21 11:45:00 | 331.30 | 2024-10-25 09:15:00 | 314.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:30:00 | 331.55 | 2024-10-25 09:15:00 | 314.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:00:00 | 331.20 | 2024-10-25 09:15:00 | 314.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 15:00:00 | 331.55 | 2024-10-25 09:15:00 | 314.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 11:45:00 | 331.30 | 2024-10-28 10:15:00 | 310.85 | STOP_HIT | 0.50 | 6.17% |
| SELL | retest2 | 2024-10-21 12:30:00 | 331.55 | 2024-10-28 10:15:00 | 310.85 | STOP_HIT | 0.50 | 6.24% |
| SELL | retest2 | 2024-10-21 14:00:00 | 331.20 | 2024-10-28 10:15:00 | 310.85 | STOP_HIT | 0.50 | 6.14% |
| SELL | retest2 | 2024-10-21 15:00:00 | 331.55 | 2024-10-28 10:15:00 | 310.85 | STOP_HIT | 0.50 | 6.24% |
| SELL | retest2 | 2024-10-30 09:15:00 | 309.55 | 2024-10-30 09:15:00 | 312.80 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-12-03 13:30:00 | 294.70 | 2024-12-05 09:15:00 | 291.85 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-12-03 15:15:00 | 294.50 | 2024-12-05 09:15:00 | 291.85 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-12-05 09:15:00 | 295.20 | 2024-12-05 09:15:00 | 291.85 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest1 | 2024-12-09 11:45:00 | 302.50 | 2024-12-12 10:15:00 | 304.05 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest1 | 2024-12-09 13:30:00 | 302.45 | 2024-12-12 10:15:00 | 304.05 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest1 | 2024-12-10 12:45:00 | 302.95 | 2024-12-12 10:15:00 | 304.05 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2024-12-11 09:15:00 | 306.35 | 2024-12-12 15:15:00 | 301.75 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-12-12 11:45:00 | 304.35 | 2024-12-12 15:15:00 | 301.75 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-12-12 13:30:00 | 304.40 | 2024-12-12 15:15:00 | 301.75 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-12-17 10:15:00 | 294.15 | 2024-12-20 10:15:00 | 297.20 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-12-17 11:00:00 | 294.00 | 2024-12-20 10:15:00 | 297.20 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-12-17 13:00:00 | 294.40 | 2024-12-20 10:15:00 | 297.20 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-12-19 12:30:00 | 294.45 | 2024-12-20 10:15:00 | 297.20 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-12-23 12:15:00 | 290.35 | 2024-12-26 09:15:00 | 297.60 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2024-12-31 09:15:00 | 292.10 | 2025-01-01 12:15:00 | 294.30 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-12-31 10:00:00 | 292.25 | 2025-01-01 12:15:00 | 294.30 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-12-31 12:45:00 | 292.20 | 2025-01-01 12:15:00 | 294.30 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-12-31 14:15:00 | 292.35 | 2025-01-01 12:15:00 | 294.30 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-12-31 15:15:00 | 292.10 | 2025-01-01 12:15:00 | 294.30 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-01-01 09:45:00 | 291.35 | 2025-01-01 12:15:00 | 294.30 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-01-09 09:15:00 | 284.20 | 2025-01-13 09:15:00 | 269.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 10:45:00 | 284.40 | 2025-01-13 09:15:00 | 270.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 09:15:00 | 284.20 | 2025-01-14 10:15:00 | 270.70 | STOP_HIT | 0.50 | 4.75% |
| SELL | retest2 | 2025-01-09 10:45:00 | 284.40 | 2025-01-14 10:15:00 | 270.70 | STOP_HIT | 0.50 | 4.82% |
| BUY | retest2 | 2025-01-22 14:45:00 | 276.90 | 2025-01-23 10:15:00 | 274.20 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-01-30 14:30:00 | 257.65 | 2025-01-31 14:15:00 | 261.05 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-01-30 15:00:00 | 257.05 | 2025-01-31 14:15:00 | 261.05 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-01-31 10:00:00 | 257.65 | 2025-01-31 14:15:00 | 261.05 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-02-07 15:00:00 | 264.45 | 2025-02-10 09:15:00 | 258.60 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-02-18 09:15:00 | 247.95 | 2025-02-19 09:15:00 | 253.95 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-02-19 10:45:00 | 251.00 | 2025-02-19 11:15:00 | 253.80 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-03-04 09:15:00 | 239.97 | 2025-03-04 10:15:00 | 246.05 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-03-11 13:45:00 | 262.75 | 2025-03-17 10:15:00 | 259.98 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-03-13 11:45:00 | 262.60 | 2025-03-17 10:15:00 | 259.98 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest1 | 2025-03-25 13:45:00 | 282.96 | 2025-03-26 12:15:00 | 277.70 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-04-01 12:15:00 | 284.80 | 2025-04-04 11:15:00 | 279.00 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-04-02 12:30:00 | 284.60 | 2025-04-04 11:15:00 | 279.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-04-04 09:45:00 | 285.75 | 2025-04-04 11:15:00 | 279.00 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-05-02 09:15:00 | 315.95 | 2025-05-06 14:15:00 | 311.65 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-05-02 13:15:00 | 312.05 | 2025-05-06 14:15:00 | 311.65 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-05-05 09:15:00 | 321.70 | 2025-05-06 14:15:00 | 311.65 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-05-12 10:30:00 | 311.20 | 2025-05-14 11:15:00 | 310.35 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-06-05 13:15:00 | 310.40 | 2025-06-06 10:15:00 | 315.75 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-06-19 15:15:00 | 311.60 | 2025-06-23 13:15:00 | 314.10 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-06-20 09:45:00 | 312.40 | 2025-06-23 13:15:00 | 314.10 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-06-20 12:30:00 | 312.70 | 2025-06-24 09:15:00 | 323.20 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2025-06-20 14:15:00 | 312.85 | 2025-06-24 09:15:00 | 323.20 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2025-06-23 09:15:00 | 312.30 | 2025-06-24 09:15:00 | 323.20 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2025-06-23 12:45:00 | 312.35 | 2025-06-24 09:15:00 | 323.20 | STOP_HIT | 1.00 | -3.47% |
| BUY | retest2 | 2025-07-04 09:15:00 | 334.10 | 2025-07-11 09:15:00 | 345.90 | STOP_HIT | 1.00 | 3.53% |
| SELL | retest2 | 2025-07-22 11:00:00 | 340.90 | 2025-07-23 14:15:00 | 344.75 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-07-30 09:15:00 | 332.35 | 2025-07-30 11:15:00 | 338.90 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest1 | 2025-08-05 09:15:00 | 313.70 | 2025-08-08 09:15:00 | 316.80 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-08-26 10:15:00 | 313.60 | 2025-09-01 14:15:00 | 314.65 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-08-26 12:15:00 | 313.35 | 2025-09-01 14:15:00 | 314.65 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-08-26 14:00:00 | 313.70 | 2025-09-01 14:15:00 | 314.65 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-08-26 14:30:00 | 313.95 | 2025-09-01 14:15:00 | 314.65 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-08-29 09:15:00 | 309.60 | 2025-09-01 14:15:00 | 314.65 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-08-29 10:45:00 | 310.10 | 2025-09-01 14:15:00 | 314.65 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-09-04 09:15:00 | 315.25 | 2025-09-04 14:15:00 | 312.70 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-10 13:30:00 | 316.60 | 2025-09-15 11:15:00 | 317.45 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2025-10-03 12:45:00 | 341.20 | 2025-10-10 12:15:00 | 341.50 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-10-03 13:15:00 | 341.00 | 2025-10-10 12:15:00 | 341.50 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2025-10-06 14:30:00 | 340.80 | 2025-10-10 12:15:00 | 341.50 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-10-15 13:45:00 | 337.40 | 2025-10-20 11:15:00 | 337.10 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-10-16 09:15:00 | 335.40 | 2025-10-20 11:15:00 | 337.10 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-10-16 11:30:00 | 336.65 | 2025-10-20 11:15:00 | 337.10 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-12-02 13:45:00 | 356.20 | 2025-12-03 13:15:00 | 359.55 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-12-03 10:15:00 | 356.05 | 2025-12-03 13:15:00 | 359.55 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-12-03 11:15:00 | 356.35 | 2025-12-03 13:15:00 | 359.55 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-12-03 11:45:00 | 356.40 | 2025-12-03 13:15:00 | 359.55 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-12-04 09:45:00 | 356.55 | 2025-12-05 12:15:00 | 360.90 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-12-04 10:45:00 | 356.90 | 2025-12-05 12:15:00 | 360.90 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-12-16 14:30:00 | 366.40 | 2025-12-18 13:15:00 | 361.75 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-12-31 09:15:00 | 373.65 | 2026-01-05 15:15:00 | 377.60 | STOP_HIT | 1.00 | 1.06% |
| SELL | retest2 | 2026-01-13 11:30:00 | 355.60 | 2026-01-16 09:15:00 | 365.85 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2026-01-14 14:15:00 | 354.95 | 2026-01-16 09:15:00 | 365.85 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2026-01-27 12:00:00 | 354.15 | 2026-01-27 14:15:00 | 356.20 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-01-27 13:30:00 | 354.05 | 2026-01-27 14:15:00 | 356.20 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2026-01-30 14:15:00 | 363.45 | 2026-02-01 12:15:00 | 357.30 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2026-01-30 14:45:00 | 364.60 | 2026-02-01 12:15:00 | 357.30 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2026-02-01 10:00:00 | 363.95 | 2026-02-01 12:15:00 | 357.30 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-02-01 10:30:00 | 363.30 | 2026-02-01 12:15:00 | 357.30 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2026-02-11 09:15:00 | 388.50 | 2026-02-12 10:15:00 | 379.85 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-02-11 15:00:00 | 387.25 | 2026-02-12 10:15:00 | 379.85 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-02-12 09:30:00 | 387.25 | 2026-02-12 10:15:00 | 379.85 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2026-02-23 12:15:00 | 371.50 | 2026-02-23 14:15:00 | 372.00 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2026-02-23 13:45:00 | 371.70 | 2026-02-23 14:15:00 | 372.00 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2026-04-06 09:15:00 | 273.70 | 2026-04-08 09:15:00 | 295.25 | STOP_HIT | 1.00 | -7.87% |
| SELL | retest2 | 2026-04-07 09:15:00 | 273.80 | 2026-04-08 09:15:00 | 295.25 | STOP_HIT | 1.00 | -7.83% |
| SELL | retest2 | 2026-04-29 14:15:00 | 305.15 | 2026-05-06 11:15:00 | 306.35 | STOP_HIT | 1.00 | -0.39% |
