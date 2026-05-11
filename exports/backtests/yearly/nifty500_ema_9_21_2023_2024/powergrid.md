# Power Grid Corporation of India Ltd. (POWERGRID)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 313.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 232 |
| ALERT1 | 155 |
| ALERT2 | 153 |
| ALERT2_SKIP | 83 |
| ALERT3 | 460 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 177 |
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 183 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 190 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 66 / 124
- **Target hits / Stop hits / Partials:** 1 / 183 / 6
- **Avg / median % per leg:** -0.09% / -0.58%
- **Sum % (uncompounded):** -17.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 100 | 43 | 43.0% | 0 | 100 | 0 | -0.03% | -3.1% |
| BUY @ 2nd Alert (retest1) | 7 | 0 | 0.0% | 0 | 7 | 0 | -0.59% | -4.1% |
| BUY @ 3rd Alert (retest2) | 93 | 43 | 46.2% | 0 | 93 | 0 | 0.01% | 1.0% |
| SELL (all) | 90 | 23 | 25.6% | 1 | 83 | 6 | -0.17% | -14.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 90 | 23 | 25.6% | 1 | 83 | 6 | -0.17% | -14.9% |
| retest1 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -0.59% | -4.1% |
| retest2 (combined) | 183 | 66 | 36.1% | 1 | 176 | 6 | -0.08% | -13.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 11:15:00 | 177.45 | 176.63 | 176.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 10:15:00 | 178.24 | 177.31 | 176.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 13:15:00 | 177.45 | 177.51 | 177.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-25 14:00:00 | 177.45 | 177.51 | 177.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 14:15:00 | 178.50 | 177.71 | 177.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 14:45:00 | 177.83 | 177.71 | 177.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 176.55 | 177.62 | 177.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 10:00:00 | 176.55 | 177.62 | 177.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 10:15:00 | 176.25 | 177.34 | 177.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 11:00:00 | 176.25 | 177.34 | 177.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 11:15:00 | 177.38 | 177.35 | 177.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 12:30:00 | 177.45 | 177.36 | 177.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 13:15:00 | 177.71 | 177.36 | 177.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-29 10:30:00 | 177.75 | 177.56 | 177.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-29 13:15:00 | 176.25 | 177.17 | 177.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 13:15:00 | 176.25 | 177.17 | 177.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-29 14:15:00 | 175.80 | 176.89 | 177.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 14:15:00 | 175.54 | 175.41 | 175.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 14:15:00 | 175.54 | 175.41 | 175.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 14:15:00 | 175.54 | 175.41 | 175.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 15:00:00 | 175.54 | 175.41 | 175.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 15:15:00 | 173.44 | 175.02 | 175.74 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 14:15:00 | 175.91 | 175.47 | 175.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 09:15:00 | 178.05 | 176.07 | 175.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 13:15:00 | 176.48 | 176.49 | 176.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-05 14:00:00 | 176.48 | 176.49 | 176.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 15:15:00 | 176.44 | 176.53 | 176.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 09:30:00 | 177.53 | 176.68 | 176.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 15:15:00 | 177.04 | 176.52 | 176.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-19 11:15:00 | 182.51 | 183.64 | 183.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 11:15:00 | 182.51 | 183.64 | 183.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 14:15:00 | 182.10 | 183.01 | 183.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 09:15:00 | 184.09 | 183.22 | 183.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 09:15:00 | 184.09 | 183.22 | 183.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 184.09 | 183.22 | 183.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 10:00:00 | 184.09 | 183.22 | 183.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2023-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 10:15:00 | 184.80 | 183.53 | 183.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-20 11:15:00 | 185.33 | 183.89 | 183.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 12:15:00 | 190.73 | 191.10 | 189.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-22 13:00:00 | 190.73 | 191.10 | 189.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 187.50 | 190.07 | 189.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 10:00:00 | 187.50 | 190.07 | 189.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 10:15:00 | 187.73 | 189.60 | 189.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-23 11:15:00 | 187.91 | 189.60 | 189.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-23 13:15:00 | 187.54 | 188.81 | 188.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 13:15:00 | 187.54 | 188.81 | 188.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-26 09:15:00 | 186.19 | 187.92 | 188.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 10:15:00 | 186.56 | 186.54 | 187.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-27 10:30:00 | 186.64 | 186.54 | 187.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 14:15:00 | 187.13 | 186.68 | 187.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 15:00:00 | 187.13 | 186.68 | 187.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 15:15:00 | 186.86 | 186.71 | 187.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-28 09:15:00 | 186.11 | 186.71 | 187.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 09:15:00 | 188.29 | 187.03 | 187.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-28 10:00:00 | 188.29 | 187.03 | 187.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2023-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 10:15:00 | 188.66 | 187.36 | 187.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 12:15:00 | 189.00 | 187.86 | 187.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 14:15:00 | 187.76 | 187.87 | 187.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 14:15:00 | 187.76 | 187.87 | 187.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 187.76 | 187.87 | 187.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 15:00:00 | 187.76 | 187.87 | 187.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 15:15:00 | 189.00 | 188.10 | 187.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 09:15:00 | 191.55 | 188.10 | 187.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 11:30:00 | 189.26 | 188.80 | 188.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-03 14:15:00 | 188.06 | 188.65 | 188.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2023-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 14:15:00 | 188.06 | 188.65 | 188.65 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 11:15:00 | 189.90 | 188.59 | 188.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 09:15:00 | 192.71 | 189.92 | 189.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 10:15:00 | 193.91 | 194.28 | 192.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-07 11:00:00 | 193.91 | 194.28 | 192.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 11:15:00 | 192.34 | 193.89 | 192.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 12:00:00 | 192.34 | 193.89 | 192.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 12:15:00 | 191.78 | 193.47 | 192.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 12:30:00 | 192.19 | 193.47 | 192.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 13:15:00 | 192.04 | 193.19 | 192.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 13:45:00 | 191.96 | 193.19 | 192.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 14:15:00 | 191.59 | 192.87 | 192.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 15:00:00 | 191.59 | 192.87 | 192.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 15:15:00 | 192.00 | 192.69 | 192.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:15:00 | 188.51 | 192.69 | 192.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2023-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 09:15:00 | 188.29 | 191.81 | 191.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 10:15:00 | 187.73 | 191.00 | 191.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 189.34 | 188.72 | 189.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-11 10:00:00 | 189.34 | 188.72 | 189.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 10:15:00 | 190.39 | 189.05 | 189.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 10:30:00 | 191.33 | 189.05 | 189.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 11:15:00 | 190.50 | 189.34 | 189.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 11:45:00 | 190.69 | 189.34 | 189.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 12:15:00 | 190.24 | 189.52 | 189.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-11 15:15:00 | 189.60 | 189.78 | 190.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-12 12:15:00 | 190.50 | 190.11 | 190.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2023-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 12:15:00 | 190.50 | 190.11 | 190.09 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 14:15:00 | 188.78 | 189.92 | 190.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 09:15:00 | 185.44 | 188.93 | 189.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-17 09:15:00 | 181.46 | 181.17 | 183.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 09:15:00 | 181.46 | 181.17 | 183.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 181.46 | 181.17 | 183.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-17 10:30:00 | 181.05 | 181.17 | 183.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-17 11:15:00 | 180.75 | 181.17 | 183.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-18 11:30:00 | 180.98 | 181.04 | 181.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-18 12:45:00 | 181.05 | 181.11 | 181.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 09:15:00 | 183.75 | 181.49 | 181.78 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-07-19 09:15:00 | 183.75 | 181.49 | 181.78 | SL hit (close>static) qty=1.00 sl=183.71 alert=retest2 |

### Cycle 13 — BUY (started 2023-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 11:15:00 | 183.08 | 182.02 | 181.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-20 14:15:00 | 183.23 | 182.46 | 182.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 14:15:00 | 183.26 | 183.28 | 182.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 14:15:00 | 183.26 | 183.28 | 182.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 14:15:00 | 183.26 | 183.28 | 182.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 15:00:00 | 183.26 | 183.28 | 182.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 15:15:00 | 182.89 | 183.21 | 182.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 09:15:00 | 182.59 | 183.21 | 182.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 183.68 | 183.30 | 182.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-24 13:00:00 | 184.43 | 183.36 | 183.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-01 12:15:00 | 188.36 | 192.83 | 192.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2023-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 12:15:00 | 188.36 | 192.83 | 192.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 11:15:00 | 187.20 | 189.63 | 191.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 14:15:00 | 186.41 | 186.19 | 187.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-03 15:00:00 | 186.41 | 186.19 | 187.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 185.25 | 185.50 | 186.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 09:15:00 | 183.11 | 185.65 | 186.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-11 11:15:00 | 182.96 | 182.04 | 181.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2023-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 11:15:00 | 182.96 | 182.04 | 181.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-16 13:15:00 | 184.09 | 183.29 | 182.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 09:15:00 | 183.53 | 184.15 | 183.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 09:15:00 | 183.53 | 184.15 | 183.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 183.53 | 184.15 | 183.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 09:45:00 | 182.96 | 184.15 | 183.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 10:15:00 | 183.38 | 183.99 | 183.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 11:15:00 | 182.63 | 183.99 | 183.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 11:15:00 | 181.76 | 183.55 | 183.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 12:00:00 | 181.76 | 183.55 | 183.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 12:15:00 | 181.99 | 183.24 | 183.18 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-08-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 13:15:00 | 182.29 | 183.05 | 183.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 09:15:00 | 181.35 | 182.59 | 182.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 09:15:00 | 183.60 | 181.87 | 182.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 09:15:00 | 183.60 | 181.87 | 182.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 183.60 | 181.87 | 182.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 09:45:00 | 183.64 | 181.87 | 182.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 183.71 | 182.24 | 182.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 11:15:00 | 184.20 | 182.24 | 182.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2023-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 11:15:00 | 184.35 | 182.66 | 182.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 12:15:00 | 185.06 | 183.14 | 182.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 10:15:00 | 185.70 | 185.72 | 184.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-23 10:30:00 | 185.81 | 185.72 | 184.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 13:15:00 | 186.19 | 186.99 | 186.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 13:30:00 | 186.30 | 186.99 | 186.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 14:15:00 | 184.76 | 186.54 | 186.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 14:45:00 | 184.84 | 186.54 | 186.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 184.73 | 186.18 | 186.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 09:15:00 | 185.25 | 186.18 | 186.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2023-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 09:15:00 | 184.61 | 185.87 | 185.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 10:15:00 | 183.56 | 185.41 | 185.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 183.75 | 183.07 | 184.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-28 10:00:00 | 183.75 | 183.07 | 184.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 10:15:00 | 184.09 | 183.27 | 184.14 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 14:15:00 | 186.30 | 184.52 | 184.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-28 15:15:00 | 186.53 | 184.92 | 184.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 09:15:00 | 186.15 | 187.57 | 186.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 09:15:00 | 186.15 | 187.57 | 186.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 09:15:00 | 186.15 | 187.57 | 186.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 10:00:00 | 186.15 | 187.57 | 186.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 10:15:00 | 186.26 | 187.31 | 186.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 10:30:00 | 186.30 | 187.31 | 186.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 12:15:00 | 186.04 | 186.87 | 186.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 12:45:00 | 185.93 | 186.87 | 186.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2023-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-30 15:15:00 | 185.55 | 186.21 | 186.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 13:15:00 | 184.20 | 185.71 | 186.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 09:15:00 | 185.36 | 185.09 | 185.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 09:15:00 | 185.36 | 185.09 | 185.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 09:15:00 | 185.36 | 185.09 | 185.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 10:00:00 | 185.36 | 185.09 | 185.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 10:15:00 | 186.30 | 185.33 | 185.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 10:30:00 | 185.78 | 185.33 | 185.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 11:15:00 | 186.60 | 185.59 | 185.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 11:45:00 | 186.90 | 185.59 | 185.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2023-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 12:15:00 | 187.39 | 185.95 | 185.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 14:15:00 | 189.15 | 186.84 | 186.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 12:15:00 | 190.28 | 190.65 | 189.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-05 12:45:00 | 190.31 | 190.65 | 189.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 11:15:00 | 190.05 | 190.33 | 189.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 11:45:00 | 189.60 | 190.33 | 189.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 12:15:00 | 189.38 | 190.14 | 189.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 13:00:00 | 189.38 | 190.14 | 189.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 13:15:00 | 189.53 | 190.02 | 189.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 14:15:00 | 190.05 | 190.02 | 189.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 10:30:00 | 190.09 | 190.16 | 189.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 12:30:00 | 189.94 | 190.47 | 190.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 14:15:00 | 191.00 | 193.36 | 193.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2023-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 14:15:00 | 191.00 | 193.36 | 193.58 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-09-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 09:15:00 | 194.25 | 193.58 | 193.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-18 10:15:00 | 195.50 | 194.64 | 194.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 11:15:00 | 202.45 | 203.01 | 200.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-21 12:00:00 | 202.45 | 203.01 | 200.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 12:15:00 | 201.60 | 202.73 | 200.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 12:45:00 | 201.50 | 202.73 | 200.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 13:15:00 | 201.55 | 202.49 | 200.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 13:30:00 | 201.55 | 202.49 | 200.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 14:15:00 | 201.55 | 202.30 | 200.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 15:00:00 | 201.55 | 202.30 | 200.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 197.50 | 201.32 | 200.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-22 09:45:00 | 197.85 | 201.32 | 200.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 199.25 | 200.90 | 200.58 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 12:15:00 | 197.10 | 199.77 | 200.10 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 15:15:00 | 199.35 | 198.99 | 198.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 09:15:00 | 201.35 | 199.46 | 199.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 12:15:00 | 198.85 | 199.62 | 199.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 12:15:00 | 198.85 | 199.62 | 199.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 12:15:00 | 198.85 | 199.62 | 199.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 13:00:00 | 198.85 | 199.62 | 199.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 198.95 | 199.48 | 199.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 13:45:00 | 198.90 | 199.48 | 199.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 199.85 | 199.56 | 199.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 14:30:00 | 198.90 | 199.56 | 199.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 200.25 | 199.76 | 199.50 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 10:15:00 | 198.00 | 199.30 | 199.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 09:15:00 | 197.75 | 199.03 | 199.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-04 14:15:00 | 199.05 | 198.41 | 198.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 14:15:00 | 199.05 | 198.41 | 198.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 14:15:00 | 199.05 | 198.41 | 198.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-04 15:00:00 | 199.05 | 198.41 | 198.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 15:15:00 | 199.05 | 198.54 | 198.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 09:15:00 | 197.55 | 198.54 | 198.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 196.15 | 198.06 | 198.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 10:15:00 | 195.80 | 198.06 | 198.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 12:30:00 | 196.00 | 197.16 | 198.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 13:00:00 | 195.95 | 197.16 | 198.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 14:15:00 | 196.00 | 197.05 | 197.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 13:15:00 | 197.30 | 196.81 | 197.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 13:45:00 | 197.15 | 196.81 | 197.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 14:15:00 | 197.10 | 196.87 | 197.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-06 15:15:00 | 196.95 | 196.87 | 197.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-10 11:15:00 | 198.00 | 196.60 | 196.62 | SL hit (close>static) qty=1.00 sl=197.40 alert=retest2 |

### Cycle 27 — BUY (started 2023-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 12:15:00 | 198.00 | 196.88 | 196.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 09:15:00 | 199.00 | 198.23 | 197.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 09:15:00 | 200.10 | 200.18 | 199.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-13 10:15:00 | 199.80 | 200.18 | 199.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 200.55 | 201.07 | 200.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 10:15:00 | 201.40 | 201.07 | 200.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-20 09:15:00 | 201.70 | 205.01 | 205.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2023-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 09:15:00 | 201.70 | 205.01 | 205.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 12:15:00 | 200.20 | 202.36 | 203.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 13:15:00 | 199.95 | 199.93 | 201.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-25 14:00:00 | 199.95 | 199.93 | 201.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 15:15:00 | 198.55 | 198.49 | 199.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:15:00 | 200.85 | 198.49 | 199.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 200.10 | 198.81 | 199.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:00:00 | 200.10 | 198.81 | 199.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 201.15 | 199.28 | 199.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 11:00:00 | 201.15 | 199.28 | 199.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2023-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 13:15:00 | 200.90 | 200.11 | 200.06 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 09:15:00 | 198.35 | 199.95 | 200.01 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 15:15:00 | 201.15 | 200.21 | 200.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 14:15:00 | 202.35 | 201.61 | 200.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 15:15:00 | 201.45 | 201.58 | 201.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-01 09:15:00 | 201.40 | 201.58 | 201.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 201.05 | 201.47 | 201.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 10:00:00 | 201.05 | 201.47 | 201.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 10:15:00 | 201.05 | 201.39 | 201.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 10:30:00 | 200.80 | 201.39 | 201.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 11:15:00 | 201.15 | 201.34 | 201.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 11:30:00 | 201.15 | 201.34 | 201.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 12:15:00 | 201.10 | 201.29 | 201.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 12:30:00 | 201.05 | 201.29 | 201.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 13:15:00 | 200.75 | 201.18 | 201.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 09:15:00 | 201.60 | 201.07 | 200.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 12:00:00 | 201.20 | 201.28 | 201.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-15 15:15:00 | 210.00 | 210.74 | 210.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2023-11-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 15:15:00 | 210.00 | 210.74 | 210.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-16 09:15:00 | 207.80 | 210.15 | 210.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-17 10:15:00 | 208.45 | 208.15 | 209.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-17 10:30:00 | 208.85 | 208.15 | 209.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 11:15:00 | 209.60 | 208.44 | 209.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 11:30:00 | 209.80 | 208.44 | 209.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 12:15:00 | 209.90 | 208.73 | 209.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 12:30:00 | 210.45 | 208.73 | 209.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 14:15:00 | 209.70 | 209.08 | 209.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 15:00:00 | 209.70 | 209.08 | 209.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 15:15:00 | 209.00 | 209.06 | 209.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 09:15:00 | 210.15 | 209.06 | 209.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2023-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 09:15:00 | 210.45 | 209.34 | 209.33 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 10:15:00 | 208.35 | 209.24 | 209.36 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 09:15:00 | 211.45 | 209.48 | 209.38 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 09:15:00 | 209.70 | 210.32 | 210.39 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 11:15:00 | 211.35 | 210.45 | 210.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 09:15:00 | 212.10 | 211.01 | 210.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 10:15:00 | 210.80 | 210.97 | 210.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 10:15:00 | 210.80 | 210.97 | 210.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 10:15:00 | 210.80 | 210.97 | 210.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 11:00:00 | 210.80 | 210.97 | 210.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 11:15:00 | 209.80 | 210.73 | 210.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 12:00:00 | 209.80 | 210.73 | 210.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2023-11-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 12:15:00 | 209.30 | 210.45 | 210.49 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 10:15:00 | 212.00 | 210.60 | 210.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 10:15:00 | 212.90 | 211.37 | 210.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 12:15:00 | 229.50 | 229.77 | 226.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-08 13:00:00 | 229.50 | 229.77 | 226.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 13:15:00 | 228.10 | 229.44 | 226.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:45:00 | 227.75 | 229.44 | 226.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 230.20 | 229.33 | 227.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 10:15:00 | 230.85 | 229.33 | 227.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-18 13:15:00 | 231.60 | 233.22 | 233.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2023-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 13:15:00 | 231.60 | 233.22 | 233.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-19 09:15:00 | 231.35 | 232.54 | 232.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-19 10:15:00 | 233.15 | 232.66 | 233.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 10:15:00 | 233.15 | 232.66 | 233.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 233.15 | 232.66 | 233.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 11:00:00 | 233.15 | 232.66 | 233.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 11:15:00 | 232.40 | 232.61 | 232.95 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 15:15:00 | 233.65 | 233.17 | 233.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-20 09:15:00 | 234.90 | 233.52 | 233.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 12:15:00 | 232.35 | 233.70 | 233.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 12:15:00 | 232.35 | 233.70 | 233.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 232.35 | 233.70 | 233.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:00:00 | 232.35 | 233.70 | 233.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 228.60 | 232.68 | 233.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 227.00 | 231.55 | 232.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 10:15:00 | 230.40 | 230.38 | 231.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 10:45:00 | 230.95 | 230.38 | 231.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 11:15:00 | 232.10 | 230.73 | 231.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 12:00:00 | 232.10 | 230.73 | 231.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 12:15:00 | 232.60 | 231.10 | 231.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 12:30:00 | 232.60 | 231.10 | 231.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 13:15:00 | 232.30 | 231.34 | 231.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 13:30:00 | 232.65 | 231.34 | 231.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 15:15:00 | 232.35 | 231.76 | 231.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:15:00 | 232.65 | 231.76 | 231.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 232.90 | 231.99 | 232.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:45:00 | 234.00 | 231.99 | 232.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2023-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 10:15:00 | 232.45 | 232.08 | 232.05 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2023-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-22 12:15:00 | 231.05 | 231.92 | 231.99 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 09:15:00 | 235.15 | 232.43 | 232.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 10:15:00 | 235.95 | 234.50 | 233.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 09:15:00 | 235.15 | 236.49 | 235.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 09:15:00 | 235.15 | 236.49 | 235.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 235.15 | 236.49 | 235.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 10:00:00 | 235.15 | 236.49 | 235.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 10:15:00 | 237.00 | 236.59 | 235.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 09:15:00 | 238.40 | 236.75 | 236.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 10:45:00 | 238.45 | 237.11 | 236.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 11:15:00 | 238.35 | 237.11 | 236.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 11:45:00 | 238.50 | 237.35 | 236.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 15:15:00 | 237.10 | 237.77 | 237.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 09:15:00 | 239.00 | 237.77 | 237.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 10:00:00 | 238.25 | 237.87 | 237.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 12:30:00 | 238.40 | 237.72 | 237.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 14:00:00 | 238.55 | 237.88 | 237.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 14:15:00 | 237.40 | 237.79 | 237.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 15:00:00 | 237.40 | 237.79 | 237.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 15:15:00 | 238.00 | 237.83 | 237.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 09:15:00 | 236.50 | 237.83 | 237.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 237.20 | 237.70 | 237.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 09:30:00 | 234.90 | 237.70 | 237.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 10:15:00 | 237.05 | 237.57 | 237.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 11:45:00 | 238.25 | 237.69 | 237.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 13:15:00 | 238.35 | 237.78 | 237.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 09:15:00 | 240.95 | 237.58 | 237.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-10 13:15:00 | 239.40 | 241.54 | 241.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-01-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 13:15:00 | 239.40 | 241.54 | 241.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-15 09:15:00 | 237.75 | 239.58 | 240.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-15 11:15:00 | 240.45 | 239.65 | 240.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 11:15:00 | 240.45 | 239.65 | 240.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 11:15:00 | 240.45 | 239.65 | 240.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-15 12:00:00 | 240.45 | 239.65 | 240.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 12:15:00 | 240.25 | 239.77 | 240.19 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2024-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 15:15:00 | 241.25 | 240.41 | 240.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 09:15:00 | 241.65 | 240.66 | 240.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 11:15:00 | 240.25 | 240.60 | 240.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 11:15:00 | 240.25 | 240.60 | 240.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 11:15:00 | 240.25 | 240.60 | 240.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 11:45:00 | 240.60 | 240.60 | 240.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2024-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 12:15:00 | 238.65 | 240.21 | 240.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 234.05 | 238.42 | 239.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 14:15:00 | 235.30 | 234.27 | 235.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-19 15:00:00 | 235.30 | 234.27 | 235.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 15:15:00 | 236.70 | 234.76 | 235.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-20 09:15:00 | 237.90 | 234.76 | 235.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 09:15:00 | 239.70 | 235.75 | 236.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-20 09:45:00 | 239.90 | 235.75 | 236.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2024-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 10:15:00 | 240.20 | 236.64 | 236.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-23 09:15:00 | 245.30 | 239.28 | 237.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 14:15:00 | 238.00 | 240.53 | 239.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 14:15:00 | 238.00 | 240.53 | 239.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 14:15:00 | 238.00 | 240.53 | 239.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 15:00:00 | 238.00 | 240.53 | 239.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 15:15:00 | 236.75 | 239.77 | 239.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-24 09:15:00 | 238.55 | 239.77 | 239.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 274.95 | 281.27 | 276.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-06 09:30:00 | 274.70 | 281.27 | 276.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 10:15:00 | 274.75 | 279.97 | 276.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-06 10:30:00 | 273.20 | 279.97 | 276.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 15:15:00 | 275.20 | 276.30 | 275.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-07 09:15:00 | 272.30 | 276.30 | 275.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2024-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 10:15:00 | 271.35 | 274.65 | 274.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-07 11:15:00 | 269.25 | 273.57 | 274.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-08 09:15:00 | 283.30 | 273.08 | 273.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 09:15:00 | 283.30 | 273.08 | 273.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 283.30 | 273.08 | 273.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 09:30:00 | 287.45 | 273.08 | 273.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-02-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 10:15:00 | 280.35 | 274.54 | 274.18 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 11:15:00 | 271.70 | 275.03 | 275.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 09:15:00 | 267.25 | 272.80 | 274.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-12 13:15:00 | 275.30 | 272.43 | 273.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 13:15:00 | 275.30 | 272.43 | 273.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 13:15:00 | 275.30 | 272.43 | 273.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-12 14:00:00 | 275.30 | 272.43 | 273.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 14:15:00 | 269.85 | 271.92 | 273.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 09:15:00 | 265.25 | 271.58 | 272.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 11:15:00 | 268.75 | 270.73 | 272.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 12:00:00 | 267.55 | 270.09 | 271.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 09:15:00 | 268.00 | 269.76 | 271.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 271.45 | 270.12 | 270.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 10:45:00 | 271.95 | 270.12 | 270.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 11:15:00 | 270.95 | 270.29 | 270.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 11:30:00 | 271.80 | 270.29 | 270.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 12:15:00 | 270.90 | 270.41 | 270.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 12:30:00 | 270.40 | 270.41 | 270.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 13:15:00 | 271.40 | 270.61 | 270.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 14:00:00 | 271.40 | 270.61 | 270.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 272.90 | 271.07 | 271.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 15:00:00 | 272.90 | 271.07 | 271.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-14 15:15:00 | 274.05 | 271.66 | 271.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2024-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 15:15:00 | 274.05 | 271.66 | 271.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 276.85 | 272.70 | 271.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 09:15:00 | 274.60 | 277.81 | 275.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 09:15:00 | 274.60 | 277.81 | 275.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 09:15:00 | 274.60 | 277.81 | 275.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 10:00:00 | 274.60 | 277.81 | 275.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 10:15:00 | 276.90 | 277.63 | 275.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-16 11:15:00 | 277.70 | 277.63 | 275.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-16 12:00:00 | 277.25 | 277.55 | 275.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-16 12:45:00 | 277.30 | 277.40 | 275.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 14:00:00 | 278.00 | 276.57 | 276.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 14:15:00 | 276.30 | 276.52 | 276.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 15:00:00 | 276.30 | 276.52 | 276.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 15:15:00 | 276.10 | 276.44 | 276.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 09:15:00 | 282.50 | 276.44 | 276.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-22 09:15:00 | 275.90 | 280.95 | 281.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-02-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 09:15:00 | 275.90 | 280.95 | 281.12 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 12:15:00 | 284.65 | 280.95 | 280.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-26 09:15:00 | 287.65 | 282.63 | 281.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 09:15:00 | 290.90 | 291.02 | 288.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-28 09:45:00 | 290.15 | 291.02 | 288.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 10:15:00 | 287.90 | 290.40 | 288.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 11:00:00 | 287.90 | 290.40 | 288.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 11:15:00 | 285.40 | 289.40 | 288.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 12:00:00 | 285.40 | 289.40 | 288.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 12:15:00 | 283.30 | 288.18 | 287.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 13:00:00 | 283.30 | 288.18 | 287.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2024-02-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 13:15:00 | 282.70 | 287.08 | 287.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 14:15:00 | 279.70 | 285.61 | 286.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 283.50 | 281.27 | 283.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 14:15:00 | 283.50 | 281.27 | 283.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 283.50 | 281.27 | 283.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 15:00:00 | 283.50 | 281.27 | 283.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 283.30 | 281.68 | 283.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:15:00 | 285.40 | 281.68 | 283.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 287.45 | 282.83 | 283.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:45:00 | 287.05 | 282.83 | 283.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2024-03-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 11:15:00 | 288.95 | 284.65 | 284.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-04 09:15:00 | 293.10 | 287.91 | 286.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 10:15:00 | 292.25 | 292.90 | 290.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-05 10:45:00 | 293.05 | 292.90 | 290.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 288.45 | 292.79 | 291.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 288.45 | 292.79 | 291.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 287.95 | 291.82 | 291.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 11:00:00 | 287.95 | 291.82 | 291.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2024-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 12:15:00 | 289.35 | 290.71 | 290.83 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-06 14:15:00 | 293.95 | 291.50 | 291.18 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 10:15:00 | 286.75 | 290.80 | 291.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 11:15:00 | 286.00 | 289.84 | 290.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-12 11:15:00 | 287.50 | 286.68 | 288.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 11:15:00 | 287.50 | 286.68 | 288.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 11:15:00 | 287.50 | 286.68 | 288.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-12 12:00:00 | 287.50 | 286.68 | 288.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 12:15:00 | 285.85 | 286.52 | 288.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-12 15:00:00 | 284.90 | 286.23 | 287.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 09:15:00 | 270.65 | 283.24 | 286.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-15 14:15:00 | 265.85 | 263.98 | 267.96 | SL hit (close>ema200) qty=0.50 sl=263.98 alert=retest2 |

### Cycle 61 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 271.10 | 264.53 | 263.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 12:15:00 | 272.00 | 267.87 | 265.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 270.65 | 274.16 | 271.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 270.65 | 274.16 | 271.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 270.65 | 274.16 | 271.72 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 14:15:00 | 270.10 | 271.10 | 271.11 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 09:15:00 | 276.05 | 271.98 | 271.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 12:15:00 | 277.00 | 273.88 | 272.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-02 12:15:00 | 278.90 | 279.04 | 277.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-02 13:00:00 | 278.90 | 279.04 | 277.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 15:15:00 | 280.65 | 280.71 | 279.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 09:15:00 | 283.20 | 280.71 | 279.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-04 11:15:00 | 275.90 | 279.92 | 279.50 | SL hit (close<static) qty=1.00 sl=279.50 alert=retest2 |

### Cycle 64 — SELL (started 2024-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 14:15:00 | 277.50 | 278.90 | 279.09 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 09:15:00 | 282.75 | 279.08 | 278.87 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-04-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 11:15:00 | 280.60 | 281.47 | 281.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 12:15:00 | 278.10 | 280.80 | 281.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 14:15:00 | 274.45 | 274.02 | 275.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 15:00:00 | 274.45 | 274.02 | 275.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 281.45 | 275.50 | 276.03 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2024-04-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 10:15:00 | 284.50 | 277.30 | 276.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 14:15:00 | 285.25 | 283.61 | 282.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 14:15:00 | 291.75 | 293.15 | 291.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 15:00:00 | 291.75 | 293.15 | 291.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 291.95 | 292.76 | 291.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 09:45:00 | 291.10 | 292.76 | 291.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 291.25 | 292.46 | 291.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 10:45:00 | 291.55 | 292.46 | 291.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 11:15:00 | 292.20 | 292.41 | 291.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 13:15:00 | 292.80 | 292.38 | 291.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 14:15:00 | 292.85 | 292.33 | 291.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 15:00:00 | 293.45 | 292.56 | 291.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 09:15:00 | 301.85 | 306.65 | 306.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 09:15:00 | 301.85 | 306.65 | 306.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 10:15:00 | 297.70 | 304.86 | 306.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 10:15:00 | 298.70 | 298.54 | 301.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 11:00:00 | 298.70 | 298.54 | 301.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 301.50 | 299.56 | 301.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:45:00 | 301.50 | 299.56 | 301.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 300.95 | 299.84 | 301.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 14:00:00 | 300.95 | 299.84 | 301.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 14:15:00 | 301.80 | 300.23 | 301.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 15:00:00 | 301.80 | 300.23 | 301.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 15:15:00 | 301.25 | 300.43 | 301.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-09 09:15:00 | 303.05 | 300.43 | 301.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 301.05 | 300.56 | 301.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-09 09:30:00 | 303.00 | 300.56 | 301.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 10:15:00 | 299.75 | 300.40 | 301.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 13:15:00 | 297.70 | 300.23 | 301.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-10 09:15:00 | 302.20 | 299.39 | 300.26 | SL hit (close>static) qty=1.00 sl=301.70 alert=retest2 |

### Cycle 69 — BUY (started 2024-05-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 13:15:00 | 302.55 | 300.86 | 300.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-10 14:15:00 | 303.50 | 301.39 | 301.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-13 09:15:00 | 300.65 | 301.56 | 301.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-13 09:15:00 | 300.65 | 301.56 | 301.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 300.65 | 301.56 | 301.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:30:00 | 298.90 | 301.56 | 301.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 10:15:00 | 301.35 | 301.52 | 301.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 10:45:00 | 299.10 | 301.52 | 301.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 11:15:00 | 303.95 | 302.01 | 301.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 12:45:00 | 304.15 | 302.37 | 301.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 13:45:00 | 305.50 | 303.17 | 302.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 11:15:00 | 314.70 | 319.52 | 319.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 11:15:00 | 314.70 | 319.52 | 319.78 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 11:15:00 | 321.65 | 319.72 | 319.57 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 15:15:00 | 319.05 | 319.54 | 319.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 314.75 | 318.58 | 319.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 11:15:00 | 319.00 | 318.33 | 318.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 11:15:00 | 319.00 | 318.33 | 318.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 319.00 | 318.33 | 318.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 12:00:00 | 319.00 | 318.33 | 318.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 12:15:00 | 320.70 | 318.81 | 319.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 13:00:00 | 320.70 | 318.81 | 319.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 13:15:00 | 319.70 | 318.99 | 319.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 14:30:00 | 319.05 | 318.77 | 319.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 341.00 | 315.60 | 313.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 341.00 | 315.60 | 313.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 10:15:00 | 341.60 | 320.80 | 316.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 321.10 | 330.96 | 324.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 321.10 | 330.96 | 324.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 321.10 | 330.96 | 324.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 314.40 | 330.96 | 324.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 305.00 | 325.77 | 322.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 305.00 | 325.77 | 322.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 283.35 | 317.28 | 319.37 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-06-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 12:15:00 | 308.30 | 304.28 | 304.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 321.00 | 309.61 | 306.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 14:15:00 | 316.10 | 316.70 | 314.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 15:00:00 | 316.10 | 316.70 | 314.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 321.75 | 322.06 | 320.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:30:00 | 319.55 | 322.06 | 320.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 11:15:00 | 321.75 | 321.83 | 320.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 11:45:00 | 321.25 | 321.83 | 320.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 12:15:00 | 320.45 | 321.55 | 320.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 13:00:00 | 320.45 | 321.55 | 320.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 13:15:00 | 320.25 | 321.29 | 320.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 13:45:00 | 319.95 | 321.29 | 320.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 328.30 | 328.42 | 325.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 326.45 | 328.42 | 325.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 327.50 | 328.43 | 326.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 324.45 | 328.43 | 326.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 323.25 | 327.39 | 326.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:15:00 | 323.75 | 327.39 | 326.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 324.80 | 326.87 | 326.40 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2024-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 12:15:00 | 323.90 | 325.83 | 325.98 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 11:15:00 | 328.05 | 326.23 | 326.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 13:15:00 | 328.95 | 326.81 | 326.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 14:15:00 | 325.75 | 326.60 | 326.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 14:15:00 | 325.75 | 326.60 | 326.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 325.75 | 326.60 | 326.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 15:00:00 | 325.75 | 326.60 | 326.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 324.90 | 326.26 | 326.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 327.45 | 326.26 | 326.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 329.95 | 331.20 | 329.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 13:00:00 | 329.95 | 331.20 | 329.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 329.60 | 330.88 | 329.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 14:00:00 | 329.60 | 330.88 | 329.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 327.50 | 330.20 | 329.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 15:00:00 | 327.50 | 330.20 | 329.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 327.25 | 329.61 | 329.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:15:00 | 326.00 | 329.61 | 329.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 09:15:00 | 325.45 | 328.78 | 329.02 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 14:15:00 | 332.05 | 329.08 | 328.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 09:15:00 | 334.95 | 330.67 | 329.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 13:15:00 | 331.85 | 332.65 | 331.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 13:15:00 | 331.85 | 332.65 | 331.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 331.85 | 332.65 | 331.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:45:00 | 331.35 | 332.65 | 331.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 330.30 | 332.18 | 330.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 15:00:00 | 330.30 | 332.18 | 330.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 331.30 | 332.00 | 330.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:15:00 | 326.60 | 332.00 | 330.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 327.85 | 331.17 | 330.71 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 11:15:00 | 328.45 | 330.18 | 330.31 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 333.60 | 330.58 | 330.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 13:15:00 | 335.45 | 332.86 | 331.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 14:15:00 | 335.00 | 335.51 | 334.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 14:15:00 | 335.00 | 335.51 | 334.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 335.00 | 335.51 | 334.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 335.00 | 335.51 | 334.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 335.10 | 335.43 | 334.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 333.45 | 335.43 | 334.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 337.85 | 335.91 | 334.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 11:45:00 | 338.30 | 336.57 | 335.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 12:30:00 | 338.15 | 336.86 | 335.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 14:30:00 | 338.05 | 337.64 | 335.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 10:00:00 | 339.05 | 338.18 | 336.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 340.65 | 339.42 | 338.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 12:00:00 | 341.65 | 339.87 | 338.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 14:30:00 | 341.95 | 340.58 | 339.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 09:15:00 | 341.90 | 340.65 | 339.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 11:30:00 | 342.45 | 341.62 | 340.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 343.20 | 343.91 | 342.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:00:00 | 343.20 | 343.91 | 342.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 342.80 | 343.69 | 342.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:45:00 | 343.25 | 343.69 | 342.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 13:15:00 | 343.30 | 343.61 | 342.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 10:45:00 | 344.65 | 343.60 | 342.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 12:00:00 | 344.35 | 343.75 | 342.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 12:45:00 | 344.05 | 343.84 | 343.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 14:30:00 | 343.85 | 343.84 | 343.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 342.60 | 343.59 | 343.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 341.30 | 343.38 | 343.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 344.55 | 343.62 | 343.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 11:00:00 | 345.05 | 344.09 | 343.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 14:15:00 | 342.10 | 343.78 | 343.72 | SL hit (close<static) qty=1.00 sl=342.45 alert=retest2 |

### Cycle 82 — SELL (started 2024-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 15:15:00 | 342.75 | 343.57 | 343.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 338.15 | 342.41 | 343.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 13:15:00 | 341.80 | 341.71 | 342.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-18 14:00:00 | 341.80 | 341.71 | 342.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 341.40 | 341.65 | 342.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:45:00 | 342.05 | 341.65 | 342.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 336.05 | 340.42 | 341.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:30:00 | 338.75 | 340.42 | 341.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 338.65 | 335.92 | 338.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 338.65 | 335.92 | 338.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 337.70 | 336.27 | 338.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 338.60 | 336.27 | 338.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 336.95 | 336.41 | 338.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:45:00 | 337.05 | 336.41 | 338.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 338.05 | 336.70 | 337.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 338.30 | 336.70 | 337.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 338.50 | 337.06 | 337.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 338.50 | 337.06 | 337.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 338.70 | 337.39 | 338.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 336.40 | 337.39 | 338.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 334.55 | 336.82 | 337.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 10:15:00 | 334.20 | 336.82 | 337.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:00:00 | 334.20 | 336.39 | 337.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 09:15:00 | 333.20 | 334.54 | 336.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 15:15:00 | 337.05 | 336.59 | 336.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2024-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 15:15:00 | 337.05 | 336.59 | 336.57 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 09:15:00 | 333.80 | 336.03 | 336.32 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 11:15:00 | 339.85 | 337.13 | 336.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 342.65 | 339.35 | 338.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 09:15:00 | 341.05 | 342.59 | 340.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 09:15:00 | 341.05 | 342.59 | 340.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 341.05 | 342.59 | 340.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 09:15:00 | 354.15 | 341.71 | 340.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 344.85 | 353.43 | 353.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 344.85 | 353.43 | 353.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 343.70 | 350.13 | 352.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 11:15:00 | 346.95 | 346.24 | 348.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 12:00:00 | 346.95 | 346.24 | 348.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 345.85 | 344.17 | 346.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 09:45:00 | 346.10 | 344.17 | 346.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 346.70 | 344.68 | 346.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:30:00 | 348.40 | 344.68 | 346.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 346.60 | 345.06 | 346.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:00:00 | 346.60 | 345.06 | 346.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 350.15 | 346.08 | 346.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:00:00 | 350.15 | 346.08 | 346.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 351.00 | 347.06 | 347.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:00:00 | 351.00 | 347.06 | 347.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 14:15:00 | 352.40 | 348.13 | 347.78 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 10:15:00 | 345.10 | 347.72 | 347.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 12:15:00 | 342.00 | 346.39 | 347.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 09:15:00 | 345.60 | 344.76 | 345.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 09:15:00 | 345.60 | 344.76 | 345.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 345.60 | 344.76 | 345.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:30:00 | 348.15 | 344.76 | 345.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 346.15 | 345.04 | 345.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 11:00:00 | 346.15 | 345.04 | 345.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 11:15:00 | 344.55 | 344.94 | 345.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 11:30:00 | 347.00 | 344.94 | 345.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 12:15:00 | 345.60 | 345.07 | 345.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 12:45:00 | 345.70 | 345.07 | 345.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 347.05 | 345.47 | 345.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:00:00 | 347.05 | 345.47 | 345.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 346.70 | 345.71 | 346.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 15:15:00 | 345.30 | 345.71 | 346.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 09:15:00 | 341.60 | 338.04 | 337.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 341.60 | 338.04 | 337.79 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 12:15:00 | 338.95 | 339.65 | 339.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-21 14:15:00 | 336.40 | 339.02 | 339.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 11:15:00 | 336.00 | 334.94 | 336.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 11:15:00 | 336.00 | 334.94 | 336.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 336.00 | 334.94 | 336.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:30:00 | 335.95 | 334.94 | 336.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 336.70 | 335.29 | 336.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:00:00 | 336.70 | 335.29 | 336.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 336.35 | 335.50 | 336.24 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 341.15 | 336.93 | 336.74 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-08-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 14:15:00 | 335.30 | 337.69 | 337.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 14:15:00 | 334.30 | 336.55 | 337.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 15:15:00 | 335.00 | 333.42 | 334.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 15:15:00 | 335.00 | 333.42 | 334.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 335.00 | 333.42 | 334.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 335.80 | 333.42 | 334.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 337.35 | 334.21 | 334.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:45:00 | 337.00 | 334.21 | 334.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 340.00 | 335.36 | 335.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 340.00 | 335.36 | 335.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2024-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 11:15:00 | 340.00 | 336.29 | 335.83 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 15:15:00 | 335.95 | 336.58 | 336.61 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 11:15:00 | 337.00 | 336.66 | 336.64 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 13:15:00 | 335.15 | 336.39 | 336.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 14:15:00 | 334.15 | 335.95 | 336.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 12:15:00 | 334.05 | 333.12 | 334.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 12:15:00 | 334.05 | 333.12 | 334.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 12:15:00 | 334.05 | 333.12 | 334.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 12:45:00 | 333.90 | 333.12 | 334.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 13:15:00 | 333.15 | 333.12 | 333.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 13:30:00 | 333.60 | 333.12 | 333.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 330.90 | 332.68 | 333.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 15:00:00 | 330.90 | 332.68 | 333.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 327.25 | 331.21 | 332.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 11:00:00 | 325.60 | 330.09 | 332.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 13:00:00 | 326.10 | 328.69 | 331.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 324.40 | 328.95 | 330.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 11:30:00 | 326.45 | 327.73 | 329.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 328.70 | 327.29 | 328.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:45:00 | 328.25 | 327.29 | 328.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 328.10 | 327.45 | 328.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 333.20 | 327.45 | 328.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 332.10 | 328.38 | 329.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-10 11:15:00 | 333.80 | 330.21 | 329.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 11:15:00 | 333.80 | 330.21 | 329.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 12:15:00 | 334.20 | 331.01 | 330.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 334.65 | 334.90 | 333.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 14:00:00 | 334.65 | 334.90 | 333.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 333.10 | 334.54 | 333.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 333.10 | 334.54 | 333.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 333.00 | 334.23 | 333.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 336.25 | 334.23 | 333.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 13:15:00 | 334.40 | 334.84 | 333.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 13:45:00 | 334.05 | 335.14 | 334.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 11:15:00 | 335.00 | 337.29 | 337.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2024-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 11:15:00 | 335.00 | 337.29 | 337.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 13:15:00 | 334.15 | 336.23 | 336.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 336.90 | 335.76 | 336.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 336.90 | 335.76 | 336.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 336.90 | 335.76 | 336.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 336.90 | 335.76 | 336.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 336.90 | 335.99 | 336.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:45:00 | 337.20 | 335.99 | 336.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 334.70 | 335.73 | 336.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 12:15:00 | 333.10 | 335.73 | 336.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 13:00:00 | 333.80 | 335.35 | 336.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 09:15:00 | 337.15 | 335.57 | 335.88 | SL hit (close>static) qty=1.00 sl=336.95 alert=retest2 |

### Cycle 99 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 341.40 | 336.74 | 336.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 10:15:00 | 342.25 | 340.09 | 338.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 13:15:00 | 340.55 | 340.60 | 339.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 14:00:00 | 340.55 | 340.60 | 339.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 358.80 | 361.75 | 357.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:30:00 | 356.45 | 361.75 | 357.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 355.75 | 360.55 | 357.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:00:00 | 355.75 | 360.55 | 357.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 355.75 | 359.59 | 357.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:30:00 | 355.45 | 359.59 | 357.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2024-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 15:15:00 | 354.65 | 356.52 | 356.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 350.55 | 355.33 | 355.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 11:15:00 | 355.90 | 354.90 | 355.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 11:15:00 | 355.90 | 354.90 | 355.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 355.90 | 354.90 | 355.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:00:00 | 355.90 | 354.90 | 355.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 356.15 | 355.15 | 355.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:45:00 | 355.90 | 355.15 | 355.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 354.25 | 354.97 | 355.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:45:00 | 355.30 | 354.97 | 355.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 353.30 | 353.78 | 354.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 348.70 | 351.77 | 353.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 331.26 | 338.81 | 343.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-09 09:15:00 | 329.05 | 328.58 | 332.33 | SL hit (close>ema200) qty=0.50 sl=328.58 alert=retest2 |

### Cycle 101 — BUY (started 2024-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 11:15:00 | 335.00 | 332.80 | 332.58 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 11:15:00 | 331.40 | 332.92 | 332.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 13:15:00 | 330.15 | 331.99 | 332.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 14:15:00 | 330.60 | 329.89 | 330.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 14:15:00 | 330.60 | 329.89 | 330.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 330.60 | 329.89 | 330.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 14:45:00 | 330.30 | 329.89 | 330.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 15:15:00 | 330.40 | 329.99 | 330.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:15:00 | 330.45 | 329.99 | 330.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 328.80 | 329.75 | 330.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 12:15:00 | 328.15 | 329.57 | 330.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 12:30:00 | 328.30 | 329.70 | 330.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 13:15:00 | 328.25 | 329.70 | 330.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 15:00:00 | 327.10 | 329.09 | 329.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 327.25 | 328.39 | 329.24 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-17 14:15:00 | 331.50 | 329.58 | 329.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2024-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 14:15:00 | 331.50 | 329.58 | 329.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 09:15:00 | 332.30 | 330.37 | 329.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-18 15:15:00 | 331.00 | 331.64 | 330.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 15:15:00 | 331.00 | 331.64 | 330.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 331.00 | 331.64 | 330.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:30:00 | 331.00 | 331.82 | 331.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 331.25 | 331.71 | 331.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 11:00:00 | 331.25 | 331.71 | 331.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 329.90 | 331.35 | 330.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 11:45:00 | 330.00 | 331.35 | 330.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 331.80 | 331.44 | 331.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 13:45:00 | 332.80 | 331.56 | 331.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-22 09:15:00 | 333.70 | 331.38 | 331.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-22 10:15:00 | 328.00 | 330.43 | 330.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 328.00 | 330.43 | 330.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 326.05 | 329.55 | 330.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 320.00 | 319.73 | 323.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 09:45:00 | 320.40 | 319.73 | 323.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 316.55 | 319.10 | 321.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 10:15:00 | 316.35 | 319.10 | 321.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 13:45:00 | 315.95 | 316.90 | 319.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 09:30:00 | 314.90 | 317.68 | 318.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 12:15:00 | 316.35 | 317.19 | 317.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 319.00 | 317.50 | 317.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:00:00 | 319.00 | 317.50 | 317.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 321.05 | 318.21 | 318.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 15:00:00 | 321.05 | 318.21 | 318.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-29 15:15:00 | 320.20 | 318.61 | 318.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 15:15:00 | 320.20 | 318.61 | 318.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 322.90 | 319.47 | 318.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 319.10 | 320.85 | 319.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 13:15:00 | 319.10 | 320.85 | 319.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 319.10 | 320.85 | 319.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:00:00 | 319.10 | 320.85 | 319.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 318.00 | 320.28 | 319.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 15:00:00 | 318.00 | 320.28 | 319.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 319.00 | 320.03 | 319.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 318.25 | 320.03 | 319.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 321.25 | 320.27 | 319.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 10:15:00 | 321.85 | 320.27 | 319.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 10:45:00 | 322.15 | 320.35 | 319.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:15:00 | 322.25 | 320.57 | 320.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 315.80 | 319.84 | 319.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 315.80 | 319.84 | 319.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 312.65 | 318.41 | 319.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 316.05 | 315.16 | 316.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 316.05 | 315.16 | 316.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 316.05 | 315.16 | 316.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 316.05 | 315.16 | 316.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 316.35 | 315.39 | 316.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 316.35 | 315.39 | 316.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 316.65 | 315.65 | 316.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 318.25 | 315.65 | 316.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 316.80 | 315.88 | 316.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:30:00 | 318.90 | 315.88 | 316.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 315.25 | 315.75 | 316.38 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 320.20 | 317.14 | 316.93 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 09:15:00 | 314.20 | 316.99 | 316.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 11:15:00 | 312.30 | 315.54 | 316.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 11:15:00 | 314.40 | 314.18 | 315.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-08 12:00:00 | 314.40 | 314.18 | 315.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 316.45 | 314.66 | 315.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:30:00 | 317.15 | 314.66 | 315.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 314.50 | 314.63 | 315.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 322.45 | 314.63 | 315.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 09:15:00 | 324.50 | 316.60 | 315.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-11 10:15:00 | 329.75 | 319.23 | 317.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 12:15:00 | 327.50 | 327.58 | 324.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-12 13:00:00 | 327.50 | 327.58 | 324.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 322.90 | 326.64 | 323.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 14:00:00 | 322.90 | 326.64 | 323.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 322.45 | 325.80 | 323.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 14:45:00 | 322.80 | 325.80 | 323.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 15:15:00 | 322.65 | 325.17 | 323.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:15:00 | 321.45 | 325.17 | 323.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 321.50 | 323.29 | 323.03 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2024-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 11:15:00 | 317.30 | 322.09 | 322.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 10:15:00 | 316.20 | 318.95 | 320.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 315.80 | 313.20 | 315.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 09:15:00 | 315.80 | 313.20 | 315.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 315.80 | 313.20 | 315.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:30:00 | 316.05 | 313.20 | 315.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 316.70 | 313.90 | 315.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:15:00 | 317.70 | 313.90 | 315.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 09:15:00 | 319.40 | 315.99 | 315.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 12:15:00 | 322.45 | 318.75 | 317.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 11:15:00 | 338.95 | 340.00 | 335.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 12:00:00 | 338.95 | 340.00 | 335.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 339.35 | 339.33 | 336.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 13:00:00 | 340.90 | 339.69 | 337.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 09:15:00 | 336.05 | 338.70 | 337.70 | SL hit (close<static) qty=1.00 sl=336.50 alert=retest2 |

### Cycle 112 — SELL (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 11:15:00 | 334.05 | 337.03 | 337.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 09:15:00 | 329.75 | 334.18 | 335.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 13:15:00 | 330.80 | 328.80 | 329.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 13:15:00 | 330.80 | 328.80 | 329.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 13:15:00 | 330.80 | 328.80 | 329.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 13:45:00 | 331.00 | 328.80 | 329.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 14:15:00 | 329.40 | 328.92 | 329.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 10:15:00 | 328.25 | 329.20 | 329.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 14:45:00 | 325.10 | 326.09 | 326.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 15:15:00 | 328.55 | 326.09 | 326.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 09:15:00 | 330.55 | 327.37 | 327.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 09:15:00 | 330.55 | 327.37 | 327.17 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 13:15:00 | 326.05 | 328.19 | 328.46 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 13:15:00 | 329.20 | 328.21 | 328.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-12 15:15:00 | 329.75 | 328.66 | 328.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 326.70 | 328.27 | 328.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 09:15:00 | 326.70 | 328.27 | 328.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 326.70 | 328.27 | 328.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:45:00 | 325.70 | 328.27 | 328.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 327.95 | 328.21 | 328.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 11:15:00 | 328.85 | 328.21 | 328.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 12:00:00 | 329.00 | 332.04 | 331.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 13:15:00 | 329.65 | 331.13 | 331.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 329.65 | 331.13 | 331.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 09:15:00 | 323.25 | 329.23 | 330.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 12:15:00 | 323.30 | 322.97 | 325.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 13:00:00 | 323.30 | 322.97 | 325.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 323.95 | 322.55 | 324.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:45:00 | 324.50 | 322.55 | 324.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 321.20 | 322.28 | 323.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:15:00 | 320.30 | 322.28 | 323.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 15:15:00 | 310.90 | 309.23 | 309.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 15:15:00 | 310.90 | 309.23 | 309.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 10:15:00 | 311.70 | 309.89 | 309.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 313.70 | 315.18 | 313.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 313.70 | 315.18 | 313.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 313.70 | 315.18 | 313.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 311.60 | 315.18 | 313.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 309.00 | 313.94 | 313.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 309.00 | 313.94 | 313.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 308.15 | 312.79 | 312.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 306.90 | 312.79 | 312.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 308.20 | 311.87 | 312.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 306.15 | 310.72 | 311.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 09:15:00 | 306.15 | 305.99 | 307.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 13:15:00 | 307.50 | 305.83 | 307.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 307.50 | 305.83 | 307.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:00:00 | 307.50 | 305.83 | 307.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 306.80 | 306.02 | 307.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 15:15:00 | 305.80 | 306.02 | 307.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:45:00 | 306.10 | 305.85 | 306.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:15:00 | 290.51 | 295.42 | 299.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:15:00 | 290.80 | 295.42 | 299.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 292.65 | 291.89 | 296.29 | SL hit (close>ema200) qty=0.50 sl=291.89 alert=retest2 |

### Cycle 119 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 301.55 | 296.18 | 296.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 11:15:00 | 304.00 | 300.21 | 298.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 304.90 | 305.44 | 303.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 304.90 | 305.44 | 303.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 304.20 | 305.19 | 303.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:30:00 | 304.40 | 305.19 | 303.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 307.20 | 305.59 | 303.85 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 298.45 | 303.31 | 303.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 13:15:00 | 296.35 | 299.82 | 301.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 09:15:00 | 297.85 | 297.02 | 298.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 297.85 | 297.02 | 298.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 297.85 | 297.02 | 298.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 290.30 | 297.22 | 298.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 295.75 | 289.09 | 288.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 295.75 | 289.09 | 288.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 10:15:00 | 296.45 | 290.56 | 289.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 296.90 | 300.16 | 297.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 296.90 | 300.16 | 297.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 296.90 | 300.16 | 297.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 298.70 | 300.16 | 297.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 290.55 | 298.24 | 296.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 290.55 | 298.24 | 296.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 292.25 | 297.04 | 296.41 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 290.20 | 295.67 | 295.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 280.65 | 291.79 | 294.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 12:15:00 | 283.95 | 283.50 | 286.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 13:00:00 | 283.95 | 283.50 | 286.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 285.25 | 284.18 | 286.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 288.55 | 284.18 | 286.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 289.70 | 285.28 | 286.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 10:30:00 | 284.40 | 285.70 | 286.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 09:15:00 | 270.18 | 281.58 | 283.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-12 09:15:00 | 255.96 | 261.82 | 267.55 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 123 — BUY (started 2025-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 14:15:00 | 263.35 | 260.19 | 259.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 13:15:00 | 265.25 | 262.48 | 261.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 12:15:00 | 264.30 | 264.50 | 263.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-19 13:00:00 | 264.30 | 264.50 | 263.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 14:15:00 | 263.20 | 264.13 | 263.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 14:30:00 | 262.60 | 264.13 | 263.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 15:15:00 | 263.20 | 263.95 | 263.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:15:00 | 263.10 | 263.95 | 263.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 263.30 | 263.82 | 263.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 11:00:00 | 264.85 | 264.02 | 263.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 12:00:00 | 265.00 | 264.22 | 263.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 11:15:00 | 261.05 | 263.39 | 263.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 11:15:00 | 261.05 | 263.39 | 263.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 13:15:00 | 260.30 | 262.41 | 263.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 12:15:00 | 256.55 | 256.24 | 257.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 12:30:00 | 256.10 | 256.24 | 257.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 252.75 | 251.63 | 252.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:45:00 | 252.75 | 251.63 | 252.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 252.90 | 251.88 | 252.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:15:00 | 252.75 | 251.88 | 252.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 255.30 | 252.57 | 253.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 255.30 | 252.57 | 253.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 254.45 | 252.94 | 253.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:30:00 | 254.40 | 252.94 | 253.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2025-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 13:15:00 | 254.55 | 253.60 | 253.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 260.00 | 255.01 | 254.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 09:15:00 | 262.05 | 262.18 | 259.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 10:00:00 | 262.05 | 262.18 | 259.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 262.15 | 263.91 | 262.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:00:00 | 262.15 | 263.91 | 262.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 262.05 | 263.54 | 262.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:45:00 | 261.90 | 263.54 | 262.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 263.25 | 263.48 | 262.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:15:00 | 269.90 | 263.46 | 262.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 11:45:00 | 265.00 | 266.95 | 266.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 10:15:00 | 266.10 | 266.99 | 267.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 10:15:00 | 266.10 | 266.99 | 267.05 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 13:15:00 | 267.55 | 267.16 | 267.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 270.65 | 267.90 | 267.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 11:15:00 | 290.25 | 290.67 | 287.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 11:45:00 | 290.75 | 290.67 | 287.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 290.85 | 292.11 | 290.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 15:00:00 | 290.85 | 292.11 | 290.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 294.30 | 292.58 | 290.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 15:00:00 | 295.10 | 293.63 | 291.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 12:15:00 | 289.30 | 291.32 | 291.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 12:15:00 | 289.30 | 291.32 | 291.35 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 294.25 | 291.42 | 291.32 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 290.40 | 291.22 | 291.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 12:15:00 | 288.40 | 290.48 | 290.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 09:15:00 | 290.00 | 287.94 | 288.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 290.00 | 287.94 | 288.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 290.00 | 287.94 | 288.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:30:00 | 289.60 | 287.94 | 288.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 291.60 | 288.68 | 289.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:45:00 | 291.60 | 288.68 | 289.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 293.50 | 289.64 | 289.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 12:15:00 | 297.25 | 291.16 | 290.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 12:15:00 | 296.00 | 296.22 | 293.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 13:00:00 | 296.00 | 296.22 | 293.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 13:15:00 | 295.05 | 295.99 | 293.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:45:00 | 295.10 | 295.99 | 293.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 293.90 | 295.57 | 293.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 14:45:00 | 292.55 | 295.57 | 293.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 15:15:00 | 294.00 | 295.26 | 293.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-07 09:15:00 | 290.95 | 295.26 | 293.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 291.55 | 294.52 | 293.74 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2025-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 11:15:00 | 288.30 | 292.58 | 292.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 12:15:00 | 285.50 | 291.17 | 292.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 291.00 | 290.50 | 291.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 289.80 | 290.50 | 291.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 287.70 | 289.94 | 291.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 285.45 | 289.72 | 291.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:30:00 | 285.60 | 289.69 | 290.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 09:15:00 | 295.75 | 290.88 | 291.03 | SL hit (close>static) qty=1.00 sl=294.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 10:15:00 | 294.50 | 291.61 | 291.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 298.30 | 294.10 | 292.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 14:15:00 | 304.45 | 304.79 | 301.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 15:00:00 | 304.45 | 304.79 | 301.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 302.35 | 304.90 | 303.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:00:00 | 302.35 | 304.90 | 303.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 305.40 | 305.00 | 303.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 306.35 | 305.00 | 303.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 308.00 | 312.92 | 313.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 308.00 | 312.92 | 313.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 305.90 | 311.51 | 312.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 309.80 | 308.40 | 310.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 309.80 | 308.40 | 310.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 309.80 | 308.40 | 310.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 309.80 | 308.40 | 310.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 309.55 | 308.63 | 310.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:00:00 | 309.55 | 308.63 | 310.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 309.40 | 308.79 | 309.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:30:00 | 310.55 | 308.79 | 309.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 309.70 | 308.97 | 309.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 13:00:00 | 309.70 | 308.97 | 309.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 308.80 | 308.94 | 309.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 14:30:00 | 308.05 | 308.85 | 309.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 09:15:00 | 306.70 | 308.72 | 309.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 10:30:00 | 308.10 | 306.13 | 306.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 11:45:00 | 307.30 | 306.49 | 307.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 307.30 | 306.65 | 307.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 15:15:00 | 305.20 | 307.20 | 307.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 10:45:00 | 304.90 | 306.77 | 307.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 10:15:00 | 309.60 | 306.24 | 306.36 | SL hit (close>static) qty=1.00 sl=308.75 alert=retest2 |

### Cycle 135 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 309.65 | 306.93 | 306.66 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 306.20 | 307.05 | 307.06 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 14:15:00 | 307.70 | 307.18 | 307.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 09:15:00 | 310.25 | 307.83 | 307.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 11:15:00 | 310.05 | 310.94 | 309.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 11:15:00 | 310.05 | 310.94 | 309.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 310.05 | 310.94 | 309.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 12:00:00 | 310.05 | 310.94 | 309.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 309.10 | 310.57 | 309.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:00:00 | 309.10 | 310.57 | 309.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 308.40 | 310.14 | 309.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 308.40 | 310.14 | 309.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 308.95 | 309.90 | 309.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:45:00 | 307.70 | 309.90 | 309.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 305.00 | 308.92 | 309.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 299.85 | 307.11 | 308.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 308.15 | 303.03 | 304.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 308.15 | 303.03 | 304.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 308.15 | 303.03 | 304.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:45:00 | 310.30 | 303.03 | 304.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 308.00 | 304.03 | 305.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 11:15:00 | 308.70 | 304.03 | 305.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 309.20 | 306.07 | 305.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 309.90 | 307.32 | 306.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 304.10 | 306.68 | 306.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 304.10 | 306.68 | 306.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 304.10 | 306.68 | 306.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 09:45:00 | 303.25 | 306.68 | 306.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 10:15:00 | 302.90 | 305.92 | 306.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 12:15:00 | 299.40 | 304.00 | 305.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 12:15:00 | 295.25 | 294.68 | 297.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-15 13:00:00 | 295.25 | 294.68 | 297.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 299.30 | 295.61 | 297.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:00:00 | 299.30 | 295.61 | 297.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 299.45 | 296.37 | 298.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 298.25 | 296.99 | 298.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 12:15:00 | 298.35 | 298.23 | 298.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 13:15:00 | 299.80 | 298.65 | 298.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 13:15:00 | 299.80 | 298.65 | 298.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 14:15:00 | 300.30 | 298.98 | 298.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 302.45 | 302.45 | 301.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 302.45 | 302.45 | 301.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 302.45 | 302.45 | 301.10 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 298.10 | 300.76 | 300.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 14:15:00 | 296.05 | 298.68 | 299.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 15:15:00 | 292.00 | 291.86 | 294.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 09:15:00 | 294.10 | 291.86 | 294.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 295.30 | 292.55 | 294.80 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 298.05 | 296.19 | 295.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 301.40 | 297.53 | 296.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 13:15:00 | 297.20 | 298.43 | 297.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 13:15:00 | 297.20 | 298.43 | 297.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 297.20 | 298.43 | 297.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:45:00 | 297.40 | 298.43 | 297.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 296.75 | 298.10 | 297.41 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 09:15:00 | 293.25 | 296.93 | 296.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 292.30 | 293.74 | 294.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 292.95 | 292.84 | 294.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 10:00:00 | 292.95 | 292.84 | 294.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 293.20 | 292.78 | 293.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 293.20 | 292.78 | 293.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 291.40 | 292.49 | 293.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:30:00 | 292.55 | 292.49 | 293.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 289.60 | 291.69 | 292.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:45:00 | 293.00 | 291.69 | 292.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 292.15 | 291.42 | 292.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:30:00 | 292.00 | 291.42 | 292.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 292.05 | 291.55 | 292.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:30:00 | 292.25 | 291.55 | 292.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 292.50 | 291.74 | 292.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:30:00 | 292.85 | 291.74 | 292.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 292.15 | 291.82 | 292.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:30:00 | 291.40 | 292.14 | 292.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 10:30:00 | 290.90 | 291.69 | 292.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 10:15:00 | 293.80 | 290.56 | 290.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 293.80 | 290.56 | 290.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 11:15:00 | 295.00 | 291.44 | 290.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 12:15:00 | 294.15 | 294.23 | 292.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 13:00:00 | 294.15 | 294.23 | 292.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 298.00 | 299.53 | 298.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 298.00 | 299.53 | 298.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 295.90 | 298.81 | 298.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:00:00 | 295.90 | 298.81 | 298.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 295.50 | 297.55 | 297.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 293.10 | 296.33 | 297.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 288.80 | 287.21 | 289.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 288.80 | 287.21 | 289.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 288.80 | 287.21 | 289.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:45:00 | 285.85 | 287.46 | 288.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:15:00 | 285.85 | 287.32 | 287.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:30:00 | 286.25 | 287.05 | 287.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:15:00 | 286.20 | 286.94 | 287.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 286.50 | 286.65 | 287.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 290.45 | 287.31 | 287.48 | SL hit (close>static) qty=1.00 sl=290.30 alert=retest2 |

### Cycle 147 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 291.70 | 288.19 | 287.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 13:15:00 | 292.65 | 290.16 | 288.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 289.15 | 290.73 | 289.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 289.15 | 290.73 | 289.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 289.15 | 290.73 | 289.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:45:00 | 289.30 | 290.73 | 289.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 290.35 | 290.65 | 289.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:15:00 | 290.85 | 290.65 | 289.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 12:15:00 | 291.05 | 290.61 | 289.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 292.05 | 290.97 | 290.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 10:45:00 | 290.55 | 291.11 | 290.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 289.60 | 290.81 | 290.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:00:00 | 289.60 | 290.81 | 290.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 12:15:00 | 288.40 | 290.33 | 290.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 288.40 | 290.33 | 290.18 | SL hit (close<static) qty=1.00 sl=289.15 alert=retest2 |

### Cycle 148 — SELL (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 13:15:00 | 286.55 | 289.57 | 289.85 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 14:15:00 | 290.35 | 289.42 | 289.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 15:15:00 | 290.80 | 289.69 | 289.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 12:15:00 | 289.90 | 289.95 | 289.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 13:00:00 | 289.90 | 289.95 | 289.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 292.65 | 290.49 | 289.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 15:00:00 | 293.95 | 291.18 | 290.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 13:15:00 | 295.40 | 296.81 | 296.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 295.40 | 296.81 | 296.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 14:15:00 | 293.85 | 294.82 | 295.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 12:15:00 | 294.40 | 294.26 | 294.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 12:15:00 | 294.40 | 294.26 | 294.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 294.40 | 294.26 | 294.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:00:00 | 294.40 | 294.26 | 294.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 295.15 | 294.47 | 294.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:45:00 | 295.65 | 294.47 | 294.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 296.00 | 294.78 | 294.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 296.00 | 294.78 | 294.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 295.90 | 295.00 | 295.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 12:15:00 | 295.50 | 295.00 | 295.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 14:15:00 | 296.00 | 295.12 | 295.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 296.00 | 295.12 | 295.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 09:15:00 | 296.75 | 295.50 | 295.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 12:15:00 | 298.75 | 299.38 | 298.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 13:00:00 | 298.75 | 299.38 | 298.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 298.30 | 299.16 | 298.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 298.30 | 299.16 | 298.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 299.45 | 299.22 | 298.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 300.65 | 299.17 | 298.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 297.40 | 298.79 | 298.46 | SL hit (close<static) qty=1.00 sl=298.20 alert=retest2 |

### Cycle 152 — SELL (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 15:15:00 | 297.85 | 298.47 | 298.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 13:15:00 | 297.40 | 298.31 | 298.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 09:15:00 | 297.00 | 296.94 | 297.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 09:15:00 | 297.00 | 296.94 | 297.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 297.00 | 296.94 | 297.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:30:00 | 298.80 | 296.94 | 297.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 295.30 | 295.19 | 296.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:45:00 | 295.10 | 295.19 | 296.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 296.30 | 295.41 | 296.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 296.30 | 295.41 | 296.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 297.00 | 295.73 | 296.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:45:00 | 296.80 | 295.73 | 296.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 296.40 | 295.86 | 296.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 295.25 | 296.33 | 296.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 11:15:00 | 297.15 | 296.35 | 296.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 11:15:00 | 297.15 | 296.35 | 296.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 12:15:00 | 297.75 | 296.63 | 296.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 15:15:00 | 297.00 | 297.12 | 296.77 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 10:15:00 | 298.70 | 297.34 | 296.90 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 10:45:00 | 298.80 | 297.71 | 297.11 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 14:00:00 | 298.55 | 298.17 | 297.50 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 297.55 | 298.51 | 297.98 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-24 11:15:00 | 297.55 | 298.51 | 297.98 | SL hit (close<ema400) qty=1.00 sl=297.98 alert=retest1 |

### Cycle 154 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 294.35 | 297.90 | 297.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 294.00 | 297.12 | 297.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 10:15:00 | 294.40 | 293.97 | 295.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 11:00:00 | 294.40 | 293.97 | 295.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 293.05 | 292.76 | 293.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:30:00 | 293.50 | 292.76 | 293.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 293.00 | 292.86 | 293.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 293.45 | 292.86 | 293.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 294.75 | 293.24 | 293.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:00:00 | 294.75 | 293.24 | 293.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 295.60 | 293.71 | 293.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:00:00 | 295.60 | 293.71 | 293.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 295.50 | 294.07 | 294.00 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 14:15:00 | 289.35 | 293.42 | 293.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 286.50 | 288.40 | 289.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 286.00 | 285.20 | 286.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 286.00 | 285.20 | 286.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 286.00 | 285.20 | 286.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 286.00 | 285.20 | 286.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 285.85 | 285.33 | 286.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 285.70 | 285.33 | 286.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 284.50 | 285.17 | 285.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:30:00 | 283.85 | 284.71 | 285.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 286.80 | 285.20 | 285.22 | SL hit (close>static) qty=1.00 sl=286.25 alert=retest2 |

### Cycle 157 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 286.05 | 285.37 | 285.29 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 284.90 | 285.27 | 285.29 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 287.15 | 285.65 | 285.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 10:15:00 | 288.50 | 286.22 | 285.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 15:15:00 | 288.65 | 288.74 | 287.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 09:15:00 | 289.05 | 288.74 | 287.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 289.50 | 290.01 | 289.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 288.95 | 290.01 | 289.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 289.15 | 289.84 | 289.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 289.15 | 289.84 | 289.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 287.80 | 289.43 | 289.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:00:00 | 287.80 | 289.43 | 289.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 288.00 | 289.14 | 288.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:45:00 | 288.15 | 289.14 | 288.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 13:15:00 | 287.30 | 288.78 | 288.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 10:15:00 | 286.30 | 287.69 | 288.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 13:15:00 | 283.95 | 283.86 | 285.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-22 14:00:00 | 283.95 | 283.86 | 285.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 284.15 | 283.70 | 284.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:30:00 | 284.50 | 283.70 | 284.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 284.15 | 283.79 | 284.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:45:00 | 284.45 | 283.79 | 284.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 278.70 | 276.23 | 277.42 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 280.00 | 278.11 | 278.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 285.75 | 280.31 | 279.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 14:15:00 | 286.00 | 286.05 | 284.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 14:30:00 | 286.00 | 286.05 | 284.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 284.30 | 285.66 | 284.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 284.30 | 285.66 | 284.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 284.75 | 285.47 | 284.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:15:00 | 284.15 | 285.47 | 284.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 283.25 | 285.03 | 284.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 283.25 | 285.03 | 284.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 282.20 | 284.46 | 284.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 282.20 | 284.46 | 284.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 281.55 | 283.40 | 283.65 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 11:15:00 | 284.75 | 283.82 | 283.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 15:15:00 | 285.55 | 284.64 | 284.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 10:15:00 | 284.20 | 284.71 | 284.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 10:15:00 | 284.20 | 284.71 | 284.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 284.20 | 284.71 | 284.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 284.20 | 284.71 | 284.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 284.00 | 284.57 | 284.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:00:00 | 284.00 | 284.57 | 284.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 283.90 | 284.44 | 284.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:30:00 | 284.30 | 284.44 | 284.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 284.60 | 284.47 | 284.29 | EMA400 retest candle locked (from upside) |

### Cycle 164 — SELL (started 2025-09-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 14:15:00 | 282.75 | 284.12 | 284.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 09:15:00 | 282.15 | 283.53 | 283.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 10:15:00 | 283.75 | 283.57 | 283.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 11:00:00 | 283.75 | 283.57 | 283.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 283.90 | 283.64 | 283.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:45:00 | 284.05 | 283.64 | 283.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 282.90 | 283.49 | 283.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:15:00 | 282.70 | 283.49 | 283.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 14:15:00 | 284.00 | 283.58 | 283.76 | SL hit (close>static) qty=1.00 sl=283.90 alert=retest2 |

### Cycle 165 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 285.85 | 284.10 | 283.97 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 14:15:00 | 283.00 | 283.77 | 283.88 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 284.90 | 284.10 | 284.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 11:15:00 | 286.60 | 284.60 | 284.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 11:15:00 | 285.85 | 286.23 | 285.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 12:00:00 | 285.85 | 286.23 | 285.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 285.75 | 286.13 | 285.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:30:00 | 285.45 | 286.13 | 285.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 286.75 | 286.97 | 286.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:45:00 | 287.10 | 286.97 | 286.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 285.90 | 286.76 | 286.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 285.90 | 286.76 | 286.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 286.10 | 286.63 | 286.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 287.30 | 286.64 | 286.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:30:00 | 287.35 | 287.61 | 287.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 11:30:00 | 287.10 | 287.51 | 287.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 286.20 | 287.06 | 287.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 286.20 | 287.06 | 287.08 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 14:15:00 | 287.35 | 287.12 | 287.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 15:15:00 | 287.50 | 287.20 | 287.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 11:15:00 | 287.15 | 287.33 | 287.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 11:15:00 | 287.15 | 287.33 | 287.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 287.15 | 287.33 | 287.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 287.15 | 287.33 | 287.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 286.55 | 287.18 | 287.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:30:00 | 286.35 | 287.18 | 287.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 287.20 | 287.18 | 287.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:30:00 | 287.85 | 287.47 | 287.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 13:15:00 | 286.95 | 287.32 | 287.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 13:15:00 | 286.95 | 287.32 | 287.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 286.10 | 287.07 | 287.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 10:15:00 | 287.05 | 286.92 | 287.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 10:15:00 | 287.05 | 286.92 | 287.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 287.05 | 286.92 | 287.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 287.05 | 286.92 | 287.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 286.70 | 286.88 | 287.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:30:00 | 287.15 | 286.88 | 287.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 286.95 | 286.89 | 287.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:30:00 | 287.15 | 286.89 | 287.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 286.95 | 286.90 | 287.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:45:00 | 287.00 | 286.90 | 287.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 286.35 | 286.79 | 286.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 15:00:00 | 286.35 | 286.79 | 286.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 287.90 | 286.94 | 287.01 | EMA400 retest candle locked (from downside) |

### Cycle 171 — BUY (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 11:15:00 | 287.85 | 287.18 | 287.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 14:15:00 | 288.80 | 287.74 | 287.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 11:15:00 | 290.90 | 291.88 | 290.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 11:15:00 | 290.90 | 291.88 | 290.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 290.90 | 291.88 | 290.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:00:00 | 290.90 | 291.88 | 290.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 288.15 | 291.14 | 290.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:00:00 | 288.15 | 291.14 | 290.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 286.00 | 290.11 | 289.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:45:00 | 286.00 | 290.11 | 289.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — SELL (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 14:15:00 | 284.00 | 288.89 | 289.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 281.25 | 286.61 | 288.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 285.00 | 283.60 | 285.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 285.00 | 283.60 | 285.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 285.65 | 284.01 | 285.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 285.65 | 284.01 | 285.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 282.80 | 283.77 | 285.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:30:00 | 284.60 | 283.77 | 285.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 283.40 | 282.48 | 283.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:45:00 | 281.70 | 282.42 | 283.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:15:00 | 281.70 | 282.42 | 283.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:45:00 | 280.80 | 282.15 | 283.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:00:00 | 281.70 | 281.33 | 282.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 283.90 | 281.62 | 282.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:45:00 | 283.75 | 281.62 | 282.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 283.90 | 282.07 | 282.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:30:00 | 283.70 | 282.07 | 282.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 284.95 | 282.65 | 282.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 284.95 | 282.65 | 282.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 286.85 | 283.49 | 282.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 285.15 | 285.67 | 284.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:00:00 | 285.15 | 285.67 | 284.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 285.80 | 288.36 | 287.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 285.80 | 288.36 | 287.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 285.55 | 287.80 | 287.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:45:00 | 285.25 | 287.80 | 287.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 13:15:00 | 284.70 | 286.60 | 286.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 283.45 | 285.54 | 286.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 285.55 | 285.54 | 286.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:00:00 | 285.55 | 285.54 | 286.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 286.00 | 285.64 | 286.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:30:00 | 286.65 | 285.64 | 286.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 286.10 | 285.73 | 286.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 14:00:00 | 285.40 | 285.66 | 286.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 290.20 | 286.72 | 286.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 290.20 | 286.72 | 286.42 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 285.60 | 286.95 | 287.10 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 290.45 | 287.51 | 287.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 10:15:00 | 291.50 | 288.31 | 287.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 289.85 | 291.22 | 290.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 289.85 | 291.22 | 290.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 289.85 | 291.22 | 290.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:30:00 | 290.25 | 291.22 | 290.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 290.50 | 291.08 | 290.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 289.55 | 291.08 | 290.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 289.75 | 290.81 | 290.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:00:00 | 289.75 | 290.81 | 290.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 288.10 | 290.27 | 290.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 288.10 | 290.27 | 290.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 288.85 | 289.99 | 290.02 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 290.85 | 289.99 | 289.99 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 288.70 | 289.73 | 289.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 12:15:00 | 288.50 | 289.49 | 289.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 290.05 | 288.98 | 289.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 290.05 | 288.98 | 289.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 290.05 | 288.98 | 289.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 290.05 | 288.98 | 289.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 289.95 | 289.18 | 289.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:30:00 | 290.60 | 289.18 | 289.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — BUY (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 13:15:00 | 289.65 | 289.47 | 289.45 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 289.15 | 289.39 | 289.41 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 290.30 | 289.57 | 289.50 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 288.10 | 289.33 | 289.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 287.60 | 288.98 | 289.25 | Break + close below crossover candle low |

### Cycle 185 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 292.25 | 289.52 | 289.41 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 287.90 | 289.59 | 289.82 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 293.80 | 290.62 | 290.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 11:15:00 | 295.00 | 291.50 | 290.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 292.40 | 293.59 | 292.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 292.40 | 293.59 | 292.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 292.40 | 293.59 | 292.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 292.40 | 293.59 | 292.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 292.05 | 293.28 | 292.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 291.15 | 293.28 | 292.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 292.85 | 293.20 | 292.25 | EMA400 retest candle locked (from upside) |

### Cycle 188 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 289.65 | 291.48 | 291.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 289.30 | 291.04 | 291.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 12:15:00 | 290.05 | 289.38 | 290.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 13:00:00 | 290.05 | 289.38 | 290.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 288.00 | 289.10 | 289.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 279.25 | 288.71 | 289.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 265.29 | 268.34 | 271.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 268.15 | 267.83 | 269.70 | SL hit (close>ema200) qty=0.50 sl=267.83 alert=retest2 |

### Cycle 189 — BUY (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 15:15:00 | 270.05 | 269.25 | 269.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 10:15:00 | 270.85 | 269.66 | 269.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 14:15:00 | 274.20 | 274.29 | 272.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 15:00:00 | 274.20 | 274.29 | 272.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 273.40 | 274.09 | 273.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 09:15:00 | 277.25 | 274.79 | 273.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 09:30:00 | 276.05 | 276.79 | 275.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 10:15:00 | 276.65 | 276.79 | 275.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 12:15:00 | 274.45 | 276.03 | 276.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — SELL (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 12:15:00 | 274.45 | 276.03 | 276.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 13:15:00 | 273.55 | 275.53 | 275.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 276.50 | 275.73 | 275.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 276.50 | 275.73 | 275.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 276.50 | 275.73 | 275.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 276.50 | 275.73 | 275.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 275.50 | 275.68 | 275.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 274.00 | 275.68 | 275.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:00:00 | 274.80 | 275.50 | 275.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 13:15:00 | 275.25 | 275.25 | 275.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 11:00:00 | 274.85 | 274.80 | 275.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 275.05 | 274.85 | 275.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:30:00 | 275.50 | 274.85 | 275.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 274.95 | 274.87 | 275.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:00:00 | 274.95 | 274.87 | 275.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 275.00 | 274.90 | 275.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 14:00:00 | 275.00 | 274.90 | 275.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 275.20 | 274.96 | 275.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:00:00 | 275.20 | 274.96 | 275.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 274.40 | 274.85 | 275.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 275.55 | 274.85 | 275.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 273.75 | 274.63 | 274.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 272.35 | 274.23 | 274.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 15:15:00 | 269.10 | 268.79 | 268.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — BUY (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 15:15:00 | 269.10 | 268.79 | 268.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 269.70 | 268.97 | 268.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 267.75 | 269.27 | 269.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 267.75 | 269.27 | 269.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 267.75 | 269.27 | 269.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 267.75 | 269.27 | 269.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 267.40 | 268.90 | 269.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 266.10 | 268.34 | 268.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 266.45 | 265.29 | 266.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 266.45 | 265.29 | 266.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 266.45 | 265.29 | 266.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:00:00 | 266.45 | 265.29 | 266.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 265.25 | 265.29 | 266.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 263.25 | 265.58 | 265.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:45:00 | 264.50 | 265.41 | 265.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 264.10 | 265.12 | 265.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 262.90 | 260.58 | 260.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 262.90 | 260.58 | 260.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 263.40 | 261.14 | 260.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 14:15:00 | 267.95 | 267.95 | 266.58 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 09:15:00 | 268.85 | 268.02 | 266.73 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 10:45:00 | 268.75 | 268.12 | 267.00 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 11:15:00 | 268.85 | 268.12 | 267.00 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 11:45:00 | 268.75 | 268.22 | 267.15 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 266.80 | 267.82 | 267.15 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-26 13:15:00 | 266.80 | 267.82 | 267.15 | SL hit (close<ema400) qty=1.00 sl=267.15 alert=retest1 |

### Cycle 194 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 262.80 | 266.13 | 266.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 262.70 | 265.44 | 266.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 263.80 | 261.26 | 262.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 263.80 | 261.26 | 262.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 263.80 | 261.26 | 262.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 263.80 | 261.26 | 262.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 264.50 | 261.91 | 262.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:30:00 | 264.60 | 261.91 | 262.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 264.30 | 263.12 | 263.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 264.90 | 263.48 | 263.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 270.25 | 270.65 | 269.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 270.25 | 270.65 | 269.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 270.25 | 270.65 | 269.27 | EMA400 retest candle locked (from upside) |

### Cycle 196 — SELL (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 09:15:00 | 266.65 | 268.40 | 268.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 11:15:00 | 264.80 | 267.33 | 268.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 261.50 | 261.05 | 263.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 261.50 | 261.05 | 263.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 261.50 | 261.05 | 263.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 12:00:00 | 259.70 | 260.87 | 262.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 260.50 | 258.84 | 258.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 260.50 | 258.84 | 258.81 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 256.80 | 258.60 | 258.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 256.05 | 257.47 | 257.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 256.75 | 255.79 | 256.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 256.75 | 255.79 | 256.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 256.75 | 255.79 | 256.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:45:00 | 256.50 | 255.79 | 256.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 256.15 | 255.86 | 256.64 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 259.40 | 257.16 | 256.91 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 254.60 | 256.81 | 256.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 253.75 | 255.77 | 256.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 11:15:00 | 254.75 | 254.68 | 255.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 11:30:00 | 255.15 | 254.68 | 255.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 255.00 | 254.28 | 254.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 257.50 | 254.28 | 254.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 257.65 | 254.95 | 255.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:00:00 | 257.65 | 254.95 | 255.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 259.40 | 255.84 | 255.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 10:15:00 | 260.85 | 258.83 | 257.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 10:15:00 | 256.95 | 259.42 | 258.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 10:15:00 | 256.95 | 259.42 | 258.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 256.95 | 259.42 | 258.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:00:00 | 256.95 | 259.42 | 258.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 256.10 | 258.76 | 258.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 256.10 | 258.76 | 258.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — SELL (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 12:15:00 | 254.50 | 257.91 | 257.99 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 262.00 | 257.86 | 257.83 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 254.60 | 257.66 | 257.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 252.20 | 256.57 | 257.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 253.95 | 253.90 | 255.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:45:00 | 254.85 | 253.90 | 255.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 256.25 | 254.17 | 255.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:00:00 | 256.25 | 254.17 | 255.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 262.25 | 255.78 | 256.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 262.25 | 255.78 | 256.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — BUY (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 13:15:00 | 266.90 | 258.01 | 257.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 14:15:00 | 270.25 | 260.46 | 258.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 287.10 | 287.16 | 280.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 11:00:00 | 287.10 | 287.16 | 280.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 288.45 | 290.29 | 288.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:00:00 | 288.45 | 290.29 | 288.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 289.35 | 290.10 | 288.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 14:15:00 | 289.65 | 290.10 | 288.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 15:00:00 | 289.50 | 289.98 | 288.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 291.30 | 289.82 | 288.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 289.85 | 292.32 | 292.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 289.85 | 292.32 | 292.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 11:15:00 | 289.35 | 291.72 | 292.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 294.40 | 290.33 | 291.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 294.40 | 290.33 | 291.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 294.40 | 290.33 | 291.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 294.40 | 290.33 | 291.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 298.80 | 292.03 | 291.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 12:15:00 | 299.75 | 294.53 | 292.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 11:15:00 | 297.50 | 297.78 | 295.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-17 11:30:00 | 297.40 | 297.78 | 295.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 298.20 | 299.80 | 298.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 298.20 | 299.80 | 298.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 297.55 | 299.35 | 298.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:45:00 | 298.15 | 299.35 | 298.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 295.45 | 297.68 | 297.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 293.55 | 296.43 | 297.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 298.25 | 296.50 | 297.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 10:15:00 | 298.25 | 296.50 | 297.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 298.25 | 296.50 | 297.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:00:00 | 298.25 | 296.50 | 297.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 300.10 | 297.22 | 297.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 300.10 | 297.22 | 297.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 298.65 | 297.68 | 297.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 300.85 | 298.68 | 298.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 305.20 | 305.46 | 303.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 13:00:00 | 305.20 | 305.46 | 303.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 304.90 | 305.62 | 304.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 304.90 | 305.62 | 304.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 302.10 | 304.92 | 304.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:00:00 | 302.10 | 304.92 | 304.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 302.35 | 304.40 | 304.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 13:00:00 | 302.35 | 304.40 | 304.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — SELL (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 14:15:00 | 302.35 | 303.67 | 303.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 11:15:00 | 301.50 | 303.00 | 303.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 13:15:00 | 294.25 | 294.03 | 296.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 14:00:00 | 294.25 | 294.03 | 296.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 296.35 | 293.63 | 295.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:00:00 | 296.35 | 293.63 | 295.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 297.40 | 294.39 | 295.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:30:00 | 298.65 | 294.39 | 295.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 297.15 | 295.45 | 296.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:30:00 | 298.55 | 295.45 | 296.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 298.90 | 296.38 | 296.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 298.90 | 296.38 | 296.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 299.00 | 296.91 | 296.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 301.15 | 297.76 | 297.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 299.15 | 299.76 | 298.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-06 14:45:00 | 299.00 | 299.76 | 298.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 300.00 | 299.81 | 298.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 291.80 | 299.81 | 298.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 289.75 | 297.80 | 297.86 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 299.30 | 296.64 | 296.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 10:15:00 | 300.05 | 298.52 | 297.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 298.50 | 299.43 | 298.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 298.50 | 299.43 | 298.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 298.50 | 299.43 | 298.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 298.50 | 299.43 | 298.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 298.70 | 299.28 | 298.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 297.55 | 299.28 | 298.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 300.00 | 299.43 | 298.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 301.80 | 299.43 | 298.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 13:15:00 | 300.50 | 302.38 | 301.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 14:00:00 | 300.65 | 302.04 | 301.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 14:30:00 | 301.20 | 301.83 | 301.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 301.00 | 301.66 | 301.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:15:00 | 299.25 | 301.66 | 301.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-16 09:15:00 | 297.15 | 300.76 | 300.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 214 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 297.15 | 300.76 | 300.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 291.90 | 298.99 | 299.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 297.90 | 297.48 | 298.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 297.90 | 297.48 | 298.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 297.90 | 297.48 | 298.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 299.30 | 297.48 | 298.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 296.90 | 297.38 | 298.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 294.70 | 297.37 | 298.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 302.25 | 298.09 | 298.19 | SL hit (close>static) qty=1.00 sl=302.00 alert=retest2 |

### Cycle 215 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 302.80 | 299.04 | 298.61 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 298.20 | 298.67 | 298.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 296.50 | 298.24 | 298.52 | Break + close below crossover candle low |

### Cycle 217 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 304.10 | 299.30 | 298.95 | EMA200 above EMA400 |

### Cycle 218 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 294.10 | 298.69 | 299.12 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-23 14:15:00 | 301.55 | 299.45 | 299.30 | EMA200 above EMA400 |

### Cycle 220 — SELL (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 09:15:00 | 294.00 | 298.45 | 298.88 | EMA200 below EMA400 |

### Cycle 221 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 300.15 | 298.98 | 298.90 | EMA200 above EMA400 |

### Cycle 222 — SELL (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 13:15:00 | 297.05 | 298.69 | 298.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-25 14:15:00 | 294.95 | 297.94 | 298.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 12:15:00 | 296.25 | 295.73 | 296.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 12:15:00 | 296.25 | 295.73 | 296.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 296.25 | 295.73 | 296.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:45:00 | 297.15 | 295.73 | 296.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 296.55 | 295.89 | 296.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:30:00 | 296.60 | 295.89 | 296.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 295.50 | 295.82 | 296.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:45:00 | 295.70 | 295.82 | 296.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 295.50 | 295.73 | 296.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:30:00 | 297.00 | 295.73 | 296.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 297.15 | 296.00 | 296.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 12:00:00 | 297.15 | 296.00 | 296.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 12:15:00 | 296.30 | 296.06 | 296.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:00:00 | 294.85 | 295.84 | 296.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 15:15:00 | 295.00 | 292.09 | 291.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 223 — BUY (started 2026-04-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 15:15:00 | 295.00 | 292.09 | 291.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 296.50 | 294.52 | 293.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 12:15:00 | 294.75 | 295.32 | 294.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-08 12:45:00 | 294.85 | 295.32 | 294.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 13:15:00 | 294.50 | 295.15 | 294.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 13:45:00 | 294.30 | 295.15 | 294.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 294.80 | 295.08 | 294.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 09:15:00 | 297.30 | 295.07 | 294.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 316.50 | 318.90 | 319.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 224 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 316.50 | 318.90 | 319.00 | EMA200 below EMA400 |

### Cycle 225 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 323.10 | 318.37 | 318.27 | EMA200 above EMA400 |

### Cycle 226 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 316.30 | 319.22 | 319.57 | EMA200 below EMA400 |

### Cycle 227 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 321.70 | 319.27 | 319.23 | EMA200 above EMA400 |

### Cycle 228 — SELL (started 2026-05-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 13:15:00 | 318.00 | 319.15 | 319.21 | EMA200 below EMA400 |

### Cycle 229 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 319.65 | 319.30 | 319.27 | EMA200 above EMA400 |

### Cycle 230 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 318.30 | 319.10 | 319.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 10:15:00 | 316.45 | 318.57 | 318.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 12:15:00 | 320.50 | 318.55 | 318.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 12:15:00 | 320.50 | 318.55 | 318.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 320.50 | 318.55 | 318.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:00:00 | 320.50 | 318.55 | 318.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 320.70 | 318.98 | 319.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:00:00 | 320.70 | 318.98 | 319.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 231 — BUY (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 14:15:00 | 319.70 | 319.13 | 319.07 | EMA200 above EMA400 |

### Cycle 232 — SELL (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 15:15:00 | 318.50 | 319.00 | 319.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 09:15:00 | 316.90 | 318.58 | 318.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 14:15:00 | 313.90 | 313.90 | 314.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 14:15:00 | 313.90 | 313.90 | 314.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 313.90 | 313.90 | 314.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:30:00 | 315.05 | 313.90 | 314.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-22 11:45:00 | 176.21 | 2023-05-24 11:15:00 | 177.45 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2023-05-23 10:15:00 | 176.44 | 2023-05-24 11:15:00 | 177.45 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2023-05-26 12:30:00 | 177.45 | 2023-05-29 13:15:00 | 176.25 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2023-05-26 13:15:00 | 177.71 | 2023-05-29 13:15:00 | 176.25 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2023-05-29 10:30:00 | 177.75 | 2023-05-29 13:15:00 | 176.25 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2023-06-06 09:30:00 | 177.53 | 2023-06-19 11:15:00 | 182.51 | STOP_HIT | 1.00 | 2.81% |
| BUY | retest2 | 2023-06-06 15:15:00 | 177.04 | 2023-06-19 11:15:00 | 182.51 | STOP_HIT | 1.00 | 3.09% |
| BUY | retest2 | 2023-06-23 11:15:00 | 187.91 | 2023-06-23 13:15:00 | 187.54 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2023-06-30 09:15:00 | 191.55 | 2023-07-03 14:15:00 | 188.06 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2023-06-30 11:30:00 | 189.26 | 2023-07-03 14:15:00 | 188.06 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2023-07-11 15:15:00 | 189.60 | 2023-07-12 12:15:00 | 190.50 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2023-07-17 10:30:00 | 181.05 | 2023-07-19 09:15:00 | 183.75 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2023-07-17 11:15:00 | 180.75 | 2023-07-19 09:15:00 | 183.75 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2023-07-18 11:30:00 | 180.98 | 2023-07-19 09:15:00 | 183.75 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2023-07-18 12:45:00 | 181.05 | 2023-07-19 09:15:00 | 183.75 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2023-07-24 13:00:00 | 184.43 | 2023-08-01 12:15:00 | 188.36 | STOP_HIT | 1.00 | 2.13% |
| SELL | retest2 | 2023-08-08 09:15:00 | 183.11 | 2023-08-11 11:15:00 | 182.96 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2023-09-06 14:15:00 | 190.05 | 2023-09-12 14:15:00 | 191.00 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2023-09-07 10:30:00 | 190.09 | 2023-09-12 14:15:00 | 191.00 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2023-09-07 12:30:00 | 189.94 | 2023-09-12 14:15:00 | 191.00 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2023-10-05 10:15:00 | 195.80 | 2023-10-10 11:15:00 | 198.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2023-10-05 12:30:00 | 196.00 | 2023-10-10 12:15:00 | 198.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2023-10-05 13:00:00 | 195.95 | 2023-10-10 12:15:00 | 198.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2023-10-05 14:15:00 | 196.00 | 2023-10-10 12:15:00 | 198.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2023-10-06 15:15:00 | 196.95 | 2023-10-10 12:15:00 | 198.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2023-10-16 10:15:00 | 201.40 | 2023-10-20 09:15:00 | 201.70 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2023-11-02 09:15:00 | 201.60 | 2023-11-15 15:15:00 | 210.00 | STOP_HIT | 1.00 | 4.17% |
| BUY | retest2 | 2023-11-02 12:00:00 | 201.20 | 2023-11-15 15:15:00 | 210.00 | STOP_HIT | 1.00 | 4.37% |
| BUY | retest2 | 2023-12-11 10:15:00 | 230.85 | 2023-12-18 13:15:00 | 231.60 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2024-01-01 09:15:00 | 238.40 | 2024-01-10 13:15:00 | 239.40 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2024-01-01 10:45:00 | 238.45 | 2024-01-10 13:15:00 | 239.40 | STOP_HIT | 1.00 | 0.40% |
| BUY | retest2 | 2024-01-01 11:15:00 | 238.35 | 2024-01-10 13:15:00 | 239.40 | STOP_HIT | 1.00 | 0.44% |
| BUY | retest2 | 2024-01-01 11:45:00 | 238.50 | 2024-01-10 13:15:00 | 239.40 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2024-01-02 09:15:00 | 239.00 | 2024-01-10 13:15:00 | 239.40 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2024-01-02 10:00:00 | 238.25 | 2024-01-10 13:15:00 | 239.40 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2024-01-02 12:30:00 | 238.40 | 2024-01-10 13:15:00 | 239.40 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2024-01-02 14:00:00 | 238.55 | 2024-01-10 13:15:00 | 239.40 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2024-01-03 11:45:00 | 238.25 | 2024-01-10 13:15:00 | 239.40 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2024-01-03 13:15:00 | 238.35 | 2024-01-10 13:15:00 | 239.40 | STOP_HIT | 1.00 | 0.44% |
| BUY | retest2 | 2024-01-04 09:15:00 | 240.95 | 2024-01-10 13:15:00 | 239.40 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-02-13 09:15:00 | 265.25 | 2024-02-14 15:15:00 | 274.05 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2024-02-13 11:15:00 | 268.75 | 2024-02-14 15:15:00 | 274.05 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2024-02-13 12:00:00 | 267.55 | 2024-02-14 15:15:00 | 274.05 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-02-14 09:15:00 | 268.00 | 2024-02-14 15:15:00 | 274.05 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2024-02-16 11:15:00 | 277.70 | 2024-02-22 09:15:00 | 275.90 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-02-16 12:00:00 | 277.25 | 2024-02-22 09:15:00 | 275.90 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2024-02-16 12:45:00 | 277.30 | 2024-02-22 09:15:00 | 275.90 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-02-19 14:00:00 | 278.00 | 2024-02-22 09:15:00 | 275.90 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-02-20 09:15:00 | 282.50 | 2024-02-22 09:15:00 | 275.90 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2024-03-12 15:00:00 | 284.90 | 2024-03-13 09:15:00 | 270.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-12 15:00:00 | 284.90 | 2024-03-15 14:15:00 | 265.85 | STOP_HIT | 0.50 | 6.69% |
| BUY | retest2 | 2024-04-04 09:15:00 | 283.20 | 2024-04-04 11:15:00 | 275.90 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2024-04-29 13:15:00 | 292.80 | 2024-05-07 09:15:00 | 301.85 | STOP_HIT | 1.00 | 3.09% |
| BUY | retest2 | 2024-04-29 14:15:00 | 292.85 | 2024-05-07 09:15:00 | 301.85 | STOP_HIT | 1.00 | 3.07% |
| BUY | retest2 | 2024-04-29 15:00:00 | 293.45 | 2024-05-07 09:15:00 | 301.85 | STOP_HIT | 1.00 | 2.86% |
| SELL | retest2 | 2024-05-09 13:15:00 | 297.70 | 2024-05-10 09:15:00 | 302.20 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-05-13 12:45:00 | 304.15 | 2024-05-23 11:15:00 | 314.70 | STOP_HIT | 1.00 | 3.47% |
| BUY | retest2 | 2024-05-13 13:45:00 | 305.50 | 2024-05-23 11:15:00 | 314.70 | STOP_HIT | 1.00 | 3.01% |
| SELL | retest2 | 2024-05-27 14:30:00 | 319.05 | 2024-06-03 09:15:00 | 341.00 | STOP_HIT | 1.00 | -6.88% |
| BUY | retest2 | 2024-07-05 11:45:00 | 338.30 | 2024-07-16 14:15:00 | 342.10 | STOP_HIT | 1.00 | 1.12% |
| BUY | retest2 | 2024-07-05 12:30:00 | 338.15 | 2024-07-16 14:15:00 | 342.10 | STOP_HIT | 1.00 | 1.17% |
| BUY | retest2 | 2024-07-05 14:30:00 | 338.05 | 2024-07-16 14:15:00 | 342.10 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest2 | 2024-07-08 10:00:00 | 339.05 | 2024-07-16 14:15:00 | 342.10 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2024-07-09 12:00:00 | 341.65 | 2024-07-16 14:15:00 | 342.10 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2024-07-09 14:30:00 | 341.95 | 2024-07-16 15:15:00 | 342.75 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2024-07-10 09:15:00 | 341.90 | 2024-07-16 15:15:00 | 342.75 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2024-07-10 11:30:00 | 342.45 | 2024-07-16 15:15:00 | 342.75 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2024-07-12 10:45:00 | 344.65 | 2024-07-16 15:15:00 | 342.75 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-07-12 12:00:00 | 344.35 | 2024-07-16 15:15:00 | 342.75 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-07-12 12:45:00 | 344.05 | 2024-07-16 15:15:00 | 342.75 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-07-12 14:30:00 | 343.85 | 2024-07-16 15:15:00 | 342.75 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-07-16 11:00:00 | 345.05 | 2024-07-16 15:15:00 | 342.75 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-07-23 10:15:00 | 334.20 | 2024-07-24 15:15:00 | 337.05 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-07-23 12:00:00 | 334.20 | 2024-07-24 15:15:00 | 337.05 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-07-24 09:15:00 | 333.20 | 2024-07-24 15:15:00 | 337.05 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-07-30 09:15:00 | 354.15 | 2024-08-05 10:15:00 | 344.85 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-08-09 15:15:00 | 345.30 | 2024-08-19 09:15:00 | 341.60 | STOP_HIT | 1.00 | 1.07% |
| SELL | retest2 | 2024-09-06 11:00:00 | 325.60 | 2024-09-10 11:15:00 | 333.80 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2024-09-06 13:00:00 | 326.10 | 2024-09-10 11:15:00 | 333.80 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-09-09 09:15:00 | 324.40 | 2024-09-10 11:15:00 | 333.80 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2024-09-09 11:30:00 | 326.45 | 2024-09-10 11:15:00 | 333.80 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2024-09-12 09:15:00 | 336.25 | 2024-09-17 11:15:00 | 335.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-09-12 13:15:00 | 334.40 | 2024-09-17 11:15:00 | 335.00 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2024-09-12 13:45:00 | 334.05 | 2024-09-17 11:15:00 | 335.00 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2024-09-19 12:15:00 | 333.10 | 2024-09-20 09:15:00 | 337.15 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-09-19 13:00:00 | 333.80 | 2024-09-20 09:15:00 | 337.15 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 348.70 | 2024-10-07 10:15:00 | 331.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 348.70 | 2024-10-09 09:15:00 | 329.05 | STOP_HIT | 0.50 | 5.64% |
| SELL | retest2 | 2024-10-15 12:15:00 | 328.15 | 2024-10-17 14:15:00 | 331.50 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-10-16 12:30:00 | 328.30 | 2024-10-17 14:15:00 | 331.50 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-10-16 13:15:00 | 328.25 | 2024-10-17 14:15:00 | 331.50 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-10-16 15:00:00 | 327.10 | 2024-10-17 14:15:00 | 331.50 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-10-21 13:45:00 | 332.80 | 2024-10-22 10:15:00 | 328.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-10-22 09:15:00 | 333.70 | 2024-10-22 10:15:00 | 328.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-10-25 10:15:00 | 316.35 | 2024-10-29 15:15:00 | 320.20 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-10-25 13:45:00 | 315.95 | 2024-10-29 15:15:00 | 320.20 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-10-29 09:30:00 | 314.90 | 2024-10-29 15:15:00 | 320.20 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-10-29 12:15:00 | 316.35 | 2024-10-29 15:15:00 | 320.20 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-10-31 10:15:00 | 321.85 | 2024-11-04 09:15:00 | 315.80 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-10-31 10:45:00 | 322.15 | 2024-11-04 09:15:00 | 315.80 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-11-01 18:15:00 | 322.25 | 2024-11-04 09:15:00 | 315.80 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2024-11-27 13:00:00 | 340.90 | 2024-11-28 09:15:00 | 336.05 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-12-04 10:15:00 | 328.25 | 2024-12-06 09:15:00 | 330.55 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-12-05 14:45:00 | 325.10 | 2024-12-06 09:15:00 | 330.55 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-12-05 15:15:00 | 328.55 | 2024-12-06 09:15:00 | 330.55 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-12-13 11:15:00 | 328.85 | 2024-12-17 13:15:00 | 329.65 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2024-12-17 12:00:00 | 329.00 | 2024-12-17 13:15:00 | 329.65 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2024-12-20 12:15:00 | 320.30 | 2025-01-01 15:15:00 | 310.90 | STOP_HIT | 1.00 | 2.93% |
| SELL | retest2 | 2025-01-08 15:15:00 | 305.80 | 2025-01-13 13:15:00 | 290.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 10:45:00 | 306.10 | 2025-01-13 13:15:00 | 290.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 15:15:00 | 305.80 | 2025-01-14 10:15:00 | 292.65 | STOP_HIT | 0.50 | 4.30% |
| SELL | retest2 | 2025-01-09 10:45:00 | 306.10 | 2025-01-14 10:15:00 | 292.65 | STOP_HIT | 0.50 | 4.39% |
| SELL | retest2 | 2025-01-27 09:15:00 | 290.30 | 2025-01-30 09:15:00 | 295.75 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-02-06 10:30:00 | 284.40 | 2025-02-07 09:15:00 | 270.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 10:30:00 | 284.40 | 2025-02-12 09:15:00 | 255.96 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-20 11:00:00 | 264.85 | 2025-02-21 11:15:00 | 261.05 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-02-20 12:00:00 | 265.00 | 2025-02-21 11:15:00 | 261.05 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-03-10 09:15:00 | 269.90 | 2025-03-17 10:15:00 | 266.10 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-03-12 11:45:00 | 265.00 | 2025-03-17 10:15:00 | 266.10 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2025-03-27 15:00:00 | 295.10 | 2025-03-28 12:15:00 | 289.30 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-04-08 10:30:00 | 285.45 | 2025-04-09 09:15:00 | 295.75 | STOP_HIT | 1.00 | -3.61% |
| SELL | retest2 | 2025-04-08 11:30:00 | 285.60 | 2025-04-09 09:15:00 | 295.75 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2025-04-17 11:15:00 | 306.35 | 2025-04-25 09:15:00 | 308.00 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2025-04-28 14:30:00 | 308.05 | 2025-05-05 10:15:00 | 309.60 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-04-29 09:15:00 | 306.70 | 2025-05-05 10:15:00 | 309.60 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-04-30 10:30:00 | 308.10 | 2025-05-05 11:15:00 | 309.65 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-04-30 11:45:00 | 307.30 | 2025-05-05 11:15:00 | 309.65 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-04-30 15:15:00 | 305.20 | 2025-05-05 11:15:00 | 309.65 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-05-02 10:45:00 | 304.90 | 2025-05-05 11:15:00 | 309.65 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-05-16 09:15:00 | 298.25 | 2025-05-16 13:15:00 | 299.80 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-05-16 12:15:00 | 298.35 | 2025-05-16 13:15:00 | 299.80 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-06-03 09:30:00 | 291.40 | 2025-06-05 10:15:00 | 293.80 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-06-03 10:30:00 | 290.90 | 2025-06-05 10:15:00 | 293.80 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-06-18 12:45:00 | 285.85 | 2025-06-20 09:15:00 | 290.45 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-06-19 09:15:00 | 285.85 | 2025-06-20 09:15:00 | 290.45 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-06-19 10:30:00 | 286.25 | 2025-06-20 09:15:00 | 290.45 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-06-19 12:15:00 | 286.20 | 2025-06-20 09:15:00 | 290.45 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-06-23 11:15:00 | 290.85 | 2025-06-24 12:15:00 | 288.40 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-06-23 12:15:00 | 291.05 | 2025-06-24 12:15:00 | 288.40 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-06-24 09:15:00 | 292.05 | 2025-06-24 12:15:00 | 288.40 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-06-24 10:45:00 | 290.55 | 2025-06-24 12:15:00 | 288.40 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-06-26 15:00:00 | 293.95 | 2025-07-02 13:15:00 | 295.40 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2025-07-07 12:15:00 | 295.50 | 2025-07-07 14:15:00 | 296.00 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-07-11 09:15:00 | 300.65 | 2025-07-11 10:15:00 | 297.40 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-07-14 10:00:00 | 299.95 | 2025-07-14 12:15:00 | 298.10 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-07-14 11:30:00 | 299.80 | 2025-07-14 12:15:00 | 298.10 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-07-14 12:00:00 | 299.90 | 2025-07-14 12:15:00 | 298.10 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-07-22 10:15:00 | 295.25 | 2025-07-22 11:15:00 | 297.15 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest1 | 2025-07-23 10:15:00 | 298.70 | 2025-07-24 11:15:00 | 297.55 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-07-23 10:45:00 | 298.80 | 2025-07-24 11:15:00 | 297.55 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-07-23 14:00:00 | 298.55 | 2025-07-24 11:15:00 | 297.55 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-08-11 09:30:00 | 283.85 | 2025-08-12 09:15:00 | 286.80 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-09-09 13:15:00 | 282.70 | 2025-09-09 14:15:00 | 284.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-09-16 09:15:00 | 287.30 | 2025-09-17 13:15:00 | 286.20 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-09-17 10:30:00 | 287.35 | 2025-09-17 13:15:00 | 286.20 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-09-17 11:30:00 | 287.10 | 2025-09-17 13:15:00 | 286.20 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-09-18 14:30:00 | 287.85 | 2025-09-19 13:15:00 | 286.95 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-09-30 10:45:00 | 281.70 | 2025-10-03 11:15:00 | 284.95 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-09-30 11:15:00 | 281.70 | 2025-10-03 11:15:00 | 284.95 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-09-30 11:45:00 | 280.80 | 2025-10-03 11:15:00 | 284.95 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-10-01 11:00:00 | 281.70 | 2025-10-03 11:15:00 | 284.95 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-10-09 14:00:00 | 285.40 | 2025-10-10 09:15:00 | 290.20 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-11-04 09:15:00 | 279.25 | 2025-11-11 10:15:00 | 265.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 09:15:00 | 279.25 | 2025-11-11 15:15:00 | 268.15 | STOP_HIT | 0.50 | 3.97% |
| BUY | retest2 | 2025-11-20 09:15:00 | 277.25 | 2025-11-24 12:15:00 | 274.45 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-11-21 09:30:00 | 276.05 | 2025-11-24 12:15:00 | 274.45 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-11-21 10:15:00 | 276.65 | 2025-11-24 12:15:00 | 274.45 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-11-25 09:15:00 | 274.00 | 2025-12-04 15:15:00 | 269.10 | STOP_HIT | 1.00 | 1.79% |
| SELL | retest2 | 2025-11-25 10:00:00 | 274.80 | 2025-12-04 15:15:00 | 269.10 | STOP_HIT | 1.00 | 2.07% |
| SELL | retest2 | 2025-11-25 13:15:00 | 275.25 | 2025-12-04 15:15:00 | 269.10 | STOP_HIT | 1.00 | 2.23% |
| SELL | retest2 | 2025-11-26 11:00:00 | 274.85 | 2025-12-04 15:15:00 | 269.10 | STOP_HIT | 1.00 | 2.09% |
| SELL | retest2 | 2025-11-28 09:15:00 | 272.35 | 2025-12-04 15:15:00 | 269.10 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2025-12-11 09:15:00 | 263.25 | 2025-12-19 13:15:00 | 262.90 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-12-11 14:45:00 | 264.50 | 2025-12-19 13:15:00 | 262.90 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2025-12-12 10:45:00 | 264.10 | 2025-12-19 13:15:00 | 262.90 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest1 | 2025-12-26 09:15:00 | 268.85 | 2025-12-26 13:15:00 | 266.80 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest1 | 2025-12-26 10:45:00 | 268.75 | 2025-12-26 13:15:00 | 266.80 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest1 | 2025-12-26 11:15:00 | 268.85 | 2025-12-26 13:15:00 | 266.80 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest1 | 2025-12-26 11:45:00 | 268.75 | 2025-12-26 13:15:00 | 266.80 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-01-09 12:00:00 | 259.70 | 2026-01-16 09:15:00 | 260.50 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-02-09 14:15:00 | 289.65 | 2026-02-13 10:15:00 | 289.85 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2026-02-09 15:00:00 | 289.50 | 2026-02-13 10:15:00 | 289.85 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2026-02-10 09:15:00 | 291.30 | 2026-02-13 10:15:00 | 289.85 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2026-03-12 10:15:00 | 301.80 | 2026-03-16 09:15:00 | 297.15 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-03-13 13:15:00 | 300.50 | 2026-03-16 09:15:00 | 297.15 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-03-13 14:00:00 | 300.65 | 2026-03-16 09:15:00 | 297.15 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-03-13 14:30:00 | 301.20 | 2026-03-16 09:15:00 | 297.15 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-03-17 11:15:00 | 294.70 | 2026-03-18 09:15:00 | 302.25 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2026-04-01 11:00:00 | 294.85 | 2026-04-06 15:15:00 | 295.00 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2026-04-09 09:15:00 | 297.30 | 2026-04-24 09:15:00 | 316.50 | STOP_HIT | 1.00 | 6.46% |
