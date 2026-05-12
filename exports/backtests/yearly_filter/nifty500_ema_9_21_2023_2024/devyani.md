# Devyani International Ltd. (DEVYANI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 118.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 230 |
| ALERT1 | 144 |
| ALERT2 | 143 |
| ALERT2_SKIP | 84 |
| ALERT3 | 414 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 198 |
| PARTIAL | 34 |
| TARGET_HIT | 14 |
| STOP_HIT | 183 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 231 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 106 / 125
- **Target hits / Stop hits / Partials:** 14 / 183 / 34
- **Avg / median % per leg:** 1.10% / -0.37%
- **Sum % (uncompounded):** 254.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 82 | 18 | 22.0% | 9 | 73 | 0 | 0.16% | 13.3% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.91% | -1.9% |
| BUY @ 3rd Alert (retest2) | 81 | 18 | 22.2% | 9 | 72 | 0 | 0.19% | 15.2% |
| SELL (all) | 149 | 88 | 59.1% | 5 | 110 | 34 | 1.62% | 241.1% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.98% | -1.0% |
| SELL @ 3rd Alert (retest2) | 148 | 88 | 59.5% | 5 | 109 | 34 | 1.64% | 242.1% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.44% | -2.9% |
| retest2 (combined) | 229 | 106 | 46.3% | 14 | 181 | 34 | 1.12% | 257.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 09:15:00 | 177.55 | 176.34 | 176.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-15 10:15:00 | 180.00 | 177.07 | 176.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-16 15:15:00 | 181.95 | 183.25 | 181.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-16 15:15:00 | 181.95 | 183.25 | 181.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 15:15:00 | 181.95 | 183.25 | 181.48 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 12:15:00 | 172.60 | 180.66 | 180.83 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 12:15:00 | 182.50 | 180.20 | 180.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-18 14:15:00 | 183.80 | 181.31 | 180.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-19 09:15:00 | 179.55 | 181.27 | 180.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 09:15:00 | 179.55 | 181.27 | 180.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 09:15:00 | 179.55 | 181.27 | 180.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 10:00:00 | 179.55 | 181.27 | 180.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 10:15:00 | 179.35 | 180.89 | 180.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 10:30:00 | 179.50 | 180.89 | 180.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2023-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 12:15:00 | 178.95 | 180.17 | 180.33 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 14:15:00 | 182.80 | 180.81 | 180.60 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-22 10:15:00 | 179.70 | 180.35 | 180.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-22 11:15:00 | 179.10 | 180.10 | 180.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-23 12:15:00 | 178.45 | 177.64 | 178.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 12:15:00 | 178.45 | 177.64 | 178.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 12:15:00 | 178.45 | 177.64 | 178.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 13:00:00 | 178.45 | 177.64 | 178.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 13:15:00 | 179.20 | 177.95 | 178.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 13:30:00 | 178.50 | 177.95 | 178.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 14:15:00 | 177.00 | 177.76 | 178.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-23 15:15:00 | 176.10 | 177.76 | 178.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-24 09:15:00 | 181.00 | 178.14 | 178.58 | SL hit (close>static) qty=1.00 sl=179.35 alert=retest2 |

### Cycle 7 — BUY (started 2023-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 10:15:00 | 182.40 | 178.99 | 178.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-24 12:15:00 | 182.65 | 180.29 | 179.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 14:15:00 | 184.30 | 187.96 | 186.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-29 14:15:00 | 184.30 | 187.96 | 186.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 14:15:00 | 184.30 | 187.96 | 186.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-29 15:00:00 | 184.30 | 187.96 | 186.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 15:15:00 | 184.20 | 187.21 | 186.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 09:30:00 | 185.50 | 186.69 | 185.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-30 10:15:00 | 183.20 | 186.00 | 185.72 | SL hit (close<static) qty=1.00 sl=183.25 alert=retest2 |

### Cycle 8 — SELL (started 2023-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 11:15:00 | 183.45 | 185.49 | 185.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-30 13:15:00 | 181.15 | 184.30 | 184.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 14:15:00 | 183.80 | 181.21 | 182.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 14:15:00 | 183.80 | 181.21 | 182.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 14:15:00 | 183.80 | 181.21 | 182.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 15:00:00 | 183.80 | 181.21 | 182.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 15:15:00 | 185.50 | 182.07 | 182.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-01 09:30:00 | 183.45 | 182.42 | 182.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-01 11:15:00 | 185.15 | 183.28 | 183.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2023-06-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 11:15:00 | 185.15 | 183.28 | 183.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 12:15:00 | 187.45 | 184.12 | 183.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 12:15:00 | 184.55 | 184.99 | 184.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 12:15:00 | 184.55 | 184.99 | 184.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 12:15:00 | 184.55 | 184.99 | 184.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 13:00:00 | 184.55 | 184.99 | 184.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 14:15:00 | 185.45 | 185.06 | 184.54 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 12:15:00 | 183.35 | 184.24 | 184.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-05 13:15:00 | 182.90 | 183.98 | 184.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-07 09:15:00 | 182.50 | 181.16 | 182.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-07 09:15:00 | 182.50 | 181.16 | 182.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 09:15:00 | 182.50 | 181.16 | 182.19 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-06-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-08 11:15:00 | 182.65 | 182.29 | 182.26 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-06-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 13:15:00 | 181.90 | 182.24 | 182.25 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-06-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-08 15:15:00 | 183.00 | 182.32 | 182.28 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 13:15:00 | 181.55 | 182.21 | 182.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 14:15:00 | 180.60 | 181.89 | 182.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 09:15:00 | 182.70 | 180.02 | 180.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 09:15:00 | 182.70 | 180.02 | 180.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 182.70 | 180.02 | 180.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 09:30:00 | 183.75 | 180.02 | 180.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 10:15:00 | 182.75 | 180.57 | 180.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 10:30:00 | 182.80 | 180.57 | 180.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2023-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 11:15:00 | 182.90 | 181.03 | 181.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 13:15:00 | 183.30 | 181.81 | 181.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 14:15:00 | 190.45 | 190.56 | 188.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-15 14:45:00 | 189.60 | 190.56 | 188.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 15:15:00 | 189.00 | 190.25 | 188.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 09:15:00 | 191.60 | 190.25 | 188.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 10:30:00 | 190.70 | 190.38 | 188.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-21 11:15:00 | 191.00 | 192.59 | 192.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2023-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 11:15:00 | 191.00 | 192.59 | 192.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-21 14:15:00 | 189.95 | 191.58 | 192.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-22 09:15:00 | 193.00 | 191.61 | 192.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 09:15:00 | 193.00 | 191.61 | 192.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 193.00 | 191.61 | 192.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 09:30:00 | 192.80 | 191.61 | 192.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 190.80 | 191.45 | 191.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 10:45:00 | 190.80 | 191.45 | 191.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 14:15:00 | 191.00 | 191.06 | 191.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 14:30:00 | 191.30 | 191.06 | 191.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 188.55 | 189.33 | 190.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 09:30:00 | 190.35 | 189.33 | 190.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 193.65 | 190.19 | 190.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 11:00:00 | 193.65 | 190.19 | 190.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2023-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 11:15:00 | 193.20 | 190.79 | 190.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-26 14:15:00 | 194.85 | 192.34 | 191.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-27 09:15:00 | 191.60 | 192.48 | 191.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 191.60 | 192.48 | 191.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 191.60 | 192.48 | 191.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:00:00 | 191.60 | 192.48 | 191.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 191.25 | 192.23 | 191.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:45:00 | 191.65 | 192.23 | 191.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 11:15:00 | 190.50 | 191.89 | 191.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 12:00:00 | 190.50 | 191.89 | 191.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2023-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 13:15:00 | 190.45 | 191.19 | 191.29 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 12:15:00 | 190.30 | 189.23 | 189.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-05 15:15:00 | 190.50 | 189.72 | 189.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-06 14:15:00 | 192.15 | 192.26 | 191.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-06 14:45:00 | 191.80 | 192.26 | 191.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 191.25 | 192.01 | 191.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-10 09:15:00 | 193.05 | 191.18 | 191.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-10 12:15:00 | 193.20 | 192.04 | 191.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-10 12:15:00 | 189.90 | 191.61 | 191.37 | SL hit (close<static) qty=1.00 sl=190.40 alert=retest2 |

### Cycle 20 — SELL (started 2023-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 13:15:00 | 188.80 | 191.05 | 191.14 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 11:15:00 | 192.10 | 190.68 | 190.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-13 11:15:00 | 195.50 | 192.40 | 191.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-14 12:15:00 | 194.20 | 194.24 | 193.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-14 12:45:00 | 193.90 | 194.24 | 193.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 15:15:00 | 194.70 | 194.42 | 193.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-17 09:30:00 | 195.30 | 194.43 | 193.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-18 09:15:00 | 191.75 | 193.30 | 193.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2023-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 09:15:00 | 191.75 | 193.30 | 193.39 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 14:15:00 | 195.80 | 193.76 | 193.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-18 15:15:00 | 197.25 | 194.46 | 193.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 09:15:00 | 197.35 | 197.95 | 196.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 09:15:00 | 197.35 | 197.95 | 196.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 197.35 | 197.95 | 196.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 09:30:00 | 197.10 | 197.95 | 196.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 13:15:00 | 196.60 | 197.35 | 196.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 13:30:00 | 196.65 | 197.35 | 196.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 14:15:00 | 196.60 | 197.20 | 196.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 14:30:00 | 196.25 | 197.20 | 196.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 15:15:00 | 195.60 | 196.88 | 196.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 09:15:00 | 196.35 | 196.88 | 196.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 198.10 | 197.12 | 196.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 11:45:00 | 198.60 | 197.58 | 196.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 14:15:00 | 198.55 | 198.05 | 197.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-24 11:15:00 | 196.10 | 197.09 | 197.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2023-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 11:15:00 | 196.10 | 197.09 | 197.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 14:15:00 | 195.55 | 196.63 | 196.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-25 13:15:00 | 197.55 | 195.86 | 196.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 13:15:00 | 197.55 | 195.86 | 196.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 13:15:00 | 197.55 | 195.86 | 196.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 14:00:00 | 197.55 | 195.86 | 196.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2023-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 14:15:00 | 199.20 | 196.53 | 196.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-26 09:15:00 | 202.70 | 198.10 | 197.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 12:15:00 | 198.20 | 198.67 | 197.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 12:15:00 | 198.20 | 198.67 | 197.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 12:15:00 | 198.20 | 198.67 | 197.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-26 12:30:00 | 197.85 | 198.67 | 197.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 13:15:00 | 198.15 | 198.56 | 197.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-26 13:45:00 | 197.90 | 198.56 | 197.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 14:15:00 | 195.35 | 197.92 | 197.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-26 15:00:00 | 195.35 | 197.92 | 197.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 15:15:00 | 195.50 | 197.44 | 197.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-27 09:15:00 | 196.50 | 197.44 | 197.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-27 09:15:00 | 195.70 | 197.09 | 197.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2023-07-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 09:15:00 | 195.70 | 197.09 | 197.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-27 10:15:00 | 195.00 | 196.67 | 197.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-28 10:15:00 | 195.70 | 195.12 | 195.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-28 10:15:00 | 195.70 | 195.12 | 195.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 10:15:00 | 195.70 | 195.12 | 195.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 10:45:00 | 195.55 | 195.12 | 195.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 11:15:00 | 196.15 | 195.33 | 195.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 11:30:00 | 196.30 | 195.33 | 195.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 12:15:00 | 198.00 | 195.86 | 196.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 13:00:00 | 198.00 | 195.86 | 196.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 13:15:00 | 197.00 | 196.09 | 196.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 13:30:00 | 198.10 | 196.09 | 196.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2023-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 14:15:00 | 196.95 | 196.26 | 196.25 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-31 09:15:00 | 194.70 | 196.02 | 196.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-31 10:15:00 | 193.95 | 195.61 | 195.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-01 09:15:00 | 195.15 | 194.07 | 194.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 09:15:00 | 195.15 | 194.07 | 194.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 09:15:00 | 195.15 | 194.07 | 194.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 09:30:00 | 195.05 | 194.07 | 194.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 193.70 | 194.00 | 194.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-03 09:30:00 | 191.95 | 194.08 | 194.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-03 12:45:00 | 192.95 | 193.85 | 194.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-03 13:15:00 | 192.95 | 193.85 | 194.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-03 14:45:00 | 193.00 | 193.77 | 194.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 15:15:00 | 194.60 | 193.94 | 194.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 09:15:00 | 196.55 | 193.94 | 194.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-08-04 09:15:00 | 195.65 | 194.28 | 194.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2023-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 09:15:00 | 195.65 | 194.28 | 194.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 10:15:00 | 197.20 | 194.87 | 194.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-04 12:15:00 | 191.10 | 194.53 | 194.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 12:15:00 | 191.10 | 194.53 | 194.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 12:15:00 | 191.10 | 194.53 | 194.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-04 12:30:00 | 192.95 | 194.53 | 194.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2023-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-04 13:15:00 | 188.25 | 193.27 | 193.89 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 14:15:00 | 193.85 | 192.21 | 192.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-10 09:15:00 | 194.45 | 192.83 | 192.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 14:15:00 | 198.45 | 198.51 | 196.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-11 15:00:00 | 198.45 | 198.51 | 196.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 197.30 | 198.19 | 196.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 09:30:00 | 196.45 | 198.19 | 196.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 10:15:00 | 196.30 | 197.81 | 196.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 10:30:00 | 196.35 | 197.81 | 196.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 11:15:00 | 196.20 | 197.49 | 196.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 11:45:00 | 196.20 | 197.49 | 196.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 199.70 | 197.68 | 196.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 11:00:00 | 201.00 | 199.17 | 198.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-18 11:15:00 | 196.90 | 198.38 | 198.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2023-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 11:15:00 | 196.90 | 198.38 | 198.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-21 09:15:00 | 194.20 | 197.12 | 197.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-22 12:15:00 | 196.10 | 195.47 | 196.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 12:15:00 | 196.10 | 195.47 | 196.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 12:15:00 | 196.10 | 195.47 | 196.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 12:30:00 | 196.30 | 195.47 | 196.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 13:15:00 | 196.30 | 195.64 | 196.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 14:00:00 | 196.30 | 195.64 | 196.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 14:15:00 | 196.00 | 195.71 | 196.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 15:00:00 | 196.00 | 195.71 | 196.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 15:15:00 | 196.10 | 195.79 | 196.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 09:15:00 | 197.75 | 195.79 | 196.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 197.30 | 196.09 | 196.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 13:00:00 | 196.00 | 196.30 | 196.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 13:30:00 | 196.00 | 196.20 | 196.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 14:00:00 | 195.80 | 196.20 | 196.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 15:00:00 | 195.30 | 196.02 | 196.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 09:15:00 | 195.00 | 195.58 | 195.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 10:45:00 | 194.65 | 195.45 | 195.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 11:15:00 | 194.75 | 195.45 | 195.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 12:00:00 | 194.75 | 195.31 | 195.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 13:30:00 | 194.70 | 195.04 | 195.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 13:15:00 | 193.05 | 191.46 | 192.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 14:00:00 | 193.05 | 191.46 | 192.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 14:15:00 | 192.55 | 191.68 | 192.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 15:15:00 | 193.00 | 191.68 | 192.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 15:15:00 | 193.00 | 191.94 | 192.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 09:15:00 | 193.15 | 191.94 | 192.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 11:15:00 | 192.40 | 192.41 | 192.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 15:15:00 | 191.45 | 192.15 | 192.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-30 11:15:00 | 193.80 | 192.56 | 192.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2023-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 11:15:00 | 193.80 | 192.56 | 192.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 09:15:00 | 196.40 | 194.34 | 193.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 15:15:00 | 194.95 | 195.39 | 194.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 15:15:00 | 194.95 | 195.39 | 194.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 15:15:00 | 194.95 | 195.39 | 194.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-01 09:15:00 | 197.00 | 195.39 | 194.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-05 10:15:00 | 216.70 | 203.89 | 200.12 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 211.10 | 219.09 | 220.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 209.80 | 217.24 | 219.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 10:15:00 | 213.40 | 212.53 | 215.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 10:15:00 | 213.40 | 212.53 | 215.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 10:15:00 | 213.40 | 212.53 | 215.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 10:45:00 | 215.65 | 212.53 | 215.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 213.55 | 211.35 | 213.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 12:15:00 | 210.45 | 211.82 | 212.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-18 09:15:00 | 210.40 | 211.20 | 212.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-18 10:45:00 | 210.70 | 211.16 | 211.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-18 11:30:00 | 210.60 | 210.98 | 211.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 15:15:00 | 210.20 | 210.47 | 211.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-20 09:15:00 | 209.70 | 210.47 | 211.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-20 10:00:00 | 209.80 | 210.33 | 211.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-20 12:30:00 | 209.95 | 209.87 | 210.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-20 13:00:00 | 209.85 | 209.87 | 210.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 14:15:00 | 208.45 | 207.37 | 208.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 14:30:00 | 208.70 | 207.37 | 208.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 15:15:00 | 208.70 | 207.64 | 208.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 09:15:00 | 206.70 | 207.64 | 208.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 206.70 | 207.45 | 208.01 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-09-26 10:15:00 | 209.50 | 207.98 | 207.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2023-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 10:15:00 | 209.50 | 207.98 | 207.95 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-09-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 15:15:00 | 207.00 | 208.43 | 208.52 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 09:15:00 | 212.40 | 209.23 | 208.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 10:15:00 | 212.80 | 209.94 | 209.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 09:15:00 | 215.00 | 216.76 | 214.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 09:15:00 | 215.00 | 216.76 | 214.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 215.00 | 216.76 | 214.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 09:30:00 | 214.80 | 216.76 | 214.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 10:15:00 | 220.25 | 217.46 | 215.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 09:30:00 | 221.50 | 217.96 | 216.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-06 10:45:00 | 221.50 | 219.27 | 218.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-06 12:15:00 | 221.70 | 219.54 | 218.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-09 13:15:00 | 216.50 | 218.34 | 218.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2023-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 13:15:00 | 216.50 | 218.34 | 218.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-11 13:15:00 | 215.30 | 217.21 | 217.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-16 09:15:00 | 212.85 | 212.34 | 213.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-16 10:00:00 | 212.85 | 212.34 | 213.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 211.00 | 211.57 | 212.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 12:45:00 | 210.50 | 211.19 | 212.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 13:30:00 | 210.40 | 210.71 | 211.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 09:15:00 | 199.97 | 205.23 | 207.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 09:15:00 | 199.88 | 205.23 | 207.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-20 15:15:00 | 197.25 | 196.19 | 199.59 | SL hit (close>ema200) qty=0.50 sl=196.19 alert=retest2 |

### Cycle 39 — BUY (started 2023-11-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 14:15:00 | 184.25 | 183.58 | 183.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 11:15:00 | 185.45 | 184.37 | 183.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-06 09:15:00 | 184.50 | 185.04 | 184.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 09:15:00 | 184.50 | 185.04 | 184.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 184.50 | 185.04 | 184.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 13:45:00 | 189.65 | 186.40 | 185.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-08 11:15:00 | 184.65 | 187.37 | 187.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2023-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 11:15:00 | 184.65 | 187.37 | 187.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 09:15:00 | 181.00 | 184.87 | 186.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 11:15:00 | 182.00 | 181.48 | 183.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-10 12:00:00 | 182.00 | 181.48 | 183.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 14:15:00 | 182.60 | 182.02 | 183.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 15:00:00 | 182.60 | 182.02 | 183.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 15:15:00 | 183.10 | 182.24 | 183.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-12 18:15:00 | 185.00 | 182.24 | 183.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 184.85 | 182.76 | 183.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 09:15:00 | 184.00 | 182.76 | 183.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 10:15:00 | 184.50 | 183.16 | 183.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-13 11:15:00 | 183.60 | 183.44 | 183.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2023-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 11:15:00 | 183.60 | 183.44 | 183.44 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 12:15:00 | 183.25 | 183.40 | 183.42 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2023-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 13:15:00 | 183.60 | 183.44 | 183.44 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2023-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 14:15:00 | 183.10 | 183.37 | 183.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 15:15:00 | 182.50 | 183.20 | 183.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 13:15:00 | 182.50 | 182.39 | 182.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 13:15:00 | 182.50 | 182.39 | 182.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 13:15:00 | 182.50 | 182.39 | 182.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 13:45:00 | 182.45 | 182.39 | 182.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 15:15:00 | 182.50 | 182.35 | 182.71 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2023-11-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-16 12:15:00 | 183.50 | 182.89 | 182.88 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-11-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 13:15:00 | 182.55 | 182.82 | 182.85 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 11:15:00 | 183.70 | 182.96 | 182.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-17 14:15:00 | 185.35 | 183.56 | 183.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 09:15:00 | 183.50 | 183.85 | 183.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 09:15:00 | 183.50 | 183.85 | 183.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 183.50 | 183.85 | 183.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 10:00:00 | 183.50 | 183.85 | 183.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 10:15:00 | 183.70 | 183.82 | 183.43 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2023-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 15:15:00 | 182.10 | 183.07 | 183.18 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2023-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 09:15:00 | 185.70 | 183.60 | 183.41 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2023-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 12:15:00 | 181.95 | 184.00 | 184.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 14:15:00 | 180.00 | 182.89 | 183.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 15:15:00 | 181.50 | 181.39 | 182.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-24 09:15:00 | 181.60 | 181.39 | 182.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 182.45 | 181.60 | 182.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 10:00:00 | 182.45 | 181.60 | 182.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 181.95 | 181.67 | 182.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 09:30:00 | 181.65 | 181.35 | 181.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-04 12:15:00 | 178.00 | 176.81 | 176.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2023-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 12:15:00 | 178.00 | 176.81 | 176.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 14:15:00 | 180.10 | 177.64 | 177.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 13:15:00 | 186.25 | 186.62 | 185.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-08 13:45:00 | 186.15 | 186.62 | 185.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 185.55 | 186.37 | 185.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-11 10:00:00 | 185.55 | 186.37 | 185.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 11:15:00 | 185.60 | 186.14 | 185.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 14:15:00 | 186.90 | 186.11 | 185.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 09:15:00 | 188.05 | 185.86 | 185.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-12 15:15:00 | 185.10 | 185.99 | 185.83 | SL hit (close<static) qty=1.00 sl=185.15 alert=retest2 |

### Cycle 52 — SELL (started 2023-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 10:15:00 | 184.35 | 185.47 | 185.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 12:15:00 | 182.40 | 184.65 | 185.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 09:15:00 | 184.90 | 183.85 | 184.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 184.90 | 183.85 | 184.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 184.90 | 183.85 | 184.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-14 10:30:00 | 183.25 | 183.50 | 184.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-14 14:00:00 | 183.50 | 183.43 | 184.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-15 12:15:00 | 184.95 | 184.28 | 184.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2023-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 12:15:00 | 184.95 | 184.28 | 184.24 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2023-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 09:15:00 | 183.00 | 184.26 | 184.27 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2023-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 09:15:00 | 193.10 | 185.48 | 184.69 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2023-12-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 10:15:00 | 184.80 | 189.03 | 189.20 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2023-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 09:15:00 | 191.20 | 189.00 | 188.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 09:15:00 | 193.10 | 191.95 | 191.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 15:15:00 | 193.00 | 193.00 | 192.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-29 09:15:00 | 191.70 | 193.00 | 192.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 190.45 | 192.49 | 191.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 10:00:00 | 190.45 | 192.49 | 191.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 10:15:00 | 190.55 | 192.10 | 191.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 11:15:00 | 191.40 | 192.10 | 191.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-02 09:15:00 | 190.35 | 191.96 | 192.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 09:15:00 | 190.35 | 191.96 | 192.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 10:15:00 | 189.85 | 191.54 | 191.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-03 13:15:00 | 191.20 | 190.31 | 190.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-03 13:15:00 | 191.20 | 190.31 | 190.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 13:15:00 | 191.20 | 190.31 | 190.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 13:45:00 | 191.00 | 190.31 | 190.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 14:15:00 | 190.50 | 190.35 | 190.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-04 09:15:00 | 190.20 | 190.38 | 190.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-11 10:15:00 | 189.35 | 187.36 | 187.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2024-01-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 10:15:00 | 189.35 | 187.36 | 187.10 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 10:15:00 | 186.40 | 187.21 | 187.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 13:15:00 | 185.90 | 186.78 | 187.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-15 13:15:00 | 186.25 | 186.16 | 186.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-15 13:45:00 | 186.30 | 186.16 | 186.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 187.80 | 186.46 | 186.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 10:00:00 | 187.80 | 186.46 | 186.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 10:15:00 | 187.05 | 186.58 | 186.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 10:30:00 | 187.55 | 186.58 | 186.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2024-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 11:15:00 | 187.20 | 186.70 | 186.65 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 09:15:00 | 181.90 | 185.81 | 186.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 11:15:00 | 179.20 | 181.47 | 182.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 12:15:00 | 178.65 | 178.21 | 179.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 13:00:00 | 178.65 | 178.21 | 179.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 14:15:00 | 178.30 | 176.75 | 177.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 15:00:00 | 178.30 | 176.75 | 177.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 15:15:00 | 179.15 | 177.23 | 178.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-29 09:15:00 | 175.55 | 177.23 | 178.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-31 09:15:00 | 182.05 | 176.10 | 175.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2024-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 09:15:00 | 182.05 | 176.10 | 175.77 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 11:15:00 | 176.50 | 178.00 | 178.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 12:15:00 | 173.65 | 177.13 | 177.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 14:15:00 | 153.15 | 151.92 | 154.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 14:15:00 | 153.15 | 151.92 | 154.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 14:15:00 | 153.15 | 151.92 | 154.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-09 15:00:00 | 153.15 | 151.92 | 154.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 15:15:00 | 154.90 | 152.51 | 154.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-12 09:15:00 | 154.90 | 152.51 | 154.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 151.50 | 152.31 | 154.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 11:45:00 | 150.70 | 151.76 | 153.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 09:15:00 | 148.10 | 151.76 | 152.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 12:00:00 | 149.85 | 151.16 | 152.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 14:30:00 | 150.80 | 151.01 | 151.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 151.35 | 151.13 | 151.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 10:30:00 | 152.15 | 151.13 | 151.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 11:15:00 | 152.45 | 151.39 | 151.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 12:00:00 | 152.45 | 151.39 | 151.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 12:15:00 | 152.40 | 151.59 | 151.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 13:00:00 | 152.40 | 151.59 | 151.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-14 14:15:00 | 154.20 | 152.29 | 152.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2024-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 14:15:00 | 154.20 | 152.29 | 152.18 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-02-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 15:15:00 | 150.25 | 152.19 | 152.41 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 09:15:00 | 158.25 | 153.40 | 152.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 12:15:00 | 161.50 | 156.30 | 154.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 09:15:00 | 163.00 | 163.08 | 160.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-20 09:30:00 | 163.55 | 163.08 | 160.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 165.00 | 166.68 | 165.04 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 10:15:00 | 161.90 | 164.46 | 164.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-23 15:15:00 | 161.00 | 162.44 | 163.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-26 12:15:00 | 161.70 | 161.51 | 162.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-26 13:00:00 | 161.70 | 161.51 | 162.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 15:15:00 | 161.20 | 161.59 | 162.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 09:15:00 | 157.05 | 161.59 | 162.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-02 11:15:00 | 157.90 | 155.77 | 155.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2024-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 11:15:00 | 157.90 | 155.77 | 155.50 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 11:15:00 | 153.70 | 155.51 | 155.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 13:15:00 | 152.00 | 154.54 | 155.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 09:15:00 | 155.25 | 154.16 | 154.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 09:15:00 | 155.25 | 154.16 | 154.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 155.25 | 154.16 | 154.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 09:30:00 | 154.80 | 154.16 | 154.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 154.60 | 154.25 | 154.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 10:30:00 | 155.05 | 154.25 | 154.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 11:15:00 | 154.95 | 154.39 | 154.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 11:45:00 | 155.20 | 154.39 | 154.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 12:15:00 | 155.05 | 154.52 | 154.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 13:00:00 | 155.05 | 154.52 | 154.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 13:15:00 | 154.90 | 154.60 | 154.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 14:15:00 | 153.65 | 154.60 | 154.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 15:00:00 | 154.50 | 154.58 | 154.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-06 09:15:00 | 154.30 | 154.60 | 154.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-07 09:15:00 | 156.15 | 154.20 | 154.29 | SL hit (close>static) qty=1.00 sl=155.20 alert=retest2 |

### Cycle 71 — BUY (started 2024-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 10:15:00 | 156.00 | 154.56 | 154.44 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 11:15:00 | 154.00 | 154.72 | 154.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 10:15:00 | 151.60 | 153.62 | 154.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 14:15:00 | 148.10 | 147.92 | 150.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-13 15:00:00 | 148.10 | 147.92 | 150.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 149.25 | 148.01 | 149.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 12:00:00 | 149.25 | 148.01 | 149.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 13:15:00 | 151.35 | 148.82 | 149.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 14:00:00 | 151.35 | 148.82 | 149.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 151.05 | 149.26 | 149.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 14:30:00 | 151.80 | 149.26 | 149.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-03-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 09:15:00 | 152.00 | 150.17 | 150.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 11:15:00 | 153.15 | 151.06 | 150.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 09:15:00 | 154.50 | 155.83 | 154.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-19 10:00:00 | 154.50 | 155.83 | 154.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 155.40 | 155.74 | 154.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-19 12:30:00 | 156.10 | 155.82 | 154.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 11:00:00 | 156.80 | 156.11 | 155.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-21 11:45:00 | 156.10 | 156.71 | 156.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-21 12:30:00 | 156.15 | 156.64 | 156.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 14:15:00 | 156.05 | 156.49 | 156.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-21 14:30:00 | 156.40 | 156.49 | 156.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 15:15:00 | 156.25 | 156.44 | 156.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-22 09:15:00 | 156.55 | 156.44 | 156.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 09:15:00 | 157.00 | 156.55 | 156.26 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-03-22 15:15:00 | 155.25 | 156.07 | 156.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-03-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 15:15:00 | 155.25 | 156.07 | 156.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 09:15:00 | 154.50 | 155.75 | 156.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 14:15:00 | 155.00 | 154.93 | 155.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 14:15:00 | 155.00 | 154.93 | 155.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 14:15:00 | 155.00 | 154.93 | 155.43 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2024-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 11:15:00 | 156.50 | 155.65 | 155.64 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 12:15:00 | 154.20 | 155.36 | 155.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 14:15:00 | 151.85 | 154.47 | 155.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 09:15:00 | 154.00 | 153.91 | 154.68 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-28 12:30:00 | 152.80 | 153.44 | 154.26 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 154.30 | 152.69 | 153.57 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-04-01 09:15:00 | 154.30 | 152.69 | 153.57 | SL hit (close>ema400) qty=1.00 sl=153.57 alert=retest1 |

### Cycle 77 — BUY (started 2024-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 13:15:00 | 155.30 | 153.98 | 153.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 09:15:00 | 156.35 | 154.79 | 154.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-02 15:15:00 | 155.70 | 155.77 | 155.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-02 15:15:00 | 155.70 | 155.77 | 155.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 15:15:00 | 155.70 | 155.77 | 155.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 09:15:00 | 155.05 | 155.77 | 155.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 154.90 | 155.60 | 155.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 10:00:00 | 154.90 | 155.60 | 155.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 10:15:00 | 153.70 | 155.22 | 155.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 11:00:00 | 153.70 | 155.22 | 155.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2024-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 12:15:00 | 154.45 | 154.82 | 154.85 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 15:15:00 | 155.40 | 154.70 | 154.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-05 09:15:00 | 156.90 | 155.14 | 154.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-08 10:15:00 | 156.30 | 156.36 | 155.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-08 10:15:00 | 156.30 | 156.36 | 155.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 10:15:00 | 156.30 | 156.36 | 155.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 11:00:00 | 156.30 | 156.36 | 155.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 12:15:00 | 158.10 | 156.67 | 156.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 14:30:00 | 159.20 | 157.36 | 156.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 10:15:00 | 159.40 | 162.99 | 163.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2024-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 10:15:00 | 159.40 | 162.99 | 163.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 12:15:00 | 158.50 | 161.59 | 162.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 160.50 | 160.18 | 161.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 160.50 | 160.18 | 161.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 160.50 | 160.18 | 161.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:45:00 | 160.90 | 160.18 | 161.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 163.45 | 160.80 | 161.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:45:00 | 163.90 | 160.80 | 161.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-04-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 10:15:00 | 164.40 | 161.52 | 161.40 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-04-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 10:15:00 | 160.00 | 161.54 | 161.61 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-04-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 13:15:00 | 162.20 | 161.71 | 161.67 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-04-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 15:15:00 | 161.05 | 161.57 | 161.61 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 09:15:00 | 161.95 | 161.65 | 161.64 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-22 11:15:00 | 160.15 | 161.37 | 161.52 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 14:15:00 | 163.05 | 161.74 | 161.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 164.30 | 162.46 | 162.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 10:15:00 | 165.90 | 166.30 | 165.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 11:00:00 | 165.90 | 166.30 | 165.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 13:15:00 | 168.55 | 166.63 | 165.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 13:30:00 | 167.15 | 166.63 | 165.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 167.25 | 168.42 | 167.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 09:30:00 | 167.30 | 168.42 | 167.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 168.20 | 168.38 | 167.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 10:30:00 | 167.25 | 168.38 | 167.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 12:15:00 | 166.55 | 167.96 | 167.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 13:00:00 | 166.55 | 167.96 | 167.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 13:15:00 | 166.30 | 167.63 | 167.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 14:00:00 | 166.30 | 167.63 | 167.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2024-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 15:15:00 | 166.80 | 167.34 | 167.37 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 09:15:00 | 167.75 | 167.42 | 167.41 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 10:15:00 | 166.90 | 167.32 | 167.36 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 11:15:00 | 168.00 | 167.45 | 167.42 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 14:15:00 | 166.45 | 167.31 | 167.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 11:15:00 | 165.75 | 166.76 | 167.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-02 13:15:00 | 167.05 | 166.72 | 166.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 13:15:00 | 167.05 | 166.72 | 166.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 13:15:00 | 167.05 | 166.72 | 166.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 14:00:00 | 167.05 | 166.72 | 166.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 14:15:00 | 164.80 | 166.34 | 166.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 11:45:00 | 164.50 | 165.70 | 166.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 12:45:00 | 164.30 | 165.52 | 166.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 13:30:00 | 164.20 | 165.26 | 166.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 09:30:00 | 164.30 | 165.14 | 165.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 11:15:00 | 165.60 | 163.93 | 164.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 12:00:00 | 165.60 | 163.93 | 164.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 163.70 | 163.89 | 164.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 12:30:00 | 167.65 | 163.89 | 164.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 13:15:00 | 163.75 | 163.86 | 164.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 14:15:00 | 161.85 | 163.86 | 164.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 10:15:00 | 156.28 | 160.08 | 161.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 10:15:00 | 156.09 | 160.08 | 161.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 10:15:00 | 155.99 | 160.08 | 161.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 10:15:00 | 156.09 | 160.08 | 161.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 09:15:00 | 153.76 | 156.56 | 158.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-14 14:15:00 | 157.30 | 156.40 | 157.74 | SL hit (close>ema200) qty=0.50 sl=156.40 alert=retest2 |

### Cycle 93 — BUY (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 09:15:00 | 153.15 | 152.39 | 152.30 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 11:15:00 | 152.15 | 152.56 | 152.60 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 13:15:00 | 153.15 | 152.54 | 152.49 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 151.40 | 152.28 | 152.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 150.05 | 151.18 | 151.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 12:15:00 | 151.30 | 151.05 | 151.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 12:15:00 | 151.30 | 151.05 | 151.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 12:15:00 | 151.30 | 151.05 | 151.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:00:00 | 151.30 | 151.05 | 151.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 151.75 | 151.13 | 151.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 14:30:00 | 151.70 | 151.13 | 151.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 151.85 | 151.27 | 151.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:15:00 | 151.30 | 151.27 | 151.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 150.85 | 151.19 | 151.36 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 14:15:00 | 154.30 | 151.96 | 151.66 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 146.15 | 151.16 | 151.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 145.40 | 150.01 | 151.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 13:15:00 | 149.85 | 149.65 | 150.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 13:15:00 | 149.85 | 149.65 | 150.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 13:15:00 | 149.85 | 149.65 | 150.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 13:30:00 | 151.70 | 149.65 | 150.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 152.05 | 150.10 | 150.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 152.05 | 150.10 | 150.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 152.55 | 150.59 | 150.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:45:00 | 152.55 | 150.59 | 150.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 154.25 | 151.32 | 151.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 12:15:00 | 156.55 | 152.37 | 151.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 09:15:00 | 178.20 | 179.37 | 177.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 09:30:00 | 178.66 | 179.37 | 177.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 178.15 | 179.13 | 177.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 11:15:00 | 178.51 | 179.13 | 177.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 11:00:00 | 178.27 | 178.92 | 178.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 12:30:00 | 178.31 | 178.53 | 178.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 13:45:00 | 178.25 | 178.44 | 178.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 178.30 | 178.41 | 178.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 14:30:00 | 178.21 | 178.41 | 178.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 178.90 | 178.51 | 178.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:15:00 | 175.81 | 178.51 | 178.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-18 09:15:00 | 173.02 | 177.41 | 177.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2024-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 09:15:00 | 173.02 | 177.41 | 177.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 13:15:00 | 172.53 | 173.86 | 175.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 173.53 | 170.89 | 172.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 173.53 | 170.89 | 172.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 173.53 | 170.89 | 172.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:00:00 | 173.53 | 170.89 | 172.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 172.14 | 171.14 | 172.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:30:00 | 171.95 | 171.14 | 172.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 172.90 | 171.49 | 172.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 12:00:00 | 172.90 | 171.49 | 172.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 172.20 | 171.63 | 172.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 14:15:00 | 171.53 | 171.75 | 172.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 13:15:00 | 162.95 | 164.94 | 166.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-28 13:15:00 | 164.80 | 164.30 | 165.36 | SL hit (close>ema200) qty=0.50 sl=164.30 alert=retest2 |

### Cycle 101 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 166.99 | 165.88 | 165.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 15:15:00 | 167.70 | 166.78 | 166.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 09:15:00 | 166.04 | 166.63 | 166.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 09:15:00 | 166.04 | 166.63 | 166.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 166.04 | 166.63 | 166.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:00:00 | 166.04 | 166.63 | 166.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 166.04 | 166.51 | 166.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:30:00 | 165.93 | 166.51 | 166.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 11:15:00 | 166.50 | 166.51 | 166.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 11:30:00 | 166.64 | 166.51 | 166.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 167.20 | 166.65 | 166.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 13:15:00 | 168.00 | 166.65 | 166.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 10:45:00 | 167.34 | 167.38 | 166.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:45:00 | 167.39 | 167.32 | 166.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 12:15:00 | 167.35 | 167.32 | 166.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 166.90 | 167.21 | 166.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 13:30:00 | 166.85 | 167.21 | 166.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 168.11 | 167.39 | 167.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 14:30:00 | 166.81 | 167.39 | 167.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 166.00 | 167.13 | 167.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-05 09:15:00 | 166.00 | 167.13 | 167.00 | SL hit (close<static) qty=1.00 sl=166.40 alert=retest2 |

### Cycle 102 — SELL (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 10:15:00 | 165.90 | 166.88 | 166.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 14:15:00 | 165.31 | 166.15 | 166.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 14:15:00 | 164.40 | 164.38 | 165.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 14:15:00 | 164.40 | 164.38 | 165.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 164.40 | 164.38 | 165.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 15:00:00 | 164.40 | 164.38 | 165.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 165.00 | 164.50 | 165.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:15:00 | 164.68 | 164.50 | 165.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 165.20 | 164.64 | 165.22 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2024-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 15:15:00 | 166.65 | 165.62 | 165.50 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 164.50 | 165.40 | 165.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 163.68 | 165.05 | 165.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 13:15:00 | 163.51 | 162.75 | 163.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 13:15:00 | 163.51 | 162.75 | 163.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 163.51 | 162.75 | 163.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 13:45:00 | 163.35 | 162.75 | 163.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 162.79 | 162.76 | 163.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 15:15:00 | 163.75 | 162.76 | 163.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 163.75 | 162.95 | 163.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 163.40 | 162.95 | 163.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 164.10 | 163.18 | 163.41 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 10:15:00 | 165.50 | 163.65 | 163.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 11:15:00 | 169.60 | 164.84 | 164.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 171.99 | 174.26 | 172.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 09:15:00 | 171.99 | 174.26 | 172.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 171.99 | 174.26 | 172.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 171.99 | 174.26 | 172.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 172.25 | 173.86 | 172.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:45:00 | 172.64 | 173.86 | 172.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 173.00 | 173.69 | 172.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:30:00 | 172.38 | 173.69 | 172.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 175.90 | 174.65 | 173.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:45:00 | 175.67 | 174.65 | 173.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 172.46 | 174.07 | 173.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 172.46 | 174.07 | 173.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 175.40 | 174.34 | 173.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 15:00:00 | 181.00 | 177.53 | 175.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 10:00:00 | 180.05 | 178.61 | 176.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 13:15:00 | 175.51 | 177.65 | 177.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-07-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 13:15:00 | 175.51 | 177.65 | 177.69 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 12:15:00 | 179.05 | 177.81 | 177.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 14:15:00 | 180.01 | 178.52 | 178.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 09:15:00 | 180.60 | 180.69 | 179.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 10:00:00 | 180.60 | 180.69 | 179.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 179.35 | 180.27 | 179.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 179.35 | 180.27 | 179.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 179.45 | 180.10 | 179.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:15:00 | 178.35 | 180.10 | 179.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 09:15:00 | 176.65 | 179.41 | 179.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 11:15:00 | 175.30 | 178.13 | 178.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 11:15:00 | 175.39 | 175.35 | 176.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-02 12:00:00 | 175.39 | 175.35 | 176.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 12:15:00 | 177.37 | 175.75 | 176.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 12:45:00 | 177.29 | 175.75 | 176.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 13:15:00 | 177.13 | 176.03 | 176.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 09:15:00 | 170.09 | 176.55 | 176.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 14:15:00 | 180.29 | 174.99 | 175.58 | SL hit (close>static) qty=1.00 sl=177.99 alert=retest2 |

### Cycle 109 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 179.72 | 176.52 | 176.20 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 13:15:00 | 176.11 | 176.78 | 176.80 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 14:15:00 | 177.01 | 176.83 | 176.82 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 15:15:00 | 176.35 | 176.73 | 176.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 09:15:00 | 174.03 | 176.19 | 176.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 14:15:00 | 171.21 | 171.15 | 172.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-12 15:00:00 | 171.21 | 171.15 | 172.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 170.83 | 171.22 | 172.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 11:45:00 | 170.47 | 171.03 | 172.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 15:00:00 | 170.16 | 170.22 | 170.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 09:15:00 | 173.61 | 170.92 | 170.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 173.61 | 170.92 | 170.60 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2024-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 15:15:00 | 169.60 | 170.42 | 170.50 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 10:15:00 | 171.24 | 170.68 | 170.61 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 11:15:00 | 169.99 | 170.54 | 170.55 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 09:15:00 | 174.60 | 171.17 | 170.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 13:15:00 | 175.25 | 173.26 | 172.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 09:15:00 | 183.51 | 184.08 | 181.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-26 10:00:00 | 183.51 | 184.08 | 181.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 180.70 | 183.27 | 181.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:00:00 | 180.70 | 183.27 | 181.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 179.80 | 182.58 | 181.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:30:00 | 179.85 | 182.58 | 181.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 181.00 | 181.64 | 181.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:45:00 | 180.97 | 181.64 | 181.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 179.00 | 181.12 | 181.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 11:45:00 | 178.99 | 181.12 | 181.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 12:15:00 | 178.99 | 180.69 | 180.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 13:15:00 | 177.50 | 180.05 | 180.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 09:15:00 | 180.65 | 179.44 | 180.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 180.65 | 179.44 | 180.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 180.65 | 179.44 | 180.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:15:00 | 181.89 | 179.44 | 180.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 181.68 | 179.89 | 180.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:45:00 | 181.94 | 179.89 | 180.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 179.63 | 179.52 | 179.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 10:30:00 | 178.73 | 179.08 | 179.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 11:15:00 | 179.10 | 177.05 | 177.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 11:15:00 | 179.10 | 177.05 | 177.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 13:15:00 | 180.20 | 179.12 | 178.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-09 09:15:00 | 179.80 | 182.32 | 181.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 09:15:00 | 179.80 | 182.32 | 181.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 179.80 | 182.32 | 181.63 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2024-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 14:15:00 | 180.35 | 181.26 | 181.31 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 15:15:00 | 182.08 | 181.43 | 181.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 186.39 | 182.42 | 181.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 09:15:00 | 186.73 | 188.16 | 186.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 186.73 | 188.16 | 186.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 186.73 | 188.16 | 186.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:00:00 | 186.73 | 188.16 | 186.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 185.14 | 187.55 | 186.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:00:00 | 185.14 | 187.55 | 186.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 185.61 | 187.17 | 186.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:30:00 | 185.59 | 187.17 | 186.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 186.60 | 186.98 | 186.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 14:00:00 | 186.60 | 186.98 | 186.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 186.89 | 186.96 | 186.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 15:15:00 | 186.80 | 186.96 | 186.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 186.80 | 186.93 | 186.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 09:15:00 | 191.00 | 186.93 | 186.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 09:15:00 | 185.68 | 187.64 | 187.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 185.68 | 187.64 | 187.85 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2024-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 12:15:00 | 190.90 | 188.03 | 187.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 09:15:00 | 197.39 | 190.70 | 189.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 09:15:00 | 194.89 | 195.38 | 192.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-19 10:00:00 | 194.89 | 195.38 | 192.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 192.40 | 194.76 | 193.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:45:00 | 192.00 | 194.76 | 193.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 191.56 | 194.12 | 192.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:00:00 | 191.56 | 194.12 | 192.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 192.51 | 193.80 | 192.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 14:45:00 | 193.60 | 194.22 | 193.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-23 09:15:00 | 212.96 | 207.40 | 201.81 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 14:15:00 | 209.07 | 211.13 | 211.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 203.51 | 209.25 | 210.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 12:15:00 | 174.65 | 174.24 | 177.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 13:00:00 | 174.65 | 174.24 | 177.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 177.62 | 175.37 | 177.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 177.62 | 175.37 | 177.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 177.85 | 175.87 | 177.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 178.80 | 175.87 | 177.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 180.65 | 176.82 | 178.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 180.65 | 176.82 | 178.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 180.01 | 177.46 | 178.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 11:15:00 | 182.39 | 177.46 | 178.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2024-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 12:15:00 | 181.00 | 178.86 | 178.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 182.50 | 180.10 | 179.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 10:15:00 | 179.48 | 179.98 | 179.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 11:00:00 | 179.48 | 179.98 | 179.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 179.00 | 179.78 | 179.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:30:00 | 178.54 | 179.78 | 179.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 178.43 | 179.51 | 179.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:00:00 | 178.43 | 179.51 | 179.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 183.36 | 180.52 | 179.87 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 12:15:00 | 179.00 | 180.46 | 180.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 12:15:00 | 178.52 | 179.09 | 179.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 13:15:00 | 180.13 | 179.30 | 179.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 13:15:00 | 180.13 | 179.30 | 179.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 180.13 | 179.30 | 179.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:00:00 | 180.13 | 179.30 | 179.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 181.44 | 179.73 | 179.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 15:00:00 | 181.44 | 179.73 | 179.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2024-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 15:15:00 | 181.17 | 180.02 | 179.98 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 09:15:00 | 179.68 | 179.95 | 179.95 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 10:15:00 | 180.22 | 180.00 | 179.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 12:15:00 | 181.22 | 180.29 | 180.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 179.41 | 180.77 | 180.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 179.41 | 180.77 | 180.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 179.41 | 180.77 | 180.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:45:00 | 179.61 | 180.77 | 180.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 178.24 | 180.26 | 180.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 177.71 | 179.75 | 180.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 15:15:00 | 175.90 | 175.89 | 177.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-21 09:15:00 | 173.60 | 175.89 | 177.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 172.87 | 175.29 | 176.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:00:00 | 171.59 | 173.14 | 174.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 11:15:00 | 170.88 | 169.82 | 171.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 11:45:00 | 170.97 | 169.91 | 171.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 14:00:00 | 170.29 | 170.24 | 171.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 163.01 | 166.28 | 168.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 162.34 | 165.30 | 167.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 162.42 | 165.30 | 167.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 161.78 | 165.30 | 167.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 164.05 | 163.65 | 165.50 | SL hit (close>ema200) qty=0.50 sl=163.65 alert=retest2 |

### Cycle 131 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 168.01 | 165.26 | 165.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 14:15:00 | 168.59 | 166.90 | 165.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 15:15:00 | 168.25 | 168.66 | 167.59 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-01 18:00:00 | 170.99 | 169.13 | 167.90 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 167.73 | 169.17 | 168.15 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 167.73 | 169.17 | 168.15 | SL hit (close<ema400) qty=1.00 sl=168.15 alert=retest1 |

### Cycle 132 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 170.98 | 171.55 | 171.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 168.97 | 171.03 | 171.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 11:15:00 | 172.16 | 171.12 | 171.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 11:15:00 | 172.16 | 171.12 | 171.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 172.16 | 171.12 | 171.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 12:00:00 | 172.16 | 171.12 | 171.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 12:15:00 | 173.75 | 171.64 | 171.52 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 171.20 | 171.91 | 172.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 11:15:00 | 168.27 | 171.18 | 171.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 15:15:00 | 171.50 | 169.10 | 170.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 15:15:00 | 171.50 | 169.10 | 170.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 15:15:00 | 171.50 | 169.10 | 170.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 171.07 | 169.49 | 170.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 166.30 | 168.85 | 169.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 11:15:00 | 166.23 | 168.85 | 169.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 14:15:00 | 167.92 | 163.04 | 162.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 14:15:00 | 167.92 | 163.04 | 162.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 168.16 | 165.39 | 164.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 166.60 | 166.70 | 165.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 09:45:00 | 166.03 | 166.70 | 165.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 165.70 | 166.34 | 165.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 165.01 | 166.12 | 165.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 165.74 | 166.04 | 165.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:45:00 | 165.54 | 166.04 | 165.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 165.99 | 166.03 | 165.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:45:00 | 165.75 | 166.03 | 165.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 165.96 | 166.02 | 165.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 12:45:00 | 165.98 | 166.02 | 165.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 165.70 | 165.95 | 165.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 14:00:00 | 165.70 | 165.95 | 165.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 165.84 | 165.93 | 165.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 15:00:00 | 165.84 | 165.93 | 165.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 166.36 | 166.02 | 165.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:15:00 | 166.66 | 166.02 | 165.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 166.78 | 166.17 | 165.92 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 09:15:00 | 165.27 | 165.81 | 165.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 15:15:00 | 165.07 | 165.41 | 165.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 165.59 | 165.45 | 165.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 165.59 | 165.45 | 165.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 165.59 | 165.45 | 165.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:30:00 | 165.13 | 165.45 | 165.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 165.23 | 165.41 | 165.57 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2024-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 14:15:00 | 166.51 | 165.75 | 165.68 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2024-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 15:15:00 | 165.15 | 165.73 | 165.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 11:15:00 | 164.56 | 165.28 | 165.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 09:15:00 | 164.53 | 163.34 | 164.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 09:15:00 | 164.53 | 163.34 | 164.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 164.53 | 163.34 | 164.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:30:00 | 165.15 | 163.34 | 164.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 166.00 | 163.87 | 164.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:00:00 | 166.00 | 163.87 | 164.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2024-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 13:15:00 | 168.07 | 164.92 | 164.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 14:15:00 | 168.81 | 165.70 | 165.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 09:15:00 | 170.21 | 171.35 | 169.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-10 09:45:00 | 170.72 | 171.35 | 169.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 12:15:00 | 169.85 | 170.68 | 169.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 12:30:00 | 169.40 | 170.68 | 169.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 171.48 | 170.57 | 169.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 14:15:00 | 171.95 | 170.68 | 169.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 13:15:00 | 169.23 | 170.58 | 170.38 | SL hit (close<static) qty=1.00 sl=169.50 alert=retest2 |

### Cycle 140 — SELL (started 2024-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 15:15:00 | 169.80 | 170.21 | 170.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 168.66 | 169.90 | 170.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-17 09:15:00 | 166.79 | 166.30 | 167.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 09:15:00 | 166.79 | 166.30 | 167.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 166.79 | 166.30 | 167.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:00:00 | 166.79 | 166.30 | 167.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 166.68 | 166.38 | 167.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 11:00:00 | 166.68 | 166.38 | 167.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 166.89 | 166.47 | 167.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:30:00 | 167.15 | 166.47 | 167.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 165.98 | 166.37 | 167.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:45:00 | 166.00 | 166.37 | 167.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 170.88 | 167.28 | 167.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:30:00 | 171.49 | 167.28 | 167.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2024-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 15:15:00 | 171.44 | 168.11 | 167.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-20 09:15:00 | 173.30 | 171.00 | 170.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 12:15:00 | 171.60 | 171.83 | 170.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 13:00:00 | 171.60 | 171.83 | 170.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 172.40 | 171.95 | 170.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:45:00 | 173.70 | 171.95 | 170.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 181.51 | 179.36 | 177.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 10:45:00 | 176.89 | 179.36 | 177.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 185.21 | 188.10 | 185.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 13:00:00 | 185.21 | 188.10 | 185.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 182.63 | 187.00 | 185.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:00:00 | 182.63 | 187.00 | 185.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 10:15:00 | 180.85 | 184.02 | 184.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 11:15:00 | 180.10 | 183.24 | 183.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 14:15:00 | 182.48 | 182.34 | 183.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 14:15:00 | 182.48 | 182.34 | 183.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 182.48 | 182.34 | 183.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:30:00 | 182.92 | 182.34 | 183.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 184.57 | 182.76 | 183.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:00:00 | 184.57 | 182.76 | 183.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 183.75 | 182.96 | 183.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 11:30:00 | 183.08 | 183.03 | 183.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 12:15:00 | 182.85 | 183.03 | 183.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 13:45:00 | 182.42 | 182.92 | 183.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 09:15:00 | 181.85 | 183.03 | 183.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 181.96 | 182.81 | 183.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-02 11:15:00 | 186.41 | 183.48 | 183.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 11:15:00 | 186.41 | 183.48 | 183.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 189.75 | 185.36 | 184.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 13:15:00 | 190.62 | 190.62 | 187.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 13:30:00 | 191.35 | 190.62 | 187.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 190.87 | 196.57 | 193.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:00:00 | 190.87 | 196.57 | 193.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 200.34 | 197.33 | 194.49 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 189.45 | 193.55 | 193.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 184.23 | 187.55 | 189.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-13 09:15:00 | 183.79 | 183.04 | 185.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-13 10:00:00 | 183.79 | 183.04 | 185.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 179.60 | 179.06 | 180.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 180.50 | 179.06 | 180.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 184.57 | 180.16 | 181.31 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 187.50 | 182.49 | 182.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 12:15:00 | 195.76 | 185.14 | 183.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 09:15:00 | 184.77 | 188.03 | 185.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 09:15:00 | 184.77 | 188.03 | 185.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 184.77 | 188.03 | 185.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:00:00 | 184.77 | 188.03 | 185.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 184.55 | 187.33 | 185.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:45:00 | 184.86 | 187.33 | 185.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 11:15:00 | 185.33 | 186.93 | 185.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 12:15:00 | 187.00 | 186.93 | 185.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 14:15:00 | 183.80 | 186.09 | 185.53 | SL hit (close<static) qty=1.00 sl=184.27 alert=retest2 |

### Cycle 146 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 184.92 | 185.25 | 185.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 11:15:00 | 184.30 | 185.03 | 185.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 12:15:00 | 181.60 | 180.72 | 182.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 12:15:00 | 181.60 | 180.72 | 182.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 12:15:00 | 181.60 | 180.72 | 182.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 13:00:00 | 181.60 | 180.72 | 182.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 182.00 | 181.08 | 182.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 14:30:00 | 181.97 | 181.08 | 182.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 182.01 | 181.26 | 182.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 179.31 | 181.26 | 182.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 180.04 | 181.02 | 181.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:30:00 | 178.99 | 180.05 | 181.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:00:00 | 178.25 | 180.05 | 181.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 170.04 | 173.02 | 176.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 169.34 | 173.02 | 176.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 09:15:00 | 161.09 | 166.13 | 170.75 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 147 — BUY (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 14:15:00 | 168.85 | 167.36 | 167.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 169.58 | 168.00 | 167.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 09:15:00 | 182.75 | 184.10 | 179.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-04 09:45:00 | 183.00 | 184.10 | 179.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 12:15:00 | 180.14 | 182.66 | 181.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 13:00:00 | 180.14 | 182.66 | 181.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 13:15:00 | 180.70 | 182.27 | 181.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 14:30:00 | 181.31 | 182.11 | 181.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 09:30:00 | 183.35 | 182.34 | 181.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 13:15:00 | 179.70 | 181.57 | 181.52 | SL hit (close<static) qty=1.00 sl=180.06 alert=retest2 |

### Cycle 148 — SELL (started 2025-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 14:15:00 | 180.45 | 181.35 | 181.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 178.99 | 180.78 | 181.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 177.17 | 176.79 | 178.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 14:15:00 | 177.17 | 176.79 | 178.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 177.17 | 176.79 | 178.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 14:30:00 | 176.83 | 176.79 | 178.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 177.78 | 176.99 | 178.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:15:00 | 179.15 | 176.99 | 178.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 177.44 | 177.08 | 178.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:30:00 | 180.97 | 177.08 | 178.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 159.88 | 159.19 | 161.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 159.88 | 159.19 | 161.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 159.00 | 159.31 | 160.92 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 169.45 | 162.12 | 161.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 15:15:00 | 170.00 | 165.84 | 163.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 173.39 | 174.81 | 170.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 173.39 | 174.81 | 170.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 177.49 | 176.36 | 173.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 10:30:00 | 179.68 | 177.08 | 174.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 14:15:00 | 171.12 | 176.05 | 174.68 | SL hit (close<static) qty=1.00 sl=172.18 alert=retest2 |

### Cycle 150 — SELL (started 2025-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 12:15:00 | 172.45 | 174.84 | 174.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 168.55 | 172.86 | 173.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 168.39 | 165.00 | 167.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 14:15:00 | 168.39 | 165.00 | 167.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 168.39 | 165.00 | 167.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 15:00:00 | 168.39 | 165.00 | 167.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 168.01 | 165.60 | 167.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 165.17 | 165.60 | 167.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 10:00:00 | 167.22 | 166.13 | 166.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 11:45:00 | 167.99 | 166.61 | 166.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 14:15:00 | 170.86 | 167.80 | 167.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 170.86 | 167.80 | 167.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 174.64 | 169.55 | 168.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 14:15:00 | 171.70 | 172.19 | 170.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 15:00:00 | 171.70 | 172.19 | 170.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 171.00 | 171.95 | 170.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 09:15:00 | 171.93 | 171.95 | 170.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:15:00 | 171.83 | 171.72 | 170.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-07 11:15:00 | 169.20 | 171.13 | 170.38 | SL hit (close<static) qty=1.00 sl=170.26 alert=retest2 |

### Cycle 152 — SELL (started 2025-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 15:15:00 | 168.99 | 170.04 | 170.04 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 09:15:00 | 170.79 | 170.19 | 170.11 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 10:15:00 | 168.18 | 169.79 | 169.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 11:15:00 | 167.47 | 169.32 | 169.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 151.85 | 150.26 | 152.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 10:00:00 | 151.85 | 150.26 | 152.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 152.10 | 150.92 | 152.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 14:00:00 | 152.10 | 150.92 | 152.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 153.39 | 151.42 | 152.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 14:45:00 | 153.47 | 151.42 | 152.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 15:15:00 | 153.40 | 151.81 | 152.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 09:15:00 | 154.04 | 151.81 | 152.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2025-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 12:15:00 | 153.42 | 152.72 | 152.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 15:15:00 | 153.70 | 153.14 | 152.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 10:15:00 | 152.90 | 153.24 | 152.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 10:15:00 | 152.90 | 153.24 | 152.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 152.90 | 153.24 | 152.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 11:00:00 | 152.90 | 153.24 | 152.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 11:15:00 | 152.51 | 153.09 | 152.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 11:30:00 | 152.94 | 153.09 | 152.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 12:15:00 | 151.95 | 152.86 | 152.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 12:30:00 | 151.95 | 152.86 | 152.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2025-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-20 13:15:00 | 152.15 | 152.72 | 152.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-20 14:15:00 | 151.04 | 152.38 | 152.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 15:15:00 | 150.56 | 150.25 | 151.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-24 09:15:00 | 150.07 | 150.25 | 151.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 149.45 | 150.09 | 150.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-24 14:15:00 | 148.95 | 149.88 | 150.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 14:15:00 | 149.35 | 148.87 | 149.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 14:15:00 | 149.90 | 149.09 | 149.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 149.90 | 149.09 | 149.05 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 13:15:00 | 148.95 | 149.11 | 149.12 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 14:15:00 | 149.58 | 149.20 | 149.17 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 15:15:00 | 148.01 | 148.96 | 149.06 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 149.98 | 149.17 | 149.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 152.16 | 150.29 | 149.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 149.35 | 151.11 | 150.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 149.35 | 151.11 | 150.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 149.35 | 151.11 | 150.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 149.35 | 151.11 | 150.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 149.57 | 150.80 | 150.61 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 147.47 | 150.14 | 150.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 146.64 | 149.08 | 149.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 142.00 | 141.11 | 144.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 15:15:00 | 142.00 | 141.11 | 144.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 142.00 | 141.11 | 144.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 145.90 | 141.11 | 144.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 144.72 | 141.83 | 144.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:30:00 | 145.05 | 141.83 | 144.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 145.37 | 142.54 | 144.23 | EMA400 retest candle locked (from downside) |

### Cycle 163 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 146.31 | 145.21 | 145.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 153.43 | 147.73 | 146.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 172.59 | 173.68 | 171.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 10:00:00 | 172.59 | 173.68 | 171.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 175.05 | 173.95 | 171.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 172.95 | 173.95 | 171.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 174.99 | 177.09 | 175.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 174.99 | 177.09 | 175.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 173.24 | 176.32 | 175.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 173.24 | 176.32 | 175.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 173.21 | 175.70 | 174.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 12:45:00 | 174.22 | 175.57 | 174.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 15:15:00 | 176.21 | 177.04 | 177.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2025-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 15:15:00 | 176.21 | 177.04 | 177.04 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 177.99 | 177.23 | 177.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 10:15:00 | 178.45 | 177.47 | 177.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 177.11 | 177.99 | 177.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 177.11 | 177.99 | 177.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 177.11 | 177.99 | 177.69 | EMA400 retest candle locked (from upside) |

### Cycle 166 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 177.00 | 177.51 | 177.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 174.81 | 176.77 | 177.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 179.30 | 177.28 | 177.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 10:15:00 | 179.30 | 177.28 | 177.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 179.30 | 177.28 | 177.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 179.30 | 177.28 | 177.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 11:15:00 | 179.00 | 177.62 | 177.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 12:15:00 | 180.00 | 178.10 | 177.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 09:15:00 | 179.82 | 180.26 | 179.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 09:15:00 | 179.82 | 180.26 | 179.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 179.82 | 180.26 | 179.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:45:00 | 180.33 | 180.26 | 179.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 177.23 | 179.66 | 178.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:45:00 | 176.87 | 179.66 | 178.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 176.80 | 179.09 | 178.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:30:00 | 177.20 | 179.09 | 178.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 174.25 | 177.71 | 178.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 14:15:00 | 173.65 | 176.90 | 177.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 13:15:00 | 175.40 | 172.88 | 174.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 13:15:00 | 175.40 | 172.88 | 174.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 175.40 | 172.88 | 174.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 14:00:00 | 175.40 | 172.88 | 174.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 174.81 | 173.27 | 174.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:15:00 | 178.00 | 173.27 | 174.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 178.00 | 174.21 | 175.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 177.40 | 174.21 | 175.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 176.90 | 174.75 | 175.28 | EMA400 retest candle locked (from downside) |

### Cycle 169 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 176.26 | 175.67 | 175.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 176.93 | 176.28 | 175.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 176.21 | 176.27 | 175.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 10:30:00 | 176.32 | 176.27 | 175.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 176.07 | 176.23 | 176.00 | EMA400 retest candle locked (from upside) |

### Cycle 170 — SELL (started 2025-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 15:15:00 | 175.30 | 175.83 | 175.86 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 09:15:00 | 176.32 | 175.93 | 175.90 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 14:15:00 | 174.18 | 175.66 | 175.83 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 11:15:00 | 177.20 | 175.94 | 175.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 12:15:00 | 178.40 | 176.43 | 176.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 11:15:00 | 183.69 | 184.40 | 182.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 12:00:00 | 183.69 | 184.40 | 182.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 182.21 | 183.89 | 182.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 182.21 | 183.89 | 182.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 183.28 | 183.77 | 182.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:15:00 | 182.80 | 183.77 | 182.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 182.80 | 183.57 | 182.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 09:15:00 | 183.65 | 183.57 | 182.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 14:45:00 | 183.75 | 184.14 | 183.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:00:00 | 183.31 | 183.95 | 183.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 180.36 | 183.02 | 183.02 | SL hit (close<static) qty=1.00 sl=182.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 12:15:00 | 180.91 | 182.60 | 182.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 09:15:00 | 179.19 | 180.25 | 181.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 181.34 | 180.47 | 181.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 10:15:00 | 181.34 | 180.47 | 181.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 181.34 | 180.47 | 181.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 181.34 | 180.47 | 181.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 179.31 | 180.24 | 181.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:30:00 | 182.50 | 180.24 | 181.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 181.74 | 180.54 | 181.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:45:00 | 180.66 | 180.54 | 181.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 181.20 | 180.67 | 181.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 14:15:00 | 180.56 | 180.67 | 181.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 14:15:00 | 171.53 | 173.69 | 175.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 12:15:00 | 173.57 | 173.05 | 174.71 | SL hit (close>ema200) qty=0.50 sl=173.05 alert=retest2 |

### Cycle 175 — BUY (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 14:15:00 | 169.01 | 168.71 | 168.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 170.87 | 169.16 | 168.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 11:15:00 | 173.01 | 173.20 | 171.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 12:00:00 | 173.01 | 173.20 | 171.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 172.76 | 173.03 | 172.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 172.76 | 173.03 | 172.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 172.10 | 172.84 | 172.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 171.23 | 172.84 | 172.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 171.98 | 172.67 | 172.53 | EMA400 retest candle locked (from upside) |

### Cycle 176 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 171.00 | 172.17 | 172.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 12:15:00 | 169.96 | 171.72 | 172.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 167.00 | 166.09 | 167.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:45:00 | 167.03 | 166.09 | 167.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 166.67 | 166.35 | 167.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 168.77 | 166.35 | 167.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 167.76 | 166.63 | 167.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 167.67 | 166.63 | 167.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 167.75 | 166.86 | 167.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:00:00 | 167.75 | 166.86 | 167.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 168.58 | 167.20 | 167.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:00:00 | 168.58 | 167.20 | 167.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 13:15:00 | 169.59 | 167.68 | 167.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 15:15:00 | 170.00 | 168.50 | 168.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 170.00 | 170.17 | 169.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 11:00:00 | 170.00 | 170.17 | 169.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 169.32 | 170.00 | 169.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:30:00 | 169.21 | 170.00 | 169.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 167.65 | 169.53 | 169.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 167.65 | 169.53 | 169.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 168.67 | 169.36 | 169.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 14:15:00 | 169.90 | 169.36 | 169.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 11:15:00 | 168.14 | 169.03 | 169.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 168.14 | 169.03 | 169.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 14:15:00 | 166.53 | 168.15 | 168.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 14:15:00 | 167.38 | 166.51 | 167.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 14:15:00 | 167.38 | 166.51 | 167.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 167.38 | 166.51 | 167.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 15:00:00 | 167.38 | 166.51 | 167.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 167.00 | 166.61 | 167.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 169.52 | 166.61 | 167.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 168.95 | 167.08 | 167.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 11:00:00 | 168.17 | 167.29 | 167.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 14:15:00 | 169.10 | 167.83 | 167.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — BUY (started 2025-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 14:15:00 | 169.10 | 167.83 | 167.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 169.70 | 168.39 | 168.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 170.68 | 170.92 | 169.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:00:00 | 170.68 | 170.92 | 169.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 173.00 | 172.02 | 170.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:30:00 | 171.40 | 172.02 | 170.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 171.76 | 171.82 | 171.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 12:30:00 | 171.93 | 171.82 | 171.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 171.88 | 171.83 | 171.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 172.92 | 171.74 | 171.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 09:15:00 | 170.41 | 171.47 | 171.19 | SL hit (close<static) qty=1.00 sl=171.15 alert=retest2 |

### Cycle 180 — SELL (started 2025-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 11:15:00 | 169.23 | 170.73 | 170.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 12:15:00 | 167.92 | 170.17 | 170.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 14:15:00 | 168.68 | 168.01 | 168.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 14:15:00 | 168.68 | 168.01 | 168.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 168.68 | 168.01 | 168.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 15:00:00 | 168.68 | 168.01 | 168.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 168.84 | 168.17 | 168.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 168.27 | 168.17 | 168.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 167.27 | 167.99 | 168.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 12:00:00 | 166.63 | 167.56 | 168.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 15:15:00 | 166.60 | 165.97 | 166.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 09:15:00 | 172.91 | 167.46 | 167.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 172.91 | 167.46 | 167.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 10:15:00 | 174.35 | 172.22 | 170.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 13:15:00 | 172.24 | 172.67 | 171.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-07 14:00:00 | 172.24 | 172.67 | 171.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 173.64 | 172.77 | 171.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:45:00 | 173.39 | 172.77 | 171.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 172.00 | 172.52 | 171.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:30:00 | 171.60 | 172.52 | 171.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 171.56 | 172.33 | 171.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:45:00 | 171.00 | 172.33 | 171.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 171.49 | 172.16 | 171.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 171.49 | 172.16 | 171.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 171.50 | 172.03 | 171.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 174.49 | 172.03 | 171.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 11:00:00 | 172.05 | 172.03 | 171.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 14:15:00 | 171.06 | 171.50 | 171.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — SELL (started 2025-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 14:15:00 | 171.06 | 171.50 | 171.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 10:15:00 | 170.25 | 171.13 | 171.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 11:15:00 | 171.15 | 171.14 | 171.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 11:15:00 | 171.15 | 171.14 | 171.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 171.15 | 171.14 | 171.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:30:00 | 171.32 | 171.14 | 171.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 171.63 | 171.24 | 171.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 171.50 | 171.24 | 171.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 171.70 | 171.34 | 171.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:30:00 | 171.66 | 171.34 | 171.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 169.84 | 171.06 | 171.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:30:00 | 169.34 | 170.65 | 171.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 172.67 | 169.67 | 169.77 | SL hit (close>static) qty=1.00 sl=171.50 alert=retest2 |

### Cycle 183 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 172.33 | 170.20 | 170.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 174.10 | 171.38 | 170.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 12:15:00 | 173.88 | 173.98 | 172.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 13:00:00 | 173.88 | 173.98 | 172.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 173.34 | 173.81 | 172.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 173.83 | 173.81 | 172.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 11:15:00 | 174.58 | 175.02 | 175.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — SELL (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 11:15:00 | 174.58 | 175.02 | 175.05 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 13:15:00 | 175.30 | 175.07 | 175.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 14:15:00 | 177.45 | 175.55 | 175.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 15:15:00 | 175.50 | 175.54 | 175.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 09:15:00 | 175.36 | 175.54 | 175.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 177.44 | 175.92 | 175.50 | EMA400 retest candle locked (from upside) |

### Cycle 186 — SELL (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 15:15:00 | 175.00 | 175.30 | 175.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 173.13 | 174.49 | 174.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 163.03 | 162.62 | 163.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 163.03 | 162.62 | 163.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 163.03 | 162.62 | 163.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 163.03 | 162.62 | 163.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 163.49 | 162.84 | 163.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 163.49 | 162.84 | 163.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 163.87 | 163.05 | 163.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 163.87 | 163.05 | 163.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 164.01 | 163.24 | 163.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 164.01 | 163.24 | 163.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 163.90 | 163.37 | 163.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 163.00 | 163.37 | 163.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 161.50 | 163.00 | 163.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 159.66 | 161.85 | 162.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 14:45:00 | 160.38 | 159.98 | 161.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 14:15:00 | 160.02 | 159.94 | 160.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 15:15:00 | 152.36 | 154.09 | 155.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 09:15:00 | 151.68 | 153.51 | 154.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 09:15:00 | 152.02 | 153.51 | 154.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-13 13:15:00 | 156.15 | 153.36 | 154.18 | SL hit (close>ema200) qty=0.50 sl=153.36 alert=retest2 |

### Cycle 187 — BUY (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 15:15:00 | 159.03 | 155.40 | 155.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 10:15:00 | 165.74 | 158.21 | 156.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 09:15:00 | 172.80 | 173.25 | 170.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 09:30:00 | 172.67 | 173.25 | 170.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 170.82 | 171.90 | 170.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 170.61 | 171.90 | 170.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 170.00 | 171.52 | 170.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 170.51 | 171.52 | 170.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 170.81 | 171.38 | 170.66 | EMA400 retest candle locked (from upside) |

### Cycle 188 — SELL (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 15:15:00 | 169.49 | 170.34 | 170.38 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 09:15:00 | 172.65 | 170.80 | 170.58 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 169.47 | 170.41 | 170.45 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 13:15:00 | 172.05 | 170.74 | 170.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 14:15:00 | 174.60 | 171.51 | 170.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 09:15:00 | 173.67 | 174.44 | 173.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:00:00 | 173.67 | 174.44 | 173.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 174.89 | 174.53 | 173.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 175.13 | 174.37 | 173.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 13:00:00 | 176.30 | 174.72 | 174.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 09:30:00 | 175.66 | 174.91 | 174.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 10:45:00 | 175.14 | 174.92 | 174.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 178.12 | 175.56 | 174.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:30:00 | 179.66 | 176.28 | 175.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 10:15:00 | 183.49 | 185.61 | 185.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — SELL (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 10:15:00 | 183.49 | 185.61 | 185.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 12:15:00 | 181.90 | 183.44 | 184.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 14:15:00 | 177.70 | 176.49 | 177.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 14:15:00 | 177.70 | 176.49 | 177.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 177.70 | 176.49 | 177.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 177.70 | 176.49 | 177.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 179.10 | 177.02 | 178.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 180.87 | 177.02 | 178.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 178.66 | 177.34 | 178.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:30:00 | 182.45 | 177.34 | 178.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 177.13 | 177.30 | 178.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:30:00 | 176.58 | 177.03 | 177.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 180.10 | 176.55 | 176.76 | SL hit (close>static) qty=1.00 sl=178.80 alert=retest2 |

### Cycle 193 — BUY (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 10:15:00 | 178.89 | 177.02 | 176.96 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 175.10 | 176.67 | 176.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 10:15:00 | 174.89 | 176.31 | 176.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 12:15:00 | 170.63 | 169.10 | 171.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 13:00:00 | 170.63 | 169.10 | 171.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 170.33 | 169.35 | 170.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 169.31 | 169.78 | 170.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:00:00 | 169.33 | 169.69 | 170.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 168.30 | 168.82 | 169.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 13:45:00 | 169.73 | 169.40 | 169.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 171.33 | 169.79 | 169.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 171.33 | 169.79 | 169.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 170.90 | 170.01 | 169.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 170.90 | 170.01 | 169.91 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 09:15:00 | 168.87 | 169.78 | 169.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 10:15:00 | 168.06 | 169.44 | 169.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 09:15:00 | 166.44 | 166.27 | 167.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 166.44 | 166.27 | 167.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 166.44 | 166.27 | 167.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:30:00 | 167.01 | 166.27 | 167.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 163.39 | 164.54 | 165.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 11:30:00 | 162.50 | 163.85 | 164.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 12:00:00 | 162.80 | 163.86 | 164.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 14:00:00 | 162.70 | 163.49 | 164.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 14:30:00 | 162.71 | 163.22 | 163.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 163.35 | 163.13 | 163.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:00:00 | 162.30 | 162.97 | 163.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 10:30:00 | 162.26 | 161.66 | 162.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 15:15:00 | 164.00 | 162.87 | 162.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 164.00 | 162.87 | 162.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 12:15:00 | 165.90 | 163.95 | 163.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 11:15:00 | 166.50 | 167.18 | 166.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 12:00:00 | 166.50 | 167.18 | 166.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 165.75 | 166.89 | 166.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 12:30:00 | 165.76 | 166.89 | 166.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 165.82 | 166.68 | 166.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 166.30 | 166.26 | 166.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 165.50 | 166.38 | 166.34 | SL hit (close<static) qty=1.00 sl=165.51 alert=retest2 |

### Cycle 198 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 165.07 | 166.12 | 166.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 164.11 | 165.54 | 165.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 165.48 | 163.82 | 164.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 165.48 | 163.82 | 164.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 165.48 | 163.82 | 164.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 165.75 | 163.82 | 164.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 165.91 | 164.24 | 164.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:45:00 | 165.88 | 164.24 | 164.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — BUY (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 12:15:00 | 166.44 | 164.98 | 164.79 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 164.02 | 164.63 | 164.65 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 167.50 | 165.12 | 164.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 14:15:00 | 167.74 | 165.64 | 165.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 13:15:00 | 166.04 | 166.67 | 165.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 14:00:00 | 166.04 | 166.67 | 165.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 167.44 | 166.83 | 166.11 | EMA400 retest candle locked (from upside) |

### Cycle 202 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 161.61 | 165.37 | 165.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 161.37 | 164.57 | 165.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 163.10 | 163.03 | 164.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 10:00:00 | 163.10 | 163.03 | 164.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 161.96 | 162.45 | 163.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 14:15:00 | 161.88 | 162.45 | 163.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 14:45:00 | 161.42 | 162.18 | 163.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:45:00 | 161.08 | 161.82 | 162.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 12:15:00 | 161.62 | 161.78 | 162.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 158.44 | 159.26 | 160.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:45:00 | 159.20 | 159.26 | 160.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 153.79 | 156.36 | 158.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 153.35 | 156.36 | 158.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 153.03 | 156.36 | 158.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 153.54 | 156.36 | 158.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 11:15:00 | 150.32 | 149.97 | 153.01 | SL hit (close>ema200) qty=0.50 sl=149.97 alert=retest2 |

### Cycle 203 — BUY (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 15:15:00 | 142.70 | 141.09 | 140.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 11:15:00 | 143.58 | 141.62 | 141.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 13:15:00 | 144.62 | 145.28 | 144.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 14:00:00 | 144.62 | 145.28 | 144.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 148.00 | 145.95 | 144.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 147.15 | 145.95 | 144.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 146.59 | 145.99 | 145.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 14:15:00 | 147.00 | 145.99 | 145.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 144.30 | 145.98 | 145.33 | SL hit (close<static) qty=1.00 sl=144.75 alert=retest2 |

### Cycle 204 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 144.14 | 144.92 | 145.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 142.00 | 144.34 | 144.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 14:15:00 | 139.80 | 139.70 | 141.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 15:00:00 | 139.80 | 139.70 | 141.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 141.15 | 139.99 | 140.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 141.15 | 139.99 | 140.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 140.85 | 140.16 | 140.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 142.17 | 140.16 | 140.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 140.14 | 140.16 | 140.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 13:45:00 | 139.63 | 140.02 | 140.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 15:00:00 | 139.15 | 139.85 | 140.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:30:00 | 138.98 | 139.55 | 140.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 09:15:00 | 132.65 | 134.15 | 135.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 09:15:00 | 132.19 | 134.15 | 135.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 09:15:00 | 132.03 | 134.15 | 135.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 135.00 | 133.84 | 134.67 | SL hit (close>ema200) qty=0.50 sl=133.84 alert=retest2 |

### Cycle 205 — BUY (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 14:15:00 | 136.83 | 135.25 | 135.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 14:15:00 | 138.49 | 135.89 | 135.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 09:15:00 | 134.58 | 135.85 | 135.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 134.58 | 135.85 | 135.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 134.58 | 135.85 | 135.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 134.58 | 135.85 | 135.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 136.70 | 136.02 | 135.66 | EMA400 retest candle locked (from upside) |

### Cycle 206 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 134.50 | 135.65 | 135.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 133.89 | 135.06 | 135.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 134.21 | 134.17 | 134.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 134.21 | 134.17 | 134.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 134.78 | 134.29 | 134.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 134.78 | 134.29 | 134.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 135.24 | 134.48 | 134.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:45:00 | 135.09 | 134.48 | 134.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 134.98 | 134.58 | 134.79 | EMA400 retest candle locked (from downside) |

### Cycle 207 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 135.20 | 134.92 | 134.91 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 14:15:00 | 134.70 | 134.90 | 134.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 10:15:00 | 134.53 | 134.77 | 134.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 134.68 | 134.61 | 134.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 134.68 | 134.61 | 134.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 134.68 | 134.61 | 134.72 | EMA400 retest candle locked (from downside) |

### Cycle 209 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 134.89 | 134.78 | 134.78 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 14:15:00 | 134.50 | 134.74 | 134.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 15:15:00 | 134.00 | 134.60 | 134.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 126.56 | 126.43 | 127.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 126.56 | 126.43 | 127.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 126.56 | 126.43 | 127.68 | EMA400 retest candle locked (from downside) |

### Cycle 211 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 131.00 | 128.19 | 128.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 132.31 | 129.32 | 128.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 12:15:00 | 143.97 | 144.20 | 141.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 13:00:00 | 143.97 | 144.20 | 141.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 142.50 | 144.10 | 142.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:15:00 | 144.91 | 143.38 | 142.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 14:15:00 | 142.11 | 142.96 | 142.80 | SL hit (close<static) qty=1.00 sl=142.21 alert=retest2 |

### Cycle 212 — SELL (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 11:15:00 | 143.74 | 146.30 | 146.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 140.63 | 144.66 | 145.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 135.15 | 133.57 | 135.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 13:00:00 | 135.15 | 133.57 | 135.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 135.82 | 134.02 | 135.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:45:00 | 136.09 | 134.02 | 135.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 135.82 | 134.38 | 135.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 135.76 | 134.38 | 135.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 134.28 | 134.06 | 134.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:45:00 | 134.65 | 134.06 | 134.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 135.05 | 134.26 | 134.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:45:00 | 135.08 | 134.26 | 134.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 135.19 | 134.44 | 134.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 135.39 | 134.44 | 134.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 135.01 | 134.56 | 134.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:45:00 | 135.29 | 134.56 | 134.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 134.70 | 134.59 | 134.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 134.70 | 134.59 | 134.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 134.45 | 134.56 | 134.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:30:00 | 134.80 | 134.56 | 134.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 134.87 | 134.62 | 134.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:00:00 | 134.87 | 134.62 | 134.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 134.97 | 134.69 | 134.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:30:00 | 134.82 | 134.69 | 134.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 134.62 | 134.68 | 134.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:30:00 | 135.14 | 134.68 | 134.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 134.68 | 134.68 | 134.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:15:00 | 133.59 | 134.68 | 134.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 133.43 | 134.43 | 134.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:00:00 | 132.40 | 134.02 | 134.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:30:00 | 132.75 | 133.60 | 134.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:00:00 | 132.75 | 133.60 | 134.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 14:15:00 | 132.83 | 133.46 | 134.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 126.11 | 128.93 | 130.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 126.11 | 128.93 | 130.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 126.19 | 128.93 | 130.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 125.78 | 127.36 | 129.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 12:15:00 | 126.64 | 126.48 | 128.17 | SL hit (close>ema200) qty=0.50 sl=126.48 alert=retest2 |

### Cycle 213 — BUY (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 12:15:00 | 116.05 | 114.26 | 114.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 14:15:00 | 116.76 | 115.14 | 114.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 116.61 | 116.90 | 116.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 15:15:00 | 116.61 | 116.90 | 116.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 116.61 | 116.90 | 116.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 114.44 | 116.90 | 116.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 114.39 | 116.40 | 115.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 114.39 | 116.40 | 115.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 114.08 | 115.93 | 115.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:30:00 | 113.85 | 115.93 | 115.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 214 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 114.10 | 115.31 | 115.44 | EMA200 below EMA400 |

### Cycle 215 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 116.80 | 115.60 | 115.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 117.11 | 116.04 | 115.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 14:15:00 | 116.38 | 116.42 | 116.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 14:15:00 | 116.38 | 116.42 | 116.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 116.38 | 116.42 | 116.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-03 14:45:00 | 116.19 | 116.42 | 116.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 116.30 | 116.40 | 116.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 118.48 | 116.40 | 116.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-05 09:15:00 | 130.33 | 123.66 | 120.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 216 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 131.72 | 133.72 | 133.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 130.29 | 132.12 | 132.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 131.55 | 130.25 | 131.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 131.55 | 130.25 | 131.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 131.55 | 130.25 | 131.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 131.55 | 130.25 | 131.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 131.24 | 130.45 | 131.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 15:15:00 | 131.00 | 131.18 | 131.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 133.24 | 131.57 | 131.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 217 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 133.24 | 131.57 | 131.53 | EMA200 above EMA400 |

### Cycle 218 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 131.00 | 131.84 | 131.88 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 12:15:00 | 132.49 | 131.97 | 131.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 14:15:00 | 133.16 | 132.29 | 132.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 09:15:00 | 131.82 | 132.31 | 132.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 131.82 | 132.31 | 132.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 131.82 | 132.31 | 132.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:00:00 | 131.82 | 132.31 | 132.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 132.04 | 132.25 | 132.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:45:00 | 132.61 | 132.25 | 132.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 132.79 | 132.36 | 132.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:30:00 | 132.09 | 132.36 | 132.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 133.09 | 132.51 | 132.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:45:00 | 132.23 | 132.51 | 132.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 132.41 | 132.69 | 132.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:00:00 | 132.41 | 132.69 | 132.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 132.83 | 132.72 | 132.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:45:00 | 132.69 | 132.72 | 132.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 132.46 | 132.67 | 132.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:45:00 | 132.18 | 132.67 | 132.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 133.27 | 132.79 | 132.56 | EMA400 retest candle locked (from upside) |

### Cycle 220 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 130.37 | 132.45 | 132.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 129.58 | 131.23 | 131.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 11:15:00 | 131.10 | 130.33 | 131.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 11:15:00 | 131.10 | 130.33 | 131.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 131.10 | 130.33 | 131.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:45:00 | 131.31 | 130.33 | 131.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 132.00 | 130.66 | 131.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 132.00 | 130.66 | 131.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 131.21 | 130.77 | 131.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 130.59 | 130.97 | 131.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 10:45:00 | 130.75 | 131.05 | 131.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 124.06 | 127.23 | 128.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 124.21 | 127.23 | 128.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-05 09:15:00 | 117.67 | 119.64 | 122.26 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 221 — BUY (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 10:15:00 | 116.70 | 112.84 | 112.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 11:15:00 | 117.60 | 113.79 | 113.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 112.24 | 115.20 | 114.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 112.24 | 115.20 | 114.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 112.24 | 115.20 | 114.27 | EMA400 retest candle locked (from upside) |

### Cycle 222 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 109.51 | 113.02 | 113.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 105.51 | 110.44 | 111.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 12:15:00 | 110.37 | 107.77 | 109.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 12:15:00 | 110.37 | 107.77 | 109.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 110.37 | 107.77 | 109.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:45:00 | 110.58 | 107.77 | 109.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 108.79 | 107.98 | 109.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 107.89 | 107.98 | 109.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 106.55 | 108.93 | 109.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 102.50 | 103.88 | 105.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 101.22 | 103.88 | 105.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 15:15:00 | 101.60 | 101.25 | 103.05 | SL hit (close>ema200) qty=0.50 sl=101.25 alert=retest2 |

### Cycle 223 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 104.22 | 102.94 | 102.89 | EMA200 above EMA400 |

### Cycle 224 — SELL (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 12:15:00 | 102.10 | 102.79 | 102.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 09:15:00 | 99.88 | 101.94 | 102.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 96.83 | 96.34 | 98.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 12:15:00 | 97.83 | 96.64 | 97.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 97.83 | 96.64 | 97.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:00:00 | 97.83 | 96.64 | 97.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 97.43 | 96.80 | 97.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 93.10 | 96.86 | 97.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 13:45:00 | 95.93 | 95.35 | 96.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 14:30:00 | 96.25 | 95.71 | 96.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 95.01 | 95.94 | 96.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 97.39 | 96.21 | 96.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:30:00 | 98.55 | 96.21 | 96.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 99.87 | 96.94 | 96.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 225 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 99.87 | 96.94 | 96.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 105.40 | 100.25 | 98.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 104.50 | 104.61 | 102.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 104.50 | 104.61 | 102.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 106.52 | 107.87 | 106.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:30:00 | 107.21 | 107.11 | 106.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 107.96 | 106.91 | 106.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 10:30:00 | 107.30 | 107.48 | 107.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 09:45:00 | 107.25 | 107.91 | 107.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 107.90 | 107.91 | 107.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 11:45:00 | 108.40 | 107.89 | 107.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 12:45:00 | 108.65 | 107.89 | 107.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 13:15:00 | 106.92 | 107.70 | 107.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 226 — SELL (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 13:15:00 | 106.92 | 107.70 | 107.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 14:15:00 | 106.64 | 107.49 | 107.64 | Break + close below crossover candle low |

### Cycle 227 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 109.78 | 107.82 | 107.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 13:15:00 | 110.43 | 109.10 | 108.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 09:15:00 | 112.15 | 112.66 | 111.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 10:15:00 | 111.53 | 112.44 | 111.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 111.53 | 112.44 | 111.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:45:00 | 111.26 | 112.44 | 111.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 110.98 | 112.15 | 111.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:45:00 | 110.85 | 112.15 | 111.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 111.26 | 111.97 | 111.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 13:15:00 | 111.74 | 111.97 | 111.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 14:15:00 | 110.20 | 111.47 | 111.15 | SL hit (close<static) qty=1.00 sl=110.86 alert=retest2 |

### Cycle 228 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 109.06 | 110.80 | 110.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 107.40 | 110.12 | 110.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 112.41 | 108.99 | 109.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 112.41 | 108.99 | 109.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 112.41 | 108.99 | 109.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 113.05 | 108.99 | 109.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 112.16 | 109.62 | 109.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 112.16 | 109.62 | 109.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 229 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 113.05 | 110.31 | 110.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 113.35 | 110.92 | 110.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 110.88 | 111.31 | 110.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 09:15:00 | 110.88 | 111.31 | 110.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 110.88 | 111.31 | 110.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 10:15:00 | 112.18 | 111.31 | 110.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 10:45:00 | 112.31 | 111.47 | 110.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 11:45:00 | 112.16 | 111.58 | 111.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 13:15:00 | 112.03 | 111.65 | 111.12 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 112.05 | 111.73 | 111.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:45:00 | 111.50 | 111.73 | 111.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 112.40 | 112.02 | 111.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 111.52 | 112.02 | 111.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Target hit | 2026-04-29 13:15:00 | 123.40 | 116.94 | 114.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 230 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 118.46 | 120.05 | 120.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 15:15:00 | 117.85 | 119.34 | 119.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 14:15:00 | 121.08 | 118.97 | 119.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 14:15:00 | 121.08 | 118.97 | 119.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 121.08 | 118.97 | 119.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:00:00 | 121.08 | 118.97 | 119.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 120.51 | 119.28 | 119.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 120.87 | 119.28 | 119.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 118.93 | 119.41 | 119.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 09:30:00 | 118.22 | 118.92 | 119.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 12:00:00 | 118.35 | 118.85 | 119.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 12:30:00 | 118.18 | 118.73 | 119.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-23 15:15:00 | 176.10 | 2023-05-24 09:15:00 | 181.00 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2023-05-30 09:30:00 | 185.50 | 2023-05-30 10:15:00 | 183.20 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2023-06-01 09:30:00 | 183.45 | 2023-06-01 11:15:00 | 185.15 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-06-16 09:15:00 | 191.60 | 2023-06-21 11:15:00 | 191.00 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2023-06-16 10:30:00 | 190.70 | 2023-06-21 11:15:00 | 191.00 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2023-07-10 09:15:00 | 193.05 | 2023-07-10 12:15:00 | 189.90 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2023-07-10 12:15:00 | 193.20 | 2023-07-10 12:15:00 | 189.90 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2023-07-17 09:30:00 | 195.30 | 2023-07-18 09:15:00 | 191.75 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2023-07-21 11:45:00 | 198.60 | 2023-07-24 11:15:00 | 196.10 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2023-07-21 14:15:00 | 198.55 | 2023-07-24 11:15:00 | 196.10 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2023-07-27 09:15:00 | 196.50 | 2023-07-27 09:15:00 | 195.70 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2023-08-03 09:30:00 | 191.95 | 2023-08-04 09:15:00 | 195.65 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2023-08-03 12:45:00 | 192.95 | 2023-08-04 09:15:00 | 195.65 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2023-08-03 13:15:00 | 192.95 | 2023-08-04 09:15:00 | 195.65 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2023-08-03 14:45:00 | 193.00 | 2023-08-04 09:15:00 | 195.65 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2023-08-17 11:00:00 | 201.00 | 2023-08-18 11:15:00 | 196.90 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2023-08-23 13:00:00 | 196.00 | 2023-08-30 11:15:00 | 193.80 | STOP_HIT | 1.00 | 1.12% |
| SELL | retest2 | 2023-08-23 13:30:00 | 196.00 | 2023-08-30 11:15:00 | 193.80 | STOP_HIT | 1.00 | 1.12% |
| SELL | retest2 | 2023-08-23 14:00:00 | 195.80 | 2023-08-30 11:15:00 | 193.80 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2023-08-23 15:00:00 | 195.30 | 2023-08-30 11:15:00 | 193.80 | STOP_HIT | 1.00 | 0.77% |
| SELL | retest2 | 2023-08-24 10:45:00 | 194.65 | 2023-08-30 11:15:00 | 193.80 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2023-08-24 11:15:00 | 194.75 | 2023-08-30 11:15:00 | 193.80 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2023-08-24 12:00:00 | 194.75 | 2023-08-30 11:15:00 | 193.80 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2023-08-24 13:30:00 | 194.70 | 2023-08-30 11:15:00 | 193.80 | STOP_HIT | 1.00 | 0.46% |
| SELL | retest2 | 2023-08-29 15:15:00 | 191.45 | 2023-08-30 11:15:00 | 193.80 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2023-09-01 09:15:00 | 197.00 | 2023-09-05 10:15:00 | 216.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-09-15 12:15:00 | 210.45 | 2023-09-26 10:15:00 | 209.50 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2023-09-18 09:15:00 | 210.40 | 2023-09-26 10:15:00 | 209.50 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2023-09-18 10:45:00 | 210.70 | 2023-09-26 10:15:00 | 209.50 | STOP_HIT | 1.00 | 0.57% |
| SELL | retest2 | 2023-09-18 11:30:00 | 210.60 | 2023-09-26 10:15:00 | 209.50 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2023-09-20 09:15:00 | 209.70 | 2023-09-26 10:15:00 | 209.50 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2023-09-20 10:00:00 | 209.80 | 2023-09-26 10:15:00 | 209.50 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2023-09-20 12:30:00 | 209.95 | 2023-09-26 10:15:00 | 209.50 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2023-09-20 13:00:00 | 209.85 | 2023-09-26 10:15:00 | 209.50 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2023-10-05 09:30:00 | 221.50 | 2023-10-09 13:15:00 | 216.50 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2023-10-06 10:45:00 | 221.50 | 2023-10-09 13:15:00 | 216.50 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2023-10-06 12:15:00 | 221.70 | 2023-10-09 13:15:00 | 216.50 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2023-10-17 12:45:00 | 210.50 | 2023-10-19 09:15:00 | 199.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-17 13:30:00 | 210.40 | 2023-10-19 09:15:00 | 199.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-17 12:45:00 | 210.50 | 2023-10-20 15:15:00 | 197.25 | STOP_HIT | 0.50 | 6.29% |
| SELL | retest2 | 2023-10-17 13:30:00 | 210.40 | 2023-10-20 15:15:00 | 197.25 | STOP_HIT | 0.50 | 6.25% |
| BUY | retest2 | 2023-11-06 13:45:00 | 189.65 | 2023-11-08 11:15:00 | 184.65 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2023-11-13 09:15:00 | 184.00 | 2023-11-13 11:15:00 | 183.60 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2023-11-13 10:15:00 | 184.50 | 2023-11-13 11:15:00 | 183.60 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2023-11-28 09:30:00 | 181.65 | 2023-12-04 12:15:00 | 178.00 | STOP_HIT | 1.00 | 2.01% |
| BUY | retest2 | 2023-12-11 14:15:00 | 186.90 | 2023-12-12 15:15:00 | 185.10 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2023-12-12 09:15:00 | 188.05 | 2023-12-12 15:15:00 | 185.10 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2023-12-14 10:30:00 | 183.25 | 2023-12-15 12:15:00 | 184.95 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2023-12-14 14:00:00 | 183.50 | 2023-12-15 12:15:00 | 184.95 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2023-12-29 11:15:00 | 191.40 | 2024-01-02 09:15:00 | 190.35 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-01-04 09:15:00 | 190.20 | 2024-01-11 10:15:00 | 189.35 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2024-01-29 09:15:00 | 175.55 | 2024-01-31 09:15:00 | 182.05 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2024-02-12 11:45:00 | 150.70 | 2024-02-14 14:15:00 | 154.20 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2024-02-13 09:15:00 | 148.10 | 2024-02-14 14:15:00 | 154.20 | STOP_HIT | 1.00 | -4.12% |
| SELL | retest2 | 2024-02-13 12:00:00 | 149.85 | 2024-02-14 14:15:00 | 154.20 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2024-02-13 14:30:00 | 150.80 | 2024-02-14 14:15:00 | 154.20 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2024-02-27 09:15:00 | 157.05 | 2024-03-02 11:15:00 | 157.90 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-03-05 14:15:00 | 153.65 | 2024-03-07 09:15:00 | 156.15 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-03-05 15:00:00 | 154.50 | 2024-03-07 09:15:00 | 156.15 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-03-06 09:15:00 | 154.30 | 2024-03-07 09:15:00 | 156.15 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-03-19 12:30:00 | 156.10 | 2024-03-22 15:15:00 | 155.25 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-03-20 11:00:00 | 156.80 | 2024-03-22 15:15:00 | 155.25 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-03-21 11:45:00 | 156.10 | 2024-03-22 15:15:00 | 155.25 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-03-21 12:30:00 | 156.15 | 2024-03-22 15:15:00 | 155.25 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2024-03-28 12:30:00 | 152.80 | 2024-04-01 09:15:00 | 154.30 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-04-08 14:30:00 | 159.20 | 2024-04-15 10:15:00 | 159.40 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2024-05-03 11:45:00 | 164.50 | 2024-05-13 10:15:00 | 156.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 12:45:00 | 164.30 | 2024-05-13 10:15:00 | 156.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 13:30:00 | 164.20 | 2024-05-13 10:15:00 | 155.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-06 09:30:00 | 164.30 | 2024-05-13 10:15:00 | 156.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-07 14:15:00 | 161.85 | 2024-05-14 09:15:00 | 153.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 11:45:00 | 164.50 | 2024-05-14 14:15:00 | 157.30 | STOP_HIT | 0.50 | 4.38% |
| SELL | retest2 | 2024-05-03 12:45:00 | 164.30 | 2024-05-14 14:15:00 | 157.30 | STOP_HIT | 0.50 | 4.26% |
| SELL | retest2 | 2024-05-03 13:30:00 | 164.20 | 2024-05-14 14:15:00 | 157.30 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2024-05-06 09:30:00 | 164.30 | 2024-05-14 14:15:00 | 157.30 | STOP_HIT | 0.50 | 4.26% |
| SELL | retest2 | 2024-05-07 14:15:00 | 161.85 | 2024-05-14 14:15:00 | 157.30 | STOP_HIT | 0.50 | 2.81% |
| BUY | retest2 | 2024-06-13 11:15:00 | 178.51 | 2024-06-18 09:15:00 | 173.02 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2024-06-14 11:00:00 | 178.27 | 2024-06-18 09:15:00 | 173.02 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2024-06-14 12:30:00 | 178.31 | 2024-06-18 09:15:00 | 173.02 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2024-06-14 13:45:00 | 178.25 | 2024-06-18 09:15:00 | 173.02 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2024-06-21 14:15:00 | 171.53 | 2024-06-27 13:15:00 | 162.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-21 14:15:00 | 171.53 | 2024-06-28 13:15:00 | 164.80 | STOP_HIT | 0.50 | 3.92% |
| BUY | retest2 | 2024-07-03 13:15:00 | 168.00 | 2024-07-05 09:15:00 | 166.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-07-04 10:45:00 | 167.34 | 2024-07-05 09:15:00 | 166.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-07-04 11:45:00 | 167.39 | 2024-07-05 09:15:00 | 166.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-07-04 12:15:00 | 167.35 | 2024-07-05 09:15:00 | 166.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-07-24 15:00:00 | 181.00 | 2024-07-26 13:15:00 | 175.51 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2024-07-25 10:00:00 | 180.05 | 2024-07-26 13:15:00 | 175.51 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2024-08-05 09:15:00 | 170.09 | 2024-08-05 14:15:00 | 180.29 | STOP_HIT | 1.00 | -6.00% |
| SELL | retest2 | 2024-08-06 09:15:00 | 176.42 | 2024-08-06 11:15:00 | 179.72 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-08-13 11:45:00 | 170.47 | 2024-08-19 09:15:00 | 173.61 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-08-16 15:00:00 | 170.16 | 2024-08-19 09:15:00 | 173.61 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-08-29 10:30:00 | 178.73 | 2024-09-03 11:15:00 | 179.10 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2024-09-13 09:15:00 | 191.00 | 2024-09-17 09:15:00 | 185.68 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2024-09-19 14:45:00 | 193.60 | 2024-09-23 09:15:00 | 212.96 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-22 10:00:00 | 171.59 | 2024-10-25 09:15:00 | 163.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 11:15:00 | 170.88 | 2024-10-25 10:15:00 | 162.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 11:45:00 | 170.97 | 2024-10-25 10:15:00 | 162.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 14:00:00 | 170.29 | 2024-10-25 10:15:00 | 161.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 10:00:00 | 171.59 | 2024-10-28 10:15:00 | 164.05 | STOP_HIT | 0.50 | 4.39% |
| SELL | retest2 | 2024-10-23 11:15:00 | 170.88 | 2024-10-28 10:15:00 | 164.05 | STOP_HIT | 0.50 | 4.00% |
| SELL | retest2 | 2024-10-23 11:45:00 | 170.97 | 2024-10-28 10:15:00 | 164.05 | STOP_HIT | 0.50 | 4.05% |
| SELL | retest2 | 2024-10-23 14:00:00 | 170.29 | 2024-10-28 10:15:00 | 164.05 | STOP_HIT | 0.50 | 3.66% |
| SELL | retest2 | 2024-10-29 12:30:00 | 163.32 | 2024-10-30 09:15:00 | 167.32 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2024-10-29 14:45:00 | 163.76 | 2024-10-30 09:15:00 | 167.32 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2024-10-29 15:15:00 | 163.51 | 2024-10-30 09:15:00 | 167.32 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest1 | 2024-11-01 18:00:00 | 170.99 | 2024-11-04 09:15:00 | 167.73 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-11-05 12:15:00 | 171.00 | 2024-11-08 13:15:00 | 170.98 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2024-11-05 14:45:00 | 171.47 | 2024-11-08 13:15:00 | 170.98 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-11-08 09:45:00 | 171.00 | 2024-11-08 13:15:00 | 170.98 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2024-11-08 11:30:00 | 171.12 | 2024-11-08 13:15:00 | 170.98 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2024-11-14 11:15:00 | 166.23 | 2024-11-22 14:15:00 | 167.92 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-12-11 14:15:00 | 171.95 | 2024-12-12 13:15:00 | 169.23 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-01-01 11:30:00 | 183.08 | 2025-01-02 11:15:00 | 186.41 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-01-01 12:15:00 | 182.85 | 2025-01-02 11:15:00 | 186.41 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-01-01 13:45:00 | 182.42 | 2025-01-02 11:15:00 | 186.41 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-01-02 09:15:00 | 181.85 | 2025-01-02 11:15:00 | 186.41 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-01-16 12:15:00 | 187.00 | 2025-01-16 14:15:00 | 183.80 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-01-23 14:30:00 | 178.99 | 2025-01-27 09:15:00 | 170.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 15:00:00 | 178.25 | 2025-01-27 09:15:00 | 169.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 14:30:00 | 178.99 | 2025-01-28 09:15:00 | 161.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-23 15:00:00 | 178.25 | 2025-01-28 09:15:00 | 160.43 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-05 14:30:00 | 181.31 | 2025-02-06 13:15:00 | 179.70 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-02-06 09:30:00 | 183.35 | 2025-02-06 13:15:00 | 179.70 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-02-24 10:30:00 | 179.68 | 2025-02-24 14:15:00 | 171.12 | STOP_HIT | 1.00 | -4.76% |
| BUY | retest2 | 2025-02-25 09:45:00 | 177.99 | 2025-02-27 12:15:00 | 172.45 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2025-02-25 14:45:00 | 178.00 | 2025-02-27 12:15:00 | 172.45 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-03-04 09:15:00 | 165.17 | 2025-03-05 14:15:00 | 170.86 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2025-03-05 10:00:00 | 167.22 | 2025-03-05 14:15:00 | 170.86 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-03-05 11:45:00 | 167.99 | 2025-03-05 14:15:00 | 170.86 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-03-07 09:15:00 | 171.93 | 2025-03-07 11:15:00 | 169.20 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-03-07 10:15:00 | 171.83 | 2025-03-07 11:15:00 | 169.20 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-03-24 14:15:00 | 148.95 | 2025-03-27 14:15:00 | 149.90 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-03-26 14:15:00 | 149.35 | 2025-03-27 14:15:00 | 149.90 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-04-25 12:45:00 | 174.22 | 2025-05-02 15:15:00 | 176.21 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2025-05-20 09:15:00 | 183.65 | 2025-05-21 11:15:00 | 180.36 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-05-20 14:45:00 | 183.75 | 2025-05-21 11:15:00 | 180.36 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-05-21 10:00:00 | 183.31 | 2025-05-21 11:15:00 | 180.36 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-05-23 14:15:00 | 180.56 | 2025-05-27 14:15:00 | 171.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-23 14:15:00 | 180.56 | 2025-05-28 12:15:00 | 173.57 | STOP_HIT | 0.50 | 3.87% |
| BUY | retest2 | 2025-06-19 14:15:00 | 169.90 | 2025-06-20 11:15:00 | 168.14 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-06-24 11:00:00 | 168.17 | 2025-06-24 14:15:00 | 169.10 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-06-30 09:15:00 | 172.92 | 2025-06-30 09:15:00 | 170.41 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-07-02 12:00:00 | 166.63 | 2025-07-04 09:15:00 | 172.91 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2025-07-03 15:15:00 | 166.60 | 2025-07-04 09:15:00 | 172.91 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2025-07-09 09:15:00 | 174.49 | 2025-07-09 14:15:00 | 171.06 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-07-09 11:00:00 | 172.05 | 2025-07-09 14:15:00 | 171.06 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-07-11 10:30:00 | 169.34 | 2025-07-15 09:15:00 | 172.67 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-07-17 09:15:00 | 173.83 | 2025-07-22 11:15:00 | 174.58 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-08-06 09:15:00 | 159.66 | 2025-08-12 15:15:00 | 152.36 | PARTIAL | 0.50 | 4.57% |
| SELL | retest2 | 2025-08-06 14:45:00 | 160.38 | 2025-08-13 09:15:00 | 151.68 | PARTIAL | 0.50 | 5.43% |
| SELL | retest2 | 2025-08-07 14:15:00 | 160.02 | 2025-08-13 09:15:00 | 152.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-06 09:15:00 | 159.66 | 2025-08-13 13:15:00 | 156.15 | STOP_HIT | 0.50 | 2.20% |
| SELL | retest2 | 2025-08-06 14:45:00 | 160.38 | 2025-08-13 13:15:00 | 156.15 | STOP_HIT | 0.50 | 2.64% |
| SELL | retest2 | 2025-08-07 14:15:00 | 160.02 | 2025-08-13 13:15:00 | 156.15 | STOP_HIT | 0.50 | 2.42% |
| BUY | retest2 | 2025-09-01 09:15:00 | 175.13 | 2025-09-15 10:15:00 | 183.49 | STOP_HIT | 1.00 | 4.77% |
| BUY | retest2 | 2025-09-01 13:00:00 | 176.30 | 2025-09-15 10:15:00 | 183.49 | STOP_HIT | 1.00 | 4.08% |
| BUY | retest2 | 2025-09-02 09:30:00 | 175.66 | 2025-09-15 10:15:00 | 183.49 | STOP_HIT | 1.00 | 4.46% |
| BUY | retest2 | 2025-09-02 10:45:00 | 175.14 | 2025-09-15 10:15:00 | 183.49 | STOP_HIT | 1.00 | 4.77% |
| BUY | retest2 | 2025-09-04 09:30:00 | 179.66 | 2025-09-15 10:15:00 | 183.49 | STOP_HIT | 1.00 | 2.13% |
| SELL | retest2 | 2025-09-22 13:30:00 | 176.58 | 2025-09-24 09:15:00 | 180.10 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-09-30 09:15:00 | 169.31 | 2025-10-01 15:15:00 | 170.90 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-09-30 10:00:00 | 169.33 | 2025-10-01 15:15:00 | 170.90 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-10-01 09:15:00 | 168.30 | 2025-10-01 15:15:00 | 170.90 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-10-01 13:45:00 | 169.73 | 2025-10-01 15:15:00 | 170.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-10-09 11:30:00 | 162.50 | 2025-10-15 15:15:00 | 164.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-10-13 12:00:00 | 162.80 | 2025-10-15 15:15:00 | 164.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-10-13 14:00:00 | 162.70 | 2025-10-15 15:15:00 | 164.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-10-13 14:30:00 | 162.71 | 2025-10-15 15:15:00 | 164.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-10-14 11:00:00 | 162.30 | 2025-10-15 15:15:00 | 164.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-10-15 10:30:00 | 162.26 | 2025-10-15 15:15:00 | 164.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-10-21 13:45:00 | 166.30 | 2025-10-24 09:15:00 | 165.50 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-11-03 14:15:00 | 161.88 | 2025-11-07 09:15:00 | 153.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 14:45:00 | 161.42 | 2025-11-07 09:15:00 | 153.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 09:45:00 | 161.08 | 2025-11-07 09:15:00 | 153.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 12:15:00 | 161.62 | 2025-11-07 09:15:00 | 153.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 14:15:00 | 161.88 | 2025-11-10 11:15:00 | 150.32 | STOP_HIT | 0.50 | 7.14% |
| SELL | retest2 | 2025-11-03 14:45:00 | 161.42 | 2025-11-10 11:15:00 | 150.32 | STOP_HIT | 0.50 | 6.88% |
| SELL | retest2 | 2025-11-04 09:45:00 | 161.08 | 2025-11-10 11:15:00 | 150.32 | STOP_HIT | 0.50 | 6.68% |
| SELL | retest2 | 2025-11-04 12:15:00 | 161.62 | 2025-11-10 11:15:00 | 150.32 | STOP_HIT | 0.50 | 6.99% |
| SELL | retest2 | 2025-11-17 11:15:00 | 140.48 | 2025-11-17 15:15:00 | 142.70 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-11-17 12:00:00 | 140.72 | 2025-11-17 15:15:00 | 142.70 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-11-17 12:30:00 | 140.71 | 2025-11-17 15:15:00 | 142.70 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-11-17 13:30:00 | 140.42 | 2025-11-17 15:15:00 | 142.70 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-11-20 14:15:00 | 147.00 | 2025-11-21 09:15:00 | 144.30 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-11-26 13:45:00 | 139.63 | 2025-12-02 09:15:00 | 132.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 15:00:00 | 139.15 | 2025-12-02 09:15:00 | 132.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 09:30:00 | 138.98 | 2025-12-02 09:15:00 | 132.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 13:45:00 | 139.63 | 2025-12-03 09:15:00 | 135.00 | STOP_HIT | 0.50 | 3.32% |
| SELL | retest2 | 2025-11-26 15:00:00 | 139.15 | 2025-12-03 09:15:00 | 135.00 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2025-11-27 09:30:00 | 138.98 | 2025-12-03 09:15:00 | 135.00 | STOP_HIT | 0.50 | 2.86% |
| BUY | retest2 | 2025-12-30 10:15:00 | 144.91 | 2025-12-30 14:15:00 | 142.11 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-12-31 11:00:00 | 145.00 | 2026-01-02 09:15:00 | 159.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-31 13:15:00 | 148.45 | 2026-01-02 09:15:00 | 159.50 | TARGET_HIT | 1.00 | 7.44% |
| BUY | retest2 | 2026-01-01 13:45:00 | 145.00 | 2026-01-05 11:15:00 | 143.74 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-01-16 11:00:00 | 132.40 | 2026-01-20 11:15:00 | 126.11 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2026-01-16 12:30:00 | 132.75 | 2026-01-20 11:15:00 | 126.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 13:00:00 | 132.75 | 2026-01-20 11:15:00 | 126.19 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2026-01-16 14:15:00 | 132.83 | 2026-01-20 14:15:00 | 125.78 | PARTIAL | 0.50 | 5.31% |
| SELL | retest2 | 2026-01-16 11:00:00 | 132.40 | 2026-01-21 12:15:00 | 126.64 | STOP_HIT | 0.50 | 4.35% |
| SELL | retest2 | 2026-01-16 12:30:00 | 132.75 | 2026-01-21 12:15:00 | 126.64 | STOP_HIT | 0.50 | 4.60% |
| SELL | retest2 | 2026-01-16 13:00:00 | 132.75 | 2026-01-21 12:15:00 | 126.64 | STOP_HIT | 0.50 | 4.60% |
| SELL | retest2 | 2026-01-16 14:15:00 | 132.83 | 2026-01-21 12:15:00 | 126.64 | STOP_HIT | 0.50 | 4.66% |
| SELL | retest2 | 2026-01-22 10:45:00 | 124.38 | 2026-01-23 10:15:00 | 118.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:45:00 | 124.38 | 2026-01-27 09:15:00 | 111.94 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-04 09:15:00 | 118.48 | 2026-02-05 09:15:00 | 130.33 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-17 15:15:00 | 131.00 | 2026-02-18 09:15:00 | 133.24 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-02-26 09:15:00 | 130.59 | 2026-03-02 09:15:00 | 124.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 10:45:00 | 130.75 | 2026-03-02 09:15:00 | 124.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 09:15:00 | 130.59 | 2026-03-05 09:15:00 | 117.67 | TARGET_HIT | 0.50 | 9.89% |
| SELL | retest2 | 2026-02-26 10:45:00 | 130.75 | 2026-03-05 10:15:00 | 117.53 | TARGET_HIT | 0.50 | 10.11% |
| SELL | retest2 | 2026-03-17 14:15:00 | 107.89 | 2026-03-23 09:15:00 | 102.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 106.55 | 2026-03-23 09:15:00 | 101.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 14:15:00 | 107.89 | 2026-03-23 15:15:00 | 101.60 | STOP_HIT | 0.50 | 5.83% |
| SELL | retest2 | 2026-03-19 09:15:00 | 106.55 | 2026-03-23 15:15:00 | 101.60 | STOP_HIT | 0.50 | 4.65% |
| SELL | retest2 | 2026-04-02 09:15:00 | 93.10 | 2026-04-06 11:15:00 | 99.87 | STOP_HIT | 1.00 | -7.27% |
| SELL | retest2 | 2026-04-02 13:45:00 | 95.93 | 2026-04-06 11:15:00 | 99.87 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2026-04-02 14:30:00 | 96.25 | 2026-04-06 11:15:00 | 99.87 | STOP_HIT | 1.00 | -3.76% |
| SELL | retest2 | 2026-04-06 09:15:00 | 95.01 | 2026-04-06 11:15:00 | 99.87 | STOP_HIT | 1.00 | -5.12% |
| BUY | retest2 | 2026-04-13 13:30:00 | 107.21 | 2026-04-20 13:15:00 | 106.92 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2026-04-15 09:15:00 | 107.96 | 2026-04-20 13:15:00 | 106.92 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2026-04-16 10:30:00 | 107.30 | 2026-04-20 13:15:00 | 106.92 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2026-04-20 09:45:00 | 107.25 | 2026-04-20 13:15:00 | 106.92 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-04-20 11:45:00 | 108.40 | 2026-04-20 13:15:00 | 106.92 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-04-20 12:45:00 | 108.65 | 2026-04-20 13:15:00 | 106.92 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-04-23 13:15:00 | 111.74 | 2026-04-23 14:15:00 | 110.20 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-04-23 14:30:00 | 111.65 | 2026-04-23 15:15:00 | 110.30 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2026-04-28 10:15:00 | 112.18 | 2026-04-29 13:15:00 | 123.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-28 10:45:00 | 112.31 | 2026-04-29 13:15:00 | 123.38 | TARGET_HIT | 1.00 | 9.85% |
| BUY | retest2 | 2026-04-28 11:45:00 | 112.16 | 2026-04-29 13:15:00 | 123.23 | TARGET_HIT | 1.00 | 9.87% |
| BUY | retest2 | 2026-04-28 13:15:00 | 112.03 | 2026-04-29 14:15:00 | 123.54 | TARGET_HIT | 1.00 | 10.27% |
