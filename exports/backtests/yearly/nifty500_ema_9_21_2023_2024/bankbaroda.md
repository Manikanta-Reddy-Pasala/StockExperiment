# Bank of Baroda (BANKBARODA)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 263.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 222 |
| ALERT1 | 162 |
| ALERT2 | 158 |
| ALERT2_SKIP | 75 |
| ALERT3 | 430 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 165 |
| PARTIAL | 25 |
| TARGET_HIT | 6 |
| STOP_HIT | 165 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 194 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 87 / 107
- **Target hits / Stop hits / Partials:** 6 / 163 / 25
- **Avg / median % per leg:** 0.95% / -0.39%
- **Sum % (uncompounded):** 184.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 73 | 28 | 38.4% | 5 | 68 | 0 | 0.65% | 47.4% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -1.29% | -2.6% |
| BUY @ 3rd Alert (retest2) | 71 | 27 | 38.0% | 5 | 66 | 0 | 0.70% | 49.9% |
| SELL (all) | 121 | 59 | 48.8% | 1 | 95 | 25 | 1.13% | 137.0% |
| SELL @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 2 | 1 | 3.10% | 9.3% |
| SELL @ 3rd Alert (retest2) | 118 | 56 | 47.5% | 1 | 93 | 24 | 1.08% | 127.7% |
| retest1 (combined) | 5 | 4 | 80.0% | 0 | 4 | 1 | 1.35% | 6.7% |
| retest2 (combined) | 189 | 83 | 43.9% | 6 | 159 | 24 | 0.94% | 177.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 09:15:00 | 181.60 | 179.33 | 179.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-15 10:15:00 | 182.90 | 180.04 | 179.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-17 11:15:00 | 184.60 | 185.64 | 184.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 11:15:00 | 184.60 | 185.64 | 184.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 11:15:00 | 184.60 | 185.64 | 184.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-17 12:00:00 | 184.60 | 185.64 | 184.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 13:15:00 | 185.05 | 185.31 | 184.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-18 09:15:00 | 186.80 | 185.54 | 184.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-18 13:15:00 | 182.20 | 184.24 | 184.18 | SL hit (close<static) qty=1.00 sl=183.75 alert=retest2 |

### Cycle 2 — SELL (started 2023-05-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 14:15:00 | 180.45 | 183.48 | 183.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 09:15:00 | 179.90 | 182.24 | 183.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 14:15:00 | 181.80 | 181.16 | 182.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 14:15:00 | 181.80 | 181.16 | 182.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 181.80 | 181.16 | 182.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 15:00:00 | 181.80 | 181.16 | 182.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 15:15:00 | 181.70 | 181.27 | 182.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:15:00 | 181.80 | 181.27 | 182.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 180.40 | 181.09 | 181.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 12:30:00 | 179.45 | 180.74 | 181.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-23 09:15:00 | 183.55 | 181.37 | 181.61 | SL hit (close>static) qty=1.00 sl=182.25 alert=retest2 |

### Cycle 3 — BUY (started 2023-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 10:15:00 | 183.50 | 181.80 | 181.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 12:15:00 | 184.25 | 182.62 | 182.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 11:15:00 | 182.70 | 183.18 | 182.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 11:15:00 | 182.70 | 183.18 | 182.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 11:15:00 | 182.70 | 183.18 | 182.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 12:00:00 | 182.70 | 183.18 | 182.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 12:15:00 | 182.40 | 183.03 | 182.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 12:45:00 | 182.45 | 183.03 | 182.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 13:15:00 | 182.65 | 182.95 | 182.68 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-05-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 11:15:00 | 180.90 | 182.25 | 182.43 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 10:15:00 | 184.55 | 182.68 | 182.48 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 12:15:00 | 183.05 | 183.37 | 183.38 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 14:15:00 | 185.20 | 183.71 | 183.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 10:15:00 | 185.40 | 184.43 | 183.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 14:15:00 | 184.90 | 185.14 | 184.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-01 15:00:00 | 184.90 | 185.14 | 184.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 15:15:00 | 184.70 | 185.05 | 184.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 09:15:00 | 185.90 | 185.05 | 184.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 10:00:00 | 185.15 | 185.94 | 185.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-06 10:15:00 | 184.50 | 185.65 | 185.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2023-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 10:15:00 | 184.50 | 185.65 | 185.68 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 12:15:00 | 185.90 | 185.70 | 185.70 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 13:15:00 | 185.55 | 185.67 | 185.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 14:15:00 | 185.45 | 185.63 | 185.66 | Break + close below crossover candle low |

### Cycle 11 — BUY (started 2023-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 09:15:00 | 187.40 | 185.92 | 185.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 10:15:00 | 188.05 | 186.35 | 185.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 11:15:00 | 186.50 | 187.34 | 186.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 11:15:00 | 186.50 | 187.34 | 186.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 186.50 | 187.34 | 186.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:00:00 | 186.50 | 187.34 | 186.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 187.25 | 187.32 | 186.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:30:00 | 186.95 | 187.32 | 186.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 15:15:00 | 187.60 | 187.50 | 187.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 09:15:00 | 186.50 | 187.50 | 187.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 09:15:00 | 187.35 | 187.47 | 187.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 09:30:00 | 186.85 | 187.47 | 187.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 10:15:00 | 186.80 | 187.34 | 187.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 11:00:00 | 186.80 | 187.34 | 187.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 11:15:00 | 186.40 | 187.15 | 187.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 11:30:00 | 186.30 | 187.15 | 187.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2023-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 12:15:00 | 185.20 | 186.76 | 186.85 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 14:15:00 | 187.00 | 186.48 | 186.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 15:15:00 | 187.50 | 187.11 | 186.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 09:15:00 | 187.05 | 187.10 | 186.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-15 09:15:00 | 187.05 | 187.10 | 186.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 09:15:00 | 187.05 | 187.10 | 186.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-15 10:15:00 | 187.70 | 187.10 | 186.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-15 10:15:00 | 186.35 | 186.95 | 186.92 | SL hit (close<static) qty=1.00 sl=186.45 alert=retest2 |

### Cycle 14 — SELL (started 2023-06-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 11:15:00 | 185.85 | 186.73 | 186.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-15 12:15:00 | 184.90 | 186.36 | 186.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 09:15:00 | 186.90 | 185.32 | 185.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 09:15:00 | 186.90 | 185.32 | 185.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 186.90 | 185.32 | 185.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 10:00:00 | 186.90 | 185.32 | 185.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 10:15:00 | 187.00 | 185.66 | 186.03 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 13:15:00 | 187.15 | 186.30 | 186.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 14:15:00 | 188.15 | 186.67 | 186.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 09:15:00 | 195.75 | 196.68 | 194.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-22 10:00:00 | 195.75 | 196.68 | 194.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 194.15 | 196.03 | 194.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 12:00:00 | 194.15 | 196.03 | 194.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 12:15:00 | 193.25 | 195.48 | 194.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 12:45:00 | 192.85 | 195.48 | 194.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2023-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 09:15:00 | 192.45 | 194.10 | 194.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-26 09:15:00 | 189.50 | 192.21 | 193.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 11:15:00 | 191.25 | 191.19 | 191.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-27 11:45:00 | 191.40 | 191.19 | 191.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 09:15:00 | 190.95 | 191.06 | 191.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-28 12:30:00 | 190.00 | 190.69 | 191.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-30 09:15:00 | 187.85 | 190.47 | 191.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-30 10:00:00 | 190.10 | 190.39 | 190.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-30 11:45:00 | 190.35 | 190.35 | 190.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 12:15:00 | 190.15 | 190.31 | 190.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-30 13:15:00 | 189.50 | 190.31 | 190.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-03 09:15:00 | 191.90 | 190.56 | 190.71 | SL hit (close>static) qty=1.00 sl=190.75 alert=retest2 |

### Cycle 17 — BUY (started 2023-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 10:15:00 | 192.65 | 190.98 | 190.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-03 12:15:00 | 195.45 | 192.23 | 191.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-06 10:15:00 | 204.00 | 204.07 | 201.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-06 10:30:00 | 204.10 | 204.07 | 201.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 204.20 | 204.65 | 203.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 10:45:00 | 203.55 | 204.65 | 203.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 12:15:00 | 204.20 | 204.49 | 203.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 12:30:00 | 203.35 | 204.49 | 203.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 11:15:00 | 205.10 | 206.39 | 205.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 12:00:00 | 205.10 | 206.39 | 205.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 12:15:00 | 205.10 | 206.13 | 205.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 13:00:00 | 205.10 | 206.13 | 205.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 13:15:00 | 205.35 | 205.98 | 205.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 14:15:00 | 205.00 | 205.98 | 205.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 14:15:00 | 203.35 | 205.45 | 204.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 15:00:00 | 203.35 | 205.45 | 204.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 15:15:00 | 203.85 | 205.13 | 204.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-11 09:15:00 | 204.60 | 205.13 | 204.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-11 10:00:00 | 204.65 | 205.03 | 204.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-12 14:15:00 | 204.50 | 205.01 | 205.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2023-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 14:15:00 | 204.50 | 205.01 | 205.02 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-07-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 11:15:00 | 205.55 | 205.11 | 205.05 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-07-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 12:15:00 | 203.55 | 204.80 | 204.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 13:15:00 | 196.90 | 203.22 | 204.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 14:15:00 | 198.60 | 198.43 | 200.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-14 15:00:00 | 198.60 | 198.43 | 200.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 13:15:00 | 199.10 | 198.30 | 199.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-18 12:00:00 | 197.95 | 199.44 | 199.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-18 12:45:00 | 197.75 | 199.16 | 199.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-18 13:45:00 | 198.35 | 198.92 | 199.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-19 15:15:00 | 200.00 | 199.34 | 199.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2023-07-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 15:15:00 | 200.00 | 199.34 | 199.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-20 09:15:00 | 201.40 | 199.75 | 199.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 12:15:00 | 199.80 | 199.92 | 199.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 12:15:00 | 199.80 | 199.92 | 199.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 12:15:00 | 199.80 | 199.92 | 199.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 13:00:00 | 199.80 | 199.92 | 199.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 13:15:00 | 199.30 | 199.79 | 199.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 13:30:00 | 199.15 | 199.79 | 199.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 14:15:00 | 199.20 | 199.68 | 199.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 15:00:00 | 199.20 | 199.68 | 199.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 15:15:00 | 199.00 | 199.54 | 199.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 09:15:00 | 201.45 | 199.54 | 199.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-21 12:15:00 | 198.20 | 199.47 | 199.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2023-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 12:15:00 | 198.20 | 199.47 | 199.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-25 14:15:00 | 196.10 | 197.24 | 198.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 10:15:00 | 197.20 | 197.03 | 197.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-26 11:00:00 | 197.20 | 197.03 | 197.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 11:15:00 | 197.90 | 197.21 | 197.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 12:00:00 | 197.90 | 197.21 | 197.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 12:15:00 | 197.55 | 197.27 | 197.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-26 15:15:00 | 197.30 | 197.48 | 197.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-27 09:15:00 | 199.95 | 197.94 | 197.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2023-07-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 09:15:00 | 199.95 | 197.94 | 197.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 10:15:00 | 201.65 | 198.68 | 198.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-27 15:15:00 | 199.15 | 199.24 | 198.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-28 09:15:00 | 198.70 | 199.24 | 198.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 09:15:00 | 199.80 | 199.35 | 198.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-28 15:00:00 | 201.20 | 199.96 | 199.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-01 12:45:00 | 201.30 | 201.71 | 201.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-02 09:15:00 | 198.75 | 200.87 | 200.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2023-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 09:15:00 | 198.75 | 200.87 | 200.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 10:15:00 | 196.30 | 199.96 | 200.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 10:15:00 | 195.00 | 193.89 | 195.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-04 11:00:00 | 195.00 | 193.89 | 195.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 11:15:00 | 195.75 | 194.26 | 195.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 11:45:00 | 195.75 | 194.26 | 195.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 12:15:00 | 195.25 | 194.46 | 195.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 13:15:00 | 193.30 | 194.46 | 195.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 11:00:00 | 192.60 | 191.79 | 192.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 14:30:00 | 194.60 | 193.11 | 193.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-08 15:15:00 | 194.95 | 193.47 | 193.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2023-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 15:15:00 | 194.95 | 193.47 | 193.27 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-08-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 12:15:00 | 191.95 | 193.17 | 193.31 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 12:15:00 | 193.65 | 193.23 | 193.19 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 14:15:00 | 192.55 | 193.10 | 193.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 09:15:00 | 188.85 | 192.17 | 192.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 09:15:00 | 190.00 | 188.03 | 189.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 09:15:00 | 190.00 | 188.03 | 189.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 190.00 | 188.03 | 189.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 10:00:00 | 190.00 | 188.03 | 189.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 10:15:00 | 189.80 | 188.38 | 189.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 10:45:00 | 190.30 | 188.38 | 189.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 12:15:00 | 188.90 | 188.60 | 189.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 12:30:00 | 189.25 | 188.60 | 189.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 13:15:00 | 189.45 | 188.77 | 189.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 13:45:00 | 189.65 | 188.77 | 189.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 14:15:00 | 189.90 | 189.00 | 189.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 14:30:00 | 189.90 | 189.00 | 189.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 15:15:00 | 190.25 | 189.25 | 189.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 09:15:00 | 190.65 | 189.25 | 189.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2023-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 09:15:00 | 192.00 | 189.80 | 189.59 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-22 15:15:00 | 190.30 | 190.53 | 190.56 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 09:15:00 | 190.80 | 190.59 | 190.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 10:15:00 | 192.35 | 190.94 | 190.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 11:15:00 | 192.70 | 193.09 | 192.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-24 12:00:00 | 192.70 | 193.09 | 192.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 13:15:00 | 192.35 | 192.90 | 192.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 14:00:00 | 192.35 | 192.90 | 192.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 14:15:00 | 191.80 | 192.68 | 192.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 14:45:00 | 192.05 | 192.68 | 192.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 191.80 | 192.50 | 192.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 09:15:00 | 191.80 | 192.50 | 192.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 10:15:00 | 189.70 | 191.62 | 191.84 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 15:15:00 | 191.10 | 190.79 | 190.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 09:15:00 | 191.80 | 190.99 | 190.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 09:15:00 | 191.15 | 191.51 | 191.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 09:15:00 | 191.15 | 191.51 | 191.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 191.15 | 191.51 | 191.25 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-08-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 10:15:00 | 188.60 | 190.92 | 191.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 12:15:00 | 188.00 | 189.99 | 190.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 10:15:00 | 189.10 | 188.58 | 189.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-01 11:00:00 | 189.10 | 188.58 | 189.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 11:15:00 | 190.00 | 188.87 | 189.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 11:30:00 | 189.75 | 188.87 | 189.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 12:15:00 | 191.75 | 189.44 | 189.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 13:00:00 | 191.75 | 189.44 | 189.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 13:15:00 | 191.60 | 189.87 | 189.92 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 14:15:00 | 190.65 | 190.03 | 189.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 10:15:00 | 192.60 | 190.85 | 190.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 09:15:00 | 195.10 | 195.29 | 193.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-06 09:30:00 | 195.35 | 195.29 | 193.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 12:15:00 | 194.85 | 195.31 | 194.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 12:45:00 | 194.45 | 195.31 | 194.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 15:15:00 | 194.85 | 195.07 | 194.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 09:15:00 | 195.30 | 195.07 | 194.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 09:45:00 | 195.35 | 195.05 | 194.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 10:30:00 | 195.40 | 195.21 | 194.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 11:45:00 | 195.30 | 195.26 | 194.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 199.65 | 200.96 | 199.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 199.35 | 200.96 | 199.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 202.90 | 201.35 | 199.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 13:30:00 | 206.90 | 202.53 | 201.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-18 09:15:00 | 214.83 | 211.49 | 209.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2023-09-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 13:15:00 | 207.70 | 212.43 | 213.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 14:15:00 | 207.25 | 211.39 | 212.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 09:15:00 | 211.90 | 210.74 | 211.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 09:15:00 | 211.90 | 210.74 | 211.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 211.90 | 210.74 | 211.97 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 11:15:00 | 216.90 | 212.89 | 212.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 12:15:00 | 218.25 | 215.86 | 214.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 11:15:00 | 216.05 | 216.51 | 215.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-26 12:00:00 | 216.05 | 216.51 | 215.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 12:15:00 | 214.10 | 216.03 | 215.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 13:00:00 | 214.10 | 216.03 | 215.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 13:15:00 | 213.95 | 215.61 | 215.32 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 15:15:00 | 213.95 | 214.95 | 215.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-27 09:15:00 | 212.55 | 214.47 | 214.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-27 10:15:00 | 214.75 | 214.53 | 214.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-27 10:15:00 | 214.75 | 214.53 | 214.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 214.75 | 214.53 | 214.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 11:00:00 | 214.75 | 214.53 | 214.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 11:15:00 | 213.80 | 214.38 | 214.72 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-09-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 09:15:00 | 215.55 | 214.92 | 214.89 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-09-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 13:15:00 | 212.75 | 214.50 | 214.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 14:15:00 | 210.45 | 213.69 | 214.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 09:15:00 | 213.70 | 213.42 | 214.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 09:15:00 | 213.70 | 213.42 | 214.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 213.70 | 213.42 | 214.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 09:45:00 | 213.65 | 213.42 | 214.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 10:15:00 | 213.00 | 213.34 | 213.99 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 11:15:00 | 215.00 | 214.19 | 214.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 12:15:00 | 215.35 | 214.42 | 214.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 09:15:00 | 213.75 | 215.50 | 214.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 09:15:00 | 213.75 | 215.50 | 214.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 213.75 | 215.50 | 214.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 10:00:00 | 213.75 | 215.50 | 214.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 10:15:00 | 213.65 | 215.13 | 214.83 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 12:15:00 | 212.10 | 214.15 | 214.41 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2023-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 12:15:00 | 214.55 | 213.71 | 213.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 13:15:00 | 215.45 | 214.06 | 213.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 10:15:00 | 213.85 | 214.40 | 214.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 10:15:00 | 213.85 | 214.40 | 214.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 10:15:00 | 213.85 | 214.40 | 214.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-09 10:30:00 | 213.75 | 214.40 | 214.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 11:15:00 | 213.20 | 214.16 | 214.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-09 12:00:00 | 213.20 | 214.16 | 214.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2023-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 12:15:00 | 212.25 | 213.78 | 213.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 14:15:00 | 211.85 | 213.17 | 213.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 10:15:00 | 212.75 | 212.74 | 213.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 10:15:00 | 212.75 | 212.74 | 213.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 212.75 | 212.74 | 213.22 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2023-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 14:15:00 | 214.35 | 213.42 | 213.42 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-11 09:15:00 | 208.25 | 212.46 | 212.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 09:15:00 | 202.80 | 207.16 | 208.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-16 11:15:00 | 205.15 | 204.61 | 206.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-16 12:00:00 | 205.15 | 204.61 | 206.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 12:15:00 | 205.90 | 204.87 | 206.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-16 13:00:00 | 205.90 | 204.87 | 206.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 13:15:00 | 204.85 | 204.86 | 205.97 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2023-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 12:15:00 | 207.60 | 206.47 | 206.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-18 09:15:00 | 207.90 | 207.05 | 206.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 10:15:00 | 205.45 | 206.73 | 206.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 10:15:00 | 205.45 | 206.73 | 206.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 205.45 | 206.73 | 206.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 11:00:00 | 205.45 | 206.73 | 206.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2023-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 11:15:00 | 205.15 | 206.41 | 206.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 12:15:00 | 204.30 | 205.99 | 206.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 10:15:00 | 204.60 | 204.57 | 205.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-19 11:00:00 | 204.60 | 204.57 | 205.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 11:15:00 | 205.35 | 204.72 | 205.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 11:30:00 | 205.40 | 204.72 | 205.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 12:15:00 | 205.05 | 204.79 | 205.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 12:30:00 | 205.05 | 204.79 | 205.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 13:15:00 | 204.95 | 204.82 | 205.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 14:00:00 | 204.95 | 204.82 | 205.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 14:15:00 | 205.10 | 204.88 | 205.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 15:00:00 | 205.10 | 204.88 | 205.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 15:15:00 | 205.20 | 204.94 | 205.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 09:15:00 | 204.35 | 204.94 | 205.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 203.55 | 204.66 | 205.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 11:45:00 | 202.60 | 204.12 | 204.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 12:15:00 | 202.50 | 204.12 | 204.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 14:15:00 | 202.50 | 203.53 | 204.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 192.47 | 194.67 | 197.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 192.38 | 194.67 | 197.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 192.38 | 194.67 | 197.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-27 09:15:00 | 193.65 | 191.82 | 194.28 | SL hit (close>ema200) qty=0.50 sl=191.82 alert=retest2 |

### Cycle 49 — BUY (started 2023-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 15:15:00 | 197.50 | 195.24 | 195.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 11:15:00 | 197.55 | 195.85 | 195.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 10:15:00 | 197.20 | 197.53 | 196.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-31 11:00:00 | 197.20 | 197.53 | 196.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 11:15:00 | 195.85 | 197.19 | 196.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 12:00:00 | 195.85 | 197.19 | 196.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 12:15:00 | 196.70 | 197.10 | 196.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-31 14:00:00 | 197.30 | 197.14 | 196.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-01 10:00:00 | 197.35 | 196.91 | 196.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-01 12:15:00 | 195.60 | 196.56 | 196.55 | SL hit (close<static) qty=1.00 sl=195.65 alert=retest2 |

### Cycle 50 — SELL (started 2023-11-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 13:15:00 | 196.40 | 196.53 | 196.54 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2023-11-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 09:15:00 | 198.95 | 196.76 | 196.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 09:15:00 | 202.95 | 199.43 | 198.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-06 09:15:00 | 196.15 | 201.05 | 200.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 09:15:00 | 196.15 | 201.05 | 200.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 196.15 | 201.05 | 200.01 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2023-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 11:15:00 | 195.60 | 199.09 | 199.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-07 09:15:00 | 193.80 | 196.43 | 197.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 15:15:00 | 192.90 | 192.60 | 193.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-09 09:15:00 | 193.35 | 192.60 | 193.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 193.50 | 192.78 | 193.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 09:45:00 | 193.85 | 192.78 | 193.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 11:15:00 | 193.40 | 193.03 | 193.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 11:30:00 | 193.60 | 193.03 | 193.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 13:15:00 | 193.65 | 193.15 | 193.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-09 15:00:00 | 192.80 | 193.08 | 193.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-10 09:30:00 | 192.80 | 193.05 | 193.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-10 10:15:00 | 192.90 | 193.05 | 193.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-10 14:15:00 | 194.55 | 193.72 | 193.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2023-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 14:15:00 | 194.55 | 193.72 | 193.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 15:15:00 | 195.05 | 193.99 | 193.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-13 09:15:00 | 194.20 | 194.21 | 193.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-13 10:15:00 | 193.90 | 194.21 | 193.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 10:15:00 | 194.30 | 194.23 | 193.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-13 10:30:00 | 194.00 | 194.23 | 193.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 196.10 | 198.03 | 197.47 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2023-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 13:15:00 | 196.30 | 197.19 | 197.20 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2023-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 09:15:00 | 197.80 | 197.21 | 197.19 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2023-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 09:15:00 | 195.90 | 197.16 | 197.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-21 10:15:00 | 195.15 | 196.76 | 197.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-22 09:15:00 | 196.25 | 196.22 | 196.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 09:15:00 | 196.25 | 196.22 | 196.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 196.25 | 196.22 | 196.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-22 09:45:00 | 196.45 | 196.22 | 196.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 196.20 | 195.21 | 195.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 10:00:00 | 196.20 | 195.21 | 195.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 10:15:00 | 195.30 | 195.23 | 195.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 12:00:00 | 194.95 | 195.17 | 195.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 13:00:00 | 195.10 | 195.16 | 195.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 13:30:00 | 194.90 | 195.08 | 195.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-28 12:15:00 | 197.40 | 195.08 | 194.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2023-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 12:15:00 | 197.40 | 195.08 | 194.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 09:15:00 | 199.95 | 197.59 | 196.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 12:15:00 | 197.90 | 198.03 | 197.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-30 12:30:00 | 198.30 | 198.03 | 197.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 13:15:00 | 196.95 | 197.82 | 197.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 14:00:00 | 196.95 | 197.82 | 197.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 14:15:00 | 197.15 | 197.68 | 197.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 14:30:00 | 197.05 | 197.68 | 197.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 15:15:00 | 197.90 | 197.73 | 197.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-01 09:15:00 | 199.55 | 197.73 | 197.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-13 13:15:00 | 219.51 | 217.67 | 215.88 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 219.10 | 223.60 | 223.80 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2023-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 09:15:00 | 225.15 | 223.17 | 223.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 09:15:00 | 232.40 | 226.09 | 224.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 15:15:00 | 232.50 | 232.57 | 230.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-29 09:15:00 | 232.70 | 232.57 | 230.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 230.00 | 232.06 | 230.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 10:00:00 | 230.00 | 232.06 | 230.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 10:15:00 | 230.30 | 231.71 | 230.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 10:30:00 | 230.30 | 231.71 | 230.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 11:15:00 | 230.90 | 231.54 | 230.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 11:30:00 | 230.35 | 231.54 | 230.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 12:15:00 | 231.10 | 231.46 | 230.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 13:00:00 | 231.10 | 231.46 | 230.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 13:15:00 | 229.55 | 231.07 | 230.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 14:00:00 | 229.55 | 231.07 | 230.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 14:15:00 | 231.00 | 231.06 | 230.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 09:15:00 | 233.00 | 231.02 | 230.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 11:30:00 | 232.80 | 232.92 | 232.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 14:00:00 | 232.10 | 232.55 | 232.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-02 14:15:00 | 230.50 | 232.14 | 232.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 14:15:00 | 230.50 | 232.14 | 232.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 09:15:00 | 228.50 | 231.21 | 231.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-03 10:15:00 | 232.80 | 231.53 | 231.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-03 10:15:00 | 232.80 | 231.53 | 231.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 10:15:00 | 232.80 | 231.53 | 231.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 11:00:00 | 232.80 | 231.53 | 231.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 11:15:00 | 233.40 | 231.90 | 231.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 11:45:00 | 233.95 | 231.90 | 231.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2024-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 12:15:00 | 234.40 | 232.40 | 232.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 10:15:00 | 235.30 | 233.64 | 232.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 11:15:00 | 236.60 | 236.67 | 235.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 11:15:00 | 236.60 | 236.67 | 235.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 11:15:00 | 236.60 | 236.67 | 235.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 11:45:00 | 236.55 | 236.67 | 235.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 13:15:00 | 233.60 | 235.99 | 235.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 14:00:00 | 233.60 | 235.99 | 235.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 14:15:00 | 234.90 | 235.78 | 235.21 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 09:15:00 | 227.30 | 233.84 | 234.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 14:15:00 | 222.85 | 228.03 | 230.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 13:15:00 | 223.70 | 223.30 | 225.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-10 14:00:00 | 223.70 | 223.30 | 225.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 224.80 | 223.79 | 225.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 10:15:00 | 226.80 | 223.79 | 225.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 10:15:00 | 224.70 | 223.97 | 225.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 10:30:00 | 225.35 | 223.97 | 225.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 11:15:00 | 225.35 | 224.25 | 225.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 11:45:00 | 225.55 | 224.25 | 225.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 12:15:00 | 226.05 | 224.61 | 225.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 12:45:00 | 226.05 | 224.61 | 225.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 13:15:00 | 224.60 | 224.61 | 225.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 14:00:00 | 224.60 | 224.61 | 225.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 14:15:00 | 226.05 | 224.89 | 225.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 14:45:00 | 226.40 | 224.89 | 225.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 15:15:00 | 226.00 | 225.12 | 225.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-12 09:15:00 | 225.85 | 225.12 | 225.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2024-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 09:15:00 | 227.45 | 225.58 | 225.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-12 10:15:00 | 230.65 | 226.60 | 226.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 12:15:00 | 230.90 | 231.35 | 230.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-16 13:00:00 | 230.90 | 231.35 | 230.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 13:15:00 | 231.70 | 231.42 | 230.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:30:00 | 230.65 | 231.42 | 230.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 229.90 | 231.13 | 230.39 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2024-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 13:15:00 | 228.65 | 229.81 | 229.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 14:15:00 | 226.15 | 229.08 | 229.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 10:15:00 | 227.85 | 227.79 | 228.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-18 11:00:00 | 227.85 | 227.79 | 228.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 11:15:00 | 229.75 | 228.18 | 228.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 12:00:00 | 229.75 | 228.18 | 228.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 12:15:00 | 227.15 | 227.98 | 228.72 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 12:15:00 | 230.30 | 229.00 | 228.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 14:15:00 | 230.60 | 229.33 | 229.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 09:15:00 | 228.75 | 229.44 | 229.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 09:15:00 | 228.75 | 229.44 | 229.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 09:15:00 | 228.75 | 229.44 | 229.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 10:00:00 | 228.75 | 229.44 | 229.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 10:15:00 | 229.60 | 229.47 | 229.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-20 14:00:00 | 232.50 | 230.07 | 229.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-23 11:15:00 | 225.00 | 229.30 | 229.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 11:15:00 | 225.00 | 229.30 | 229.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 13:15:00 | 224.45 | 227.67 | 228.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 09:15:00 | 226.35 | 226.03 | 227.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 09:15:00 | 226.35 | 226.03 | 227.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 226.35 | 226.03 | 227.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 10:00:00 | 226.35 | 226.03 | 227.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 10:15:00 | 227.55 | 226.33 | 227.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 11:00:00 | 227.55 | 226.33 | 227.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 11:15:00 | 224.95 | 226.06 | 227.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 12:30:00 | 224.40 | 226.33 | 226.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 14:00:00 | 224.40 | 225.94 | 226.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-29 09:15:00 | 230.70 | 227.22 | 227.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2024-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 09:15:00 | 230.70 | 227.22 | 227.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 10:15:00 | 234.20 | 228.62 | 227.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 09:15:00 | 235.50 | 236.86 | 234.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 09:15:00 | 235.50 | 236.86 | 234.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 235.50 | 236.86 | 234.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 09:30:00 | 233.85 | 236.86 | 234.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 10:15:00 | 238.55 | 237.20 | 234.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 11:15:00 | 240.10 | 237.20 | 234.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 12:45:00 | 239.70 | 238.19 | 235.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-06 12:15:00 | 248.55 | 253.18 | 253.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-06 12:15:00 | 248.55 | 253.18 | 253.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-07 12:15:00 | 244.95 | 248.85 | 250.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-08 09:15:00 | 254.30 | 248.61 | 249.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 09:15:00 | 254.30 | 248.61 | 249.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 254.30 | 248.61 | 249.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 10:00:00 | 254.30 | 248.61 | 249.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 250.10 | 248.91 | 249.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 10:45:00 | 250.25 | 248.91 | 249.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 11:15:00 | 252.90 | 249.71 | 250.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 12:00:00 | 252.90 | 249.71 | 250.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 12:15:00 | 253.40 | 250.45 | 250.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 12:30:00 | 253.30 | 250.45 | 250.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-02-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 13:15:00 | 255.00 | 251.36 | 250.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-09 12:15:00 | 262.00 | 254.39 | 252.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 11:15:00 | 259.10 | 259.61 | 256.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-12 12:00:00 | 259.10 | 259.61 | 256.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 13:15:00 | 259.00 | 259.32 | 257.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 13:45:00 | 257.40 | 259.32 | 257.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 14:15:00 | 254.35 | 258.33 | 256.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 15:00:00 | 254.35 | 258.33 | 256.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 15:15:00 | 254.40 | 257.54 | 256.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 09:15:00 | 249.60 | 257.54 | 256.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 11:15:00 | 254.50 | 256.48 | 256.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 11:30:00 | 254.80 | 256.48 | 256.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 12:15:00 | 255.95 | 256.37 | 256.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-13 13:15:00 | 256.85 | 256.37 | 256.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 10:45:00 | 257.50 | 257.36 | 256.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-22 12:15:00 | 272.60 | 273.66 | 273.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-02-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 12:15:00 | 272.60 | 273.66 | 273.75 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-02-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 14:15:00 | 275.35 | 274.12 | 273.95 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 10:15:00 | 269.80 | 273.38 | 273.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 11:15:00 | 262.85 | 268.02 | 269.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 12:15:00 | 262.90 | 262.86 | 265.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 13:00:00 | 262.90 | 262.86 | 265.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 265.90 | 263.36 | 265.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 15:00:00 | 265.90 | 263.36 | 265.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 266.50 | 263.99 | 265.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:15:00 | 268.75 | 263.99 | 265.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 11:15:00 | 267.80 | 265.68 | 265.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 11:30:00 | 268.40 | 265.68 | 265.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-03-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 12:15:00 | 269.30 | 266.41 | 266.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 13:15:00 | 270.25 | 267.17 | 266.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 274.40 | 275.87 | 273.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 09:15:00 | 274.40 | 275.87 | 273.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 274.40 | 275.87 | 273.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 274.40 | 275.87 | 273.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 278.75 | 276.44 | 274.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 13:00:00 | 283.00 | 278.11 | 275.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 13:45:00 | 282.20 | 279.01 | 276.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-07 13:45:00 | 281.85 | 280.92 | 278.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-07 14:45:00 | 282.45 | 281.06 | 279.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 11:15:00 | 279.65 | 281.37 | 279.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 12:00:00 | 279.65 | 281.37 | 279.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 12:15:00 | 278.65 | 280.82 | 279.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 13:00:00 | 278.65 | 280.82 | 279.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 13:15:00 | 278.25 | 280.31 | 279.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 14:15:00 | 277.15 | 280.31 | 279.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-03-12 09:15:00 | 278.20 | 279.06 | 279.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 09:15:00 | 278.20 | 279.06 | 279.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 11:15:00 | 273.10 | 277.09 | 278.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-15 15:15:00 | 254.90 | 254.84 | 259.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-18 09:15:00 | 257.15 | 254.84 | 259.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 247.25 | 250.14 | 252.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 09:30:00 | 251.00 | 250.14 | 252.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 13:15:00 | 249.85 | 249.66 | 251.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 13:45:00 | 250.75 | 249.66 | 251.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 255.95 | 250.99 | 251.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 10:00:00 | 255.95 | 250.99 | 251.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2024-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 12:15:00 | 255.00 | 252.89 | 252.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 14:15:00 | 256.15 | 253.92 | 253.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 09:15:00 | 260.10 | 260.78 | 258.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-27 10:00:00 | 260.10 | 260.78 | 258.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 13:15:00 | 259.50 | 260.75 | 259.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 14:00:00 | 259.50 | 260.75 | 259.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 258.90 | 260.38 | 259.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 15:00:00 | 258.90 | 260.38 | 259.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 260.50 | 260.41 | 259.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 09:15:00 | 260.65 | 260.41 | 259.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-05 12:15:00 | 266.85 | 269.15 | 269.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-04-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 12:15:00 | 266.85 | 269.15 | 269.23 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 09:15:00 | 270.90 | 269.36 | 269.26 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 14:15:00 | 267.55 | 269.13 | 269.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 12:15:00 | 266.25 | 268.29 | 268.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 09:15:00 | 267.50 | 266.85 | 267.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-10 09:30:00 | 267.65 | 266.85 | 267.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 10:15:00 | 268.55 | 267.19 | 267.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 10:30:00 | 268.90 | 267.19 | 267.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 11:15:00 | 268.15 | 267.38 | 267.88 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2024-04-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 14:15:00 | 270.45 | 268.48 | 268.30 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 09:15:00 | 263.25 | 267.90 | 268.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-16 09:15:00 | 259.10 | 263.18 | 265.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 09:15:00 | 260.85 | 258.56 | 261.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 09:15:00 | 260.85 | 258.56 | 261.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 260.85 | 258.56 | 261.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 10:00:00 | 260.85 | 258.56 | 261.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 262.25 | 259.30 | 261.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 11:00:00 | 262.25 | 259.30 | 261.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 11:15:00 | 262.60 | 259.96 | 261.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 11:45:00 | 263.10 | 259.96 | 261.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 12:15:00 | 262.95 | 260.56 | 261.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 12:30:00 | 263.00 | 260.56 | 261.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 14:15:00 | 258.70 | 260.26 | 261.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 09:15:00 | 254.35 | 260.25 | 261.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 14:15:00 | 261.50 | 259.24 | 259.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2024-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 14:15:00 | 261.50 | 259.24 | 259.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 15:15:00 | 261.85 | 259.76 | 259.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 14:15:00 | 260.15 | 261.13 | 260.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 14:15:00 | 260.15 | 261.13 | 260.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 14:15:00 | 260.15 | 261.13 | 260.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 15:00:00 | 260.15 | 261.13 | 260.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 15:15:00 | 260.10 | 260.93 | 260.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-24 09:45:00 | 259.10 | 260.51 | 260.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 10:15:00 | 259.15 | 260.24 | 260.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-24 10:45:00 | 258.75 | 260.24 | 260.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2024-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 11:15:00 | 259.10 | 260.01 | 260.07 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 09:15:00 | 265.10 | 260.68 | 260.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 12:15:00 | 266.15 | 262.94 | 261.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 14:15:00 | 267.80 | 268.61 | 266.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 15:00:00 | 267.80 | 268.61 | 266.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 266.40 | 268.21 | 266.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 09:45:00 | 265.50 | 268.21 | 266.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 268.30 | 268.23 | 266.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 11:15:00 | 269.00 | 268.23 | 266.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-06 09:15:00 | 263.20 | 274.32 | 275.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 09:15:00 | 263.20 | 274.32 | 275.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 11:15:00 | 259.10 | 264.79 | 268.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 264.50 | 262.33 | 265.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 09:15:00 | 264.50 | 262.33 | 265.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 264.50 | 262.33 | 265.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 09:45:00 | 257.00 | 263.24 | 264.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 15:00:00 | 253.00 | 261.33 | 263.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 10:00:00 | 258.00 | 259.00 | 260.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-15 09:15:00 | 263.70 | 260.72 | 260.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 263.70 | 260.72 | 260.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 10:15:00 | 266.30 | 261.83 | 261.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 10:15:00 | 262.45 | 263.36 | 262.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 10:15:00 | 262.45 | 263.36 | 262.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 262.45 | 263.36 | 262.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:00:00 | 262.45 | 263.36 | 262.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 262.05 | 263.10 | 262.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:00:00 | 262.05 | 263.10 | 262.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 260.80 | 262.64 | 262.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:00:00 | 260.80 | 262.64 | 262.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 259.90 | 262.09 | 262.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:45:00 | 260.20 | 262.09 | 262.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 264.00 | 262.93 | 262.47 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2024-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 12:15:00 | 260.40 | 262.22 | 262.25 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 09:15:00 | 263.05 | 262.28 | 262.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 10:15:00 | 264.35 | 262.69 | 262.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 10:15:00 | 264.05 | 264.57 | 263.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-22 11:00:00 | 264.05 | 264.57 | 263.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 264.05 | 264.46 | 263.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:15:00 | 263.90 | 264.46 | 263.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 263.80 | 264.33 | 263.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:30:00 | 263.50 | 264.33 | 263.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 264.15 | 264.30 | 263.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:30:00 | 263.80 | 264.30 | 263.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 263.35 | 264.11 | 263.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:45:00 | 262.90 | 264.11 | 263.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 263.65 | 264.02 | 263.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 09:15:00 | 268.10 | 264.02 | 263.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 12:15:00 | 266.60 | 268.39 | 268.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 266.60 | 268.39 | 268.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 15:15:00 | 263.00 | 266.35 | 267.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 12:15:00 | 263.90 | 263.52 | 264.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-30 13:00:00 | 263.90 | 263.52 | 264.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 265.45 | 263.91 | 264.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:45:00 | 266.25 | 263.91 | 264.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 262.45 | 263.62 | 264.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 10:00:00 | 261.75 | 263.22 | 264.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 12:00:00 | 262.10 | 262.93 | 263.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 287.50 | 268.81 | 266.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 287.50 | 268.81 | 266.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 11:15:00 | 289.95 | 275.99 | 270.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 278.00 | 286.28 | 278.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 278.00 | 286.28 | 278.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 278.00 | 286.28 | 278.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 276.50 | 286.28 | 278.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 253.45 | 279.71 | 276.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 253.45 | 279.71 | 276.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 239.85 | 271.74 | 273.02 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 270.05 | 263.45 | 262.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 276.90 | 271.12 | 268.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 13:15:00 | 275.65 | 276.22 | 273.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 14:00:00 | 275.65 | 276.22 | 273.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 278.40 | 276.20 | 274.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 10:15:00 | 278.65 | 276.20 | 274.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 10:15:00 | 284.20 | 284.92 | 285.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 10:15:00 | 284.20 | 284.92 | 285.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 11:15:00 | 283.10 | 284.56 | 284.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 11:15:00 | 281.20 | 280.89 | 282.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-24 12:00:00 | 281.20 | 280.89 | 282.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 282.25 | 281.27 | 282.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:30:00 | 280.55 | 281.14 | 281.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 12:45:00 | 280.20 | 280.83 | 281.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 15:15:00 | 279.55 | 280.88 | 281.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 09:45:00 | 280.80 | 280.57 | 281.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 281.15 | 280.69 | 281.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:00:00 | 281.15 | 280.69 | 281.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 282.90 | 281.13 | 281.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:45:00 | 283.25 | 281.13 | 281.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 280.70 | 281.04 | 281.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 10:30:00 | 280.20 | 281.18 | 281.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 11:45:00 | 280.20 | 280.97 | 281.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 12:15:00 | 266.52 | 270.35 | 273.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 12:15:00 | 266.19 | 270.35 | 273.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 12:15:00 | 265.57 | 270.35 | 273.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 12:15:00 | 266.76 | 270.35 | 273.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 12:15:00 | 266.19 | 270.35 | 273.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 12:15:00 | 266.19 | 270.35 | 273.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-03 11:15:00 | 267.35 | 267.29 | 270.23 | SL hit (close>ema200) qty=0.50 sl=267.29 alert=retest2 |

### Cycle 93 — BUY (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 10:15:00 | 271.25 | 269.34 | 269.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 12:15:00 | 272.30 | 270.26 | 269.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 09:15:00 | 266.05 | 270.74 | 270.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 09:15:00 | 266.05 | 270.74 | 270.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 266.05 | 270.74 | 270.27 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 265.55 | 269.70 | 269.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 11:15:00 | 264.30 | 268.62 | 269.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 09:15:00 | 258.10 | 257.17 | 258.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 09:15:00 | 258.10 | 257.17 | 258.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 258.10 | 257.17 | 258.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:00:00 | 258.10 | 257.17 | 258.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 258.40 | 255.18 | 256.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 13:45:00 | 258.25 | 255.18 | 256.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 258.60 | 255.87 | 256.39 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 258.30 | 256.82 | 256.76 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 12:15:00 | 255.80 | 257.10 | 257.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 254.15 | 255.89 | 256.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 254.40 | 252.73 | 254.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 254.40 | 252.73 | 254.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 254.40 | 252.73 | 254.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:30:00 | 254.20 | 252.73 | 254.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 254.35 | 253.05 | 254.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 254.70 | 253.05 | 254.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 255.55 | 253.55 | 254.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:00:00 | 255.55 | 253.55 | 254.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 254.85 | 253.81 | 254.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 14:15:00 | 253.90 | 253.84 | 254.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 15:00:00 | 253.20 | 253.71 | 254.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:30:00 | 253.35 | 253.40 | 254.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-29 09:15:00 | 257.55 | 249.97 | 249.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 257.55 | 249.97 | 249.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 10:15:00 | 260.30 | 252.04 | 250.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 13:15:00 | 255.40 | 255.42 | 253.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 14:00:00 | 255.40 | 255.42 | 253.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 253.80 | 255.31 | 254.10 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 253.05 | 253.71 | 253.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 11:15:00 | 251.90 | 253.35 | 253.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 14:15:00 | 240.30 | 239.55 | 243.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-05 14:30:00 | 241.15 | 239.55 | 243.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 243.75 | 240.34 | 243.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 241.35 | 241.34 | 242.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 240.55 | 241.34 | 242.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 09:45:00 | 240.60 | 240.37 | 242.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:30:00 | 241.35 | 241.74 | 241.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 243.10 | 242.01 | 242.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 11:00:00 | 243.10 | 242.01 | 242.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-08 11:15:00 | 243.80 | 242.37 | 242.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 11:15:00 | 243.80 | 242.37 | 242.19 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 14:15:00 | 240.90 | 242.00 | 242.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 15:15:00 | 240.80 | 241.76 | 241.95 | Break + close below crossover candle low |

### Cycle 101 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 243.65 | 242.14 | 242.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 13:15:00 | 247.60 | 243.82 | 242.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 12:15:00 | 244.85 | 245.08 | 244.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 12:45:00 | 245.00 | 245.08 | 244.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 244.90 | 245.07 | 244.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 15:00:00 | 244.90 | 245.07 | 244.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 245.75 | 245.17 | 244.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:30:00 | 245.20 | 245.17 | 244.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 244.45 | 245.16 | 244.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:00:00 | 244.45 | 245.16 | 244.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 242.85 | 244.70 | 244.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:45:00 | 242.70 | 244.70 | 244.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 242.80 | 244.32 | 244.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 242.80 | 244.32 | 244.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 14:15:00 | 241.65 | 243.78 | 244.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 15:15:00 | 241.05 | 243.24 | 243.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 241.10 | 240.97 | 242.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 241.10 | 240.97 | 242.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 241.10 | 240.97 | 242.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 10:15:00 | 240.85 | 240.97 | 242.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 11:00:00 | 240.65 | 240.90 | 241.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 15:15:00 | 243.50 | 242.30 | 242.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 15:15:00 | 243.50 | 242.30 | 242.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 248.35 | 243.51 | 242.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 12:15:00 | 252.20 | 252.28 | 250.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 13:00:00 | 252.20 | 252.28 | 250.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 252.95 | 253.98 | 252.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 252.65 | 253.98 | 252.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 254.05 | 253.99 | 252.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 11:15:00 | 254.50 | 253.99 | 252.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 14:15:00 | 252.10 | 253.33 | 252.93 | SL hit (close<static) qty=1.00 sl=252.65 alert=retest2 |

### Cycle 104 — SELL (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 10:15:00 | 251.35 | 252.74 | 252.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 12:15:00 | 250.80 | 251.64 | 252.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 12:15:00 | 250.10 | 249.89 | 250.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 12:45:00 | 250.00 | 249.89 | 250.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 13:15:00 | 249.15 | 249.74 | 250.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 13:30:00 | 250.05 | 249.74 | 250.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 250.10 | 249.88 | 250.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 252.60 | 249.88 | 250.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 253.25 | 250.55 | 250.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:45:00 | 253.00 | 250.55 | 250.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 10:15:00 | 252.60 | 250.96 | 250.79 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 15:15:00 | 249.85 | 250.79 | 250.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 09:15:00 | 248.70 | 250.38 | 250.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 11:15:00 | 250.50 | 250.23 | 250.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 11:15:00 | 250.50 | 250.23 | 250.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 250.50 | 250.23 | 250.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:30:00 | 250.60 | 250.23 | 250.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 250.15 | 250.21 | 250.47 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2024-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 13:15:00 | 253.90 | 250.95 | 250.78 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 09:15:00 | 246.15 | 250.61 | 251.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 10:15:00 | 245.50 | 249.59 | 250.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 235.25 | 235.04 | 238.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 14:45:00 | 235.65 | 235.04 | 238.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 235.90 | 235.76 | 236.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:45:00 | 236.35 | 235.76 | 236.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 236.60 | 235.93 | 236.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:30:00 | 236.45 | 235.93 | 236.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 233.65 | 234.24 | 235.52 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 237.25 | 235.86 | 235.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 238.05 | 236.30 | 236.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 14:15:00 | 239.20 | 239.48 | 238.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 15:00:00 | 239.20 | 239.48 | 238.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 238.60 | 239.34 | 238.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:15:00 | 239.10 | 239.34 | 238.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 239.00 | 239.27 | 238.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 11:30:00 | 239.70 | 239.34 | 238.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 12:00:00 | 239.60 | 239.34 | 238.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 09:15:00 | 240.00 | 239.53 | 239.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 10:30:00 | 239.90 | 239.56 | 239.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 238.75 | 239.46 | 239.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:30:00 | 239.10 | 239.46 | 239.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 238.05 | 239.17 | 239.10 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-18 14:15:00 | 238.05 | 239.17 | 239.10 | SL hit (close<static) qty=1.00 sl=238.20 alert=retest2 |

### Cycle 110 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 237.40 | 238.82 | 238.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 236.75 | 238.41 | 238.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 235.95 | 235.89 | 237.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 235.95 | 235.89 | 237.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 237.35 | 236.35 | 237.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:30:00 | 236.25 | 237.03 | 237.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 09:15:00 | 240.20 | 237.19 | 237.28 | SL hit (close>static) qty=1.00 sl=238.55 alert=retest2 |

### Cycle 111 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 244.20 | 238.60 | 237.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 11:15:00 | 245.95 | 243.55 | 241.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 14:15:00 | 243.70 | 244.06 | 242.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 15:00:00 | 243.70 | 244.06 | 242.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 241.60 | 243.52 | 242.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 241.60 | 243.52 | 242.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 241.10 | 243.03 | 242.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:45:00 | 241.15 | 243.03 | 242.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 241.15 | 242.35 | 241.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 13:15:00 | 242.45 | 242.35 | 241.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 12:15:00 | 246.00 | 247.84 | 247.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 12:15:00 | 246.00 | 247.84 | 247.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 245.23 | 247.32 | 247.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 250.09 | 247.19 | 247.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 250.09 | 247.19 | 247.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 250.09 | 247.19 | 247.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:00:00 | 250.09 | 247.19 | 247.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 10:15:00 | 252.90 | 248.33 | 247.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 11:15:00 | 254.53 | 249.57 | 248.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 15:15:00 | 250.20 | 250.52 | 249.42 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-07 09:15:00 | 251.98 | 250.52 | 249.42 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 249.85 | 250.39 | 249.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 10:00:00 | 249.85 | 250.39 | 249.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 245.14 | 249.34 | 249.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-07 10:15:00 | 245.14 | 249.34 | 249.06 | SL hit (close<ema400) qty=1.00 sl=249.06 alert=retest1 |

### Cycle 114 — SELL (started 2024-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 11:15:00 | 244.73 | 248.42 | 248.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 13:15:00 | 241.27 | 246.08 | 247.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 246.79 | 245.06 | 246.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 10:15:00 | 246.79 | 245.06 | 246.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 246.79 | 245.06 | 246.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 246.79 | 245.06 | 246.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 246.55 | 245.36 | 246.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:45:00 | 247.39 | 245.36 | 246.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 246.11 | 245.51 | 246.41 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 249.17 | 247.01 | 246.90 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 11:15:00 | 245.25 | 246.85 | 246.86 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 251.70 | 247.54 | 247.10 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 10:15:00 | 244.85 | 246.97 | 247.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 11:15:00 | 243.00 | 246.17 | 246.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 09:15:00 | 244.09 | 244.05 | 245.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 244.09 | 244.05 | 245.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 244.09 | 244.05 | 245.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:45:00 | 244.57 | 244.05 | 245.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 244.65 | 243.96 | 244.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 15:00:00 | 244.65 | 243.96 | 244.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 15:15:00 | 245.50 | 244.27 | 244.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:15:00 | 247.44 | 244.27 | 244.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 245.92 | 244.60 | 244.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 10:45:00 | 245.00 | 244.62 | 244.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 12:15:00 | 244.84 | 244.73 | 244.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 12:15:00 | 244.81 | 243.22 | 243.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 12:15:00 | 244.81 | 243.22 | 243.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 13:15:00 | 246.75 | 243.93 | 243.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 12:15:00 | 246.00 | 246.11 | 245.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 12:15:00 | 246.00 | 246.11 | 245.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 246.00 | 246.11 | 245.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 13:00:00 | 246.00 | 246.11 | 245.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 245.60 | 245.90 | 245.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:45:00 | 245.13 | 245.90 | 245.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 245.20 | 245.76 | 245.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:15:00 | 242.11 | 245.76 | 245.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 242.18 | 245.05 | 244.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:15:00 | 239.82 | 245.05 | 244.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 240.15 | 244.07 | 244.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 237.85 | 242.82 | 243.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 13:15:00 | 236.99 | 235.75 | 238.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 14:00:00 | 236.99 | 235.75 | 238.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 14:15:00 | 238.12 | 236.22 | 238.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 14:45:00 | 239.60 | 236.22 | 238.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 15:15:00 | 237.79 | 236.53 | 238.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 09:15:00 | 237.57 | 236.53 | 238.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 240.43 | 237.31 | 238.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:00:00 | 240.43 | 237.31 | 238.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 243.50 | 238.55 | 239.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:45:00 | 244.56 | 238.55 | 239.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2024-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 11:15:00 | 242.55 | 239.35 | 239.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 13:15:00 | 244.45 | 240.82 | 240.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 09:15:00 | 237.59 | 241.33 | 240.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 09:15:00 | 237.59 | 241.33 | 240.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 237.59 | 241.33 | 240.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:00:00 | 237.59 | 241.33 | 240.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 237.89 | 240.64 | 240.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 235.81 | 240.64 | 240.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2024-10-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 12:15:00 | 239.01 | 240.00 | 240.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 13:15:00 | 237.03 | 239.40 | 239.79 | Break + close below crossover candle low |

### Cycle 123 — BUY (started 2024-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 09:15:00 | 249.19 | 241.29 | 240.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 10:15:00 | 251.60 | 243.35 | 241.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 252.83 | 253.33 | 250.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 14:00:00 | 252.83 | 253.33 | 250.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 251.24 | 252.61 | 250.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 250.93 | 252.61 | 250.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 253.47 | 252.78 | 251.08 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 248.15 | 250.68 | 250.84 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2024-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 14:15:00 | 252.30 | 251.10 | 251.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 13:15:00 | 258.00 | 253.55 | 252.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 14:15:00 | 262.50 | 262.90 | 260.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 15:00:00 | 262.50 | 262.90 | 260.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 260.05 | 262.34 | 260.52 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 257.70 | 259.60 | 259.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 256.60 | 259.00 | 259.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 259.05 | 257.98 | 258.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 259.05 | 257.98 | 258.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 259.05 | 257.98 | 258.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 259.05 | 257.98 | 258.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 258.45 | 258.07 | 258.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:30:00 | 259.30 | 258.07 | 258.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 259.10 | 258.28 | 258.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 12:45:00 | 259.35 | 258.28 | 258.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 13:15:00 | 257.30 | 258.08 | 258.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 10:30:00 | 257.00 | 257.94 | 258.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 14:15:00 | 244.15 | 248.87 | 252.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-18 12:15:00 | 242.15 | 241.04 | 244.72 | SL hit (close>ema200) qty=0.50 sl=241.04 alert=retest2 |

### Cycle 127 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 246.55 | 237.28 | 236.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 248.55 | 245.54 | 242.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 245.35 | 246.59 | 244.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 10:00:00 | 245.35 | 246.59 | 244.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 249.10 | 247.54 | 246.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 15:15:00 | 251.45 | 247.67 | 246.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 11:15:00 | 245.10 | 247.37 | 246.92 | SL hit (close<static) qty=1.00 sl=245.60 alert=retest2 |

### Cycle 128 — SELL (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 09:15:00 | 244.95 | 246.35 | 246.53 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 252.69 | 247.40 | 246.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 10:15:00 | 253.54 | 248.63 | 247.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 09:15:00 | 259.57 | 259.62 | 257.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-06 10:00:00 | 259.57 | 259.62 | 257.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 262.82 | 263.15 | 261.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:30:00 | 262.85 | 263.15 | 261.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 262.16 | 262.95 | 261.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:45:00 | 261.53 | 262.95 | 261.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 11:15:00 | 262.53 | 262.87 | 262.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 11:30:00 | 262.08 | 262.87 | 262.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 12:15:00 | 261.94 | 262.68 | 262.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 12:45:00 | 261.88 | 262.68 | 262.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 13:15:00 | 262.44 | 262.64 | 262.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 14:15:00 | 262.94 | 262.64 | 262.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 14:45:00 | 263.45 | 262.62 | 262.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 15:15:00 | 263.65 | 262.62 | 262.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 10:15:00 | 263.08 | 262.78 | 262.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 13:15:00 | 261.82 | 262.94 | 262.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 13:45:00 | 261.92 | 262.94 | 262.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 261.07 | 262.57 | 262.42 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-11 14:15:00 | 261.07 | 262.57 | 262.42 | SL hit (close<static) qty=1.00 sl=261.52 alert=retest2 |

### Cycle 130 — SELL (started 2024-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 15:15:00 | 260.42 | 262.14 | 262.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 258.23 | 261.36 | 261.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 12:15:00 | 257.20 | 256.94 | 258.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 13:00:00 | 257.20 | 256.94 | 258.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 258.21 | 257.59 | 258.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 258.60 | 257.59 | 258.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 257.24 | 257.40 | 258.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:15:00 | 258.07 | 257.40 | 258.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 258.33 | 257.59 | 258.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:15:00 | 258.37 | 257.59 | 258.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 257.57 | 257.58 | 258.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:15:00 | 257.23 | 257.95 | 258.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 10:15:00 | 258.63 | 258.08 | 258.33 | SL hit (close>static) qty=1.00 sl=258.60 alert=retest2 |

### Cycle 131 — BUY (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 15:15:00 | 246.85 | 246.29 | 246.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 10:15:00 | 248.01 | 246.77 | 246.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 11:15:00 | 245.73 | 246.56 | 246.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 11:15:00 | 245.73 | 246.56 | 246.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 245.73 | 246.56 | 246.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:00:00 | 245.73 | 246.56 | 246.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 245.61 | 246.37 | 246.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:00:00 | 245.61 | 246.37 | 246.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 13:15:00 | 246.00 | 246.30 | 246.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 14:15:00 | 244.98 | 246.03 | 246.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 11:15:00 | 246.48 | 245.49 | 245.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 11:15:00 | 246.48 | 245.49 | 245.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 246.48 | 245.49 | 245.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:45:00 | 246.73 | 245.49 | 245.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 243.51 | 245.09 | 245.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 13:15:00 | 242.33 | 245.09 | 245.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 10:15:00 | 242.97 | 240.50 | 240.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 10:15:00 | 243.91 | 241.19 | 240.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 10:15:00 | 243.91 | 241.19 | 240.94 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 232.38 | 239.94 | 240.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 12:15:00 | 228.97 | 234.94 | 237.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 231.99 | 231.50 | 235.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:30:00 | 232.65 | 231.50 | 235.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 233.25 | 231.85 | 233.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:00:00 | 233.25 | 231.85 | 233.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 232.99 | 232.08 | 233.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 231.75 | 232.22 | 233.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:15:00 | 232.28 | 232.36 | 233.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:45:00 | 231.30 | 231.88 | 232.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:15:00 | 220.16 | 223.71 | 226.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:15:00 | 220.67 | 223.71 | 226.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:15:00 | 219.74 | 223.71 | 226.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 222.48 | 221.59 | 224.87 | SL hit (close>ema200) qty=0.50 sl=221.59 alert=retest2 |

### Cycle 135 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 229.10 | 225.03 | 224.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 10:15:00 | 230.97 | 228.62 | 227.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 230.18 | 231.24 | 229.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 230.18 | 231.24 | 229.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 229.25 | 230.84 | 229.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:30:00 | 229.18 | 230.84 | 229.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 230.10 | 230.69 | 229.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 229.94 | 230.69 | 229.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 230.30 | 230.61 | 229.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:30:00 | 229.85 | 230.61 | 229.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 230.48 | 230.59 | 229.82 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 227.30 | 229.43 | 229.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 12:15:00 | 225.55 | 228.66 | 229.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 229.05 | 228.10 | 228.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 229.05 | 228.10 | 228.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 229.05 | 228.10 | 228.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 229.05 | 228.10 | 228.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 228.29 | 228.14 | 228.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 09:15:00 | 225.87 | 228.14 | 228.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 10:45:00 | 227.59 | 228.03 | 228.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 12:15:00 | 229.71 | 228.35 | 228.63 | SL hit (close>static) qty=1.00 sl=229.18 alert=retest2 |

### Cycle 137 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 218.21 | 213.08 | 212.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 10:15:00 | 219.11 | 214.29 | 213.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 12:15:00 | 218.90 | 219.01 | 217.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 12:45:00 | 218.34 | 219.01 | 217.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 219.01 | 219.29 | 217.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 219.01 | 219.29 | 217.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 217.91 | 219.01 | 217.86 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2025-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 15:15:00 | 216.25 | 217.12 | 217.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 214.15 | 216.53 | 216.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 212.56 | 211.07 | 212.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 11:15:00 | 212.56 | 211.07 | 212.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 212.56 | 211.07 | 212.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 212.56 | 211.07 | 212.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 213.65 | 211.59 | 212.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 213.65 | 211.59 | 212.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 211.76 | 211.62 | 212.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 213.37 | 211.62 | 212.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 212.65 | 211.83 | 212.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 212.65 | 211.83 | 212.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 212.52 | 211.97 | 212.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:15:00 | 212.95 | 211.97 | 212.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 212.69 | 212.11 | 212.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 10:15:00 | 212.30 | 212.11 | 212.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 10:45:00 | 212.26 | 212.18 | 212.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 11:15:00 | 212.35 | 212.18 | 212.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 201.69 | 205.62 | 208.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 201.65 | 205.62 | 208.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 201.73 | 205.62 | 208.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 13:15:00 | 205.55 | 205.16 | 207.15 | SL hit (close>ema200) qty=0.50 sl=205.16 alert=retest2 |

### Cycle 139 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 209.41 | 207.12 | 206.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 211.03 | 208.80 | 207.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 211.00 | 211.86 | 210.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 211.00 | 211.86 | 210.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 210.73 | 211.63 | 210.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 210.73 | 211.63 | 210.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 210.53 | 211.41 | 210.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 12:00:00 | 210.53 | 211.41 | 210.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 12:15:00 | 210.12 | 211.15 | 210.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:00:00 | 210.12 | 211.15 | 210.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 210.32 | 210.99 | 210.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:45:00 | 209.58 | 210.99 | 210.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 210.57 | 210.90 | 210.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:30:00 | 210.39 | 210.90 | 210.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 210.09 | 210.74 | 210.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 207.71 | 210.74 | 210.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 207.83 | 210.16 | 210.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:30:00 | 206.82 | 210.16 | 210.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 207.40 | 209.61 | 209.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 207.16 | 208.27 | 208.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 10:15:00 | 196.73 | 196.19 | 198.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 11:00:00 | 196.73 | 196.19 | 198.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 201.51 | 197.21 | 197.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:00:00 | 201.51 | 197.21 | 197.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 203.73 | 198.51 | 198.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 205.40 | 202.05 | 200.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 205.98 | 206.64 | 204.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 205.98 | 206.64 | 204.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 204.95 | 206.31 | 204.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:00:00 | 204.95 | 206.31 | 204.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 204.89 | 206.02 | 204.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:45:00 | 204.69 | 206.02 | 204.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 206.03 | 206.02 | 204.96 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 204.02 | 204.66 | 204.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 12:15:00 | 202.10 | 203.86 | 204.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 09:15:00 | 203.72 | 203.39 | 203.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 203.72 | 203.39 | 203.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 203.72 | 203.39 | 203.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:45:00 | 202.40 | 203.51 | 203.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 12:15:00 | 202.94 | 203.51 | 203.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 12:45:00 | 202.14 | 203.17 | 203.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 09:15:00 | 206.76 | 203.52 | 203.62 | SL hit (close>static) qty=1.00 sl=204.80 alert=retest2 |

### Cycle 143 — BUY (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 10:15:00 | 205.85 | 203.99 | 203.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 10:15:00 | 208.09 | 206.32 | 205.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 10:15:00 | 221.55 | 222.15 | 219.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:30:00 | 221.36 | 222.15 | 219.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 220.80 | 222.74 | 221.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 14:00:00 | 220.80 | 222.74 | 221.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 220.33 | 222.26 | 221.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 15:00:00 | 220.33 | 222.26 | 221.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 229.25 | 228.71 | 227.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 15:00:00 | 231.78 | 229.56 | 228.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 10:00:00 | 231.64 | 230.35 | 229.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 224.77 | 232.82 | 232.63 | SL hit (close<static) qty=1.00 sl=227.01 alert=retest2 |

### Cycle 144 — SELL (started 2025-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 10:15:00 | 228.85 | 232.03 | 232.29 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 235.95 | 232.08 | 231.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 15:15:00 | 236.50 | 234.36 | 233.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 09:15:00 | 231.77 | 233.84 | 232.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-09 09:15:00 | 231.77 | 233.84 | 232.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 231.77 | 233.84 | 232.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 10:00:00 | 231.77 | 233.84 | 232.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 10:15:00 | 230.18 | 233.11 | 232.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 11:00:00 | 230.18 | 233.11 | 232.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 11:15:00 | 231.39 | 232.76 | 232.54 | EMA400 retest candle locked (from upside) |

### Cycle 146 — SELL (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 13:15:00 | 229.87 | 232.02 | 232.23 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 233.81 | 232.23 | 232.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 13:15:00 | 234.96 | 233.27 | 232.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 15:15:00 | 251.97 | 252.03 | 248.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 09:15:00 | 251.33 | 252.03 | 248.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 248.52 | 251.33 | 248.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 248.52 | 251.33 | 248.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 248.98 | 250.86 | 248.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:30:00 | 247.76 | 250.86 | 248.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 249.17 | 250.52 | 248.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 13:15:00 | 249.90 | 250.27 | 248.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 247.83 | 251.20 | 250.64 | SL hit (close<static) qty=1.00 sl=248.50 alert=retest2 |

### Cycle 148 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 246.39 | 250.24 | 250.25 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 252.67 | 249.96 | 249.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 256.67 | 252.33 | 251.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 15:15:00 | 252.09 | 253.34 | 252.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 15:15:00 | 252.09 | 253.34 | 252.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 252.09 | 253.34 | 252.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 250.99 | 253.34 | 252.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 251.10 | 252.89 | 252.16 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2025-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 13:15:00 | 250.07 | 251.53 | 251.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 249.60 | 250.92 | 251.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 10:15:00 | 249.65 | 249.09 | 249.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 10:15:00 | 249.65 | 249.09 | 249.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 249.65 | 249.09 | 249.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:00:00 | 249.65 | 249.09 | 249.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 249.92 | 249.26 | 249.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 249.92 | 249.26 | 249.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 249.06 | 249.22 | 249.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 13:15:00 | 249.69 | 249.22 | 249.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 249.84 | 249.34 | 249.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:00:00 | 249.84 | 249.34 | 249.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 248.93 | 249.26 | 249.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:45:00 | 249.30 | 249.26 | 249.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 249.50 | 249.31 | 249.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 249.21 | 249.31 | 249.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 245.30 | 248.51 | 249.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:30:00 | 244.01 | 247.35 | 248.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 14:15:00 | 231.81 | 241.27 | 245.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-05-08 13:15:00 | 219.61 | 222.94 | 229.09 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 151 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 228.23 | 224.57 | 224.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 233.41 | 227.02 | 225.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 12:15:00 | 231.79 | 232.04 | 230.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 12:45:00 | 231.16 | 232.04 | 230.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 236.91 | 239.15 | 238.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 236.91 | 239.15 | 238.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 236.68 | 238.65 | 237.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 237.75 | 238.51 | 237.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 12:15:00 | 237.55 | 238.51 | 238.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 09:15:00 | 238.77 | 241.14 | 241.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 09:15:00 | 238.77 | 241.14 | 241.19 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 15:15:00 | 241.20 | 241.11 | 241.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 242.30 | 241.44 | 241.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 09:15:00 | 241.90 | 242.23 | 241.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 241.90 | 242.23 | 241.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 241.90 | 242.23 | 241.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:45:00 | 242.15 | 242.23 | 241.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 241.81 | 242.15 | 241.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:45:00 | 242.14 | 242.15 | 241.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 241.91 | 242.10 | 241.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:45:00 | 241.92 | 242.10 | 241.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 242.10 | 242.10 | 241.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:30:00 | 241.47 | 242.10 | 241.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 241.65 | 242.01 | 241.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:00:00 | 241.65 | 242.01 | 241.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 243.30 | 242.27 | 241.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 15:15:00 | 243.64 | 242.27 | 241.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:00:00 | 243.66 | 242.85 | 242.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 13:15:00 | 249.45 | 251.99 | 252.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 249.45 | 251.99 | 252.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 09:15:00 | 242.50 | 249.79 | 251.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 248.65 | 247.05 | 248.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 248.65 | 247.05 | 248.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 248.65 | 247.05 | 248.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:30:00 | 249.07 | 247.05 | 248.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 249.96 | 247.63 | 248.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 11:00:00 | 249.96 | 247.63 | 248.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 11:15:00 | 249.00 | 247.90 | 248.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 15:15:00 | 247.80 | 248.24 | 248.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 235.41 | 241.75 | 243.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 15:15:00 | 239.80 | 239.63 | 241.51 | SL hit (close>ema200) qty=0.50 sl=239.63 alert=retest2 |

### Cycle 155 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 239.11 | 235.05 | 234.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 242.42 | 239.02 | 238.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 244.61 | 246.40 | 244.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 10:00:00 | 244.61 | 246.40 | 244.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 245.34 | 246.19 | 244.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 245.34 | 246.19 | 244.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 244.42 | 245.84 | 244.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:00:00 | 244.42 | 245.84 | 244.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 247.10 | 246.09 | 244.45 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 14:15:00 | 243.00 | 244.33 | 244.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 10:15:00 | 242.14 | 243.48 | 244.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 243.63 | 241.55 | 242.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 243.63 | 241.55 | 242.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 243.63 | 241.55 | 242.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 243.19 | 241.55 | 242.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 244.00 | 242.04 | 242.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:45:00 | 243.40 | 242.19 | 242.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 11:15:00 | 240.36 | 239.36 | 239.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — BUY (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 11:15:00 | 240.36 | 239.36 | 239.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 244.03 | 240.92 | 240.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 10:15:00 | 247.25 | 247.42 | 245.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 11:00:00 | 247.25 | 247.42 | 245.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 245.51 | 246.41 | 245.66 | EMA400 retest candle locked (from upside) |

### Cycle 158 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 244.40 | 245.31 | 245.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 241.91 | 244.63 | 245.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 244.88 | 244.27 | 244.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 12:15:00 | 244.88 | 244.27 | 244.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 244.88 | 244.27 | 244.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:45:00 | 244.97 | 244.27 | 244.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 243.40 | 244.09 | 244.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 14:45:00 | 243.22 | 243.96 | 244.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 242.75 | 243.86 | 244.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 246.90 | 242.77 | 242.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 14:15:00 | 246.90 | 242.77 | 242.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-25 09:15:00 | 247.30 | 244.35 | 243.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 12:15:00 | 244.05 | 244.45 | 243.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 12:15:00 | 244.05 | 244.45 | 243.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 244.05 | 244.45 | 243.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 245.59 | 244.04 | 243.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 10:00:00 | 245.30 | 244.30 | 243.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 242.95 | 244.03 | 243.67 | SL hit (close<static) qty=1.00 sl=243.01 alert=retest2 |

### Cycle 160 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 240.95 | 243.24 | 243.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 240.39 | 242.67 | 243.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 242.55 | 242.00 | 242.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 242.55 | 242.00 | 242.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 242.55 | 242.00 | 242.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 242.55 | 242.00 | 242.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 241.15 | 241.83 | 242.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:15:00 | 242.94 | 241.83 | 242.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 242.65 | 241.99 | 242.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:45:00 | 240.56 | 241.96 | 242.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 14:15:00 | 240.81 | 241.96 | 242.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 14:15:00 | 241.24 | 239.18 | 239.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 241.24 | 239.18 | 239.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 12:15:00 | 242.39 | 240.80 | 240.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 241.14 | 241.25 | 240.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 241.14 | 241.25 | 240.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 241.14 | 241.25 | 240.70 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 12:15:00 | 237.85 | 240.25 | 240.36 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 242.00 | 240.46 | 240.42 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 12:15:00 | 239.61 | 240.39 | 240.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 239.35 | 240.05 | 240.27 | Break + close below crossover candle low |

### Cycle 165 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 243.74 | 240.61 | 240.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 244.04 | 242.29 | 241.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 13:15:00 | 242.90 | 243.59 | 242.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 13:15:00 | 242.90 | 243.59 | 242.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 242.90 | 243.59 | 242.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:30:00 | 242.72 | 243.59 | 242.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 242.91 | 243.46 | 242.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 243.97 | 243.35 | 242.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 11:15:00 | 241.91 | 243.06 | 242.75 | SL hit (close<static) qty=1.00 sl=242.64 alert=retest2 |

### Cycle 166 — SELL (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 13:15:00 | 241.53 | 242.53 | 242.55 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 14:15:00 | 242.99 | 242.55 | 242.51 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 15:15:00 | 242.19 | 242.48 | 242.48 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 244.36 | 242.85 | 242.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 14:15:00 | 246.98 | 244.42 | 243.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 11:15:00 | 245.45 | 245.77 | 244.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 12:00:00 | 245.45 | 245.77 | 244.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 244.65 | 245.55 | 244.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:00:00 | 244.65 | 245.55 | 244.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 244.90 | 245.42 | 244.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:45:00 | 244.81 | 245.42 | 244.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 245.20 | 245.55 | 245.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:30:00 | 245.00 | 245.55 | 245.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 244.50 | 245.34 | 245.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 244.05 | 245.34 | 245.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 243.43 | 244.96 | 244.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 243.43 | 244.96 | 244.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 243.54 | 244.68 | 244.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 10:15:00 | 242.23 | 243.97 | 244.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 241.96 | 241.89 | 242.98 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 11:45:00 | 241.23 | 241.65 | 242.68 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 233.95 | 233.34 | 234.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 234.55 | 233.34 | 234.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 233.71 | 233.54 | 234.55 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-01 13:15:00 | 234.59 | 233.87 | 234.53 | SL hit (close>ema400) qty=1.00 sl=234.53 alert=retest1 |

### Cycle 171 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 237.38 | 235.27 | 235.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 238.70 | 236.72 | 235.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 237.58 | 237.72 | 236.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:00:00 | 237.58 | 237.72 | 236.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 236.43 | 237.46 | 236.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 236.43 | 237.46 | 236.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 236.33 | 237.23 | 236.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:15:00 | 235.90 | 237.23 | 236.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 234.40 | 236.39 | 236.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 233.89 | 235.89 | 236.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 234.14 | 234.06 | 235.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 14:00:00 | 234.14 | 234.06 | 235.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 235.90 | 234.56 | 235.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 235.90 | 234.56 | 235.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 236.23 | 234.89 | 235.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:30:00 | 236.26 | 234.89 | 235.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 235.81 | 235.34 | 235.31 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-09-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 14:15:00 | 234.64 | 235.17 | 235.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 10:15:00 | 233.97 | 234.76 | 235.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 14:15:00 | 234.40 | 234.36 | 234.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 15:00:00 | 234.40 | 234.36 | 234.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 234.41 | 234.37 | 234.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 235.96 | 234.37 | 234.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 237.21 | 234.94 | 234.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 237.92 | 235.53 | 235.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 13:15:00 | 238.48 | 238.55 | 237.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 14:00:00 | 238.48 | 238.55 | 237.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 238.30 | 238.40 | 237.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 14:15:00 | 239.69 | 237.74 | 237.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 10:15:00 | 239.32 | 238.40 | 237.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 10:45:00 | 239.30 | 238.56 | 238.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 11:30:00 | 240.34 | 238.73 | 238.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 251.16 | 252.14 | 250.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 250.89 | 252.14 | 250.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 252.29 | 252.01 | 250.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 249.86 | 252.01 | 250.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 255.62 | 254.71 | 253.53 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 251.09 | 253.08 | 253.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 251.09 | 253.08 | 253.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 247.98 | 250.91 | 252.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 252.07 | 250.45 | 251.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 252.07 | 250.45 | 251.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 252.07 | 250.45 | 251.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 252.07 | 250.45 | 251.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 251.05 | 250.57 | 251.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 251.87 | 250.57 | 251.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 250.84 | 250.62 | 251.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:45:00 | 251.25 | 250.62 | 251.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 253.58 | 251.21 | 251.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:00:00 | 253.58 | 251.21 | 251.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 253.50 | 251.67 | 251.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 253.50 | 251.67 | 251.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 14:15:00 | 254.02 | 252.14 | 251.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 09:15:00 | 257.79 | 253.58 | 252.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 10:15:00 | 256.75 | 257.52 | 255.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 10:15:00 | 256.75 | 257.52 | 255.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 256.75 | 257.52 | 255.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:45:00 | 256.45 | 257.52 | 255.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 255.70 | 257.15 | 255.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 255.70 | 257.15 | 255.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 258.15 | 257.35 | 255.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 13:30:00 | 260.35 | 257.73 | 256.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 260.40 | 262.52 | 262.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 260.40 | 262.52 | 262.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 258.90 | 261.80 | 262.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 13:15:00 | 262.55 | 261.67 | 262.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 13:15:00 | 262.55 | 261.67 | 262.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 262.55 | 261.67 | 262.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:45:00 | 262.80 | 261.67 | 262.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 261.90 | 261.72 | 262.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 09:30:00 | 261.30 | 261.71 | 262.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 11:15:00 | 263.60 | 262.19 | 262.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 263.60 | 262.19 | 262.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 13:15:00 | 264.80 | 262.94 | 262.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 265.60 | 267.11 | 266.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 265.60 | 267.11 | 266.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 265.60 | 267.11 | 266.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 265.60 | 267.11 | 266.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 265.70 | 266.83 | 266.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:15:00 | 265.00 | 266.83 | 266.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 265.05 | 266.47 | 266.09 | EMA400 retest candle locked (from upside) |

### Cycle 180 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 264.20 | 265.77 | 265.82 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 267.80 | 265.97 | 265.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 268.75 | 266.53 | 266.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 10:15:00 | 267.20 | 267.62 | 266.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 10:15:00 | 267.20 | 267.62 | 266.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 267.20 | 267.62 | 266.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:00:00 | 267.20 | 267.62 | 266.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 266.55 | 267.40 | 266.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:00:00 | 266.55 | 267.40 | 266.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 267.20 | 267.36 | 266.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:45:00 | 265.90 | 267.36 | 266.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 268.00 | 267.49 | 267.07 | EMA400 retest candle locked (from upside) |

### Cycle 182 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 264.60 | 266.51 | 266.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 263.15 | 265.69 | 266.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 266.40 | 265.36 | 265.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 266.40 | 265.36 | 265.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 266.40 | 265.36 | 265.87 | EMA400 retest candle locked (from downside) |

### Cycle 183 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 269.30 | 266.73 | 266.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 12:15:00 | 271.50 | 267.68 | 266.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 11:15:00 | 269.65 | 269.87 | 268.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 11:15:00 | 269.65 | 269.87 | 268.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 269.65 | 269.87 | 268.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:00:00 | 269.65 | 269.87 | 268.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 269.45 | 269.78 | 268.89 | EMA400 retest candle locked (from upside) |

### Cycle 184 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 267.50 | 268.27 | 268.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 266.00 | 267.36 | 267.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 269.80 | 267.27 | 267.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 269.80 | 267.27 | 267.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 269.80 | 267.27 | 267.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:30:00 | 269.25 | 267.27 | 267.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 269.55 | 267.73 | 267.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 11:15:00 | 269.00 | 267.73 | 267.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 11:15:00 | 269.80 | 268.14 | 267.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 269.80 | 268.14 | 267.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 12:15:00 | 270.80 | 268.67 | 268.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 10:15:00 | 275.05 | 275.39 | 273.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 11:00:00 | 275.05 | 275.39 | 273.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 275.20 | 275.07 | 273.98 | EMA400 retest candle locked (from upside) |

### Cycle 186 — SELL (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 15:15:00 | 272.40 | 273.46 | 273.55 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 274.85 | 273.74 | 273.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 11:15:00 | 279.00 | 274.94 | 274.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 12:15:00 | 287.65 | 288.63 | 285.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 12:30:00 | 286.80 | 288.63 | 285.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 286.25 | 287.55 | 286.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:30:00 | 285.50 | 287.55 | 286.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 287.20 | 287.48 | 286.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 12:30:00 | 288.45 | 287.29 | 286.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 12:45:00 | 288.15 | 287.56 | 287.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 282.00 | 286.48 | 286.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 282.00 | 286.48 | 286.81 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 14:15:00 | 287.05 | 284.95 | 284.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 291.25 | 286.44 | 285.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 14:15:00 | 288.00 | 288.03 | 286.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 15:00:00 | 288.00 | 288.03 | 286.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 288.65 | 288.18 | 287.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:00:00 | 290.40 | 288.75 | 287.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:45:00 | 290.00 | 288.97 | 288.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 11:15:00 | 290.10 | 288.97 | 288.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 13:45:00 | 289.95 | 291.00 | 290.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 288.35 | 290.47 | 290.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 14:45:00 | 288.20 | 290.47 | 290.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 288.10 | 290.00 | 289.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 286.95 | 290.00 | 289.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 286.20 | 289.24 | 289.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 286.20 | 289.24 | 289.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 284.95 | 288.38 | 289.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 284.40 | 283.67 | 285.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 10:00:00 | 284.40 | 283.67 | 285.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 284.85 | 283.94 | 285.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:00:00 | 284.85 | 283.94 | 285.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 286.85 | 284.61 | 285.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 286.85 | 284.61 | 285.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 287.50 | 285.19 | 285.46 | EMA400 retest candle locked (from downside) |

### Cycle 191 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 291.20 | 286.68 | 286.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 09:15:00 | 296.40 | 291.04 | 289.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 14:15:00 | 297.25 | 298.75 | 295.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 14:15:00 | 297.25 | 298.75 | 295.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 297.25 | 298.75 | 295.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 297.25 | 298.75 | 295.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 292.05 | 297.14 | 295.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 292.05 | 297.14 | 295.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 287.55 | 295.22 | 294.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 287.55 | 295.22 | 294.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — SELL (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 11:15:00 | 288.50 | 293.88 | 294.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 13:15:00 | 286.95 | 291.56 | 293.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 289.25 | 288.41 | 289.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 10:00:00 | 289.25 | 288.41 | 289.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 290.15 | 288.76 | 289.94 | EMA400 retest candle locked (from downside) |

### Cycle 193 — BUY (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 14:15:00 | 292.95 | 290.65 | 290.52 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 288.00 | 290.34 | 290.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 287.45 | 289.76 | 290.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 287.00 | 286.24 | 287.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:00:00 | 287.00 | 286.24 | 287.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 288.20 | 286.63 | 287.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 288.20 | 286.63 | 287.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 288.15 | 286.94 | 287.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:15:00 | 289.00 | 286.94 | 287.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 289.50 | 287.45 | 287.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:45:00 | 289.70 | 287.45 | 287.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 289.30 | 288.25 | 288.25 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 287.30 | 288.10 | 288.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 15:15:00 | 285.10 | 286.96 | 287.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 13:15:00 | 285.05 | 284.56 | 285.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 14:00:00 | 285.05 | 284.56 | 285.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 284.30 | 284.11 | 284.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:45:00 | 286.20 | 284.11 | 284.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 285.30 | 284.35 | 284.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:00:00 | 285.30 | 284.35 | 284.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 285.05 | 284.49 | 284.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 14:15:00 | 284.90 | 284.49 | 284.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 14:45:00 | 285.00 | 284.63 | 284.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 15:15:00 | 284.55 | 284.63 | 284.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 09:45:00 | 285.00 | 283.12 | 283.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 286.05 | 283.70 | 283.86 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 286.00 | 284.16 | 284.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — BUY (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 11:15:00 | 286.00 | 284.16 | 284.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 12:15:00 | 287.45 | 284.82 | 284.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 15:15:00 | 287.90 | 287.95 | 286.75 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 09:15:00 | 291.45 | 287.95 | 286.75 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 292.45 | 292.96 | 292.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 291.95 | 292.96 | 292.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 292.10 | 292.79 | 292.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:00:00 | 292.10 | 292.79 | 292.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 292.10 | 292.65 | 292.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 292.10 | 292.65 | 292.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 291.85 | 292.49 | 292.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-24 11:15:00 | 291.85 | 292.49 | 292.06 | SL hit (close<ema400) qty=1.00 sl=292.06 alert=retest1 |

### Cycle 198 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 290.45 | 291.59 | 291.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 289.30 | 290.93 | 291.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 11:15:00 | 289.65 | 288.92 | 289.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 11:15:00 | 289.65 | 288.92 | 289.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 289.65 | 288.92 | 289.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:00:00 | 289.65 | 288.92 | 289.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 288.70 | 288.87 | 289.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 13:15:00 | 288.15 | 288.87 | 289.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:45:00 | 287.65 | 288.15 | 288.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 11:30:00 | 288.30 | 288.04 | 288.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 13:15:00 | 291.85 | 289.00 | 289.14 | SL hit (close>static) qty=1.00 sl=289.75 alert=retest2 |

### Cycle 199 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 293.50 | 289.90 | 289.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 296.00 | 291.60 | 290.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 303.50 | 305.98 | 304.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 13:15:00 | 303.50 | 305.98 | 304.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 303.50 | 305.98 | 304.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:00:00 | 303.50 | 305.98 | 304.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 305.60 | 305.91 | 304.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 305.90 | 305.65 | 304.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 302.20 | 305.51 | 305.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 302.20 | 305.51 | 305.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 300.35 | 304.48 | 305.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 303.80 | 302.77 | 304.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 303.80 | 302.77 | 304.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 303.80 | 302.77 | 304.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:00:00 | 303.80 | 302.77 | 304.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 303.00 | 302.82 | 303.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:30:00 | 304.50 | 302.82 | 303.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 302.70 | 302.90 | 303.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:45:00 | 303.85 | 302.90 | 303.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 297.65 | 301.00 | 302.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 11:00:00 | 296.70 | 300.14 | 302.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 12:15:00 | 306.60 | 302.38 | 301.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 306.60 | 302.38 | 301.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 14:15:00 | 307.80 | 304.15 | 302.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 12:15:00 | 307.45 | 307.95 | 305.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 13:00:00 | 307.45 | 307.95 | 305.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 307.50 | 307.75 | 306.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:45:00 | 305.75 | 307.75 | 306.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 305.75 | 307.35 | 306.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:00:00 | 305.75 | 307.35 | 306.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 305.85 | 307.05 | 306.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 12:00:00 | 305.85 | 307.05 | 306.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 305.95 | 306.83 | 306.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 13:00:00 | 305.95 | 306.83 | 306.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 306.30 | 306.72 | 306.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 13:45:00 | 306.35 | 306.72 | 306.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 307.40 | 306.86 | 306.22 | EMA400 retest candle locked (from upside) |

### Cycle 202 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 303.15 | 305.75 | 305.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 300.40 | 303.96 | 304.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 303.40 | 303.18 | 304.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 303.40 | 303.18 | 304.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 303.40 | 303.18 | 304.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:45:00 | 304.30 | 303.18 | 304.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 299.80 | 302.50 | 303.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 14:15:00 | 297.40 | 300.97 | 302.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 306.80 | 301.57 | 302.55 | SL hit (close>static) qty=1.00 sl=304.25 alert=retest2 |

### Cycle 203 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 306.20 | 303.60 | 303.31 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 300.70 | 303.05 | 303.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 298.10 | 302.06 | 302.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 299.80 | 299.49 | 301.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 09:45:00 | 299.30 | 299.49 | 301.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 301.95 | 299.25 | 300.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 301.95 | 299.25 | 300.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 303.00 | 300.00 | 300.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 302.55 | 300.00 | 300.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 303.35 | 301.24 | 301.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 12:15:00 | 305.20 | 302.03 | 301.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 11:15:00 | 303.20 | 304.34 | 303.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 11:15:00 | 303.20 | 304.34 | 303.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 303.20 | 304.34 | 303.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 303.20 | 304.34 | 303.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 303.10 | 304.09 | 303.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 13:15:00 | 303.50 | 304.09 | 303.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 304.00 | 304.07 | 303.16 | EMA400 retest candle locked (from upside) |

### Cycle 206 — SELL (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 10:15:00 | 299.90 | 302.60 | 302.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 299.00 | 301.49 | 302.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 284.25 | 279.89 | 285.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 284.25 | 279.89 | 285.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 284.25 | 279.89 | 285.00 | EMA400 retest candle locked (from downside) |

### Cycle 207 — BUY (started 2026-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 12:15:00 | 287.70 | 285.58 | 285.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 291.95 | 286.86 | 286.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 288.85 | 289.58 | 288.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 288.85 | 289.58 | 288.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 288.85 | 289.58 | 288.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:45:00 | 288.50 | 289.58 | 288.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 286.40 | 288.94 | 288.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 285.60 | 288.94 | 288.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 287.25 | 288.60 | 288.23 | EMA400 retest candle locked (from upside) |

### Cycle 208 — SELL (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 13:15:00 | 286.70 | 287.76 | 287.88 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 289.50 | 288.11 | 288.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 293.15 | 289.15 | 288.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 14:15:00 | 290.50 | 290.61 | 289.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-09 15:00:00 | 290.50 | 290.61 | 289.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 291.35 | 290.76 | 289.87 | EMA400 retest candle locked (from upside) |

### Cycle 210 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 286.55 | 289.71 | 289.82 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 11:15:00 | 290.50 | 290.01 | 289.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 13:15:00 | 292.05 | 290.52 | 290.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 10:15:00 | 290.70 | 290.85 | 290.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 10:15:00 | 290.70 | 290.85 | 290.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 290.70 | 290.85 | 290.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:45:00 | 290.45 | 290.85 | 290.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 287.90 | 290.26 | 290.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 287.90 | 290.26 | 290.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 288.75 | 289.96 | 290.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 286.30 | 289.12 | 289.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 14:15:00 | 287.70 | 287.47 | 288.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 15:00:00 | 287.70 | 287.47 | 288.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 287.90 | 287.59 | 288.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:00:00 | 287.90 | 287.59 | 288.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 289.80 | 288.04 | 288.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:00:00 | 289.80 | 288.04 | 288.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 290.90 | 288.61 | 288.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:30:00 | 290.70 | 288.61 | 288.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — BUY (started 2026-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 13:15:00 | 292.50 | 289.39 | 289.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 15:15:00 | 293.30 | 290.60 | 289.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 303.35 | 303.92 | 300.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 10:00:00 | 303.35 | 303.92 | 300.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 310.45 | 311.53 | 309.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 11:15:00 | 313.65 | 311.70 | 309.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:15:00 | 313.60 | 311.88 | 310.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 13:45:00 | 313.55 | 312.76 | 311.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 14:30:00 | 313.70 | 313.45 | 311.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 315.85 | 320.49 | 319.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-02 12:15:00 | 315.15 | 318.19 | 318.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 214 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 315.15 | 318.19 | 318.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 13:15:00 | 314.10 | 317.38 | 317.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 15:15:00 | 301.60 | 301.43 | 305.02 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:15:00 | 296.60 | 301.43 | 305.02 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 281.77 | 294.46 | 299.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 292.00 | 289.57 | 293.65 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 292.00 | 289.57 | 293.65 | SL hit (close>ema200) qty=0.50 sl=289.57 alert=retest1 |

### Cycle 215 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 284.90 | 282.34 | 282.12 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 276.25 | 281.79 | 282.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 274.60 | 278.58 | 280.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 280.05 | 276.81 | 278.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 280.05 | 276.81 | 278.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 280.05 | 276.81 | 278.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 270.50 | 279.54 | 279.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 10:45:00 | 275.00 | 271.76 | 272.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 12:15:00 | 274.60 | 272.47 | 272.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 13:15:00 | 274.35 | 273.01 | 272.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 217 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 274.35 | 273.01 | 272.89 | EMA200 above EMA400 |

### Cycle 218 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 262.10 | 270.79 | 271.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 261.35 | 268.90 | 270.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 252.25 | 252.23 | 258.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 09:30:00 | 253.06 | 252.23 | 258.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 257.38 | 250.17 | 251.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:30:00 | 258.30 | 250.17 | 251.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 219 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 257.60 | 253.24 | 252.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 260.00 | 254.59 | 253.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 11:15:00 | 256.01 | 256.04 | 254.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:45:00 | 256.20 | 256.04 | 254.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 269.91 | 274.53 | 272.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:30:00 | 271.32 | 273.99 | 272.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 279.55 | 281.79 | 282.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 220 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 279.55 | 281.79 | 282.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 12:15:00 | 276.67 | 280.69 | 281.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 274.03 | 273.90 | 276.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 14:30:00 | 274.99 | 273.90 | 276.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 273.95 | 273.97 | 276.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:00:00 | 273.68 | 273.91 | 276.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:00:00 | 273.34 | 273.87 | 275.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 15:15:00 | 273.50 | 273.95 | 275.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 270.30 | 266.13 | 265.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 221 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 270.30 | 266.13 | 265.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 271.10 | 267.12 | 266.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 10:15:00 | 268.75 | 269.16 | 268.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 10:45:00 | 268.85 | 269.16 | 268.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 268.45 | 269.02 | 268.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:30:00 | 268.40 | 269.02 | 268.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 267.55 | 268.72 | 268.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 267.55 | 268.72 | 268.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — SELL (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 13:15:00 | 261.85 | 267.35 | 267.59 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-18 09:15:00 | 186.80 | 2023-05-18 13:15:00 | 182.20 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2023-05-22 12:30:00 | 179.45 | 2023-05-23 09:15:00 | 183.55 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2023-06-02 09:15:00 | 185.90 | 2023-06-06 10:15:00 | 184.50 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2023-06-06 10:00:00 | 185.15 | 2023-06-06 10:15:00 | 184.50 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2023-06-15 10:15:00 | 187.70 | 2023-06-15 10:15:00 | 186.35 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2023-06-28 12:30:00 | 190.00 | 2023-07-03 09:15:00 | 191.90 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2023-06-30 09:15:00 | 187.85 | 2023-07-03 10:15:00 | 192.65 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2023-06-30 10:00:00 | 190.10 | 2023-07-03 10:15:00 | 192.65 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2023-06-30 11:45:00 | 190.35 | 2023-07-03 10:15:00 | 192.65 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2023-06-30 13:15:00 | 189.50 | 2023-07-03 10:15:00 | 192.65 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2023-07-11 09:15:00 | 204.60 | 2023-07-12 14:15:00 | 204.50 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2023-07-11 10:00:00 | 204.65 | 2023-07-12 14:15:00 | 204.50 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2023-07-18 12:00:00 | 197.95 | 2023-07-19 15:15:00 | 200.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2023-07-18 12:45:00 | 197.75 | 2023-07-19 15:15:00 | 200.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2023-07-18 13:45:00 | 198.35 | 2023-07-19 15:15:00 | 200.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2023-07-21 09:15:00 | 201.45 | 2023-07-21 12:15:00 | 198.20 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2023-07-26 15:15:00 | 197.30 | 2023-07-27 09:15:00 | 199.95 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2023-07-28 15:00:00 | 201.20 | 2023-08-02 09:15:00 | 198.75 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2023-08-01 12:45:00 | 201.30 | 2023-08-02 09:15:00 | 198.75 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2023-08-04 13:15:00 | 193.30 | 2023-08-08 15:15:00 | 194.95 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2023-08-08 11:00:00 | 192.60 | 2023-08-08 15:15:00 | 194.95 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2023-08-08 14:30:00 | 194.60 | 2023-08-08 15:15:00 | 194.95 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2023-09-07 09:15:00 | 195.30 | 2023-09-18 09:15:00 | 214.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-07 09:45:00 | 195.35 | 2023-09-18 09:15:00 | 214.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-07 10:30:00 | 195.40 | 2023-09-18 09:15:00 | 214.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-07 11:45:00 | 195.30 | 2023-09-18 09:15:00 | 214.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-13 13:30:00 | 206.90 | 2023-09-21 13:15:00 | 207.70 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2023-10-20 11:45:00 | 202.60 | 2023-10-26 09:15:00 | 192.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 12:15:00 | 202.50 | 2023-10-26 09:15:00 | 192.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 14:15:00 | 202.50 | 2023-10-26 09:15:00 | 192.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 11:45:00 | 202.60 | 2023-10-27 09:15:00 | 193.65 | STOP_HIT | 0.50 | 4.42% |
| SELL | retest2 | 2023-10-20 12:15:00 | 202.50 | 2023-10-27 09:15:00 | 193.65 | STOP_HIT | 0.50 | 4.37% |
| SELL | retest2 | 2023-10-20 14:15:00 | 202.50 | 2023-10-27 09:15:00 | 193.65 | STOP_HIT | 0.50 | 4.37% |
| BUY | retest2 | 2023-10-31 14:00:00 | 197.30 | 2023-11-01 12:15:00 | 195.60 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2023-11-01 10:00:00 | 197.35 | 2023-11-01 12:15:00 | 195.60 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2023-11-09 15:00:00 | 192.80 | 2023-11-10 14:15:00 | 194.55 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2023-11-10 09:30:00 | 192.80 | 2023-11-10 14:15:00 | 194.55 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2023-11-10 10:15:00 | 192.90 | 2023-11-10 14:15:00 | 194.55 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2023-11-23 12:00:00 | 194.95 | 2023-11-28 12:15:00 | 197.40 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2023-11-23 13:00:00 | 195.10 | 2023-11-28 12:15:00 | 197.40 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2023-11-23 13:30:00 | 194.90 | 2023-11-28 12:15:00 | 197.40 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2023-12-01 09:15:00 | 199.55 | 2023-12-13 13:15:00 | 219.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-01 09:15:00 | 233.00 | 2024-01-02 14:15:00 | 230.50 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-01-02 11:30:00 | 232.80 | 2024-01-02 14:15:00 | 230.50 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-01-02 14:00:00 | 232.10 | 2024-01-02 14:15:00 | 230.50 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-01-20 14:00:00 | 232.50 | 2024-01-23 11:15:00 | 225.00 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2024-01-25 12:30:00 | 224.40 | 2024-01-29 09:15:00 | 230.70 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2024-01-25 14:00:00 | 224.40 | 2024-01-29 09:15:00 | 230.70 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2024-01-31 11:15:00 | 240.10 | 2024-02-06 12:15:00 | 248.55 | STOP_HIT | 1.00 | 3.52% |
| BUY | retest2 | 2024-01-31 12:45:00 | 239.70 | 2024-02-06 12:15:00 | 248.55 | STOP_HIT | 1.00 | 3.69% |
| BUY | retest2 | 2024-02-13 13:15:00 | 256.85 | 2024-02-22 12:15:00 | 272.60 | STOP_HIT | 1.00 | 6.13% |
| BUY | retest2 | 2024-02-14 10:45:00 | 257.50 | 2024-02-22 12:15:00 | 272.60 | STOP_HIT | 1.00 | 5.86% |
| BUY | retest2 | 2024-03-06 13:00:00 | 283.00 | 2024-03-12 09:15:00 | 278.20 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-03-06 13:45:00 | 282.20 | 2024-03-12 09:15:00 | 278.20 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-03-07 13:45:00 | 281.85 | 2024-03-12 09:15:00 | 278.20 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-03-07 14:45:00 | 282.45 | 2024-03-12 09:15:00 | 278.20 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-03-28 09:15:00 | 260.65 | 2024-04-05 12:15:00 | 266.85 | STOP_HIT | 1.00 | 2.38% |
| SELL | retest2 | 2024-04-19 09:15:00 | 254.35 | 2024-04-22 14:15:00 | 261.50 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2024-04-29 11:15:00 | 269.00 | 2024-05-06 09:15:00 | 263.20 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-05-10 09:45:00 | 257.00 | 2024-05-15 09:15:00 | 263.70 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2024-05-10 15:00:00 | 253.00 | 2024-05-15 09:15:00 | 263.70 | STOP_HIT | 1.00 | -4.23% |
| SELL | retest2 | 2024-05-14 10:00:00 | 258.00 | 2024-05-15 09:15:00 | 263.70 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-05-23 09:15:00 | 268.10 | 2024-05-28 12:15:00 | 266.60 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-05-31 10:00:00 | 261.75 | 2024-06-03 09:15:00 | 287.50 | STOP_HIT | 1.00 | -9.84% |
| SELL | retest2 | 2024-05-31 12:00:00 | 262.10 | 2024-06-03 09:15:00 | 287.50 | STOP_HIT | 1.00 | -9.69% |
| BUY | retest2 | 2024-06-12 10:15:00 | 278.65 | 2024-06-21 10:15:00 | 284.20 | STOP_HIT | 1.00 | 1.99% |
| SELL | retest2 | 2024-06-25 11:30:00 | 280.55 | 2024-07-02 12:15:00 | 266.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-25 12:45:00 | 280.20 | 2024-07-02 12:15:00 | 266.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-25 15:15:00 | 279.55 | 2024-07-02 12:15:00 | 265.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-26 09:45:00 | 280.80 | 2024-07-02 12:15:00 | 266.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-27 10:30:00 | 280.20 | 2024-07-02 12:15:00 | 266.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-27 11:45:00 | 280.20 | 2024-07-02 12:15:00 | 266.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-25 11:30:00 | 280.55 | 2024-07-03 11:15:00 | 267.35 | STOP_HIT | 0.50 | 4.71% |
| SELL | retest2 | 2024-06-25 12:45:00 | 280.20 | 2024-07-03 11:15:00 | 267.35 | STOP_HIT | 0.50 | 4.59% |
| SELL | retest2 | 2024-06-25 15:15:00 | 279.55 | 2024-07-03 11:15:00 | 267.35 | STOP_HIT | 0.50 | 4.36% |
| SELL | retest2 | 2024-06-26 09:45:00 | 280.80 | 2024-07-03 11:15:00 | 267.35 | STOP_HIT | 0.50 | 4.79% |
| SELL | retest2 | 2024-06-27 10:30:00 | 280.20 | 2024-07-03 11:15:00 | 267.35 | STOP_HIT | 0.50 | 4.59% |
| SELL | retest2 | 2024-06-27 11:45:00 | 280.20 | 2024-07-03 11:15:00 | 267.35 | STOP_HIT | 0.50 | 4.59% |
| SELL | retest2 | 2024-07-22 14:15:00 | 253.90 | 2024-07-29 09:15:00 | 257.55 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-07-22 15:00:00 | 253.20 | 2024-07-29 09:15:00 | 257.55 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-07-23 09:30:00 | 253.35 | 2024-07-29 09:15:00 | 257.55 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-08-06 13:30:00 | 241.35 | 2024-08-08 11:15:00 | 243.80 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-08-06 14:00:00 | 240.55 | 2024-08-08 11:15:00 | 243.80 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-08-07 09:45:00 | 240.60 | 2024-08-08 11:15:00 | 243.80 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-08-08 09:30:00 | 241.35 | 2024-08-08 11:15:00 | 243.80 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-08-16 10:15:00 | 240.85 | 2024-08-16 15:15:00 | 243.50 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-08-16 11:00:00 | 240.65 | 2024-08-16 15:15:00 | 243.50 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-08-23 11:15:00 | 254.50 | 2024-08-23 14:15:00 | 252.10 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-09-17 11:30:00 | 239.70 | 2024-09-18 14:15:00 | 238.05 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-09-17 12:00:00 | 239.60 | 2024-09-18 14:15:00 | 238.05 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-09-18 09:15:00 | 240.00 | 2024-09-18 14:15:00 | 238.05 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-09-18 10:30:00 | 239.90 | 2024-09-18 14:15:00 | 238.05 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-09-20 13:30:00 | 236.25 | 2024-09-23 09:15:00 | 240.20 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-09-25 13:15:00 | 242.45 | 2024-10-03 12:15:00 | 246.00 | STOP_HIT | 1.00 | 1.46% |
| BUY | retest1 | 2024-10-07 09:15:00 | 251.98 | 2024-10-07 10:15:00 | 245.14 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2024-10-15 10:45:00 | 245.00 | 2024-10-18 12:15:00 | 244.81 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2024-10-15 12:15:00 | 244.84 | 2024-10-18 12:15:00 | 244.81 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2024-11-12 10:30:00 | 257.00 | 2024-11-13 14:15:00 | 244.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 10:30:00 | 257.00 | 2024-11-18 12:15:00 | 242.15 | STOP_HIT | 0.50 | 5.78% |
| BUY | retest2 | 2024-11-28 15:15:00 | 251.45 | 2024-11-29 11:15:00 | 245.10 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2024-12-10 14:15:00 | 262.94 | 2024-12-11 14:15:00 | 261.07 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-12-10 14:45:00 | 263.45 | 2024-12-11 14:15:00 | 261.07 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-12-10 15:15:00 | 263.65 | 2024-12-11 14:15:00 | 261.07 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-12-11 10:15:00 | 263.08 | 2024-12-11 14:15:00 | 261.07 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-12-17 10:15:00 | 257.23 | 2024-12-17 10:15:00 | 258.63 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-12-17 11:30:00 | 256.55 | 2024-12-20 13:15:00 | 243.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 11:30:00 | 256.55 | 2024-12-23 10:15:00 | 245.30 | STOP_HIT | 0.50 | 4.39% |
| SELL | retest2 | 2024-12-30 13:15:00 | 242.33 | 2025-01-03 10:15:00 | 243.91 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-01-03 10:15:00 | 242.97 | 2025-01-03 10:15:00 | 243.91 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-01-09 09:15:00 | 231.75 | 2025-01-13 13:15:00 | 220.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 10:15:00 | 232.28 | 2025-01-13 13:15:00 | 220.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 10:45:00 | 231.30 | 2025-01-13 13:15:00 | 219.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 09:15:00 | 231.75 | 2025-01-14 09:15:00 | 222.48 | STOP_HIT | 0.50 | 4.00% |
| SELL | retest2 | 2025-01-09 10:15:00 | 232.28 | 2025-01-14 09:15:00 | 222.48 | STOP_HIT | 0.50 | 4.22% |
| SELL | retest2 | 2025-01-09 10:45:00 | 231.30 | 2025-01-14 09:15:00 | 222.48 | STOP_HIT | 0.50 | 3.81% |
| SELL | retest2 | 2025-01-23 09:15:00 | 225.87 | 2025-01-23 12:15:00 | 229.71 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-01-23 10:45:00 | 227.59 | 2025-01-23 12:15:00 | 229.71 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-01-24 09:30:00 | 227.39 | 2025-01-31 09:15:00 | 216.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 12:30:00 | 227.90 | 2025-01-31 09:15:00 | 216.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:30:00 | 227.39 | 2025-02-01 10:15:00 | 216.22 | STOP_HIT | 0.50 | 4.91% |
| SELL | retest2 | 2025-01-24 12:30:00 | 227.90 | 2025-02-01 10:15:00 | 216.22 | STOP_HIT | 0.50 | 5.13% |
| SELL | retest2 | 2025-01-28 10:30:00 | 220.84 | 2025-02-01 13:15:00 | 209.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-28 10:30:00 | 220.84 | 2025-02-04 09:15:00 | 212.98 | STOP_HIT | 0.50 | 3.56% |
| SELL | retest2 | 2025-01-31 09:15:00 | 214.80 | 2025-02-05 09:15:00 | 218.21 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-02-13 10:15:00 | 212.30 | 2025-02-17 09:15:00 | 201.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 10:45:00 | 212.26 | 2025-02-17 09:15:00 | 201.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 11:15:00 | 212.35 | 2025-02-17 09:15:00 | 201.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 10:15:00 | 212.30 | 2025-02-17 13:15:00 | 205.55 | STOP_HIT | 0.50 | 3.18% |
| SELL | retest2 | 2025-02-13 10:45:00 | 212.26 | 2025-02-17 13:15:00 | 205.55 | STOP_HIT | 0.50 | 3.16% |
| SELL | retest2 | 2025-02-13 11:15:00 | 212.35 | 2025-02-17 13:15:00 | 205.55 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2025-03-12 11:45:00 | 202.40 | 2025-03-13 09:15:00 | 206.76 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-03-12 12:15:00 | 202.94 | 2025-03-13 09:15:00 | 206.76 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-03-12 12:45:00 | 202.14 | 2025-03-13 09:15:00 | 206.76 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-04-02 15:00:00 | 231.78 | 2025-04-07 09:15:00 | 224.77 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2025-04-03 10:00:00 | 231.64 | 2025-04-07 09:15:00 | 224.77 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2025-04-23 13:15:00 | 249.90 | 2025-04-25 09:15:00 | 247.83 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-05-06 11:30:00 | 244.01 | 2025-05-06 14:15:00 | 231.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 11:30:00 | 244.01 | 2025-05-08 13:15:00 | 219.61 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-05-21 09:30:00 | 237.75 | 2025-05-27 09:15:00 | 238.77 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2025-05-21 12:15:00 | 237.55 | 2025-05-27 09:15:00 | 238.77 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2025-05-29 15:15:00 | 243.64 | 2025-06-05 13:15:00 | 249.45 | STOP_HIT | 1.00 | 2.38% |
| BUY | retest2 | 2025-05-30 11:00:00 | 243.66 | 2025-06-05 13:15:00 | 249.45 | STOP_HIT | 1.00 | 2.38% |
| SELL | retest2 | 2025-06-09 15:15:00 | 247.80 | 2025-06-13 09:15:00 | 235.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 15:15:00 | 247.80 | 2025-06-13 15:15:00 | 239.80 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2025-07-07 11:45:00 | 243.40 | 2025-07-14 11:15:00 | 240.36 | STOP_HIT | 1.00 | 1.25% |
| SELL | retest2 | 2025-07-21 14:45:00 | 243.22 | 2025-07-24 14:15:00 | 246.90 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-07-22 09:15:00 | 242.75 | 2025-07-24 14:15:00 | 246.90 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-07-28 09:15:00 | 245.59 | 2025-07-28 10:15:00 | 242.95 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-07-28 10:00:00 | 245.30 | 2025-07-28 10:15:00 | 242.95 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-07-30 13:45:00 | 240.56 | 2025-08-04 14:15:00 | 241.24 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-07-30 14:15:00 | 240.81 | 2025-08-04 14:15:00 | 241.24 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-08-13 09:15:00 | 243.97 | 2025-08-13 11:15:00 | 241.91 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest1 | 2025-08-25 11:45:00 | 241.23 | 2025-09-01 13:15:00 | 234.59 | STOP_HIT | 1.00 | 2.75% |
| BUY | retest2 | 2025-09-15 14:15:00 | 239.69 | 2025-09-26 09:15:00 | 251.09 | STOP_HIT | 1.00 | 4.76% |
| BUY | retest2 | 2025-09-16 10:15:00 | 239.32 | 2025-09-26 09:15:00 | 251.09 | STOP_HIT | 1.00 | 4.92% |
| BUY | retest2 | 2025-09-16 10:45:00 | 239.30 | 2025-09-26 09:15:00 | 251.09 | STOP_HIT | 1.00 | 4.93% |
| BUY | retest2 | 2025-09-16 11:30:00 | 240.34 | 2025-09-26 09:15:00 | 251.09 | STOP_HIT | 1.00 | 4.47% |
| BUY | retest2 | 2025-10-01 13:30:00 | 260.35 | 2025-10-08 09:15:00 | 260.40 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2025-10-09 09:30:00 | 261.30 | 2025-10-09 11:15:00 | 263.60 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-10-27 11:15:00 | 269.00 | 2025-10-27 11:15:00 | 269.80 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-11-07 12:30:00 | 288.45 | 2025-11-11 09:15:00 | 282.00 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-11-10 12:45:00 | 288.15 | 2025-11-11 09:15:00 | 282.00 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-11-19 10:00:00 | 290.40 | 2025-11-21 09:15:00 | 286.20 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-11-19 10:45:00 | 290.00 | 2025-11-21 09:15:00 | 286.20 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-11-19 11:15:00 | 290.10 | 2025-11-21 09:15:00 | 286.20 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-11-20 13:45:00 | 289.95 | 2025-11-21 09:15:00 | 286.20 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-12-15 14:15:00 | 284.90 | 2025-12-17 11:15:00 | 286.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-12-15 14:45:00 | 285.00 | 2025-12-17 11:15:00 | 286.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-12-15 15:15:00 | 284.55 | 2025-12-17 11:15:00 | 286.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-12-17 09:45:00 | 285.00 | 2025-12-17 11:15:00 | 286.00 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-12-19 09:15:00 | 291.45 | 2025-12-24 11:15:00 | 291.85 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-12-29 13:15:00 | 288.15 | 2025-12-30 13:15:00 | 291.85 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-12-30 10:45:00 | 287.65 | 2025-12-30 13:15:00 | 291.85 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-12-30 11:30:00 | 288.30 | 2025-12-30 13:15:00 | 291.85 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2026-01-07 09:15:00 | 305.90 | 2026-01-08 12:15:00 | 302.20 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-01-12 11:00:00 | 296.70 | 2026-01-14 12:15:00 | 306.60 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2026-01-21 14:15:00 | 297.40 | 2026-01-22 09:15:00 | 306.80 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2026-02-24 11:15:00 | 313.65 | 2026-03-02 12:15:00 | 315.15 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2026-02-25 09:15:00 | 313.60 | 2026-03-02 12:15:00 | 315.15 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest2 | 2026-02-25 13:45:00 | 313.55 | 2026-03-02 12:15:00 | 315.15 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2026-02-25 14:30:00 | 313.70 | 2026-03-02 12:15:00 | 315.15 | STOP_HIT | 1.00 | 0.46% |
| SELL | retest1 | 2026-03-06 09:15:00 | 296.60 | 2026-03-09 09:15:00 | 281.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-06 09:15:00 | 296.60 | 2026-03-10 09:15:00 | 292.00 | STOP_HIT | 0.50 | 1.55% |
| SELL | retest2 | 2026-03-11 14:30:00 | 290.40 | 2026-03-16 10:15:00 | 275.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 14:15:00 | 290.40 | 2026-03-16 10:15:00 | 275.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 14:30:00 | 290.40 | 2026-03-16 14:15:00 | 280.00 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2026-03-12 14:15:00 | 290.40 | 2026-03-16 14:15:00 | 280.00 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2026-03-23 09:15:00 | 270.50 | 2026-03-25 13:15:00 | 274.35 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-03-25 10:45:00 | 275.00 | 2026-03-25 13:15:00 | 274.35 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2026-03-25 12:15:00 | 274.60 | 2026-03-25 13:15:00 | 274.35 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2026-04-13 10:30:00 | 271.32 | 2026-04-23 10:15:00 | 279.55 | STOP_HIT | 1.00 | 3.03% |
| SELL | retest2 | 2026-04-27 11:00:00 | 273.68 | 2026-05-06 14:15:00 | 270.30 | STOP_HIT | 1.00 | 1.24% |
| SELL | retest2 | 2026-04-27 14:00:00 | 273.34 | 2026-05-06 14:15:00 | 270.30 | STOP_HIT | 1.00 | 1.11% |
| SELL | retest2 | 2026-04-27 15:15:00 | 273.50 | 2026-05-06 14:15:00 | 270.30 | STOP_HIT | 1.00 | 1.17% |
