# Bandhan Bank Ltd. (BANDHANBNK)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 206.25
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 147 |
| ALERT1 | 110 |
| ALERT2 | 107 |
| ALERT2_SKIP | 51 |
| ALERT3 | 275 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 96 |
| PARTIAL | 13 |
| TARGET_HIT | 1 |
| STOP_HIT | 99 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 113 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 72
- **Target hits / Stop hits / Partials:** 1 / 99 / 13
- **Avg / median % per leg:** 0.29% / -0.96%
- **Sum % (uncompounded):** 32.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 47 | 14 | 29.8% | 1 | 46 | 0 | -0.34% | -16.2% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.50% | -6.0% |
| BUY @ 3rd Alert (retest2) | 43 | 14 | 32.6% | 1 | 42 | 0 | -0.24% | -10.2% |
| SELL (all) | 66 | 27 | 40.9% | 0 | 53 | 13 | 0.74% | 48.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 66 | 27 | 40.9% | 0 | 53 | 13 | 0.74% | 48.7% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.50% | -6.0% |
| retest2 (combined) | 109 | 41 | 37.6% | 1 | 95 | 13 | 0.35% | 38.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 09:15:00 | 181.70 | 187.77 | 187.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 10:15:00 | 180.55 | 186.32 | 187.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 11:15:00 | 182.50 | 181.62 | 183.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-17 11:45:00 | 182.60 | 181.62 | 183.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 182.65 | 181.78 | 182.91 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 13:15:00 | 184.60 | 183.28 | 183.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 11:15:00 | 185.80 | 184.44 | 183.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 14:15:00 | 186.65 | 186.65 | 185.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-24 15:00:00 | 186.65 | 186.65 | 185.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 185.60 | 186.44 | 185.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 188.05 | 186.44 | 185.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 12:15:00 | 188.30 | 189.78 | 189.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 12:15:00 | 188.30 | 189.78 | 189.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 14:15:00 | 186.30 | 188.76 | 189.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 187.60 | 187.35 | 188.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 14:00:00 | 187.60 | 187.35 | 188.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 188.55 | 187.59 | 188.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 188.55 | 187.59 | 188.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 188.20 | 187.72 | 188.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 193.45 | 187.72 | 188.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 191.90 | 188.55 | 188.57 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 191.90 | 189.22 | 188.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 193.05 | 190.39 | 189.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 186.40 | 191.41 | 190.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 186.40 | 191.41 | 190.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 186.40 | 191.41 | 190.47 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 178.30 | 188.78 | 189.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 176.00 | 186.23 | 188.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 182.80 | 181.55 | 184.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 182.80 | 181.55 | 184.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 185.60 | 182.36 | 184.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:30:00 | 185.65 | 182.36 | 184.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 187.95 | 183.48 | 184.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:00:00 | 187.95 | 183.48 | 184.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 189.65 | 184.71 | 185.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:00:00 | 189.65 | 184.71 | 185.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 190.60 | 185.89 | 185.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 192.30 | 187.86 | 186.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 197.50 | 197.98 | 195.58 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:15:00 | 198.76 | 197.98 | 195.58 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:00:00 | 198.91 | 198.16 | 195.88 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 12:45:00 | 198.81 | 198.29 | 196.51 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 13:15:00 | 199.69 | 198.29 | 196.51 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 200.48 | 198.71 | 197.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-12 12:15:00 | 196.05 | 198.08 | 197.36 | SL hit (close<ema400) qty=1.00 sl=197.36 alert=retest1 |

### Cycle 7 — SELL (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 11:15:00 | 195.11 | 196.77 | 196.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 15:15:00 | 193.86 | 195.35 | 196.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 09:15:00 | 196.09 | 194.92 | 195.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 09:15:00 | 196.09 | 194.92 | 195.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 196.09 | 194.92 | 195.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:00:00 | 196.09 | 194.92 | 195.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 196.81 | 195.30 | 195.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:45:00 | 196.75 | 195.30 | 195.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 197.34 | 195.71 | 195.71 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2024-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 12:15:00 | 197.59 | 196.08 | 195.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 13:15:00 | 198.02 | 196.47 | 196.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 14:15:00 | 198.18 | 198.81 | 197.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 15:00:00 | 198.18 | 198.81 | 197.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 207.20 | 208.13 | 205.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 205.20 | 208.13 | 205.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 204.76 | 207.22 | 206.47 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 11:15:00 | 202.63 | 205.77 | 205.91 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 10:15:00 | 208.13 | 205.66 | 205.63 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 15:15:00 | 204.82 | 205.71 | 205.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 09:15:00 | 204.33 | 205.43 | 205.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 204.88 | 202.70 | 203.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 09:15:00 | 204.88 | 202.70 | 203.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 204.88 | 202.70 | 203.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:00:00 | 204.88 | 202.70 | 203.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 205.52 | 203.26 | 203.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 12:15:00 | 204.19 | 203.56 | 203.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 14:30:00 | 203.80 | 203.77 | 204.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 10:15:00 | 205.38 | 204.24 | 204.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 205.38 | 204.24 | 204.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 11:15:00 | 206.09 | 204.61 | 204.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 204.77 | 205.23 | 204.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 204.77 | 205.23 | 204.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 204.77 | 205.23 | 204.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:00:00 | 204.77 | 205.23 | 204.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 205.60 | 205.31 | 204.88 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 201.24 | 204.34 | 204.51 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 12:15:00 | 208.56 | 205.10 | 204.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 13:15:00 | 209.59 | 206.00 | 205.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 12:15:00 | 207.33 | 207.98 | 206.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 13:00:00 | 207.33 | 207.98 | 206.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 207.97 | 207.97 | 206.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 13:45:00 | 207.23 | 207.97 | 206.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 204.61 | 207.26 | 206.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:30:00 | 203.83 | 207.26 | 206.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 11:15:00 | 205.15 | 206.43 | 206.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 13:15:00 | 204.23 | 205.78 | 206.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 14:15:00 | 204.55 | 203.96 | 204.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 14:15:00 | 204.55 | 203.96 | 204.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 204.55 | 203.96 | 204.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 14:45:00 | 204.63 | 203.96 | 204.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 203.10 | 203.80 | 204.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 14:30:00 | 202.10 | 203.42 | 204.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:15:00 | 191.99 | 199.72 | 202.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-11 10:15:00 | 195.70 | 195.08 | 197.97 | SL hit (close>ema200) qty=0.50 sl=195.08 alert=retest2 |

### Cycle 16 — BUY (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 10:15:00 | 197.12 | 195.66 | 195.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 13:15:00 | 200.01 | 196.46 | 195.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 197.23 | 197.71 | 196.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-18 11:00:00 | 197.23 | 197.71 | 196.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 197.17 | 197.61 | 196.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 12:30:00 | 196.99 | 197.61 | 196.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 196.93 | 197.47 | 196.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:00:00 | 196.93 | 197.47 | 196.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 196.00 | 197.18 | 196.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:45:00 | 195.90 | 197.18 | 196.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 196.40 | 197.02 | 196.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:15:00 | 194.17 | 197.02 | 196.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 192.10 | 196.04 | 196.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 15:15:00 | 191.50 | 193.32 | 194.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 195.91 | 193.84 | 194.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 195.91 | 193.84 | 194.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 195.91 | 193.84 | 194.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 195.91 | 193.84 | 194.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 195.86 | 194.24 | 194.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 196.60 | 194.24 | 194.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2024-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 13:15:00 | 197.36 | 195.51 | 195.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 11:15:00 | 199.08 | 196.95 | 196.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 12:15:00 | 194.17 | 196.39 | 195.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 194.17 | 196.39 | 195.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 194.17 | 196.39 | 195.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 195.55 | 196.39 | 195.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 194.77 | 196.07 | 195.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 15:00:00 | 195.38 | 195.93 | 195.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 192.41 | 195.26 | 195.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 09:15:00 | 192.41 | 195.26 | 195.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-24 10:15:00 | 189.56 | 194.12 | 195.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 11:15:00 | 189.05 | 187.49 | 189.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 11:15:00 | 189.05 | 187.49 | 189.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 11:15:00 | 189.05 | 187.49 | 189.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 12:00:00 | 189.05 | 187.49 | 189.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 12:15:00 | 188.79 | 187.75 | 189.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 12:30:00 | 189.54 | 187.75 | 189.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 13:15:00 | 189.00 | 188.00 | 189.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 13:45:00 | 189.07 | 188.00 | 189.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 192.24 | 188.85 | 189.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 15:00:00 | 192.24 | 188.85 | 189.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 15:15:00 | 192.60 | 189.60 | 189.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 09:15:00 | 210.02 | 189.60 | 189.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 211.84 | 194.05 | 191.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 14:15:00 | 219.74 | 209.28 | 201.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 09:15:00 | 217.58 | 217.91 | 211.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 09:30:00 | 215.59 | 217.91 | 211.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 213.93 | 216.60 | 214.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 213.93 | 216.60 | 214.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 213.76 | 216.03 | 214.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:30:00 | 213.70 | 216.03 | 214.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 209.99 | 213.91 | 213.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 205.69 | 211.38 | 212.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 203.21 | 202.65 | 205.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 203.21 | 202.65 | 205.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 203.21 | 202.65 | 205.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 12:00:00 | 202.20 | 202.76 | 204.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 202.10 | 203.12 | 204.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 10:30:00 | 202.50 | 202.77 | 203.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 12:15:00 | 202.68 | 202.79 | 203.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 201.10 | 201.10 | 202.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 200.50 | 201.10 | 202.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:00:00 | 200.57 | 200.99 | 202.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 12:00:00 | 200.04 | 200.80 | 202.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 09:15:00 | 192.09 | 197.07 | 198.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 09:15:00 | 191.99 | 197.07 | 198.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 09:15:00 | 192.38 | 197.07 | 198.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 09:15:00 | 192.55 | 197.07 | 198.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:15:00 | 190.47 | 193.48 | 195.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:15:00 | 190.54 | 193.48 | 195.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:15:00 | 190.04 | 193.48 | 195.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-16 10:15:00 | 192.85 | 191.24 | 193.10 | SL hit (close>ema200) qty=0.50 sl=191.24 alert=retest2 |

### Cycle 22 — BUY (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 10:15:00 | 195.28 | 193.30 | 193.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 11:15:00 | 195.62 | 193.77 | 193.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 14:15:00 | 205.02 | 205.04 | 201.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 15:00:00 | 205.02 | 205.04 | 201.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 203.22 | 204.22 | 202.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:45:00 | 202.98 | 204.22 | 202.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 201.95 | 203.76 | 202.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 201.95 | 203.76 | 202.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 201.51 | 203.31 | 202.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 201.49 | 203.31 | 202.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 10:15:00 | 200.27 | 202.28 | 202.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 09:15:00 | 198.55 | 200.61 | 201.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 198.10 | 195.34 | 196.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 198.10 | 195.34 | 196.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 198.10 | 195.34 | 196.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:00:00 | 198.10 | 195.34 | 196.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 197.37 | 195.75 | 196.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:45:00 | 197.97 | 195.75 | 196.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 14:15:00 | 200.54 | 197.71 | 197.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 11:15:00 | 202.00 | 200.00 | 198.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 201.48 | 202.33 | 201.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 14:15:00 | 201.48 | 202.33 | 201.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 201.48 | 202.33 | 201.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 201.48 | 202.33 | 201.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 201.50 | 202.17 | 201.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 203.02 | 202.17 | 201.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 202.30 | 202.19 | 201.27 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2024-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 15:15:00 | 199.60 | 201.07 | 201.08 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 09:15:00 | 203.76 | 201.60 | 201.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 10:15:00 | 205.69 | 202.42 | 201.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 15:15:00 | 203.70 | 203.89 | 202.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 09:15:00 | 202.27 | 203.89 | 202.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 198.02 | 202.71 | 202.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 198.02 | 202.71 | 202.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 197.65 | 201.70 | 201.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 196.39 | 199.32 | 200.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 197.34 | 196.33 | 197.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 197.34 | 196.33 | 197.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 197.34 | 196.33 | 197.83 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 199.96 | 198.68 | 198.55 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 11:15:00 | 196.89 | 198.23 | 198.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 14:15:00 | 195.80 | 197.79 | 198.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 14:15:00 | 197.41 | 196.76 | 197.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 14:15:00 | 197.41 | 196.76 | 197.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 197.41 | 196.76 | 197.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 14:45:00 | 197.04 | 196.76 | 197.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 196.95 | 196.80 | 197.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:15:00 | 200.06 | 196.80 | 197.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 202.70 | 197.98 | 197.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 10:15:00 | 204.69 | 199.32 | 198.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 14:15:00 | 205.37 | 205.75 | 203.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 14:45:00 | 205.57 | 205.75 | 203.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 208.93 | 210.89 | 209.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:45:00 | 210.00 | 210.89 | 209.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 207.20 | 210.16 | 209.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:45:00 | 206.10 | 210.16 | 209.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 211.78 | 210.30 | 209.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 14:45:00 | 210.36 | 210.30 | 209.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 210.09 | 210.94 | 210.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 210.09 | 210.94 | 210.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 15:15:00 | 209.87 | 210.72 | 210.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:15:00 | 209.95 | 210.72 | 210.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 208.78 | 210.33 | 210.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 10:00:00 | 208.78 | 210.33 | 210.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 208.95 | 210.06 | 210.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 10:30:00 | 208.49 | 210.06 | 210.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 13:15:00 | 214.50 | 210.89 | 210.40 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2024-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 11:15:00 | 208.72 | 210.21 | 210.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 205.37 | 208.33 | 209.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 205.50 | 204.36 | 205.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 14:15:00 | 205.50 | 204.36 | 205.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 205.50 | 204.36 | 205.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 205.50 | 204.36 | 205.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 205.80 | 204.65 | 205.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 206.12 | 204.65 | 205.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 206.14 | 204.95 | 205.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:45:00 | 205.81 | 204.95 | 205.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 205.82 | 205.12 | 205.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:45:00 | 205.99 | 205.12 | 205.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 205.41 | 205.18 | 205.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:30:00 | 205.92 | 205.18 | 205.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 204.73 | 205.09 | 205.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 13:00:00 | 204.73 | 205.09 | 205.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 188.39 | 188.50 | 191.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:15:00 | 187.28 | 188.50 | 191.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 11:15:00 | 189.85 | 188.15 | 188.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 189.85 | 188.15 | 188.06 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 15:15:00 | 187.90 | 188.39 | 188.45 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 09:15:00 | 204.90 | 191.69 | 189.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 10:15:00 | 208.40 | 195.03 | 191.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 09:15:00 | 202.39 | 203.58 | 198.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 10:00:00 | 202.39 | 203.58 | 198.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 15:15:00 | 200.19 | 201.31 | 199.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:15:00 | 198.90 | 201.31 | 199.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 196.24 | 200.29 | 198.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 196.24 | 200.29 | 198.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 195.74 | 199.38 | 198.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:00:00 | 195.74 | 199.38 | 198.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2024-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 12:15:00 | 194.18 | 197.54 | 197.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 09:15:00 | 194.00 | 195.59 | 196.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 10:15:00 | 191.21 | 190.86 | 192.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 10:30:00 | 191.00 | 190.86 | 192.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 192.16 | 191.30 | 192.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:00:00 | 192.16 | 191.30 | 192.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 192.01 | 191.44 | 192.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 15:00:00 | 191.95 | 191.55 | 192.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 182.35 | 186.31 | 188.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 10:15:00 | 178.58 | 178.40 | 182.43 | SL hit (close>ema200) qty=0.50 sl=178.40 alert=retest2 |

### Cycle 36 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 183.93 | 178.54 | 178.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 13:15:00 | 184.45 | 179.72 | 178.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 09:15:00 | 179.53 | 181.10 | 179.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 09:15:00 | 179.53 | 181.10 | 179.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 179.53 | 181.10 | 179.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:00:00 | 179.53 | 181.10 | 179.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 178.80 | 180.64 | 179.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 11:00:00 | 178.80 | 180.64 | 179.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 178.01 | 180.12 | 179.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 11:45:00 | 177.43 | 180.12 | 179.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 178.99 | 179.34 | 179.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:15:00 | 178.88 | 179.34 | 179.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 178.71 | 179.21 | 179.19 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 10:15:00 | 178.93 | 179.16 | 179.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 13:15:00 | 177.66 | 178.74 | 178.96 | Break + close below crossover candle low |

### Cycle 38 — BUY (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 09:15:00 | 181.74 | 179.01 | 179.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 183.80 | 181.46 | 180.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 178.66 | 181.25 | 180.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 178.66 | 181.25 | 180.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 178.66 | 181.25 | 180.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 178.66 | 181.25 | 180.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 177.87 | 180.57 | 180.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 177.87 | 180.57 | 180.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 179.33 | 179.96 | 180.03 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 09:15:00 | 180.29 | 180.06 | 180.06 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 10:15:00 | 179.75 | 180.00 | 180.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 11:15:00 | 179.00 | 179.80 | 179.94 | Break + close below crossover candle low |

### Cycle 42 — BUY (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 13:15:00 | 181.79 | 180.10 | 180.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 183.48 | 181.11 | 180.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 11:15:00 | 182.05 | 182.55 | 181.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 11:15:00 | 182.05 | 182.55 | 181.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 182.05 | 182.55 | 181.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 182.05 | 182.55 | 181.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 182.65 | 182.57 | 181.93 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 179.12 | 181.24 | 181.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 178.55 | 180.70 | 181.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 12:15:00 | 167.56 | 167.18 | 169.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 13:00:00 | 167.56 | 167.18 | 169.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 171.64 | 167.52 | 168.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 171.64 | 167.52 | 168.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 171.69 | 168.36 | 168.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 171.99 | 168.36 | 168.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 170.95 | 168.87 | 168.63 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 166.65 | 168.70 | 168.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 164.61 | 167.88 | 168.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 167.85 | 166.78 | 167.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 167.85 | 166.78 | 167.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 167.85 | 166.78 | 167.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:30:00 | 167.59 | 166.78 | 167.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 167.40 | 166.91 | 167.38 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2024-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 12:15:00 | 169.95 | 167.81 | 167.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 173.15 | 169.77 | 168.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 170.80 | 170.96 | 170.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 09:15:00 | 170.80 | 170.96 | 170.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 170.80 | 170.96 | 170.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:30:00 | 173.00 | 171.63 | 171.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 10:15:00 | 168.44 | 170.63 | 170.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 10:15:00 | 168.44 | 170.63 | 170.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 12:15:00 | 167.69 | 169.65 | 170.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 11:15:00 | 169.10 | 168.94 | 169.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 11:15:00 | 169.10 | 168.94 | 169.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 169.10 | 168.94 | 169.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:30:00 | 169.37 | 168.94 | 169.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 12:15:00 | 169.48 | 169.05 | 169.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 13:00:00 | 169.48 | 169.05 | 169.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 169.80 | 169.20 | 169.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 14:00:00 | 169.80 | 169.20 | 169.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 170.59 | 169.48 | 169.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 14:30:00 | 170.90 | 169.48 | 169.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 174.44 | 170.70 | 170.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 10:15:00 | 174.48 | 171.46 | 170.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 176.35 | 176.74 | 174.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 09:45:00 | 176.50 | 176.74 | 174.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 174.67 | 176.91 | 176.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 174.67 | 176.91 | 176.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 177.72 | 177.07 | 176.19 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 13:15:00 | 175.21 | 176.13 | 176.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 14:15:00 | 174.57 | 175.82 | 176.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 175.80 | 175.62 | 175.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 175.80 | 175.62 | 175.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 175.80 | 175.62 | 175.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:00:00 | 175.80 | 175.62 | 175.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 175.40 | 175.58 | 175.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:30:00 | 175.92 | 175.58 | 175.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 175.44 | 175.18 | 175.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 15:00:00 | 175.44 | 175.18 | 175.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 175.85 | 175.31 | 175.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:15:00 | 176.35 | 175.31 | 175.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 175.38 | 175.33 | 175.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 10:30:00 | 175.25 | 175.28 | 175.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:15:00 | 166.49 | 169.98 | 172.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-13 14:15:00 | 169.83 | 169.24 | 170.99 | SL hit (close>ema200) qty=0.50 sl=169.24 alert=retest2 |

### Cycle 50 — BUY (started 2024-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 09:15:00 | 166.90 | 163.84 | 163.51 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 10:15:00 | 161.10 | 164.27 | 164.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 10:15:00 | 160.44 | 161.20 | 162.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 11:15:00 | 161.20 | 161.20 | 161.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-30 11:30:00 | 161.22 | 161.20 | 161.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 159.59 | 159.14 | 159.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:30:00 | 160.30 | 159.14 | 159.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 160.12 | 159.34 | 159.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:45:00 | 160.24 | 159.34 | 159.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 159.39 | 159.35 | 159.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 09:45:00 | 159.17 | 159.48 | 159.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 10:45:00 | 158.42 | 159.34 | 159.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 13:15:00 | 160.80 | 159.67 | 159.73 | SL hit (close>static) qty=1.00 sl=160.30 alert=retest2 |

### Cycle 52 — BUY (started 2025-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 14:15:00 | 160.97 | 159.93 | 159.84 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 12:15:00 | 158.48 | 159.71 | 159.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 13:15:00 | 157.90 | 159.35 | 159.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 154.01 | 153.97 | 155.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 11:00:00 | 154.01 | 153.97 | 155.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 144.90 | 141.94 | 144.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 144.90 | 141.94 | 144.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 144.83 | 142.52 | 144.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:30:00 | 144.90 | 142.52 | 144.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 145.78 | 143.51 | 144.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:00:00 | 145.78 | 143.51 | 144.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 147.72 | 144.35 | 144.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:00:00 | 147.72 | 144.35 | 144.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2025-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 15:15:00 | 149.05 | 145.29 | 145.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 10:15:00 | 149.90 | 146.78 | 145.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 150.19 | 150.51 | 149.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 11:00:00 | 150.19 | 150.51 | 149.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 151.79 | 151.19 | 150.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 11:45:00 | 152.50 | 151.52 | 150.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 11:45:00 | 152.60 | 152.06 | 151.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 09:15:00 | 149.29 | 151.05 | 151.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 149.29 | 151.05 | 151.14 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 151.41 | 150.90 | 150.89 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 09:15:00 | 149.75 | 150.83 | 150.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 149.47 | 150.15 | 150.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 11:15:00 | 148.65 | 148.56 | 149.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-27 12:00:00 | 148.65 | 148.56 | 149.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 13:15:00 | 149.72 | 148.90 | 149.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 14:00:00 | 149.72 | 148.90 | 149.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 14:15:00 | 149.47 | 149.01 | 149.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 09:15:00 | 148.95 | 149.15 | 149.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 151.85 | 149.49 | 149.52 | SL hit (close>static) qty=1.00 sl=149.99 alert=retest2 |

### Cycle 58 — BUY (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 13:15:00 | 152.27 | 150.04 | 149.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 09:15:00 | 152.80 | 150.94 | 150.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-29 12:15:00 | 149.64 | 150.85 | 150.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 12:15:00 | 149.64 | 150.85 | 150.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 149.64 | 150.85 | 150.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-29 13:00:00 | 149.64 | 150.85 | 150.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 150.26 | 150.73 | 150.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-29 15:00:00 | 150.55 | 150.70 | 150.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 09:15:00 | 150.62 | 150.58 | 150.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 10:30:00 | 150.53 | 150.48 | 150.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 11:00:00 | 150.54 | 150.48 | 150.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 150.49 | 150.48 | 150.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:30:00 | 150.33 | 150.48 | 150.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 150.31 | 150.45 | 150.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:00:00 | 150.31 | 150.45 | 150.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-30 13:15:00 | 148.65 | 150.09 | 150.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 148.65 | 150.09 | 150.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 14:15:00 | 148.35 | 149.74 | 150.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 10:15:00 | 150.71 | 149.37 | 149.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 10:15:00 | 150.71 | 149.37 | 149.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 150.71 | 149.37 | 149.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 11:00:00 | 150.71 | 149.37 | 149.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 152.00 | 149.90 | 149.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 12:00:00 | 152.00 | 149.90 | 149.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2025-01-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 12:15:00 | 152.00 | 150.32 | 150.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 152.88 | 151.15 | 150.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 151.20 | 151.39 | 150.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 151.20 | 151.39 | 150.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 151.20 | 151.39 | 150.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 150.40 | 151.39 | 150.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 151.14 | 151.34 | 150.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 151.14 | 151.34 | 150.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 152.68 | 151.61 | 151.02 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 147.11 | 150.58 | 150.69 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 14:15:00 | 151.09 | 149.87 | 149.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 153.75 | 150.84 | 150.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 153.08 | 153.43 | 152.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 153.08 | 153.43 | 152.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 153.08 | 153.43 | 152.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 152.64 | 153.43 | 152.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 152.53 | 153.25 | 152.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:00:00 | 152.53 | 153.25 | 152.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 151.55 | 152.91 | 152.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:00:00 | 151.55 | 152.91 | 152.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 151.89 | 152.71 | 152.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 151.89 | 152.71 | 152.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 152.70 | 152.45 | 152.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 152.96 | 152.45 | 152.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:30:00 | 153.29 | 153.04 | 152.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 11:15:00 | 151.95 | 152.83 | 152.43 | SL hit (close<static) qty=1.00 sl=152.02 alert=retest2 |

### Cycle 63 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 149.95 | 152.05 | 152.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 148.98 | 151.16 | 151.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 09:15:00 | 132.74 | 130.88 | 132.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 09:15:00 | 132.74 | 130.88 | 132.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 132.74 | 130.88 | 132.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 132.74 | 130.88 | 132.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 132.28 | 131.16 | 132.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 133.00 | 131.16 | 132.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 133.00 | 131.53 | 132.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 11:45:00 | 133.10 | 131.53 | 132.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 12:15:00 | 132.37 | 131.70 | 132.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 14:15:00 | 131.81 | 131.88 | 132.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 14:15:00 | 133.51 | 132.21 | 132.91 | SL hit (close>static) qty=1.00 sl=133.09 alert=retest2 |

### Cycle 64 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 135.43 | 133.36 | 133.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 14:15:00 | 136.59 | 134.59 | 133.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 13:15:00 | 139.02 | 139.39 | 138.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 13:45:00 | 139.14 | 139.39 | 138.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 137.55 | 138.91 | 138.12 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2025-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 13:15:00 | 136.39 | 137.48 | 137.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 14:15:00 | 135.58 | 137.10 | 137.42 | Break + close below crossover candle low |

### Cycle 66 — BUY (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 09:15:00 | 141.35 | 137.60 | 137.57 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 14:15:00 | 137.08 | 137.59 | 137.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 136.68 | 137.33 | 137.51 | Break + close below crossover candle low |

### Cycle 68 — BUY (started 2025-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-28 10:15:00 | 139.14 | 137.69 | 137.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-28 14:15:00 | 140.86 | 138.61 | 138.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-03 11:15:00 | 139.22 | 139.31 | 138.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 11:15:00 | 139.22 | 139.31 | 138.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 11:15:00 | 139.22 | 139.31 | 138.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-03 12:00:00 | 139.22 | 139.31 | 138.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 141.20 | 139.69 | 138.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-04 09:30:00 | 142.45 | 140.34 | 139.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 139.72 | 147.49 | 148.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 139.72 | 147.49 | 148.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 12:15:00 | 138.21 | 139.75 | 140.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 140.02 | 139.29 | 140.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 140.02 | 139.29 | 140.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 140.02 | 139.29 | 140.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:30:00 | 140.17 | 139.29 | 140.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 140.45 | 139.74 | 140.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 14:00:00 | 140.45 | 139.74 | 140.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 140.46 | 139.88 | 140.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 15:00:00 | 140.46 | 139.88 | 140.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 15:15:00 | 140.90 | 140.08 | 140.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 09:15:00 | 141.36 | 140.08 | 140.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2025-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 10:15:00 | 141.13 | 140.46 | 140.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 10:15:00 | 141.88 | 141.16 | 140.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 141.41 | 141.57 | 141.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 09:15:00 | 142.03 | 141.57 | 141.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 142.81 | 141.81 | 141.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:15:00 | 144.24 | 141.81 | 141.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 13:15:00 | 145.64 | 147.58 | 147.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 13:15:00 | 145.64 | 147.58 | 147.81 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 10:15:00 | 149.00 | 147.85 | 147.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 10:15:00 | 150.95 | 149.23 | 148.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 152.56 | 154.04 | 152.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 152.56 | 154.04 | 152.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 152.56 | 154.04 | 152.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:45:00 | 152.70 | 154.04 | 152.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 154.66 | 154.16 | 152.50 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 142.09 | 151.12 | 151.72 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 150.45 | 149.25 | 149.19 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 09:15:00 | 146.41 | 148.68 | 148.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 10:15:00 | 143.59 | 147.67 | 148.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 09:15:00 | 149.03 | 147.30 | 147.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 09:15:00 | 149.03 | 147.30 | 147.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 149.03 | 147.30 | 147.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 10:30:00 | 148.20 | 147.77 | 147.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 11:15:00 | 149.90 | 148.20 | 148.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 149.90 | 148.20 | 148.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 152.18 | 149.63 | 148.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 15:15:00 | 168.30 | 168.34 | 165.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 09:15:00 | 168.90 | 168.34 | 165.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 167.42 | 169.92 | 168.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 167.42 | 169.92 | 168.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 167.13 | 169.36 | 168.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 167.13 | 169.36 | 168.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2025-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 13:15:00 | 166.88 | 168.30 | 168.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 14:15:00 | 165.84 | 167.81 | 168.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 168.30 | 165.31 | 166.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 168.30 | 165.31 | 166.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 168.30 | 165.31 | 166.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:30:00 | 169.68 | 165.31 | 166.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 169.92 | 166.23 | 166.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 11:00:00 | 169.92 | 166.23 | 166.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 11:15:00 | 168.38 | 166.66 | 166.63 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 10:15:00 | 164.64 | 166.63 | 166.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 10:15:00 | 163.18 | 165.07 | 165.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 165.17 | 163.37 | 164.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 165.17 | 163.37 | 164.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 165.17 | 163.37 | 164.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 165.17 | 163.37 | 164.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 165.51 | 163.80 | 164.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:00:00 | 165.51 | 163.80 | 164.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 164.71 | 164.32 | 164.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 13:30:00 | 164.94 | 164.32 | 164.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 164.44 | 164.34 | 164.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:45:00 | 164.73 | 164.34 | 164.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 164.40 | 164.36 | 164.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 163.20 | 164.36 | 164.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 161.55 | 163.79 | 164.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 10:15:00 | 161.11 | 163.79 | 164.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 09:45:00 | 161.15 | 159.66 | 160.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 10:30:00 | 161.25 | 160.01 | 160.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 11:15:00 | 161.10 | 160.01 | 160.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 158.55 | 159.90 | 160.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 14:45:00 | 157.60 | 159.42 | 160.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 163.35 | 158.33 | 158.69 | SL hit (close>static) qty=1.00 sl=160.84 alert=retest2 |

### Cycle 80 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 163.55 | 159.38 | 159.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 165.09 | 162.43 | 160.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 162.20 | 162.88 | 161.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 162.20 | 162.88 | 161.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 164.92 | 163.19 | 162.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 11:15:00 | 165.99 | 163.61 | 162.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 12:00:00 | 165.67 | 164.02 | 162.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 166.38 | 168.62 | 168.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 166.38 | 168.62 | 168.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 164.90 | 167.87 | 168.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 10:15:00 | 167.84 | 167.42 | 168.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 10:15:00 | 167.84 | 167.42 | 168.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 167.84 | 167.42 | 168.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 167.84 | 167.42 | 168.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 166.21 | 167.18 | 167.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 10:30:00 | 165.84 | 166.91 | 167.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 10:15:00 | 168.29 | 166.28 | 166.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 10:15:00 | 168.29 | 166.28 | 166.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 170.01 | 167.02 | 166.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 15:15:00 | 168.73 | 169.16 | 168.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 09:45:00 | 169.30 | 169.23 | 168.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 168.76 | 169.84 | 169.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 168.76 | 169.84 | 169.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 168.05 | 169.49 | 169.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:45:00 | 167.92 | 169.49 | 169.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 174.84 | 170.49 | 169.67 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 169.50 | 170.83 | 170.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 167.50 | 170.16 | 170.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 10:15:00 | 170.80 | 168.21 | 168.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 10:15:00 | 170.80 | 168.21 | 168.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 170.80 | 168.21 | 168.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:00:00 | 170.80 | 168.21 | 168.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 172.19 | 169.00 | 169.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:00:00 | 172.19 | 169.00 | 169.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 173.04 | 169.81 | 169.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 182.56 | 173.76 | 171.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 11:15:00 | 182.34 | 182.81 | 179.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 11:30:00 | 182.13 | 182.81 | 179.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 180.40 | 181.73 | 179.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 179.62 | 181.73 | 179.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 182.22 | 182.12 | 181.02 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 177.83 | 180.58 | 180.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 177.09 | 179.47 | 180.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 175.65 | 175.37 | 176.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:00:00 | 175.65 | 175.37 | 176.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 178.28 | 176.10 | 176.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 178.28 | 176.10 | 176.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 178.44 | 176.57 | 177.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:30:00 | 178.50 | 176.57 | 177.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 178.64 | 177.43 | 177.41 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 11:15:00 | 177.14 | 177.37 | 177.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 12:15:00 | 176.16 | 177.13 | 177.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 179.00 | 176.77 | 176.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 179.00 | 176.77 | 176.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 179.00 | 176.77 | 176.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:45:00 | 180.60 | 176.77 | 176.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 178.59 | 177.13 | 177.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 11:15:00 | 180.18 | 177.74 | 177.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 179.38 | 179.41 | 178.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 09:30:00 | 179.19 | 179.41 | 178.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 178.23 | 179.31 | 178.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:45:00 | 178.25 | 179.31 | 178.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 178.70 | 179.19 | 178.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 177.68 | 179.19 | 178.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 178.45 | 179.04 | 178.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 14:45:00 | 178.55 | 179.04 | 178.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 177.30 | 178.69 | 178.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 09:30:00 | 178.82 | 178.63 | 178.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:15:00 | 179.12 | 178.63 | 178.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 10:15:00 | 179.09 | 179.35 | 179.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:45:00 | 178.91 | 179.16 | 179.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 179.33 | 179.19 | 179.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:30:00 | 178.82 | 179.19 | 179.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 180.31 | 179.42 | 179.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 15:15:00 | 180.80 | 179.55 | 179.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 10:30:00 | 180.80 | 180.21 | 179.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:30:00 | 181.04 | 180.65 | 180.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 184.05 | 186.93 | 187.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 10:15:00 | 184.05 | 186.93 | 187.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 15:15:00 | 183.35 | 184.93 | 185.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 175.74 | 175.55 | 177.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 12:15:00 | 177.43 | 176.18 | 177.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 177.43 | 176.18 | 177.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:00:00 | 177.43 | 176.18 | 177.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 178.23 | 176.59 | 177.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:00:00 | 178.23 | 176.59 | 177.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 178.24 | 176.92 | 177.46 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 11:15:00 | 178.46 | 177.83 | 177.77 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 176.52 | 177.67 | 177.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 175.57 | 177.25 | 177.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 176.49 | 176.25 | 176.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 176.49 | 176.25 | 176.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 176.49 | 176.25 | 176.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 176.39 | 176.25 | 176.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 176.15 | 176.23 | 176.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 175.52 | 176.23 | 176.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 181.80 | 176.74 | 176.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 181.80 | 176.74 | 176.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 10:15:00 | 183.43 | 180.29 | 178.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 15:15:00 | 185.40 | 185.93 | 184.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 15:15:00 | 185.40 | 185.93 | 184.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 185.40 | 185.93 | 184.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 183.35 | 185.93 | 184.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 182.28 | 185.20 | 184.28 | EMA400 retest candle locked (from upside) |

### Cycle 93 — SELL (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 11:15:00 | 181.40 | 183.70 | 183.71 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 183.90 | 181.93 | 181.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 14:15:00 | 184.14 | 182.99 | 182.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 180.72 | 182.72 | 182.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 180.72 | 182.72 | 182.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 180.72 | 182.72 | 182.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 180.72 | 182.72 | 182.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 179.63 | 182.10 | 182.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 11:15:00 | 177.68 | 181.22 | 181.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 10:15:00 | 165.79 | 165.13 | 167.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 11:00:00 | 165.79 | 165.13 | 167.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 167.26 | 165.87 | 167.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:00:00 | 166.51 | 166.79 | 167.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 12:30:00 | 166.59 | 166.53 | 167.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 167.53 | 165.44 | 165.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 09:15:00 | 167.53 | 165.44 | 165.43 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 165.08 | 165.63 | 165.66 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 166.08 | 165.72 | 165.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 10:15:00 | 166.73 | 166.11 | 165.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 10:15:00 | 166.19 | 166.67 | 166.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 10:15:00 | 166.19 | 166.67 | 166.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 166.19 | 166.67 | 166.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:45:00 | 166.08 | 166.67 | 166.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 165.45 | 166.43 | 166.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 165.45 | 166.43 | 166.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 165.66 | 166.27 | 166.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:30:00 | 165.00 | 166.27 | 166.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 13:15:00 | 165.66 | 166.15 | 166.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 163.87 | 165.56 | 165.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 167.47 | 164.76 | 165.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 167.47 | 164.76 | 165.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 167.47 | 164.76 | 165.13 | EMA400 retest candle locked (from downside) |

### Cycle 100 — BUY (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 11:15:00 | 166.85 | 165.50 | 165.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 168.16 | 166.03 | 165.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 09:15:00 | 174.22 | 175.58 | 173.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 10:00:00 | 174.22 | 175.58 | 173.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 174.88 | 175.44 | 174.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 174.69 | 175.44 | 174.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 174.05 | 175.16 | 174.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:00:00 | 174.05 | 175.16 | 174.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 173.80 | 174.89 | 174.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:15:00 | 173.57 | 174.89 | 174.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 173.84 | 174.68 | 174.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:30:00 | 173.65 | 174.68 | 174.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 173.38 | 174.42 | 173.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 173.38 | 174.42 | 173.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 10:15:00 | 172.19 | 173.46 | 173.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 171.47 | 172.84 | 173.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 165.00 | 164.22 | 166.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:30:00 | 165.31 | 164.22 | 166.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 163.09 | 162.06 | 163.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 12:00:00 | 163.09 | 162.06 | 163.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 162.80 | 162.21 | 162.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 12:30:00 | 163.10 | 162.21 | 162.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 161.40 | 162.05 | 162.84 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 165.00 | 163.47 | 163.27 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 162.43 | 163.33 | 163.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 161.92 | 163.06 | 163.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 163.56 | 162.96 | 163.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 163.56 | 162.96 | 163.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 163.56 | 162.96 | 163.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 163.56 | 162.96 | 163.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 163.36 | 163.04 | 163.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:15:00 | 163.35 | 163.04 | 163.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 163.35 | 163.10 | 163.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 163.74 | 163.10 | 163.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 164.91 | 163.46 | 163.37 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 15:15:00 | 163.50 | 163.80 | 163.83 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 165.65 | 164.17 | 164.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 167.75 | 164.88 | 164.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 166.77 | 166.78 | 165.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 12:45:00 | 166.80 | 166.78 | 165.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 166.09 | 166.54 | 166.05 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 164.75 | 165.80 | 165.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 09:15:00 | 163.33 | 164.94 | 165.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 14:15:00 | 162.75 | 162.72 | 163.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 14:30:00 | 162.71 | 162.72 | 163.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 162.90 | 162.80 | 163.38 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 14:15:00 | 164.34 | 163.67 | 163.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 167.49 | 164.54 | 164.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 166.22 | 166.29 | 165.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 09:30:00 | 166.51 | 166.29 | 165.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 165.34 | 166.10 | 165.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:45:00 | 165.47 | 166.10 | 165.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 166.43 | 166.16 | 165.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 12:30:00 | 166.64 | 166.44 | 165.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 164.11 | 165.79 | 165.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 13:15:00 | 164.11 | 165.79 | 165.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 162.87 | 165.21 | 165.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 14:15:00 | 162.08 | 161.97 | 163.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:30:00 | 162.04 | 161.97 | 163.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 162.52 | 162.10 | 163.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:45:00 | 161.93 | 161.93 | 163.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 13:15:00 | 161.05 | 158.75 | 158.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2025-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 13:15:00 | 161.05 | 158.75 | 158.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 14:15:00 | 162.17 | 159.44 | 158.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 10:15:00 | 160.92 | 161.88 | 160.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 10:15:00 | 160.92 | 161.88 | 160.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 160.92 | 161.88 | 160.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 160.92 | 161.88 | 160.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 162.53 | 162.01 | 161.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 12:30:00 | 163.17 | 162.10 | 161.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 13:30:00 | 163.39 | 162.31 | 161.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:30:00 | 163.22 | 164.45 | 164.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 161.30 | 163.82 | 164.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 161.30 | 163.82 | 164.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 161.08 | 163.27 | 163.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 11:15:00 | 162.21 | 161.75 | 162.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 12:00:00 | 162.21 | 161.75 | 162.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 162.69 | 161.90 | 162.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 162.69 | 161.90 | 162.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 163.09 | 162.14 | 162.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 163.09 | 162.14 | 162.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 163.28 | 162.37 | 162.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 164.16 | 162.37 | 162.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 165.24 | 162.94 | 162.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 11:15:00 | 165.61 | 163.75 | 163.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 15:15:00 | 167.80 | 167.84 | 166.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 09:15:00 | 165.57 | 167.84 | 166.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 164.20 | 167.11 | 166.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 164.20 | 167.11 | 166.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 163.56 | 166.40 | 166.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 163.56 | 166.40 | 166.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 162.51 | 165.62 | 165.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 162.08 | 164.91 | 165.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 164.00 | 163.73 | 164.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 11:00:00 | 164.00 | 163.73 | 164.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 163.87 | 163.76 | 164.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 163.87 | 163.76 | 164.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 162.66 | 163.52 | 164.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 163.81 | 163.52 | 164.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 161.44 | 163.02 | 163.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:45:00 | 161.17 | 162.67 | 163.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 12:30:00 | 161.30 | 162.28 | 163.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 15:00:00 | 161.33 | 162.00 | 162.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:15:00 | 161.00 | 161.94 | 162.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 163.18 | 161.89 | 162.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 163.18 | 161.89 | 162.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 163.02 | 162.11 | 162.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:00:00 | 163.02 | 162.11 | 162.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 161.34 | 161.96 | 162.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:30:00 | 163.11 | 161.96 | 162.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 164.39 | 162.05 | 162.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:00:00 | 164.39 | 162.05 | 162.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 166.74 | 162.99 | 162.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 166.74 | 162.99 | 162.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 169.31 | 166.54 | 165.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 13:15:00 | 170.45 | 170.99 | 169.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 14:00:00 | 170.45 | 170.99 | 169.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 172.48 | 173.22 | 172.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:00:00 | 172.48 | 173.22 | 172.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 172.64 | 173.10 | 172.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:30:00 | 172.00 | 173.10 | 172.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 172.11 | 172.91 | 172.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 172.11 | 172.91 | 172.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 172.40 | 172.80 | 172.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 173.00 | 172.80 | 172.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 172.20 | 172.59 | 172.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 171.93 | 172.59 | 172.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 172.19 | 172.51 | 172.42 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 12:15:00 | 170.72 | 172.15 | 172.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 15:15:00 | 169.64 | 171.32 | 171.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 153.59 | 153.51 | 155.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 10:45:00 | 153.43 | 153.51 | 155.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 153.94 | 153.80 | 154.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 10:15:00 | 153.51 | 153.80 | 154.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 12:15:00 | 155.28 | 152.69 | 152.69 | SL hit (close>static) qty=1.00 sl=155.27 alert=retest2 |

### Cycle 116 — BUY (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 13:15:00 | 155.60 | 153.27 | 152.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 15:15:00 | 155.95 | 154.21 | 153.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 154.59 | 154.96 | 154.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 14:00:00 | 154.59 | 154.96 | 154.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 154.10 | 154.79 | 154.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 154.10 | 154.79 | 154.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 154.20 | 154.67 | 154.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 154.70 | 154.67 | 154.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 153.88 | 154.51 | 154.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 11:00:00 | 155.00 | 154.61 | 154.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:15:00 | 155.01 | 154.66 | 154.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:15:00 | 155.00 | 154.47 | 154.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 10:15:00 | 153.81 | 154.78 | 154.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 153.81 | 154.78 | 154.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 153.07 | 154.13 | 154.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 10:15:00 | 150.19 | 150.01 | 150.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 11:00:00 | 150.19 | 150.01 | 150.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 150.20 | 149.69 | 150.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 151.85 | 149.69 | 150.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 151.54 | 150.06 | 150.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:30:00 | 151.70 | 150.06 | 150.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 151.41 | 150.33 | 150.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 151.41 | 150.33 | 150.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 151.21 | 150.51 | 150.42 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 149.85 | 150.49 | 150.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 12:15:00 | 149.47 | 150.28 | 150.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 09:15:00 | 150.51 | 150.09 | 150.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 150.51 | 150.09 | 150.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 150.51 | 150.09 | 150.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 10:30:00 | 150.05 | 150.15 | 150.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 12:15:00 | 149.95 | 150.17 | 150.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 14:00:00 | 150.18 | 150.19 | 150.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 14:45:00 | 150.22 | 150.24 | 150.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 15:15:00 | 150.62 | 150.31 | 150.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 15:15:00 | 150.62 | 150.31 | 150.30 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 149.78 | 150.23 | 150.27 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 10:15:00 | 151.46 | 150.44 | 150.33 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 15:15:00 | 149.95 | 150.30 | 150.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 147.54 | 149.75 | 150.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 138.28 | 137.79 | 140.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:00:00 | 138.28 | 137.79 | 140.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 140.80 | 138.39 | 140.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 140.80 | 138.39 | 140.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 140.31 | 138.78 | 140.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 140.31 | 138.78 | 140.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 140.44 | 139.11 | 140.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 140.47 | 139.11 | 140.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 141.24 | 139.54 | 140.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 141.24 | 139.54 | 140.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 140.85 | 139.80 | 140.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 141.02 | 139.80 | 140.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 142.10 | 140.26 | 140.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:00:00 | 142.10 | 140.26 | 140.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 142.14 | 140.64 | 140.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:30:00 | 142.80 | 140.64 | 140.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 142.12 | 140.93 | 140.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 142.60 | 141.51 | 141.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 12:15:00 | 150.38 | 150.38 | 148.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 12:45:00 | 150.28 | 150.38 | 148.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 149.70 | 150.04 | 148.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:30:00 | 149.96 | 149.78 | 149.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 148.15 | 149.22 | 148.93 | SL hit (close<static) qty=1.00 sl=148.50 alert=retest2 |

### Cycle 125 — SELL (started 2025-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 15:15:00 | 147.56 | 148.61 | 148.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 146.26 | 148.14 | 148.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 149.00 | 146.99 | 147.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 149.00 | 146.99 | 147.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 149.00 | 146.99 | 147.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 149.00 | 146.99 | 147.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 148.67 | 147.33 | 147.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 149.63 | 147.33 | 147.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 149.30 | 147.95 | 147.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 149.80 | 148.32 | 148.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 13:15:00 | 148.71 | 149.64 | 149.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 13:15:00 | 148.71 | 149.64 | 149.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 13:15:00 | 148.71 | 149.64 | 149.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 14:00:00 | 148.71 | 149.64 | 149.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 148.96 | 149.50 | 149.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:45:00 | 149.25 | 149.36 | 149.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 10:30:00 | 149.35 | 149.46 | 149.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:45:00 | 149.46 | 149.58 | 149.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 147.91 | 149.13 | 149.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 10:15:00 | 147.91 | 149.13 | 149.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 11:15:00 | 147.32 | 148.77 | 149.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 11:15:00 | 145.07 | 145.03 | 146.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 12:00:00 | 145.07 | 145.03 | 146.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 146.17 | 145.34 | 146.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 146.17 | 145.34 | 146.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 146.11 | 145.49 | 146.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 146.11 | 145.49 | 146.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 145.85 | 145.57 | 146.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 145.68 | 145.57 | 146.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 145.95 | 145.64 | 146.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:00:00 | 145.95 | 145.64 | 146.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 145.74 | 145.66 | 146.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 11:45:00 | 145.22 | 145.59 | 145.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:15:00 | 145.07 | 145.69 | 145.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:00:00 | 145.18 | 145.69 | 145.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:00:00 | 144.78 | 145.56 | 145.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 144.62 | 144.71 | 145.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 13:00:00 | 143.51 | 144.59 | 145.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 14:30:00 | 143.75 | 144.33 | 144.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 148.76 | 145.32 | 145.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 148.76 | 145.32 | 145.18 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 145.35 | 146.85 | 146.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 144.99 | 146.48 | 146.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 143.61 | 143.54 | 144.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 144.02 | 143.54 | 144.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 141.90 | 143.25 | 144.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 141.81 | 142.69 | 143.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:45:00 | 141.60 | 142.16 | 143.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 12:15:00 | 144.95 | 143.53 | 143.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 144.95 | 143.53 | 143.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 14:15:00 | 145.33 | 143.95 | 143.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 12:15:00 | 144.07 | 144.43 | 144.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 12:15:00 | 144.07 | 144.43 | 144.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 144.07 | 144.43 | 144.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:15:00 | 143.93 | 144.43 | 144.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 143.50 | 144.24 | 144.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 143.50 | 144.24 | 144.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 144.90 | 144.37 | 144.13 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 142.49 | 143.80 | 143.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 141.50 | 142.72 | 143.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 139.44 | 139.05 | 140.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:30:00 | 140.15 | 139.05 | 140.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 142.65 | 139.70 | 140.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 15:00:00 | 142.65 | 139.70 | 140.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 143.00 | 140.36 | 140.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 148.80 | 142.05 | 141.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 10:15:00 | 148.04 | 148.09 | 145.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-27 10:45:00 | 148.09 | 148.09 | 145.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 151.14 | 152.02 | 150.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:30:00 | 150.56 | 152.02 | 150.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 154.40 | 152.45 | 151.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 12:00:00 | 155.25 | 153.29 | 151.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 13:30:00 | 155.09 | 153.76 | 152.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 15:00:00 | 154.88 | 153.99 | 152.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 150.10 | 152.59 | 152.32 | SL hit (close<static) qty=1.00 sl=150.32 alert=retest2 |

### Cycle 133 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 149.31 | 151.93 | 152.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 146.38 | 149.17 | 150.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 148.75 | 148.02 | 149.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 148.75 | 148.02 | 149.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 152.15 | 148.96 | 149.61 | EMA400 retest candle locked (from downside) |

### Cycle 134 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 154.65 | 150.77 | 150.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 157.25 | 154.05 | 152.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 12:15:00 | 156.88 | 156.98 | 155.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 13:00:00 | 156.88 | 156.98 | 155.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 156.20 | 157.07 | 155.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:45:00 | 156.20 | 157.07 | 155.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 157.04 | 157.06 | 156.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 156.19 | 157.06 | 156.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 168.17 | 167.81 | 165.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 168.53 | 167.68 | 166.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 14:45:00 | 168.80 | 168.11 | 166.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 15:15:00 | 164.60 | 166.42 | 166.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 164.60 | 166.42 | 166.50 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 167.24 | 166.62 | 166.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 14:15:00 | 168.00 | 167.19 | 166.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 170.93 | 171.20 | 170.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 170.93 | 171.20 | 170.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 169.41 | 170.87 | 170.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 169.41 | 170.87 | 170.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 168.99 | 170.49 | 170.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 172.05 | 170.49 | 170.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-26 10:15:00 | 189.26 | 183.70 | 180.16 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 179.50 | 181.18 | 181.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 178.51 | 180.65 | 181.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 13:15:00 | 178.25 | 177.09 | 178.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 14:00:00 | 178.25 | 177.09 | 178.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 177.82 | 177.24 | 178.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:30:00 | 178.43 | 177.24 | 178.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 181.00 | 177.91 | 178.53 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 12:15:00 | 180.27 | 179.15 | 179.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 13:15:00 | 182.07 | 179.73 | 179.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 183.05 | 183.50 | 182.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-06 15:00:00 | 183.05 | 183.50 | 182.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 183.00 | 183.40 | 182.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 173.54 | 183.40 | 182.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 173.99 | 181.52 | 181.37 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 174.25 | 180.06 | 180.72 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 181.89 | 179.34 | 179.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 182.99 | 180.07 | 179.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 182.03 | 182.38 | 181.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 15:00:00 | 182.03 | 182.38 | 181.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 181.50 | 182.20 | 181.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 179.20 | 182.20 | 181.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 177.62 | 181.29 | 180.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:00:00 | 177.62 | 181.29 | 180.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 11:15:00 | 178.71 | 180.42 | 180.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 15:15:00 | 177.75 | 179.26 | 179.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 161.05 | 160.76 | 164.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 13:15:00 | 163.61 | 162.02 | 164.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 163.61 | 162.02 | 164.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:45:00 | 164.36 | 162.02 | 164.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 160.07 | 159.61 | 161.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:00:00 | 157.70 | 159.27 | 160.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:45:00 | 157.93 | 158.93 | 160.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:30:00 | 157.79 | 158.82 | 160.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 152.36 | 158.75 | 160.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 149.81 | 156.78 | 159.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 150.03 | 156.78 | 159.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 149.90 | 156.78 | 159.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 150.28 | 148.90 | 152.13 | SL hit (close>ema200) qty=0.50 sl=148.90 alert=retest2 |

### Cycle 142 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 154.67 | 152.84 | 152.70 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 149.94 | 152.57 | 152.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 149.40 | 151.94 | 152.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 146.98 | 144.37 | 146.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 146.98 | 144.37 | 146.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 146.98 | 144.37 | 146.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 146.79 | 144.37 | 146.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 146.13 | 144.72 | 146.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 142.20 | 146.97 | 147.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 14:30:00 | 144.87 | 144.87 | 145.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 15:15:00 | 145.80 | 144.87 | 145.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:45:00 | 145.45 | 145.27 | 145.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 146.73 | 145.56 | 145.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:45:00 | 146.94 | 145.56 | 145.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 147.90 | 146.03 | 146.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 147.90 | 146.03 | 146.06 | SL hit (close>static) qty=1.00 sl=147.59 alert=retest2 |

### Cycle 144 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 149.65 | 146.75 | 146.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 150.40 | 147.48 | 146.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 148.48 | 149.07 | 147.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 148.48 | 149.07 | 147.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 148.28 | 148.83 | 148.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 13:00:00 | 148.28 | 148.83 | 148.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 165.60 | 167.10 | 164.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:30:00 | 166.40 | 166.40 | 164.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 172.05 | 174.83 | 174.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 172.05 | 174.83 | 174.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 170.96 | 173.23 | 173.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 13:15:00 | 172.41 | 171.99 | 173.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 14:00:00 | 172.41 | 171.99 | 173.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 174.40 | 172.47 | 173.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 174.40 | 172.47 | 173.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 174.60 | 172.90 | 173.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 178.16 | 172.90 | 173.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 176.26 | 173.57 | 173.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 179.59 | 174.77 | 174.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 14:15:00 | 178.56 | 180.23 | 178.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 14:15:00 | 178.56 | 180.23 | 178.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 178.56 | 180.23 | 178.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 178.56 | 180.23 | 178.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 178.00 | 179.78 | 178.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 195.49 | 179.78 | 178.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 13:15:00 | 204.56 | 205.80 | 205.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 13:15:00 | 204.56 | 205.80 | 205.82 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-13 11:30:00 | 186.85 | 2024-05-16 09:15:00 | 181.70 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2024-05-27 09:15:00 | 188.05 | 2024-05-30 12:15:00 | 188.30 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest1 | 2024-06-11 09:15:00 | 198.76 | 2024-06-12 12:15:00 | 196.05 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest1 | 2024-06-11 10:00:00 | 198.91 | 2024-06-12 12:15:00 | 196.05 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest1 | 2024-06-11 12:45:00 | 198.81 | 2024-06-12 12:15:00 | 196.05 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest1 | 2024-06-11 13:15:00 | 199.69 | 2024-06-12 12:15:00 | 196.05 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-06-28 12:15:00 | 204.19 | 2024-07-01 10:15:00 | 205.38 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-06-28 14:30:00 | 203.80 | 2024-07-01 10:15:00 | 205.38 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-07-09 14:30:00 | 202.10 | 2024-07-10 10:15:00 | 191.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-09 14:30:00 | 202.10 | 2024-07-11 10:15:00 | 195.70 | STOP_HIT | 0.50 | 3.17% |
| BUY | retest2 | 2024-07-23 15:00:00 | 195.38 | 2024-07-24 09:15:00 | 192.41 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-08-07 12:00:00 | 202.20 | 2024-08-13 09:15:00 | 192.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 09:15:00 | 202.10 | 2024-08-13 09:15:00 | 191.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 10:30:00 | 202.50 | 2024-08-13 09:15:00 | 192.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 12:15:00 | 202.68 | 2024-08-13 09:15:00 | 192.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 10:15:00 | 200.50 | 2024-08-14 09:15:00 | 190.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 11:00:00 | 200.57 | 2024-08-14 09:15:00 | 190.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 12:00:00 | 200.04 | 2024-08-14 09:15:00 | 190.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-07 12:00:00 | 202.20 | 2024-08-16 10:15:00 | 192.85 | STOP_HIT | 0.50 | 4.62% |
| SELL | retest2 | 2024-08-08 09:15:00 | 202.10 | 2024-08-16 10:15:00 | 192.85 | STOP_HIT | 0.50 | 4.58% |
| SELL | retest2 | 2024-08-08 10:30:00 | 202.50 | 2024-08-16 10:15:00 | 192.85 | STOP_HIT | 0.50 | 4.77% |
| SELL | retest2 | 2024-08-08 12:15:00 | 202.68 | 2024-08-16 10:15:00 | 192.85 | STOP_HIT | 0.50 | 4.85% |
| SELL | retest2 | 2024-08-09 10:15:00 | 200.50 | 2024-08-16 10:15:00 | 192.85 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2024-08-09 11:00:00 | 200.57 | 2024-08-16 10:15:00 | 192.85 | STOP_HIT | 0.50 | 3.85% |
| SELL | retest2 | 2024-08-09 12:00:00 | 200.04 | 2024-08-16 10:15:00 | 192.85 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2024-10-07 10:15:00 | 187.28 | 2024-10-09 11:15:00 | 189.85 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-10-18 15:00:00 | 191.95 | 2024-10-22 09:15:00 | 182.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 15:00:00 | 191.95 | 2024-10-23 10:15:00 | 178.58 | STOP_HIT | 0.50 | 6.97% |
| BUY | retest2 | 2024-11-28 09:30:00 | 173.00 | 2024-11-29 10:15:00 | 168.44 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2024-12-11 10:30:00 | 175.25 | 2024-12-13 10:15:00 | 166.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-11 10:30:00 | 175.25 | 2024-12-13 14:15:00 | 169.83 | STOP_HIT | 0.50 | 3.09% |
| SELL | retest2 | 2025-01-02 09:45:00 | 159.17 | 2025-01-02 13:15:00 | 160.80 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-01-02 10:45:00 | 158.42 | 2025-01-02 13:15:00 | 160.80 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-01-20 11:45:00 | 152.50 | 2025-01-22 09:15:00 | 149.29 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-01-21 11:45:00 | 152.60 | 2025-01-22 09:15:00 | 149.29 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-01-28 09:15:00 | 148.95 | 2025-01-28 12:15:00 | 151.85 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-01-29 15:00:00 | 150.55 | 2025-01-30 13:15:00 | 148.65 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-01-30 09:15:00 | 150.62 | 2025-01-30 13:15:00 | 148.65 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-01-30 10:30:00 | 150.53 | 2025-01-30 13:15:00 | 148.65 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-01-30 11:00:00 | 150.54 | 2025-01-30 13:15:00 | 148.65 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-02-07 09:15:00 | 152.96 | 2025-02-07 11:15:00 | 151.95 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-02-07 10:30:00 | 153.29 | 2025-02-07 11:15:00 | 151.95 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-02-07 12:30:00 | 153.30 | 2025-02-07 13:15:00 | 151.97 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-02-19 14:15:00 | 131.81 | 2025-02-19 14:15:00 | 133.51 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-03-04 09:30:00 | 142.45 | 2025-03-11 09:15:00 | 139.72 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-03-21 10:15:00 | 144.24 | 2025-03-28 13:15:00 | 145.64 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2025-04-11 10:30:00 | 148.20 | 2025-04-11 11:15:00 | 149.90 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-05-06 10:15:00 | 161.11 | 2025-05-12 09:15:00 | 163.35 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-05-08 09:45:00 | 161.15 | 2025-05-12 10:15:00 | 163.55 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-05-08 10:30:00 | 161.25 | 2025-05-12 10:15:00 | 163.55 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-05-08 11:15:00 | 161.10 | 2025-05-12 10:15:00 | 163.55 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-05-08 14:45:00 | 157.60 | 2025-05-12 10:15:00 | 163.55 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2025-05-14 11:15:00 | 165.99 | 2025-05-20 13:15:00 | 166.38 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2025-05-14 12:00:00 | 165.67 | 2025-05-20 13:15:00 | 166.38 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-05-22 10:30:00 | 165.84 | 2025-05-27 10:15:00 | 168.29 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-06-20 09:30:00 | 178.82 | 2025-07-03 10:15:00 | 184.05 | STOP_HIT | 1.00 | 2.92% |
| BUY | retest2 | 2025-06-20 10:15:00 | 179.12 | 2025-07-03 10:15:00 | 184.05 | STOP_HIT | 1.00 | 2.75% |
| BUY | retest2 | 2025-06-23 10:15:00 | 179.09 | 2025-07-03 10:15:00 | 184.05 | STOP_HIT | 1.00 | 2.77% |
| BUY | retest2 | 2025-06-23 11:45:00 | 178.91 | 2025-07-03 10:15:00 | 184.05 | STOP_HIT | 1.00 | 2.87% |
| BUY | retest2 | 2025-06-23 15:15:00 | 180.80 | 2025-07-03 10:15:00 | 184.05 | STOP_HIT | 1.00 | 1.80% |
| BUY | retest2 | 2025-06-24 10:30:00 | 180.80 | 2025-07-03 10:15:00 | 184.05 | STOP_HIT | 1.00 | 1.80% |
| BUY | retest2 | 2025-06-25 09:30:00 | 181.04 | 2025-07-03 10:15:00 | 184.05 | STOP_HIT | 1.00 | 1.66% |
| SELL | retest2 | 2025-07-14 11:15:00 | 175.52 | 2025-07-15 09:15:00 | 181.80 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2025-08-05 10:00:00 | 166.51 | 2025-08-08 09:15:00 | 167.53 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-08-05 12:30:00 | 166.59 | 2025-08-08 09:15:00 | 167.53 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-09-19 12:30:00 | 166.64 | 2025-09-22 13:15:00 | 164.11 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-09-24 10:45:00 | 161.93 | 2025-09-29 13:15:00 | 161.05 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2025-10-01 12:30:00 | 163.17 | 2025-10-08 10:15:00 | 161.30 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-10-01 13:30:00 | 163.39 | 2025-10-08 10:15:00 | 161.30 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-10-08 09:30:00 | 163.22 | 2025-10-08 10:15:00 | 161.30 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-10-16 10:45:00 | 161.17 | 2025-10-20 10:15:00 | 166.74 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2025-10-16 12:30:00 | 161.30 | 2025-10-20 10:15:00 | 166.74 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2025-10-16 15:00:00 | 161.33 | 2025-10-20 10:15:00 | 166.74 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2025-10-17 09:15:00 | 161.00 | 2025-10-20 10:15:00 | 166.74 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2025-11-10 10:15:00 | 153.51 | 2025-11-12 12:15:00 | 155.28 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-11-14 11:00:00 | 155.00 | 2025-11-18 10:15:00 | 153.81 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-11-14 12:15:00 | 155.01 | 2025-11-18 10:15:00 | 153.81 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-11-14 15:15:00 | 155.00 | 2025-11-18 10:15:00 | 153.81 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-11-28 10:30:00 | 150.05 | 2025-11-28 15:15:00 | 150.62 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-11-28 12:15:00 | 149.95 | 2025-11-28 15:15:00 | 150.62 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-11-28 14:00:00 | 150.18 | 2025-11-28 15:15:00 | 150.62 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-11-28 14:45:00 | 150.22 | 2025-11-28 15:15:00 | 150.62 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-12-17 09:30:00 | 149.96 | 2025-12-17 12:15:00 | 148.15 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-12-23 09:45:00 | 149.25 | 2025-12-24 10:15:00 | 147.91 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-12-23 10:30:00 | 149.35 | 2025-12-24 10:15:00 | 147.91 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-23 14:45:00 | 149.46 | 2025-12-24 10:15:00 | 147.91 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-12-30 11:45:00 | 145.22 | 2026-01-05 09:15:00 | 148.76 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-12-31 11:15:00 | 145.07 | 2026-01-05 09:15:00 | 148.76 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2026-01-01 10:00:00 | 145.18 | 2026-01-05 09:15:00 | 148.76 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2026-01-01 12:00:00 | 144.78 | 2026-01-05 09:15:00 | 148.76 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2026-01-02 13:00:00 | 143.51 | 2026-01-05 09:15:00 | 148.76 | STOP_HIT | 1.00 | -3.66% |
| SELL | retest2 | 2026-01-02 14:30:00 | 143.75 | 2026-01-05 09:15:00 | 148.76 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2026-01-13 11:30:00 | 141.81 | 2026-01-14 12:15:00 | 144.95 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2026-01-13 13:45:00 | 141.60 | 2026-01-14 12:15:00 | 144.95 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2026-01-30 12:00:00 | 155.25 | 2026-02-01 11:15:00 | 150.10 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2026-01-30 13:30:00 | 155.09 | 2026-02-01 11:15:00 | 150.10 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2026-01-30 15:00:00 | 154.88 | 2026-02-01 11:15:00 | 150.10 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2026-02-12 13:15:00 | 168.53 | 2026-02-13 15:15:00 | 164.60 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-02-12 14:45:00 | 168.80 | 2026-02-13 15:15:00 | 164.60 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2026-02-20 09:15:00 | 172.05 | 2026-02-26 10:15:00 | 189.26 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-20 12:00:00 | 157.70 | 2026-03-23 09:15:00 | 149.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:45:00 | 157.93 | 2026-03-23 09:15:00 | 150.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 14:30:00 | 157.79 | 2026-03-23 09:15:00 | 149.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:00:00 | 157.70 | 2026-03-24 12:15:00 | 150.28 | STOP_HIT | 0.50 | 4.71% |
| SELL | retest2 | 2026-03-20 13:45:00 | 157.93 | 2026-03-24 12:15:00 | 150.28 | STOP_HIT | 0.50 | 4.84% |
| SELL | retest2 | 2026-03-20 14:30:00 | 157.79 | 2026-03-24 12:15:00 | 150.28 | STOP_HIT | 0.50 | 4.76% |
| SELL | retest2 | 2026-03-23 09:15:00 | 152.36 | 2026-03-25 13:15:00 | 154.67 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-04-02 09:15:00 | 142.20 | 2026-04-06 11:15:00 | 147.90 | STOP_HIT | 1.00 | -4.01% |
| SELL | retest2 | 2026-04-02 14:30:00 | 144.87 | 2026-04-06 11:15:00 | 147.90 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2026-04-02 15:15:00 | 145.80 | 2026-04-06 11:15:00 | 147.90 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-04-06 09:45:00 | 145.45 | 2026-04-06 11:15:00 | 147.90 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2026-04-13 13:30:00 | 166.40 | 2026-04-23 09:15:00 | 172.05 | STOP_HIT | 1.00 | 3.40% |
| BUY | retest2 | 2026-04-29 09:15:00 | 195.49 | 2026-05-08 13:15:00 | 204.56 | STOP_HIT | 1.00 | 4.64% |
