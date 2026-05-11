# Hindustan Petroleum Corporation Ltd. (HINDPETRO)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 387.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 198 |
| ALERT1 | 144 |
| ALERT2 | 143 |
| ALERT2_SKIP | 61 |
| ALERT3 | 408 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 175 |
| PARTIAL | 20 |
| TARGET_HIT | 8 |
| STOP_HIT | 171 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 199 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 74 / 125
- **Target hits / Stop hits / Partials:** 8 / 171 / 20
- **Avg / median % per leg:** 0.38% / -0.78%
- **Sum % (uncompounded):** 75.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 89 | 27 | 30.3% | 8 | 80 | 1 | 0.40% | 35.6% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| BUY @ 3rd Alert (retest2) | 87 | 25 | 28.7% | 7 | 80 | 0 | 0.24% | 20.6% |
| SELL (all) | 110 | 47 | 42.7% | 0 | 91 | 19 | 0.37% | 40.4% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.08% | -9.2% |
| SELL @ 3rd Alert (retest2) | 107 | 47 | 43.9% | 0 | 88 | 19 | 0.46% | 49.6% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 1.15% | 5.8% |
| retest2 (combined) | 194 | 72 | 37.1% | 7 | 168 | 19 | 0.36% | 70.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 13:15:00 | 171.97 | 173.27 | 173.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 10:15:00 | 171.33 | 172.40 | 172.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 11:15:00 | 170.10 | 170.04 | 171.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-19 11:30:00 | 170.37 | 170.04 | 171.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 13:15:00 | 170.17 | 170.16 | 171.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 13:30:00 | 170.23 | 170.16 | 171.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 171.17 | 170.36 | 171.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 15:00:00 | 171.17 | 170.36 | 171.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 15:15:00 | 170.17 | 170.32 | 170.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 09:15:00 | 168.37 | 170.32 | 170.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 12:15:00 | 170.07 | 170.13 | 170.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 13:30:00 | 170.13 | 170.20 | 170.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-23 09:15:00 | 172.93 | 170.76 | 170.80 | SL hit (close>static) qty=1.00 sl=171.27 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 10:15:00 | 174.07 | 171.42 | 171.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 11:15:00 | 175.20 | 172.18 | 171.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 10:15:00 | 174.07 | 174.29 | 173.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-24 11:00:00 | 174.07 | 174.29 | 173.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 12:15:00 | 172.93 | 173.96 | 173.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 13:00:00 | 172.93 | 173.96 | 173.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 13:15:00 | 172.80 | 173.73 | 173.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 14:00:00 | 172.80 | 173.73 | 173.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 14:15:00 | 172.27 | 173.43 | 173.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 15:00:00 | 172.27 | 173.43 | 173.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2023-05-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 09:15:00 | 170.30 | 172.62 | 172.71 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 09:15:00 | 174.67 | 172.10 | 171.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 10:15:00 | 175.73 | 172.82 | 172.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 12:15:00 | 174.83 | 174.88 | 173.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-01 13:00:00 | 174.83 | 174.88 | 173.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 14:15:00 | 173.60 | 174.52 | 173.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-01 15:00:00 | 173.60 | 174.52 | 173.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 15:15:00 | 173.50 | 174.31 | 173.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 09:15:00 | 173.27 | 174.31 | 173.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 10:15:00 | 172.90 | 173.76 | 173.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 11:30:00 | 173.47 | 173.73 | 173.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-02 12:15:00 | 172.90 | 173.56 | 173.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 12:15:00 | 172.90 | 173.56 | 173.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-02 13:15:00 | 172.77 | 173.40 | 173.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-05 14:15:00 | 170.93 | 170.91 | 171.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-05 14:30:00 | 170.37 | 170.91 | 171.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 09:15:00 | 172.80 | 171.36 | 171.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-06 09:30:00 | 172.77 | 171.36 | 171.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 10:15:00 | 173.20 | 171.73 | 172.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-06 10:30:00 | 174.03 | 171.73 | 172.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2023-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 13:15:00 | 174.00 | 172.45 | 172.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-06 14:15:00 | 174.90 | 172.94 | 172.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 09:15:00 | 180.60 | 181.41 | 178.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 09:15:00 | 180.60 | 181.41 | 178.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 09:15:00 | 180.60 | 181.41 | 178.27 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 11:15:00 | 176.07 | 177.87 | 177.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 12:15:00 | 175.07 | 177.31 | 177.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 09:15:00 | 178.17 | 176.51 | 177.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 09:15:00 | 178.17 | 176.51 | 177.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 178.17 | 176.51 | 177.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 09:30:00 | 178.60 | 176.51 | 177.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 10:15:00 | 178.93 | 176.99 | 177.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 11:00:00 | 178.93 | 176.99 | 177.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2023-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 12:15:00 | 180.23 | 177.90 | 177.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-12 14:15:00 | 182.20 | 179.08 | 178.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 14:15:00 | 183.93 | 184.07 | 182.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-14 15:00:00 | 183.93 | 184.07 | 182.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 11:15:00 | 182.60 | 183.65 | 182.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 11:45:00 | 182.83 | 183.65 | 182.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 12:15:00 | 182.57 | 183.44 | 182.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 12:45:00 | 182.33 | 183.44 | 182.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 13:15:00 | 182.70 | 183.29 | 182.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 14:15:00 | 182.10 | 183.29 | 182.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 14:15:00 | 182.67 | 183.17 | 182.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 14:30:00 | 182.43 | 183.17 | 182.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 15:15:00 | 182.67 | 183.07 | 182.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-16 09:15:00 | 181.57 | 183.07 | 182.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 180.97 | 182.65 | 182.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-16 10:00:00 | 180.97 | 182.65 | 182.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2023-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-16 10:15:00 | 181.27 | 182.37 | 182.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-20 09:15:00 | 180.30 | 181.58 | 181.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 12:15:00 | 181.93 | 181.41 | 181.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 12:15:00 | 181.93 | 181.41 | 181.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 12:15:00 | 181.93 | 181.41 | 181.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 13:00:00 | 181.93 | 181.41 | 181.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 13:15:00 | 181.13 | 181.35 | 181.70 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 10:15:00 | 183.27 | 182.00 | 181.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-22 10:15:00 | 183.77 | 182.52 | 182.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 11:15:00 | 182.33 | 182.48 | 182.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-22 12:00:00 | 182.33 | 182.48 | 182.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 12:15:00 | 182.07 | 182.40 | 182.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 12:45:00 | 181.80 | 182.40 | 182.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 13:15:00 | 182.23 | 182.36 | 182.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 14:15:00 | 182.50 | 182.36 | 182.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 14:15:00 | 183.27 | 182.55 | 182.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-22 15:15:00 | 183.73 | 182.55 | 182.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-23 09:15:00 | 178.70 | 181.97 | 182.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2023-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 09:15:00 | 178.70 | 181.97 | 182.10 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 12:15:00 | 181.73 | 179.94 | 179.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 14:15:00 | 182.57 | 180.62 | 180.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 09:15:00 | 187.97 | 188.07 | 185.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-04 09:30:00 | 187.93 | 188.07 | 185.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 12:15:00 | 185.50 | 187.16 | 185.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 13:00:00 | 185.50 | 187.16 | 185.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 13:15:00 | 185.37 | 186.80 | 185.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 14:15:00 | 184.83 | 186.80 | 185.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 14:15:00 | 186.00 | 186.64 | 185.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 14:30:00 | 185.13 | 186.64 | 185.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 186.80 | 186.60 | 185.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 09:45:00 | 185.57 | 186.60 | 185.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 199.93 | 201.50 | 198.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 11:15:00 | 198.07 | 201.50 | 198.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 11:15:00 | 196.67 | 200.54 | 198.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 12:00:00 | 196.67 | 200.54 | 198.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 12:15:00 | 198.10 | 200.05 | 198.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-10 13:45:00 | 198.53 | 199.87 | 198.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-11 12:15:00 | 197.17 | 198.37 | 198.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-11 12:15:00 | 197.17 | 198.37 | 198.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-11 13:15:00 | 196.77 | 198.05 | 198.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 10:15:00 | 193.33 | 192.78 | 194.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-14 11:00:00 | 193.33 | 192.78 | 194.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 193.63 | 192.51 | 193.40 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 12:15:00 | 195.53 | 193.96 | 193.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-18 09:15:00 | 197.33 | 195.43 | 194.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 11:15:00 | 194.60 | 195.31 | 194.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 11:15:00 | 194.60 | 195.31 | 194.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 11:15:00 | 194.60 | 195.31 | 194.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 12:00:00 | 194.60 | 195.31 | 194.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 12:15:00 | 194.13 | 195.08 | 194.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 13:00:00 | 194.13 | 195.08 | 194.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 13:15:00 | 193.77 | 194.81 | 194.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 13:45:00 | 194.00 | 194.81 | 194.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2023-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 15:15:00 | 193.90 | 194.46 | 194.49 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 11:15:00 | 195.67 | 194.65 | 194.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 13:15:00 | 196.97 | 195.33 | 194.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 15:15:00 | 200.00 | 200.03 | 198.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-21 09:15:00 | 200.63 | 200.03 | 198.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 201.00 | 201.20 | 200.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 09:30:00 | 201.00 | 201.20 | 200.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 15:15:00 | 200.67 | 201.23 | 200.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 09:15:00 | 201.67 | 201.23 | 200.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 201.83 | 201.35 | 200.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 10:00:00 | 202.60 | 201.08 | 200.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 10:30:00 | 203.13 | 201.71 | 201.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 14:30:00 | 203.17 | 202.06 | 201.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-27 11:15:00 | 196.90 | 200.46 | 200.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2023-07-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 11:15:00 | 196.90 | 200.46 | 200.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-27 12:15:00 | 194.57 | 199.28 | 200.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 11:15:00 | 189.37 | 188.76 | 192.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-31 12:00:00 | 189.37 | 188.76 | 192.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 11:15:00 | 190.80 | 189.72 | 190.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 12:00:00 | 190.80 | 189.72 | 190.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 12:15:00 | 189.47 | 189.67 | 190.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-02 09:15:00 | 187.20 | 189.75 | 190.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-04 13:15:00 | 177.84 | 179.73 | 182.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-08-07 14:15:00 | 178.70 | 177.77 | 179.58 | SL hit (close>ema200) qty=0.50 sl=177.77 alert=retest2 |

### Cycle 18 — BUY (started 2023-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 11:15:00 | 178.40 | 177.74 | 177.72 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 09:15:00 | 176.10 | 177.51 | 177.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-16 10:15:00 | 174.33 | 175.53 | 176.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 09:15:00 | 175.30 | 174.88 | 175.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 09:15:00 | 175.30 | 174.88 | 175.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 175.30 | 174.88 | 175.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 11:45:00 | 174.37 | 174.81 | 175.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 10:00:00 | 174.07 | 174.52 | 175.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 13:00:00 | 174.33 | 174.38 | 174.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 15:00:00 | 174.13 | 174.41 | 174.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 173.57 | 174.18 | 174.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 09:30:00 | 173.93 | 174.18 | 174.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 11:15:00 | 174.23 | 174.08 | 174.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 12:00:00 | 174.23 | 174.08 | 174.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 12:15:00 | 174.50 | 174.16 | 174.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 12:45:00 | 174.63 | 174.16 | 174.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 13:15:00 | 174.00 | 174.13 | 174.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 13:30:00 | 174.53 | 174.13 | 174.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 14:15:00 | 173.70 | 174.04 | 174.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 14:30:00 | 174.03 | 174.04 | 174.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 09:15:00 | 174.13 | 174.03 | 174.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 09:30:00 | 174.93 | 174.03 | 174.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 11:15:00 | 174.13 | 174.04 | 174.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 11:45:00 | 174.27 | 174.04 | 174.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 12:15:00 | 174.23 | 174.08 | 174.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 12:45:00 | 174.37 | 174.08 | 174.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 13:15:00 | 173.63 | 173.99 | 174.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 13:30:00 | 174.30 | 173.99 | 174.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 173.80 | 173.79 | 174.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 09:45:00 | 173.57 | 173.79 | 174.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 13:15:00 | 173.57 | 173.53 | 173.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 14:00:00 | 173.57 | 173.53 | 173.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 173.43 | 173.51 | 173.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 14:45:00 | 173.57 | 173.51 | 173.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 09:15:00 | 174.83 | 173.71 | 173.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-24 10:00:00 | 174.83 | 173.71 | 173.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-08-24 10:15:00 | 176.93 | 174.36 | 174.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2023-08-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 10:15:00 | 176.93 | 174.36 | 174.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-25 09:15:00 | 177.73 | 176.26 | 175.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-25 11:15:00 | 176.13 | 176.26 | 175.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-25 12:00:00 | 176.13 | 176.26 | 175.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 12:15:00 | 174.63 | 175.93 | 175.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 13:00:00 | 174.63 | 175.93 | 175.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 13:15:00 | 174.33 | 175.61 | 175.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 13:30:00 | 174.47 | 175.61 | 175.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2023-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 15:15:00 | 173.83 | 175.07 | 175.10 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 09:15:00 | 176.27 | 175.31 | 175.20 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-30 09:15:00 | 172.43 | 175.54 | 175.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 09:15:00 | 169.37 | 172.11 | 173.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 14:15:00 | 166.70 | 166.24 | 168.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-01 15:00:00 | 166.70 | 166.24 | 168.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 168.27 | 166.79 | 168.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 10:00:00 | 168.27 | 166.79 | 168.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 10:15:00 | 168.30 | 167.09 | 168.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 11:15:00 | 168.30 | 167.09 | 168.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 11:15:00 | 168.33 | 167.34 | 168.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 11:30:00 | 168.37 | 167.34 | 168.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 12:15:00 | 168.43 | 167.56 | 168.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 12:30:00 | 168.20 | 167.56 | 168.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 13:15:00 | 168.67 | 167.78 | 168.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 13:45:00 | 168.20 | 167.78 | 168.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 14:15:00 | 168.63 | 167.95 | 168.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 15:00:00 | 168.63 | 167.95 | 168.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 15:15:00 | 169.00 | 168.16 | 168.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-05 09:15:00 | 169.43 | 168.16 | 168.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2023-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 09:15:00 | 171.07 | 168.74 | 168.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 12:15:00 | 171.20 | 170.13 | 169.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 14:15:00 | 169.77 | 170.14 | 169.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-07 15:00:00 | 169.77 | 170.14 | 169.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 15:15:00 | 169.13 | 169.94 | 169.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-08 09:45:00 | 170.17 | 170.07 | 169.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-08 11:00:00 | 170.63 | 170.18 | 169.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-08 13:00:00 | 170.33 | 170.32 | 169.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 12:15:00 | 169.20 | 172.56 | 172.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2023-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 12:15:00 | 169.20 | 172.56 | 172.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 14:15:00 | 167.70 | 171.07 | 171.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 12:15:00 | 169.97 | 169.84 | 170.86 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-13 15:15:00 | 169.33 | 169.88 | 170.70 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 170.43 | 169.90 | 170.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:45:00 | 171.13 | 169.90 | 170.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 10:15:00 | 172.57 | 170.43 | 170.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-09-14 10:15:00 | 172.57 | 170.43 | 170.75 | SL hit (close>ema400) qty=1.00 sl=170.75 alert=retest1 |

### Cycle 26 — BUY (started 2023-09-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 13:15:00 | 171.60 | 170.93 | 170.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 14:15:00 | 172.37 | 171.22 | 171.06 | Break + close above crossover candle high |

### Cycle 27 — SELL (started 2023-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 09:15:00 | 168.73 | 170.87 | 170.93 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 12:15:00 | 172.33 | 170.25 | 170.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-21 09:15:00 | 175.80 | 172.16 | 171.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 12:15:00 | 172.90 | 173.12 | 172.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-21 13:00:00 | 172.90 | 173.12 | 172.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 15:15:00 | 173.87 | 174.55 | 173.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-25 09:15:00 | 172.53 | 174.55 | 173.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 171.03 | 173.84 | 173.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-25 10:00:00 | 171.03 | 173.84 | 173.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 10:15:00 | 171.67 | 173.41 | 173.35 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-25 11:15:00 | 172.00 | 173.13 | 173.23 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 09:15:00 | 174.73 | 173.53 | 173.38 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 09:15:00 | 171.03 | 173.39 | 173.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-27 10:15:00 | 170.47 | 172.80 | 173.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 09:15:00 | 171.03 | 168.81 | 170.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 09:15:00 | 171.03 | 168.81 | 170.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 171.03 | 168.81 | 170.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 10:00:00 | 171.03 | 168.81 | 170.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 10:15:00 | 169.53 | 168.96 | 170.06 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 10:15:00 | 171.57 | 170.40 | 170.35 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 14:15:00 | 169.33 | 170.19 | 170.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 09:15:00 | 168.37 | 169.72 | 170.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 170.47 | 167.83 | 168.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 170.47 | 167.83 | 168.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 170.47 | 167.83 | 168.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 11:15:00 | 169.17 | 168.15 | 168.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-06 09:15:00 | 170.00 | 168.89 | 168.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2023-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 09:15:00 | 170.00 | 168.89 | 168.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 12:15:00 | 172.07 | 170.07 | 169.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 167.10 | 170.23 | 169.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 167.10 | 170.23 | 169.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 167.10 | 170.23 | 169.82 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 11:15:00 | 167.17 | 169.11 | 169.35 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 10:15:00 | 169.67 | 168.61 | 168.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 11:15:00 | 170.00 | 168.89 | 168.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 10:15:00 | 171.93 | 171.94 | 170.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-13 10:30:00 | 171.77 | 171.94 | 170.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 14:15:00 | 170.93 | 171.87 | 171.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 15:00:00 | 170.93 | 171.87 | 171.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 15:15:00 | 171.00 | 171.69 | 171.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 09:15:00 | 168.07 | 171.69 | 171.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 11:15:00 | 171.77 | 171.31 | 171.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 12:15:00 | 171.97 | 171.31 | 171.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 09:30:00 | 172.47 | 171.97 | 171.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 12:45:00 | 171.87 | 173.33 | 172.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 14:15:00 | 171.13 | 172.78 | 172.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2023-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 14:15:00 | 171.13 | 172.78 | 172.79 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-10-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-19 14:15:00 | 174.33 | 172.95 | 172.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-19 15:15:00 | 174.87 | 173.33 | 172.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-20 09:15:00 | 172.50 | 173.17 | 172.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 09:15:00 | 172.50 | 173.17 | 172.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 172.50 | 173.17 | 172.94 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 11:15:00 | 169.27 | 172.16 | 172.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 12:15:00 | 167.57 | 171.25 | 172.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 167.17 | 167.12 | 168.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-25 09:45:00 | 167.53 | 167.12 | 168.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 162.67 | 161.94 | 163.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:45:00 | 163.60 | 161.94 | 163.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 09:15:00 | 162.87 | 162.18 | 163.00 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 13:15:00 | 164.67 | 163.44 | 163.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 13:15:00 | 165.37 | 164.27 | 163.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 15:15:00 | 171.43 | 171.51 | 170.04 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 09:15:00 | 173.60 | 171.51 | 170.04 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 11:15:00 | 182.28 | 177.15 | 174.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2023-11-08 09:15:00 | 190.96 | 185.05 | 179.43 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 41 — SELL (started 2023-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 10:15:00 | 206.57 | 208.19 | 208.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-20 13:15:00 | 204.80 | 207.17 | 207.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 14:15:00 | 203.27 | 203.22 | 204.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-21 15:00:00 | 203.27 | 203.22 | 204.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 206.73 | 203.94 | 205.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-22 09:45:00 | 208.50 | 203.94 | 205.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 10:15:00 | 203.83 | 203.92 | 204.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 12:15:00 | 202.40 | 203.80 | 204.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 13:00:00 | 202.33 | 203.51 | 204.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 14:15:00 | 202.30 | 203.35 | 204.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-23 11:15:00 | 208.17 | 204.93 | 204.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2023-11-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 11:15:00 | 208.17 | 204.93 | 204.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 13:15:00 | 209.93 | 206.61 | 205.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 13:15:00 | 210.80 | 211.51 | 209.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-24 13:30:00 | 210.37 | 211.51 | 209.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 12:15:00 | 225.87 | 227.65 | 224.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 12:30:00 | 225.20 | 227.65 | 224.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 10:15:00 | 252.50 | 254.69 | 253.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 11:00:00 | 252.50 | 254.69 | 253.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 11:15:00 | 252.70 | 254.29 | 253.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 11:00:00 | 253.20 | 252.74 | 252.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-11 11:15:00 | 251.33 | 252.46 | 252.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 11:15:00 | 251.33 | 252.46 | 252.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 12:15:00 | 241.53 | 250.27 | 251.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 09:15:00 | 243.97 | 241.51 | 244.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 09:15:00 | 243.97 | 241.51 | 244.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 243.97 | 241.51 | 244.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 09:45:00 | 242.70 | 241.51 | 244.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 10:15:00 | 245.67 | 242.34 | 244.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 11:00:00 | 245.67 | 242.34 | 244.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 11:15:00 | 245.03 | 242.88 | 244.66 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 14:15:00 | 249.67 | 245.56 | 245.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 15:15:00 | 250.93 | 246.63 | 246.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 09:15:00 | 253.57 | 254.41 | 251.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 13:15:00 | 253.27 | 254.09 | 252.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 13:15:00 | 253.27 | 254.09 | 252.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 13:30:00 | 252.83 | 254.09 | 252.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 14:15:00 | 252.63 | 253.80 | 252.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 15:00:00 | 252.63 | 253.80 | 252.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 252.00 | 253.31 | 252.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:00:00 | 252.00 | 253.31 | 252.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 252.97 | 253.24 | 252.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:30:00 | 253.00 | 253.24 | 252.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 14:15:00 | 254.00 | 253.84 | 253.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 14:45:00 | 252.87 | 253.84 | 253.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 09:15:00 | 253.77 | 253.96 | 253.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 09:45:00 | 254.10 | 253.96 | 253.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 254.93 | 254.15 | 253.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 10:30:00 | 254.07 | 254.15 | 253.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 250.77 | 253.64 | 253.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:00:00 | 250.77 | 253.64 | 253.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 246.00 | 252.11 | 252.67 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 10:15:00 | 260.23 | 250.12 | 249.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 10:15:00 | 266.87 | 256.91 | 254.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 09:15:00 | 268.53 | 269.88 | 263.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-01 14:15:00 | 266.17 | 266.88 | 265.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 14:15:00 | 266.17 | 266.88 | 265.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 15:00:00 | 266.17 | 266.88 | 265.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 15:15:00 | 266.20 | 266.75 | 265.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 09:15:00 | 266.53 | 266.75 | 265.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 09:15:00 | 267.83 | 266.96 | 265.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:15:00 | 264.13 | 266.96 | 265.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 265.33 | 266.64 | 265.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:30:00 | 263.67 | 266.64 | 265.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 11:15:00 | 266.73 | 266.65 | 265.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 12:15:00 | 267.70 | 266.65 | 265.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-11 09:15:00 | 294.47 | 289.11 | 287.46 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2024-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 11:15:00 | 298.60 | 306.75 | 306.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 12:15:00 | 298.03 | 305.01 | 306.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 09:15:00 | 300.33 | 300.04 | 303.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 10:00:00 | 300.33 | 300.04 | 303.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 301.30 | 299.96 | 301.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:45:00 | 303.27 | 299.96 | 301.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 301.67 | 300.31 | 301.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:15:00 | 303.03 | 300.31 | 301.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 303.27 | 300.90 | 301.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:45:00 | 304.00 | 300.90 | 301.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 298.80 | 300.48 | 301.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 11:15:00 | 297.97 | 300.48 | 301.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 11:45:00 | 298.07 | 299.91 | 301.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 13:15:00 | 285.83 | 299.79 | 301.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-25 13:15:00 | 283.07 | 296.53 | 299.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-25 13:15:00 | 283.17 | 296.53 | 299.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-29 09:15:00 | 295.43 | 293.70 | 297.22 | SL hit (close>ema200) qty=0.50 sl=293.70 alert=retest2 |

### Cycle 48 — BUY (started 2024-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 14:15:00 | 301.17 | 298.52 | 298.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-30 09:15:00 | 313.63 | 302.24 | 300.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 09:15:00 | 311.23 | 312.59 | 307.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-31 10:00:00 | 311.23 | 312.59 | 307.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 11:15:00 | 307.93 | 311.09 | 308.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 12:00:00 | 307.93 | 311.09 | 308.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 12:15:00 | 308.43 | 310.56 | 308.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 12:45:00 | 307.73 | 310.56 | 308.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 13:15:00 | 308.00 | 310.05 | 308.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 13:30:00 | 308.90 | 310.05 | 308.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 14:15:00 | 308.63 | 309.76 | 308.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 09:15:00 | 311.40 | 309.68 | 308.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 10:30:00 | 309.87 | 309.53 | 308.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 14:45:00 | 310.80 | 309.07 | 308.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-05 10:15:00 | 340.86 | 329.21 | 321.37 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 330.90 | 345.29 | 345.94 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 12:15:00 | 348.77 | 338.82 | 337.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 13:15:00 | 352.87 | 341.63 | 339.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 11:15:00 | 374.53 | 377.40 | 366.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-16 12:00:00 | 374.53 | 377.40 | 366.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 13:15:00 | 370.53 | 374.34 | 371.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 14:00:00 | 370.53 | 374.34 | 371.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 14:15:00 | 369.47 | 373.37 | 371.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 15:00:00 | 369.47 | 373.37 | 371.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 15:15:00 | 369.67 | 372.63 | 370.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 09:15:00 | 369.23 | 372.63 | 370.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2024-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 10:15:00 | 362.47 | 368.74 | 369.32 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-20 14:15:00 | 374.10 | 369.69 | 369.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-20 15:15:00 | 377.33 | 371.22 | 370.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-21 11:15:00 | 370.77 | 371.41 | 370.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 11:15:00 | 370.77 | 371.41 | 370.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 11:15:00 | 370.77 | 371.41 | 370.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 11:30:00 | 371.00 | 371.41 | 370.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 12:15:00 | 373.67 | 371.86 | 370.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 13:30:00 | 375.27 | 371.25 | 370.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 14:15:00 | 363.53 | 369.71 | 369.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2024-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 14:15:00 | 363.53 | 369.71 | 369.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-23 09:15:00 | 353.97 | 359.10 | 362.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-26 13:15:00 | 355.13 | 350.17 | 354.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 13:15:00 | 355.13 | 350.17 | 354.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 13:15:00 | 355.13 | 350.17 | 354.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 14:00:00 | 355.13 | 350.17 | 354.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 14:15:00 | 361.13 | 352.36 | 354.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 15:00:00 | 361.13 | 352.36 | 354.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 15:15:00 | 361.97 | 354.29 | 355.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 09:15:00 | 364.57 | 354.29 | 355.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 346.40 | 341.88 | 344.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:30:00 | 350.53 | 341.88 | 344.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 10:15:00 | 344.67 | 342.44 | 344.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 10:30:00 | 345.60 | 342.44 | 344.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 11:15:00 | 346.10 | 343.17 | 344.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 12:00:00 | 346.10 | 343.17 | 344.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 13:15:00 | 347.07 | 344.38 | 344.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 14:00:00 | 347.07 | 344.38 | 344.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 15:15:00 | 344.87 | 344.47 | 344.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 09:15:00 | 344.77 | 344.47 | 344.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 09:15:00 | 345.93 | 344.76 | 344.83 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 11:15:00 | 346.47 | 345.11 | 344.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-04 09:15:00 | 347.87 | 345.60 | 345.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 338.70 | 348.19 | 348.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 09:15:00 | 338.70 | 348.19 | 348.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 338.70 | 348.19 | 348.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 338.70 | 348.19 | 348.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 10:15:00 | 341.40 | 346.83 | 347.44 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-12 13:15:00 | 347.37 | 343.35 | 343.09 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 09:15:00 | 336.83 | 341.83 | 342.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 10:15:00 | 331.33 | 339.73 | 341.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 331.27 | 329.70 | 334.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 10:00:00 | 331.27 | 329.70 | 334.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 332.67 | 330.29 | 334.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 10:30:00 | 335.13 | 330.29 | 334.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 333.47 | 330.93 | 334.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 11:45:00 | 334.30 | 330.93 | 334.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 12:15:00 | 332.33 | 331.21 | 334.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 09:15:00 | 319.37 | 332.24 | 334.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 11:15:00 | 303.40 | 320.08 | 327.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-20 10:15:00 | 300.63 | 300.27 | 305.75 | SL hit (close>ema200) qty=0.50 sl=300.27 alert=retest2 |

### Cycle 58 — BUY (started 2024-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 12:15:00 | 310.63 | 306.37 | 306.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 14:15:00 | 315.03 | 308.83 | 307.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 314.30 | 315.25 | 312.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 314.30 | 315.25 | 312.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 314.30 | 315.25 | 312.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 10:30:00 | 314.80 | 315.58 | 313.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 09:15:00 | 320.77 | 315.55 | 314.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 15:15:00 | 315.33 | 316.59 | 315.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-01 11:15:00 | 314.70 | 316.01 | 316.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2024-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-01 11:15:00 | 314.70 | 316.01 | 316.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-01 13:15:00 | 312.87 | 315.00 | 315.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 09:15:00 | 319.33 | 315.34 | 315.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-02 09:15:00 | 319.33 | 315.34 | 315.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 319.33 | 315.34 | 315.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-02 10:00:00 | 319.33 | 315.34 | 315.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2024-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-02 10:15:00 | 316.73 | 315.62 | 315.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 14:15:00 | 320.97 | 317.35 | 316.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 10:15:00 | 317.60 | 318.43 | 317.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-03 10:45:00 | 317.70 | 318.43 | 317.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 11:15:00 | 317.87 | 318.32 | 317.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 12:00:00 | 317.87 | 318.32 | 317.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 12:15:00 | 317.90 | 318.23 | 317.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 12:45:00 | 317.73 | 318.23 | 317.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 13:15:00 | 317.57 | 318.10 | 317.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 14:00:00 | 317.57 | 318.10 | 317.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 14:15:00 | 316.70 | 317.82 | 317.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 15:00:00 | 316.70 | 317.82 | 317.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 15:15:00 | 316.23 | 317.50 | 317.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 09:15:00 | 317.33 | 317.50 | 317.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-04 09:15:00 | 314.33 | 316.87 | 316.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2024-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 09:15:00 | 314.33 | 316.87 | 316.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 10:15:00 | 312.33 | 315.96 | 316.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 13:15:00 | 307.07 | 306.95 | 310.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-05 14:00:00 | 307.07 | 306.95 | 310.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 311.43 | 308.04 | 309.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-08 11:00:00 | 309.27 | 308.28 | 309.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-08 12:45:00 | 309.80 | 308.57 | 309.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-10 09:15:00 | 318.50 | 308.79 | 308.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2024-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 09:15:00 | 318.50 | 308.79 | 308.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 10:15:00 | 320.63 | 311.16 | 309.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 12:15:00 | 319.87 | 320.93 | 317.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-12 13:00:00 | 319.87 | 320.93 | 317.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 312.50 | 318.49 | 317.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 09:30:00 | 307.90 | 318.49 | 317.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2024-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 12:15:00 | 312.00 | 315.83 | 316.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 14:15:00 | 310.97 | 314.41 | 315.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 319.13 | 314.97 | 315.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 319.13 | 314.97 | 315.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 319.13 | 314.97 | 315.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:00:00 | 319.13 | 314.97 | 315.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 318.13 | 315.60 | 315.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 12:00:00 | 314.90 | 315.46 | 315.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-18 09:15:00 | 324.07 | 315.69 | 315.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2024-04-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 09:15:00 | 324.07 | 315.69 | 315.41 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-04-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 10:15:00 | 312.60 | 316.92 | 317.11 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-04-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 14:15:00 | 319.50 | 317.45 | 317.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 09:15:00 | 332.67 | 320.65 | 318.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 12:15:00 | 322.47 | 326.21 | 324.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 12:15:00 | 322.47 | 326.21 | 324.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 12:15:00 | 322.47 | 326.21 | 324.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 13:00:00 | 322.47 | 326.21 | 324.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 13:15:00 | 324.13 | 325.79 | 324.19 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 11:15:00 | 321.87 | 323.35 | 323.42 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 13:15:00 | 326.13 | 323.89 | 323.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 13:15:00 | 327.53 | 325.25 | 324.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 13:15:00 | 327.50 | 327.90 | 326.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 13:30:00 | 328.77 | 327.90 | 326.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 330.80 | 336.65 | 334.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 14:45:00 | 330.93 | 336.65 | 334.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 330.20 | 335.36 | 334.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 09:15:00 | 338.20 | 335.36 | 334.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-06 14:15:00 | 342.27 | 347.64 | 347.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 14:15:00 | 342.27 | 347.64 | 347.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 12:15:00 | 341.27 | 345.09 | 346.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 349.77 | 345.33 | 346.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 09:15:00 | 349.77 | 345.33 | 346.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 349.77 | 345.33 | 346.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 09:45:00 | 351.33 | 345.33 | 346.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 347.63 | 345.79 | 346.21 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 11:15:00 | 349.67 | 346.56 | 346.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 12:15:00 | 351.00 | 347.45 | 346.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 09:15:00 | 345.00 | 347.47 | 347.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-09 09:15:00 | 345.00 | 347.47 | 347.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 345.00 | 347.47 | 347.15 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 10:15:00 | 342.90 | 346.56 | 346.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 12:15:00 | 340.80 | 345.19 | 346.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-13 14:15:00 | 332.90 | 329.90 | 333.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-13 14:15:00 | 332.90 | 329.90 | 333.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 332.90 | 329.90 | 333.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 14:45:00 | 334.27 | 329.90 | 333.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 331.97 | 330.32 | 333.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:15:00 | 331.33 | 330.32 | 333.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 327.10 | 329.67 | 333.04 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 13:15:00 | 335.83 | 332.87 | 332.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 14:15:00 | 337.33 | 333.76 | 333.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 10:15:00 | 332.37 | 334.53 | 333.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 10:15:00 | 332.37 | 334.53 | 333.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 332.37 | 334.53 | 333.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:00:00 | 332.37 | 334.53 | 333.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 332.20 | 334.07 | 333.56 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2024-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 13:15:00 | 328.23 | 332.29 | 332.80 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 11:15:00 | 335.07 | 333.12 | 333.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 12:15:00 | 336.63 | 333.82 | 333.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-18 12:15:00 | 336.27 | 336.28 | 335.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 09:15:00 | 337.27 | 336.28 | 335.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 345.50 | 338.12 | 336.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 10:15:00 | 347.50 | 338.12 | 336.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 12:15:00 | 360.17 | 362.36 | 362.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 12:15:00 | 360.17 | 362.36 | 362.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 355.10 | 359.66 | 361.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 11:15:00 | 359.80 | 359.60 | 360.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-30 12:00:00 | 359.80 | 359.60 | 360.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 12:15:00 | 358.73 | 359.42 | 360.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 12:30:00 | 360.83 | 359.42 | 360.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 359.03 | 359.34 | 360.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:45:00 | 360.03 | 359.34 | 360.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 356.20 | 357.78 | 359.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:30:00 | 360.00 | 357.78 | 359.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 359.10 | 358.22 | 359.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:00:00 | 359.10 | 358.22 | 359.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 360.03 | 358.58 | 359.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:00:00 | 360.03 | 358.58 | 359.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 361.97 | 359.26 | 359.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:45:00 | 362.50 | 359.26 | 359.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 358.33 | 359.08 | 359.53 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 390.93 | 365.81 | 362.53 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 330.03 | 364.98 | 367.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 15:15:00 | 329.33 | 346.00 | 356.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 337.93 | 335.55 | 346.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:00:00 | 337.93 | 335.55 | 346.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 348.87 | 338.37 | 344.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 349.23 | 338.37 | 344.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 356.70 | 342.04 | 345.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 356.70 | 342.04 | 345.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 12:15:00 | 343.63 | 343.32 | 345.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 14:15:00 | 341.50 | 343.47 | 345.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-07 09:30:00 | 342.03 | 343.33 | 345.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-07 10:15:00 | 341.30 | 343.33 | 345.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 15:15:00 | 348.73 | 344.37 | 344.56 | SL hit (close>static) qty=1.00 sl=348.60 alert=retest2 |

### Cycle 78 — BUY (started 2024-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 09:15:00 | 352.73 | 346.04 | 345.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-12 09:15:00 | 354.93 | 350.43 | 348.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 13:15:00 | 350.17 | 350.97 | 349.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 14:00:00 | 350.17 | 350.97 | 349.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 351.37 | 351.26 | 350.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:15:00 | 352.43 | 351.26 | 350.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 15:15:00 | 353.33 | 351.34 | 350.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 14:45:00 | 352.83 | 354.86 | 354.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 345.73 | 352.87 | 353.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 09:15:00 | 345.73 | 352.87 | 353.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 10:15:00 | 340.10 | 346.41 | 348.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 10:15:00 | 344.25 | 342.40 | 344.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 10:15:00 | 344.25 | 342.40 | 344.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 344.25 | 342.40 | 344.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:00:00 | 344.25 | 342.40 | 344.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 342.20 | 342.36 | 344.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 12:30:00 | 341.05 | 342.22 | 344.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 13:15:00 | 340.05 | 342.22 | 344.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 09:15:00 | 333.70 | 330.53 | 330.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 09:15:00 | 333.70 | 330.53 | 330.12 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 13:15:00 | 327.50 | 330.06 | 330.33 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 12:15:00 | 331.25 | 330.36 | 330.28 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 13:15:00 | 329.55 | 330.20 | 330.21 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 14:15:00 | 331.65 | 330.49 | 330.35 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 326.40 | 329.74 | 330.03 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 14:15:00 | 336.20 | 330.79 | 330.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 15:15:00 | 336.60 | 331.96 | 330.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 11:15:00 | 346.60 | 347.57 | 342.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-12 11:45:00 | 345.80 | 347.57 | 342.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 342.10 | 345.15 | 342.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 14:30:00 | 342.25 | 345.15 | 342.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 342.50 | 344.62 | 342.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 344.30 | 344.62 | 342.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 347.75 | 345.25 | 342.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:15:00 | 353.25 | 346.93 | 344.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 10:15:00 | 349.30 | 355.37 | 353.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 11:15:00 | 348.10 | 352.70 | 352.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 11:15:00 | 348.10 | 352.70 | 352.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 12:15:00 | 345.35 | 351.23 | 352.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 349.95 | 347.57 | 349.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 349.95 | 347.57 | 349.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 349.95 | 347.57 | 349.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 349.95 | 347.57 | 349.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 352.70 | 348.59 | 350.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 352.90 | 348.59 | 350.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 347.85 | 348.45 | 349.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 12:15:00 | 347.10 | 348.45 | 349.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 329.75 | 345.01 | 347.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-23 14:15:00 | 346.10 | 344.95 | 346.70 | SL hit (close>ema200) qty=0.50 sl=344.95 alert=retest2 |

### Cycle 88 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 352.60 | 348.10 | 347.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 13:15:00 | 355.30 | 350.01 | 348.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 11:15:00 | 394.05 | 394.59 | 387.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 12:00:00 | 394.05 | 394.59 | 387.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 389.60 | 392.94 | 389.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:30:00 | 388.65 | 392.94 | 389.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 388.25 | 392.00 | 389.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:45:00 | 388.15 | 392.00 | 389.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 386.85 | 390.97 | 389.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:45:00 | 387.90 | 390.97 | 389.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 392.05 | 390.54 | 389.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:30:00 | 390.35 | 390.54 | 389.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 390.00 | 390.63 | 389.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 10:15:00 | 392.00 | 390.63 | 389.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 12:45:00 | 392.35 | 391.61 | 390.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 386.20 | 389.96 | 389.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 386.20 | 389.96 | 389.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 381.45 | 388.25 | 389.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 14:15:00 | 388.60 | 385.66 | 387.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 14:15:00 | 388.60 | 385.66 | 387.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 14:15:00 | 388.60 | 385.66 | 387.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-05 14:45:00 | 388.45 | 385.66 | 387.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 15:15:00 | 389.10 | 386.35 | 387.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:15:00 | 394.40 | 386.35 | 387.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 394.25 | 387.93 | 388.17 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2024-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 10:15:00 | 394.00 | 389.14 | 388.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-06 11:15:00 | 394.60 | 390.24 | 389.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-06 13:15:00 | 390.30 | 391.12 | 389.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-06 14:00:00 | 390.30 | 391.12 | 389.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 14:15:00 | 385.15 | 389.93 | 389.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-06 15:00:00 | 385.15 | 389.93 | 389.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 15:15:00 | 387.55 | 389.45 | 389.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 09:15:00 | 395.00 | 389.45 | 389.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 11:15:00 | 388.25 | 393.75 | 392.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 11:45:00 | 390.00 | 393.00 | 392.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 13:15:00 | 389.50 | 391.85 | 392.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 13:15:00 | 389.50 | 391.85 | 392.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 14:15:00 | 388.60 | 391.20 | 391.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 13:15:00 | 380.10 | 377.26 | 381.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 13:15:00 | 380.10 | 377.26 | 381.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 380.10 | 377.26 | 381.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:00:00 | 380.10 | 377.26 | 381.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 379.90 | 377.79 | 380.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:30:00 | 383.00 | 377.79 | 380.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 376.35 | 372.23 | 375.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 10:00:00 | 376.35 | 372.23 | 375.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 372.25 | 372.23 | 374.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 10:30:00 | 376.35 | 372.23 | 374.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 375.35 | 372.86 | 375.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 12:00:00 | 375.35 | 372.86 | 375.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 375.65 | 373.41 | 375.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 12:45:00 | 375.80 | 373.41 | 375.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 13:15:00 | 376.05 | 373.94 | 375.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 13:45:00 | 376.20 | 373.94 | 375.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 373.10 | 373.77 | 374.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 15:15:00 | 372.50 | 373.77 | 374.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 09:15:00 | 377.10 | 374.23 | 374.97 | SL hit (close>static) qty=1.00 sl=376.65 alert=retest2 |

### Cycle 92 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 379.00 | 375.77 | 375.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 380.30 | 377.62 | 376.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 13:15:00 | 405.90 | 406.77 | 401.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 14:00:00 | 405.90 | 406.77 | 401.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 404.75 | 407.91 | 405.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 404.75 | 407.91 | 405.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 406.60 | 407.65 | 405.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 405.90 | 407.65 | 405.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 403.15 | 406.75 | 405.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:00:00 | 403.15 | 406.75 | 405.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 400.95 | 405.59 | 404.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:00:00 | 400.95 | 405.59 | 404.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 403.35 | 405.16 | 404.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:00:00 | 403.35 | 405.16 | 404.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 403.95 | 404.92 | 404.64 | EMA400 retest candle locked (from upside) |

### Cycle 93 — SELL (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 09:15:00 | 400.60 | 404.02 | 404.31 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 10:15:00 | 406.85 | 404.59 | 404.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 11:15:00 | 410.75 | 405.82 | 405.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 12:15:00 | 404.80 | 405.62 | 405.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 12:15:00 | 404.80 | 405.62 | 405.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 404.80 | 405.62 | 405.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 13:00:00 | 404.80 | 405.62 | 405.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 404.85 | 405.46 | 405.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 09:15:00 | 407.90 | 405.07 | 404.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 13:30:00 | 405.90 | 406.13 | 405.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 14:45:00 | 407.85 | 406.25 | 405.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 09:15:00 | 403.75 | 405.84 | 405.64 | SL hit (close<static) qty=1.00 sl=404.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 402.75 | 405.22 | 405.38 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 12:15:00 | 411.90 | 406.66 | 406.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-29 14:15:00 | 416.45 | 409.16 | 407.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 09:15:00 | 426.25 | 427.19 | 422.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 10:00:00 | 426.25 | 427.19 | 422.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 440.25 | 445.82 | 440.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 440.25 | 445.82 | 440.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 437.75 | 444.20 | 440.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:30:00 | 440.20 | 444.20 | 440.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 439.00 | 443.16 | 440.51 | EMA400 retest candle locked (from upside) |

### Cycle 97 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 434.45 | 438.88 | 438.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 425.95 | 435.44 | 437.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 13:15:00 | 422.05 | 421.97 | 426.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 13:45:00 | 421.60 | 421.97 | 426.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 417.85 | 421.17 | 425.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 10:45:00 | 417.20 | 420.27 | 424.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:15:00 | 396.34 | 404.19 | 407.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-20 09:15:00 | 399.45 | 399.40 | 402.86 | SL hit (close>ema200) qty=0.50 sl=399.40 alert=retest2 |

### Cycle 98 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 11:15:00 | 406.75 | 402.29 | 402.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 410.00 | 405.09 | 403.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 408.70 | 413.18 | 409.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 408.70 | 413.18 | 409.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 408.70 | 413.18 | 409.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 408.70 | 413.18 | 409.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 413.95 | 413.33 | 410.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 13:15:00 | 414.35 | 413.32 | 410.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 12:45:00 | 414.45 | 414.19 | 412.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 11:15:00 | 419.75 | 433.89 | 434.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 419.75 | 433.89 | 434.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 417.65 | 430.64 | 432.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 15:15:00 | 394.20 | 391.83 | 398.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-09 09:15:00 | 409.65 | 391.83 | 398.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 405.30 | 394.53 | 399.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:45:00 | 399.20 | 397.36 | 399.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 12:30:00 | 399.20 | 397.65 | 399.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:45:00 | 399.35 | 397.91 | 399.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:00:00 | 398.20 | 397.84 | 399.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 396.50 | 397.57 | 398.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:30:00 | 397.50 | 397.57 | 398.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 394.80 | 393.12 | 394.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 15:00:00 | 394.80 | 393.12 | 394.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 395.40 | 393.58 | 394.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 396.20 | 393.58 | 394.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 401.25 | 395.11 | 395.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:30:00 | 401.05 | 395.11 | 395.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-14 11:15:00 | 397.10 | 396.07 | 395.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 11:15:00 | 397.10 | 396.07 | 395.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 13:15:00 | 401.65 | 397.42 | 396.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 10:15:00 | 429.20 | 431.59 | 423.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 11:00:00 | 429.20 | 431.59 | 423.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 425.70 | 431.01 | 426.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:00:00 | 425.70 | 431.01 | 426.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 431.75 | 431.16 | 427.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 12:00:00 | 433.95 | 431.72 | 428.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 433.65 | 431.85 | 429.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 12:15:00 | 424.65 | 428.98 | 428.62 | SL hit (close<static) qty=1.00 sl=425.15 alert=retest2 |

### Cycle 101 — SELL (started 2024-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 13:15:00 | 419.50 | 427.09 | 427.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 14:15:00 | 415.15 | 424.70 | 426.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 11:15:00 | 399.70 | 399.54 | 404.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 11:45:00 | 399.20 | 399.54 | 404.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 405.05 | 401.35 | 404.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 15:00:00 | 405.05 | 401.35 | 404.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 403.70 | 401.82 | 404.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 09:15:00 | 400.00 | 401.82 | 404.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 389.00 | 399.25 | 403.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 10:15:00 | 387.00 | 399.25 | 403.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 09:45:00 | 384.30 | 383.31 | 391.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 12:30:00 | 387.80 | 385.39 | 390.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 15:15:00 | 387.50 | 384.06 | 386.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 387.50 | 384.75 | 386.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 09:15:00 | 383.50 | 384.75 | 386.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 11:00:00 | 386.45 | 385.43 | 386.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 09:15:00 | 367.65 | 377.53 | 380.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 09:15:00 | 368.41 | 377.53 | 380.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 09:15:00 | 368.12 | 377.53 | 380.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 09:15:00 | 367.13 | 377.53 | 380.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 10:15:00 | 365.08 | 375.86 | 379.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 10:15:00 | 364.32 | 375.86 | 379.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-05 13:15:00 | 373.50 | 369.13 | 372.23 | SL hit (close>ema200) qty=0.50 sl=369.13 alert=retest2 |

### Cycle 102 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 387.00 | 374.43 | 374.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 10:15:00 | 389.70 | 377.49 | 375.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 09:15:00 | 386.05 | 391.59 | 387.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 09:15:00 | 386.05 | 391.59 | 387.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 386.05 | 391.59 | 387.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:00:00 | 386.05 | 391.59 | 387.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 384.55 | 390.18 | 387.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:00:00 | 384.55 | 390.18 | 387.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 383.15 | 388.77 | 386.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 383.15 | 388.77 | 386.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2024-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 14:15:00 | 383.05 | 385.51 | 385.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 10:15:00 | 379.10 | 381.96 | 383.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 374.15 | 373.62 | 376.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 10:15:00 | 372.75 | 373.44 | 376.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 372.75 | 373.44 | 376.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 377.70 | 373.44 | 376.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 376.50 | 374.05 | 376.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 11:45:00 | 377.30 | 374.05 | 376.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 374.65 | 374.17 | 375.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 13:45:00 | 373.35 | 374.13 | 375.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 14:15:00 | 371.30 | 374.13 | 375.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:30:00 | 369.55 | 373.11 | 374.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 12:30:00 | 372.80 | 372.46 | 374.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 372.20 | 370.75 | 372.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:30:00 | 372.60 | 370.75 | 372.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 371.30 | 370.86 | 372.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 12:15:00 | 369.35 | 370.71 | 372.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 379.40 | 365.05 | 364.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 379.40 | 365.05 | 364.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 385.55 | 376.64 | 371.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 14:15:00 | 379.70 | 380.40 | 375.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 15:00:00 | 379.70 | 380.40 | 375.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 378.00 | 379.32 | 376.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:45:00 | 376.75 | 379.32 | 376.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 384.45 | 380.72 | 378.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 10:00:00 | 385.75 | 381.93 | 380.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 10:45:00 | 385.85 | 382.69 | 381.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 11:30:00 | 386.00 | 383.30 | 381.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 13:15:00 | 386.35 | 383.76 | 382.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 389.35 | 385.40 | 383.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 13:45:00 | 392.10 | 388.75 | 386.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 14:15:00 | 392.30 | 388.75 | 386.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 15:15:00 | 391.85 | 389.28 | 387.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 12:15:00 | 402.35 | 407.91 | 408.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 12:15:00 | 402.35 | 407.91 | 408.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 14:15:00 | 397.40 | 403.23 | 405.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 10:15:00 | 406.75 | 402.41 | 404.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 10:15:00 | 406.75 | 402.41 | 404.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 10:15:00 | 406.75 | 402.41 | 404.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:00:00 | 406.75 | 402.41 | 404.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 409.55 | 403.84 | 404.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:45:00 | 410.10 | 403.84 | 404.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 13:15:00 | 406.85 | 405.50 | 405.47 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 14:15:00 | 398.50 | 405.30 | 405.81 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 09:15:00 | 410.00 | 405.71 | 405.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 10:15:00 | 417.60 | 408.09 | 406.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 10:15:00 | 411.75 | 412.04 | 409.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-26 11:00:00 | 411.75 | 412.04 | 409.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 11:15:00 | 409.40 | 411.51 | 409.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 12:00:00 | 409.40 | 411.51 | 409.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 412.25 | 411.66 | 409.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 12:30:00 | 410.05 | 411.66 | 409.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 412.35 | 415.17 | 413.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 15:00:00 | 412.35 | 415.17 | 413.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 409.70 | 414.08 | 413.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 407.10 | 414.08 | 413.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 09:15:00 | 407.00 | 412.66 | 412.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 402.70 | 408.17 | 410.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 10:15:00 | 408.05 | 407.05 | 408.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 10:15:00 | 408.05 | 407.05 | 408.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 408.05 | 407.05 | 408.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:00:00 | 408.05 | 407.05 | 408.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 406.20 | 406.85 | 408.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 12:30:00 | 407.75 | 406.85 | 408.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 409.95 | 407.47 | 408.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 409.95 | 407.47 | 408.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 409.10 | 407.79 | 408.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 09:45:00 | 406.20 | 407.45 | 408.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 13:15:00 | 413.75 | 409.02 | 408.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2025-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 13:15:00 | 413.75 | 409.02 | 408.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 12:15:00 | 414.00 | 411.02 | 410.43 | Break + close above crossover candle high |

### Cycle 111 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 399.55 | 409.38 | 409.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 393.10 | 406.13 | 408.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 09:15:00 | 389.25 | 389.14 | 394.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 10:00:00 | 389.25 | 389.14 | 394.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 391.80 | 390.30 | 393.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 386.95 | 390.44 | 392.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:30:00 | 389.85 | 390.34 | 392.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 12:15:00 | 390.05 | 388.17 | 389.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 13:15:00 | 389.85 | 388.83 | 389.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 14:15:00 | 388.30 | 388.92 | 389.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 14:45:00 | 389.00 | 388.92 | 389.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 367.60 | 384.30 | 387.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 370.36 | 384.30 | 387.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 370.55 | 384.30 | 387.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 370.36 | 384.30 | 387.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 371.75 | 369.97 | 376.35 | SL hit (close>ema200) qty=0.50 sl=369.97 alert=retest2 |

### Cycle 112 — BUY (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 10:15:00 | 369.55 | 362.96 | 362.27 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 12:15:00 | 362.40 | 365.64 | 365.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 11:15:00 | 360.20 | 363.52 | 364.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 349.40 | 347.75 | 351.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 349.40 | 347.75 | 351.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 350.00 | 348.20 | 351.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:00:00 | 350.00 | 348.20 | 351.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 350.30 | 349.25 | 351.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 350.40 | 349.25 | 351.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 350.00 | 349.40 | 351.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:30:00 | 350.45 | 349.40 | 351.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 350.50 | 349.62 | 351.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:45:00 | 350.85 | 349.62 | 351.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 352.85 | 348.54 | 349.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:45:00 | 352.75 | 348.54 | 349.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 352.30 | 349.29 | 350.06 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 13:15:00 | 351.10 | 350.55 | 350.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 10:15:00 | 355.45 | 352.09 | 351.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 10:15:00 | 354.90 | 355.58 | 353.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 10:15:00 | 354.90 | 355.58 | 353.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 354.90 | 355.58 | 353.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:00:00 | 354.90 | 355.58 | 353.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 351.40 | 354.74 | 353.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 349.05 | 354.74 | 353.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 336.20 | 351.03 | 352.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 320.95 | 342.33 | 347.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 330.60 | 328.99 | 336.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 09:45:00 | 331.35 | 328.99 | 336.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 334.30 | 331.37 | 334.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 346.45 | 331.37 | 334.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 345.60 | 334.22 | 335.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:45:00 | 344.80 | 334.22 | 335.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 11:15:00 | 341.75 | 337.39 | 336.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 09:15:00 | 347.20 | 342.35 | 339.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 11:15:00 | 342.65 | 343.18 | 340.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 12:00:00 | 342.65 | 343.18 | 340.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 340.90 | 342.72 | 340.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 341.95 | 342.72 | 340.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 342.80 | 342.74 | 341.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:45:00 | 344.65 | 343.16 | 341.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 12:30:00 | 345.50 | 343.20 | 341.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 337.80 | 341.62 | 341.55 | SL hit (close<static) qty=1.00 sl=340.90 alert=retest2 |

### Cycle 117 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 338.85 | 341.06 | 341.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 332.40 | 337.67 | 339.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 12:15:00 | 313.60 | 312.15 | 316.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 13:00:00 | 313.60 | 312.15 | 316.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 314.05 | 312.53 | 316.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:45:00 | 313.50 | 312.53 | 316.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 318.35 | 313.70 | 316.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 318.35 | 313.70 | 316.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 320.00 | 314.96 | 316.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 313.95 | 314.96 | 316.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 314.60 | 314.95 | 316.52 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 324.70 | 318.32 | 317.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 325.00 | 321.31 | 319.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 323.60 | 327.81 | 324.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 323.60 | 327.81 | 324.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 323.60 | 327.81 | 324.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 323.60 | 327.81 | 324.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 323.95 | 327.04 | 324.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:15:00 | 323.65 | 327.04 | 324.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 323.65 | 326.36 | 324.69 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 319.95 | 323.55 | 323.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 12:15:00 | 319.00 | 322.31 | 323.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 298.40 | 295.46 | 300.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 298.40 | 295.46 | 300.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 302.05 | 297.69 | 300.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 302.05 | 297.69 | 300.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 309.70 | 300.09 | 300.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 309.70 | 300.09 | 300.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2025-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 11:15:00 | 307.50 | 301.57 | 301.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 13:15:00 | 315.45 | 305.59 | 303.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 09:15:00 | 333.95 | 334.84 | 327.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 10:00:00 | 333.95 | 334.84 | 327.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 329.35 | 332.46 | 330.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 329.35 | 332.46 | 330.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 331.25 | 332.22 | 330.11 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 325.85 | 329.03 | 329.13 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 09:15:00 | 330.10 | 329.03 | 328.97 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 11:15:00 | 326.95 | 328.67 | 328.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 09:15:00 | 325.00 | 327.23 | 328.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 327.65 | 325.79 | 326.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 327.65 | 325.79 | 326.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 327.65 | 325.79 | 326.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 10:00:00 | 323.90 | 325.12 | 325.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 10:45:00 | 323.90 | 324.96 | 325.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 11:15:00 | 323.80 | 324.96 | 325.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 11:15:00 | 327.45 | 325.45 | 325.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2025-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 11:15:00 | 327.45 | 325.45 | 325.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 10:15:00 | 334.15 | 328.74 | 327.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 10:15:00 | 357.45 | 358.36 | 351.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:30:00 | 357.25 | 358.36 | 351.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 356.10 | 358.79 | 355.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:00:00 | 356.10 | 358.79 | 355.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 353.40 | 357.71 | 355.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:45:00 | 354.45 | 357.71 | 355.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 352.00 | 356.57 | 355.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 15:00:00 | 352.00 | 356.57 | 355.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 356.45 | 355.59 | 355.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 11:30:00 | 356.85 | 356.04 | 355.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 11:15:00 | 357.25 | 359.96 | 359.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 12:00:00 | 358.15 | 359.60 | 359.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 09:15:00 | 354.40 | 359.56 | 359.41 | SL hit (close<static) qty=1.00 sl=354.50 alert=retest2 |

### Cycle 125 — SELL (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 10:15:00 | 358.05 | 359.26 | 359.28 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 361.05 | 359.60 | 359.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 362.25 | 360.31 | 359.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 11:15:00 | 361.20 | 361.54 | 360.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 11:15:00 | 361.20 | 361.54 | 360.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 361.20 | 361.54 | 360.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 11:45:00 | 361.10 | 361.54 | 360.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 362.75 | 361.78 | 360.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 13:15:00 | 363.10 | 361.78 | 360.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 15:15:00 | 363.00 | 362.17 | 361.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 357.70 | 361.41 | 361.02 | SL hit (close<static) qty=1.00 sl=360.30 alert=retest2 |

### Cycle 127 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 355.20 | 359.83 | 360.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 352.60 | 358.38 | 359.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-04 14:15:00 | 358.00 | 357.49 | 358.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-04 15:00:00 | 358.00 | 357.49 | 358.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 15:15:00 | 359.30 | 357.85 | 359.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 09:15:00 | 356.00 | 357.85 | 359.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 10:15:00 | 361.00 | 357.86 | 358.76 | SL hit (close>static) qty=1.00 sl=360.00 alert=retest2 |

### Cycle 128 — BUY (started 2025-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 14:15:00 | 363.15 | 359.86 | 359.46 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 15:15:00 | 348.00 | 357.49 | 358.42 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 363.85 | 359.14 | 358.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 12:15:00 | 365.25 | 360.36 | 359.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 10:15:00 | 378.55 | 380.04 | 376.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 11:00:00 | 378.55 | 380.04 | 376.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 11:15:00 | 378.05 | 379.64 | 376.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-15 11:45:00 | 377.00 | 379.64 | 376.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 381.00 | 379.54 | 377.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 11:15:00 | 384.15 | 379.83 | 377.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 12:15:00 | 387.25 | 391.70 | 391.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2025-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 12:15:00 | 387.25 | 391.70 | 391.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 382.30 | 388.36 | 389.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 389.20 | 384.48 | 386.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 389.20 | 384.48 | 386.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 389.20 | 384.48 | 386.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 389.20 | 384.48 | 386.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 388.95 | 385.37 | 386.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:00:00 | 388.95 | 385.37 | 386.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 391.70 | 387.56 | 387.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 394.10 | 390.76 | 389.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 14:15:00 | 390.70 | 391.46 | 390.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-29 15:00:00 | 390.70 | 391.46 | 390.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 391.15 | 391.40 | 390.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 09:15:00 | 398.30 | 391.40 | 390.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 10:15:00 | 389.90 | 391.65 | 390.65 | SL hit (close<static) qty=1.00 sl=390.05 alert=retest2 |

### Cycle 133 — SELL (started 2025-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 13:15:00 | 381.35 | 389.48 | 389.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 14:15:00 | 379.45 | 387.47 | 388.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 389.15 | 386.05 | 387.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 389.15 | 386.05 | 387.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 389.15 | 386.05 | 387.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 389.15 | 386.05 | 387.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 386.20 | 386.08 | 387.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 385.10 | 386.08 | 387.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 405.00 | 389.28 | 388.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 405.00 | 389.28 | 388.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 10:15:00 | 409.80 | 393.39 | 390.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 10:15:00 | 401.50 | 404.05 | 398.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 10:45:00 | 403.05 | 404.05 | 398.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 401.00 | 403.44 | 398.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:30:00 | 398.90 | 403.44 | 398.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 399.90 | 402.73 | 399.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 12:30:00 | 399.80 | 402.73 | 399.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 399.75 | 402.13 | 399.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:00:00 | 399.75 | 402.13 | 399.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 396.75 | 401.06 | 398.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:45:00 | 395.50 | 401.06 | 398.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 397.40 | 400.33 | 398.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 399.35 | 400.33 | 398.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 400.60 | 400.11 | 398.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:15:00 | 403.85 | 400.11 | 398.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 14:15:00 | 397.00 | 399.00 | 398.76 | SL hit (close<static) qty=1.00 sl=397.10 alert=retest2 |

### Cycle 135 — SELL (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 15:15:00 | 396.95 | 398.59 | 398.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 10:15:00 | 393.80 | 397.43 | 398.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 393.10 | 388.19 | 390.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 393.10 | 388.19 | 390.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 393.10 | 388.19 | 390.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 11:45:00 | 391.75 | 389.70 | 390.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 09:15:00 | 394.30 | 390.06 | 389.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — BUY (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 09:15:00 | 394.30 | 390.06 | 389.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 12:15:00 | 397.60 | 393.05 | 391.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 10:15:00 | 406.20 | 407.17 | 403.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 11:00:00 | 406.20 | 407.17 | 403.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 404.95 | 406.40 | 404.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 406.00 | 406.40 | 404.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 408.80 | 406.88 | 404.93 | EMA400 retest candle locked (from upside) |

### Cycle 137 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 395.10 | 403.08 | 403.95 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 406.30 | 402.70 | 402.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 10:15:00 | 414.50 | 409.01 | 406.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 410.70 | 412.44 | 409.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 410.70 | 412.44 | 409.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 410.70 | 412.44 | 409.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 410.70 | 412.44 | 409.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 410.65 | 412.66 | 410.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 410.65 | 412.66 | 410.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 413.00 | 412.73 | 410.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 15:00:00 | 413.00 | 412.73 | 410.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 411.00 | 412.38 | 410.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:15:00 | 409.50 | 412.38 | 410.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 412.15 | 412.34 | 411.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:30:00 | 411.35 | 412.34 | 411.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 417.30 | 413.33 | 411.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:30:00 | 413.85 | 413.33 | 411.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 413.25 | 415.43 | 414.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:00:00 | 413.25 | 415.43 | 414.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 414.45 | 415.23 | 414.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 15:00:00 | 414.90 | 414.82 | 414.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 10:45:00 | 415.20 | 415.14 | 414.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 12:15:00 | 413.30 | 414.77 | 414.38 | SL hit (close<static) qty=1.00 sl=413.35 alert=retest2 |

### Cycle 139 — SELL (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 14:15:00 | 411.00 | 413.81 | 414.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 15:15:00 | 410.05 | 413.05 | 413.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 404.00 | 403.71 | 406.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 11:45:00 | 403.45 | 403.71 | 406.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 405.10 | 403.98 | 405.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 405.10 | 403.98 | 405.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 406.45 | 404.47 | 405.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 406.45 | 404.47 | 405.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 406.00 | 404.78 | 405.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:15:00 | 408.45 | 404.78 | 405.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 408.00 | 405.42 | 406.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:00:00 | 408.00 | 405.42 | 406.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 405.80 | 405.50 | 406.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:00:00 | 403.80 | 405.47 | 406.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 15:15:00 | 407.50 | 405.66 | 405.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 15:15:00 | 407.50 | 405.66 | 405.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 10:15:00 | 408.15 | 406.45 | 405.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 15:15:00 | 407.50 | 407.63 | 406.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 09:15:00 | 407.20 | 407.63 | 406.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 407.85 | 407.67 | 406.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:45:00 | 406.35 | 407.67 | 406.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 407.60 | 407.66 | 406.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 407.60 | 407.66 | 406.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 406.30 | 407.39 | 406.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:00:00 | 406.30 | 407.39 | 406.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 406.90 | 407.29 | 406.87 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2025-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 15:15:00 | 405.05 | 406.35 | 406.50 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 09:15:00 | 416.70 | 408.42 | 407.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 10:15:00 | 420.85 | 410.90 | 408.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 396.85 | 410.84 | 410.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 396.85 | 410.84 | 410.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 396.85 | 410.84 | 410.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 396.85 | 410.84 | 410.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 395.85 | 407.84 | 408.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 394.10 | 405.10 | 407.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 388.20 | 387.68 | 393.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 10:45:00 | 389.80 | 387.68 | 393.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 394.80 | 389.61 | 392.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 394.80 | 389.61 | 392.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 395.85 | 390.86 | 392.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 395.85 | 390.86 | 392.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 395.20 | 391.72 | 393.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 398.40 | 391.72 | 393.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 398.65 | 394.09 | 394.02 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-06-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 15:15:00 | 392.50 | 394.04 | 394.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 11:15:00 | 390.75 | 393.05 | 393.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 14:15:00 | 393.75 | 392.77 | 393.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 14:15:00 | 393.75 | 392.77 | 393.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 393.75 | 392.77 | 393.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:30:00 | 393.95 | 392.77 | 393.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 394.10 | 393.04 | 393.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 394.50 | 393.04 | 393.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 391.50 | 392.73 | 393.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:30:00 | 390.55 | 392.89 | 393.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 13:15:00 | 389.85 | 392.86 | 393.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 15:15:00 | 390.00 | 392.54 | 392.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 390.10 | 392.22 | 392.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 392.10 | 391.52 | 392.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 392.10 | 391.52 | 392.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 391.15 | 391.44 | 392.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:30:00 | 389.65 | 390.71 | 391.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 14:15:00 | 393.60 | 390.57 | 391.03 | SL hit (close>static) qty=1.00 sl=393.45 alert=retest2 |

### Cycle 146 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 408.10 | 394.52 | 392.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 10:15:00 | 416.80 | 409.74 | 405.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 10:15:00 | 436.60 | 437.18 | 431.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 10:45:00 | 436.15 | 437.18 | 431.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 435.00 | 437.66 | 435.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 435.20 | 437.66 | 435.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 435.50 | 437.23 | 435.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:45:00 | 433.60 | 437.23 | 435.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 436.85 | 437.15 | 435.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:45:00 | 435.00 | 437.15 | 435.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 439.00 | 437.50 | 435.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 11:30:00 | 441.65 | 437.99 | 437.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:45:00 | 442.00 | 439.58 | 438.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 10:15:00 | 443.50 | 446.92 | 447.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 443.50 | 446.92 | 447.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 14:15:00 | 441.35 | 443.94 | 445.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 434.15 | 433.02 | 436.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 434.15 | 433.02 | 436.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 434.15 | 433.02 | 436.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 10:45:00 | 433.00 | 433.20 | 435.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 11:15:00 | 433.30 | 433.20 | 435.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 10:15:00 | 443.15 | 435.82 | 435.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 10:15:00 | 443.15 | 435.82 | 435.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 11:15:00 | 444.20 | 437.49 | 436.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 441.75 | 442.24 | 439.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 441.75 | 442.24 | 439.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 441.75 | 442.24 | 439.54 | EMA400 retest candle locked (from upside) |

### Cycle 149 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 434.85 | 439.05 | 439.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 433.00 | 437.06 | 438.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 433.40 | 432.74 | 434.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 433.40 | 432.74 | 434.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 433.40 | 432.74 | 434.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:00:00 | 430.20 | 432.12 | 433.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 12:15:00 | 430.25 | 430.33 | 431.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 13:00:00 | 430.70 | 430.40 | 431.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 433.95 | 432.48 | 432.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 433.95 | 432.48 | 432.37 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 426.00 | 431.54 | 432.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 423.85 | 428.51 | 430.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 428.80 | 426.89 | 429.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 428.80 | 426.89 | 429.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 428.80 | 426.89 | 429.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 427.70 | 426.89 | 429.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 425.65 | 426.64 | 428.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:30:00 | 427.55 | 426.64 | 428.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 423.90 | 422.72 | 424.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:30:00 | 424.70 | 422.72 | 424.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 426.00 | 423.38 | 424.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:45:00 | 426.15 | 423.38 | 424.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 426.40 | 423.98 | 424.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:15:00 | 418.65 | 423.98 | 424.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 11:15:00 | 432.00 | 424.62 | 424.87 | SL hit (close>static) qty=1.00 sl=426.60 alert=retest2 |

### Cycle 152 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 430.45 | 425.79 | 425.37 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 411.50 | 423.05 | 424.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 10:15:00 | 409.80 | 416.28 | 419.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 407.35 | 406.85 | 411.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:00:00 | 407.35 | 406.85 | 411.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 402.65 | 398.39 | 400.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 402.65 | 398.39 | 400.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 403.00 | 399.31 | 400.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 409.00 | 399.31 | 400.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 10:15:00 | 409.55 | 402.56 | 402.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 11:15:00 | 412.85 | 404.62 | 403.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 10:15:00 | 404.50 | 407.18 | 405.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 10:15:00 | 404.50 | 407.18 | 405.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 404.50 | 407.18 | 405.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 404.50 | 407.18 | 405.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 405.85 | 406.91 | 405.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:45:00 | 403.75 | 406.91 | 405.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 407.80 | 407.09 | 405.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:45:00 | 405.05 | 407.09 | 405.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 407.70 | 408.75 | 407.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 407.70 | 408.75 | 407.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 408.05 | 408.61 | 407.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 408.75 | 408.53 | 407.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 11:15:00 | 408.70 | 408.83 | 407.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 12:15:00 | 408.80 | 408.74 | 408.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 12:45:00 | 410.05 | 409.09 | 408.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 398.00 | 407.77 | 408.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 398.00 | 407.77 | 408.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 10:15:00 | 396.35 | 405.49 | 406.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 390.05 | 389.43 | 394.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 10:00:00 | 390.05 | 389.43 | 394.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 394.95 | 390.53 | 394.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 394.95 | 390.53 | 394.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 393.70 | 391.17 | 394.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:45:00 | 394.05 | 391.17 | 394.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 393.15 | 391.94 | 394.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:30:00 | 394.60 | 391.94 | 394.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 395.40 | 392.63 | 394.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:45:00 | 395.70 | 392.63 | 394.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 395.05 | 393.11 | 394.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 393.25 | 393.11 | 394.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 392.80 | 393.10 | 394.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 11:15:00 | 391.95 | 393.10 | 394.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 12:15:00 | 395.20 | 393.61 | 394.25 | SL hit (close>static) qty=1.00 sl=394.50 alert=retest2 |

### Cycle 156 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 384.80 | 382.67 | 382.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 385.90 | 383.32 | 382.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 384.05 | 384.44 | 383.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 384.05 | 384.44 | 383.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 384.05 | 384.44 | 383.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 384.05 | 384.44 | 383.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 384.65 | 384.48 | 383.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:30:00 | 384.20 | 384.48 | 383.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 388.60 | 385.39 | 384.25 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 383.10 | 385.10 | 385.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 381.35 | 384.22 | 384.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 385.20 | 384.05 | 384.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 385.20 | 384.05 | 384.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 385.20 | 384.05 | 384.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 385.20 | 384.05 | 384.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 383.55 | 383.95 | 384.53 | EMA400 retest candle locked (from downside) |

### Cycle 158 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 392.10 | 385.64 | 385.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 12:15:00 | 394.00 | 389.41 | 387.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 11:15:00 | 392.60 | 392.73 | 390.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 12:00:00 | 392.60 | 392.73 | 390.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 390.20 | 391.95 | 390.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:00:00 | 390.20 | 391.95 | 390.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 390.00 | 391.56 | 390.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:15:00 | 389.60 | 391.56 | 390.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 389.60 | 391.17 | 390.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 389.65 | 391.17 | 390.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 392.35 | 392.06 | 391.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 397.25 | 392.13 | 391.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-29 10:15:00 | 436.98 | 427.89 | 425.09 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 10:15:00 | 435.75 | 436.52 | 436.56 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 12:15:00 | 440.55 | 437.27 | 436.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 446.70 | 439.77 | 438.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 441.15 | 441.31 | 439.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:45:00 | 440.70 | 441.31 | 439.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 460.10 | 458.46 | 456.47 | EMA400 retest candle locked (from upside) |

### Cycle 161 — SELL (started 2025-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 15:15:00 | 452.90 | 455.45 | 455.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 446.05 | 453.57 | 454.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 451.80 | 450.51 | 452.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 15:00:00 | 451.80 | 450.51 | 452.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 447.50 | 450.16 | 452.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:15:00 | 446.15 | 450.16 | 452.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:00:00 | 446.00 | 449.33 | 451.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 454.40 | 448.58 | 448.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 454.40 | 448.58 | 448.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 13:15:00 | 456.65 | 452.65 | 451.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 14:15:00 | 452.55 | 452.63 | 451.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 14:45:00 | 452.50 | 452.63 | 451.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 163 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 440.45 | 450.52 | 450.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 14:15:00 | 439.65 | 444.56 | 447.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 449.85 | 442.30 | 443.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 449.85 | 442.30 | 443.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 449.85 | 442.30 | 443.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:45:00 | 449.00 | 442.30 | 443.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 448.15 | 443.47 | 444.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:00:00 | 447.60 | 444.30 | 444.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 452.40 | 445.92 | 445.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 452.40 | 445.92 | 445.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 13:15:00 | 454.00 | 447.54 | 446.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 13:15:00 | 484.50 | 485.29 | 480.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 14:00:00 | 484.50 | 485.29 | 480.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 473.50 | 482.52 | 480.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 473.50 | 482.52 | 480.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 475.45 | 481.11 | 480.22 | EMA400 retest candle locked (from upside) |

### Cycle 165 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 472.35 | 478.27 | 479.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 469.75 | 474.58 | 476.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 477.65 | 474.76 | 476.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 12:15:00 | 477.65 | 474.76 | 476.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 477.65 | 474.76 | 476.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 477.65 | 474.76 | 476.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 476.50 | 475.11 | 476.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 478.75 | 475.11 | 476.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 476.10 | 476.08 | 476.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 483.15 | 476.08 | 476.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 484.20 | 477.71 | 477.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 487.30 | 481.25 | 479.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 14:15:00 | 485.50 | 486.03 | 483.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 14:45:00 | 485.25 | 486.03 | 483.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 483.55 | 485.54 | 483.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 487.85 | 485.54 | 483.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:15:00 | 485.60 | 485.49 | 483.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:45:00 | 485.75 | 485.53 | 484.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 14:15:00 | 485.70 | 485.55 | 484.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 486.40 | 485.72 | 484.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 486.70 | 485.66 | 484.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 09:15:00 | 483.85 | 485.30 | 484.62 | SL hit (close<static) qty=1.00 sl=484.10 alert=retest2 |

### Cycle 167 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 479.65 | 483.31 | 483.78 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 490.65 | 483.62 | 483.53 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 09:15:00 | 482.70 | 484.37 | 484.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 10:15:00 | 478.15 | 483.13 | 483.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 13:15:00 | 479.50 | 478.35 | 480.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 13:15:00 | 479.50 | 478.35 | 480.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 479.50 | 478.35 | 480.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:45:00 | 480.90 | 478.35 | 480.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 478.20 | 478.32 | 479.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 475.00 | 478.43 | 479.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 14:15:00 | 451.25 | 455.97 | 458.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 12:15:00 | 451.50 | 450.06 | 452.65 | SL hit (close>ema200) qty=0.50 sl=450.06 alert=retest2 |

### Cycle 170 — BUY (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 12:15:00 | 451.75 | 449.25 | 449.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 10:15:00 | 455.45 | 451.04 | 450.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 12:15:00 | 450.80 | 451.28 | 450.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 12:15:00 | 450.80 | 451.28 | 450.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 450.80 | 451.28 | 450.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 12:30:00 | 450.95 | 451.28 | 450.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 449.75 | 450.97 | 450.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:00:00 | 449.75 | 450.97 | 450.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 450.00 | 450.78 | 450.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:30:00 | 448.65 | 450.78 | 450.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 448.80 | 450.38 | 450.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 450.75 | 450.38 | 450.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 445.50 | 449.41 | 449.70 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 452.85 | 449.80 | 449.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 456.25 | 451.09 | 450.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 463.00 | 463.79 | 459.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 463.00 | 463.79 | 459.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 463.30 | 462.83 | 460.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:30:00 | 464.00 | 463.38 | 460.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 459.20 | 464.62 | 463.53 | SL hit (close<static) qty=1.00 sl=460.20 alert=retest2 |

### Cycle 173 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 469.30 | 472.34 | 472.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 465.85 | 471.04 | 471.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 471.50 | 468.81 | 469.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 471.50 | 468.81 | 469.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 471.50 | 468.81 | 469.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:45:00 | 471.35 | 468.81 | 469.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 473.35 | 469.72 | 470.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:30:00 | 473.45 | 469.72 | 470.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 12:15:00 | 471.25 | 470.64 | 470.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 15:15:00 | 475.75 | 472.30 | 471.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 12:15:00 | 473.90 | 474.35 | 472.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-30 13:00:00 | 473.90 | 474.35 | 472.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 471.45 | 473.77 | 472.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:45:00 | 470.05 | 473.77 | 472.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 468.80 | 472.77 | 472.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 468.80 | 472.77 | 472.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 489.00 | 496.15 | 494.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 489.00 | 496.15 | 494.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 492.80 | 495.48 | 494.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:30:00 | 491.70 | 495.48 | 494.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 479.40 | 491.55 | 492.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 476.00 | 488.44 | 491.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 449.60 | 448.94 | 454.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:00:00 | 449.60 | 448.94 | 454.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 452.20 | 444.50 | 446.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:45:00 | 451.00 | 444.50 | 446.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 454.05 | 446.41 | 447.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:00:00 | 454.05 | 446.41 | 447.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 12:15:00 | 453.60 | 449.41 | 448.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 14:15:00 | 458.15 | 451.99 | 450.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 10:15:00 | 453.30 | 453.69 | 451.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 10:45:00 | 453.95 | 453.69 | 451.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 452.95 | 453.34 | 452.05 | EMA400 retest candle locked (from upside) |

### Cycle 177 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 440.80 | 450.88 | 451.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 438.45 | 444.25 | 447.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 09:15:00 | 430.20 | 428.90 | 433.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-23 10:00:00 | 430.20 | 428.90 | 433.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 429.55 | 421.07 | 423.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:45:00 | 429.75 | 421.07 | 423.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 436.40 | 424.14 | 424.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 436.40 | 424.14 | 424.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 431.90 | 425.69 | 425.02 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 425.25 | 428.64 | 428.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 09:15:00 | 421.65 | 426.53 | 427.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 13:15:00 | 427.65 | 423.68 | 425.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 13:15:00 | 427.65 | 423.68 | 425.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 427.65 | 423.68 | 425.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 427.65 | 423.68 | 425.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 430.70 | 425.08 | 426.21 | EMA400 retest candle locked (from downside) |

### Cycle 180 — BUY (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 09:15:00 | 447.70 | 430.71 | 428.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 13:15:00 | 448.20 | 440.11 | 434.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 10:15:00 | 445.15 | 445.42 | 439.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-03 10:45:00 | 444.50 | 445.42 | 439.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 458.75 | 462.83 | 461.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 465.65 | 461.17 | 461.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:15:00 | 467.00 | 461.77 | 461.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 458.50 | 461.17 | 461.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 458.50 | 461.17 | 461.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 455.75 | 460.08 | 460.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 449.20 | 448.97 | 451.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 14:00:00 | 449.20 | 448.97 | 451.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 452.00 | 449.57 | 451.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 452.00 | 449.57 | 451.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 451.90 | 450.04 | 451.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 447.00 | 450.04 | 451.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 14:15:00 | 454.30 | 450.57 | 451.05 | SL hit (close>static) qty=1.00 sl=452.60 alert=retest2 |

### Cycle 182 — BUY (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 15:15:00 | 454.70 | 451.40 | 451.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 12:15:00 | 455.35 | 453.51 | 452.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 446.85 | 453.55 | 453.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 446.85 | 453.55 | 453.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 446.85 | 453.55 | 453.00 | EMA400 retest candle locked (from upside) |

### Cycle 183 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 447.45 | 452.33 | 452.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 444.90 | 450.85 | 451.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 435.50 | 433.59 | 438.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 10:15:00 | 440.35 | 434.94 | 438.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 440.35 | 434.94 | 438.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 440.35 | 434.94 | 438.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 438.50 | 435.65 | 438.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:15:00 | 438.10 | 435.65 | 438.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:45:00 | 437.60 | 436.02 | 438.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 14:30:00 | 437.15 | 436.99 | 438.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 10:15:00 | 441.90 | 438.63 | 439.10 | SL hit (close>static) qty=1.00 sl=441.40 alert=retest2 |

### Cycle 184 — BUY (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 12:15:00 | 441.45 | 439.61 | 439.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 14:15:00 | 447.45 | 441.50 | 440.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 09:15:00 | 439.00 | 441.96 | 440.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 439.00 | 441.96 | 440.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 439.00 | 441.96 | 440.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 439.00 | 441.96 | 440.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 438.30 | 441.23 | 440.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:15:00 | 435.40 | 441.23 | 440.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — SELL (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 11:15:00 | 434.15 | 439.81 | 440.03 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 14:15:00 | 443.85 | 440.00 | 439.50 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 438.50 | 439.41 | 439.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 429.85 | 437.50 | 438.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 409.25 | 408.46 | 417.76 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:15:00 | 402.25 | 408.46 | 417.76 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:15:00 | 401.60 | 407.32 | 416.39 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 416.65 | 408.43 | 413.95 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 416.65 | 408.43 | 413.95 | SL hit (close>ema400) qty=1.00 sl=413.95 alert=retest1 |

### Cycle 188 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 336.10 | 330.91 | 330.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 344.75 | 333.68 | 331.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 14:15:00 | 340.75 | 342.22 | 339.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 14:15:00 | 340.75 | 342.22 | 339.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 340.75 | 342.22 | 339.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 340.75 | 342.22 | 339.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 338.35 | 341.13 | 339.40 | EMA400 retest candle locked (from upside) |

### Cycle 189 — SELL (started 2026-03-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 14:15:00 | 335.50 | 338.49 | 338.62 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 344.40 | 339.06 | 338.72 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 14:15:00 | 335.00 | 338.25 | 338.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 319.50 | 333.98 | 336.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 12:15:00 | 326.70 | 324.13 | 328.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 13:00:00 | 326.70 | 324.13 | 328.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 327.50 | 324.80 | 328.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 13:30:00 | 326.00 | 324.80 | 328.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 327.65 | 325.37 | 327.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:30:00 | 328.95 | 325.37 | 327.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 15:15:00 | 328.85 | 326.07 | 328.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:15:00 | 324.20 | 326.07 | 328.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 14:15:00 | 332.00 | 325.78 | 326.68 | SL hit (close>static) qty=1.00 sl=330.60 alert=retest2 |

### Cycle 192 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 358.45 | 332.93 | 329.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 11:15:00 | 365.70 | 343.76 | 335.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 357.40 | 357.74 | 349.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 356.50 | 357.74 | 349.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 348.50 | 357.52 | 355.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 349.80 | 357.52 | 355.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 12:15:00 | 347.35 | 352.38 | 353.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — SELL (started 2026-04-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 12:15:00 | 347.35 | 352.38 | 353.01 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 365.35 | 353.90 | 353.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 12:15:00 | 368.60 | 360.48 | 356.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 369.55 | 370.33 | 367.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 09:45:00 | 369.05 | 370.33 | 367.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 382.20 | 382.58 | 378.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 11:15:00 | 384.10 | 382.61 | 378.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 374.45 | 381.47 | 380.13 | SL hit (close<static) qty=1.00 sl=376.30 alert=retest2 |

### Cycle 195 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 374.30 | 378.55 | 378.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 371.90 | 374.85 | 376.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 374.80 | 373.75 | 375.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 376.20 | 373.75 | 375.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 381.75 | 375.35 | 376.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 383.95 | 375.35 | 376.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 381.70 | 376.62 | 376.56 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 372.25 | 378.72 | 378.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 370.10 | 377.00 | 378.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 375.55 | 375.00 | 376.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 14:45:00 | 376.30 | 375.00 | 376.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 376.10 | 375.16 | 376.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 372.90 | 375.60 | 376.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 15:00:00 | 373.25 | 374.95 | 375.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 369.15 | 374.85 | 375.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 13:15:00 | 373.35 | 372.55 | 374.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 373.40 | 372.72 | 374.10 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-06 10:15:00 | 381.15 | 375.85 | 375.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 381.15 | 375.85 | 375.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 11:15:00 | 382.20 | 377.12 | 375.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 391.40 | 393.77 | 388.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:45:00 | 391.75 | 393.77 | 388.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 388.85 | 392.40 | 388.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:00:00 | 388.85 | 392.40 | 388.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 388.45 | 391.61 | 388.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 388.45 | 391.61 | 388.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 385.30 | 390.35 | 388.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 385.30 | 390.35 | 388.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 386.45 | 389.57 | 388.36 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-16 09:15:00 | 173.77 | 2023-05-17 12:15:00 | 172.17 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2023-05-16 10:00:00 | 173.30 | 2023-05-17 12:15:00 | 172.17 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2023-05-16 10:30:00 | 173.83 | 2023-05-17 12:15:00 | 172.17 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2023-05-22 09:15:00 | 168.37 | 2023-05-23 09:15:00 | 172.93 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2023-05-22 12:15:00 | 170.07 | 2023-05-23 09:15:00 | 172.93 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2023-05-22 13:30:00 | 170.13 | 2023-05-23 09:15:00 | 172.93 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2023-06-02 11:30:00 | 173.47 | 2023-06-02 12:15:00 | 172.90 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2023-06-22 15:15:00 | 183.73 | 2023-06-23 09:15:00 | 178.70 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2023-07-10 13:45:00 | 198.53 | 2023-07-11 12:15:00 | 197.17 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2023-07-26 10:00:00 | 202.60 | 2023-07-27 11:15:00 | 196.90 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2023-07-26 10:30:00 | 203.13 | 2023-07-27 11:15:00 | 196.90 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2023-07-26 14:30:00 | 203.17 | 2023-07-27 11:15:00 | 196.90 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2023-08-02 09:15:00 | 187.20 | 2023-08-04 13:15:00 | 177.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-02 09:15:00 | 187.20 | 2023-08-07 14:15:00 | 178.70 | STOP_HIT | 0.50 | 4.54% |
| SELL | retest2 | 2023-08-17 11:45:00 | 174.37 | 2023-08-24 10:15:00 | 176.93 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2023-08-18 10:00:00 | 174.07 | 2023-08-24 10:15:00 | 176.93 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2023-08-18 13:00:00 | 174.33 | 2023-08-24 10:15:00 | 176.93 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2023-08-18 15:00:00 | 174.13 | 2023-08-24 10:15:00 | 176.93 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2023-09-08 09:45:00 | 170.17 | 2023-09-12 12:15:00 | 169.20 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2023-09-08 11:00:00 | 170.63 | 2023-09-12 12:15:00 | 169.20 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2023-09-08 13:00:00 | 170.33 | 2023-09-12 12:15:00 | 169.20 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest1 | 2023-09-13 15:15:00 | 169.33 | 2023-09-14 10:15:00 | 172.57 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2023-09-14 12:30:00 | 170.97 | 2023-09-14 13:15:00 | 171.60 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2023-10-05 11:15:00 | 169.17 | 2023-10-06 09:15:00 | 170.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-10-16 12:15:00 | 171.97 | 2023-10-18 14:15:00 | 171.13 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-10-17 09:30:00 | 172.47 | 2023-10-18 14:15:00 | 171.13 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2023-10-18 12:45:00 | 171.87 | 2023-10-18 14:15:00 | 171.13 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2023-11-06 09:15:00 | 173.60 | 2023-11-07 11:15:00 | 182.28 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-11-06 09:15:00 | 173.60 | 2023-11-08 09:15:00 | 190.96 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2023-11-13 11:15:00 | 201.03 | 2023-11-17 09:15:00 | 221.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-13 13:00:00 | 202.83 | 2023-11-17 09:15:00 | 223.11 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-11-22 12:15:00 | 202.40 | 2023-11-23 11:15:00 | 208.17 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2023-11-22 13:00:00 | 202.33 | 2023-11-23 11:15:00 | 208.17 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2023-11-22 14:15:00 | 202.30 | 2023-11-23 11:15:00 | 208.17 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2023-12-11 11:00:00 | 253.20 | 2023-12-11 11:15:00 | 251.33 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-01-02 12:15:00 | 267.70 | 2024-01-11 09:15:00 | 294.47 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-01-25 11:15:00 | 297.97 | 2024-01-25 13:15:00 | 283.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-25 11:45:00 | 298.07 | 2024-01-25 13:15:00 | 283.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-25 11:15:00 | 297.97 | 2024-01-29 09:15:00 | 295.43 | STOP_HIT | 0.50 | 0.85% |
| SELL | retest2 | 2024-01-25 11:45:00 | 298.07 | 2024-01-29 09:15:00 | 295.43 | STOP_HIT | 0.50 | 0.89% |
| SELL | retest2 | 2024-01-25 13:15:00 | 285.83 | 2024-01-29 13:15:00 | 305.00 | STOP_HIT | 1.00 | -6.71% |
| BUY | retest2 | 2024-02-01 09:15:00 | 311.40 | 2024-02-05 10:15:00 | 340.86 | TARGET_HIT | 1.00 | 9.46% |
| BUY | retest2 | 2024-02-01 10:30:00 | 309.87 | 2024-02-05 11:15:00 | 341.88 | TARGET_HIT | 1.00 | 10.33% |
| BUY | retest2 | 2024-02-01 14:45:00 | 310.80 | 2024-02-05 13:15:00 | 342.54 | TARGET_HIT | 1.00 | 10.21% |
| BUY | retest2 | 2024-02-21 13:30:00 | 375.27 | 2024-02-21 14:15:00 | 363.53 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2024-03-15 09:15:00 | 319.37 | 2024-03-15 11:15:00 | 303.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-15 09:15:00 | 319.37 | 2024-03-20 10:15:00 | 300.63 | STOP_HIT | 0.50 | 5.87% |
| BUY | retest2 | 2024-03-26 10:30:00 | 314.80 | 2024-04-01 11:15:00 | 314.70 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2024-03-27 09:15:00 | 320.77 | 2024-04-01 11:15:00 | 314.70 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-03-27 15:15:00 | 315.33 | 2024-04-01 11:15:00 | 314.70 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2024-04-04 09:15:00 | 317.33 | 2024-04-04 09:15:00 | 314.33 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-04-08 11:00:00 | 309.27 | 2024-04-10 09:15:00 | 318.50 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2024-04-08 12:45:00 | 309.80 | 2024-04-10 09:15:00 | 318.50 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2024-04-16 12:00:00 | 314.90 | 2024-04-18 09:15:00 | 324.07 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest2 | 2024-05-02 09:15:00 | 338.20 | 2024-05-06 14:15:00 | 342.27 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest2 | 2024-05-21 10:15:00 | 347.50 | 2024-05-29 12:15:00 | 360.17 | STOP_HIT | 1.00 | 3.65% |
| SELL | retest2 | 2024-06-06 14:15:00 | 341.50 | 2024-06-07 15:15:00 | 348.73 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2024-06-07 09:30:00 | 342.03 | 2024-06-07 15:15:00 | 348.73 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-06-07 10:15:00 | 341.30 | 2024-06-07 15:15:00 | 348.73 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2024-06-13 12:15:00 | 352.43 | 2024-06-19 09:15:00 | 345.73 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-06-13 15:15:00 | 353.33 | 2024-06-19 09:15:00 | 345.73 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2024-06-18 14:45:00 | 352.83 | 2024-06-19 09:15:00 | 345.73 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2024-06-24 12:30:00 | 341.05 | 2024-07-05 09:15:00 | 333.70 | STOP_HIT | 1.00 | 2.16% |
| SELL | retest2 | 2024-06-24 13:15:00 | 340.05 | 2024-07-05 09:15:00 | 333.70 | STOP_HIT | 1.00 | 1.87% |
| BUY | retest2 | 2024-07-16 09:15:00 | 353.25 | 2024-07-19 11:15:00 | 348.10 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-07-19 10:15:00 | 349.30 | 2024-07-19 11:15:00 | 348.10 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2024-07-22 12:15:00 | 347.10 | 2024-07-23 12:15:00 | 329.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 12:15:00 | 347.10 | 2024-07-23 14:15:00 | 346.10 | STOP_HIT | 0.50 | 0.29% |
| BUY | retest2 | 2024-08-02 10:15:00 | 392.00 | 2024-08-05 09:15:00 | 386.20 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-08-02 12:45:00 | 392.35 | 2024-08-05 09:15:00 | 386.20 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-08-07 09:15:00 | 395.00 | 2024-08-08 13:15:00 | 389.50 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-08-08 11:15:00 | 388.25 | 2024-08-08 13:15:00 | 389.50 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2024-08-08 11:45:00 | 390.00 | 2024-08-08 13:15:00 | 389.50 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2024-08-14 15:15:00 | 372.50 | 2024-08-16 09:15:00 | 377.10 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-08-28 09:15:00 | 407.90 | 2024-08-29 09:15:00 | 403.75 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-08-28 13:30:00 | 405.90 | 2024-08-29 09:15:00 | 403.75 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-08-28 14:45:00 | 407.85 | 2024-08-29 09:15:00 | 403.75 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-09-11 10:45:00 | 417.20 | 2024-09-19 10:15:00 | 396.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-11 10:45:00 | 417.20 | 2024-09-20 09:15:00 | 399.45 | STOP_HIT | 0.50 | 4.25% |
| BUY | retest2 | 2024-09-25 13:15:00 | 414.35 | 2024-10-03 11:15:00 | 419.75 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2024-09-26 12:45:00 | 414.45 | 2024-10-03 11:15:00 | 419.75 | STOP_HIT | 1.00 | 1.28% |
| SELL | retest2 | 2024-10-09 11:45:00 | 399.20 | 2024-10-14 11:15:00 | 397.10 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest2 | 2024-10-09 12:30:00 | 399.20 | 2024-10-14 11:15:00 | 397.10 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest2 | 2024-10-09 13:45:00 | 399.35 | 2024-10-14 11:15:00 | 397.10 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2024-10-10 11:00:00 | 398.20 | 2024-10-14 11:15:00 | 397.10 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2024-10-18 12:00:00 | 433.95 | 2024-10-21 12:15:00 | 424.65 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-10-21 09:15:00 | 433.65 | 2024-10-21 12:15:00 | 424.65 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-10-25 10:15:00 | 387.00 | 2024-11-04 09:15:00 | 367.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-28 09:45:00 | 384.30 | 2024-11-04 09:15:00 | 368.41 | PARTIAL | 0.50 | 4.13% |
| SELL | retest2 | 2024-10-28 12:30:00 | 387.80 | 2024-11-04 09:15:00 | 368.12 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2024-10-29 15:15:00 | 387.50 | 2024-11-04 09:15:00 | 367.13 | PARTIAL | 0.50 | 5.26% |
| SELL | retest2 | 2024-10-30 09:15:00 | 383.50 | 2024-11-04 10:15:00 | 365.08 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2024-10-30 11:00:00 | 386.45 | 2024-11-04 10:15:00 | 364.32 | PARTIAL | 0.50 | 5.73% |
| SELL | retest2 | 2024-10-25 10:15:00 | 387.00 | 2024-11-05 13:15:00 | 373.50 | STOP_HIT | 0.50 | 3.49% |
| SELL | retest2 | 2024-10-28 09:45:00 | 384.30 | 2024-11-05 13:15:00 | 373.50 | STOP_HIT | 0.50 | 2.81% |
| SELL | retest2 | 2024-10-28 12:30:00 | 387.80 | 2024-11-05 13:15:00 | 373.50 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2024-10-29 15:15:00 | 387.50 | 2024-11-05 13:15:00 | 373.50 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2024-10-30 09:15:00 | 383.50 | 2024-11-05 13:15:00 | 373.50 | STOP_HIT | 0.50 | 2.61% |
| SELL | retest2 | 2024-10-30 11:00:00 | 386.45 | 2024-11-05 13:15:00 | 373.50 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2024-11-14 13:45:00 | 373.35 | 2024-11-25 09:15:00 | 379.40 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-11-14 14:15:00 | 371.30 | 2024-11-25 09:15:00 | 379.40 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2024-11-18 09:30:00 | 369.55 | 2024-11-25 09:15:00 | 379.40 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2024-11-18 12:30:00 | 372.80 | 2024-11-25 09:15:00 | 379.40 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2024-11-19 12:15:00 | 369.35 | 2024-11-25 09:15:00 | 379.40 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2024-12-03 10:00:00 | 385.75 | 2024-12-16 12:15:00 | 402.35 | STOP_HIT | 1.00 | 4.30% |
| BUY | retest2 | 2024-12-03 10:45:00 | 385.85 | 2024-12-16 12:15:00 | 402.35 | STOP_HIT | 1.00 | 4.28% |
| BUY | retest2 | 2024-12-03 11:30:00 | 386.00 | 2024-12-16 12:15:00 | 402.35 | STOP_HIT | 1.00 | 4.24% |
| BUY | retest2 | 2024-12-03 13:15:00 | 386.35 | 2024-12-16 12:15:00 | 402.35 | STOP_HIT | 1.00 | 4.14% |
| BUY | retest2 | 2024-12-05 13:45:00 | 392.10 | 2024-12-16 12:15:00 | 402.35 | STOP_HIT | 1.00 | 2.61% |
| BUY | retest2 | 2024-12-05 14:15:00 | 392.30 | 2024-12-16 12:15:00 | 402.35 | STOP_HIT | 1.00 | 2.56% |
| BUY | retest2 | 2024-12-05 15:15:00 | 391.85 | 2024-12-16 12:15:00 | 402.35 | STOP_HIT | 1.00 | 2.68% |
| SELL | retest2 | 2025-01-01 09:45:00 | 406.20 | 2025-01-01 13:15:00 | 413.75 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-01-09 09:15:00 | 386.95 | 2025-01-13 09:15:00 | 367.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 10:30:00 | 389.85 | 2025-01-13 09:15:00 | 370.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 12:15:00 | 390.05 | 2025-01-13 09:15:00 | 370.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 13:15:00 | 389.85 | 2025-01-13 09:15:00 | 370.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 09:15:00 | 386.95 | 2025-01-14 10:15:00 | 371.75 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2025-01-09 10:30:00 | 389.85 | 2025-01-14 10:15:00 | 371.75 | STOP_HIT | 0.50 | 4.64% |
| SELL | retest2 | 2025-01-10 12:15:00 | 390.05 | 2025-01-14 10:15:00 | 371.75 | STOP_HIT | 0.50 | 4.69% |
| SELL | retest2 | 2025-01-10 13:15:00 | 389.85 | 2025-01-14 10:15:00 | 371.75 | STOP_HIT | 0.50 | 4.64% |
| SELL | retest2 | 2025-01-15 11:15:00 | 371.00 | 2025-01-21 10:15:00 | 369.55 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2025-02-07 10:45:00 | 344.65 | 2025-02-10 09:15:00 | 337.80 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-02-07 12:30:00 | 345.50 | 2025-02-10 09:15:00 | 337.80 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-03-18 10:00:00 | 323.90 | 2025-03-19 11:15:00 | 327.45 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-03-18 10:45:00 | 323.90 | 2025-03-19 11:15:00 | 327.45 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-03-18 11:15:00 | 323.80 | 2025-03-19 11:15:00 | 327.45 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-03-27 11:30:00 | 356.85 | 2025-04-02 09:15:00 | 354.40 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-04-01 11:15:00 | 357.25 | 2025-04-02 09:15:00 | 354.40 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-04-01 12:00:00 | 358.15 | 2025-04-02 09:15:00 | 354.40 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-04-03 13:15:00 | 363.10 | 2025-04-04 09:15:00 | 357.70 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-04-03 15:15:00 | 363.00 | 2025-04-04 09:15:00 | 357.70 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-04-07 09:15:00 | 356.00 | 2025-04-07 10:15:00 | 361.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-04-07 12:00:00 | 357.70 | 2025-04-07 12:15:00 | 360.35 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-04-16 11:15:00 | 384.15 | 2025-04-23 12:15:00 | 387.25 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2025-04-30 09:15:00 | 398.30 | 2025-04-30 10:15:00 | 389.90 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-04-30 11:45:00 | 392.05 | 2025-04-30 13:15:00 | 381.35 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-05-02 11:15:00 | 385.10 | 2025-05-05 09:15:00 | 405.00 | STOP_HIT | 1.00 | -5.17% |
| BUY | retest2 | 2025-05-07 11:15:00 | 403.85 | 2025-05-07 14:15:00 | 397.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-05-12 11:45:00 | 391.75 | 2025-05-14 09:15:00 | 394.30 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-05-29 15:00:00 | 414.90 | 2025-05-30 12:15:00 | 413.30 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-05-30 10:45:00 | 415.20 | 2025-05-30 12:15:00 | 413.30 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-06-05 13:00:00 | 403.80 | 2025-06-06 15:15:00 | 407.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-06-19 10:30:00 | 390.55 | 2025-06-23 14:15:00 | 393.60 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-06-19 13:15:00 | 389.85 | 2025-06-24 09:15:00 | 408.10 | STOP_HIT | 1.00 | -4.68% |
| SELL | retest2 | 2025-06-19 15:15:00 | 390.00 | 2025-06-24 09:15:00 | 408.10 | STOP_HIT | 1.00 | -4.64% |
| SELL | retest2 | 2025-06-20 12:15:00 | 390.10 | 2025-06-24 09:15:00 | 408.10 | STOP_HIT | 1.00 | -4.61% |
| SELL | retest2 | 2025-06-23 09:30:00 | 389.65 | 2025-06-24 09:15:00 | 408.10 | STOP_HIT | 1.00 | -4.74% |
| BUY | retest2 | 2025-07-04 11:30:00 | 441.65 | 2025-07-10 10:15:00 | 443.50 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2025-07-04 14:45:00 | 442.00 | 2025-07-10 10:15:00 | 443.50 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-07-15 10:45:00 | 433.00 | 2025-07-16 10:15:00 | 443.15 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-07-15 11:15:00 | 433.30 | 2025-07-16 10:15:00 | 443.15 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-07-22 12:00:00 | 430.20 | 2025-07-24 09:15:00 | 433.95 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-07-23 12:15:00 | 430.25 | 2025-07-24 09:15:00 | 433.95 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-07-23 13:00:00 | 430.70 | 2025-07-24 09:15:00 | 433.95 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-07-30 09:15:00 | 418.65 | 2025-07-30 11:15:00 | 432.00 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2025-08-13 09:15:00 | 408.75 | 2025-08-14 09:15:00 | 398.00 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2025-08-13 11:15:00 | 408.70 | 2025-08-14 09:15:00 | 398.00 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-08-13 12:15:00 | 408.80 | 2025-08-14 09:15:00 | 398.00 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2025-08-13 12:45:00 | 410.05 | 2025-08-14 09:15:00 | 398.00 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2025-08-20 11:15:00 | 391.95 | 2025-08-20 12:15:00 | 395.20 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-08-20 13:45:00 | 392.35 | 2025-08-21 09:15:00 | 395.10 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-08-20 14:30:00 | 392.40 | 2025-08-21 09:15:00 | 395.10 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-08-22 09:15:00 | 389.65 | 2025-09-01 15:15:00 | 384.80 | STOP_HIT | 1.00 | 1.24% |
| SELL | retest2 | 2025-08-26 09:30:00 | 386.15 | 2025-09-01 15:15:00 | 384.80 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2025-09-11 09:15:00 | 397.25 | 2025-09-29 10:15:00 | 436.98 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-14 10:15:00 | 446.15 | 2025-10-16 10:15:00 | 454.40 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-10-14 11:00:00 | 446.00 | 2025-10-16 10:15:00 | 454.40 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-10-27 12:00:00 | 447.60 | 2025-10-27 12:15:00 | 452.40 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-11-13 09:15:00 | 487.85 | 2025-11-14 09:15:00 | 483.85 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-11-13 10:15:00 | 485.60 | 2025-11-14 10:15:00 | 479.90 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-11-13 10:45:00 | 485.75 | 2025-11-14 10:15:00 | 479.90 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-11-13 14:15:00 | 485.70 | 2025-11-14 10:15:00 | 479.90 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-11-14 09:15:00 | 486.70 | 2025-11-14 10:15:00 | 479.90 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-11-21 09:15:00 | 475.00 | 2025-12-01 14:15:00 | 451.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 09:15:00 | 475.00 | 2025-12-03 12:15:00 | 451.50 | STOP_HIT | 0.50 | 4.95% |
| BUY | retest2 | 2025-12-16 14:30:00 | 464.00 | 2025-12-18 09:15:00 | 459.20 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-12-18 13:00:00 | 464.10 | 2025-12-24 15:15:00 | 469.30 | STOP_HIT | 1.00 | 1.12% |
| BUY | retest2 | 2025-12-18 15:00:00 | 463.95 | 2025-12-24 15:15:00 | 469.30 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2025-12-19 10:15:00 | 465.75 | 2025-12-24 15:15:00 | 469.30 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2025-12-19 15:00:00 | 469.75 | 2025-12-24 15:15:00 | 469.30 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2026-02-11 09:15:00 | 465.65 | 2026-02-12 09:15:00 | 458.50 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-02-11 10:15:00 | 467.00 | 2026-02-12 09:15:00 | 458.50 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-02-17 09:15:00 | 447.00 | 2026-02-17 14:15:00 | 454.30 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2026-02-23 12:15:00 | 438.10 | 2026-02-24 10:15:00 | 441.90 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-02-23 12:45:00 | 437.60 | 2026-02-24 10:15:00 | 441.90 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-02-23 14:30:00 | 437.15 | 2026-02-24 10:15:00 | 441.90 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest1 | 2026-03-05 10:15:00 | 402.25 | 2026-03-05 14:15:00 | 416.65 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest1 | 2026-03-05 11:15:00 | 401.60 | 2026-03-05 14:15:00 | 416.65 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2026-03-06 12:15:00 | 409.30 | 2026-03-09 09:15:00 | 388.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:45:00 | 405.85 | 2026-03-09 09:15:00 | 385.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 12:15:00 | 409.30 | 2026-03-10 13:15:00 | 385.80 | STOP_HIT | 0.50 | 5.74% |
| SELL | retest2 | 2026-03-06 14:45:00 | 405.85 | 2026-03-10 13:15:00 | 385.80 | STOP_HIT | 0.50 | 4.94% |
| SELL | retest2 | 2026-04-07 09:15:00 | 324.20 | 2026-04-07 14:15:00 | 332.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2026-04-13 10:15:00 | 349.80 | 2026-04-13 12:15:00 | 347.35 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2026-04-22 11:15:00 | 384.10 | 2026-04-23 09:15:00 | 374.45 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2026-05-04 13:15:00 | 372.90 | 2026-05-06 10:15:00 | 381.15 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2026-05-04 15:00:00 | 373.25 | 2026-05-06 10:15:00 | 381.15 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2026-05-05 09:15:00 | 369.15 | 2026-05-06 10:15:00 | 381.15 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2026-05-05 13:15:00 | 373.35 | 2026-05-06 10:15:00 | 381.15 | STOP_HIT | 1.00 | -2.09% |
