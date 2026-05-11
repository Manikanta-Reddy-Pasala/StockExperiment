# Authum Investment & Infrastructure Ltd. (AIIL)

## Backtest Summary

- **Window:** 2024-04-23 09:15:00 → 2026-05-08 15:15:00 (3535 bars)
- **Last close:** 494.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 147 |
| ALERT1 | 94 |
| ALERT2 | 93 |
| ALERT2_SKIP | 48 |
| ALERT3 | 230 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 124 |
| PARTIAL | 20 |
| TARGET_HIT | 22 |
| STOP_HIT | 108 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 150 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 60 / 90
- **Target hits / Stop hits / Partials:** 22 / 108 / 20
- **Avg / median % per leg:** 0.86% / -1.00%
- **Sum % (uncompounded):** 129.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 68 | 22 | 32.4% | 12 | 54 | 2 | 0.78% | 53.1% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 0 | 5 | 2 | 1.04% | 7.3% |
| BUY @ 3rd Alert (retest2) | 61 | 18 | 29.5% | 12 | 49 | 0 | 0.75% | 45.9% |
| SELL (all) | 82 | 38 | 46.3% | 10 | 54 | 18 | 0.93% | 76.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.23% | -0.2% |
| SELL @ 3rd Alert (retest2) | 81 | 38 | 46.9% | 10 | 53 | 18 | 0.95% | 76.7% |
| retest1 (combined) | 8 | 4 | 50.0% | 0 | 6 | 2 | 0.88% | 7.0% |
| retest2 (combined) | 142 | 56 | 39.4% | 22 | 102 | 18 | 0.86% | 122.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 165.01 | 155.72 | 154.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 10:15:00 | 167.25 | 163.30 | 159.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 12:15:00 | 163.79 | 164.09 | 160.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 13:00:00 | 163.79 | 164.09 | 160.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 13:15:00 | 164.69 | 164.21 | 161.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 167.41 | 164.45 | 161.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 15:15:00 | 170.00 | 171.24 | 171.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 15:15:00 | 170.00 | 171.24 | 171.39 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 10:15:00 | 175.65 | 170.32 | 170.16 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 14:15:00 | 167.37 | 170.39 | 170.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 09:15:00 | 163.97 | 168.81 | 169.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 165.11 | 165.08 | 167.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 165.11 | 165.08 | 167.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 165.11 | 165.08 | 167.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 14:45:00 | 161.80 | 163.14 | 164.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 10:15:00 | 172.32 | 165.71 | 165.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 10:15:00 | 172.32 | 165.71 | 165.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 14:15:00 | 185.99 | 171.57 | 168.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 170.36 | 174.08 | 170.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 170.36 | 174.08 | 170.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 170.36 | 174.08 | 170.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 170.36 | 174.08 | 170.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 164.37 | 172.14 | 170.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 164.37 | 172.14 | 170.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 164.42 | 170.60 | 169.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 13:30:00 | 167.45 | 170.19 | 169.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 15:00:00 | 166.36 | 169.43 | 169.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 15:15:00 | 164.28 | 168.40 | 168.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 15:15:00 | 164.28 | 168.40 | 168.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 162.54 | 167.23 | 168.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 11:15:00 | 171.43 | 167.90 | 168.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 11:15:00 | 171.43 | 167.90 | 168.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 171.43 | 167.90 | 168.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:30:00 | 169.49 | 167.90 | 168.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 171.74 | 168.67 | 168.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:45:00 | 172.24 | 168.67 | 168.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 13:15:00 | 172.03 | 169.34 | 169.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 187.60 | 173.65 | 171.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 11:15:00 | 204.59 | 205.37 | 197.01 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 15:00:00 | 208.68 | 205.77 | 199.25 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 09:15:00 | 219.11 | 209.99 | 202.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-06-11 14:15:00 | 210.43 | 213.00 | 207.14 | SL hit (close<ema200) qty=0.50 sl=213.00 alert=retest1 |

### Cycle 8 — SELL (started 2024-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 14:15:00 | 216.24 | 219.45 | 219.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 15:15:00 | 215.00 | 218.56 | 219.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 213.86 | 210.85 | 214.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 213.86 | 210.85 | 214.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 213.86 | 210.85 | 214.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:30:00 | 214.90 | 210.85 | 214.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 216.69 | 212.02 | 214.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:00:00 | 216.69 | 212.02 | 214.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 219.18 | 213.45 | 214.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:00:00 | 219.18 | 213.45 | 214.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 13:15:00 | 221.18 | 215.89 | 215.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 09:15:00 | 225.77 | 218.52 | 216.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 11:15:00 | 225.95 | 227.95 | 225.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-25 12:00:00 | 225.95 | 227.95 | 225.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 225.37 | 227.44 | 225.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:45:00 | 224.13 | 227.44 | 225.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 229.50 | 227.85 | 225.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 14:15:00 | 229.66 | 227.85 | 225.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 15:00:00 | 229.96 | 228.27 | 226.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 09:15:00 | 232.73 | 230.19 | 228.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 13:15:00 | 222.36 | 227.46 | 227.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 222.36 | 227.46 | 227.91 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 10:15:00 | 229.99 | 228.37 | 228.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 14:15:00 | 231.99 | 229.44 | 228.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 09:15:00 | 225.98 | 228.97 | 228.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 225.98 | 228.97 | 228.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 225.98 | 228.97 | 228.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:45:00 | 227.32 | 228.97 | 228.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 10:15:00 | 226.44 | 228.46 | 228.49 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 229.78 | 228.72 | 228.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 230.22 | 229.05 | 228.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 229.35 | 229.46 | 229.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 11:15:00 | 229.35 | 229.46 | 229.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 229.35 | 229.46 | 229.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 229.35 | 229.46 | 229.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 226.49 | 228.86 | 228.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 226.49 | 228.86 | 228.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 232.55 | 229.60 | 229.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:45:00 | 227.39 | 229.60 | 229.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 228.00 | 229.60 | 229.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:00:00 | 228.00 | 229.60 | 229.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 227.30 | 229.14 | 229.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 11:00:00 | 227.30 | 229.14 | 229.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 11:15:00 | 227.98 | 228.91 | 229.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 09:15:00 | 226.51 | 228.01 | 228.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 11:15:00 | 228.48 | 228.00 | 228.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 11:15:00 | 228.48 | 228.00 | 228.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 228.48 | 228.00 | 228.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 09:45:00 | 226.82 | 227.96 | 228.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 11:00:00 | 226.00 | 227.57 | 228.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 14:00:00 | 226.75 | 227.61 | 227.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 10:15:00 | 230.95 | 227.27 | 227.59 | SL hit (close>static) qty=1.00 sl=229.52 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 11:15:00 | 230.87 | 227.99 | 227.89 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 225.85 | 228.89 | 228.95 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 11:15:00 | 230.55 | 229.14 | 229.05 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 13:15:00 | 228.47 | 229.06 | 229.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 14:15:00 | 227.34 | 228.71 | 228.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 13:15:00 | 212.33 | 212.27 | 215.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-18 13:45:00 | 213.37 | 212.27 | 215.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 204.50 | 210.34 | 213.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 10:45:00 | 203.52 | 208.98 | 213.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 11:30:00 | 203.92 | 208.04 | 212.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 12:15:00 | 203.35 | 208.04 | 212.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 13:00:00 | 203.39 | 207.11 | 211.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 200.00 | 202.22 | 205.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:30:00 | 199.13 | 201.93 | 205.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:15:00 | 198.71 | 201.71 | 204.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 193.34 | 200.57 | 203.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 193.72 | 200.57 | 203.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 193.18 | 200.57 | 203.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 193.22 | 200.57 | 203.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 14:15:00 | 198.36 | 200.46 | 203.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 14:45:00 | 198.74 | 199.97 | 202.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 15:15:00 | 201.00 | 200.18 | 202.74 | SL hit (close>ema200) qty=0.50 sl=200.18 alert=retest2 |

### Cycle 19 — BUY (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 09:15:00 | 205.28 | 203.06 | 202.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 11:15:00 | 208.73 | 205.83 | 204.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 12:15:00 | 208.01 | 209.29 | 207.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 13:00:00 | 208.01 | 209.29 | 207.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 13:15:00 | 207.92 | 209.02 | 207.55 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2024-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 11:15:00 | 206.04 | 206.88 | 206.90 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 09:15:00 | 210.60 | 206.97 | 206.84 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 202.45 | 208.96 | 209.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 11:15:00 | 202.19 | 207.61 | 208.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 203.60 | 202.55 | 205.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 10:00:00 | 203.60 | 202.55 | 205.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 15:15:00 | 203.80 | 201.84 | 203.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 09:15:00 | 203.06 | 201.84 | 203.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 208.00 | 203.07 | 204.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 09:45:00 | 207.69 | 203.07 | 204.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 210.40 | 204.53 | 204.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:00:00 | 210.40 | 204.53 | 204.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 210.04 | 205.64 | 205.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 13:15:00 | 215.62 | 208.73 | 206.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 10:15:00 | 282.05 | 282.57 | 269.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 10:45:00 | 282.78 | 282.57 | 269.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 275.57 | 280.01 | 273.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 271.04 | 280.01 | 273.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 274.20 | 278.85 | 273.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:00:00 | 274.20 | 278.85 | 273.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 273.60 | 277.80 | 273.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:45:00 | 274.22 | 277.80 | 273.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 274.81 | 277.20 | 273.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 12:30:00 | 273.21 | 277.20 | 273.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 13:15:00 | 276.88 | 277.14 | 274.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 14:15:00 | 279.25 | 277.14 | 274.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 14:45:00 | 277.18 | 277.23 | 274.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 09:15:00 | 278.00 | 276.58 | 274.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 15:15:00 | 277.79 | 278.43 | 276.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 280.94 | 278.83 | 277.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:45:00 | 277.56 | 278.83 | 277.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 11:15:00 | 277.35 | 279.14 | 277.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 12:00:00 | 277.35 | 279.14 | 277.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 12:15:00 | 278.70 | 279.05 | 277.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 12:30:00 | 277.80 | 279.05 | 277.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 278.79 | 279.14 | 277.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 15:00:00 | 278.79 | 279.14 | 277.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 15:15:00 | 276.22 | 278.56 | 277.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 09:15:00 | 282.40 | 278.56 | 277.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 10:15:00 | 279.05 | 278.60 | 277.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-20 14:15:00 | 304.90 | 284.34 | 281.06 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 15:15:00 | 340.08 | 344.42 | 344.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 327.41 | 341.02 | 342.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 338.98 | 336.86 | 339.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 15:00:00 | 338.98 | 336.86 | 339.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 333.20 | 335.89 | 338.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 09:45:00 | 326.20 | 331.20 | 332.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 316.40 | 329.64 | 331.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 09:15:00 | 336.94 | 324.00 | 323.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 336.94 | 324.00 | 323.07 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 10:15:00 | 323.08 | 327.42 | 327.77 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 09:15:00 | 337.31 | 328.74 | 327.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 09:15:00 | 353.72 | 339.74 | 335.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 13:15:00 | 341.74 | 344.45 | 339.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 14:00:00 | 341.74 | 344.45 | 339.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 340.90 | 342.99 | 340.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 12:00:00 | 340.90 | 342.99 | 340.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 341.92 | 342.78 | 340.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 09:15:00 | 343.16 | 341.54 | 340.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 09:15:00 | 338.92 | 341.02 | 340.47 | SL hit (close<static) qty=1.00 sl=340.51 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 12:15:00 | 339.15 | 340.13 | 340.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 14:15:00 | 338.23 | 339.73 | 339.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 10:15:00 | 341.50 | 339.75 | 339.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 10:15:00 | 341.50 | 339.75 | 339.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 341.50 | 339.75 | 339.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:30:00 | 341.53 | 339.75 | 339.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 340.33 | 339.86 | 339.93 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2024-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 15:15:00 | 341.60 | 340.14 | 340.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 09:15:00 | 344.40 | 340.99 | 340.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 10:15:00 | 340.40 | 340.87 | 340.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 10:15:00 | 340.40 | 340.87 | 340.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 340.40 | 340.87 | 340.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:45:00 | 340.40 | 340.87 | 340.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 343.81 | 341.46 | 340.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:30:00 | 341.00 | 341.46 | 340.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 340.65 | 342.30 | 341.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 340.65 | 342.30 | 341.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 340.00 | 341.84 | 341.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 347.20 | 341.84 | 341.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 10:15:00 | 343.20 | 354.53 | 355.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 343.20 | 354.53 | 355.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-08 09:15:00 | 338.71 | 345.08 | 349.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 14:15:00 | 341.69 | 341.56 | 346.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 15:00:00 | 341.69 | 341.56 | 346.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 352.25 | 343.32 | 346.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:45:00 | 352.46 | 343.32 | 346.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 352.44 | 345.14 | 346.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:45:00 | 352.00 | 345.14 | 346.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 359.00 | 347.91 | 347.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 12:15:00 | 359.02 | 350.14 | 348.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 09:15:00 | 361.95 | 363.37 | 359.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 09:30:00 | 360.74 | 363.37 | 359.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 358.98 | 362.49 | 359.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:30:00 | 359.12 | 362.49 | 359.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 359.61 | 361.92 | 359.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:45:00 | 359.62 | 361.92 | 359.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 359.78 | 361.49 | 359.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 12:30:00 | 358.93 | 361.49 | 359.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 362.32 | 361.65 | 359.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 368.42 | 361.24 | 359.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 12:15:00 | 358.41 | 363.86 | 361.80 | SL hit (close<static) qty=1.00 sl=358.44 alert=retest2 |

### Cycle 32 — SELL (started 2024-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 15:15:00 | 356.40 | 360.61 | 360.69 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 09:15:00 | 365.30 | 361.55 | 361.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 11:15:00 | 369.74 | 367.08 | 364.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 364.75 | 367.61 | 365.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 364.75 | 367.61 | 365.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 364.75 | 367.61 | 365.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 364.75 | 367.61 | 365.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 367.37 | 367.56 | 366.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 12:30:00 | 367.99 | 367.46 | 366.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 12:00:00 | 368.64 | 367.19 | 366.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 13:15:00 | 368.00 | 367.15 | 366.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 15:15:00 | 365.79 | 366.44 | 366.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 15:15:00 | 365.79 | 366.44 | 366.48 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 09:15:00 | 374.11 | 367.97 | 367.17 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 09:15:00 | 331.33 | 365.28 | 367.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 12:15:00 | 330.60 | 349.57 | 359.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 15:15:00 | 334.00 | 333.47 | 342.35 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:30:00 | 325.60 | 330.53 | 335.95 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 320.00 | 322.45 | 328.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 09:15:00 | 318.84 | 322.45 | 328.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 12:00:00 | 319.29 | 320.74 | 326.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 15:15:00 | 326.34 | 322.69 | 325.55 | SL hit (close>ema400) qty=1.00 sl=325.55 alert=retest1 |

### Cycle 37 — BUY (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 09:15:00 | 329.79 | 323.69 | 323.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 335.91 | 328.82 | 326.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 329.97 | 330.04 | 327.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 329.97 | 330.04 | 327.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 329.97 | 330.04 | 327.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:45:00 | 329.80 | 330.04 | 327.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 338.00 | 335.97 | 332.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 10:45:00 | 339.17 | 336.44 | 332.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 339.00 | 335.82 | 333.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 11:30:00 | 343.01 | 338.00 | 335.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 09:15:00 | 334.64 | 338.74 | 338.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 334.64 | 338.74 | 338.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 12:15:00 | 326.07 | 329.11 | 331.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 319.00 | 317.02 | 321.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 11:15:00 | 317.80 | 316.16 | 318.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 317.80 | 316.16 | 318.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:00:00 | 317.80 | 316.16 | 318.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 316.86 | 316.30 | 318.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:30:00 | 318.52 | 316.30 | 318.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 315.81 | 316.20 | 318.06 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 323.73 | 319.42 | 319.12 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 316.64 | 319.07 | 319.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-22 13:15:00 | 310.10 | 313.24 | 315.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 312.43 | 312.31 | 314.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 312.43 | 312.31 | 314.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 312.43 | 312.31 | 314.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 14:30:00 | 311.34 | 312.07 | 313.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 15:00:00 | 310.41 | 312.07 | 313.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-27 09:15:00 | 316.60 | 314.42 | 314.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 09:15:00 | 316.60 | 314.42 | 314.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 322.98 | 316.93 | 315.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 14:15:00 | 318.15 | 318.47 | 316.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 15:00:00 | 318.15 | 318.47 | 316.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 316.39 | 318.08 | 317.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:30:00 | 316.52 | 318.08 | 317.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 317.04 | 317.87 | 317.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 10:00:00 | 318.19 | 317.04 | 316.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 11:15:00 | 318.44 | 317.20 | 316.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 12:15:00 | 323.04 | 323.69 | 323.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 12:15:00 | 323.04 | 323.69 | 323.75 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 09:15:00 | 328.24 | 324.10 | 323.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 10:15:00 | 335.39 | 326.36 | 324.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 371.35 | 373.54 | 366.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-13 09:30:00 | 370.00 | 373.54 | 366.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 367.68 | 372.37 | 366.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:30:00 | 365.58 | 372.37 | 366.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 366.20 | 370.40 | 366.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:30:00 | 366.87 | 370.40 | 366.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 367.60 | 369.84 | 366.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 09:45:00 | 373.59 | 369.43 | 367.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 11:15:00 | 365.63 | 368.50 | 367.06 | SL hit (close<static) qty=1.00 sl=365.67 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 363.00 | 366.14 | 366.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 359.35 | 364.32 | 365.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 14:15:00 | 351.37 | 350.97 | 354.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 15:00:00 | 351.37 | 350.97 | 354.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 332.80 | 335.63 | 338.98 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 14:15:00 | 341.75 | 339.30 | 339.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 09:15:00 | 345.88 | 340.95 | 339.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 13:15:00 | 340.62 | 341.90 | 340.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 13:15:00 | 340.62 | 341.90 | 340.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 340.62 | 341.90 | 340.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:00:00 | 340.62 | 341.90 | 340.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 341.33 | 341.78 | 340.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:15:00 | 340.42 | 341.78 | 340.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 340.42 | 341.51 | 340.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:15:00 | 338.80 | 341.51 | 340.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 338.56 | 340.92 | 340.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:45:00 | 337.81 | 340.92 | 340.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 340.72 | 340.88 | 340.58 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2024-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 12:15:00 | 339.09 | 340.29 | 340.35 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 14:15:00 | 341.45 | 340.57 | 340.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 15:15:00 | 342.00 | 340.85 | 340.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 13:15:00 | 374.20 | 374.20 | 365.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 13:30:00 | 373.70 | 374.20 | 365.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 380.59 | 386.66 | 380.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 380.59 | 386.66 | 380.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 378.91 | 385.11 | 380.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 376.94 | 385.11 | 380.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 379.51 | 383.99 | 380.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 12:30:00 | 379.45 | 383.99 | 380.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 373.67 | 381.93 | 379.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:00:00 | 373.67 | 381.93 | 379.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 375.40 | 379.37 | 378.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 09:15:00 | 393.00 | 379.37 | 378.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 12:15:00 | 378.07 | 380.78 | 380.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 12:15:00 | 378.07 | 380.78 | 380.87 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 11:15:00 | 381.89 | 380.94 | 380.82 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 12:15:00 | 378.14 | 380.38 | 380.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 13:15:00 | 373.00 | 378.90 | 379.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 10:15:00 | 374.70 | 373.50 | 376.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-10 11:00:00 | 374.70 | 373.50 | 376.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 11:15:00 | 375.87 | 373.97 | 376.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 13:45:00 | 372.55 | 373.33 | 375.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 11:00:00 | 365.66 | 371.25 | 374.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 12:15:00 | 353.92 | 365.86 | 370.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 14:15:00 | 347.38 | 359.55 | 367.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-14 13:15:00 | 335.30 | 350.28 | 358.77 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 12:15:00 | 366.31 | 357.92 | 356.98 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 11:15:00 | 349.60 | 357.92 | 358.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 13:15:00 | 346.92 | 354.44 | 356.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 09:15:00 | 359.90 | 354.56 | 355.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 359.90 | 354.56 | 355.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 359.90 | 354.56 | 355.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:00:00 | 359.90 | 354.56 | 355.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 369.40 | 357.53 | 357.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 11:15:00 | 376.81 | 361.39 | 358.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 13:15:00 | 369.69 | 371.67 | 367.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 13:15:00 | 369.69 | 371.67 | 367.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 369.69 | 371.67 | 367.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:00:00 | 369.69 | 371.67 | 367.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 366.59 | 370.65 | 367.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:45:00 | 367.50 | 370.65 | 367.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 366.60 | 369.84 | 367.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 354.14 | 369.84 | 367.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 353.48 | 363.37 | 364.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 345.25 | 359.75 | 362.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 356.91 | 354.23 | 359.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:15:00 | 355.46 | 354.23 | 359.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 355.46 | 354.48 | 358.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:30:00 | 365.03 | 357.53 | 359.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 370.10 | 360.05 | 360.74 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 11:15:00 | 368.00 | 361.64 | 361.40 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 09:15:00 | 355.80 | 361.36 | 361.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 340.39 | 352.98 | 356.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 338.40 | 332.00 | 338.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 338.40 | 332.00 | 338.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 338.40 | 332.00 | 338.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 338.40 | 332.00 | 338.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 336.36 | 332.88 | 338.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 336.36 | 332.88 | 338.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 337.69 | 333.84 | 338.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:30:00 | 337.38 | 333.84 | 338.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 336.93 | 334.46 | 338.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:15:00 | 334.15 | 338.63 | 339.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 14:00:00 | 334.90 | 333.68 | 335.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 15:15:00 | 339.40 | 335.52 | 336.08 | SL hit (close>static) qty=1.00 sl=338.56 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 10:15:00 | 340.71 | 337.27 | 336.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 14:15:00 | 343.00 | 338.51 | 337.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 337.50 | 338.87 | 337.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 337.50 | 338.87 | 337.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 337.50 | 338.87 | 337.92 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 12:15:00 | 335.41 | 337.38 | 337.41 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 14:15:00 | 340.00 | 337.44 | 337.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 09:15:00 | 346.16 | 339.59 | 338.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 11:15:00 | 357.91 | 358.51 | 354.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-07 12:00:00 | 357.91 | 358.51 | 354.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 353.31 | 357.23 | 354.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 13:45:00 | 354.13 | 357.23 | 354.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 360.00 | 357.78 | 355.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 367.19 | 358.09 | 355.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 10:15:00 | 361.65 | 358.63 | 355.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 15:15:00 | 361.98 | 358.82 | 357.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 348.89 | 357.34 | 356.70 | SL hit (close<static) qty=1.00 sl=352.86 alert=retest2 |

### Cycle 60 — SELL (started 2025-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 10:15:00 | 347.85 | 355.44 | 355.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 11:15:00 | 346.90 | 353.73 | 355.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 13:15:00 | 354.80 | 353.23 | 354.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-11 13:15:00 | 354.80 | 353.23 | 354.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 13:15:00 | 354.80 | 353.23 | 354.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 14:00:00 | 354.80 | 353.23 | 354.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 14:15:00 | 353.00 | 353.18 | 354.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 14:45:00 | 353.86 | 353.18 | 354.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 14:15:00 | 339.95 | 334.92 | 339.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 15:00:00 | 339.95 | 334.92 | 339.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 15:15:00 | 340.00 | 335.93 | 339.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 330.15 | 335.93 | 339.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:15:00 | 334.33 | 336.15 | 339.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 317.61 | 329.24 | 333.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 12:15:00 | 313.64 | 322.75 | 329.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-18 11:15:00 | 300.90 | 312.77 | 321.21 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 61 — BUY (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 11:15:00 | 313.03 | 305.89 | 305.55 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 301.84 | 305.57 | 305.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 294.50 | 300.60 | 303.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 15:15:00 | 306.75 | 301.77 | 303.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 15:15:00 | 306.75 | 301.77 | 303.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 306.75 | 301.77 | 303.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 298.20 | 301.77 | 303.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 09:45:00 | 298.40 | 300.96 | 302.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 09:15:00 | 298.02 | 298.40 | 300.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-28 09:15:00 | 268.38 | 291.60 | 295.05 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-03-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 10:15:00 | 310.81 | 292.65 | 291.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-03 12:15:00 | 317.00 | 300.38 | 295.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 11:15:00 | 308.00 | 308.06 | 302.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-04 12:00:00 | 308.00 | 308.06 | 302.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 301.00 | 306.11 | 302.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 15:00:00 | 301.00 | 306.11 | 302.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 15:15:00 | 299.18 | 304.73 | 302.47 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 10:15:00 | 301.02 | 302.45 | 302.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 11:15:00 | 295.12 | 300.99 | 301.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 14:15:00 | 302.00 | 299.62 | 300.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 14:15:00 | 302.00 | 299.62 | 300.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 302.00 | 299.62 | 300.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:00:00 | 302.00 | 299.62 | 300.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 302.60 | 300.21 | 301.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 299.80 | 300.21 | 301.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 299.12 | 299.99 | 300.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 10:15:00 | 296.50 | 299.99 | 300.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 10:45:00 | 296.96 | 299.40 | 300.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 12:15:00 | 282.11 | 287.26 | 289.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 13:15:00 | 281.68 | 286.64 | 288.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 287.37 | 286.31 | 288.12 | SL hit (close>ema200) qty=0.50 sl=286.31 alert=retest2 |

### Cycle 65 — BUY (started 2025-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 13:15:00 | 292.52 | 289.41 | 289.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 297.29 | 290.99 | 289.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 11:15:00 | 315.26 | 316.42 | 310.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 11:45:00 | 315.13 | 316.42 | 310.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 312.92 | 315.67 | 311.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:45:00 | 311.25 | 315.67 | 311.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 310.94 | 314.72 | 311.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 310.94 | 314.72 | 311.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 310.00 | 313.78 | 310.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 314.60 | 313.78 | 310.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 316.00 | 315.21 | 314.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:45:00 | 314.95 | 314.54 | 314.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 12:15:00 | 311.80 | 313.76 | 313.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 311.80 | 313.76 | 313.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 310.00 | 313.01 | 313.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 315.99 | 312.03 | 312.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 09:15:00 | 315.99 | 312.03 | 312.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 315.99 | 312.03 | 312.88 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 12:15:00 | 314.28 | 313.50 | 313.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 13:15:00 | 321.74 | 315.15 | 314.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 09:15:00 | 348.62 | 351.77 | 343.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-02 09:45:00 | 347.86 | 351.77 | 343.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 344.91 | 353.47 | 351.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 344.91 | 353.47 | 351.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 344.51 | 351.68 | 350.77 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 343.05 | 349.95 | 350.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 341.28 | 348.22 | 349.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 320.71 | 315.35 | 326.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 10:00:00 | 320.71 | 315.35 | 326.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 324.41 | 319.36 | 325.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 14:00:00 | 324.41 | 319.36 | 325.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 326.24 | 320.74 | 325.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 14:30:00 | 326.80 | 320.74 | 325.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 15:15:00 | 330.80 | 322.75 | 325.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 317.10 | 322.75 | 325.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 332.68 | 323.35 | 322.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 332.68 | 323.35 | 322.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 12:15:00 | 333.84 | 328.12 | 324.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 15:15:00 | 370.80 | 372.22 | 364.04 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:15:00 | 377.02 | 372.22 | 364.04 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:45:00 | 374.32 | 371.03 | 364.24 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 366.58 | 369.53 | 366.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-24 09:15:00 | 366.58 | 369.53 | 366.66 | SL hit (close<ema400) qty=1.00 sl=366.66 alert=retest1 |

### Cycle 70 — SELL (started 2025-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 14:15:00 | 356.10 | 365.15 | 365.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 346.18 | 360.21 | 363.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 355.16 | 351.20 | 355.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 355.16 | 351.20 | 355.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 355.16 | 351.20 | 355.79 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 11:15:00 | 360.52 | 357.08 | 356.75 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 10:15:00 | 352.00 | 356.67 | 356.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 11:15:00 | 347.48 | 354.83 | 356.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 362.92 | 352.41 | 353.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 362.92 | 352.41 | 353.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 362.92 | 352.41 | 353.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:30:00 | 363.94 | 352.41 | 353.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 358.22 | 353.57 | 354.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 12:00:00 | 354.68 | 353.79 | 354.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 14:15:00 | 355.80 | 354.55 | 354.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 14:15:00 | 359.30 | 355.50 | 355.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 14:15:00 | 359.30 | 355.50 | 355.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 09:15:00 | 373.76 | 359.87 | 357.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 365.08 | 368.90 | 364.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 365.08 | 368.90 | 364.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 365.08 | 368.90 | 364.40 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2025-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 15:15:00 | 356.00 | 361.89 | 362.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 345.44 | 358.60 | 360.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 14:15:00 | 357.20 | 354.26 | 357.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 15:00:00 | 357.20 | 354.26 | 357.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 357.00 | 354.81 | 357.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 364.34 | 354.81 | 357.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 368.88 | 357.62 | 358.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:00:00 | 368.88 | 357.62 | 358.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 365.54 | 359.21 | 359.08 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 351.30 | 357.93 | 358.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 349.00 | 356.14 | 357.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 15:15:00 | 351.00 | 350.16 | 353.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-12 09:15:00 | 366.52 | 350.16 | 353.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 372.70 | 354.67 | 354.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 372.70 | 354.67 | 354.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 373.64 | 358.46 | 356.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 12:15:00 | 393.60 | 375.84 | 368.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 10:15:00 | 423.00 | 426.45 | 416.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 11:00:00 | 423.00 | 426.45 | 416.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 440.34 | 443.12 | 437.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 440.34 | 443.12 | 437.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 432.86 | 441.07 | 437.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 432.86 | 441.07 | 437.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 441.36 | 441.13 | 437.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 15:00:00 | 444.60 | 441.74 | 438.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 443.56 | 441.59 | 438.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 15:15:00 | 442.20 | 439.53 | 438.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 15:00:00 | 442.40 | 443.46 | 442.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 446.56 | 443.46 | 442.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 457.56 | 444.83 | 444.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-03 09:15:00 | 489.06 | 479.45 | 472.52 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 495.34 | 500.13 | 500.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 490.84 | 497.99 | 499.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 492.30 | 489.58 | 492.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 492.30 | 489.58 | 492.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 497.42 | 491.15 | 492.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 497.42 | 491.15 | 492.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 498.80 | 492.68 | 493.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 498.52 | 492.68 | 493.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 501.12 | 494.37 | 494.18 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 490.84 | 493.56 | 493.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 14:15:00 | 487.40 | 491.68 | 492.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 501.50 | 492.99 | 493.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 501.50 | 492.99 | 493.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 501.50 | 492.99 | 493.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:45:00 | 511.90 | 492.99 | 493.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 496.72 | 493.73 | 493.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 14:15:00 | 506.38 | 498.79 | 496.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 15:15:00 | 518.00 | 518.62 | 514.39 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:15:00 | 522.80 | 518.62 | 514.39 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 531.06 | 537.78 | 532.65 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-26 10:15:00 | 531.06 | 537.78 | 532.65 | SL hit (close<ema400) qty=1.00 sl=532.65 alert=retest1 |

### Cycle 82 — SELL (started 2025-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 14:15:00 | 525.88 | 529.59 | 529.83 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 530.76 | 530.08 | 530.03 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 14:15:00 | 524.98 | 529.89 | 530.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 10:15:00 | 516.88 | 525.33 | 527.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 14:15:00 | 525.78 | 522.33 | 525.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 14:15:00 | 525.78 | 522.33 | 525.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 525.78 | 522.33 | 525.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 525.78 | 522.33 | 525.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 521.10 | 522.08 | 524.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 516.96 | 522.08 | 524.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 09:15:00 | 491.11 | 503.12 | 511.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-02 14:15:00 | 501.96 | 501.83 | 507.83 | SL hit (close>ema200) qty=0.50 sl=501.83 alert=retest2 |

### Cycle 85 — BUY (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 12:15:00 | 534.40 | 512.80 | 510.91 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 522.02 | 531.66 | 532.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 12:15:00 | 514.74 | 526.63 | 529.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 14:15:00 | 526.78 | 525.53 | 528.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 14:15:00 | 526.78 | 525.53 | 528.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 526.78 | 525.53 | 528.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:30:00 | 528.00 | 525.53 | 528.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 531.40 | 526.62 | 528.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:45:00 | 526.20 | 527.12 | 528.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 550.52 | 529.02 | 528.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 550.52 | 529.02 | 528.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 558.94 | 544.55 | 536.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 539.52 | 544.96 | 539.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 539.52 | 544.96 | 539.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 539.52 | 544.96 | 539.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:00:00 | 539.52 | 544.96 | 539.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 532.52 | 542.48 | 538.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:30:00 | 530.22 | 542.48 | 538.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 540.00 | 541.98 | 538.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 12:15:00 | 542.28 | 541.98 | 538.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 15:15:00 | 542.40 | 540.88 | 538.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 13:15:00 | 554.34 | 571.22 | 571.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 13:15:00 | 554.34 | 571.22 | 571.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 510.84 | 556.46 | 564.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 14:15:00 | 551.00 | 539.94 | 551.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 14:15:00 | 551.00 | 539.94 | 551.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 551.00 | 539.94 | 551.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 15:00:00 | 551.00 | 539.94 | 551.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 542.00 | 540.35 | 550.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 538.68 | 540.35 | 550.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 10:00:00 | 539.78 | 540.24 | 549.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:15:00 | 537.50 | 540.61 | 548.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 12:15:00 | 540.24 | 538.37 | 542.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 543.08 | 539.31 | 542.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:30:00 | 545.06 | 539.31 | 542.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 553.30 | 542.11 | 543.86 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-29 13:15:00 | 553.30 | 542.11 | 543.86 | SL hit (close>static) qty=1.00 sl=551.38 alert=retest2 |

### Cycle 89 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 565.00 | 546.69 | 545.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 576.92 | 555.66 | 550.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 561.78 | 564.72 | 558.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 561.78 | 564.72 | 558.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 561.78 | 564.72 | 558.79 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 553.64 | 557.22 | 557.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 15:15:00 | 550.00 | 554.87 | 556.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 555.22 | 554.06 | 555.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 11:15:00 | 555.22 | 554.06 | 555.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 555.22 | 554.06 | 555.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 555.22 | 554.06 | 555.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 555.76 | 554.40 | 555.49 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 560.56 | 556.59 | 556.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 14:15:00 | 575.64 | 563.43 | 560.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 573.64 | 574.29 | 569.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 13:15:00 | 565.00 | 571.94 | 569.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 565.00 | 571.94 | 569.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:00:00 | 565.00 | 571.94 | 569.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 572.44 | 572.04 | 569.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:30:00 | 567.30 | 572.04 | 569.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 577.86 | 572.97 | 570.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:30:00 | 571.52 | 572.97 | 570.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 571.28 | 574.39 | 572.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 14:00:00 | 571.28 | 574.39 | 572.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 570.08 | 573.53 | 572.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 570.08 | 573.53 | 572.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 566.44 | 572.11 | 571.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 570.36 | 572.11 | 571.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 563.80 | 570.45 | 570.84 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 573.00 | 570.31 | 570.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 12:15:00 | 577.18 | 573.49 | 571.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 15:15:00 | 572.94 | 574.28 | 572.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 15:15:00 | 572.94 | 574.28 | 572.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 572.94 | 574.28 | 572.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 577.98 | 574.28 | 572.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 10:15:00 | 577.76 | 588.65 | 583.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 10:15:00 | 570.98 | 585.11 | 582.29 | SL hit (close<static) qty=1.00 sl=572.30 alert=retest2 |

### Cycle 94 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 551.84 | 578.46 | 579.52 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 601.40 | 581.79 | 580.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 603.80 | 586.19 | 582.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 591.54 | 593.17 | 587.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 15:00:00 | 591.54 | 593.17 | 587.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 582.08 | 592.10 | 589.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:00:00 | 582.08 | 592.10 | 589.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 585.98 | 590.87 | 589.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 14:30:00 | 587.66 | 589.02 | 588.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 09:15:00 | 580.00 | 586.60 | 587.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 09:15:00 | 580.00 | 586.60 | 587.40 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 12:15:00 | 592.68 | 588.59 | 588.16 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 13:15:00 | 582.68 | 587.41 | 587.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 571.80 | 584.29 | 586.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 582.62 | 581.79 | 584.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 10:00:00 | 582.62 | 581.79 | 584.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 586.18 | 582.67 | 584.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 585.00 | 582.67 | 584.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 584.22 | 582.98 | 584.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:30:00 | 576.80 | 582.21 | 584.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 14:45:00 | 581.02 | 582.36 | 583.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 15:15:00 | 577.00 | 582.36 | 583.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 589.82 | 584.36 | 584.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 11:15:00 | 589.82 | 584.36 | 584.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 12:15:00 | 592.96 | 586.08 | 585.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 15:15:00 | 602.00 | 604.33 | 597.98 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-26 10:00:00 | 615.88 | 606.64 | 599.60 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 646.67 | 635.74 | 624.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 614.00 | 637.28 | 631.99 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 614.00 | 637.28 | 631.99 | SL hit (close<ema200) qty=0.50 sl=637.28 alert=retest1 |

### Cycle 100 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 628.26 | 637.81 | 638.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 614.98 | 633.25 | 636.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 12:15:00 | 618.42 | 617.14 | 624.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 12:45:00 | 618.38 | 617.14 | 624.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 617.98 | 617.21 | 622.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:15:00 | 608.40 | 616.78 | 620.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 625.52 | 620.78 | 620.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 625.52 | 620.78 | 620.25 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 15:15:00 | 618.40 | 621.22 | 621.38 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 625.94 | 622.16 | 621.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 630.40 | 625.65 | 623.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 11:15:00 | 653.00 | 653.47 | 647.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 11:45:00 | 652.92 | 653.47 | 647.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 638.30 | 650.44 | 648.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 638.30 | 650.44 | 648.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 640.28 | 648.41 | 647.57 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 11:15:00 | 636.58 | 646.05 | 646.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 633.02 | 643.44 | 645.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 14:15:00 | 636.28 | 635.85 | 639.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 14:15:00 | 636.28 | 635.85 | 639.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 636.28 | 635.85 | 639.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 15:15:00 | 626.00 | 635.85 | 639.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 594.70 | 599.88 | 604.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 618.72 | 601.57 | 603.69 | SL hit (close>ema200) qty=0.50 sl=601.57 alert=retest2 |

### Cycle 105 — BUY (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 11:15:00 | 612.98 | 606.41 | 605.68 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 15:15:00 | 602.88 | 605.16 | 605.36 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 09:15:00 | 618.00 | 607.73 | 606.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 09:15:00 | 619.68 | 614.52 | 610.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 14:15:00 | 616.56 | 616.71 | 613.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 15:15:00 | 619.84 | 616.71 | 613.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 631.58 | 634.48 | 628.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:00:00 | 631.58 | 634.48 | 628.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 653.34 | 657.16 | 654.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 652.88 | 657.16 | 654.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 651.38 | 656.01 | 654.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:45:00 | 651.08 | 656.01 | 654.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 635.56 | 650.19 | 651.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 622.78 | 636.93 | 643.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 645.66 | 632.99 | 638.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 645.66 | 632.99 | 638.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 645.66 | 632.99 | 638.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 645.66 | 632.99 | 638.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 646.98 | 635.79 | 639.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 646.98 | 635.79 | 639.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 651.06 | 642.86 | 641.84 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 641.92 | 643.06 | 643.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 14:15:00 | 633.10 | 641.07 | 642.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 638.88 | 634.54 | 636.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 638.88 | 634.54 | 636.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 638.88 | 634.54 | 636.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 638.32 | 634.54 | 636.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 637.60 | 635.15 | 636.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 09:15:00 | 632.76 | 635.15 | 636.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 11:15:00 | 601.12 | 607.68 | 612.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 12:15:00 | 601.70 | 600.83 | 605.73 | SL hit (close>ema200) qty=0.50 sl=600.83 alert=retest2 |

### Cycle 111 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 558.02 | 551.53 | 551.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 15:15:00 | 562.20 | 553.66 | 552.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 562.00 | 565.97 | 561.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 09:45:00 | 562.60 | 565.97 | 561.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 564.80 | 565.74 | 561.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:15:00 | 561.58 | 565.74 | 561.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 562.40 | 565.07 | 561.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:30:00 | 561.66 | 565.07 | 561.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 561.30 | 564.31 | 561.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 561.30 | 564.31 | 561.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 561.64 | 563.78 | 561.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:30:00 | 561.40 | 563.78 | 561.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 554.66 | 561.96 | 560.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 554.66 | 561.96 | 560.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 554.60 | 560.48 | 560.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 549.34 | 560.48 | 560.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 09:15:00 | 544.02 | 557.19 | 558.81 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 12:15:00 | 560.48 | 558.23 | 558.06 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 09:15:00 | 550.88 | 557.17 | 557.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 10:15:00 | 548.46 | 555.43 | 556.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 12:15:00 | 542.78 | 541.91 | 545.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 13:00:00 | 542.78 | 541.91 | 545.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 534.28 | 534.90 | 539.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 526.48 | 539.79 | 540.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 10:15:00 | 541.00 | 531.81 | 533.87 | SL hit (close>static) qty=1.00 sl=540.72 alert=retest2 |

### Cycle 115 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 543.48 | 535.79 | 535.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 13:15:00 | 548.22 | 540.29 | 538.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 536.80 | 541.62 | 539.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 536.80 | 541.62 | 539.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 536.80 | 541.62 | 539.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 536.80 | 541.62 | 539.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 541.14 | 541.52 | 539.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 11:15:00 | 542.64 | 541.52 | 539.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:15:00 | 542.66 | 541.44 | 539.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 528.12 | 538.47 | 539.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 09:15:00 | 528.12 | 538.47 | 539.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 522.60 | 528.50 | 532.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 521.54 | 521.01 | 523.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 521.54 | 521.01 | 523.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 531.02 | 522.60 | 523.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 531.02 | 522.60 | 523.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 530.54 | 524.19 | 524.02 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 521.72 | 524.66 | 524.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 520.16 | 523.76 | 524.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 516.78 | 509.37 | 512.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 516.78 | 509.37 | 512.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 516.78 | 509.37 | 512.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 517.98 | 509.37 | 512.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 516.92 | 510.88 | 512.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 516.92 | 510.88 | 512.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 519.80 | 514.01 | 513.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 529.40 | 520.22 | 517.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 13:15:00 | 512.64 | 521.25 | 518.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 13:15:00 | 512.64 | 521.25 | 518.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 512.64 | 521.25 | 518.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:00:00 | 512.64 | 521.25 | 518.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 522.04 | 521.41 | 519.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:30:00 | 521.70 | 521.41 | 519.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 515.56 | 520.17 | 518.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 516.04 | 520.17 | 518.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 521.18 | 520.37 | 519.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 515.84 | 520.37 | 519.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 517.80 | 519.86 | 519.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:00:00 | 517.80 | 519.86 | 519.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 521.54 | 520.20 | 519.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:30:00 | 519.90 | 520.20 | 519.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 550.56 | 555.95 | 548.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 549.64 | 555.95 | 548.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 543.18 | 553.39 | 548.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:45:00 | 541.30 | 553.39 | 548.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 545.00 | 551.72 | 547.77 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2025-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 15:15:00 | 540.00 | 545.08 | 545.66 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 551.84 | 546.43 | 546.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 11:15:00 | 553.28 | 548.65 | 547.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 573.72 | 575.27 | 567.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:45:00 | 573.00 | 575.27 | 567.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 578.46 | 580.12 | 577.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 578.48 | 580.12 | 577.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 576.18 | 579.33 | 577.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 578.74 | 579.33 | 577.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 579.72 | 579.41 | 577.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 10:30:00 | 582.06 | 579.34 | 578.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 13:15:00 | 581.90 | 579.39 | 578.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 571.56 | 576.99 | 577.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 571.56 | 576.99 | 577.50 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 11:15:00 | 579.32 | 577.94 | 577.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 13:15:00 | 586.80 | 580.04 | 578.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 15:15:00 | 624.20 | 625.19 | 615.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 09:15:00 | 619.48 | 625.19 | 615.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 613.62 | 622.88 | 614.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 613.62 | 622.88 | 614.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 615.40 | 621.38 | 614.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:00:00 | 619.20 | 616.13 | 614.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:30:00 | 619.14 | 616.63 | 614.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:15:00 | 619.12 | 616.63 | 614.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:45:00 | 619.16 | 617.18 | 615.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 617.98 | 624.86 | 621.89 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-07 13:15:00 | 618.50 | 620.26 | 620.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 13:15:00 | 618.50 | 620.26 | 620.31 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 09:15:00 | 624.98 | 620.70 | 620.47 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 616.92 | 620.05 | 620.22 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 15:15:00 | 622.00 | 620.32 | 620.19 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 617.00 | 619.66 | 619.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 612.24 | 616.51 | 618.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 618.08 | 614.46 | 616.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 618.08 | 614.46 | 616.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 618.08 | 614.46 | 616.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 13:15:00 | 609.78 | 614.09 | 615.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 620.00 | 617.10 | 617.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 620.00 | 617.10 | 617.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 646.30 | 622.94 | 619.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 15:15:00 | 645.00 | 653.02 | 639.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 636.00 | 657.44 | 650.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 636.00 | 657.44 | 650.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:30:00 | 634.40 | 657.44 | 650.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 630.30 | 652.01 | 648.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:00:00 | 630.30 | 652.01 | 648.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 12:15:00 | 628.00 | 643.59 | 645.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 617.40 | 634.66 | 640.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 519.00 | 514.43 | 532.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 15:00:00 | 519.00 | 514.43 | 532.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 508.90 | 504.42 | 515.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 499.60 | 511.39 | 515.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 500.90 | 507.71 | 511.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 495.30 | 506.81 | 509.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:15:00 | 501.30 | 500.75 | 505.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 506.65 | 501.58 | 504.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:00:00 | 506.65 | 501.58 | 504.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 510.60 | 503.38 | 504.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:30:00 | 508.90 | 503.38 | 504.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 526.60 | 508.60 | 506.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 526.60 | 508.60 | 506.86 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 13:15:00 | 504.20 | 512.66 | 513.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 15:15:00 | 502.20 | 509.18 | 511.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 501.40 | 497.98 | 502.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 11:15:00 | 501.40 | 497.98 | 502.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 501.40 | 497.98 | 502.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:45:00 | 501.15 | 497.98 | 502.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 505.25 | 499.43 | 502.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:45:00 | 505.35 | 499.43 | 502.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 500.60 | 499.66 | 502.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 506.50 | 499.66 | 502.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 505.80 | 500.89 | 502.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 505.80 | 500.89 | 502.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 504.00 | 501.51 | 502.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 496.95 | 501.51 | 502.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 497.40 | 500.69 | 502.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 484.00 | 498.86 | 500.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 494.50 | 493.83 | 495.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 13:15:00 | 500.85 | 497.22 | 496.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 13:15:00 | 500.85 | 497.22 | 496.86 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 487.15 | 494.78 | 495.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 10:15:00 | 484.95 | 488.97 | 491.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 500.70 | 488.85 | 490.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 14:15:00 | 500.70 | 488.85 | 490.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 500.70 | 488.85 | 490.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:45:00 | 500.40 | 488.85 | 490.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 501.65 | 491.41 | 491.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 501.75 | 491.41 | 491.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 503.80 | 493.89 | 492.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 513.20 | 500.17 | 497.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 09:15:00 | 504.00 | 506.23 | 502.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 504.00 | 506.23 | 502.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 504.00 | 506.23 | 502.57 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 13:15:00 | 499.75 | 502.36 | 502.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 497.45 | 500.98 | 501.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 15:15:00 | 500.00 | 498.14 | 499.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 15:15:00 | 500.00 | 498.14 | 499.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 500.00 | 498.14 | 499.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:45:00 | 493.90 | 496.80 | 498.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:30:00 | 493.60 | 495.58 | 497.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 14:45:00 | 494.50 | 494.81 | 497.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 15:15:00 | 490.00 | 494.81 | 497.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 11:15:00 | 469.20 | 476.97 | 483.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 11:15:00 | 468.92 | 476.97 | 483.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 11:15:00 | 469.77 | 476.97 | 483.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 11:15:00 | 465.50 | 476.97 | 483.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 444.51 | 459.05 | 471.58 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 137 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 478.80 | 437.38 | 432.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 484.90 | 453.74 | 441.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 09:15:00 | 452.10 | 462.80 | 449.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 09:45:00 | 455.50 | 462.80 | 449.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 454.60 | 459.61 | 450.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:30:00 | 455.00 | 459.61 | 450.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 452.25 | 456.75 | 451.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 452.25 | 456.75 | 451.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 452.10 | 455.82 | 451.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 434.55 | 455.82 | 451.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 436.90 | 452.03 | 449.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 434.75 | 452.03 | 449.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 11:15:00 | 441.20 | 448.13 | 448.40 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 14:15:00 | 450.30 | 448.76 | 448.62 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 15:15:00 | 445.50 | 448.11 | 448.33 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 09:15:00 | 473.15 | 453.12 | 450.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 490.70 | 478.22 | 470.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 12:15:00 | 480.05 | 481.05 | 473.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 12:45:00 | 476.65 | 481.05 | 473.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 484.40 | 482.74 | 476.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 13:00:00 | 494.80 | 483.69 | 480.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 489.65 | 482.69 | 482.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 11:15:00 | 488.45 | 483.58 | 482.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 15:15:00 | 473.40 | 497.70 | 500.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 473.40 | 497.70 | 500.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 444.45 | 487.05 | 495.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 11:15:00 | 422.10 | 421.32 | 432.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 12:00:00 | 422.10 | 421.32 | 432.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 432.65 | 419.49 | 423.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:30:00 | 433.00 | 419.49 | 423.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 437.85 | 425.73 | 425.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 12:15:00 | 439.50 | 428.48 | 426.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 433.45 | 434.78 | 430.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 433.45 | 434.78 | 430.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 433.45 | 434.78 | 430.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 433.45 | 434.78 | 430.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 431.55 | 434.34 | 432.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 431.55 | 434.34 | 432.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 429.50 | 433.37 | 431.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 441.45 | 433.37 | 431.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 423.60 | 437.16 | 435.88 | SL hit (close<static) qty=1.00 sl=428.00 alert=retest2 |

### Cycle 144 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 427.70 | 433.77 | 434.47 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 444.00 | 435.81 | 434.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 13:15:00 | 446.80 | 440.62 | 437.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 12:15:00 | 444.80 | 444.83 | 441.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 12:45:00 | 444.80 | 444.83 | 441.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 464.50 | 468.03 | 464.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 11:00:00 | 464.50 | 468.03 | 464.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 466.55 | 467.74 | 464.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 13:15:00 | 470.50 | 467.67 | 464.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-28 10:15:00 | 517.55 | 495.89 | 490.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 484.00 | 491.73 | 492.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 473.00 | 487.99 | 490.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 463.30 | 457.60 | 464.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 463.30 | 457.60 | 464.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 463.30 | 457.60 | 464.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 463.30 | 457.60 | 464.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 467.90 | 459.66 | 464.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:00:00 | 467.90 | 459.66 | 464.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 466.15 | 460.96 | 464.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:30:00 | 466.40 | 460.96 | 464.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 467.95 | 463.22 | 465.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:45:00 | 467.10 | 463.22 | 465.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 473.00 | 466.72 | 466.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 485.20 | 470.41 | 468.19 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-16 09:15:00 | 167.41 | 2024-05-21 15:15:00 | 170.00 | STOP_HIT | 1.00 | 1.55% |
| SELL | retest2 | 2024-05-30 14:45:00 | 161.80 | 2024-05-31 10:15:00 | 172.32 | STOP_HIT | 1.00 | -6.50% |
| BUY | retest2 | 2024-06-04 13:30:00 | 167.45 | 2024-06-04 15:15:00 | 164.28 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-06-04 15:00:00 | 166.36 | 2024-06-04 15:15:00 | 164.28 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest1 | 2024-06-10 15:00:00 | 208.68 | 2024-06-11 09:15:00 | 219.11 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-06-10 15:00:00 | 208.68 | 2024-06-11 14:15:00 | 210.43 | STOP_HIT | 0.50 | 0.84% |
| BUY | retest2 | 2024-06-12 11:00:00 | 216.74 | 2024-06-18 14:15:00 | 216.24 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-06-12 13:00:00 | 216.28 | 2024-06-18 14:15:00 | 216.24 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2024-06-12 13:45:00 | 215.90 | 2024-06-18 14:15:00 | 216.24 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2024-06-25 14:15:00 | 229.66 | 2024-06-27 13:15:00 | 222.36 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2024-06-25 15:00:00 | 229.96 | 2024-06-27 13:15:00 | 222.36 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2024-06-27 09:15:00 | 232.73 | 2024-06-27 13:15:00 | 222.36 | STOP_HIT | 1.00 | -4.46% |
| SELL | retest2 | 2024-07-05 09:45:00 | 226.82 | 2024-07-08 10:15:00 | 230.95 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-07-05 11:00:00 | 226.00 | 2024-07-08 10:15:00 | 230.95 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-07-05 14:00:00 | 226.75 | 2024-07-08 10:15:00 | 230.95 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-07-19 10:45:00 | 203.52 | 2024-07-23 12:15:00 | 193.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 11:30:00 | 203.92 | 2024-07-23 12:15:00 | 193.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 12:15:00 | 203.35 | 2024-07-23 12:15:00 | 193.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 13:00:00 | 203.39 | 2024-07-23 12:15:00 | 193.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 10:45:00 | 203.52 | 2024-07-23 15:15:00 | 201.00 | STOP_HIT | 0.50 | 1.24% |
| SELL | retest2 | 2024-07-19 11:30:00 | 203.92 | 2024-07-23 15:15:00 | 201.00 | STOP_HIT | 0.50 | 1.43% |
| SELL | retest2 | 2024-07-19 12:15:00 | 203.35 | 2024-07-23 15:15:00 | 201.00 | STOP_HIT | 0.50 | 1.16% |
| SELL | retest2 | 2024-07-19 13:00:00 | 203.39 | 2024-07-23 15:15:00 | 201.00 | STOP_HIT | 0.50 | 1.18% |
| SELL | retest2 | 2024-07-23 09:30:00 | 199.13 | 2024-07-25 09:15:00 | 205.28 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2024-07-23 11:15:00 | 198.71 | 2024-07-25 09:15:00 | 205.28 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2024-07-23 14:15:00 | 198.36 | 2024-07-25 09:15:00 | 205.28 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2024-07-23 14:45:00 | 198.74 | 2024-07-25 09:15:00 | 205.28 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2024-07-24 10:15:00 | 200.59 | 2024-07-25 09:15:00 | 205.28 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2024-07-24 13:15:00 | 199.85 | 2024-07-25 09:15:00 | 205.28 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2024-08-14 14:15:00 | 279.25 | 2024-08-20 14:15:00 | 304.90 | TARGET_HIT | 1.00 | 9.18% |
| BUY | retest2 | 2024-08-14 14:45:00 | 277.18 | 2024-08-20 14:15:00 | 305.57 | TARGET_HIT | 1.00 | 10.24% |
| BUY | retest2 | 2024-08-16 09:15:00 | 278.00 | 2024-08-21 09:15:00 | 307.18 | TARGET_HIT | 1.00 | 10.49% |
| BUY | retest2 | 2024-08-16 15:15:00 | 277.79 | 2024-08-21 09:15:00 | 305.80 | TARGET_HIT | 1.00 | 10.08% |
| BUY | retest2 | 2024-08-20 09:15:00 | 282.40 | 2024-08-21 09:15:00 | 310.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-20 10:15:00 | 279.05 | 2024-08-21 09:15:00 | 306.96 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-06 09:45:00 | 326.20 | 2024-09-13 09:15:00 | 336.94 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2024-09-09 09:15:00 | 316.40 | 2024-09-13 09:15:00 | 336.94 | STOP_HIT | 1.00 | -6.49% |
| BUY | retest2 | 2024-09-25 09:15:00 | 343.16 | 2024-09-25 09:15:00 | 338.92 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-09-30 09:15:00 | 347.20 | 2024-10-07 10:15:00 | 343.20 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-10-14 09:15:00 | 368.42 | 2024-10-14 12:15:00 | 358.41 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2024-10-17 12:30:00 | 367.99 | 2024-10-18 15:15:00 | 365.79 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-10-18 12:00:00 | 368.64 | 2024-10-18 15:15:00 | 365.79 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-10-18 13:15:00 | 368.00 | 2024-10-18 15:15:00 | 365.79 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest1 | 2024-10-25 09:30:00 | 325.60 | 2024-10-28 15:15:00 | 326.34 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2024-10-28 09:15:00 | 318.84 | 2024-10-30 15:15:00 | 332.31 | STOP_HIT | 1.00 | -4.22% |
| SELL | retest2 | 2024-10-28 12:00:00 | 319.29 | 2024-10-30 15:15:00 | 332.31 | STOP_HIT | 1.00 | -4.08% |
| SELL | retest2 | 2024-10-29 10:45:00 | 319.51 | 2024-10-30 15:15:00 | 332.31 | STOP_HIT | 1.00 | -4.01% |
| SELL | retest2 | 2024-10-29 11:45:00 | 319.80 | 2024-10-30 15:15:00 | 332.31 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2024-10-30 09:15:00 | 320.54 | 2024-10-30 15:15:00 | 332.31 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest2 | 2024-11-05 10:45:00 | 339.17 | 2024-11-08 09:15:00 | 334.64 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-11-06 09:15:00 | 339.00 | 2024-11-08 09:15:00 | 334.64 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-11-06 11:30:00 | 343.01 | 2024-11-08 09:15:00 | 334.64 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2024-11-25 14:30:00 | 311.34 | 2024-11-27 09:15:00 | 316.60 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-11-25 15:00:00 | 310.41 | 2024-11-27 09:15:00 | 316.60 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-12-02 10:00:00 | 318.19 | 2024-12-05 12:15:00 | 323.04 | STOP_HIT | 1.00 | 1.52% |
| BUY | retest2 | 2024-12-02 11:15:00 | 318.44 | 2024-12-05 12:15:00 | 323.04 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest2 | 2024-12-16 09:45:00 | 373.59 | 2024-12-16 11:15:00 | 365.63 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-12-16 13:15:00 | 368.04 | 2024-12-16 14:15:00 | 365.39 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-01-07 09:15:00 | 393.00 | 2025-01-08 12:15:00 | 378.07 | STOP_HIT | 1.00 | -3.80% |
| SELL | retest2 | 2025-01-10 13:45:00 | 372.55 | 2025-01-13 12:15:00 | 353.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-13 11:00:00 | 365.66 | 2025-01-13 14:15:00 | 347.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 13:45:00 | 372.55 | 2025-01-14 13:15:00 | 335.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-13 11:00:00 | 365.66 | 2025-01-15 09:15:00 | 354.54 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2025-01-30 13:15:00 | 334.15 | 2025-01-31 15:15:00 | 339.40 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-01-31 14:00:00 | 334.90 | 2025-01-31 15:15:00 | 339.40 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-02-10 09:15:00 | 367.19 | 2025-02-11 09:15:00 | 348.89 | STOP_HIT | 1.00 | -4.98% |
| BUY | retest2 | 2025-02-10 10:15:00 | 361.65 | 2025-02-11 09:15:00 | 348.89 | STOP_HIT | 1.00 | -3.53% |
| BUY | retest2 | 2025-02-10 15:15:00 | 361.98 | 2025-02-11 09:15:00 | 348.89 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2025-02-14 09:15:00 | 330.15 | 2025-02-17 09:15:00 | 317.61 | PARTIAL | 0.50 | 3.80% |
| SELL | retest2 | 2025-02-14 10:15:00 | 334.33 | 2025-02-17 12:15:00 | 313.64 | PARTIAL | 0.50 | 6.19% |
| SELL | retest2 | 2025-02-14 09:15:00 | 330.15 | 2025-02-18 11:15:00 | 300.90 | TARGET_HIT | 0.50 | 8.86% |
| SELL | retest2 | 2025-02-14 10:15:00 | 334.33 | 2025-02-18 12:15:00 | 297.13 | TARGET_HIT | 0.50 | 11.13% |
| SELL | retest2 | 2025-02-25 09:15:00 | 298.20 | 2025-02-28 09:15:00 | 268.38 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-25 09:45:00 | 298.40 | 2025-02-28 09:15:00 | 268.56 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-27 09:15:00 | 298.02 | 2025-02-28 09:15:00 | 268.22 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-10 10:15:00 | 296.50 | 2025-03-17 12:15:00 | 282.11 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2025-03-10 10:45:00 | 296.96 | 2025-03-17 13:15:00 | 281.68 | PARTIAL | 0.50 | 5.15% |
| SELL | retest2 | 2025-03-10 10:15:00 | 296.50 | 2025-03-18 09:15:00 | 287.37 | STOP_HIT | 0.50 | 3.08% |
| SELL | retest2 | 2025-03-10 10:45:00 | 296.96 | 2025-03-18 09:15:00 | 287.37 | STOP_HIT | 0.50 | 3.23% |
| BUY | retest2 | 2025-03-24 09:15:00 | 314.60 | 2025-03-26 12:15:00 | 311.80 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-03-26 09:15:00 | 316.00 | 2025-03-26 12:15:00 | 311.80 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-03-26 10:45:00 | 314.95 | 2025-03-26 12:15:00 | 311.80 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-04-09 09:15:00 | 317.10 | 2025-04-15 09:15:00 | 332.68 | STOP_HIT | 1.00 | -4.91% |
| BUY | retest1 | 2025-04-23 09:15:00 | 377.02 | 2025-04-24 09:15:00 | 366.58 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest1 | 2025-04-23 09:45:00 | 374.32 | 2025-04-24 09:15:00 | 366.58 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-05-02 12:00:00 | 354.68 | 2025-05-02 14:15:00 | 359.30 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-05-02 14:15:00 | 355.80 | 2025-05-02 14:15:00 | 359.30 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-05-21 15:00:00 | 444.60 | 2025-06-03 09:15:00 | 489.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-22 09:15:00 | 443.56 | 2025-06-03 09:15:00 | 487.92 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-22 15:15:00 | 442.20 | 2025-06-03 09:15:00 | 486.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-26 15:00:00 | 442.40 | 2025-06-03 09:15:00 | 486.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-28 09:15:00 | 457.56 | 2025-06-04 12:15:00 | 503.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2025-06-24 09:15:00 | 522.80 | 2025-06-26 10:15:00 | 531.06 | STOP_HIT | 1.00 | 1.58% |
| SELL | retest2 | 2025-07-01 09:15:00 | 516.96 | 2025-07-02 09:15:00 | 491.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-01 09:15:00 | 516.96 | 2025-07-02 14:15:00 | 501.96 | STOP_HIT | 0.50 | 2.90% |
| SELL | retest2 | 2025-07-14 11:45:00 | 526.20 | 2025-07-15 09:15:00 | 550.52 | STOP_HIT | 1.00 | -4.62% |
| BUY | retest2 | 2025-07-16 12:15:00 | 542.28 | 2025-07-24 13:15:00 | 554.34 | STOP_HIT | 1.00 | 2.22% |
| BUY | retest2 | 2025-07-16 15:15:00 | 542.40 | 2025-07-24 13:15:00 | 554.34 | STOP_HIT | 1.00 | 2.20% |
| SELL | retest2 | 2025-07-28 09:15:00 | 538.68 | 2025-07-29 13:15:00 | 553.30 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-07-28 10:00:00 | 539.78 | 2025-07-29 13:15:00 | 553.30 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-07-28 12:15:00 | 537.50 | 2025-07-29 13:15:00 | 553.30 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2025-07-29 12:15:00 | 540.24 | 2025-07-29 13:15:00 | 553.30 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-08-13 09:15:00 | 577.98 | 2025-08-14 10:15:00 | 570.98 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-08-14 10:15:00 | 577.76 | 2025-08-14 10:15:00 | 570.98 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-08-19 14:30:00 | 587.66 | 2025-08-20 09:15:00 | 580.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-08-21 12:30:00 | 576.80 | 2025-08-22 11:15:00 | 589.82 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-08-21 14:45:00 | 581.02 | 2025-08-22 11:15:00 | 589.82 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-08-21 15:15:00 | 577.00 | 2025-08-22 11:15:00 | 589.82 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest1 | 2025-08-26 10:00:00 | 615.88 | 2025-08-29 09:15:00 | 646.67 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-08-26 10:00:00 | 615.88 | 2025-09-01 09:15:00 | 614.00 | STOP_HIT | 0.50 | -0.31% |
| BUY | retest2 | 2025-09-01 14:45:00 | 644.16 | 2025-09-04 10:15:00 | 628.26 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-09-03 09:15:00 | 646.78 | 2025-09-04 10:15:00 | 628.26 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-09-04 09:15:00 | 638.68 | 2025-09-04 10:15:00 | 628.26 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-09-08 15:15:00 | 608.40 | 2025-09-10 09:15:00 | 625.52 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-09-19 15:15:00 | 626.00 | 2025-09-26 13:15:00 | 594.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 15:15:00 | 626.00 | 2025-09-29 09:15:00 | 618.72 | STOP_HIT | 0.50 | 1.16% |
| SELL | retest2 | 2025-10-23 09:15:00 | 632.76 | 2025-10-28 11:15:00 | 601.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-23 09:15:00 | 632.76 | 2025-10-29 12:15:00 | 601.70 | STOP_HIT | 0.50 | 4.91% |
| SELL | retest2 | 2025-11-25 09:15:00 | 526.48 | 2025-11-26 10:15:00 | 541.00 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-11-28 11:15:00 | 542.64 | 2025-12-01 09:15:00 | 528.12 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-11-28 13:15:00 | 542.66 | 2025-12-01 09:15:00 | 528.12 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-12-29 10:30:00 | 582.06 | 2025-12-30 09:15:00 | 571.56 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-12-29 13:15:00 | 581.90 | 2025-12-30 09:15:00 | 571.56 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2026-01-05 11:00:00 | 619.20 | 2026-01-07 13:15:00 | 618.50 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2026-01-05 11:30:00 | 619.14 | 2026-01-07 13:15:00 | 618.50 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2026-01-05 12:15:00 | 619.12 | 2026-01-07 13:15:00 | 618.50 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2026-01-05 12:45:00 | 619.16 | 2026-01-07 13:15:00 | 618.50 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2026-01-12 13:15:00 | 609.78 | 2026-01-12 15:15:00 | 620.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-01-30 09:15:00 | 499.60 | 2026-02-03 09:15:00 | 526.60 | STOP_HIT | 1.00 | -5.40% |
| SELL | retest2 | 2026-02-01 09:15:00 | 500.90 | 2026-02-03 09:15:00 | 526.60 | STOP_HIT | 1.00 | -5.13% |
| SELL | retest2 | 2026-02-01 12:15:00 | 495.30 | 2026-02-03 09:15:00 | 526.60 | STOP_HIT | 1.00 | -6.32% |
| SELL | retest2 | 2026-02-02 10:15:00 | 501.30 | 2026-02-03 09:15:00 | 526.60 | STOP_HIT | 1.00 | -5.05% |
| SELL | retest2 | 2026-02-11 09:15:00 | 484.00 | 2026-02-12 13:15:00 | 500.85 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2026-02-12 09:15:00 | 494.50 | 2026-02-12 13:15:00 | 500.85 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-02-25 11:45:00 | 493.90 | 2026-03-02 11:15:00 | 469.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 12:30:00 | 493.60 | 2026-03-02 11:15:00 | 468.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 14:45:00 | 494.50 | 2026-03-02 11:15:00 | 469.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 15:15:00 | 490.00 | 2026-03-02 11:15:00 | 465.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:45:00 | 493.90 | 2026-03-04 09:15:00 | 444.51 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 12:30:00 | 493.60 | 2026-03-04 09:15:00 | 444.24 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 14:45:00 | 494.50 | 2026-03-04 09:15:00 | 445.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 15:15:00 | 490.00 | 2026-03-04 09:15:00 | 441.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-06 10:45:00 | 432.85 | 2026-03-09 09:15:00 | 411.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 10:45:00 | 432.85 | 2026-03-10 09:15:00 | 437.75 | STOP_HIT | 0.50 | -1.13% |
| SELL | retest2 | 2026-03-10 10:00:00 | 437.75 | 2026-03-10 10:15:00 | 465.20 | STOP_HIT | 1.00 | -6.27% |
| BUY | retest2 | 2026-03-20 13:00:00 | 494.80 | 2026-03-27 15:15:00 | 473.40 | STOP_HIT | 1.00 | -4.32% |
| BUY | retest2 | 2026-03-24 09:15:00 | 489.65 | 2026-03-27 15:15:00 | 473.40 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2026-03-24 11:15:00 | 488.45 | 2026-03-27 15:15:00 | 473.40 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2026-04-10 09:15:00 | 441.45 | 2026-04-13 09:15:00 | 423.60 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest2 | 2026-04-22 13:15:00 | 470.50 | 2026-04-28 10:15:00 | 517.55 | TARGET_HIT | 1.00 | 10.00% |
