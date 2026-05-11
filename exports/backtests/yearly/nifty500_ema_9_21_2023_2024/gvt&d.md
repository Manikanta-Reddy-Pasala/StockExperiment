# GE Vernova T&D India Ltd. (GVT&D)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5317 bars)
- **Last close:** 4630.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 214 |
| ALERT1 | 147 |
| ALERT2 | 147 |
| ALERT2_SKIP | 85 |
| ALERT3 | 394 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 174 |
| PARTIAL | 26 |
| TARGET_HIT | 17 |
| STOP_HIT | 161 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 204 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 91 / 113
- **Target hits / Stop hits / Partials:** 17 / 161 / 26
- **Avg / median % per leg:** 0.62% / -0.80%
- **Sum % (uncompounded):** 127.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 99 | 40 | 40.4% | 15 | 81 | 3 | 0.40% | 40.0% |
| BUY @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 1 | 3 | 3 | 3.22% | 22.5% |
| BUY @ 3rd Alert (retest2) | 92 | 34 | 37.0% | 14 | 78 | 0 | 0.19% | 17.5% |
| SELL (all) | 105 | 51 | 48.6% | 2 | 80 | 23 | 0.83% | 87.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 105 | 51 | 48.6% | 2 | 80 | 23 | 0.83% | 87.3% |
| retest1 (combined) | 7 | 6 | 85.7% | 1 | 3 | 3 | 3.22% | 22.5% |
| retest2 (combined) | 197 | 85 | 43.1% | 16 | 158 | 23 | 0.53% | 104.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 14:15:00 | 210.35 | 212.79 | 212.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 11:15:00 | 208.40 | 211.88 | 212.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 14:15:00 | 211.40 | 211.11 | 211.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-17 14:45:00 | 209.50 | 211.11 | 211.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 2 — BUY (started 2023-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 09:15:00 | 226.00 | 213.89 | 212.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-18 10:15:00 | 233.00 | 217.71 | 214.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-19 12:15:00 | 228.10 | 229.42 | 224.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-19 12:45:00 | 228.20 | 229.42 | 224.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 224.50 | 229.22 | 226.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:30:00 | 223.75 | 229.22 | 226.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 10:15:00 | 227.00 | 228.78 | 226.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-22 12:45:00 | 228.50 | 228.42 | 226.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-22 14:15:00 | 221.00 | 226.20 | 225.78 | SL hit (close<static) qty=1.00 sl=223.75 alert=retest2 |

### Cycle 3 — SELL (started 2023-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-22 15:15:00 | 221.00 | 225.16 | 225.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-23 09:15:00 | 209.65 | 222.06 | 223.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-26 09:15:00 | 197.80 | 192.44 | 198.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-26 09:45:00 | 197.80 | 192.44 | 198.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 14:15:00 | 195.00 | 191.01 | 195.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 15:00:00 | 195.00 | 191.01 | 195.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 15:15:00 | 192.00 | 191.21 | 194.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-29 09:15:00 | 188.00 | 191.21 | 194.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-31 14:15:00 | 178.60 | 182.20 | 185.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-06-01 10:15:00 | 183.30 | 182.01 | 184.43 | SL hit (close>ema200) qty=0.50 sl=182.01 alert=retest2 |

### Cycle 4 — BUY (started 2023-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 11:15:00 | 188.50 | 182.89 | 182.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 12:15:00 | 189.20 | 184.15 | 183.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-09 10:15:00 | 213.95 | 214.89 | 209.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-09 10:45:00 | 212.00 | 214.89 | 209.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 11:15:00 | 214.70 | 214.86 | 209.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 11:45:00 | 214.95 | 214.86 | 209.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 12:15:00 | 213.95 | 214.67 | 209.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 13:00:00 | 213.95 | 214.67 | 209.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 13:15:00 | 210.05 | 213.75 | 209.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 13:45:00 | 211.50 | 213.75 | 209.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 14:15:00 | 208.25 | 212.65 | 209.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-12 11:30:00 | 213.75 | 211.70 | 210.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-12 12:15:00 | 213.65 | 211.70 | 210.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-13 09:30:00 | 213.90 | 210.57 | 210.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-13 10:15:00 | 205.65 | 209.58 | 209.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 10:15:00 | 205.65 | 209.58 | 209.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-13 12:15:00 | 204.70 | 208.06 | 208.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-14 09:15:00 | 209.50 | 206.79 | 207.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-14 09:15:00 | 209.50 | 206.79 | 207.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 09:15:00 | 209.50 | 206.79 | 207.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-14 10:00:00 | 209.50 | 206.79 | 207.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 10:15:00 | 207.10 | 206.85 | 207.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-14 11:15:00 | 210.00 | 206.85 | 207.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 11:15:00 | 207.95 | 207.07 | 207.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-14 13:15:00 | 206.05 | 207.14 | 207.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-15 13:15:00 | 195.75 | 201.51 | 204.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-06-16 09:15:00 | 202.00 | 200.38 | 202.97 | SL hit (close>ema200) qty=0.50 sl=200.38 alert=retest2 |

### Cycle 6 — BUY (started 2023-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 15:15:00 | 206.65 | 204.38 | 204.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 09:15:00 | 216.95 | 206.89 | 205.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 09:15:00 | 213.00 | 214.05 | 210.59 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 11:30:00 | 215.90 | 215.13 | 211.67 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-21 09:15:00 | 226.70 | 217.71 | 214.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-06-21 12:15:00 | 217.50 | 218.13 | 215.47 | SL hit (close<ema200) qty=0.50 sl=218.13 alert=retest1 |

### Cycle 7 — SELL (started 2023-06-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 12:15:00 | 211.00 | 215.18 | 215.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 15:15:00 | 210.35 | 213.19 | 214.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 14:15:00 | 210.45 | 209.57 | 211.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-23 15:00:00 | 210.45 | 209.57 | 211.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 208.00 | 209.32 | 211.12 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 14:15:00 | 213.50 | 211.98 | 211.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 09:15:00 | 215.00 | 212.75 | 212.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 09:15:00 | 218.00 | 218.15 | 215.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-28 09:30:00 | 220.00 | 218.15 | 215.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 220.00 | 220.02 | 217.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 14:45:00 | 217.20 | 220.02 | 217.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 09:15:00 | 219.00 | 219.97 | 218.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-03 11:00:00 | 223.50 | 220.88 | 219.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 09:15:00 | 227.95 | 221.85 | 220.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 10:15:00 | 223.95 | 221.88 | 220.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 14:30:00 | 223.50 | 223.04 | 221.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 15:15:00 | 222.50 | 222.93 | 221.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 09:15:00 | 218.25 | 222.93 | 221.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 217.15 | 221.78 | 221.54 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-05 11:15:00 | 218.85 | 221.06 | 221.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2023-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-05 11:15:00 | 218.85 | 221.06 | 221.25 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-07-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 15:15:00 | 225.75 | 221.62 | 221.36 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-07-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-06 09:15:00 | 219.00 | 221.09 | 221.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 10:15:00 | 214.75 | 217.89 | 219.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-07 14:15:00 | 221.05 | 217.93 | 218.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 14:15:00 | 221.05 | 217.93 | 218.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 14:15:00 | 221.05 | 217.93 | 218.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-07 15:00:00 | 221.05 | 217.93 | 218.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 15:15:00 | 221.80 | 218.71 | 219.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-10 09:45:00 | 219.65 | 219.16 | 219.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-10 13:15:00 | 220.70 | 219.25 | 219.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2023-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 13:15:00 | 220.70 | 219.25 | 219.16 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 14:15:00 | 217.95 | 218.99 | 219.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-11 12:15:00 | 215.00 | 217.08 | 217.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 14:15:00 | 220.45 | 217.07 | 217.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 14:15:00 | 220.45 | 217.07 | 217.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 220.45 | 217.07 | 217.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 15:00:00 | 220.45 | 217.07 | 217.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 15:15:00 | 220.95 | 217.84 | 218.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-12 09:15:00 | 213.05 | 217.84 | 218.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-12 10:00:00 | 219.25 | 218.13 | 218.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-12 10:15:00 | 220.95 | 218.69 | 218.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2023-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 10:15:00 | 220.95 | 218.69 | 218.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 11:15:00 | 222.00 | 219.35 | 218.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 13:15:00 | 228.00 | 228.10 | 224.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-13 14:00:00 | 228.00 | 228.10 | 224.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 14:15:00 | 226.50 | 227.78 | 224.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 15:00:00 | 226.50 | 227.78 | 224.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 09:15:00 | 236.00 | 229.15 | 225.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 11:00:00 | 237.65 | 230.85 | 226.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-18 09:15:00 | 261.42 | 248.00 | 241.25 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2023-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 10:15:00 | 244.95 | 250.67 | 251.21 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 15:15:00 | 255.00 | 251.51 | 251.27 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 13:15:00 | 248.25 | 251.03 | 251.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-27 14:15:00 | 246.00 | 249.41 | 250.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 11:15:00 | 244.00 | 242.47 | 244.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-31 11:45:00 | 244.50 | 242.47 | 244.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 12:15:00 | 250.85 | 244.15 | 245.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 13:00:00 | 250.85 | 244.15 | 245.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 13:15:00 | 250.00 | 245.32 | 245.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 13:30:00 | 253.00 | 245.32 | 245.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2023-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 14:15:00 | 253.95 | 247.04 | 246.51 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 242.80 | 247.54 | 248.04 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-08-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 12:15:00 | 252.15 | 248.65 | 248.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 11:15:00 | 258.00 | 252.75 | 250.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 09:15:00 | 267.00 | 271.18 | 264.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-08 09:45:00 | 270.00 | 271.18 | 264.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 15:15:00 | 272.75 | 272.43 | 268.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-09 09:15:00 | 285.40 | 272.43 | 268.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-08-11 09:15:00 | 313.94 | 299.79 | 290.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 15:15:00 | 344.70 | 347.89 | 348.14 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-08-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 09:15:00 | 352.10 | 348.73 | 348.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-25 09:15:00 | 358.65 | 351.93 | 350.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-25 11:15:00 | 352.00 | 352.59 | 351.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 11:15:00 | 352.00 | 352.59 | 351.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 11:15:00 | 352.00 | 352.59 | 351.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 12:00:00 | 352.00 | 352.59 | 351.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 12:15:00 | 351.95 | 352.46 | 351.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 12:30:00 | 353.65 | 352.46 | 351.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 13:15:00 | 353.45 | 352.66 | 351.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-28 09:15:00 | 362.00 | 352.26 | 351.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-28 14:15:00 | 346.95 | 355.24 | 354.06 | SL hit (close<static) qty=1.00 sl=349.25 alert=retest2 |

### Cycle 23 — SELL (started 2023-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 10:15:00 | 348.25 | 353.34 | 353.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-29 11:15:00 | 337.00 | 350.07 | 351.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-31 09:15:00 | 333.00 | 325.20 | 332.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 09:15:00 | 333.00 | 325.20 | 332.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 333.00 | 325.20 | 332.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-31 10:00:00 | 333.00 | 325.20 | 332.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 333.40 | 326.84 | 332.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-31 10:30:00 | 333.40 | 326.84 | 332.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 14:15:00 | 323.90 | 317.62 | 321.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 14:45:00 | 323.70 | 317.62 | 321.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 15:15:00 | 322.95 | 318.69 | 321.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-05 09:15:00 | 324.85 | 318.69 | 321.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2023-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 10:15:00 | 333.95 | 323.98 | 323.37 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 12:15:00 | 317.95 | 323.01 | 323.52 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-09-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 10:15:00 | 325.00 | 323.56 | 323.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 15:15:00 | 335.00 | 328.40 | 326.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-11 11:15:00 | 337.90 | 338.49 | 334.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 11:15:00 | 337.90 | 338.49 | 334.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 11:15:00 | 337.90 | 338.49 | 334.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 11:30:00 | 335.15 | 338.49 | 334.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 331.65 | 338.55 | 336.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:00:00 | 331.65 | 338.55 | 336.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 335.05 | 337.85 | 335.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-12 11:30:00 | 335.50 | 337.01 | 335.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-12 14:30:00 | 341.70 | 336.06 | 335.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-13 09:15:00 | 328.95 | 334.79 | 335.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2023-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-13 09:15:00 | 328.95 | 334.79 | 335.06 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 13:15:00 | 344.00 | 335.16 | 334.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-13 14:15:00 | 347.80 | 337.69 | 336.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-20 11:15:00 | 396.80 | 398.96 | 386.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 11:15:00 | 396.80 | 398.96 | 386.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 11:15:00 | 396.80 | 398.96 | 386.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 12:00:00 | 396.80 | 398.96 | 386.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 09:15:00 | 389.80 | 401.49 | 392.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 09:45:00 | 389.80 | 401.49 | 392.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 10:15:00 | 397.75 | 400.74 | 393.21 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 09:15:00 | 377.35 | 390.34 | 390.73 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 14:15:00 | 398.75 | 391.70 | 390.89 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 09:15:00 | 388.00 | 391.83 | 392.15 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 11:15:00 | 403.05 | 393.71 | 392.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 12:15:00 | 410.60 | 397.09 | 394.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 09:15:00 | 420.55 | 436.70 | 432.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 420.55 | 436.70 | 432.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 420.55 | 436.70 | 432.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 09:30:00 | 420.55 | 436.70 | 432.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 10:15:00 | 423.00 | 433.96 | 431.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 10:30:00 | 425.70 | 433.96 | 431.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2023-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 13:15:00 | 422.10 | 429.71 | 429.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 12:15:00 | 415.00 | 425.24 | 427.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-04 15:15:00 | 432.00 | 425.75 | 427.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 15:15:00 | 432.00 | 425.75 | 427.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 15:15:00 | 432.00 | 425.75 | 427.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 09:15:00 | 428.75 | 425.75 | 427.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 427.00 | 426.00 | 427.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 12:00:00 | 418.00 | 424.07 | 426.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 12:30:00 | 420.85 | 423.36 | 425.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:15:00 | 397.10 | 407.92 | 414.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:15:00 | 399.81 | 407.92 | 414.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-10 10:15:00 | 401.95 | 397.36 | 403.89 | SL hit (close>ema200) qty=0.50 sl=397.36 alert=retest2 |

### Cycle 34 — BUY (started 2023-10-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 15:15:00 | 410.00 | 403.85 | 403.47 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-10-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 09:15:00 | 400.00 | 403.08 | 403.15 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-10-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 10:15:00 | 406.90 | 403.85 | 403.49 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 10:15:00 | 397.95 | 402.82 | 403.36 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 09:15:00 | 414.15 | 403.19 | 403.00 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 11:15:00 | 399.95 | 405.13 | 405.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-17 14:15:00 | 395.00 | 402.21 | 403.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 09:15:00 | 401.75 | 401.29 | 402.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-18 09:30:00 | 402.00 | 401.29 | 402.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 13:15:00 | 393.20 | 386.76 | 391.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 14:00:00 | 393.20 | 386.76 | 391.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 14:15:00 | 394.00 | 388.21 | 391.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 15:00:00 | 394.00 | 388.21 | 391.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 15:15:00 | 400.00 | 390.56 | 392.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 09:15:00 | 404.00 | 390.56 | 392.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 10:15:00 | 397.90 | 393.39 | 393.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 11:45:00 | 387.00 | 393.44 | 393.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 12:45:00 | 382.65 | 392.77 | 393.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 13:45:00 | 384.00 | 391.22 | 392.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 09:15:00 | 387.55 | 390.76 | 392.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 09:15:00 | 378.00 | 388.20 | 390.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 11:30:00 | 373.85 | 384.41 | 388.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 15:15:00 | 373.35 | 380.36 | 385.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 09:15:00 | 367.65 | 374.77 | 382.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 09:15:00 | 363.52 | 374.77 | 382.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 09:15:00 | 364.80 | 374.77 | 382.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 09:15:00 | 368.17 | 374.77 | 382.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 355.16 | 362.20 | 370.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 354.68 | 362.20 | 370.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-26 12:15:00 | 365.00 | 360.88 | 368.00 | SL hit (close>ema200) qty=0.50 sl=360.88 alert=retest2 |

### Cycle 40 — BUY (started 2023-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 15:15:00 | 372.85 | 369.44 | 369.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 09:15:00 | 379.00 | 371.35 | 370.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 13:15:00 | 370.10 | 371.91 | 370.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-30 13:15:00 | 370.10 | 371.91 | 370.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 13:15:00 | 370.10 | 371.91 | 370.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-30 13:45:00 | 369.80 | 371.91 | 370.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 14:15:00 | 376.00 | 372.73 | 371.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-31 09:15:00 | 379.80 | 373.28 | 371.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-31 13:45:00 | 380.10 | 374.64 | 373.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-31 14:15:00 | 395.25 | 374.64 | 373.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-08 10:15:00 | 385.95 | 392.50 | 392.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2023-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 10:15:00 | 385.95 | 392.50 | 392.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 10:15:00 | 380.00 | 385.07 | 386.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-13 14:15:00 | 385.00 | 383.28 | 384.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-13 14:15:00 | 385.00 | 383.28 | 384.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 14:15:00 | 385.00 | 383.28 | 384.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-13 15:00:00 | 385.00 | 383.28 | 384.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 15:15:00 | 385.10 | 383.65 | 384.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 09:15:00 | 390.45 | 383.65 | 384.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 10:15:00 | 385.95 | 384.76 | 385.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-15 11:45:00 | 380.75 | 384.81 | 385.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-15 13:15:00 | 388.00 | 384.88 | 385.13 | SL hit (close>static) qty=1.00 sl=387.75 alert=retest2 |

### Cycle 42 — BUY (started 2023-11-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 15:15:00 | 390.00 | 385.89 | 385.55 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-11-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 09:15:00 | 381.20 | 384.95 | 385.15 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 12:15:00 | 399.00 | 386.55 | 385.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-17 14:15:00 | 402.20 | 391.19 | 387.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 15:15:00 | 396.00 | 396.85 | 393.15 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 09:15:00 | 400.25 | 396.85 | 393.15 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-22 09:15:00 | 420.26 | 406.09 | 400.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-11-22 10:15:00 | 403.15 | 405.50 | 400.76 | SL hit (close<ema200) qty=0.50 sl=405.50 alert=retest1 |

### Cycle 45 — SELL (started 2023-11-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 12:15:00 | 396.10 | 400.42 | 400.45 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-11-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 14:15:00 | 406.50 | 400.96 | 400.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 13:15:00 | 415.75 | 407.69 | 404.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 10:15:00 | 418.00 | 420.07 | 415.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-29 11:00:00 | 418.00 | 420.07 | 415.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 11:15:00 | 425.85 | 421.23 | 416.21 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2023-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-01 13:15:00 | 415.05 | 418.11 | 418.39 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2023-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 14:15:00 | 422.00 | 418.89 | 418.72 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2023-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-01 15:15:00 | 415.00 | 418.11 | 418.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-04 09:15:00 | 411.85 | 416.86 | 417.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-04 12:15:00 | 417.15 | 415.33 | 416.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-04 12:15:00 | 417.15 | 415.33 | 416.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 12:15:00 | 417.15 | 415.33 | 416.73 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2023-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 15:15:00 | 422.45 | 417.46 | 417.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-06 14:15:00 | 425.65 | 421.93 | 420.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-06 15:15:00 | 421.00 | 421.74 | 420.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-06 15:15:00 | 421.00 | 421.74 | 420.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 15:15:00 | 421.00 | 421.74 | 420.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-07 09:15:00 | 416.60 | 421.74 | 420.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 416.25 | 420.64 | 420.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-07 09:30:00 | 419.50 | 420.64 | 420.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2023-12-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 10:15:00 | 414.10 | 419.34 | 419.62 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2023-12-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 12:15:00 | 426.35 | 421.01 | 420.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 15:15:00 | 428.10 | 423.75 | 421.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 15:15:00 | 431.00 | 431.65 | 428.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 15:15:00 | 431.00 | 431.65 | 428.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 15:15:00 | 431.00 | 431.65 | 428.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-11 09:15:00 | 426.00 | 431.65 | 428.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 427.00 | 430.72 | 427.91 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2023-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 11:15:00 | 419.90 | 428.43 | 428.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 13:15:00 | 414.40 | 423.98 | 426.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 14:15:00 | 430.00 | 425.18 | 426.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 14:15:00 | 430.00 | 425.18 | 426.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 14:15:00 | 430.00 | 425.18 | 426.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 15:00:00 | 430.00 | 425.18 | 426.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 15:15:00 | 430.00 | 426.15 | 427.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-13 09:15:00 | 425.00 | 426.15 | 427.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-13 10:15:00 | 432.90 | 427.56 | 427.60 | SL hit (close>static) qty=1.00 sl=431.50 alert=retest2 |

### Cycle 54 — BUY (started 2023-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 11:15:00 | 435.05 | 429.06 | 428.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 12:15:00 | 441.95 | 431.64 | 429.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 09:15:00 | 466.95 | 475.87 | 467.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 09:15:00 | 466.95 | 475.87 | 467.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 466.95 | 475.87 | 467.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 09:30:00 | 460.15 | 475.87 | 467.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 10:15:00 | 473.05 | 475.30 | 467.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 13:30:00 | 478.00 | 472.64 | 468.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 15:15:00 | 478.00 | 471.32 | 468.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-19 11:15:00 | 459.05 | 468.34 | 467.77 | SL hit (close<static) qty=1.00 sl=465.20 alert=retest2 |

### Cycle 55 — SELL (started 2023-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 13:15:00 | 460.00 | 466.93 | 467.24 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2023-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 14:15:00 | 470.10 | 467.56 | 467.50 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 09:15:00 | 464.00 | 467.40 | 467.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 13:15:00 | 446.55 | 459.90 | 463.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 11:15:00 | 452.00 | 450.94 | 457.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 12:00:00 | 452.00 | 450.94 | 457.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 13:15:00 | 459.90 | 452.85 | 456.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 13:30:00 | 462.00 | 452.85 | 456.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 14:15:00 | 453.00 | 452.88 | 456.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 14:45:00 | 454.00 | 452.88 | 456.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 15:15:00 | 454.90 | 453.28 | 456.39 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2023-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 15:15:00 | 471.00 | 460.15 | 458.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 09:15:00 | 479.05 | 468.72 | 464.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 09:15:00 | 470.00 | 474.74 | 470.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 09:15:00 | 470.00 | 474.74 | 470.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 09:15:00 | 470.00 | 474.74 | 470.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 10:00:00 | 470.00 | 474.74 | 470.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 10:15:00 | 477.90 | 475.37 | 471.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-28 13:15:00 | 480.00 | 476.08 | 472.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 09:15:00 | 494.00 | 477.53 | 474.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-01 09:15:00 | 528.00 | 503.61 | 491.51 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2024-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-19 11:15:00 | 637.00 | 641.52 | 641.89 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-01-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 13:15:00 | 647.00 | 642.81 | 642.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 14:15:00 | 651.05 | 644.46 | 643.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 14:15:00 | 646.15 | 651.17 | 648.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 14:15:00 | 646.15 | 651.17 | 648.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 14:15:00 | 646.15 | 651.17 | 648.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 15:00:00 | 646.15 | 651.17 | 648.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 15:15:00 | 642.25 | 649.39 | 647.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:15:00 | 641.00 | 649.39 | 647.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 651.00 | 649.71 | 647.81 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 13:15:00 | 635.00 | 647.42 | 647.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 10:15:00 | 632.00 | 637.61 | 640.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-29 11:15:00 | 630.00 | 629.36 | 633.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-29 11:15:00 | 630.00 | 629.36 | 633.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 11:15:00 | 630.00 | 629.36 | 633.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 11:45:00 | 630.00 | 629.36 | 633.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 12:15:00 | 630.00 | 629.49 | 633.44 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 11:15:00 | 641.50 | 635.72 | 635.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-30 14:15:00 | 645.10 | 639.16 | 636.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 15:15:00 | 685.05 | 688.94 | 675.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-02 09:15:00 | 693.00 | 688.94 | 675.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 10:15:00 | 690.00 | 698.05 | 688.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 10:30:00 | 680.00 | 698.05 | 688.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 12:15:00 | 703.90 | 698.73 | 690.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 12:45:00 | 703.95 | 698.73 | 690.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 14:15:00 | 698.75 | 698.94 | 692.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 14:45:00 | 692.95 | 698.94 | 692.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 15:15:00 | 688.50 | 696.85 | 691.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-06 09:15:00 | 736.10 | 696.85 | 691.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-08 09:15:00 | 809.71 | 772.75 | 748.52 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 10:15:00 | 805.00 | 817.94 | 818.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-16 12:15:00 | 802.00 | 812.53 | 815.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-19 15:15:00 | 797.00 | 782.53 | 792.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 15:15:00 | 797.00 | 782.53 | 792.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 15:15:00 | 797.00 | 782.53 | 792.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-20 11:45:00 | 760.60 | 778.86 | 788.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-22 10:15:00 | 792.00 | 786.06 | 785.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2024-02-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 10:15:00 | 792.00 | 786.06 | 785.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-22 11:15:00 | 803.50 | 789.55 | 787.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-22 12:15:00 | 786.00 | 788.84 | 787.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 12:15:00 | 786.00 | 788.84 | 787.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 12:15:00 | 786.00 | 788.84 | 787.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 13:00:00 | 786.00 | 788.84 | 787.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 13:15:00 | 787.00 | 788.47 | 787.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-23 14:45:00 | 829.15 | 809.13 | 799.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 09:15:00 | 850.00 | 809.30 | 800.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-28 09:15:00 | 912.07 | 885.06 | 862.99 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 11:15:00 | 931.75 | 938.64 | 938.93 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-05 13:15:00 | 967.00 | 943.21 | 940.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-05 15:15:00 | 972.00 | 952.61 | 945.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 950.00 | 952.09 | 946.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 09:15:00 | 950.00 | 952.09 | 946.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 950.00 | 952.09 | 946.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 950.00 | 952.09 | 946.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 952.00 | 952.07 | 946.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:30:00 | 948.50 | 952.07 | 946.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 11:15:00 | 925.65 | 946.79 | 944.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 12:00:00 | 925.65 | 946.79 | 944.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 12:15:00 | 932.00 | 943.83 | 943.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 12:45:00 | 924.15 | 943.83 | 943.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2024-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 13:15:00 | 933.10 | 941.68 | 942.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 14:15:00 | 926.70 | 938.69 | 941.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 951.00 | 938.58 | 940.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 09:15:00 | 951.00 | 938.58 | 940.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 951.00 | 938.58 | 940.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 09:45:00 | 944.50 | 938.58 | 940.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 10:15:00 | 952.00 | 941.27 | 941.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 11:00:00 | 952.00 | 941.27 | 941.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2024-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 11:15:00 | 954.90 | 943.99 | 942.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-07 12:15:00 | 960.00 | 947.20 | 944.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-12 10:15:00 | 982.40 | 983.40 | 971.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-12 11:00:00 | 982.40 | 983.40 | 971.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 11:15:00 | 962.00 | 979.12 | 970.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 11:45:00 | 968.90 | 979.12 | 970.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 12:15:00 | 950.00 | 973.30 | 969.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 12:45:00 | 974.00 | 973.30 | 969.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 13:15:00 | 974.85 | 973.61 | 969.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 13:30:00 | 960.00 | 973.61 | 969.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 14:15:00 | 973.95 | 973.68 | 969.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 14:30:00 | 961.05 | 973.68 | 969.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 15:15:00 | 971.50 | 973.24 | 970.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 09:15:00 | 938.00 | 973.24 | 970.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2024-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 09:15:00 | 911.45 | 960.88 | 964.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 10:15:00 | 907.85 | 950.28 | 959.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-15 15:15:00 | 860.00 | 848.42 | 873.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 15:15:00 | 860.00 | 848.42 | 873.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 15:15:00 | 860.00 | 848.42 | 873.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 09:45:00 | 822.65 | 851.41 | 872.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 11:00:00 | 837.20 | 848.57 | 869.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 12:15:00 | 835.20 | 849.23 | 867.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 10:30:00 | 841.30 | 848.27 | 858.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 14:15:00 | 848.00 | 844.77 | 853.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-19 14:45:00 | 856.95 | 844.77 | 853.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 15:15:00 | 840.50 | 843.92 | 852.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-20 09:30:00 | 827.00 | 838.93 | 849.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 10:15:00 | 859.95 | 840.71 | 842.16 | SL hit (close>static) qty=1.00 sl=857.50 alert=retest2 |

### Cycle 70 — BUY (started 2024-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 11:15:00 | 862.30 | 845.03 | 843.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 12:15:00 | 869.40 | 849.90 | 846.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 11:15:00 | 868.95 | 875.37 | 868.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 11:15:00 | 868.95 | 875.37 | 868.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 11:15:00 | 868.95 | 875.37 | 868.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 11:45:00 | 865.00 | 875.37 | 868.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 12:15:00 | 879.00 | 876.09 | 869.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 12:30:00 | 868.95 | 876.09 | 869.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 15:15:00 | 890.00 | 881.33 | 873.38 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 14:15:00 | 829.20 | 867.30 | 869.60 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 15:15:00 | 875.00 | 863.27 | 861.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 13:15:00 | 883.90 | 873.34 | 867.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 14:15:00 | 962.00 | 970.89 | 951.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-05 14:45:00 | 960.00 | 970.89 | 951.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 988.00 | 976.71 | 957.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 09:30:00 | 988.50 | 976.71 | 957.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 10:15:00 | 918.30 | 965.03 | 954.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 10:30:00 | 918.30 | 965.03 | 954.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 11:15:00 | 949.15 | 961.85 | 953.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 12:15:00 | 973.00 | 961.85 | 953.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 14:15:00 | 979.25 | 966.24 | 957.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 15:00:00 | 978.25 | 968.64 | 959.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-09 14:15:00 | 947.00 | 956.26 | 956.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 14:15:00 | 947.00 | 956.26 | 956.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 09:15:00 | 922.90 | 950.03 | 953.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 13:15:00 | 905.90 | 904.49 | 915.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-15 13:45:00 | 905.95 | 904.49 | 915.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 910.00 | 903.66 | 912.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:15:00 | 905.00 | 903.66 | 912.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 920.00 | 906.93 | 913.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:45:00 | 906.35 | 906.93 | 913.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 915.40 | 908.62 | 913.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:45:00 | 918.75 | 908.62 | 913.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 12:15:00 | 910.00 | 908.90 | 912.94 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2024-04-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 10:15:00 | 933.70 | 917.80 | 915.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 12:15:00 | 936.00 | 922.91 | 918.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 11:15:00 | 925.00 | 926.46 | 922.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-19 11:45:00 | 927.50 | 926.46 | 922.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 1079.95 | 1100.48 | 1073.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 09:30:00 | 1099.00 | 1100.48 | 1073.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 13:15:00 | 1083.20 | 1090.40 | 1076.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 13:45:00 | 1078.00 | 1090.40 | 1076.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 14:15:00 | 1090.00 | 1090.32 | 1077.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 14:45:00 | 1070.05 | 1090.32 | 1077.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 1104.10 | 1093.50 | 1081.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 12:00:00 | 1129.50 | 1104.14 | 1088.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 14:15:00 | 1124.70 | 1111.82 | 1095.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 14:45:00 | 1130.00 | 1116.01 | 1098.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 14:00:00 | 1131.95 | 1145.64 | 1144.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 15:15:00 | 1139.00 | 1142.61 | 1143.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-05-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 15:15:00 | 1139.00 | 1142.61 | 1143.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 09:15:00 | 1090.00 | 1132.09 | 1138.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 09:15:00 | 1107.05 | 1105.27 | 1117.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-07 10:00:00 | 1107.05 | 1105.27 | 1117.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 1049.50 | 1066.21 | 1084.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 15:15:00 | 1038.35 | 1062.82 | 1081.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 09:15:00 | 986.43 | 1012.06 | 1040.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-10 14:15:00 | 1000.00 | 991.17 | 1017.24 | SL hit (close>ema200) qty=0.50 sl=991.17 alert=retest2 |

### Cycle 76 — BUY (started 2024-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 14:15:00 | 1043.95 | 1024.75 | 1023.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 09:15:00 | 1056.05 | 1033.45 | 1027.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 12:15:00 | 1289.00 | 1305.93 | 1257.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 13:00:00 | 1289.00 | 1305.93 | 1257.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 1335.00 | 1311.74 | 1264.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 13:30:00 | 1283.95 | 1311.74 | 1264.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 1369.80 | 1389.35 | 1364.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 13:30:00 | 1358.45 | 1389.35 | 1364.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 1359.60 | 1383.40 | 1363.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 15:00:00 | 1359.60 | 1383.40 | 1363.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 1358.45 | 1378.41 | 1363.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:15:00 | 1310.00 | 1378.41 | 1363.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1321.95 | 1367.12 | 1359.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 1315.65 | 1367.12 | 1359.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 12:15:00 | 1395.10 | 1372.06 | 1363.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 12:30:00 | 1397.90 | 1372.06 | 1363.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 1385.00 | 1374.65 | 1365.38 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 10:15:00 | 1339.50 | 1361.85 | 1362.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 11:15:00 | 1325.00 | 1340.12 | 1349.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 13:15:00 | 1276.45 | 1272.76 | 1293.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-30 13:45:00 | 1268.35 | 1272.76 | 1293.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 1296.50 | 1277.51 | 1293.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 15:00:00 | 1296.50 | 1277.51 | 1293.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 1262.55 | 1274.52 | 1291.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:30:00 | 1312.80 | 1278.81 | 1291.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 1321.00 | 1287.25 | 1294.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 11:00:00 | 1321.00 | 1287.25 | 1294.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2024-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 12:15:00 | 1331.95 | 1303.34 | 1300.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 14:15:00 | 1361.15 | 1323.17 | 1310.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 1347.75 | 1388.22 | 1365.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 1347.75 | 1388.22 | 1365.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1347.75 | 1388.22 | 1365.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 10:45:00 | 1347.75 | 1388.22 | 1365.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 1347.75 | 1380.13 | 1364.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:30:00 | 1347.75 | 1380.13 | 1364.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 09:15:00 | 1280.40 | 1344.89 | 1351.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-06 12:15:00 | 1260.05 | 1291.54 | 1311.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 15:15:00 | 1279.00 | 1276.49 | 1298.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-07 09:15:00 | 1300.00 | 1281.19 | 1298.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 1300.00 | 1281.19 | 1298.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:45:00 | 1306.00 | 1281.19 | 1298.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 1320.10 | 1288.97 | 1300.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 10:45:00 | 1320.10 | 1288.97 | 1300.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 11:15:00 | 1320.10 | 1295.20 | 1302.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 11:30:00 | 1320.10 | 1295.20 | 1302.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2024-06-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 14:15:00 | 1320.10 | 1307.35 | 1306.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 1386.10 | 1325.14 | 1314.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 15:15:00 | 1590.00 | 1590.68 | 1551.64 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:15:00 | 1624.15 | 1590.68 | 1551.64 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 1585.00 | 1594.82 | 1560.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:00:00 | 1585.00 | 1594.82 | 1560.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 14:15:00 | 1565.00 | 1578.99 | 1563.04 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-18 15:15:00 | 1560.00 | 1575.19 | 1562.76 | SL hit (close<ema400) qty=1.00 sl=1562.76 alert=retest1 |

### Cycle 81 — SELL (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 14:15:00 | 1550.00 | 1557.10 | 1557.80 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 10:15:00 | 1604.00 | 1566.47 | 1561.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 12:15:00 | 1621.00 | 1584.50 | 1571.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 14:15:00 | 1587.15 | 1590.81 | 1576.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-20 15:00:00 | 1587.15 | 1590.81 | 1576.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 1593.00 | 1591.24 | 1578.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 09:45:00 | 1596.30 | 1592.70 | 1580.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 11:15:00 | 1596.00 | 1592.68 | 1581.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 10:00:00 | 1597.85 | 1589.89 | 1585.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 11:15:00 | 1595.00 | 1589.93 | 1585.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 1575.00 | 1586.95 | 1584.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:30:00 | 1557.85 | 1586.95 | 1584.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 1574.40 | 1584.44 | 1583.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:30:00 | 1582.90 | 1584.44 | 1583.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-24 13:15:00 | 1561.40 | 1579.83 | 1581.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 13:15:00 | 1561.40 | 1579.83 | 1581.58 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 12:15:00 | 1597.85 | 1583.77 | 1582.73 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 14:15:00 | 1571.50 | 1580.73 | 1581.49 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 15:15:00 | 1599.00 | 1584.38 | 1583.08 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 10:15:00 | 1560.05 | 1578.30 | 1580.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 11:15:00 | 1549.95 | 1572.63 | 1577.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 11:15:00 | 1555.95 | 1546.30 | 1558.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 11:15:00 | 1555.95 | 1546.30 | 1558.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 1555.95 | 1546.30 | 1558.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:00:00 | 1555.95 | 1546.30 | 1558.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 1555.70 | 1548.18 | 1557.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 14:30:00 | 1524.95 | 1539.83 | 1552.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 09:15:00 | 1501.85 | 1541.21 | 1551.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-28 15:15:00 | 1448.70 | 1503.74 | 1523.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-28 15:15:00 | 1519.00 | 1503.74 | 1523.47 | SL hit (close>static) qty=0.50 sl=1503.74 alert=retest2 |

### Cycle 88 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 1557.50 | 1535.54 | 1534.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 1613.50 | 1563.12 | 1548.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 09:15:00 | 1668.00 | 1680.81 | 1651.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 09:15:00 | 1668.00 | 1680.81 | 1651.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 1668.00 | 1680.81 | 1651.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 11:00:00 | 1685.00 | 1681.65 | 1654.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 12:00:00 | 1685.60 | 1682.44 | 1657.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 10:00:00 | 1680.00 | 1686.14 | 1669.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 09:15:00 | 1693.40 | 1683.28 | 1682.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 1662.85 | 1679.20 | 1680.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 1662.85 | 1679.20 | 1680.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 11:15:00 | 1650.00 | 1672.88 | 1677.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 1692.60 | 1666.01 | 1671.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 1692.60 | 1666.01 | 1671.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 1692.60 | 1666.01 | 1671.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:00:00 | 1692.60 | 1666.01 | 1671.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 1674.70 | 1667.74 | 1671.37 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 09:15:00 | 1698.80 | 1676.89 | 1674.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 09:15:00 | 1720.80 | 1697.25 | 1687.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 10:15:00 | 1693.00 | 1696.40 | 1688.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 11:00:00 | 1693.00 | 1696.40 | 1688.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 1679.55 | 1693.03 | 1687.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 12:00:00 | 1679.55 | 1693.03 | 1687.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 1699.95 | 1694.42 | 1688.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 12:30:00 | 1674.35 | 1694.42 | 1688.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 1678.80 | 1691.29 | 1687.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 14:15:00 | 1672.00 | 1691.29 | 1687.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 1675.25 | 1688.08 | 1686.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 15:15:00 | 1680.00 | 1688.08 | 1686.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 1680.00 | 1686.47 | 1686.06 | EMA400 retest candle locked (from upside) |

### Cycle 91 — SELL (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 09:15:00 | 1660.80 | 1681.33 | 1683.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 12:15:00 | 1640.20 | 1666.52 | 1675.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 1467.65 | 1450.60 | 1478.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-24 09:30:00 | 1448.75 | 1450.60 | 1478.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 1462.70 | 1453.02 | 1477.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 14:00:00 | 1445.10 | 1456.17 | 1473.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 15:15:00 | 1445.60 | 1457.12 | 1471.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 10:15:00 | 1525.60 | 1464.45 | 1471.08 | SL hit (close>static) qty=1.00 sl=1479.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 11:15:00 | 1531.00 | 1477.76 | 1476.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 10:15:00 | 1574.05 | 1526.89 | 1504.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 1572.05 | 1572.79 | 1549.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:00:00 | 1572.05 | 1572.79 | 1549.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 1570.05 | 1572.15 | 1554.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:30:00 | 1572.95 | 1572.15 | 1554.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 1591.75 | 1576.07 | 1558.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:30:00 | 1579.95 | 1576.07 | 1558.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 1575.00 | 1578.07 | 1562.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:30:00 | 1570.00 | 1578.07 | 1562.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 1624.00 | 1588.84 | 1570.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 14:45:00 | 1584.00 | 1588.84 | 1570.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 1580.00 | 1592.86 | 1575.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:15:00 | 1592.00 | 1592.86 | 1575.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 1569.65 | 1588.22 | 1574.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:00:00 | 1569.65 | 1588.22 | 1574.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 1570.00 | 1584.57 | 1574.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:30:00 | 1563.00 | 1584.57 | 1574.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 12:15:00 | 1575.00 | 1582.66 | 1574.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 12:30:00 | 1569.00 | 1582.66 | 1574.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 1574.00 | 1580.93 | 1574.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:15:00 | 1566.95 | 1580.93 | 1574.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 1581.10 | 1580.96 | 1575.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:45:00 | 1567.50 | 1580.96 | 1575.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 1798.00 | 1740.88 | 1692.57 | EMA400 retest candle locked (from upside) |

### Cycle 93 — SELL (started 2024-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 09:15:00 | 1661.00 | 1684.47 | 1687.54 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 1716.00 | 1692.37 | 1690.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 12:15:00 | 1748.95 | 1703.69 | 1695.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 09:15:00 | 1723.00 | 1729.04 | 1712.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 09:15:00 | 1723.00 | 1729.04 | 1712.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 1723.00 | 1729.04 | 1712.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 11:15:00 | 1767.00 | 1731.72 | 1715.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 11:30:00 | 1759.45 | 1745.24 | 1732.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 13:45:00 | 1769.00 | 1751.71 | 1737.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 12:15:00 | 1713.65 | 1733.53 | 1734.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 12:15:00 | 1713.65 | 1733.53 | 1734.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 14:15:00 | 1690.00 | 1721.86 | 1728.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 09:15:00 | 1724.00 | 1718.78 | 1725.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 1724.00 | 1718.78 | 1725.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 1724.00 | 1718.78 | 1725.99 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 1749.90 | 1711.74 | 1709.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 1826.85 | 1749.94 | 1730.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 14:15:00 | 1809.00 | 1829.38 | 1800.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 14:15:00 | 1809.00 | 1829.38 | 1800.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 1809.00 | 1829.38 | 1800.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 15:00:00 | 1809.00 | 1829.38 | 1800.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1717.50 | 1803.90 | 1793.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:30:00 | 1723.90 | 1803.90 | 1793.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 10:15:00 | 1717.50 | 1786.62 | 1786.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 09:15:00 | 1656.70 | 1723.46 | 1751.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 09:15:00 | 1698.00 | 1668.96 | 1702.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 1698.00 | 1668.96 | 1702.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 1698.00 | 1668.96 | 1702.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:45:00 | 1705.05 | 1668.96 | 1702.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 1705.00 | 1676.17 | 1702.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:30:00 | 1700.00 | 1676.17 | 1702.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 1717.85 | 1684.50 | 1704.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:30:00 | 1719.35 | 1684.50 | 1704.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 1716.00 | 1690.80 | 1705.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:30:00 | 1717.70 | 1690.80 | 1705.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 1719.30 | 1696.50 | 1706.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:45:00 | 1719.25 | 1696.50 | 1706.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 1709.65 | 1699.13 | 1706.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:30:00 | 1719.35 | 1699.13 | 1706.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 1719.35 | 1703.18 | 1707.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 1700.00 | 1703.18 | 1707.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 1725.30 | 1708.20 | 1709.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:00:00 | 1725.30 | 1708.20 | 1709.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 1699.50 | 1706.46 | 1708.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:30:00 | 1721.50 | 1706.46 | 1708.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 1700.00 | 1705.17 | 1707.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 13:15:00 | 1690.00 | 1705.17 | 1707.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 10:30:00 | 1695.00 | 1691.36 | 1698.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 10:30:00 | 1695.00 | 1689.00 | 1692.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 14:15:00 | 1699.85 | 1695.17 | 1694.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 14:15:00 | 1699.85 | 1695.17 | 1694.73 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 09:15:00 | 1676.00 | 1691.31 | 1693.05 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 15:15:00 | 1732.15 | 1681.73 | 1678.30 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 14:15:00 | 1660.95 | 1684.53 | 1685.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 09:15:00 | 1644.90 | 1674.28 | 1680.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 1640.30 | 1639.62 | 1656.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 09:15:00 | 1640.30 | 1639.62 | 1656.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 1640.30 | 1639.62 | 1656.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:45:00 | 1649.50 | 1639.62 | 1656.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 1641.00 | 1635.49 | 1645.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 10:15:00 | 1630.00 | 1635.49 | 1645.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 14:15:00 | 1548.50 | 1613.30 | 1630.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-09 14:15:00 | 1625.00 | 1604.93 | 1615.93 | SL hit (close>ema200) qty=0.50 sl=1604.93 alert=retest2 |

### Cycle 102 — BUY (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 11:15:00 | 1665.85 | 1628.94 | 1624.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 12:15:00 | 1675.35 | 1638.23 | 1629.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 13:15:00 | 1729.20 | 1729.69 | 1704.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-12 13:45:00 | 1728.25 | 1729.69 | 1704.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 1700.00 | 1722.74 | 1707.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 10:00:00 | 1700.00 | 1722.74 | 1707.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 1697.00 | 1717.60 | 1706.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 10:45:00 | 1698.05 | 1717.60 | 1706.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 12:15:00 | 1690.60 | 1710.25 | 1705.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 13:00:00 | 1690.60 | 1710.25 | 1705.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 13:15:00 | 1715.60 | 1711.32 | 1705.97 | EMA400 retest candle locked (from upside) |

### Cycle 103 — SELL (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 10:15:00 | 1690.00 | 1701.96 | 1703.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 09:15:00 | 1655.20 | 1684.13 | 1692.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 14:15:00 | 1696.75 | 1675.75 | 1683.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 14:15:00 | 1696.75 | 1675.75 | 1683.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 1696.75 | 1675.75 | 1683.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 15:00:00 | 1696.75 | 1675.75 | 1683.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 1692.95 | 1679.19 | 1684.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:15:00 | 1670.00 | 1679.19 | 1684.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 1672.15 | 1675.42 | 1681.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:30:00 | 1687.60 | 1675.42 | 1681.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 1671.00 | 1672.70 | 1679.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 12:30:00 | 1689.00 | 1672.70 | 1679.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 1686.50 | 1671.83 | 1677.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 1686.50 | 1671.83 | 1677.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 1700.00 | 1677.46 | 1679.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 09:15:00 | 1606.85 | 1677.46 | 1679.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 09:15:00 | 1526.51 | 1620.31 | 1641.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-25 09:15:00 | 1553.55 | 1551.04 | 1564.86 | SL hit (close>ema200) qty=0.50 sl=1551.04 alert=retest2 |

### Cycle 104 — BUY (started 2024-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 12:15:00 | 1600.05 | 1574.96 | 1573.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 13:15:00 | 1620.70 | 1584.11 | 1577.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 1589.00 | 1598.36 | 1587.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 09:15:00 | 1589.00 | 1598.36 | 1587.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 1589.00 | 1598.36 | 1587.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 1578.55 | 1598.36 | 1587.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 1600.50 | 1598.79 | 1588.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 12:00:00 | 1615.35 | 1602.10 | 1590.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 13:30:00 | 1613.95 | 1606.76 | 1594.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 10:15:00 | 1626.00 | 1611.75 | 1600.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 13:15:00 | 1629.95 | 1651.67 | 1652.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 13:15:00 | 1629.95 | 1651.67 | 1652.91 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 09:15:00 | 1677.00 | 1654.65 | 1653.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 10:15:00 | 1689.00 | 1661.52 | 1656.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 12:15:00 | 1655.00 | 1661.57 | 1657.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 12:15:00 | 1655.00 | 1661.57 | 1657.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 1655.00 | 1661.57 | 1657.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:00:00 | 1655.00 | 1661.57 | 1657.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 1660.85 | 1661.43 | 1658.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:00:00 | 1660.85 | 1661.43 | 1658.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 1656.40 | 1660.42 | 1657.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:30:00 | 1660.00 | 1660.42 | 1657.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 1660.00 | 1660.34 | 1658.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:15:00 | 1686.00 | 1660.34 | 1658.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 1667.45 | 1661.76 | 1658.96 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 1596.50 | 1648.71 | 1653.28 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 1685.70 | 1644.12 | 1642.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 10:15:00 | 1711.55 | 1673.70 | 1657.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 15:15:00 | 1756.50 | 1763.23 | 1732.23 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:15:00 | 1786.55 | 1763.23 | 1732.23 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 1779.00 | 1766.39 | 1736.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:30:00 | 1794.80 | 1766.39 | 1736.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 1767.75 | 1771.80 | 1753.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 1812.00 | 1771.80 | 1753.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 11:15:00 | 1780.00 | 1775.17 | 1758.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 14:15:00 | 1875.88 | 1823.08 | 1798.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-10-17 09:15:00 | 1965.21 | 1911.17 | 1871.28 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 109 — SELL (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 11:15:00 | 1813.40 | 1864.38 | 1868.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 09:15:00 | 1790.00 | 1830.23 | 1848.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 1764.85 | 1746.90 | 1771.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 11:00:00 | 1764.85 | 1746.90 | 1771.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 1794.95 | 1756.51 | 1773.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:00:00 | 1794.95 | 1756.51 | 1773.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 1794.95 | 1764.20 | 1775.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:45:00 | 1798.00 | 1764.20 | 1775.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 14:15:00 | 1768.60 | 1767.61 | 1775.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 14:30:00 | 1790.00 | 1767.61 | 1775.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 1758.90 | 1760.69 | 1769.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 11:00:00 | 1758.90 | 1760.69 | 1769.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 11:15:00 | 1750.00 | 1758.55 | 1767.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 11:45:00 | 1771.20 | 1758.55 | 1767.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 1676.25 | 1655.87 | 1684.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 15:00:00 | 1676.25 | 1655.87 | 1684.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 15:15:00 | 1659.00 | 1656.50 | 1681.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 09:15:00 | 1635.85 | 1656.50 | 1681.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 13:00:00 | 1627.00 | 1639.84 | 1664.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-29 14:15:00 | 1689.85 | 1651.47 | 1665.93 | SL hit (close>static) qty=1.00 sl=1683.95 alert=retest2 |

### Cycle 110 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 1706.15 | 1675.17 | 1674.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 1720.15 | 1690.35 | 1681.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 09:15:00 | 1743.00 | 1772.68 | 1746.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 1743.00 | 1772.68 | 1746.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 1743.00 | 1772.68 | 1746.09 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 10:15:00 | 1707.80 | 1733.60 | 1735.46 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 14:15:00 | 1741.10 | 1737.14 | 1736.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 10:15:00 | 1777.00 | 1749.54 | 1742.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 11:15:00 | 1735.25 | 1757.05 | 1751.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 11:15:00 | 1735.25 | 1757.05 | 1751.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 1735.25 | 1757.05 | 1751.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:30:00 | 1750.00 | 1757.05 | 1751.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 12:15:00 | 1690.00 | 1743.64 | 1746.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 15:15:00 | 1675.00 | 1715.82 | 1731.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 09:15:00 | 1777.15 | 1728.09 | 1735.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 09:15:00 | 1777.15 | 1728.09 | 1735.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 1777.15 | 1728.09 | 1735.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:30:00 | 1777.15 | 1728.09 | 1735.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 11:15:00 | 1777.15 | 1745.75 | 1742.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-12 09:15:00 | 1866.00 | 1784.63 | 1763.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 1813.05 | 1838.35 | 1808.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 09:15:00 | 1813.05 | 1838.35 | 1808.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 1813.05 | 1838.35 | 1808.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:30:00 | 1816.00 | 1838.35 | 1808.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 1819.95 | 1834.67 | 1809.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 10:30:00 | 1819.95 | 1834.67 | 1809.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 11:15:00 | 1799.95 | 1827.72 | 1808.93 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2024-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 09:15:00 | 1779.95 | 1797.20 | 1799.41 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 14:15:00 | 1854.85 | 1785.55 | 1785.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 1933.00 | 1825.35 | 1804.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 09:15:00 | 1859.80 | 1886.44 | 1870.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 1859.80 | 1886.44 | 1870.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 1859.80 | 1886.44 | 1870.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:45:00 | 1866.95 | 1886.44 | 1870.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 1900.00 | 1889.15 | 1872.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 11:30:00 | 1905.50 | 1890.13 | 1874.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 09:15:00 | 1918.00 | 1885.79 | 1877.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 11:45:00 | 1909.10 | 1901.88 | 1887.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 09:15:00 | 1823.50 | 1890.16 | 1888.54 | SL hit (close<static) qty=1.00 sl=1859.80 alert=retest2 |

### Cycle 117 — SELL (started 2024-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 10:15:00 | 1823.50 | 1876.83 | 1882.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 09:15:00 | 1738.00 | 1820.38 | 1849.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 14:15:00 | 1799.75 | 1797.98 | 1824.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-27 15:00:00 | 1799.75 | 1797.98 | 1824.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 1765.65 | 1752.10 | 1761.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:00:00 | 1765.65 | 1752.10 | 1761.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 1774.00 | 1756.48 | 1762.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:45:00 | 1787.95 | 1763.16 | 1765.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 10:15:00 | 1800.00 | 1770.53 | 1768.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 11:15:00 | 1810.90 | 1778.60 | 1772.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 1859.95 | 1861.72 | 1835.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 09:45:00 | 1854.50 | 1861.72 | 1835.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 1904.00 | 1890.30 | 1865.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 11:15:00 | 1918.90 | 1897.86 | 1883.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 11:00:00 | 1924.00 | 1930.06 | 1921.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 10:15:00 | 1909.25 | 1926.05 | 1928.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 1909.25 | 1926.05 | 1928.01 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 15:15:00 | 1935.00 | 1929.37 | 1929.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 09:15:00 | 1980.00 | 1939.50 | 1933.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 09:15:00 | 2034.00 | 2056.49 | 2024.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-18 10:00:00 | 2034.00 | 2056.49 | 2024.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 2049.95 | 2055.18 | 2026.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 11:15:00 | 2048.80 | 2055.18 | 2026.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 2050.00 | 2054.15 | 2029.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 09:30:00 | 2062.65 | 2052.16 | 2037.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 11:45:00 | 2061.95 | 2053.82 | 2040.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 12:45:00 | 2061.50 | 2055.34 | 2042.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 13:45:00 | 2060.15 | 2056.09 | 2043.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 2134.95 | 2071.87 | 2052.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:30:00 | 2058.00 | 2071.87 | 2052.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 2100.00 | 2128.63 | 2102.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:30:00 | 2111.00 | 2128.63 | 2102.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 2109.95 | 2124.90 | 2103.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 10:30:00 | 2100.50 | 2124.90 | 2103.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 2099.65 | 2119.85 | 2103.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:45:00 | 2093.90 | 2119.85 | 2103.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 2113.20 | 2118.52 | 2103.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 13:30:00 | 2127.30 | 2121.01 | 2106.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 09:15:00 | 2094.00 | 2111.22 | 2105.15 | SL hit (close<static) qty=1.00 sl=2099.65 alert=retest2 |

### Cycle 121 — SELL (started 2024-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 12:15:00 | 2080.05 | 2097.59 | 2099.71 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 14:15:00 | 2088.40 | 2063.79 | 2062.01 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 09:15:00 | 2018.00 | 2060.75 | 2063.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 09:15:00 | 2000.60 | 2032.63 | 2045.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 11:15:00 | 2030.60 | 2027.54 | 2040.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-03 12:00:00 | 2030.60 | 2027.54 | 2040.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 2039.00 | 2025.62 | 2037.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:45:00 | 2043.50 | 2025.62 | 2037.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 2039.05 | 2028.30 | 2037.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:30:00 | 2049.45 | 2028.30 | 2037.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 2039.50 | 2030.54 | 2037.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 2039.95 | 2030.54 | 2037.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 2087.50 | 2041.93 | 2042.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 2087.50 | 2041.93 | 2042.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-06 10:15:00 | 2065.00 | 2046.55 | 2044.38 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 2006.90 | 2042.95 | 2043.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 11:15:00 | 1990.05 | 2016.40 | 2028.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 09:15:00 | 2016.00 | 2010.70 | 2020.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 09:15:00 | 2016.00 | 2010.70 | 2020.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 2016.00 | 2010.70 | 2020.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 1967.95 | 1994.66 | 2001.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 12:30:00 | 1970.80 | 1972.79 | 1986.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 15:15:00 | 1951.00 | 1971.37 | 1983.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-14 09:15:00 | 1869.55 | 1907.32 | 1935.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-14 09:15:00 | 1872.26 | 1907.32 | 1935.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-14 09:15:00 | 1853.45 | 1907.32 | 1935.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 14:15:00 | 1886.85 | 1882.50 | 1910.89 | SL hit (close>ema200) qty=0.50 sl=1882.50 alert=retest2 |

### Cycle 126 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 1925.40 | 1910.84 | 1909.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 13:15:00 | 1980.00 | 1926.22 | 1917.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 11:15:00 | 1946.50 | 1951.73 | 1935.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 12:00:00 | 1946.50 | 1951.73 | 1935.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 1930.10 | 1945.85 | 1935.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:00:00 | 1930.10 | 1945.85 | 1935.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 1945.90 | 1945.86 | 1936.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:30:00 | 1931.35 | 1945.86 | 1936.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 1932.10 | 1943.11 | 1936.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 1900.05 | 1943.11 | 1936.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 1919.90 | 1938.46 | 1934.71 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 10:15:00 | 1900.00 | 1930.77 | 1931.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 11:15:00 | 1891.05 | 1922.83 | 1927.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1787.00 | 1779.04 | 1819.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 1787.00 | 1779.04 | 1819.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1834.45 | 1790.12 | 1820.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 1834.45 | 1790.12 | 1820.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 1833.80 | 1798.86 | 1822.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 12:30:00 | 1800.65 | 1799.09 | 1820.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:30:00 | 1799.10 | 1798.08 | 1813.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 1710.62 | 1757.01 | 1783.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 1709.14 | 1757.01 | 1783.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 09:15:00 | 1620.59 | 1675.68 | 1723.36 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 128 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 1709.90 | 1663.77 | 1659.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 1760.30 | 1706.37 | 1683.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 1724.90 | 1763.68 | 1738.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 1724.90 | 1763.68 | 1738.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1724.90 | 1763.68 | 1738.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:45:00 | 1724.25 | 1763.68 | 1738.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1698.15 | 1750.57 | 1735.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:00:00 | 1698.15 | 1750.57 | 1735.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 1656.15 | 1719.84 | 1723.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 1640.30 | 1692.81 | 1710.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 14:15:00 | 1646.00 | 1632.60 | 1659.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 15:00:00 | 1646.00 | 1632.60 | 1659.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 1697.00 | 1647.94 | 1661.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:45:00 | 1705.00 | 1647.94 | 1661.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 1729.25 | 1664.20 | 1667.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:30:00 | 1729.25 | 1664.20 | 1667.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 11:15:00 | 1729.25 | 1677.21 | 1673.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 10:15:00 | 1736.30 | 1703.83 | 1690.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 15:15:00 | 1705.00 | 1716.24 | 1703.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-07 09:15:00 | 1697.75 | 1716.24 | 1703.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1702.00 | 1713.39 | 1703.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:30:00 | 1682.80 | 1713.39 | 1703.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1704.60 | 1711.64 | 1703.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:30:00 | 1702.20 | 1711.64 | 1703.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 1695.00 | 1708.31 | 1702.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 12:00:00 | 1695.00 | 1708.31 | 1702.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 1695.00 | 1705.65 | 1701.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 12:30:00 | 1695.35 | 1705.65 | 1701.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 1695.10 | 1701.02 | 1700.45 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 1655.10 | 1691.83 | 1696.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 12:15:00 | 1613.25 | 1650.22 | 1666.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 1563.75 | 1559.09 | 1593.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 1563.75 | 1559.09 | 1593.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1563.75 | 1559.09 | 1593.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 1563.75 | 1559.09 | 1593.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 1475.00 | 1541.46 | 1569.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 09:15:00 | 1414.95 | 1492.10 | 1528.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 09:15:00 | 1344.20 | 1378.85 | 1403.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-20 10:15:00 | 1387.85 | 1380.65 | 1402.29 | SL hit (close>ema200) qty=0.50 sl=1380.65 alert=retest2 |

### Cycle 132 — BUY (started 2025-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 15:15:00 | 1426.65 | 1411.91 | 1411.20 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 09:15:00 | 1382.65 | 1406.06 | 1408.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 10:15:00 | 1380.75 | 1401.00 | 1406.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 13:15:00 | 1399.95 | 1397.05 | 1402.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-21 14:15:00 | 1394.50 | 1397.05 | 1402.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 1438.40 | 1405.32 | 1405.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 15:00:00 | 1438.40 | 1405.32 | 1405.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 1399.70 | 1404.20 | 1405.30 | EMA400 retest candle locked (from downside) |

### Cycle 134 — BUY (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 09:15:00 | 1417.95 | 1406.95 | 1406.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-24 10:15:00 | 1468.55 | 1419.27 | 1412.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 15:15:00 | 1436.00 | 1438.80 | 1426.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-25 09:15:00 | 1429.15 | 1438.80 | 1426.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 1444.60 | 1439.96 | 1428.29 | EMA400 retest candle locked (from upside) |

### Cycle 135 — SELL (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 10:15:00 | 1399.15 | 1422.70 | 1425.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 11:15:00 | 1390.30 | 1416.22 | 1422.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 1314.45 | 1313.64 | 1341.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 14:45:00 | 1316.55 | 1313.64 | 1341.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 1348.95 | 1318.52 | 1338.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 1348.95 | 1318.52 | 1338.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 1362.00 | 1327.21 | 1341.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:45:00 | 1377.40 | 1327.21 | 1341.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 14:15:00 | 1362.55 | 1348.14 | 1347.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 1401.00 | 1360.29 | 1353.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 1371.30 | 1380.55 | 1371.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 11:15:00 | 1371.30 | 1380.55 | 1371.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 11:15:00 | 1371.30 | 1380.55 | 1371.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 12:00:00 | 1371.30 | 1380.55 | 1371.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 12:15:00 | 1376.15 | 1379.67 | 1372.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 14:45:00 | 1391.40 | 1381.73 | 1374.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:15:00 | 1394.95 | 1383.58 | 1376.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 10:15:00 | 1427.40 | 1432.51 | 1433.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 10:15:00 | 1427.40 | 1432.51 | 1433.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 12:15:00 | 1424.80 | 1430.39 | 1432.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 15:15:00 | 1428.00 | 1427.42 | 1430.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 15:15:00 | 1428.00 | 1427.42 | 1430.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 1428.00 | 1427.42 | 1430.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:15:00 | 1433.85 | 1427.42 | 1430.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 1455.65 | 1433.06 | 1432.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 10:15:00 | 1471.15 | 1440.68 | 1435.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 09:15:00 | 1505.85 | 1506.70 | 1485.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-19 13:15:00 | 1485.50 | 1503.93 | 1491.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 13:15:00 | 1485.50 | 1503.93 | 1491.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 14:00:00 | 1485.50 | 1503.93 | 1491.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 14:15:00 | 1505.15 | 1504.18 | 1492.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 09:15:00 | 1516.35 | 1505.10 | 1494.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-20 09:15:00 | 1481.40 | 1500.36 | 1492.94 | SL hit (close<static) qty=1.00 sl=1485.60 alert=retest2 |

### Cycle 139 — SELL (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 11:15:00 | 1547.00 | 1579.12 | 1579.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 1536.50 | 1565.95 | 1573.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 12:15:00 | 1583.35 | 1551.98 | 1560.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 12:15:00 | 1583.35 | 1551.98 | 1560.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 1583.35 | 1551.98 | 1560.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:45:00 | 1584.00 | 1551.98 | 1560.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 1538.25 | 1549.23 | 1558.55 | EMA400 retest candle locked (from downside) |

### Cycle 140 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 1605.00 | 1571.10 | 1567.54 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 09:15:00 | 1519.25 | 1559.89 | 1564.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 1509.55 | 1549.82 | 1559.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 1503.10 | 1500.97 | 1524.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 11:00:00 | 1503.10 | 1500.97 | 1524.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1411.35 | 1449.17 | 1474.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 13:30:00 | 1409.25 | 1429.91 | 1456.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 1338.79 | 1399.35 | 1435.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-08 15:15:00 | 1311.00 | 1309.33 | 1341.19 | SL hit (close>ema200) qty=0.50 sl=1309.33 alert=retest2 |

### Cycle 142 — BUY (started 2025-04-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 15:15:00 | 1348.00 | 1325.20 | 1323.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1378.40 | 1335.84 | 1328.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1407.80 | 1411.06 | 1384.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 09:30:00 | 1410.80 | 1411.06 | 1384.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 1520.00 | 1514.05 | 1499.64 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 1488.50 | 1497.68 | 1498.78 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 13:15:00 | 1521.50 | 1501.06 | 1499.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 14:15:00 | 1535.50 | 1507.94 | 1502.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 14:15:00 | 1570.20 | 1573.84 | 1554.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 14:30:00 | 1567.90 | 1573.84 | 1554.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 1548.50 | 1568.77 | 1554.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 1571.50 | 1568.77 | 1554.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 10:30:00 | 1572.00 | 1566.11 | 1555.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 09:15:00 | 1575.70 | 1559.43 | 1555.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 12:30:00 | 1586.00 | 1562.83 | 1558.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 1604.40 | 1574.34 | 1564.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:30:00 | 1576.60 | 1574.34 | 1564.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 1576.90 | 1578.46 | 1570.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:30:00 | 1563.50 | 1578.46 | 1570.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 1561.40 | 1575.05 | 1569.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 1561.40 | 1575.05 | 1569.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 1538.50 | 1567.74 | 1566.61 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-06 13:15:00 | 1538.50 | 1567.74 | 1566.61 | SL hit (close<static) qty=1.00 sl=1540.20 alert=retest2 |

### Cycle 145 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 1539.30 | 1562.05 | 1564.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 15:15:00 | 1526.00 | 1554.84 | 1560.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 1560.40 | 1546.53 | 1554.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 11:15:00 | 1560.40 | 1546.53 | 1554.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 1560.40 | 1546.53 | 1554.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:45:00 | 1562.20 | 1546.53 | 1554.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 1564.20 | 1550.07 | 1555.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 13:00:00 | 1564.20 | 1550.07 | 1555.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 1592.00 | 1558.45 | 1558.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 1592.00 | 1558.45 | 1558.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 14:15:00 | 1630.90 | 1572.94 | 1565.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-09 14:15:00 | 1656.30 | 1629.87 | 1613.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 09:15:00 | 1849.10 | 1852.09 | 1817.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 10:00:00 | 1849.10 | 1852.09 | 1817.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1845.00 | 1845.35 | 1830.63 | EMA400 retest candle locked (from upside) |

### Cycle 147 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 1794.00 | 1822.77 | 1824.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 15:15:00 | 1785.00 | 1815.22 | 1820.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 10:15:00 | 1845.00 | 1820.98 | 1822.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 10:15:00 | 1845.00 | 1820.98 | 1822.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1845.00 | 1820.98 | 1822.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 1845.00 | 1820.98 | 1822.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 1826.40 | 1822.06 | 1822.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 13:15:00 | 1823.90 | 1822.71 | 1823.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 14:15:00 | 1849.90 | 1828.45 | 1825.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 1849.90 | 1828.45 | 1825.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 15:15:00 | 1870.00 | 1843.37 | 1834.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 2260.30 | 2274.98 | 2226.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 14:15:00 | 2246.00 | 2271.15 | 2242.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 2246.00 | 2271.15 | 2242.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 2246.00 | 2271.15 | 2242.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 2234.80 | 2263.88 | 2242.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 2309.20 | 2263.88 | 2242.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 13:15:00 | 2332.40 | 2364.69 | 2367.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 13:15:00 | 2332.40 | 2364.69 | 2367.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 14:15:00 | 2314.90 | 2354.74 | 2362.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 15:15:00 | 2313.60 | 2311.22 | 2330.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-10 09:15:00 | 2320.40 | 2311.22 | 2330.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 2319.80 | 2312.93 | 2329.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:00:00 | 2319.80 | 2312.93 | 2329.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 2397.20 | 2323.23 | 2325.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 2370.70 | 2323.23 | 2325.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 10:15:00 | 2403.00 | 2339.18 | 2332.18 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 2290.50 | 2336.67 | 2341.04 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 2366.30 | 2319.90 | 2316.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 15:15:00 | 2387.00 | 2353.32 | 2336.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 2365.40 | 2367.75 | 2349.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 12:15:00 | 2365.40 | 2367.75 | 2349.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 2365.40 | 2367.75 | 2349.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:45:00 | 2350.00 | 2367.75 | 2349.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 2370.00 | 2368.20 | 2351.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 09:30:00 | 2392.40 | 2365.06 | 2354.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 15:15:00 | 2345.00 | 2354.60 | 2353.00 | SL hit (close<static) qty=1.00 sl=2347.60 alert=retest2 |

### Cycle 153 — SELL (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 09:15:00 | 2326.40 | 2348.96 | 2350.58 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 2408.80 | 2350.72 | 2346.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 13:15:00 | 2434.00 | 2382.42 | 2363.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 10:15:00 | 2398.30 | 2406.20 | 2383.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 11:00:00 | 2398.30 | 2406.20 | 2383.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 2395.10 | 2403.98 | 2384.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 11:30:00 | 2413.90 | 2403.98 | 2384.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 2382.60 | 2399.70 | 2383.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:45:00 | 2378.00 | 2399.70 | 2383.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 2387.00 | 2397.16 | 2384.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:45:00 | 2371.00 | 2397.16 | 2384.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 2394.90 | 2396.71 | 2385.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:15:00 | 2390.00 | 2396.71 | 2385.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 2390.00 | 2395.37 | 2385.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 2415.60 | 2395.37 | 2385.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 10:45:00 | 2412.50 | 2395.96 | 2387.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 11:15:00 | 2400.00 | 2395.96 | 2387.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 11:15:00 | 2375.50 | 2391.87 | 2386.50 | SL hit (close<static) qty=1.00 sl=2381.20 alert=retest2 |

### Cycle 155 — SELL (started 2025-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 13:15:00 | 2354.60 | 2381.65 | 2382.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 14:15:00 | 2347.60 | 2374.84 | 2379.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 10:15:00 | 2370.10 | 2368.87 | 2375.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 10:15:00 | 2370.10 | 2368.87 | 2375.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 2370.10 | 2368.87 | 2375.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:45:00 | 2362.70 | 2368.87 | 2375.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 2362.70 | 2367.64 | 2373.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:45:00 | 2370.20 | 2367.64 | 2373.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 2362.70 | 2354.14 | 2363.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:00:00 | 2362.70 | 2354.14 | 2363.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 2353.00 | 2353.91 | 2362.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 2349.50 | 2357.26 | 2361.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 12:45:00 | 2348.90 | 2358.84 | 2361.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 14:15:00 | 2348.70 | 2357.07 | 2360.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 2345.60 | 2354.21 | 2358.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 2355.00 | 2354.37 | 2357.87 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-02 11:15:00 | 2375.10 | 2361.35 | 2360.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 11:15:00 | 2375.10 | 2361.35 | 2360.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 14:15:00 | 2387.20 | 2370.18 | 2365.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 09:15:00 | 2353.40 | 2368.87 | 2365.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 2353.40 | 2368.87 | 2365.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 2353.40 | 2368.87 | 2365.53 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 2344.30 | 2360.53 | 2362.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 10:15:00 | 2332.60 | 2349.77 | 2355.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 12:15:00 | 2363.10 | 2347.10 | 2353.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 12:15:00 | 2363.10 | 2347.10 | 2353.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 2363.10 | 2347.10 | 2353.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:45:00 | 2357.60 | 2347.10 | 2353.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 2358.70 | 2349.42 | 2353.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:45:00 | 2370.00 | 2349.42 | 2353.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 2359.10 | 2351.35 | 2354.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 15:15:00 | 2345.20 | 2351.35 | 2354.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 10:30:00 | 2350.00 | 2351.35 | 2353.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:45:00 | 2349.40 | 2350.60 | 2353.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 2372.20 | 2350.31 | 2350.72 | SL hit (close>static) qty=1.00 sl=2364.60 alert=retest2 |

### Cycle 158 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 2383.90 | 2356.95 | 2353.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 2424.10 | 2381.47 | 2367.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 13:15:00 | 2387.30 | 2388.40 | 2375.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 13:45:00 | 2394.00 | 2388.40 | 2375.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 2372.00 | 2384.73 | 2377.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 2372.00 | 2384.73 | 2377.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 2400.00 | 2387.79 | 2379.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 12:15:00 | 2407.20 | 2390.93 | 2381.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 13:15:00 | 2367.00 | 2383.56 | 2379.63 | SL hit (close<static) qty=1.00 sl=2370.90 alert=retest2 |

### Cycle 159 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 2332.50 | 2373.31 | 2375.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 11:15:00 | 2320.30 | 2356.51 | 2367.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 2328.40 | 2299.84 | 2319.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 2328.40 | 2299.84 | 2319.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 2328.40 | 2299.84 | 2319.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 2322.80 | 2299.84 | 2319.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 2340.40 | 2307.95 | 2321.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 2340.40 | 2307.95 | 2321.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 2322.10 | 2310.78 | 2321.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:45:00 | 2332.90 | 2310.78 | 2321.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 2322.00 | 2313.02 | 2321.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:30:00 | 2325.40 | 2313.02 | 2321.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 2305.60 | 2311.54 | 2320.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:30:00 | 2319.40 | 2311.54 | 2320.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 2329.00 | 2311.39 | 2317.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:30:00 | 2319.20 | 2311.39 | 2317.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 2323.50 | 2313.81 | 2318.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:30:00 | 2325.50 | 2313.81 | 2318.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 2333.10 | 2313.36 | 2316.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:00:00 | 2333.10 | 2313.36 | 2316.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 14:15:00 | 2340.30 | 2318.75 | 2318.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 2360.60 | 2328.82 | 2323.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 2375.00 | 2388.87 | 2363.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 09:45:00 | 2392.20 | 2388.87 | 2363.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 2393.50 | 2389.79 | 2366.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:00:00 | 2407.90 | 2377.77 | 2368.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 12:15:00 | 2440.00 | 2474.00 | 2475.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 2440.00 | 2474.00 | 2475.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 14:15:00 | 2434.50 | 2460.66 | 2468.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 2470.00 | 2418.81 | 2430.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 12:15:00 | 2470.00 | 2418.81 | 2430.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 2470.00 | 2418.81 | 2430.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:30:00 | 2456.40 | 2418.81 | 2430.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 2476.80 | 2430.41 | 2434.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:45:00 | 2485.40 | 2430.41 | 2434.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 2475.20 | 2439.37 | 2437.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 2597.00 | 2476.76 | 2455.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 09:15:00 | 2795.50 | 2820.49 | 2742.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-04 10:00:00 | 2795.50 | 2820.49 | 2742.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 2759.80 | 2791.82 | 2747.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 15:15:00 | 2810.00 | 2782.87 | 2750.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 15:15:00 | 2787.90 | 2797.37 | 2798.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 15:15:00 | 2787.90 | 2797.37 | 2798.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 2751.90 | 2788.28 | 2794.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 11:15:00 | 2789.20 | 2786.85 | 2792.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 11:15:00 | 2789.20 | 2786.85 | 2792.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 2789.20 | 2786.85 | 2792.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 2789.20 | 2786.85 | 2792.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 2799.90 | 2789.46 | 2793.08 | EMA400 retest candle locked (from downside) |

### Cycle 164 — BUY (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 13:15:00 | 2822.10 | 2795.99 | 2795.72 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 2790.00 | 2795.43 | 2795.55 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 2815.40 | 2799.43 | 2797.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 2914.60 | 2821.96 | 2809.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 2848.90 | 2885.70 | 2857.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 2848.90 | 2885.70 | 2857.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 2848.90 | 2885.70 | 2857.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 2848.90 | 2885.70 | 2857.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 2890.00 | 2886.56 | 2860.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:30:00 | 2911.90 | 2856.92 | 2853.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 11:15:00 | 2839.70 | 2852.35 | 2851.55 | SL hit (close<static) qty=1.00 sl=2848.90 alert=retest2 |

### Cycle 167 — SELL (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 12:15:00 | 2840.00 | 2849.88 | 2850.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 14:15:00 | 2829.90 | 2844.29 | 2847.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 15:15:00 | 2821.30 | 2817.73 | 2828.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 15:15:00 | 2821.30 | 2817.73 | 2828.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 2821.30 | 2817.73 | 2828.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 2827.10 | 2817.73 | 2828.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 2792.00 | 2812.58 | 2825.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 10:30:00 | 2788.60 | 2807.07 | 2821.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 11:00:00 | 2785.00 | 2807.07 | 2821.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 14:30:00 | 2787.10 | 2801.61 | 2814.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 10:45:00 | 2789.80 | 2793.57 | 2807.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 2791.80 | 2793.21 | 2805.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:45:00 | 2812.50 | 2793.21 | 2805.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 2787.20 | 2775.48 | 2790.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 2718.60 | 2766.63 | 2776.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:00:00 | 2709.60 | 2745.17 | 2764.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 10:45:00 | 2723.70 | 2698.55 | 2712.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 13:15:00 | 2781.50 | 2730.75 | 2725.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 13:15:00 | 2781.50 | 2730.75 | 2725.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 15:15:00 | 2819.90 | 2778.26 | 2757.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 09:15:00 | 2763.60 | 2775.32 | 2757.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 10:00:00 | 2763.60 | 2775.32 | 2757.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 2783.20 | 2776.90 | 2760.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:45:00 | 2809.20 | 2773.48 | 2764.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 10:15:00 | 2756.10 | 2791.28 | 2783.49 | SL hit (close<static) qty=1.00 sl=2760.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 2729.10 | 2772.93 | 2776.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 2706.80 | 2752.35 | 2765.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 2756.20 | 2748.03 | 2761.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 2756.20 | 2748.03 | 2761.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 2756.20 | 2748.03 | 2761.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 2756.20 | 2748.03 | 2761.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 2761.20 | 2750.66 | 2761.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 2761.20 | 2750.66 | 2761.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 2777.40 | 2756.01 | 2762.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:30:00 | 2776.70 | 2756.01 | 2762.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 2760.00 | 2756.81 | 2762.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 15:00:00 | 2749.20 | 2755.30 | 2760.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:45:00 | 2752.10 | 2751.98 | 2758.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 10:30:00 | 2745.00 | 2749.90 | 2756.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:15:00 | 2751.90 | 2751.01 | 2755.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 2752.00 | 2751.21 | 2755.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:45:00 | 2737.10 | 2748.89 | 2754.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:15:00 | 2730.70 | 2749.24 | 2753.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 12:15:00 | 2773.90 | 2747.85 | 2745.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 2773.90 | 2747.85 | 2745.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 2828.00 | 2771.22 | 2757.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 13:15:00 | 2760.20 | 2786.45 | 2771.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 13:15:00 | 2760.20 | 2786.45 | 2771.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 2760.20 | 2786.45 | 2771.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 2760.20 | 2786.45 | 2771.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 2743.30 | 2777.82 | 2768.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 2743.30 | 2777.82 | 2768.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 2760.50 | 2767.14 | 2765.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:30:00 | 2752.10 | 2767.14 | 2765.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 2781.70 | 2770.05 | 2766.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:45:00 | 2789.40 | 2770.94 | 2767.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 14:45:00 | 2794.00 | 2773.87 | 2769.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:00:00 | 2786.00 | 2778.24 | 2772.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-19 10:15:00 | 3068.34 | 3001.25 | 2959.17 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 13:15:00 | 3006.30 | 3011.44 | 3011.94 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 15:15:00 | 3030.00 | 3010.39 | 3009.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 10:15:00 | 3030.30 | 3016.74 | 3012.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 11:15:00 | 2998.20 | 3013.03 | 3011.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 11:15:00 | 2998.20 | 3013.03 | 3011.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 2998.20 | 3013.03 | 3011.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:00:00 | 2998.20 | 3013.03 | 3011.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 12:15:00 | 2978.10 | 3006.05 | 3008.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 13:15:00 | 2963.00 | 2997.44 | 3004.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 15:15:00 | 2946.60 | 2944.84 | 2966.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:15:00 | 2989.70 | 2944.84 | 2966.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 2959.60 | 2947.79 | 2965.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 2997.40 | 2947.79 | 2965.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 2966.30 | 2951.50 | 2965.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 2953.70 | 2951.50 | 2965.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 2986.00 | 2958.40 | 2967.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:45:00 | 2988.80 | 2958.40 | 2967.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 3000.00 | 2966.72 | 2970.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:00:00 | 3000.00 | 2966.72 | 2970.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 14:15:00 | 2988.70 | 2974.84 | 2973.63 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 10:15:00 | 2953.40 | 2972.30 | 2973.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 11:15:00 | 2945.40 | 2966.92 | 2970.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 12:15:00 | 2967.80 | 2967.09 | 2970.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 13:00:00 | 2967.80 | 2967.09 | 2970.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 2961.10 | 2965.90 | 2969.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 15:15:00 | 2944.60 | 2964.84 | 2968.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 3034.40 | 2975.51 | 2972.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 3034.40 | 2975.51 | 2972.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 3070.00 | 3015.42 | 2995.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 13:15:00 | 3133.30 | 3146.13 | 3104.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 13:30:00 | 3129.10 | 3146.13 | 3104.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 3135.30 | 3142.89 | 3113.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:30:00 | 3137.70 | 3142.89 | 3113.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 3135.00 | 3139.85 | 3119.45 | EMA400 retest candle locked (from upside) |

### Cycle 177 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 3065.10 | 3109.00 | 3110.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 13:15:00 | 3030.50 | 3082.73 | 3097.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 3049.50 | 3037.88 | 3061.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 15:00:00 | 3049.50 | 3037.88 | 3061.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 3042.90 | 3039.29 | 3058.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 3049.00 | 3039.29 | 3058.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 3044.50 | 3025.59 | 3039.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 3033.60 | 3025.59 | 3039.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 3040.00 | 3028.47 | 3039.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 14:00:00 | 3022.50 | 3034.38 | 3040.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 2968.20 | 3031.79 | 3038.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 3004.10 | 2965.79 | 2963.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 3004.10 | 2965.79 | 2963.94 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 13:15:00 | 2962.30 | 2967.16 | 2967.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-21 14:15:00 | 2956.00 | 2964.93 | 2966.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 09:15:00 | 2921.80 | 2897.88 | 2927.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 09:15:00 | 2921.80 | 2897.88 | 2927.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 2921.80 | 2897.88 | 2927.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:30:00 | 2910.00 | 2897.88 | 2927.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 2910.10 | 2900.32 | 2925.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 11:30:00 | 2893.50 | 2900.04 | 2923.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 13:30:00 | 2900.50 | 2903.98 | 2921.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 14:45:00 | 2902.00 | 2903.52 | 2919.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:45:00 | 2873.20 | 2897.12 | 2913.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 2927.70 | 2871.15 | 2886.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 2932.70 | 2871.15 | 2886.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 2958.60 | 2888.64 | 2893.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 2958.60 | 2888.64 | 2893.41 | SL hit (close>static) qty=1.00 sl=2929.90 alert=retest2 |

### Cycle 180 — BUY (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 11:15:00 | 2950.00 | 2900.91 | 2898.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 13:15:00 | 2968.60 | 2922.27 | 2909.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 10:15:00 | 2933.50 | 2943.78 | 2925.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 11:00:00 | 2933.50 | 2943.78 | 2925.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 3034.30 | 3051.27 | 3025.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 3112.80 | 3044.26 | 3028.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 09:30:00 | 3073.60 | 3117.32 | 3106.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 3059.80 | 3097.86 | 3098.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 3059.80 | 3097.86 | 3098.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 12:15:00 | 3039.30 | 3086.15 | 3093.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 09:15:00 | 3096.30 | 3058.60 | 3075.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 09:15:00 | 3096.30 | 3058.60 | 3075.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 3096.30 | 3058.60 | 3075.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:30:00 | 3093.90 | 3058.60 | 3075.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 3071.70 | 3061.22 | 3074.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:30:00 | 3087.20 | 3061.22 | 3074.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 3098.60 | 3068.70 | 3077.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:00:00 | 3098.60 | 3068.70 | 3077.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 3104.10 | 3075.78 | 3079.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 13:30:00 | 3090.50 | 3075.42 | 3079.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 3117.50 | 3084.07 | 3082.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 3117.50 | 3084.07 | 3082.14 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 3080.40 | 3114.32 | 3116.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 3051.60 | 3092.90 | 3105.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 3088.90 | 3069.70 | 3088.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 3088.90 | 3069.70 | 3088.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 3088.90 | 3069.70 | 3088.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 3110.00 | 3069.70 | 3088.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 3065.60 | 3068.88 | 3086.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:15:00 | 3108.90 | 3068.88 | 3086.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 3069.30 | 3068.96 | 3084.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:30:00 | 3074.90 | 3068.96 | 3084.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 2997.80 | 3008.57 | 3034.31 | EMA400 retest candle locked (from downside) |

### Cycle 184 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 3068.30 | 3043.25 | 3040.79 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 2976.70 | 3044.39 | 3045.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 2943.00 | 3024.11 | 3036.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 3002.30 | 2956.27 | 2988.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 3002.30 | 2956.27 | 2988.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 3002.30 | 2956.27 | 2988.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 3002.30 | 2956.27 | 2988.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 2974.80 | 2959.98 | 2986.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:30:00 | 3003.00 | 2959.98 | 2986.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 2972.00 | 2962.38 | 2985.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:45:00 | 2987.40 | 2962.38 | 2985.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 2989.30 | 2969.34 | 2984.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 13:45:00 | 2988.80 | 2969.34 | 2984.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 2867.60 | 2948.99 | 2974.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:15:00 | 2843.00 | 2948.99 | 2974.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 14:15:00 | 2998.30 | 2942.99 | 2937.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 2998.30 | 2942.99 | 2937.22 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 2884.50 | 2928.96 | 2932.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 15:15:00 | 2880.00 | 2901.42 | 2915.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 10:15:00 | 2902.00 | 2897.96 | 2911.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 10:30:00 | 2907.40 | 2897.96 | 2911.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 2875.30 | 2890.96 | 2903.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:30:00 | 2892.20 | 2890.96 | 2903.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 2873.80 | 2886.59 | 2899.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:45:00 | 2832.00 | 2874.82 | 2891.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:30:00 | 2841.80 | 2834.40 | 2863.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 2900.00 | 2869.12 | 2867.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — BUY (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 10:15:00 | 2900.00 | 2869.12 | 2867.43 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 11:15:00 | 2827.90 | 2868.28 | 2870.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 2819.50 | 2853.25 | 2863.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 14:15:00 | 2760.00 | 2744.59 | 2774.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 14:15:00 | 2760.00 | 2744.59 | 2774.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 2760.00 | 2744.59 | 2774.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 15:00:00 | 2760.00 | 2744.59 | 2774.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 2836.30 | 2764.67 | 2776.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 2836.30 | 2764.67 | 2776.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 2865.10 | 2784.75 | 2784.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 12:15:00 | 2895.50 | 2806.90 | 2794.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 13:15:00 | 2899.10 | 2903.68 | 2863.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 14:00:00 | 2899.10 | 2903.68 | 2863.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 2914.20 | 2905.78 | 2867.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 13:30:00 | 2935.80 | 2913.51 | 2886.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 2959.60 | 2988.00 | 2989.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 2959.60 | 2988.00 | 2989.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 2846.40 | 2950.60 | 2970.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 10:15:00 | 2931.30 | 2894.08 | 2920.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 10:15:00 | 2931.30 | 2894.08 | 2920.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 2931.30 | 2894.08 | 2920.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 2931.30 | 2894.08 | 2920.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 2897.50 | 2894.76 | 2918.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:15:00 | 2880.60 | 2894.76 | 2918.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 3085.30 | 2939.40 | 2930.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 3085.30 | 2939.40 | 2930.77 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 3065.00 | 3096.24 | 3097.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 3035.40 | 3070.09 | 3081.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 10:15:00 | 3085.00 | 3073.08 | 3081.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 10:15:00 | 3085.00 | 3073.08 | 3081.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 3085.00 | 3073.08 | 3081.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:30:00 | 3074.60 | 3073.08 | 3081.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 3071.10 | 3072.68 | 3080.85 | EMA400 retest candle locked (from downside) |

### Cycle 194 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 3135.60 | 3089.60 | 3086.84 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 12:15:00 | 3081.40 | 3124.30 | 3125.53 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 3171.20 | 3106.31 | 3102.35 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 3087.10 | 3118.26 | 3118.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 3009.10 | 3096.43 | 3108.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 2812.30 | 2755.09 | 2803.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 2812.30 | 2755.09 | 2803.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 2812.30 | 2755.09 | 2803.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:45:00 | 2813.80 | 2755.09 | 2803.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 2802.70 | 2764.61 | 2803.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:15:00 | 2787.30 | 2771.31 | 2803.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 2747.90 | 2778.83 | 2796.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 14:15:00 | 2647.93 | 2693.49 | 2742.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 2610.51 | 2675.01 | 2724.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 14:15:00 | 2575.10 | 2571.98 | 2607.73 | SL hit (close>ema200) qty=0.50 sl=2571.98 alert=retest2 |

### Cycle 198 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 2646.70 | 2622.79 | 2619.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 2746.40 | 2652.67 | 2634.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 15:15:00 | 2700.00 | 2704.70 | 2674.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-27 09:15:00 | 2704.90 | 2704.70 | 2674.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 2720.00 | 2707.76 | 2678.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 15:00:00 | 2733.40 | 2708.66 | 2689.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-29 09:15:00 | 3006.74 | 2880.93 | 2804.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 199 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 3564.20 | 3627.06 | 3632.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 3552.30 | 3595.39 | 3613.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 3694.60 | 3610.48 | 3616.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 3694.60 | 3610.48 | 3616.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 3694.60 | 3610.48 | 3616.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 3694.60 | 3610.48 | 3616.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 3687.10 | 3625.80 | 3623.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 3702.60 | 3660.83 | 3643.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 10:15:00 | 3643.20 | 3657.31 | 3643.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 10:15:00 | 3643.20 | 3657.31 | 3643.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 3643.20 | 3657.31 | 3643.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:00:00 | 3643.20 | 3657.31 | 3643.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 3641.00 | 3654.05 | 3643.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:15:00 | 3626.60 | 3654.05 | 3643.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 3636.00 | 3650.44 | 3642.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:45:00 | 3621.70 | 3650.44 | 3642.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 3628.80 | 3646.11 | 3641.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:45:00 | 3626.90 | 3646.11 | 3641.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 3647.80 | 3646.45 | 3642.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 15:15:00 | 3655.00 | 3646.45 | 3642.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 10:45:00 | 3665.50 | 3650.96 | 3645.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 3566.50 | 3674.29 | 3665.72 | SL hit (close<static) qty=1.00 sl=3625.00 alert=retest2 |

### Cycle 201 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 3563.00 | 3652.04 | 3656.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 3542.40 | 3630.11 | 3646.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 3579.60 | 3569.33 | 3601.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 12:00:00 | 3579.60 | 3569.33 | 3601.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 3622.40 | 3579.94 | 3603.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:45:00 | 3642.90 | 3579.94 | 3603.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 3661.10 | 3596.17 | 3608.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:45:00 | 3662.00 | 3596.17 | 3608.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 3729.40 | 3637.54 | 3625.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 3824.10 | 3754.01 | 3715.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 3842.30 | 3849.22 | 3808.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 3865.10 | 3852.48 | 3829.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 3865.10 | 3852.48 | 3829.81 | EMA400 retest candle locked (from upside) |

### Cycle 203 — SELL (started 2026-03-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 13:15:00 | 3775.00 | 3820.33 | 3821.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 3742.10 | 3791.38 | 3806.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 3799.10 | 3721.24 | 3751.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 3799.10 | 3721.24 | 3751.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 3799.10 | 3721.24 | 3751.14 | EMA400 retest candle locked (from downside) |

### Cycle 204 — BUY (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 13:15:00 | 3818.20 | 3775.32 | 3770.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 15:15:00 | 3838.00 | 3796.43 | 3781.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 3745.60 | 3845.48 | 3828.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 3745.60 | 3845.48 | 3828.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 3745.60 | 3845.48 | 3828.37 | EMA400 retest candle locked (from upside) |

### Cycle 205 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 3697.00 | 3815.78 | 3816.43 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 3878.60 | 3809.72 | 3807.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 11:15:00 | 3885.60 | 3824.89 | 3814.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 3794.80 | 3835.93 | 3827.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 10:15:00 | 3794.80 | 3835.93 | 3827.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 3794.80 | 3835.93 | 3827.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:00:00 | 3794.80 | 3835.93 | 3827.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 3779.90 | 3824.73 | 3823.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 12:00:00 | 3779.90 | 3824.73 | 3823.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 3756.90 | 3811.16 | 3817.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 3749.90 | 3782.49 | 3800.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 3619.00 | 3543.16 | 3597.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 3619.00 | 3543.16 | 3597.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 3619.00 | 3543.16 | 3597.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 3619.00 | 3543.16 | 3597.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 3668.30 | 3568.18 | 3604.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 3668.30 | 3568.18 | 3604.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 3663.20 | 3625.82 | 3625.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 3769.70 | 3668.41 | 3646.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 3766.80 | 3779.22 | 3739.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 14:00:00 | 3766.80 | 3779.22 | 3739.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 3768.40 | 3777.05 | 3742.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 14:45:00 | 3734.40 | 3777.05 | 3742.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 3760.00 | 3773.64 | 3743.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 3846.30 | 3773.64 | 3743.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 09:15:00 | 3722.70 | 3763.45 | 3741.85 | SL hit (close<static) qty=1.00 sl=3742.50 alert=retest2 |

### Cycle 209 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 3670.20 | 3729.12 | 3732.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 3519.00 | 3680.04 | 3709.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 3560.00 | 3529.01 | 3597.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 3560.00 | 3529.01 | 3597.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 3560.00 | 3529.01 | 3597.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 3506.70 | 3515.97 | 3585.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 13:45:00 | 3501.10 | 3506.74 | 3563.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 3694.40 | 3548.93 | 3568.80 | SL hit (close>static) qty=1.00 sl=3627.70 alert=retest2 |

### Cycle 210 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 3691.40 | 3595.19 | 3587.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 3720.60 | 3620.28 | 3599.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 3727.30 | 3730.46 | 3691.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 3727.30 | 3730.46 | 3691.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 3727.30 | 3730.46 | 3691.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 3859.80 | 3685.63 | 3682.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-01 11:15:00 | 4245.78 | 3815.17 | 3749.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 14:15:00 | 3724.70 | 3757.03 | 3760.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 10:15:00 | 3686.90 | 3732.95 | 3747.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 3723.40 | 3698.42 | 3719.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 3723.40 | 3698.42 | 3719.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3723.40 | 3698.42 | 3719.61 | EMA400 retest candle locked (from downside) |

### Cycle 212 — BUY (started 2026-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 12:15:00 | 3793.90 | 3740.79 | 3735.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 3879.80 | 3766.44 | 3748.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 14:15:00 | 4097.30 | 4099.18 | 4020.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 15:00:00 | 4097.30 | 4099.18 | 4020.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 4067.80 | 4116.33 | 4079.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:00:00 | 4067.80 | 4116.33 | 4079.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 4063.40 | 4105.74 | 4077.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:15:00 | 4060.30 | 4105.74 | 4077.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 4065.00 | 4097.59 | 4076.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:45:00 | 4060.00 | 4097.59 | 4076.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 4071.20 | 4084.30 | 4074.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 15:00:00 | 4071.20 | 4084.30 | 4074.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 15:15:00 | 4070.00 | 4081.44 | 4074.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 4135.20 | 4081.44 | 4074.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-23 09:15:00 | 4548.72 | 4308.09 | 4256.08 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 213 — SELL (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 12:15:00 | 4494.70 | 4520.85 | 4521.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 4483.40 | 4508.06 | 4515.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 09:15:00 | 4518.90 | 4504.90 | 4512.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 4518.90 | 4504.90 | 4512.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 4518.90 | 4504.90 | 4512.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 4518.90 | 4504.90 | 4512.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 4538.00 | 4511.52 | 4514.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:30:00 | 4550.10 | 4511.52 | 4514.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 4529.10 | 4514.78 | 4515.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:30:00 | 4543.80 | 4514.78 | 4515.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 4500.00 | 4511.82 | 4514.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 14:15:00 | 4490.70 | 4511.82 | 4514.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 15:00:00 | 4478.50 | 4485.72 | 4494.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 4575.50 | 4509.96 | 4504.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 214 — BUY (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 09:15:00 | 4575.50 | 4509.96 | 4504.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 4718.20 | 4603.35 | 4572.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 4668.00 | 4707.98 | 4653.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 4668.00 | 4707.98 | 4653.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 4668.00 | 4707.98 | 4653.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:00:00 | 4668.00 | 4707.98 | 4653.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 4633.80 | 4693.14 | 4651.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 4633.80 | 4693.14 | 4651.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 4630.00 | 4680.51 | 4649.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:45:00 | 4636.00 | 4680.51 | 4649.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 4638.60 | 4672.13 | 4648.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 4634.90 | 4672.13 | 4648.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-22 12:45:00 | 228.50 | 2023-05-22 14:15:00 | 221.00 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2023-05-29 09:15:00 | 188.00 | 2023-05-31 14:15:00 | 178.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-05-29 09:15:00 | 188.00 | 2023-06-01 10:15:00 | 183.30 | STOP_HIT | 0.50 | 2.50% |
| BUY | retest2 | 2023-06-12 11:30:00 | 213.75 | 2023-06-13 10:15:00 | 205.65 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2023-06-12 12:15:00 | 213.65 | 2023-06-13 10:15:00 | 205.65 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2023-06-13 09:30:00 | 213.90 | 2023-06-13 10:15:00 | 205.65 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2023-06-14 13:15:00 | 206.05 | 2023-06-15 13:15:00 | 195.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-14 13:15:00 | 206.05 | 2023-06-16 09:15:00 | 202.00 | STOP_HIT | 0.50 | 1.97% |
| BUY | retest1 | 2023-06-20 11:30:00 | 215.90 | 2023-06-21 09:15:00 | 226.70 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-06-20 11:30:00 | 215.90 | 2023-06-21 12:15:00 | 217.50 | STOP_HIT | 0.50 | 0.74% |
| BUY | retest2 | 2023-07-03 11:00:00 | 223.50 | 2023-07-05 11:15:00 | 218.85 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2023-07-04 09:15:00 | 227.95 | 2023-07-05 11:15:00 | 218.85 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2023-07-04 10:15:00 | 223.95 | 2023-07-05 11:15:00 | 218.85 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2023-07-04 14:30:00 | 223.50 | 2023-07-05 11:15:00 | 218.85 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2023-07-10 09:45:00 | 219.65 | 2023-07-10 13:15:00 | 220.70 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2023-07-12 09:15:00 | 213.05 | 2023-07-12 10:15:00 | 220.95 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2023-07-12 10:00:00 | 219.25 | 2023-07-12 10:15:00 | 220.95 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2023-07-14 11:00:00 | 237.65 | 2023-07-18 09:15:00 | 261.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-09 09:15:00 | 285.40 | 2023-08-11 09:15:00 | 313.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-28 09:15:00 | 362.00 | 2023-08-28 14:15:00 | 346.95 | STOP_HIT | 1.00 | -4.16% |
| BUY | retest2 | 2023-08-29 09:15:00 | 359.50 | 2023-08-29 10:15:00 | 348.25 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2023-09-12 11:30:00 | 335.50 | 2023-09-13 09:15:00 | 328.95 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2023-09-12 14:30:00 | 341.70 | 2023-09-13 09:15:00 | 328.95 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2023-10-05 12:00:00 | 418.00 | 2023-10-09 09:15:00 | 397.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-05 12:30:00 | 420.85 | 2023-10-09 09:15:00 | 399.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-05 12:00:00 | 418.00 | 2023-10-10 10:15:00 | 401.95 | STOP_HIT | 0.50 | 3.84% |
| SELL | retest2 | 2023-10-05 12:30:00 | 420.85 | 2023-10-10 10:15:00 | 401.95 | STOP_HIT | 0.50 | 4.49% |
| SELL | retest2 | 2023-10-20 11:45:00 | 387.00 | 2023-10-25 09:15:00 | 367.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 12:45:00 | 382.65 | 2023-10-25 09:15:00 | 363.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 13:45:00 | 384.00 | 2023-10-25 09:15:00 | 364.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-23 09:15:00 | 387.55 | 2023-10-25 09:15:00 | 368.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-23 11:30:00 | 373.85 | 2023-10-26 09:15:00 | 355.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-23 15:15:00 | 373.35 | 2023-10-26 09:15:00 | 354.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 11:45:00 | 387.00 | 2023-10-26 12:15:00 | 365.00 | STOP_HIT | 0.50 | 5.68% |
| SELL | retest2 | 2023-10-20 12:45:00 | 382.65 | 2023-10-26 12:15:00 | 365.00 | STOP_HIT | 0.50 | 4.61% |
| SELL | retest2 | 2023-10-20 13:45:00 | 384.00 | 2023-10-26 12:15:00 | 365.00 | STOP_HIT | 0.50 | 4.95% |
| SELL | retest2 | 2023-10-23 09:15:00 | 387.55 | 2023-10-26 12:15:00 | 365.00 | STOP_HIT | 0.50 | 5.82% |
| SELL | retest2 | 2023-10-23 11:30:00 | 373.85 | 2023-10-26 12:15:00 | 365.00 | STOP_HIT | 0.50 | 2.37% |
| SELL | retest2 | 2023-10-23 15:15:00 | 373.35 | 2023-10-26 12:15:00 | 365.00 | STOP_HIT | 0.50 | 2.24% |
| BUY | retest2 | 2023-10-31 09:15:00 | 379.80 | 2023-11-08 10:15:00 | 385.95 | STOP_HIT | 1.00 | 1.62% |
| BUY | retest2 | 2023-10-31 13:45:00 | 380.10 | 2023-11-08 10:15:00 | 385.95 | STOP_HIT | 1.00 | 1.54% |
| BUY | retest2 | 2023-10-31 14:15:00 | 395.25 | 2023-11-08 10:15:00 | 385.95 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2023-11-15 11:45:00 | 380.75 | 2023-11-15 13:15:00 | 388.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest1 | 2023-11-21 09:15:00 | 400.25 | 2023-11-22 09:15:00 | 420.26 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-11-21 09:15:00 | 400.25 | 2023-11-22 10:15:00 | 403.15 | STOP_HIT | 0.50 | 0.72% |
| SELL | retest2 | 2023-12-13 09:15:00 | 425.00 | 2023-12-13 10:15:00 | 432.90 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2023-12-18 13:30:00 | 478.00 | 2023-12-19 11:15:00 | 459.05 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2023-12-18 15:15:00 | 478.00 | 2023-12-19 11:15:00 | 459.05 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2023-12-28 13:15:00 | 480.00 | 2024-01-01 09:15:00 | 528.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-29 09:15:00 | 494.00 | 2024-01-02 09:15:00 | 543.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-06 09:15:00 | 736.10 | 2024-02-08 09:15:00 | 809.71 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-02-20 11:45:00 | 760.60 | 2024-02-22 10:15:00 | 792.00 | STOP_HIT | 1.00 | -4.13% |
| BUY | retest2 | 2024-02-23 14:45:00 | 829.15 | 2024-02-28 09:15:00 | 912.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-26 09:15:00 | 850.00 | 2024-03-01 09:15:00 | 935.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-03-18 09:45:00 | 822.65 | 2024-03-21 10:15:00 | 859.95 | STOP_HIT | 1.00 | -4.53% |
| SELL | retest2 | 2024-03-18 11:00:00 | 837.20 | 2024-03-21 11:15:00 | 862.30 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2024-03-18 12:15:00 | 835.20 | 2024-03-21 11:15:00 | 862.30 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2024-03-19 10:30:00 | 841.30 | 2024-03-21 11:15:00 | 862.30 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2024-03-20 09:30:00 | 827.00 | 2024-03-21 11:15:00 | 862.30 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest2 | 2024-04-08 12:15:00 | 973.00 | 2024-04-09 14:15:00 | 947.00 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2024-04-08 14:15:00 | 979.25 | 2024-04-09 14:15:00 | 947.00 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2024-04-08 15:00:00 | 978.25 | 2024-04-09 14:15:00 | 947.00 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2024-04-29 12:00:00 | 1129.50 | 2024-05-03 15:15:00 | 1139.00 | STOP_HIT | 1.00 | 0.84% |
| BUY | retest2 | 2024-04-29 14:15:00 | 1124.70 | 2024-05-03 15:15:00 | 1139.00 | STOP_HIT | 1.00 | 1.27% |
| BUY | retest2 | 2024-04-29 14:45:00 | 1130.00 | 2024-05-03 15:15:00 | 1139.00 | STOP_HIT | 1.00 | 0.80% |
| BUY | retest2 | 2024-05-03 14:00:00 | 1131.95 | 2024-05-03 15:15:00 | 1139.00 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2024-05-08 15:15:00 | 1038.35 | 2024-05-10 09:15:00 | 986.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-08 15:15:00 | 1038.35 | 2024-05-10 14:15:00 | 1000.00 | STOP_HIT | 0.50 | 3.69% |
| BUY | retest1 | 2024-06-18 09:15:00 | 1624.15 | 2024-06-18 15:15:00 | 1560.00 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest2 | 2024-06-19 09:15:00 | 1590.00 | 2024-06-19 14:15:00 | 1550.00 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2024-06-21 09:45:00 | 1596.30 | 2024-06-24 13:15:00 | 1561.40 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-06-21 11:15:00 | 1596.00 | 2024-06-24 13:15:00 | 1561.40 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2024-06-24 10:00:00 | 1597.85 | 2024-06-24 13:15:00 | 1561.40 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-06-24 11:15:00 | 1595.00 | 2024-06-24 13:15:00 | 1561.40 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-06-27 14:30:00 | 1524.95 | 2024-06-28 15:15:00 | 1448.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-27 14:30:00 | 1524.95 | 2024-06-28 15:15:00 | 1519.00 | STOP_HIT | 0.50 | 0.39% |
| SELL | retest2 | 2024-06-28 09:15:00 | 1501.85 | 2024-07-01 10:15:00 | 1566.20 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2024-07-01 10:30:00 | 1525.55 | 2024-07-01 11:15:00 | 1566.25 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2024-07-05 11:00:00 | 1685.00 | 2024-07-10 09:15:00 | 1662.85 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-07-05 12:00:00 | 1685.60 | 2024-07-10 09:15:00 | 1662.85 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-07-08 10:00:00 | 1680.00 | 2024-07-10 09:15:00 | 1662.85 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-07-10 09:15:00 | 1693.40 | 2024-07-10 09:15:00 | 1662.85 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-07-24 14:00:00 | 1445.10 | 2024-07-25 10:15:00 | 1525.60 | STOP_HIT | 1.00 | -5.57% |
| SELL | retest2 | 2024-07-24 15:15:00 | 1445.60 | 2024-07-25 10:15:00 | 1525.60 | STOP_HIT | 1.00 | -5.53% |
| BUY | retest2 | 2024-08-08 11:15:00 | 1767.00 | 2024-08-12 12:15:00 | 1713.65 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2024-08-09 11:30:00 | 1759.45 | 2024-08-12 12:15:00 | 1713.65 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2024-08-09 13:45:00 | 1769.00 | 2024-08-12 12:15:00 | 1713.65 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2024-08-26 13:15:00 | 1690.00 | 2024-08-28 14:15:00 | 1699.85 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-08-27 10:30:00 | 1695.00 | 2024-08-28 14:15:00 | 1699.85 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2024-08-28 10:30:00 | 1695.00 | 2024-08-28 14:15:00 | 1699.85 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2024-09-06 10:15:00 | 1630.00 | 2024-09-06 14:15:00 | 1548.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-06 10:15:00 | 1630.00 | 2024-09-09 14:15:00 | 1625.00 | STOP_HIT | 0.50 | 0.31% |
| SELL | retest2 | 2024-09-10 10:45:00 | 1637.80 | 2024-09-10 11:15:00 | 1665.85 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-09-19 09:15:00 | 1606.85 | 2024-09-20 09:15:00 | 1526.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-19 09:15:00 | 1606.85 | 2024-09-25 09:15:00 | 1553.55 | STOP_HIT | 0.50 | 3.32% |
| BUY | retest2 | 2024-09-26 12:00:00 | 1615.35 | 2024-10-03 13:15:00 | 1629.95 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2024-09-26 13:30:00 | 1613.95 | 2024-10-03 13:15:00 | 1629.95 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest2 | 2024-09-27 10:15:00 | 1626.00 | 2024-10-03 13:15:00 | 1629.95 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest1 | 2024-10-11 09:15:00 | 1786.55 | 2024-10-15 14:15:00 | 1875.88 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-10-11 09:15:00 | 1786.55 | 2024-10-17 09:15:00 | 1965.21 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-10-14 09:15:00 | 1812.00 | 2024-10-17 09:15:00 | 1958.00 | TARGET_HIT | 1.00 | 8.06% |
| BUY | retest2 | 2024-10-14 11:15:00 | 1780.00 | 2024-10-18 11:15:00 | 1813.40 | STOP_HIT | 1.00 | 1.88% |
| SELL | retest2 | 2024-10-29 09:15:00 | 1635.85 | 2024-10-29 14:15:00 | 1689.85 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2024-10-29 13:00:00 | 1627.00 | 2024-10-29 14:15:00 | 1689.85 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest2 | 2024-11-22 11:30:00 | 1905.50 | 2024-11-26 09:15:00 | 1823.50 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2024-11-25 09:15:00 | 1918.00 | 2024-11-26 09:15:00 | 1823.50 | STOP_HIT | 1.00 | -4.93% |
| BUY | retest2 | 2024-11-25 11:45:00 | 1909.10 | 2024-11-26 09:15:00 | 1823.50 | STOP_HIT | 1.00 | -4.48% |
| BUY | retest2 | 2024-12-09 11:15:00 | 1918.90 | 2024-12-13 10:15:00 | 1909.25 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-12-11 11:00:00 | 1924.00 | 2024-12-13 10:15:00 | 1909.25 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-12-19 09:30:00 | 2062.65 | 2024-12-24 09:15:00 | 2094.00 | STOP_HIT | 1.00 | 1.52% |
| BUY | retest2 | 2024-12-19 11:45:00 | 2061.95 | 2024-12-24 12:15:00 | 2080.05 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest2 | 2024-12-19 12:45:00 | 2061.50 | 2024-12-24 12:15:00 | 2080.05 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2024-12-19 13:45:00 | 2060.15 | 2024-12-24 12:15:00 | 2080.05 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2024-12-23 13:30:00 | 2127.30 | 2024-12-24 12:15:00 | 2080.05 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-01-10 09:15:00 | 1967.95 | 2025-01-14 09:15:00 | 1869.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 12:30:00 | 1970.80 | 2025-01-14 09:15:00 | 1872.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 15:15:00 | 1951.00 | 2025-01-14 09:15:00 | 1853.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 09:15:00 | 1967.95 | 2025-01-14 14:15:00 | 1886.85 | STOP_HIT | 0.50 | 4.12% |
| SELL | retest2 | 2025-01-10 12:30:00 | 1970.80 | 2025-01-14 14:15:00 | 1886.85 | STOP_HIT | 0.50 | 4.26% |
| SELL | retest2 | 2025-01-10 15:15:00 | 1951.00 | 2025-01-14 14:15:00 | 1886.85 | STOP_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2025-01-23 12:30:00 | 1800.65 | 2025-01-27 09:15:00 | 1710.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:30:00 | 1799.10 | 2025-01-27 09:15:00 | 1709.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 12:30:00 | 1800.65 | 2025-01-28 09:15:00 | 1620.59 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 09:30:00 | 1799.10 | 2025-01-28 09:15:00 | 1619.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-17 09:15:00 | 1414.95 | 2025-02-20 09:15:00 | 1344.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-17 09:15:00 | 1414.95 | 2025-02-20 10:15:00 | 1387.85 | STOP_HIT | 0.50 | 1.92% |
| BUY | retest2 | 2025-03-06 14:45:00 | 1391.40 | 2025-03-13 10:15:00 | 1427.40 | STOP_HIT | 1.00 | 2.59% |
| BUY | retest2 | 2025-03-07 10:15:00 | 1394.95 | 2025-03-13 10:15:00 | 1427.40 | STOP_HIT | 1.00 | 2.33% |
| BUY | retest2 | 2025-03-20 09:15:00 | 1516.35 | 2025-03-20 09:15:00 | 1481.40 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-03-20 12:15:00 | 1511.45 | 2025-03-26 11:15:00 | 1547.00 | STOP_HIT | 1.00 | 2.35% |
| SELL | retest2 | 2025-04-04 13:30:00 | 1409.25 | 2025-04-07 09:15:00 | 1338.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 13:30:00 | 1409.25 | 2025-04-08 15:15:00 | 1311.00 | STOP_HIT | 0.50 | 6.97% |
| BUY | retest2 | 2025-05-02 09:15:00 | 1571.50 | 2025-05-06 13:15:00 | 1538.50 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-05-02 10:30:00 | 1572.00 | 2025-05-06 13:15:00 | 1538.50 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-05-05 09:15:00 | 1575.70 | 2025-05-06 13:15:00 | 1538.50 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-05-05 12:30:00 | 1586.00 | 2025-05-06 13:15:00 | 1538.50 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2025-05-21 13:15:00 | 1823.90 | 2025-05-21 14:15:00 | 1849.90 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-06-02 09:15:00 | 2309.20 | 2025-06-06 13:15:00 | 2332.40 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2025-06-20 09:30:00 | 2392.40 | 2025-06-20 15:15:00 | 2345.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-06-26 09:15:00 | 2415.60 | 2025-06-26 11:15:00 | 2375.50 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-06-26 10:45:00 | 2412.50 | 2025-06-26 11:15:00 | 2375.50 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-06-26 11:15:00 | 2400.00 | 2025-06-26 11:15:00 | 2375.50 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-07-01 09:15:00 | 2349.50 | 2025-07-02 11:15:00 | 2375.10 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-07-01 12:45:00 | 2348.90 | 2025-07-02 11:15:00 | 2375.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-07-01 14:15:00 | 2348.70 | 2025-07-02 11:15:00 | 2375.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-07-02 09:15:00 | 2345.60 | 2025-07-02 11:15:00 | 2375.10 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-07-04 15:15:00 | 2345.20 | 2025-07-08 09:15:00 | 2372.20 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-07-07 10:30:00 | 2350.00 | 2025-07-08 09:15:00 | 2372.20 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-07-07 11:45:00 | 2349.40 | 2025-07-08 09:15:00 | 2372.20 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-07-08 10:30:00 | 2327.80 | 2025-07-08 12:15:00 | 2383.90 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-07-10 12:15:00 | 2407.20 | 2025-07-10 13:15:00 | 2367.00 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-07-21 10:00:00 | 2407.90 | 2025-07-25 12:15:00 | 2440.00 | STOP_HIT | 1.00 | 1.33% |
| BUY | retest2 | 2025-08-04 15:15:00 | 2810.00 | 2025-08-07 15:15:00 | 2787.90 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-08-18 09:30:00 | 2911.90 | 2025-08-18 11:15:00 | 2839.70 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-08-20 10:30:00 | 2788.60 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-08-20 11:00:00 | 2785.00 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-08-20 14:30:00 | 2787.10 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-08-21 10:45:00 | 2789.80 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2025-08-26 09:15:00 | 2718.60 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-08-26 11:00:00 | 2709.60 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-08-29 10:45:00 | 2723.70 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-09-03 09:45:00 | 2809.20 | 2025-09-04 10:15:00 | 2756.10 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-09-05 15:00:00 | 2749.20 | 2025-09-10 12:15:00 | 2773.90 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-09-08 09:45:00 | 2752.10 | 2025-09-10 12:15:00 | 2773.90 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-09-08 10:30:00 | 2745.00 | 2025-09-10 12:15:00 | 2773.90 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-09-08 13:15:00 | 2751.90 | 2025-09-10 12:15:00 | 2773.90 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-09-08 14:45:00 | 2737.10 | 2025-09-10 12:15:00 | 2773.90 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-09-09 10:15:00 | 2730.70 | 2025-09-10 12:15:00 | 2773.90 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-09-12 13:45:00 | 2789.40 | 2025-09-19 10:15:00 | 3068.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-12 14:45:00 | 2794.00 | 2025-09-19 10:15:00 | 3064.60 | TARGET_HIT | 1.00 | 9.69% |
| BUY | retest2 | 2025-09-15 10:00:00 | 2786.00 | 2025-09-19 11:15:00 | 3073.40 | TARGET_HIT | 1.00 | 10.32% |
| SELL | retest2 | 2025-09-30 15:15:00 | 2944.60 | 2025-10-01 09:15:00 | 3034.40 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-10-13 14:00:00 | 3022.50 | 2025-10-20 09:15:00 | 3004.10 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2025-10-14 09:15:00 | 2968.20 | 2025-10-20 09:15:00 | 3004.10 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-10-24 11:30:00 | 2893.50 | 2025-10-28 10:15:00 | 2958.60 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-10-24 13:30:00 | 2900.50 | 2025-10-28 10:15:00 | 2958.60 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-10-24 14:45:00 | 2902.00 | 2025-10-28 10:15:00 | 2958.60 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-10-27 09:45:00 | 2873.20 | 2025-10-28 10:15:00 | 2958.60 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2025-11-03 09:15:00 | 3112.80 | 2025-11-06 11:15:00 | 3059.80 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-11-06 09:30:00 | 3073.60 | 2025-11-06 11:15:00 | 3059.80 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-11-07 13:30:00 | 3090.50 | 2025-11-10 09:15:00 | 3117.50 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-11-24 15:15:00 | 2843.00 | 2025-11-26 14:15:00 | 2998.30 | STOP_HIT | 1.00 | -5.46% |
| SELL | retest2 | 2025-12-01 11:45:00 | 2832.00 | 2025-12-03 10:15:00 | 2900.00 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-12-02 09:30:00 | 2841.80 | 2025-12-03 10:15:00 | 2900.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-12-11 13:30:00 | 2935.80 | 2025-12-17 11:15:00 | 2959.60 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest2 | 2025-12-19 12:15:00 | 2880.60 | 2025-12-22 09:15:00 | 3085.30 | STOP_HIT | 1.00 | -7.11% |
| SELL | retest2 | 2026-01-14 12:15:00 | 2787.30 | 2026-01-16 14:15:00 | 2647.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:15:00 | 2747.90 | 2026-01-19 09:15:00 | 2610.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 12:15:00 | 2787.30 | 2026-01-21 14:15:00 | 2575.10 | STOP_HIT | 0.50 | 7.61% |
| SELL | retest2 | 2026-01-16 09:15:00 | 2747.90 | 2026-01-21 14:15:00 | 2575.10 | STOP_HIT | 0.50 | 6.29% |
| BUY | retest2 | 2026-01-27 15:00:00 | 2733.40 | 2026-01-29 09:15:00 | 3006.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-17 15:15:00 | 3655.00 | 2026-02-19 09:15:00 | 3566.50 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2026-02-18 10:45:00 | 3665.50 | 2026-02-19 09:15:00 | 3566.50 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2026-03-20 09:15:00 | 3846.30 | 2026-03-20 09:15:00 | 3722.70 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2026-03-24 10:30:00 | 3506.70 | 2026-03-25 09:15:00 | 3694.40 | STOP_HIT | 1.00 | -5.35% |
| SELL | retest2 | 2026-03-24 13:45:00 | 3501.10 | 2026-03-25 09:15:00 | 3694.40 | STOP_HIT | 1.00 | -5.52% |
| BUY | retest2 | 2026-04-01 09:15:00 | 3859.80 | 2026-04-01 11:15:00 | 4245.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-01 15:00:00 | 3799.80 | 2026-04-06 14:15:00 | 3724.70 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-04-02 13:15:00 | 3820.00 | 2026-04-06 14:15:00 | 3724.70 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2026-04-02 14:30:00 | 3792.50 | 2026-04-06 14:15:00 | 3724.70 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2026-04-06 09:15:00 | 3884.00 | 2026-04-06 14:15:00 | 3724.70 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2026-04-17 09:15:00 | 4135.20 | 2026-04-23 09:15:00 | 4548.72 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 14:15:00 | 4490.70 | 2026-05-05 09:15:00 | 4575.50 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-05-04 15:00:00 | 4478.50 | 2026-05-05 09:15:00 | 4575.50 | STOP_HIT | 1.00 | -2.17% |
