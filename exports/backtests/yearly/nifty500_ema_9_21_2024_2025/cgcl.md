# Capri Global Capital Ltd. (CGCL)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 197.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 158 |
| ALERT1 | 97 |
| ALERT2 | 96 |
| ALERT2_SKIP | 43 |
| ALERT3 | 267 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 155 |
| PARTIAL | 17 |
| TARGET_HIT | 11 |
| STOP_HIT | 146 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 174 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 66 / 108
- **Target hits / Stop hits / Partials:** 11 / 146 / 17
- **Avg / median % per leg:** 0.50% / -0.85%
- **Sum % (uncompounded):** 87.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 66 | 13 | 19.7% | 4 | 61 | 1 | -0.26% | -17.2% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 1.64% | 4.9% |
| BUY @ 3rd Alert (retest2) | 63 | 11 | 17.5% | 4 | 59 | 0 | -0.35% | -22.1% |
| SELL (all) | 108 | 53 | 49.1% | 7 | 85 | 16 | 0.97% | 105.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 108 | 53 | 49.1% | 7 | 85 | 16 | 0.97% | 105.1% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 1.64% | 4.9% |
| retest2 (combined) | 171 | 64 | 37.4% | 11 | 144 | 16 | 0.49% | 83.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 13:15:00 | 225.00 | 228.24 | 228.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 15:15:00 | 223.80 | 226.23 | 227.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 10:15:00 | 226.45 | 226.15 | 226.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 10:15:00 | 226.45 | 226.15 | 226.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 226.45 | 226.15 | 226.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 10:30:00 | 227.20 | 226.15 | 226.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 226.00 | 225.63 | 226.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 226.00 | 225.63 | 226.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 225.50 | 225.60 | 226.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:15:00 | 226.55 | 225.60 | 226.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 226.80 | 225.84 | 226.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 11:30:00 | 224.50 | 225.61 | 226.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-18 09:15:00 | 229.00 | 225.28 | 225.65 | SL hit (close>static) qty=1.00 sl=227.85 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 222.55 | 214.09 | 213.42 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 205.05 | 215.22 | 215.50 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 214.95 | 213.39 | 213.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 218.00 | 214.57 | 213.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 11:15:00 | 214.90 | 215.18 | 214.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 12:00:00 | 214.90 | 215.18 | 214.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 12:15:00 | 213.50 | 214.85 | 214.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 12:30:00 | 213.95 | 214.85 | 214.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 214.95 | 214.87 | 214.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 09:15:00 | 215.90 | 214.90 | 214.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 12:45:00 | 215.70 | 215.23 | 214.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 14:15:00 | 215.35 | 215.11 | 214.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 14:45:00 | 215.30 | 215.17 | 214.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 15:15:00 | 215.00 | 215.13 | 214.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 09:15:00 | 217.42 | 215.13 | 214.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-11 09:15:00 | 214.64 | 214.96 | 214.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 09:15:00 | 214.64 | 214.96 | 214.99 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-11 10:15:00 | 215.41 | 215.05 | 215.03 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 11:15:00 | 213.91 | 214.82 | 214.93 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 10:15:00 | 216.42 | 215.10 | 214.94 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 12:15:00 | 213.65 | 214.64 | 214.75 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-06-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 14:15:00 | 224.98 | 216.10 | 215.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 09:15:00 | 230.35 | 220.15 | 217.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 14:15:00 | 222.64 | 223.04 | 220.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-18 15:00:00 | 222.64 | 223.04 | 220.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 219.70 | 222.75 | 221.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 13:00:00 | 219.70 | 222.75 | 221.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 13:15:00 | 219.79 | 222.16 | 220.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 14:00:00 | 219.79 | 222.16 | 220.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 222.46 | 223.06 | 222.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 15:00:00 | 222.46 | 223.06 | 222.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 222.10 | 222.87 | 222.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 224.27 | 222.87 | 222.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 09:45:00 | 223.54 | 223.03 | 222.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 11:30:00 | 223.66 | 223.15 | 222.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 12:15:00 | 220.58 | 222.63 | 222.24 | SL hit (close<static) qty=1.00 sl=221.55 alert=retest2 |

### Cycle 11 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 220.00 | 221.72 | 221.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 10:15:00 | 216.00 | 220.33 | 221.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 09:15:00 | 228.61 | 220.44 | 220.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 09:15:00 | 228.61 | 220.44 | 220.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 228.61 | 220.44 | 220.60 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 10:15:00 | 221.79 | 220.71 | 220.71 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 11:15:00 | 218.80 | 220.33 | 220.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 14:15:00 | 214.59 | 218.57 | 219.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 218.92 | 215.47 | 216.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 218.92 | 215.47 | 216.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 218.92 | 215.47 | 216.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 10:45:00 | 216.00 | 215.43 | 216.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 11:00:00 | 215.00 | 214.97 | 215.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 14:15:00 | 216.00 | 215.17 | 215.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 10:30:00 | 216.18 | 215.42 | 215.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 11:15:00 | 212.61 | 214.86 | 215.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 12:30:00 | 210.33 | 211.82 | 212.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 15:15:00 | 214.80 | 212.65 | 212.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2024-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 15:15:00 | 214.80 | 212.65 | 212.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 12:15:00 | 215.67 | 213.86 | 213.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 14:15:00 | 221.10 | 221.11 | 218.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-11 15:00:00 | 221.10 | 221.11 | 218.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 218.67 | 220.43 | 218.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:45:00 | 218.83 | 220.43 | 218.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 218.99 | 220.15 | 218.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:30:00 | 218.99 | 220.15 | 218.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 218.90 | 219.90 | 218.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 15:00:00 | 228.26 | 221.57 | 219.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:15:00 | 220.89 | 220.91 | 220.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 13:15:00 | 217.93 | 220.82 | 220.74 | SL hit (close<static) qty=1.00 sl=218.33 alert=retest2 |

### Cycle 15 — SELL (started 2024-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 14:15:00 | 217.45 | 220.15 | 220.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 215.97 | 218.64 | 219.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 15:15:00 | 212.40 | 212.15 | 213.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 09:15:00 | 213.00 | 212.15 | 213.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 211.50 | 212.02 | 213.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 210.05 | 212.06 | 213.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:00:00 | 210.77 | 211.80 | 213.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 14:15:00 | 210.97 | 211.70 | 212.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 14:15:00 | 210.90 | 212.55 | 212.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 212.00 | 211.95 | 212.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 09:30:00 | 212.88 | 211.95 | 212.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 212.70 | 212.10 | 212.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:00:00 | 212.70 | 212.10 | 212.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 212.30 | 212.14 | 212.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:30:00 | 212.65 | 212.14 | 212.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 212.11 | 212.14 | 212.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:30:00 | 212.67 | 212.14 | 212.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 212.80 | 212.27 | 212.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 13:30:00 | 212.87 | 212.27 | 212.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 208.80 | 211.57 | 212.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:45:00 | 212.92 | 211.57 | 212.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 211.54 | 211.16 | 211.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 13:15:00 | 208.90 | 211.64 | 211.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 14:45:00 | 209.85 | 210.69 | 211.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 14:00:00 | 210.02 | 210.96 | 211.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 12:30:00 | 210.29 | 210.81 | 210.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 211.69 | 210.35 | 210.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:45:00 | 212.30 | 210.35 | 210.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 211.89 | 210.66 | 210.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:15:00 | 212.30 | 210.66 | 210.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-01 11:15:00 | 212.58 | 211.04 | 210.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 11:15:00 | 212.58 | 211.04 | 210.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-01 14:15:00 | 225.05 | 213.97 | 212.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 12:15:00 | 216.50 | 216.97 | 214.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-02 13:00:00 | 216.50 | 216.97 | 214.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 214.00 | 216.22 | 214.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 14:45:00 | 214.73 | 216.22 | 214.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 15:15:00 | 214.00 | 215.77 | 214.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 09:15:00 | 209.81 | 215.77 | 214.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 208.12 | 213.43 | 213.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 11:15:00 | 207.70 | 212.28 | 213.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 210.30 | 210.23 | 211.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 09:30:00 | 209.99 | 210.23 | 211.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 207.14 | 207.13 | 208.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 12:45:00 | 205.15 | 206.77 | 207.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:30:00 | 205.57 | 206.43 | 207.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 09:45:00 | 205.71 | 206.05 | 207.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 204.67 | 206.05 | 207.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 206.88 | 205.35 | 206.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 15:00:00 | 206.88 | 205.35 | 206.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 206.00 | 205.48 | 206.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:45:00 | 205.70 | 205.47 | 206.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 11:15:00 | 205.92 | 205.95 | 206.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 12:15:00 | 205.85 | 205.97 | 206.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 15:15:00 | 205.50 | 205.85 | 206.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 15:15:00 | 205.50 | 205.78 | 206.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 10:15:00 | 204.77 | 205.70 | 206.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 10:15:00 | 204.71 | 202.86 | 202.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 204.71 | 202.86 | 202.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 11:15:00 | 205.18 | 203.32 | 202.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 14:15:00 | 203.62 | 203.76 | 203.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 14:30:00 | 203.95 | 203.76 | 203.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 203.64 | 204.04 | 203.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 13:00:00 | 203.64 | 204.04 | 203.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 204.98 | 204.23 | 203.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 15:15:00 | 206.00 | 204.31 | 203.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 10:15:00 | 209.39 | 212.58 | 212.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 10:15:00 | 209.39 | 212.58 | 212.66 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 09:15:00 | 214.01 | 212.67 | 212.55 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 14:15:00 | 212.20 | 213.05 | 213.06 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 215.32 | 213.50 | 213.26 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 11:15:00 | 206.91 | 212.31 | 212.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 12:15:00 | 203.91 | 210.63 | 211.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 14:15:00 | 207.70 | 207.26 | 208.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-02 14:30:00 | 207.50 | 207.26 | 208.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 207.58 | 207.44 | 208.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 10:15:00 | 209.01 | 207.44 | 208.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 210.85 | 208.12 | 208.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 11:00:00 | 210.85 | 208.12 | 208.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 11:15:00 | 209.48 | 208.40 | 208.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 13:15:00 | 209.03 | 208.69 | 208.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 14:15:00 | 212.14 | 209.46 | 209.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 14:15:00 | 212.14 | 209.46 | 209.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 15:15:00 | 213.00 | 210.17 | 209.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 12:15:00 | 213.93 | 214.45 | 213.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 13:00:00 | 213.93 | 214.45 | 213.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 13:15:00 | 213.15 | 214.19 | 213.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:00:00 | 213.15 | 214.19 | 213.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 213.13 | 213.98 | 213.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:30:00 | 213.13 | 213.98 | 213.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 214.00 | 213.98 | 213.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 211.26 | 213.98 | 213.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 212.35 | 213.66 | 213.30 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 11:15:00 | 211.63 | 212.96 | 213.03 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 14:15:00 | 214.00 | 213.07 | 213.06 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 15:15:00 | 212.10 | 212.88 | 212.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-10 13:15:00 | 211.84 | 212.77 | 212.90 | Break + close below crossover candle low |

### Cycle 28 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 214.40 | 213.10 | 213.04 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 11:15:00 | 212.91 | 213.02 | 213.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 14:15:00 | 212.03 | 212.82 | 212.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 13:15:00 | 212.17 | 211.70 | 212.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 13:15:00 | 212.17 | 211.70 | 212.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 212.17 | 211.70 | 212.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 14:00:00 | 212.17 | 211.70 | 212.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 213.01 | 211.96 | 212.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 15:00:00 | 213.01 | 211.96 | 212.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 213.50 | 212.27 | 212.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:15:00 | 215.80 | 212.27 | 212.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 216.58 | 213.13 | 212.76 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 12:15:00 | 212.67 | 213.34 | 213.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 15:15:00 | 212.26 | 212.94 | 213.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 13:15:00 | 211.66 | 211.63 | 212.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-17 14:00:00 | 211.66 | 211.63 | 212.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 212.91 | 211.89 | 212.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 15:00:00 | 212.91 | 211.89 | 212.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 212.48 | 212.00 | 212.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:15:00 | 212.58 | 212.00 | 212.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 212.29 | 212.06 | 212.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 13:00:00 | 210.65 | 211.75 | 212.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 12:15:00 | 209.55 | 206.85 | 206.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2024-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 12:15:00 | 209.55 | 206.85 | 206.81 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 11:15:00 | 206.25 | 206.81 | 206.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 15:15:00 | 204.41 | 206.32 | 206.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 13:15:00 | 205.32 | 205.19 | 205.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-30 13:45:00 | 205.54 | 205.19 | 205.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 207.00 | 205.55 | 205.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:00:00 | 207.00 | 205.55 | 205.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 206.46 | 205.74 | 205.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:15:00 | 205.64 | 205.74 | 205.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 205.60 | 205.76 | 205.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 11:30:00 | 204.86 | 205.61 | 205.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 12:15:00 | 204.10 | 205.61 | 205.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 14:15:00 | 207.15 | 205.52 | 205.73 | SL hit (close>static) qty=1.00 sl=206.10 alert=retest2 |

### Cycle 34 — BUY (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 09:15:00 | 183.44 | 182.98 | 182.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 15:15:00 | 186.20 | 184.22 | 183.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 11:15:00 | 203.70 | 204.97 | 198.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 11:30:00 | 203.04 | 204.97 | 198.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 204.36 | 204.71 | 201.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:30:00 | 209.84 | 206.67 | 204.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 09:15:00 | 209.80 | 210.00 | 208.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-07 15:15:00 | 207.50 | 208.27 | 208.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 15:15:00 | 207.50 | 208.27 | 208.30 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 09:15:00 | 208.78 | 208.38 | 208.34 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 207.59 | 208.20 | 208.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 12:15:00 | 207.11 | 207.98 | 208.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 15:15:00 | 190.45 | 190.13 | 192.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 09:15:00 | 191.34 | 190.13 | 192.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 192.69 | 190.64 | 192.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 192.69 | 190.64 | 192.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 191.89 | 190.89 | 192.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 15:00:00 | 191.00 | 191.53 | 192.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 198.99 | 189.69 | 189.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 198.99 | 189.69 | 189.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 199.20 | 191.59 | 190.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 194.50 | 196.14 | 193.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 15:00:00 | 194.50 | 196.14 | 193.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 194.95 | 195.90 | 193.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:15:00 | 194.85 | 195.90 | 193.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 200.98 | 196.92 | 194.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 12:45:00 | 202.02 | 199.08 | 195.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 13:45:00 | 202.19 | 199.89 | 196.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 13:15:00 | 206.81 | 207.78 | 207.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2024-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 13:15:00 | 206.81 | 207.78 | 207.90 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 208.99 | 208.13 | 208.04 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 12:15:00 | 207.51 | 208.13 | 208.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 09:15:00 | 206.84 | 207.48 | 207.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 15:15:00 | 206.99 | 206.85 | 207.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-06 09:15:00 | 205.81 | 206.85 | 207.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 205.55 | 206.59 | 207.13 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 09:15:00 | 212.46 | 207.58 | 207.30 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 12:15:00 | 206.73 | 207.71 | 207.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 14:15:00 | 206.06 | 207.29 | 207.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 14:15:00 | 207.20 | 206.48 | 206.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 14:15:00 | 207.20 | 206.48 | 206.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 207.20 | 206.48 | 206.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 15:00:00 | 207.20 | 206.48 | 206.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 206.75 | 206.53 | 206.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:30:00 | 204.85 | 206.11 | 206.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 194.61 | 196.02 | 197.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 194.54 | 194.07 | 195.53 | SL hit (close>ema200) qty=0.50 sl=194.07 alert=retest2 |

### Cycle 44 — BUY (started 2025-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 14:15:00 | 187.53 | 186.86 | 186.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 188.53 | 187.25 | 187.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 15:15:00 | 191.20 | 191.39 | 190.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 09:15:00 | 189.30 | 191.39 | 190.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 189.28 | 190.97 | 190.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 189.22 | 190.97 | 190.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 185.62 | 189.90 | 189.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 185.62 | 189.90 | 189.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 185.57 | 189.03 | 189.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 182.89 | 187.18 | 188.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 13:15:00 | 185.34 | 185.14 | 186.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 14:00:00 | 185.34 | 185.14 | 186.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 185.89 | 185.29 | 186.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:45:00 | 186.42 | 185.29 | 186.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 187.50 | 185.62 | 186.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 11:00:00 | 187.50 | 185.62 | 186.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 11:15:00 | 185.82 | 185.66 | 186.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:15:00 | 185.52 | 185.66 | 186.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 14:15:00 | 189.10 | 186.39 | 186.43 | SL hit (close>static) qty=1.00 sl=187.67 alert=retest2 |

### Cycle 46 — BUY (started 2025-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 15:15:00 | 188.50 | 186.81 | 186.62 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 184.01 | 186.47 | 186.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 15:15:00 | 181.30 | 183.79 | 185.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 177.59 | 177.49 | 180.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 10:00:00 | 177.59 | 177.49 | 180.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 179.55 | 178.27 | 179.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 13:45:00 | 179.42 | 178.27 | 179.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 180.75 | 178.77 | 179.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:45:00 | 180.81 | 178.77 | 179.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 182.00 | 179.41 | 180.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 181.41 | 179.41 | 180.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 179.79 | 179.63 | 180.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 181.50 | 179.63 | 180.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 176.92 | 176.89 | 178.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:45:00 | 177.73 | 176.89 | 178.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 11:15:00 | 179.27 | 177.34 | 178.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 12:00:00 | 179.27 | 177.34 | 178.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 12:15:00 | 178.76 | 177.62 | 178.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 13:15:00 | 180.41 | 177.62 | 178.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2025-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 15:15:00 | 179.40 | 178.80 | 178.78 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 10:15:00 | 178.37 | 178.72 | 178.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 12:15:00 | 177.76 | 178.46 | 178.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 14:15:00 | 178.78 | 178.41 | 178.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 14:15:00 | 178.78 | 178.41 | 178.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 178.78 | 178.41 | 178.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 15:00:00 | 178.78 | 178.41 | 178.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2025-01-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 15:15:00 | 180.00 | 178.73 | 178.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 12:15:00 | 180.87 | 179.51 | 179.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 178.63 | 180.12 | 179.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 10:15:00 | 178.63 | 180.12 | 179.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 178.63 | 180.12 | 179.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 178.63 | 180.12 | 179.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 180.00 | 180.09 | 179.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:15:00 | 179.21 | 180.09 | 179.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 179.67 | 180.01 | 179.66 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2025-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 15:15:00 | 178.50 | 179.35 | 179.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 177.89 | 178.98 | 179.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 179.50 | 178.88 | 179.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 179.50 | 178.88 | 179.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 179.50 | 178.88 | 179.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 14:30:00 | 179.36 | 178.88 | 179.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 179.50 | 179.00 | 179.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 09:15:00 | 178.29 | 179.00 | 179.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 181.30 | 179.46 | 179.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 181.30 | 179.46 | 179.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 10:15:00 | 182.98 | 180.16 | 179.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-23 15:15:00 | 181.00 | 181.66 | 180.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 15:15:00 | 181.00 | 181.66 | 180.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 181.00 | 181.66 | 180.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 09:15:00 | 199.62 | 181.66 | 180.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 15:15:00 | 185.30 | 188.12 | 188.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2025-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 15:15:00 | 185.30 | 188.12 | 188.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 176.75 | 185.85 | 187.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 14:15:00 | 179.81 | 177.25 | 179.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 14:15:00 | 179.81 | 177.25 | 179.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 179.81 | 177.25 | 179.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 15:00:00 | 179.81 | 177.25 | 179.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 15:15:00 | 181.00 | 178.00 | 179.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:15:00 | 177.80 | 179.30 | 180.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 09:15:00 | 177.57 | 177.34 | 177.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 177.57 | 177.34 | 177.31 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 10:15:00 | 176.75 | 177.22 | 177.26 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 178.18 | 177.24 | 177.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 12:15:00 | 178.70 | 177.70 | 177.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 10:15:00 | 177.41 | 178.00 | 177.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 10:15:00 | 177.41 | 178.00 | 177.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 177.41 | 178.00 | 177.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:00:00 | 177.41 | 178.00 | 177.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 177.41 | 177.88 | 177.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:15:00 | 177.42 | 177.88 | 177.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 177.06 | 177.72 | 177.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:30:00 | 176.96 | 177.72 | 177.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 13:15:00 | 176.57 | 177.49 | 177.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 175.87 | 177.16 | 177.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 176.26 | 173.87 | 174.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 14:15:00 | 176.26 | 173.87 | 174.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 176.26 | 173.87 | 174.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 14:45:00 | 177.70 | 173.87 | 174.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 175.31 | 174.16 | 174.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:15:00 | 171.89 | 174.16 | 174.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 163.30 | 170.35 | 172.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-13 14:15:00 | 166.22 | 165.49 | 167.44 | SL hit (close>ema200) qty=0.50 sl=165.49 alert=retest2 |

### Cycle 58 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 169.31 | 163.96 | 163.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 173.83 | 167.45 | 165.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 173.31 | 174.33 | 171.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 173.31 | 174.33 | 171.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 172.19 | 173.30 | 172.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:30:00 | 169.92 | 173.30 | 172.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 171.34 | 172.91 | 171.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:00:00 | 171.34 | 172.91 | 171.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 171.50 | 172.63 | 171.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 14:45:00 | 172.92 | 171.89 | 171.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 15:15:00 | 169.50 | 171.41 | 171.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 15:15:00 | 169.50 | 171.41 | 171.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 09:15:00 | 169.09 | 170.95 | 171.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 162.15 | 157.83 | 160.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 14:15:00 | 162.15 | 157.83 | 160.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 162.15 | 157.83 | 160.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 15:00:00 | 162.15 | 157.83 | 160.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 160.50 | 158.36 | 160.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 158.58 | 158.36 | 160.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:00:00 | 159.26 | 159.26 | 160.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 13:00:00 | 160.02 | 159.41 | 160.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 162.75 | 160.71 | 160.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 162.75 | 160.71 | 160.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 165.27 | 163.04 | 162.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 12:15:00 | 165.68 | 166.25 | 164.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:45:00 | 165.60 | 166.25 | 164.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 163.39 | 165.67 | 164.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 14:00:00 | 163.39 | 165.67 | 164.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 164.10 | 165.36 | 164.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 14:45:00 | 163.13 | 165.36 | 164.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 163.57 | 165.00 | 164.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 161.42 | 165.00 | 164.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2025-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 10:15:00 | 163.48 | 164.19 | 164.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 157.35 | 161.97 | 163.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 13:15:00 | 160.19 | 160.18 | 161.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 14:00:00 | 160.19 | 160.18 | 161.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 165.35 | 160.99 | 161.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:30:00 | 158.74 | 160.12 | 161.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 13:30:00 | 158.45 | 159.50 | 160.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 158.38 | 159.42 | 160.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:30:00 | 158.70 | 159.13 | 160.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 159.99 | 159.25 | 159.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 13:45:00 | 159.79 | 159.25 | 159.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 158.29 | 159.06 | 159.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:15:00 | 160.25 | 159.06 | 159.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 160.25 | 159.30 | 159.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 10:30:00 | 157.84 | 159.07 | 159.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 12:00:00 | 157.92 | 158.84 | 159.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 13:00:00 | 157.75 | 158.62 | 159.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 15:00:00 | 157.85 | 158.60 | 159.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 158.98 | 158.68 | 159.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:15:00 | 160.42 | 158.68 | 159.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 159.72 | 158.89 | 159.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:45:00 | 159.99 | 158.89 | 159.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 158.60 | 158.83 | 159.16 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 160.90 | 159.32 | 159.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 160.90 | 159.32 | 159.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 10:15:00 | 162.61 | 159.98 | 159.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 164.05 | 164.35 | 162.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 09:15:00 | 165.44 | 164.35 | 162.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 168.05 | 166.76 | 165.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 09:45:00 | 170.10 | 168.14 | 166.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 14:00:00 | 169.17 | 168.10 | 167.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 169.79 | 168.15 | 167.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 11:00:00 | 169.33 | 168.25 | 167.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-26 11:15:00 | 187.11 | 171.65 | 169.12 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 09:15:00 | 175.53 | 179.93 | 180.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 164.93 | 168.28 | 170.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 157.50 | 156.46 | 160.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 157.50 | 156.46 | 160.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 157.50 | 156.46 | 160.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 15:15:00 | 154.60 | 156.27 | 158.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 12:15:00 | 159.37 | 157.78 | 157.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2025-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 12:15:00 | 159.37 | 157.78 | 157.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 13:15:00 | 161.15 | 158.45 | 157.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 15:15:00 | 168.09 | 168.14 | 166.82 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:15:00 | 169.15 | 168.14 | 166.82 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 166.26 | 167.76 | 166.77 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 166.26 | 167.76 | 166.77 | SL hit (close<ema400) qty=1.00 sl=166.77 alert=retest1 |

### Cycle 65 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 162.66 | 166.16 | 166.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 161.12 | 165.15 | 166.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 10:15:00 | 163.53 | 162.38 | 163.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 10:15:00 | 163.53 | 162.38 | 163.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 163.53 | 162.38 | 163.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:45:00 | 162.68 | 162.38 | 163.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 165.75 | 163.06 | 164.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:30:00 | 165.25 | 163.06 | 164.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 164.28 | 163.30 | 164.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 15:15:00 | 164.15 | 164.02 | 164.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-29 09:15:00 | 166.21 | 164.48 | 164.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 166.21 | 164.48 | 164.43 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 10:15:00 | 163.26 | 164.43 | 164.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 13:15:00 | 162.09 | 163.64 | 164.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 13:15:00 | 163.15 | 162.67 | 163.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 13:15:00 | 163.15 | 162.67 | 163.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 163.15 | 162.67 | 163.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 14:00:00 | 163.15 | 162.67 | 163.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 163.30 | 162.80 | 163.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 15:00:00 | 163.30 | 162.80 | 163.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 163.34 | 162.91 | 163.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:15:00 | 171.30 | 162.91 | 163.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 169.21 | 164.17 | 163.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 10:15:00 | 175.02 | 166.34 | 164.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-05 15:15:00 | 169.20 | 169.47 | 167.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 15:15:00 | 169.20 | 169.47 | 167.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 169.20 | 169.47 | 167.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:45:00 | 166.98 | 168.94 | 167.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 166.11 | 168.37 | 167.11 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 163.57 | 166.27 | 166.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 161.41 | 164.96 | 165.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 15:15:00 | 163.29 | 163.25 | 164.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 15:15:00 | 163.29 | 163.25 | 164.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 163.29 | 163.25 | 164.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 164.00 | 163.25 | 164.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 164.04 | 163.41 | 164.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:15:00 | 163.20 | 163.68 | 164.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 167.91 | 163.17 | 163.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 167.91 | 163.17 | 163.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 170.22 | 167.78 | 166.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 13:15:00 | 168.15 | 168.45 | 167.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 14:00:00 | 168.15 | 168.45 | 167.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 167.64 | 168.43 | 167.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:30:00 | 167.74 | 168.43 | 167.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 167.56 | 168.26 | 167.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:30:00 | 167.51 | 168.26 | 167.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 167.54 | 167.83 | 167.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:30:00 | 167.49 | 167.83 | 167.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 167.98 | 167.86 | 167.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:30:00 | 167.82 | 167.86 | 167.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 12:15:00 | 167.78 | 167.84 | 167.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 12:30:00 | 167.85 | 167.84 | 167.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 13:15:00 | 167.07 | 167.69 | 167.69 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 168.63 | 167.83 | 167.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 10:15:00 | 169.74 | 168.21 | 167.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 12:15:00 | 168.25 | 168.28 | 168.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 12:45:00 | 168.32 | 168.28 | 168.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 166.72 | 167.97 | 167.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 166.72 | 167.97 | 167.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 14:15:00 | 164.97 | 167.37 | 167.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 162.91 | 165.47 | 166.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 13:15:00 | 159.90 | 159.78 | 161.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 14:00:00 | 159.90 | 159.78 | 161.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 160.46 | 159.70 | 160.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 12:15:00 | 159.72 | 159.84 | 160.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 158.00 | 159.66 | 160.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 10:30:00 | 159.60 | 159.00 | 159.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 14:15:00 | 151.73 | 155.33 | 156.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 14:15:00 | 151.62 | 155.33 | 156.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-02 11:15:00 | 158.71 | 154.61 | 155.82 | SL hit (close>ema200) qty=0.50 sl=154.61 alert=retest2 |

### Cycle 74 — BUY (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 12:15:00 | 156.13 | 154.47 | 154.42 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 14:15:00 | 150.85 | 153.91 | 154.18 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 157.31 | 154.66 | 154.41 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 15:15:00 | 153.00 | 154.86 | 154.93 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 170.45 | 157.98 | 156.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 10:15:00 | 170.64 | 160.51 | 157.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 15:15:00 | 177.28 | 177.42 | 171.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:15:00 | 177.06 | 177.42 | 171.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 174.55 | 178.09 | 176.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 174.55 | 178.09 | 176.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 174.76 | 177.42 | 176.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 174.76 | 177.42 | 176.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2025-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 10:15:00 | 173.00 | 175.42 | 175.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 14:15:00 | 170.91 | 173.24 | 174.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 170.92 | 170.59 | 172.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 12:15:00 | 170.92 | 170.59 | 172.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 170.92 | 170.59 | 172.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:45:00 | 170.50 | 170.59 | 172.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 171.70 | 170.69 | 172.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 171.95 | 170.69 | 172.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 170.30 | 170.61 | 172.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 172.71 | 170.61 | 172.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 176.47 | 171.78 | 172.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:00:00 | 176.47 | 171.78 | 172.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 177.35 | 172.90 | 172.87 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 170.94 | 172.83 | 173.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 11:15:00 | 170.03 | 172.27 | 172.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 165.90 | 165.86 | 167.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 11:15:00 | 166.20 | 165.86 | 167.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 168.27 | 166.06 | 167.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:45:00 | 168.35 | 166.06 | 167.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 166.80 | 166.21 | 167.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 167.70 | 166.21 | 167.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 167.35 | 166.44 | 167.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 12:00:00 | 166.10 | 166.48 | 167.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 175.52 | 168.55 | 167.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 175.52 | 168.55 | 167.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 12:15:00 | 177.95 | 172.10 | 169.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 11:15:00 | 174.80 | 175.27 | 172.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 12:00:00 | 174.80 | 175.27 | 172.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 172.60 | 174.40 | 173.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 172.60 | 174.40 | 173.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 170.48 | 173.61 | 173.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 170.48 | 173.61 | 173.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2025-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 12:15:00 | 170.00 | 172.29 | 172.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 09:15:00 | 169.34 | 170.39 | 171.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 11:15:00 | 171.55 | 170.25 | 170.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 11:15:00 | 171.55 | 170.25 | 170.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 171.55 | 170.25 | 170.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:00:00 | 171.55 | 170.25 | 170.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 172.40 | 170.68 | 171.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:30:00 | 172.87 | 170.68 | 171.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 13:15:00 | 173.71 | 171.29 | 171.26 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 168.99 | 171.17 | 171.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 13:15:00 | 168.48 | 169.84 | 170.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 13:15:00 | 167.77 | 167.69 | 168.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 14:00:00 | 167.77 | 167.69 | 168.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 166.90 | 166.86 | 167.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:30:00 | 166.39 | 166.71 | 167.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 10:00:00 | 166.37 | 166.36 | 166.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 11:15:00 | 170.30 | 166.94 | 166.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 11:15:00 | 170.30 | 166.94 | 166.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 09:15:00 | 171.38 | 169.10 | 168.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 11:15:00 | 174.76 | 175.80 | 174.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 11:45:00 | 174.60 | 175.80 | 174.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 173.87 | 175.41 | 174.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:00:00 | 173.87 | 175.41 | 174.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 176.75 | 175.68 | 174.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 15:15:00 | 177.40 | 175.89 | 174.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 10:00:00 | 177.45 | 176.44 | 175.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 11:00:00 | 177.38 | 176.63 | 175.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 12:45:00 | 178.88 | 177.06 | 176.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 180.91 | 181.18 | 179.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 11:15:00 | 181.55 | 181.18 | 179.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 12:45:00 | 181.48 | 181.19 | 180.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 10:15:00 | 181.45 | 182.47 | 182.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 12:15:00 | 180.70 | 184.00 | 184.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 180.70 | 184.00 | 184.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 14:15:00 | 178.40 | 181.67 | 182.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 179.66 | 179.36 | 180.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 179.66 | 179.36 | 180.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 181.08 | 179.71 | 180.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 181.08 | 179.71 | 180.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 181.80 | 180.13 | 181.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 181.89 | 180.13 | 181.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 182.70 | 180.64 | 181.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:45:00 | 182.50 | 180.64 | 181.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 182.85 | 181.08 | 181.31 | EMA400 retest candle locked (from downside) |

### Cycle 88 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 187.05 | 182.28 | 181.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 09:15:00 | 187.91 | 186.32 | 184.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 11:15:00 | 185.92 | 186.49 | 185.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 12:00:00 | 185.92 | 186.49 | 185.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 186.70 | 186.53 | 185.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:45:00 | 185.99 | 186.53 | 185.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 177.04 | 184.64 | 184.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:00:00 | 177.04 | 184.64 | 184.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 175.20 | 182.75 | 183.77 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 12:15:00 | 189.04 | 183.65 | 183.52 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 184.19 | 186.25 | 186.51 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 187.79 | 186.54 | 186.50 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 10:15:00 | 184.85 | 186.17 | 186.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 12:15:00 | 183.50 | 185.48 | 185.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 184.26 | 184.07 | 185.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 09:45:00 | 183.80 | 184.07 | 185.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 184.01 | 184.06 | 184.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:30:00 | 185.10 | 184.06 | 184.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 182.43 | 183.72 | 184.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 10:15:00 | 181.64 | 183.72 | 184.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 13:15:00 | 182.15 | 182.97 | 183.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 13:15:00 | 185.09 | 183.40 | 183.98 | SL hit (close>static) qty=1.00 sl=184.50 alert=retest2 |

### Cycle 94 — BUY (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 11:15:00 | 185.35 | 184.17 | 184.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 15:15:00 | 186.30 | 184.97 | 184.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 184.50 | 184.88 | 184.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 184.50 | 184.88 | 184.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 184.50 | 184.88 | 184.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 184.50 | 184.88 | 184.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 184.80 | 184.86 | 184.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:45:00 | 184.51 | 184.86 | 184.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 186.11 | 185.11 | 184.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 12:00:00 | 186.11 | 185.11 | 184.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 185.00 | 185.55 | 185.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 188.88 | 185.55 | 185.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 14:15:00 | 187.71 | 188.99 | 189.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 187.71 | 188.99 | 189.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 15:15:00 | 187.60 | 188.71 | 188.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 11:15:00 | 189.32 | 188.78 | 188.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 11:15:00 | 189.32 | 188.78 | 188.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 189.32 | 188.78 | 188.86 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 13:15:00 | 189.29 | 188.93 | 188.92 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 187.70 | 188.68 | 188.81 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 190.99 | 188.97 | 188.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 10:15:00 | 192.86 | 190.79 | 189.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 15:15:00 | 189.25 | 190.97 | 190.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 15:15:00 | 189.25 | 190.97 | 190.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 189.25 | 190.97 | 190.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 187.61 | 190.97 | 190.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 187.30 | 190.24 | 190.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:30:00 | 186.92 | 190.24 | 190.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 10:15:00 | 186.92 | 189.57 | 189.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 15:15:00 | 182.87 | 187.34 | 188.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 187.17 | 186.70 | 188.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 187.17 | 186.70 | 188.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 186.00 | 185.79 | 186.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 186.78 | 185.79 | 186.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 188.55 | 186.47 | 187.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 188.55 | 186.47 | 187.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 188.18 | 186.81 | 187.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 13:30:00 | 187.71 | 187.02 | 187.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 14:00:00 | 187.84 | 187.02 | 187.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 189.48 | 187.51 | 187.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 189.48 | 187.51 | 187.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 191.50 | 188.51 | 187.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 189.15 | 189.30 | 188.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 189.15 | 189.30 | 188.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 187.96 | 189.03 | 188.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 187.96 | 189.03 | 188.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 188.40 | 188.90 | 188.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 190.47 | 188.90 | 188.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 14:15:00 | 189.03 | 189.06 | 188.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 15:00:00 | 188.87 | 189.03 | 188.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:45:00 | 188.96 | 188.84 | 188.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 10:15:00 | 187.70 | 188.62 | 188.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 187.70 | 188.62 | 188.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 186.39 | 188.17 | 188.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 187.45 | 186.74 | 187.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 187.45 | 186.74 | 187.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 187.45 | 186.74 | 187.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 187.45 | 186.74 | 187.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 187.70 | 186.93 | 187.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:30:00 | 187.79 | 186.93 | 187.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 184.96 | 186.54 | 187.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 12:30:00 | 184.17 | 186.12 | 187.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 184.16 | 185.59 | 186.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 10:30:00 | 184.20 | 185.40 | 186.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 11:15:00 | 184.36 | 185.40 | 186.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 185.83 | 185.32 | 185.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 186.01 | 185.32 | 185.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 186.60 | 185.58 | 185.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 186.60 | 185.58 | 185.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 186.90 | 185.84 | 185.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:15:00 | 187.18 | 185.84 | 185.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-09 13:15:00 | 186.74 | 186.18 | 186.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 13:15:00 | 186.74 | 186.18 | 186.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 191.12 | 187.33 | 186.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 12:15:00 | 186.99 | 187.73 | 187.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 12:15:00 | 186.99 | 187.73 | 187.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 186.99 | 187.73 | 187.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:00:00 | 186.99 | 187.73 | 187.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 187.61 | 187.71 | 187.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 15:00:00 | 188.06 | 187.78 | 187.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:45:00 | 188.65 | 187.70 | 187.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 10:15:00 | 188.23 | 187.70 | 187.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 11:15:00 | 188.40 | 187.66 | 187.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 186.42 | 187.42 | 187.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:00:00 | 186.42 | 187.42 | 187.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 184.97 | 186.93 | 187.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 184.97 | 186.93 | 187.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 14:15:00 | 183.75 | 185.98 | 186.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 12:15:00 | 185.56 | 185.14 | 185.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 12:15:00 | 185.56 | 185.14 | 185.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 185.56 | 185.14 | 185.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:00:00 | 185.56 | 185.14 | 185.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 186.25 | 185.33 | 185.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:00:00 | 186.25 | 185.33 | 185.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 186.94 | 185.65 | 185.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 188.86 | 185.65 | 185.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 188.75 | 186.27 | 186.17 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 11:15:00 | 186.38 | 187.99 | 188.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 185.56 | 187.50 | 187.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 12:15:00 | 186.96 | 186.18 | 186.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 12:15:00 | 186.96 | 186.18 | 186.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 186.96 | 186.18 | 186.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:45:00 | 187.35 | 186.18 | 186.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 188.80 | 186.70 | 187.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:45:00 | 188.25 | 186.70 | 187.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 14:15:00 | 190.82 | 187.53 | 187.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 15:15:00 | 193.00 | 188.62 | 187.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 14:15:00 | 192.99 | 193.12 | 191.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 15:00:00 | 192.99 | 193.12 | 191.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 192.27 | 193.49 | 192.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 192.27 | 193.49 | 192.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 192.52 | 193.30 | 192.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 13:45:00 | 194.15 | 193.40 | 192.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 191.20 | 192.71 | 192.36 | SL hit (close<static) qty=1.00 sl=191.90 alert=retest2 |

### Cycle 107 — SELL (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 11:15:00 | 189.60 | 191.69 | 191.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 188.50 | 190.77 | 191.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 185.71 | 185.51 | 187.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:00:00 | 185.71 | 185.51 | 187.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 188.07 | 186.02 | 187.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 188.07 | 186.02 | 187.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 186.55 | 186.13 | 187.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 187.97 | 186.13 | 187.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 186.10 | 186.12 | 187.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:00:00 | 185.90 | 186.08 | 186.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:30:00 | 185.95 | 185.96 | 186.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 188.50 | 186.83 | 186.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 188.50 | 186.83 | 186.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 189.55 | 188.07 | 187.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 189.13 | 189.19 | 188.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 13:00:00 | 189.13 | 189.19 | 188.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 188.21 | 189.29 | 188.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 188.21 | 189.29 | 188.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 188.40 | 189.11 | 188.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:00:00 | 188.77 | 189.04 | 188.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 186.72 | 188.39 | 188.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 186.72 | 188.39 | 188.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 185.32 | 187.78 | 188.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 12:15:00 | 188.33 | 187.59 | 188.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 12:15:00 | 188.33 | 187.59 | 188.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 188.33 | 187.59 | 188.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:45:00 | 188.42 | 187.59 | 188.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 188.64 | 187.80 | 188.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:00:00 | 188.64 | 187.80 | 188.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 188.68 | 187.98 | 188.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 188.68 | 187.98 | 188.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 189.38 | 188.36 | 188.30 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 12:15:00 | 187.40 | 188.66 | 188.70 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 189.14 | 188.76 | 188.74 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 187.45 | 188.55 | 188.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 14:15:00 | 187.08 | 187.91 | 188.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 188.00 | 186.28 | 186.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 188.00 | 186.28 | 186.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 188.00 | 186.28 | 186.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:45:00 | 187.83 | 186.28 | 186.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 188.51 | 186.73 | 187.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 188.51 | 186.73 | 187.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 189.95 | 187.37 | 187.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 12:15:00 | 193.68 | 188.64 | 187.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 192.38 | 192.51 | 191.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 10:00:00 | 192.38 | 192.51 | 191.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 190.61 | 191.92 | 191.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:45:00 | 190.60 | 191.92 | 191.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 190.98 | 191.73 | 191.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:45:00 | 190.26 | 191.73 | 191.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 191.60 | 191.71 | 191.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 15:15:00 | 193.00 | 191.71 | 191.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:45:00 | 193.67 | 192.23 | 191.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:45:00 | 192.85 | 192.23 | 191.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 11:15:00 | 192.39 | 192.23 | 191.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 191.34 | 192.05 | 191.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 12:00:00 | 191.34 | 192.05 | 191.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 190.49 | 191.74 | 191.53 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-20 12:15:00 | 190.49 | 191.74 | 191.53 | SL hit (close<static) qty=1.00 sl=190.61 alert=retest2 |

### Cycle 115 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 202.51 | 204.77 | 205.04 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 15:15:00 | 206.50 | 205.25 | 205.22 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 202.00 | 204.86 | 205.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 201.20 | 203.18 | 204.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 14:15:00 | 197.42 | 196.73 | 198.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 15:00:00 | 197.42 | 196.73 | 198.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 196.46 | 195.92 | 197.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 197.73 | 195.92 | 197.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 197.35 | 196.26 | 197.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:15:00 | 198.40 | 196.26 | 197.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 198.21 | 196.65 | 197.32 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 201.00 | 198.01 | 197.85 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 197.35 | 197.91 | 197.92 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 199.92 | 198.10 | 197.89 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 197.05 | 198.26 | 198.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 15:15:00 | 194.10 | 196.05 | 196.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 13:15:00 | 195.80 | 194.81 | 195.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 13:15:00 | 195.80 | 194.81 | 195.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 195.80 | 194.81 | 195.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:45:00 | 196.02 | 194.81 | 195.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 194.70 | 194.79 | 195.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 15:15:00 | 193.00 | 194.79 | 195.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 12:00:00 | 194.21 | 194.49 | 195.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:30:00 | 194.16 | 194.48 | 195.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:30:00 | 193.95 | 194.37 | 194.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 192.71 | 191.96 | 192.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 13:00:00 | 192.71 | 191.96 | 192.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 191.54 | 191.88 | 192.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:30:00 | 190.75 | 191.71 | 192.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 190.50 | 189.39 | 189.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 10:15:00 | 192.59 | 189.81 | 189.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 192.59 | 189.81 | 189.47 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 11:15:00 | 189.09 | 189.61 | 189.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 187.50 | 189.08 | 189.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 188.15 | 188.02 | 188.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 15:00:00 | 188.15 | 188.02 | 188.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 189.27 | 188.25 | 188.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:15:00 | 189.64 | 188.25 | 188.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 188.30 | 188.26 | 188.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 189.51 | 188.26 | 188.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 187.44 | 187.79 | 188.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 187.44 | 187.79 | 188.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 187.05 | 187.50 | 187.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 11:15:00 | 186.75 | 187.51 | 187.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 12:00:00 | 186.03 | 187.21 | 187.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 14:15:00 | 184.00 | 183.19 | 183.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 14:15:00 | 184.00 | 183.19 | 183.16 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 182.40 | 183.11 | 183.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 12:15:00 | 181.96 | 182.88 | 183.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 09:15:00 | 182.77 | 182.62 | 182.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 182.77 | 182.62 | 182.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 182.77 | 182.62 | 182.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:45:00 | 183.70 | 182.62 | 182.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 183.01 | 182.70 | 182.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 183.01 | 182.70 | 182.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 181.80 | 182.52 | 182.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 15:00:00 | 180.92 | 182.14 | 182.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 11:15:00 | 179.22 | 178.54 | 178.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 11:15:00 | 179.22 | 178.54 | 178.49 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 176.44 | 178.17 | 178.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 11:15:00 | 176.38 | 177.81 | 178.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 11:15:00 | 176.69 | 176.40 | 177.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 11:15:00 | 176.69 | 176.40 | 177.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 176.69 | 176.40 | 177.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:45:00 | 177.60 | 176.40 | 177.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 178.19 | 176.76 | 177.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:45:00 | 178.30 | 176.76 | 177.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 178.89 | 177.19 | 177.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:45:00 | 178.70 | 177.19 | 177.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 180.08 | 177.77 | 177.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 185.50 | 179.89 | 178.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 12:15:00 | 181.46 | 181.72 | 179.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-31 13:00:00 | 181.46 | 181.72 | 179.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 185.37 | 185.31 | 184.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:30:00 | 187.35 | 186.09 | 184.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 11:30:00 | 187.00 | 187.30 | 187.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 12:45:00 | 186.74 | 187.19 | 187.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:00:00 | 186.70 | 187.09 | 187.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 187.48 | 187.17 | 187.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:30:00 | 186.90 | 187.17 | 187.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-07 15:15:00 | 184.90 | 186.72 | 186.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 15:15:00 | 184.90 | 186.72 | 186.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 181.37 | 183.82 | 184.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 10:15:00 | 181.48 | 181.01 | 181.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 11:00:00 | 181.48 | 181.01 | 181.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 181.03 | 181.02 | 181.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:45:00 | 180.70 | 180.97 | 181.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 13:15:00 | 182.40 | 181.26 | 181.80 | SL hit (close>static) qty=1.00 sl=181.87 alert=retest2 |

### Cycle 130 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 174.79 | 169.50 | 169.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 176.40 | 170.88 | 169.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 172.05 | 176.60 | 174.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 172.05 | 176.60 | 174.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 172.05 | 176.60 | 174.89 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 10:15:00 | 172.85 | 173.94 | 174.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 171.43 | 173.44 | 173.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 166.70 | 166.46 | 169.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 14:00:00 | 166.70 | 166.46 | 169.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 169.34 | 167.03 | 169.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 169.34 | 167.03 | 169.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 169.48 | 167.52 | 169.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 172.71 | 167.52 | 169.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 171.15 | 168.25 | 169.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 13:00:00 | 170.28 | 169.76 | 169.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 13:15:00 | 171.57 | 170.12 | 170.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 171.57 | 170.12 | 170.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 171.74 | 170.45 | 170.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 169.74 | 170.30 | 170.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 10:15:00 | 169.74 | 170.30 | 170.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 169.74 | 170.30 | 170.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 169.74 | 170.30 | 170.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 170.17 | 170.28 | 170.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 13:30:00 | 171.00 | 170.33 | 170.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 169.25 | 170.49 | 170.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 169.25 | 170.49 | 170.59 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 12:15:00 | 173.81 | 171.17 | 170.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 176.26 | 172.99 | 171.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 09:15:00 | 175.80 | 176.00 | 174.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 09:45:00 | 175.92 | 176.00 | 174.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 175.00 | 175.69 | 174.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:30:00 | 174.38 | 175.69 | 174.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 174.67 | 175.49 | 174.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:30:00 | 174.70 | 175.49 | 174.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 174.85 | 175.36 | 174.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:30:00 | 174.63 | 175.36 | 174.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 175.00 | 175.29 | 174.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 174.50 | 175.29 | 174.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 174.64 | 175.16 | 174.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:45:00 | 174.61 | 175.16 | 174.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 176.22 | 175.37 | 174.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 14:45:00 | 177.49 | 175.96 | 175.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:45:00 | 177.07 | 176.63 | 175.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 12:30:00 | 177.50 | 177.09 | 176.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 11:15:00 | 175.97 | 176.66 | 176.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 11:15:00 | 175.97 | 176.66 | 176.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 12:15:00 | 174.50 | 176.23 | 176.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 177.36 | 175.98 | 176.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 177.36 | 175.98 | 176.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 177.36 | 175.98 | 176.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 177.57 | 175.98 | 176.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 173.30 | 175.45 | 175.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:15:00 | 173.10 | 174.48 | 175.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 15:15:00 | 173.05 | 174.22 | 175.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 13:15:00 | 177.30 | 175.65 | 175.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — BUY (started 2026-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 13:15:00 | 177.30 | 175.65 | 175.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 14:15:00 | 178.69 | 176.26 | 175.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 175.60 | 176.51 | 176.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 10:15:00 | 175.60 | 176.51 | 176.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 175.60 | 176.51 | 176.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:45:00 | 176.07 | 176.51 | 176.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 175.10 | 176.23 | 175.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 175.10 | 176.23 | 175.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 175.00 | 175.98 | 175.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:00:00 | 175.00 | 175.98 | 175.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 174.11 | 175.61 | 175.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 173.09 | 175.10 | 175.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 11:15:00 | 173.08 | 172.90 | 173.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 11:15:00 | 173.08 | 172.90 | 173.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 173.08 | 172.90 | 173.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 169.70 | 172.65 | 173.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 14:15:00 | 161.21 | 164.51 | 166.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 10:15:00 | 152.73 | 157.42 | 160.75 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 138 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 160.69 | 157.82 | 157.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 13:15:00 | 161.41 | 159.03 | 158.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 155.20 | 158.83 | 158.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 155.20 | 158.83 | 158.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 155.20 | 158.83 | 158.34 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 153.19 | 157.70 | 157.87 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 09:15:00 | 165.37 | 158.39 | 157.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 168.08 | 164.31 | 161.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 12:15:00 | 168.72 | 168.82 | 166.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-12 13:00:00 | 168.72 | 168.82 | 166.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 167.40 | 168.31 | 166.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 11:15:00 | 170.15 | 167.80 | 166.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 12:00:00 | 169.94 | 168.23 | 167.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 09:30:00 | 170.22 | 169.54 | 168.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 11:45:00 | 169.17 | 169.56 | 168.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 172.84 | 170.16 | 169.05 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-19 12:15:00 | 168.70 | 169.61 | 169.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 168.70 | 169.61 | 169.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 167.05 | 169.10 | 169.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-19 15:15:00 | 169.57 | 169.04 | 169.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 15:15:00 | 169.57 | 169.04 | 169.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 169.57 | 169.04 | 169.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:15:00 | 169.05 | 169.04 | 169.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 167.65 | 168.76 | 169.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 10:15:00 | 167.46 | 168.76 | 169.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:00:00 | 167.28 | 168.26 | 168.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 15:15:00 | 169.90 | 169.20 | 169.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 15:15:00 | 169.90 | 169.20 | 169.18 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 162.05 | 167.77 | 168.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 161.95 | 166.60 | 167.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 161.37 | 160.99 | 163.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 161.90 | 160.99 | 163.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 171.50 | 163.05 | 163.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 171.50 | 163.05 | 163.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 171.18 | 164.68 | 164.60 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 164.00 | 165.85 | 166.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 161.95 | 164.47 | 165.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 170.04 | 164.43 | 164.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 170.04 | 164.43 | 164.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 170.04 | 164.43 | 164.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 168.43 | 164.43 | 164.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 172.80 | 166.10 | 165.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 174.90 | 167.86 | 166.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 164.48 | 170.58 | 168.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 164.48 | 170.58 | 168.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 164.48 | 170.58 | 168.69 | EMA400 retest candle locked (from upside) |

### Cycle 147 — SELL (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 12:15:00 | 164.80 | 167.62 | 167.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 10:15:00 | 163.69 | 166.00 | 166.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 165.50 | 165.40 | 166.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 166.00 | 165.56 | 166.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 166.00 | 165.56 | 166.15 | EMA400 retest candle locked (from downside) |

### Cycle 148 — BUY (started 2026-04-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 13:15:00 | 167.17 | 166.41 | 166.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 176.16 | 169.25 | 167.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 177.05 | 177.69 | 175.22 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 180.00 | 177.69 | 175.22 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 182.27 | 181.34 | 178.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 182.91 | 181.34 | 178.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:15:00 | 189.00 | 184.67 | 183.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 182.93 | 184.67 | 183.21 | SL hit (close<ema400) qty=0.50 sl=184.67 alert=retest1 |

### Cycle 149 — SELL (started 2026-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 13:15:00 | 179.77 | 182.39 | 182.50 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2026-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 14:15:00 | 185.09 | 182.93 | 182.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 185.38 | 183.75 | 183.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 10:15:00 | 183.74 | 183.75 | 183.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 10:30:00 | 183.94 | 183.75 | 183.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 11:15:00 | 183.00 | 183.60 | 183.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 12:00:00 | 183.00 | 183.60 | 183.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 12:15:00 | 182.10 | 183.30 | 183.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 13:00:00 | 182.10 | 183.30 | 183.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 13:15:00 | 182.00 | 183.04 | 183.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 13:45:00 | 181.85 | 183.04 | 183.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 14:15:00 | 181.95 | 182.82 | 182.90 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 09:15:00 | 185.50 | 183.23 | 183.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 12:15:00 | 185.90 | 184.36 | 183.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 09:15:00 | 181.92 | 184.14 | 183.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 181.92 | 184.14 | 183.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 181.92 | 184.14 | 183.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:00:00 | 181.92 | 184.14 | 183.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — SELL (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 10:15:00 | 181.00 | 183.51 | 183.57 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2026-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 13:15:00 | 185.78 | 183.30 | 183.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 14:15:00 | 187.32 | 184.10 | 183.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 13:15:00 | 185.20 | 185.41 | 184.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-23 14:00:00 | 185.20 | 185.41 | 184.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 183.80 | 185.09 | 184.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 15:00:00 | 183.80 | 185.09 | 184.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 184.23 | 184.91 | 184.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 09:15:00 | 186.34 | 184.91 | 184.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 12:45:00 | 184.87 | 184.85 | 184.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 13:30:00 | 184.88 | 184.90 | 184.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 15:15:00 | 181.99 | 184.17 | 184.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 15:15:00 | 181.99 | 184.17 | 184.38 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 186.00 | 184.75 | 184.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 12:15:00 | 187.70 | 185.68 | 185.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 185.59 | 185.66 | 185.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 14:00:00 | 185.59 | 185.66 | 185.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 185.34 | 185.60 | 185.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 185.34 | 185.60 | 185.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 184.90 | 185.46 | 185.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 184.94 | 185.46 | 185.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 184.50 | 185.27 | 185.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 185.70 | 185.27 | 185.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — SELL (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 10:15:00 | 183.95 | 185.00 | 185.06 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 186.60 | 184.63 | 184.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 188.40 | 185.86 | 185.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 11:15:00 | 200.07 | 200.26 | 197.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 11:30:00 | 200.95 | 200.26 | 197.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 198.47 | 200.28 | 198.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 198.47 | 200.28 | 198.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 199.29 | 200.08 | 198.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:30:00 | 197.97 | 200.08 | 198.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 198.70 | 199.80 | 198.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 198.70 | 199.80 | 198.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 197.75 | 199.39 | 198.82 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-17 11:30:00 | 224.50 | 2024-05-18 09:15:00 | 229.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-05-21 09:15:00 | 224.00 | 2024-05-28 10:15:00 | 212.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-21 09:15:00 | 224.00 | 2024-05-29 09:15:00 | 214.00 | STOP_HIT | 0.50 | 4.46% |
| BUY | retest2 | 2024-06-07 09:15:00 | 215.90 | 2024-06-11 09:15:00 | 214.64 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-06-07 12:45:00 | 215.70 | 2024-06-11 09:15:00 | 214.64 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2024-06-07 14:15:00 | 215.35 | 2024-06-11 09:15:00 | 214.64 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2024-06-07 14:45:00 | 215.30 | 2024-06-11 09:15:00 | 214.64 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2024-06-10 09:15:00 | 217.42 | 2024-06-11 09:15:00 | 214.64 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-06-21 09:15:00 | 224.27 | 2024-06-21 12:15:00 | 220.58 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-06-21 09:45:00 | 223.54 | 2024-06-21 12:15:00 | 220.58 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-06-21 11:30:00 | 223.66 | 2024-06-21 12:15:00 | 220.58 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-06-27 10:45:00 | 216.00 | 2024-07-09 15:15:00 | 214.80 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2024-06-28 11:00:00 | 215.00 | 2024-07-09 15:15:00 | 214.80 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2024-06-28 14:15:00 | 216.00 | 2024-07-09 15:15:00 | 214.80 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2024-07-01 10:30:00 | 216.18 | 2024-07-09 15:15:00 | 214.80 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2024-07-08 12:30:00 | 210.33 | 2024-07-09 15:15:00 | 214.80 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-07-12 15:00:00 | 228.26 | 2024-07-16 13:15:00 | 217.93 | STOP_HIT | 1.00 | -4.53% |
| BUY | retest2 | 2024-07-16 09:15:00 | 220.89 | 2024-07-16 13:15:00 | 217.93 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-07-23 12:15:00 | 210.05 | 2024-08-01 11:15:00 | 212.58 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-07-23 13:00:00 | 210.77 | 2024-08-01 11:15:00 | 212.58 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-07-23 14:15:00 | 210.97 | 2024-08-01 11:15:00 | 212.58 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-07-24 14:15:00 | 210.90 | 2024-08-01 11:15:00 | 212.58 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-07-26 13:15:00 | 208.90 | 2024-08-01 11:15:00 | 212.58 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-07-29 14:45:00 | 209.85 | 2024-08-01 11:15:00 | 212.58 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-07-30 14:00:00 | 210.02 | 2024-08-01 11:15:00 | 212.58 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-07-31 12:30:00 | 210.29 | 2024-08-01 11:15:00 | 212.58 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-08-08 12:45:00 | 205.15 | 2024-08-19 10:15:00 | 204.71 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2024-08-08 13:30:00 | 205.57 | 2024-08-19 10:15:00 | 204.71 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2024-08-09 09:45:00 | 205.71 | 2024-08-19 10:15:00 | 204.71 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2024-08-09 10:15:00 | 204.67 | 2024-08-19 10:15:00 | 204.71 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2024-08-12 09:45:00 | 205.70 | 2024-08-19 10:15:00 | 204.71 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2024-08-12 11:15:00 | 205.92 | 2024-08-19 10:15:00 | 204.71 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2024-08-12 12:15:00 | 205.85 | 2024-08-19 10:15:00 | 204.71 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2024-08-12 15:15:00 | 205.50 | 2024-08-19 10:15:00 | 204.71 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2024-08-13 10:15:00 | 204.77 | 2024-08-19 10:15:00 | 204.71 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2024-08-20 15:15:00 | 206.00 | 2024-08-27 10:15:00 | 209.39 | STOP_HIT | 1.00 | 1.65% |
| SELL | retest2 | 2024-09-03 13:15:00 | 209.03 | 2024-09-03 14:15:00 | 212.14 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-09-18 13:00:00 | 210.65 | 2024-09-26 12:15:00 | 209.55 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2024-10-01 11:30:00 | 204.86 | 2024-10-01 14:15:00 | 207.15 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-10-01 12:15:00 | 204.10 | 2024-10-01 14:15:00 | 207.15 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-10-03 09:15:00 | 204.55 | 2024-10-14 12:15:00 | 194.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 15:15:00 | 205.00 | 2024-10-14 12:15:00 | 194.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 204.55 | 2024-10-14 15:15:00 | 196.24 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2024-10-03 15:15:00 | 205.00 | 2024-10-14 15:15:00 | 196.24 | STOP_HIT | 0.50 | 4.27% |
| SELL | retest2 | 2024-10-07 10:00:00 | 201.30 | 2024-10-21 14:15:00 | 191.63 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2024-10-08 15:15:00 | 201.15 | 2024-10-21 14:15:00 | 191.63 | PARTIAL | 0.50 | 4.73% |
| SELL | retest2 | 2024-10-09 09:45:00 | 201.72 | 2024-10-21 15:15:00 | 191.24 | PARTIAL | 0.50 | 5.20% |
| SELL | retest2 | 2024-10-09 15:15:00 | 201.72 | 2024-10-21 15:15:00 | 191.09 | PARTIAL | 0.50 | 5.27% |
| SELL | retest2 | 2024-10-10 09:45:00 | 200.89 | 2024-10-22 09:15:00 | 190.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 09:45:00 | 199.41 | 2024-10-22 10:15:00 | 189.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-07 10:00:00 | 201.30 | 2024-10-23 09:15:00 | 181.55 | TARGET_HIT | 0.50 | 9.81% |
| SELL | retest2 | 2024-10-08 15:15:00 | 201.15 | 2024-10-23 09:15:00 | 181.55 | TARGET_HIT | 0.50 | 9.74% |
| SELL | retest2 | 2024-10-09 09:45:00 | 201.72 | 2024-10-24 09:15:00 | 181.17 | TARGET_HIT | 0.50 | 10.19% |
| SELL | retest2 | 2024-10-09 15:15:00 | 201.72 | 2024-10-24 09:15:00 | 181.03 | TARGET_HIT | 0.50 | 10.25% |
| SELL | retest2 | 2024-10-10 09:45:00 | 200.89 | 2024-10-24 09:15:00 | 180.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-15 09:45:00 | 199.41 | 2024-10-25 09:15:00 | 179.47 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-11-05 09:30:00 | 209.84 | 2024-11-07 15:15:00 | 207.50 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-11-07 09:15:00 | 209.80 | 2024-11-07 15:15:00 | 207.50 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-11-19 15:00:00 | 191.00 | 2024-11-25 09:15:00 | 198.99 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2024-11-26 12:45:00 | 202.02 | 2024-12-02 13:15:00 | 206.81 | STOP_HIT | 1.00 | 2.37% |
| BUY | retest2 | 2024-11-26 13:45:00 | 202.19 | 2024-12-02 13:15:00 | 206.81 | STOP_HIT | 1.00 | 2.28% |
| SELL | retest2 | 2024-12-12 09:30:00 | 204.85 | 2024-12-19 09:15:00 | 194.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 09:30:00 | 204.85 | 2024-12-20 10:15:00 | 194.54 | STOP_HIT | 0.50 | 5.03% |
| SELL | retest2 | 2025-01-08 12:15:00 | 185.52 | 2025-01-08 14:15:00 | 189.10 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-01-23 09:15:00 | 178.29 | 2025-01-23 09:15:00 | 181.30 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-01-24 09:15:00 | 199.62 | 2025-01-27 15:15:00 | 185.30 | STOP_HIT | 1.00 | -7.17% |
| SELL | retest2 | 2025-01-30 13:15:00 | 177.80 | 2025-02-04 09:15:00 | 177.57 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-02-11 09:15:00 | 171.89 | 2025-02-12 09:15:00 | 163.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 171.89 | 2025-02-13 14:15:00 | 166.22 | STOP_HIT | 0.50 | 3.30% |
| BUY | retest2 | 2025-02-24 14:45:00 | 172.92 | 2025-02-24 15:15:00 | 169.50 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-03-04 09:15:00 | 158.58 | 2025-03-05 09:15:00 | 162.75 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2025-03-04 12:00:00 | 159.26 | 2025-03-05 09:15:00 | 162.75 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-03-04 13:00:00 | 160.02 | 2025-03-05 09:15:00 | 162.75 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-03-12 11:30:00 | 158.74 | 2025-03-19 09:15:00 | 160.90 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-03-12 13:30:00 | 158.45 | 2025-03-19 09:15:00 | 160.90 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-03-13 09:15:00 | 158.38 | 2025-03-19 09:15:00 | 160.90 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-03-13 11:30:00 | 158.70 | 2025-03-19 09:15:00 | 160.90 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-03-17 10:30:00 | 157.84 | 2025-03-19 09:15:00 | 160.90 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-03-17 12:00:00 | 157.92 | 2025-03-19 09:15:00 | 160.90 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-03-17 13:00:00 | 157.75 | 2025-03-19 09:15:00 | 160.90 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-03-17 15:00:00 | 157.85 | 2025-03-19 09:15:00 | 160.90 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-03-25 09:45:00 | 170.10 | 2025-03-26 11:15:00 | 187.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-25 14:00:00 | 169.17 | 2025-03-26 11:15:00 | 186.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-26 09:15:00 | 169.79 | 2025-03-26 11:15:00 | 186.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-26 11:00:00 | 169.33 | 2025-03-26 11:15:00 | 186.26 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-09 15:15:00 | 154.60 | 2025-04-15 12:15:00 | 159.37 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest1 | 2025-04-23 09:15:00 | 169.15 | 2025-04-23 09:15:00 | 166.26 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-04-23 14:45:00 | 167.94 | 2025-04-25 09:15:00 | 162.66 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2025-04-24 09:30:00 | 168.08 | 2025-04-25 09:15:00 | 162.66 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2025-04-24 10:15:00 | 167.81 | 2025-04-25 09:15:00 | 162.66 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2025-04-24 10:45:00 | 167.79 | 2025-04-25 09:15:00 | 162.66 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2025-04-28 15:15:00 | 164.15 | 2025-04-29 09:15:00 | 166.21 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-05-08 13:15:00 | 163.20 | 2025-05-12 09:15:00 | 167.91 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2025-05-26 12:15:00 | 159.72 | 2025-05-30 14:15:00 | 151.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-27 09:15:00 | 158.00 | 2025-05-30 14:15:00 | 151.62 | PARTIAL | 0.50 | 4.04% |
| SELL | retest2 | 2025-05-26 12:15:00 | 159.72 | 2025-06-02 11:15:00 | 158.71 | STOP_HIT | 0.50 | 0.63% |
| SELL | retest2 | 2025-05-27 09:15:00 | 158.00 | 2025-06-02 11:15:00 | 158.71 | STOP_HIT | 0.50 | -0.45% |
| SELL | retest2 | 2025-05-28 10:30:00 | 159.60 | 2025-06-04 12:15:00 | 156.13 | STOP_HIT | 1.00 | 2.17% |
| SELL | retest2 | 2025-06-23 12:00:00 | 166.10 | 2025-06-24 09:15:00 | 175.52 | STOP_HIT | 1.00 | -5.67% |
| SELL | retest2 | 2025-07-04 11:30:00 | 166.39 | 2025-07-08 11:15:00 | 170.30 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-07-07 10:00:00 | 166.37 | 2025-07-08 11:15:00 | 170.30 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-07-14 15:15:00 | 177.40 | 2025-07-25 12:15:00 | 180.70 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2025-07-15 10:00:00 | 177.45 | 2025-07-25 12:15:00 | 180.70 | STOP_HIT | 1.00 | 1.83% |
| BUY | retest2 | 2025-07-15 11:00:00 | 177.38 | 2025-07-25 12:15:00 | 180.70 | STOP_HIT | 1.00 | 1.87% |
| BUY | retest2 | 2025-07-16 12:45:00 | 178.88 | 2025-07-25 12:15:00 | 180.70 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2025-07-18 11:15:00 | 181.55 | 2025-07-25 12:15:00 | 180.70 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-07-18 12:45:00 | 181.48 | 2025-07-25 12:15:00 | 180.70 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-07-23 10:15:00 | 181.45 | 2025-07-25 12:15:00 | 180.70 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-08-12 10:15:00 | 181.64 | 2025-08-12 13:15:00 | 185.09 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-08-12 13:15:00 | 182.15 | 2025-08-12 13:15:00 | 185.09 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-08-18 09:15:00 | 188.88 | 2025-08-21 14:15:00 | 187.71 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-09-01 13:30:00 | 187.71 | 2025-09-01 14:15:00 | 189.48 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-09-01 14:00:00 | 187.84 | 2025-09-01 14:15:00 | 189.48 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-09-03 09:15:00 | 190.47 | 2025-09-04 10:15:00 | 187.70 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-09-03 14:15:00 | 189.03 | 2025-09-04 10:15:00 | 187.70 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-09-03 15:00:00 | 188.87 | 2025-09-04 10:15:00 | 187.70 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-09-04 09:45:00 | 188.96 | 2025-09-04 10:15:00 | 187.70 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-09-05 12:30:00 | 184.17 | 2025-09-09 13:15:00 | 186.74 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-09-08 09:30:00 | 184.16 | 2025-09-09 13:15:00 | 186.74 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-09-08 10:30:00 | 184.20 | 2025-09-09 13:15:00 | 186.74 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-09-08 11:15:00 | 184.36 | 2025-09-09 13:15:00 | 186.74 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-09-10 15:00:00 | 188.06 | 2025-09-11 12:15:00 | 184.97 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-09-11 09:45:00 | 188.65 | 2025-09-11 12:15:00 | 184.97 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-09-11 10:15:00 | 188.23 | 2025-09-11 12:15:00 | 184.97 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-09-11 11:15:00 | 188.40 | 2025-09-11 12:15:00 | 184.97 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-09-24 13:45:00 | 194.15 | 2025-09-25 09:15:00 | 191.20 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-09-30 11:00:00 | 185.90 | 2025-10-01 14:15:00 | 188.50 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-09-30 11:30:00 | 185.95 | 2025-10-01 14:15:00 | 188.50 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-10-07 13:00:00 | 188.77 | 2025-10-08 09:15:00 | 186.72 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-10-17 15:15:00 | 193.00 | 2025-10-20 12:15:00 | 190.49 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-10-20 09:45:00 | 193.67 | 2025-10-20 12:15:00 | 190.49 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-10-20 10:45:00 | 192.85 | 2025-10-20 12:15:00 | 190.49 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-10-20 11:15:00 | 192.39 | 2025-10-20 12:15:00 | 190.49 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-11-19 15:15:00 | 193.00 | 2025-12-01 10:15:00 | 192.59 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-11-20 12:00:00 | 194.21 | 2025-12-01 10:15:00 | 192.59 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2025-11-20 13:30:00 | 194.16 | 2025-12-01 10:15:00 | 192.59 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest2 | 2025-11-20 14:30:00 | 193.95 | 2025-12-01 10:15:00 | 192.59 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2025-11-24 14:30:00 | 190.75 | 2025-12-01 10:15:00 | 192.59 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-11-27 15:15:00 | 190.50 | 2025-12-01 10:15:00 | 192.59 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-12-08 11:15:00 | 186.75 | 2025-12-12 14:15:00 | 184.00 | STOP_HIT | 1.00 | 1.47% |
| SELL | retest2 | 2025-12-08 12:00:00 | 186.03 | 2025-12-12 14:15:00 | 184.00 | STOP_HIT | 1.00 | 1.09% |
| SELL | retest2 | 2025-12-16 15:00:00 | 180.92 | 2025-12-26 11:15:00 | 179.22 | STOP_HIT | 1.00 | 0.94% |
| BUY | retest2 | 2026-01-05 11:30:00 | 187.35 | 2026-01-07 15:15:00 | 184.90 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-01-07 11:30:00 | 187.00 | 2026-01-07 15:15:00 | 184.90 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-01-07 12:45:00 | 186.74 | 2026-01-07 15:15:00 | 184.90 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2026-01-07 14:00:00 | 186.70 | 2026-01-07 15:15:00 | 184.90 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-01-14 12:45:00 | 180.70 | 2026-01-14 13:15:00 | 182.40 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-01-14 13:45:00 | 180.79 | 2026-01-16 09:15:00 | 182.17 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-01-14 14:45:00 | 180.50 | 2026-01-16 09:15:00 | 182.17 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-01-16 13:00:00 | 180.62 | 2026-01-21 09:15:00 | 171.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 11:45:00 | 175.80 | 2026-01-22 10:15:00 | 167.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 13:00:00 | 180.62 | 2026-01-22 15:15:00 | 168.75 | STOP_HIT | 0.50 | 6.57% |
| SELL | retest2 | 2026-01-20 11:45:00 | 175.80 | 2026-01-22 15:15:00 | 168.75 | STOP_HIT | 0.50 | 4.01% |
| SELL | retest2 | 2026-02-03 13:00:00 | 170.28 | 2026-02-03 13:15:00 | 171.57 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-02-04 13:30:00 | 171.00 | 2026-02-06 09:15:00 | 169.25 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2026-02-11 14:45:00 | 177.49 | 2026-02-16 11:15:00 | 175.97 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-02-12 09:45:00 | 177.07 | 2026-02-16 11:15:00 | 175.97 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-02-13 12:30:00 | 177.50 | 2026-02-16 11:15:00 | 175.97 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-02-17 14:15:00 | 173.10 | 2026-02-18 13:15:00 | 177.30 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2026-02-17 15:15:00 | 173.05 | 2026-02-18 13:15:00 | 177.30 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-02-24 09:15:00 | 169.70 | 2026-02-27 14:15:00 | 161.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 169.70 | 2026-03-04 10:15:00 | 152.73 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-13 11:15:00 | 170.15 | 2026-03-19 12:15:00 | 168.70 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-03-13 12:00:00 | 169.94 | 2026-03-19 12:15:00 | 168.70 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-03-16 09:30:00 | 170.22 | 2026-03-19 12:15:00 | 168.70 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2026-03-16 11:45:00 | 169.17 | 2026-03-19 12:15:00 | 168.70 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2026-03-20 10:15:00 | 167.46 | 2026-03-20 15:15:00 | 169.90 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-03-20 12:00:00 | 167.28 | 2026-03-20 15:15:00 | 169.90 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest1 | 2026-04-10 09:15:00 | 180.00 | 2026-04-16 09:15:00 | 189.00 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-10 09:15:00 | 180.00 | 2026-04-16 09:15:00 | 182.93 | STOP_HIT | 0.50 | 1.63% |
| BUY | retest2 | 2026-04-13 10:15:00 | 182.91 | 2026-04-16 13:15:00 | 179.77 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-04-24 09:15:00 | 186.34 | 2026-04-24 15:15:00 | 181.99 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-04-24 12:45:00 | 184.87 | 2026-04-24 15:15:00 | 181.99 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2026-04-24 13:30:00 | 184.88 | 2026-04-24 15:15:00 | 181.99 | STOP_HIT | 1.00 | -1.56% |
