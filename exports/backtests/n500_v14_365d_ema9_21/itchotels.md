# ITC Hotels Ltd. (ITCHOTELS)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 164.58
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 60 |
| ALERT1 | 46 |
| ALERT2 | 45 |
| ALERT2_SKIP | 24 |
| ALERT3 | 122 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 68 |
| PARTIAL | 9 |
| TARGET_HIT | 4 |
| STOP_HIT | 65 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 78 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 50
- **Target hits / Stop hits / Partials:** 4 / 65 / 9
- **Avg / median % per leg:** 1.28% / -0.39%
- **Sum % (uncompounded):** 99.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 5 | 14.3% | 4 | 31 | 0 | 0.49% | 17.0% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.68% | -2.7% |
| BUY @ 3rd Alert (retest2) | 34 | 5 | 14.7% | 4 | 30 | 0 | 0.58% | 19.7% |
| SELL (all) | 43 | 23 | 53.5% | 0 | 34 | 9 | 1.93% | 82.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 43 | 23 | 53.5% | 0 | 34 | 9 | 1.93% | 82.8% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.68% | -2.7% |
| retest2 (combined) | 77 | 28 | 36.4% | 4 | 64 | 9 | 1.33% | 102.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 197.79 | 188.54 | 188.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 199.01 | 195.68 | 192.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 13:15:00 | 200.60 | 200.91 | 199.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 13:15:00 | 200.60 | 200.91 | 199.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 200.60 | 200.91 | 199.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:00:00 | 200.60 | 200.91 | 199.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 202.23 | 203.02 | 202.28 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 200.40 | 201.88 | 201.96 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 204.83 | 202.54 | 202.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 13:15:00 | 208.63 | 205.02 | 203.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 14:15:00 | 214.33 | 215.06 | 211.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 15:00:00 | 214.33 | 215.06 | 211.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 214.17 | 215.37 | 213.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 214.17 | 215.37 | 213.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 214.20 | 214.99 | 213.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:45:00 | 213.99 | 214.99 | 213.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 213.61 | 214.71 | 213.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:30:00 | 213.67 | 214.71 | 213.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 213.14 | 214.40 | 213.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 213.14 | 214.40 | 213.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 213.24 | 214.17 | 213.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 15:15:00 | 213.75 | 214.17 | 213.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 209.43 | 213.15 | 213.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 209.43 | 213.15 | 213.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 10:15:00 | 209.00 | 212.32 | 212.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 213.75 | 209.08 | 209.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 213.75 | 209.08 | 209.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 213.75 | 209.08 | 209.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 213.75 | 209.08 | 209.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 214.61 | 210.19 | 210.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 214.61 | 210.19 | 210.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 11:15:00 | 216.71 | 211.49 | 210.85 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 14:15:00 | 212.93 | 213.67 | 213.73 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 217.75 | 214.41 | 214.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 218.31 | 215.79 | 214.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 15:15:00 | 218.15 | 218.18 | 217.08 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 09:15:00 | 223.47 | 218.18 | 217.08 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 217.49 | 219.48 | 218.83 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-11 12:15:00 | 217.49 | 219.48 | 218.83 | SL hit (close<ema400) qty=1.00 sl=218.83 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-06-11 13:00:00 | 217.49 | 219.48 | 218.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 216.70 | 218.92 | 218.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 216.65 | 218.92 | 218.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 217.10 | 218.24 | 218.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 215.99 | 217.79 | 218.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 11:15:00 | 214.24 | 214.16 | 215.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 214.30 | 214.19 | 215.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 214.30 | 214.19 | 215.04 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 216.77 | 215.06 | 215.05 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 214.00 | 215.04 | 215.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 11:15:00 | 213.39 | 214.42 | 214.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 211.76 | 211.29 | 212.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 211.76 | 211.29 | 212.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 211.76 | 211.29 | 212.36 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 216.09 | 212.88 | 212.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 14:15:00 | 217.31 | 215.43 | 214.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 222.90 | 224.37 | 222.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 09:15:00 | 222.90 | 224.37 | 222.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 222.90 | 224.37 | 222.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:00:00 | 222.90 | 224.37 | 222.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 221.70 | 223.83 | 222.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:00:00 | 221.70 | 223.83 | 222.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 222.17 | 223.50 | 222.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 12:15:00 | 222.65 | 223.50 | 222.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 14:45:00 | 222.57 | 222.96 | 222.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 11:45:00 | 222.61 | 222.41 | 222.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 14:00:00 | 222.96 | 222.48 | 222.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 224.77 | 223.11 | 222.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 221.70 | 222.57 | 222.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 221.70 | 222.57 | 222.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 221.70 | 222.57 | 222.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 221.70 | 222.57 | 222.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 10:15:00 | 221.70 | 222.57 | 222.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 14:15:00 | 220.82 | 221.93 | 222.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 220.72 | 220.54 | 221.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 15:00:00 | 220.72 | 220.54 | 221.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 221.00 | 220.63 | 221.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 219.71 | 220.63 | 221.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 219.95 | 219.19 | 219.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 221.64 | 219.68 | 220.11 | SL hit (close>static) qty=1.00 sl=221.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 221.64 | 219.68 | 220.11 | SL hit (close>static) qty=1.00 sl=221.35 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 223.27 | 220.80 | 220.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 13:15:00 | 223.66 | 221.37 | 220.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 13:15:00 | 229.65 | 229.69 | 227.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 13:30:00 | 229.55 | 229.69 | 227.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 230.75 | 229.90 | 228.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:30:00 | 231.94 | 230.34 | 228.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:45:00 | 231.81 | 231.09 | 229.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 14:45:00 | 231.61 | 230.98 | 230.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 228.33 | 230.23 | 229.89 | SL hit (close<static) qty=1.00 sl=228.51 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 228.33 | 230.23 | 229.89 | SL hit (close<static) qty=1.00 sl=228.51 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 228.33 | 230.23 | 229.89 | SL hit (close<static) qty=1.00 sl=228.51 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 12:15:00 | 228.49 | 229.59 | 229.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 11:15:00 | 228.21 | 229.00 | 229.27 | Break + close below crossover candle low |

### Cycle 15 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 236.27 | 230.30 | 229.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 14:15:00 | 238.55 | 231.95 | 230.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 13:15:00 | 250.75 | 251.55 | 246.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-21 14:00:00 | 250.75 | 251.55 | 246.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 247.99 | 249.88 | 248.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:45:00 | 248.09 | 249.88 | 248.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 247.79 | 249.46 | 248.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:30:00 | 248.13 | 249.46 | 248.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 246.65 | 248.90 | 247.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 246.65 | 248.90 | 247.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 245.15 | 248.03 | 247.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:30:00 | 245.41 | 248.03 | 247.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 244.97 | 247.42 | 247.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 13:15:00 | 244.00 | 246.04 | 246.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 246.96 | 245.47 | 246.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 246.96 | 245.47 | 246.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 246.96 | 245.47 | 246.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 246.96 | 245.47 | 246.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 248.80 | 246.14 | 246.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:00:00 | 248.80 | 246.14 | 246.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 247.64 | 246.44 | 246.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 12:45:00 | 247.15 | 246.44 | 246.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 13:15:00 | 247.62 | 246.67 | 246.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 13:15:00 | 247.62 | 246.67 | 246.66 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 244.05 | 246.33 | 246.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 242.94 | 245.65 | 246.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 09:15:00 | 236.83 | 235.60 | 237.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 09:15:00 | 236.83 | 235.60 | 237.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 236.83 | 235.60 | 237.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 14:15:00 | 233.51 | 235.89 | 237.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 233.12 | 235.55 | 236.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 10:00:00 | 234.25 | 235.29 | 236.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 11:15:00 | 233.87 | 235.17 | 236.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 09:15:00 | 222.54 | 227.08 | 230.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 13:15:00 | 227.60 | 226.74 | 228.99 | SL hit (close>ema200) qty=0.50 sl=226.74 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 228.26 | 227.05 | 228.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 228.26 | 227.05 | 228.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 229.90 | 227.62 | 229.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 231.75 | 227.62 | 229.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 230.81 | 228.26 | 229.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:30:00 | 230.60 | 228.26 | 229.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 230.50 | 228.70 | 229.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 231.04 | 228.70 | 229.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-05 13:15:00 | 231.08 | 229.72 | 229.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 13:15:00 | 231.08 | 229.72 | 229.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 13:15:00 | 231.08 | 229.72 | 229.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 13:15:00 | 231.08 | 229.72 | 229.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 10:15:00 | 231.81 | 230.53 | 230.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 15:15:00 | 231.69 | 231.95 | 231.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 09:15:00 | 233.41 | 231.95 | 231.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 232.45 | 232.05 | 231.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 10:15:00 | 233.67 | 232.05 | 231.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 15:00:00 | 234.84 | 232.82 | 231.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 10:00:00 | 233.65 | 233.32 | 232.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 11:00:00 | 236.36 | 233.93 | 232.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 233.75 | 235.02 | 233.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 233.75 | 235.02 | 233.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 233.00 | 234.62 | 233.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 232.08 | 234.62 | 233.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-11 11:15:00 | 229.95 | 232.54 | 232.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 11:15:00 | 229.95 | 232.54 | 232.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 11:15:00 | 229.95 | 232.54 | 232.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 11:15:00 | 229.95 | 232.54 | 232.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 11:15:00 | 229.95 | 232.54 | 232.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 12:15:00 | 229.59 | 231.95 | 232.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 231.00 | 230.46 | 231.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 231.00 | 230.46 | 231.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 231.00 | 230.46 | 231.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 12:45:00 | 229.00 | 230.18 | 231.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 10:15:00 | 228.48 | 229.37 | 230.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 14:00:00 | 229.00 | 229.17 | 229.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 234.78 | 230.23 | 229.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 234.78 | 230.23 | 229.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 234.78 | 230.23 | 229.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 234.78 | 230.23 | 229.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 235.00 | 232.67 | 231.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 13:15:00 | 233.62 | 234.15 | 232.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 14:00:00 | 233.62 | 234.15 | 232.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 242.71 | 235.72 | 233.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:15:00 | 244.12 | 235.72 | 233.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 11:15:00 | 244.10 | 244.80 | 244.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 14:15:00 | 244.25 | 244.18 | 244.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 15:15:00 | 243.25 | 243.89 | 243.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-25 15:15:00 | 243.25 | 243.89 | 243.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-25 15:15:00 | 243.25 | 243.89 | 243.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 15:15:00 | 243.25 | 243.89 | 243.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 239.06 | 242.92 | 243.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 10:15:00 | 239.73 | 239.71 | 241.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 11:00:00 | 239.73 | 239.71 | 241.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 239.00 | 239.08 | 240.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:45:00 | 242.06 | 239.08 | 240.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 244.45 | 239.50 | 239.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 243.87 | 239.50 | 239.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 242.00 | 240.00 | 239.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 248.54 | 244.87 | 242.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 248.24 | 249.95 | 247.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:00:00 | 248.24 | 249.95 | 247.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 246.31 | 249.22 | 247.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 245.92 | 249.22 | 247.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 245.65 | 248.51 | 247.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 245.65 | 248.51 | 247.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 244.20 | 246.98 | 246.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 242.37 | 246.06 | 246.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 15:15:00 | 244.39 | 243.77 | 244.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:15:00 | 242.51 | 243.77 | 244.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 244.34 | 243.88 | 244.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 244.34 | 243.88 | 244.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 245.22 | 244.15 | 244.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:45:00 | 245.08 | 244.15 | 244.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 245.80 | 244.48 | 244.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:00:00 | 244.61 | 244.51 | 244.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 246.82 | 244.10 | 243.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 246.82 | 244.10 | 243.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 09:15:00 | 248.88 | 246.99 | 246.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 247.56 | 247.81 | 246.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 244.55 | 247.08 | 246.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 244.55 | 247.08 | 246.77 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 11:15:00 | 244.00 | 246.11 | 246.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 13:15:00 | 243.33 | 245.27 | 245.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 245.74 | 244.91 | 245.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 245.74 | 244.91 | 245.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 245.74 | 244.91 | 245.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 245.52 | 244.91 | 245.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 245.58 | 245.04 | 245.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:15:00 | 245.00 | 245.04 | 245.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 243.40 | 244.75 | 245.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 09:15:00 | 232.75 | 236.64 | 238.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 13:15:00 | 231.23 | 233.40 | 235.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 227.97 | 225.87 | 227.55 | SL hit (close>ema200) qty=0.50 sl=225.87 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 227.97 | 225.87 | 227.55 | SL hit (close>ema200) qty=0.50 sl=225.87 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 217.19 | 212.98 | 212.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 15:15:00 | 218.20 | 214.03 | 213.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 13:15:00 | 215.00 | 215.86 | 214.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 13:15:00 | 215.00 | 215.86 | 214.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 215.00 | 215.86 | 214.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 215.00 | 215.86 | 214.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 216.49 | 216.63 | 215.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 215.99 | 216.63 | 215.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 222.85 | 221.49 | 220.37 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 14:15:00 | 220.00 | 221.09 | 221.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 11:15:00 | 219.81 | 220.51 | 220.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 09:15:00 | 219.70 | 219.64 | 220.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 219.70 | 219.64 | 220.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 219.70 | 219.64 | 220.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:30:00 | 220.62 | 219.64 | 220.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 217.00 | 216.94 | 218.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 11:00:00 | 216.93 | 216.98 | 217.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 10:15:00 | 206.08 | 208.24 | 210.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 10:15:00 | 206.90 | 206.72 | 208.36 | SL hit (close>ema200) qty=0.50 sl=206.72 alert=retest2 |

### Cycle 29 — BUY (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 10:15:00 | 209.15 | 208.25 | 208.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 212.65 | 209.54 | 209.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 208.00 | 209.48 | 209.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 208.00 | 209.48 | 209.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 208.00 | 209.48 | 209.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 208.00 | 209.48 | 209.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 208.06 | 209.19 | 209.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:45:00 | 207.70 | 209.19 | 209.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 207.87 | 208.93 | 209.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 15:15:00 | 207.00 | 208.13 | 208.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 208.15 | 207.98 | 208.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 11:45:00 | 207.95 | 207.98 | 208.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 207.70 | 207.92 | 208.34 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 211.02 | 208.97 | 208.73 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 208.08 | 208.98 | 209.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 206.80 | 208.14 | 208.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 208.90 | 207.89 | 208.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 208.90 | 207.89 | 208.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 208.90 | 207.89 | 208.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 208.90 | 207.89 | 208.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 208.20 | 207.96 | 208.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 207.71 | 207.96 | 208.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 207.50 | 207.86 | 208.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 11:15:00 | 206.70 | 207.65 | 208.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 209.74 | 207.85 | 207.94 | SL hit (close>static) qty=1.00 sl=209.50 alert=retest2 |

### Cycle 33 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 209.48 | 208.18 | 208.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 10:15:00 | 210.88 | 209.30 | 208.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 210.15 | 210.53 | 209.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 210.15 | 210.53 | 209.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 210.15 | 210.53 | 209.76 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 15:15:00 | 208.80 | 209.57 | 209.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 12:15:00 | 208.12 | 208.97 | 209.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 209.03 | 208.98 | 209.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 15:00:00 | 209.03 | 208.98 | 209.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 208.86 | 208.96 | 209.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 208.85 | 208.96 | 209.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 208.00 | 208.77 | 209.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 12:30:00 | 207.45 | 208.23 | 208.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 11:15:00 | 207.20 | 207.86 | 208.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 14:45:00 | 207.31 | 207.84 | 208.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 206.44 | 207.81 | 208.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 207.10 | 207.67 | 208.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 207.60 | 207.67 | 208.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 207.62 | 207.47 | 207.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:00:00 | 207.62 | 207.47 | 207.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 207.71 | 207.51 | 207.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:45:00 | 207.80 | 207.51 | 207.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 207.80 | 207.57 | 207.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 207.31 | 207.57 | 207.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 205.74 | 207.21 | 207.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 09:15:00 | 197.84 | 205.92 | 206.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 197.08 | 204.04 | 205.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 196.84 | 204.04 | 205.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 196.94 | 204.04 | 205.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 196.12 | 204.04 | 205.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 193.60 | 192.60 | 195.17 | SL hit (close>ema200) qty=0.50 sl=192.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 193.60 | 192.60 | 195.17 | SL hit (close>ema200) qty=0.50 sl=192.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 193.60 | 192.60 | 195.17 | SL hit (close>ema200) qty=0.50 sl=192.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 193.60 | 192.60 | 195.17 | SL hit (close>ema200) qty=0.50 sl=192.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 12:15:00 | 192.92 | 191.47 | 191.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 192.92 | 191.47 | 191.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 193.99 | 192.23 | 191.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 195.17 | 195.27 | 194.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:30:00 | 194.92 | 195.27 | 194.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 194.71 | 195.16 | 194.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:45:00 | 194.76 | 195.16 | 194.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 194.84 | 194.99 | 194.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 194.30 | 194.99 | 194.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 195.60 | 195.05 | 194.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:15:00 | 195.82 | 195.18 | 194.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:45:00 | 195.89 | 195.28 | 194.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:15:00 | 195.90 | 195.28 | 194.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:45:00 | 195.89 | 195.40 | 194.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 195.00 | 195.40 | 194.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:45:00 | 195.36 | 195.40 | 194.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 195.60 | 195.44 | 194.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 195.71 | 195.44 | 194.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 195.88 | 195.53 | 195.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:45:00 | 196.95 | 196.00 | 195.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 11:30:00 | 196.59 | 196.90 | 196.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 13:45:00 | 196.30 | 196.53 | 196.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 14:30:00 | 196.52 | 196.56 | 196.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 196.16 | 196.48 | 196.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 194.37 | 196.48 | 196.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 194.72 | 196.13 | 196.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 194.72 | 196.13 | 196.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 194.72 | 196.13 | 196.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 194.72 | 196.13 | 196.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 194.72 | 196.13 | 196.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 194.72 | 196.13 | 196.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 194.72 | 196.13 | 196.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 194.72 | 196.13 | 196.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 194.72 | 196.13 | 196.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 12:15:00 | 194.19 | 195.27 | 195.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 196.00 | 195.42 | 195.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 13:15:00 | 196.00 | 195.42 | 195.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 196.00 | 195.42 | 195.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 196.00 | 195.42 | 195.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 195.70 | 195.47 | 195.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:45:00 | 196.50 | 195.47 | 195.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 196.10 | 195.60 | 195.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 09:30:00 | 195.15 | 195.55 | 195.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 11:15:00 | 196.99 | 195.90 | 195.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 196.99 | 195.90 | 195.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 197.64 | 196.25 | 196.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 196.65 | 196.78 | 196.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 09:45:00 | 196.88 | 196.78 | 196.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 196.74 | 196.77 | 196.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 15:15:00 | 197.40 | 196.47 | 196.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:00:00 | 197.20 | 196.76 | 196.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 11:30:00 | 197.25 | 196.84 | 196.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 14:15:00 | 197.25 | 198.22 | 197.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 197.92 | 198.16 | 197.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 15:00:00 | 200.00 | 198.37 | 198.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 11:15:00 | 196.54 | 198.16 | 198.08 | SL hit (close<static) qty=1.00 sl=197.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 12:15:00 | 196.91 | 197.91 | 197.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 12:15:00 | 196.91 | 197.91 | 197.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 12:15:00 | 196.91 | 197.91 | 197.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 12:15:00 | 196.91 | 197.91 | 197.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 196.91 | 197.91 | 197.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 194.33 | 196.73 | 197.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 10:15:00 | 194.58 | 194.44 | 195.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 10:45:00 | 194.68 | 194.44 | 195.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 194.19 | 194.09 | 195.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 15:00:00 | 194.19 | 194.09 | 195.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 190.94 | 193.46 | 194.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 10:15:00 | 189.86 | 193.46 | 194.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:15:00 | 190.14 | 190.86 | 192.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 192.44 | 192.18 | 192.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 192.44 | 192.18 | 192.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 192.44 | 192.18 | 192.18 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 10:15:00 | 190.21 | 191.78 | 192.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 11:15:00 | 189.47 | 191.32 | 191.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 11:15:00 | 183.50 | 182.23 | 184.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 11:15:00 | 183.50 | 182.23 | 184.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 183.50 | 182.23 | 184.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:00:00 | 183.50 | 182.23 | 184.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 187.76 | 183.33 | 184.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 187.76 | 183.33 | 184.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 184.33 | 183.53 | 184.58 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 187.34 | 185.40 | 185.27 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 15:15:00 | 184.76 | 185.32 | 185.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 09:15:00 | 183.20 | 184.89 | 185.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 11:15:00 | 181.09 | 180.47 | 181.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 12:00:00 | 181.09 | 180.47 | 181.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 180.71 | 180.52 | 181.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 14:45:00 | 180.12 | 180.63 | 181.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 179.26 | 180.67 | 181.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 11:30:00 | 180.00 | 179.44 | 179.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 13:00:00 | 179.98 | 179.55 | 179.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 180.40 | 179.72 | 179.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:00:00 | 180.40 | 179.72 | 179.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 180.70 | 179.91 | 180.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 15:00:00 | 180.70 | 179.91 | 180.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 180.75 | 180.08 | 180.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 180.26 | 180.08 | 180.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-01 09:15:00 | 180.39 | 180.14 | 180.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 09:15:00 | 180.39 | 180.14 | 180.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 09:15:00 | 180.39 | 180.14 | 180.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 09:15:00 | 180.39 | 180.14 | 180.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 180.39 | 180.14 | 180.14 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 178.56 | 179.82 | 179.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 178.45 | 179.55 | 179.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 179.90 | 178.61 | 179.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 12:15:00 | 179.90 | 178.61 | 179.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 179.90 | 178.61 | 179.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 179.90 | 178.61 | 179.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 179.87 | 178.86 | 179.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:00:00 | 179.87 | 178.86 | 179.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 185.83 | 180.26 | 179.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 186.11 | 182.79 | 181.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 184.40 | 184.93 | 183.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 11:00:00 | 184.40 | 184.93 | 183.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 184.42 | 185.12 | 184.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 184.42 | 185.12 | 184.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 184.21 | 184.93 | 184.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 184.21 | 184.93 | 184.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 184.16 | 184.78 | 184.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:15:00 | 183.92 | 184.78 | 184.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 184.30 | 184.68 | 184.17 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 179.56 | 183.08 | 183.54 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2026-02-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 13:15:00 | 186.61 | 182.73 | 182.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 10:15:00 | 189.10 | 185.82 | 184.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 13:15:00 | 187.20 | 187.31 | 186.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 14:00:00 | 187.20 | 187.31 | 186.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 185.49 | 187.40 | 186.63 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 13:15:00 | 185.10 | 186.06 | 186.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 184.77 | 185.80 | 186.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 179.81 | 179.19 | 181.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 179.81 | 179.19 | 181.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 182.25 | 179.85 | 181.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 182.25 | 179.85 | 181.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 181.28 | 180.14 | 181.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 182.00 | 180.14 | 181.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 181.45 | 180.63 | 181.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:00:00 | 181.10 | 180.72 | 181.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:45:00 | 181.10 | 180.78 | 181.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:45:00 | 180.27 | 180.74 | 181.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 13:15:00 | 180.90 | 180.75 | 180.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 181.11 | 180.83 | 180.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:15:00 | 181.20 | 180.83 | 180.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 180.80 | 180.82 | 180.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 180.45 | 180.80 | 180.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 181.93 | 178.06 | 178.50 | SL hit (close>static) qty=1.00 sl=181.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 181.93 | 178.06 | 178.50 | SL hit (close>static) qty=1.00 sl=181.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 181.93 | 178.06 | 178.50 | SL hit (close>static) qty=1.00 sl=181.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 181.93 | 178.06 | 178.50 | SL hit (close>static) qty=1.00 sl=181.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 181.93 | 178.06 | 178.50 | SL hit (close>static) qty=1.00 sl=181.37 alert=retest2 |

### Cycle 49 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 183.20 | 179.09 | 178.93 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 178.33 | 179.50 | 179.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 176.94 | 178.09 | 178.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 166.11 | 165.15 | 167.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 166.11 | 165.15 | 167.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 166.74 | 165.90 | 166.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:45:00 | 166.75 | 165.90 | 166.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 168.50 | 166.42 | 167.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 162.11 | 166.42 | 167.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 13:15:00 | 154.00 | 156.43 | 158.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 153.69 | 151.80 | 154.52 | SL hit (close>ema200) qty=0.50 sl=151.80 alert=retest2 |

### Cycle 51 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 158.50 | 154.45 | 154.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 159.99 | 155.56 | 154.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 155.62 | 156.50 | 155.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 155.62 | 156.50 | 155.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 155.62 | 156.50 | 155.33 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 152.47 | 154.72 | 154.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 11:15:00 | 151.90 | 153.79 | 154.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 142.81 | 142.39 | 145.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 142.81 | 142.39 | 145.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 146.65 | 143.74 | 145.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 147.00 | 143.74 | 145.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 145.15 | 144.02 | 145.81 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 151.74 | 146.91 | 146.77 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 144.53 | 147.12 | 147.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 143.45 | 145.66 | 146.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 145.80 | 141.07 | 142.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 145.80 | 141.07 | 142.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 145.80 | 141.07 | 142.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 145.80 | 141.07 | 142.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 145.15 | 141.89 | 142.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:15:00 | 145.60 | 141.89 | 142.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 147.69 | 143.84 | 143.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 147.95 | 145.79 | 144.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 146.45 | 146.59 | 145.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 146.45 | 146.59 | 145.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 146.45 | 146.59 | 145.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:30:00 | 148.01 | 146.95 | 145.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 11:45:00 | 148.16 | 147.18 | 146.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:30:00 | 148.05 | 147.63 | 146.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 152.20 | 146.93 | 146.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 151.48 | 151.85 | 150.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 151.10 | 151.85 | 150.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 151.55 | 153.56 | 152.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 152.46 | 153.56 | 152.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 13:15:00 | 162.81 | 159.33 | 156.45 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-15 13:15:00 | 162.98 | 159.33 | 156.45 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-15 13:15:00 | 162.86 | 159.33 | 156.45 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-21 12:15:00 | 167.42 | 164.42 | 163.02 | Target hit (10%) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 161.60 | 163.08 | 163.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 161.60 | 163.08 | 163.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 160.43 | 161.95 | 162.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 160.13 | 159.30 | 160.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 160.13 | 159.30 | 160.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 160.13 | 159.30 | 160.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 160.51 | 159.30 | 160.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 160.66 | 159.57 | 160.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 160.46 | 159.57 | 160.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 160.32 | 159.72 | 160.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:15:00 | 160.80 | 159.72 | 160.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 161.40 | 160.06 | 160.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 161.40 | 160.06 | 160.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 161.55 | 160.35 | 160.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:30:00 | 161.14 | 160.42 | 160.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 14:15:00 | 160.32 | 159.89 | 159.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 14:15:00 | 160.32 | 159.89 | 159.87 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 159.21 | 159.76 | 159.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 156.31 | 159.07 | 159.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 13:15:00 | 160.02 | 158.65 | 159.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 13:15:00 | 160.02 | 158.65 | 159.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 160.02 | 158.65 | 159.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:00:00 | 160.02 | 158.65 | 159.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 160.70 | 159.06 | 159.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 15:00:00 | 160.70 | 159.06 | 159.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 160.95 | 159.44 | 159.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 161.76 | 159.90 | 159.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 10:15:00 | 161.80 | 161.98 | 161.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 10:15:00 | 161.80 | 161.98 | 161.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 161.80 | 161.98 | 161.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 161.07 | 161.98 | 161.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 161.99 | 161.98 | 161.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:00:00 | 161.99 | 161.98 | 161.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 165.68 | 166.58 | 165.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 165.88 | 166.58 | 165.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 165.15 | 166.30 | 165.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:45:00 | 165.37 | 166.30 | 165.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 164.03 | 165.84 | 165.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:00:00 | 164.03 | 165.84 | 165.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 164.56 | 165.59 | 165.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 13:15:00 | 164.97 | 165.59 | 165.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 14:15:00 | 164.44 | 165.17 | 165.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 14:15:00 | 164.44 | 165.17 | 165.24 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-27 15:15:00 | 213.75 | 2025-05-28 09:15:00 | 209.43 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest1 | 2025-06-10 09:15:00 | 223.47 | 2025-06-11 12:15:00 | 217.49 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-06-30 12:15:00 | 222.65 | 2025-07-03 10:15:00 | 221.70 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-06-30 14:45:00 | 222.57 | 2025-07-03 10:15:00 | 221.70 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-07-01 11:45:00 | 222.61 | 2025-07-03 10:15:00 | 221.70 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-07-01 14:00:00 | 222.96 | 2025-07-03 10:15:00 | 221.70 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-07-07 09:15:00 | 219.71 | 2025-07-08 10:15:00 | 221.64 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-07-08 09:30:00 | 219.95 | 2025-07-08 10:15:00 | 221.64 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-07-11 14:30:00 | 231.94 | 2025-07-15 10:15:00 | 228.33 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-07-14 09:45:00 | 231.81 | 2025-07-15 10:15:00 | 228.33 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-07-14 14:45:00 | 231.61 | 2025-07-15 10:15:00 | 228.33 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-07-24 12:45:00 | 247.15 | 2025-07-24 13:15:00 | 247.62 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-07-30 14:15:00 | 233.51 | 2025-08-04 09:15:00 | 222.54 | PARTIAL | 0.50 | 4.70% |
| SELL | retest2 | 2025-07-30 14:15:00 | 233.51 | 2025-08-04 13:15:00 | 227.60 | STOP_HIT | 0.50 | 2.53% |
| SELL | retest2 | 2025-07-31 09:15:00 | 233.12 | 2025-08-05 13:15:00 | 231.08 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-07-31 10:00:00 | 234.25 | 2025-08-05 13:15:00 | 231.08 | STOP_HIT | 1.00 | 1.35% |
| SELL | retest2 | 2025-07-31 11:15:00 | 233.87 | 2025-08-05 13:15:00 | 231.08 | STOP_HIT | 1.00 | 1.19% |
| BUY | retest2 | 2025-08-07 10:15:00 | 233.67 | 2025-08-11 11:15:00 | 229.95 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-08-07 15:00:00 | 234.84 | 2025-08-11 11:15:00 | 229.95 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-08-08 10:00:00 | 233.65 | 2025-08-11 11:15:00 | 229.95 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-08-08 11:00:00 | 236.36 | 2025-08-11 11:15:00 | 229.95 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-08-12 12:45:00 | 229.00 | 2025-08-18 10:15:00 | 234.78 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-08-13 10:15:00 | 228.48 | 2025-08-18 10:15:00 | 234.78 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-08-13 14:00:00 | 229.00 | 2025-08-18 10:15:00 | 234.78 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-08-20 10:15:00 | 244.12 | 2025-08-25 15:15:00 | 243.25 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-08-25 11:15:00 | 244.10 | 2025-08-25 15:15:00 | 243.25 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-08-25 14:15:00 | 244.25 | 2025-08-25 15:15:00 | 243.25 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-09-08 13:00:00 | 244.61 | 2025-09-10 09:15:00 | 246.82 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-09-16 11:15:00 | 245.00 | 2025-09-24 09:15:00 | 232.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 09:15:00 | 243.40 | 2025-09-24 13:15:00 | 231.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-16 11:15:00 | 245.00 | 2025-09-29 14:15:00 | 227.97 | STOP_HIT | 0.50 | 6.95% |
| SELL | retest2 | 2025-09-17 09:15:00 | 243.40 | 2025-09-29 14:15:00 | 227.97 | STOP_HIT | 0.50 | 6.34% |
| SELL | retest2 | 2025-11-03 11:00:00 | 216.93 | 2025-11-10 10:15:00 | 206.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 11:00:00 | 216.93 | 2025-11-11 10:15:00 | 206.90 | STOP_HIT | 0.50 | 4.62% |
| SELL | retest2 | 2025-11-25 11:15:00 | 206.70 | 2025-11-26 09:15:00 | 209.74 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-12-02 12:30:00 | 207.45 | 2025-12-09 09:15:00 | 197.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 11:15:00 | 207.20 | 2025-12-09 09:15:00 | 196.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 14:45:00 | 207.31 | 2025-12-09 09:15:00 | 196.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 09:15:00 | 206.44 | 2025-12-09 09:15:00 | 196.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 12:30:00 | 207.45 | 2025-12-12 09:15:00 | 193.60 | STOP_HIT | 0.50 | 6.68% |
| SELL | retest2 | 2025-12-03 11:15:00 | 207.20 | 2025-12-12 09:15:00 | 193.60 | STOP_HIT | 0.50 | 6.56% |
| SELL | retest2 | 2025-12-03 14:45:00 | 207.31 | 2025-12-12 09:15:00 | 193.60 | STOP_HIT | 0.50 | 6.61% |
| SELL | retest2 | 2025-12-04 09:15:00 | 206.44 | 2025-12-12 09:15:00 | 193.60 | STOP_HIT | 0.50 | 6.22% |
| SELL | retest2 | 2025-12-09 09:15:00 | 197.84 | 2025-12-19 12:15:00 | 192.92 | STOP_HIT | 1.00 | 2.49% |
| BUY | retest2 | 2025-12-24 11:15:00 | 195.82 | 2025-12-30 09:15:00 | 194.72 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-12-24 11:45:00 | 195.89 | 2025-12-30 09:15:00 | 194.72 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-12-24 12:15:00 | 195.90 | 2025-12-30 09:15:00 | 194.72 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-12-24 12:45:00 | 195.89 | 2025-12-30 09:15:00 | 194.72 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-12-26 10:45:00 | 196.95 | 2025-12-30 09:15:00 | 194.72 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-12-29 11:30:00 | 196.59 | 2025-12-30 09:15:00 | 194.72 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-12-29 13:45:00 | 196.30 | 2025-12-30 09:15:00 | 194.72 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-12-29 14:30:00 | 196.52 | 2025-12-30 09:15:00 | 194.72 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-12-31 09:30:00 | 195.15 | 2025-12-31 11:15:00 | 196.99 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-01-01 15:15:00 | 197.40 | 2026-01-07 11:15:00 | 196.54 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2026-01-02 10:00:00 | 197.20 | 2026-01-07 12:15:00 | 196.91 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2026-01-02 11:30:00 | 197.25 | 2026-01-07 12:15:00 | 196.91 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2026-01-05 14:15:00 | 197.25 | 2026-01-07 12:15:00 | 196.91 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2026-01-06 15:00:00 | 200.00 | 2026-01-07 12:15:00 | 196.91 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-01-12 10:15:00 | 189.86 | 2026-01-16 09:15:00 | 192.44 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-01-13 12:15:00 | 190.14 | 2026-01-16 09:15:00 | 192.44 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-01-28 14:45:00 | 180.12 | 2026-02-01 09:15:00 | 180.39 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2026-01-29 10:15:00 | 179.26 | 2026-02-01 09:15:00 | 180.39 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2026-01-30 11:30:00 | 180.00 | 2026-02-01 09:15:00 | 180.39 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2026-01-30 13:00:00 | 179.98 | 2026-02-01 09:15:00 | 180.39 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2026-02-17 14:00:00 | 181.10 | 2026-02-23 09:15:00 | 181.93 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2026-02-17 14:45:00 | 181.10 | 2026-02-23 09:15:00 | 181.93 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2026-02-18 09:45:00 | 180.27 | 2026-02-23 09:15:00 | 181.93 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-02-18 13:15:00 | 180.90 | 2026-02-23 09:15:00 | 181.93 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2026-02-19 09:15:00 | 180.45 | 2026-02-23 09:15:00 | 181.93 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-03-09 09:15:00 | 162.11 | 2026-03-13 13:15:00 | 154.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-09 09:15:00 | 162.11 | 2026-03-16 14:15:00 | 153.69 | STOP_HIT | 0.50 | 5.19% |
| BUY | retest2 | 2026-04-06 10:30:00 | 148.01 | 2026-04-15 13:15:00 | 162.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 11:45:00 | 148.16 | 2026-04-15 13:15:00 | 162.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 10:30:00 | 148.05 | 2026-04-15 13:15:00 | 162.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:15:00 | 152.20 | 2026-04-21 12:15:00 | 167.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 10:15:00 | 152.46 | 2026-04-23 10:15:00 | 161.60 | STOP_HIT | 1.00 | 6.00% |
| SELL | retest2 | 2026-04-27 14:30:00 | 161.14 | 2026-04-29 14:15:00 | 160.32 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2026-05-08 13:15:00 | 164.97 | 2026-05-08 14:15:00 | 164.44 | STOP_HIT | 1.00 | -0.32% |
