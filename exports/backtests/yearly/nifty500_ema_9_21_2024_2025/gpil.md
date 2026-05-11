# Godawari Power & Ispat Ltd. (GPIL)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 295.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 135 |
| ALERT1 | 98 |
| ALERT2 | 96 |
| ALERT2_SKIP | 40 |
| ALERT3 | 262 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 125 |
| PARTIAL | 24 |
| TARGET_HIT | 13 |
| STOP_HIT | 120 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 157 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 73 / 84
- **Target hits / Stop hits / Partials:** 13 / 120 / 24
- **Avg / median % per leg:** 0.83% / -0.25%
- **Sum % (uncompounded):** 130.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 52 | 11 | 21.2% | 4 | 48 | 0 | -0.57% | -29.8% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.08% | -8.3% |
| BUY @ 3rd Alert (retest2) | 48 | 11 | 22.9% | 4 | 44 | 0 | -0.45% | -21.5% |
| SELL (all) | 105 | 62 | 59.0% | 9 | 72 | 24 | 1.52% | 159.8% |
| SELL @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 0 | 4 | 2 | 1.30% | 7.8% |
| SELL @ 3rd Alert (retest2) | 99 | 58 | 58.6% | 9 | 68 | 22 | 1.54% | 152.1% |
| retest1 (combined) | 10 | 4 | 40.0% | 0 | 8 | 2 | -0.05% | -0.5% |
| retest2 (combined) | 147 | 69 | 46.9% | 13 | 112 | 22 | 0.89% | 130.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 10:15:00 | 177.37 | 178.35 | 178.44 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 11:15:00 | 180.13 | 178.71 | 178.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 13:15:00 | 180.42 | 179.25 | 178.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-13 14:15:00 | 179.10 | 179.22 | 178.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-13 14:45:00 | 179.45 | 179.22 | 178.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 179.20 | 179.22 | 178.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-14 09:15:00 | 180.19 | 179.22 | 178.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 15:15:00 | 180.23 | 181.13 | 181.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 15:15:00 | 180.23 | 181.13 | 181.14 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 11:15:00 | 182.80 | 181.39 | 181.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 12:15:00 | 185.20 | 182.62 | 181.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 15:15:00 | 186.00 | 186.39 | 184.70 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 09:15:00 | 189.66 | 186.39 | 184.70 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 10:30:00 | 187.66 | 187.92 | 185.72 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 188.11 | 188.71 | 187.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 12:15:00 | 189.14 | 188.71 | 187.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 09:15:00 | 187.54 | 189.39 | 188.39 | SL hit (close<ema400) qty=1.00 sl=188.39 alert=retest1 |

### Cycle 5 — SELL (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 14:15:00 | 185.38 | 187.50 | 187.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 183.57 | 186.41 | 187.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 182.56 | 178.61 | 180.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 182.56 | 178.61 | 180.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 182.56 | 178.61 | 180.65 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 14:15:00 | 183.80 | 181.53 | 181.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 10:15:00 | 184.98 | 182.82 | 182.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 186.62 | 193.22 | 190.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 186.62 | 193.22 | 190.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 186.62 | 193.22 | 190.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 186.62 | 193.22 | 190.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 182.63 | 191.10 | 189.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 182.63 | 191.10 | 189.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 188.44 | 190.11 | 189.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 14:45:00 | 186.22 | 190.11 | 189.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 189.92 | 190.07 | 189.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-05 09:15:00 | 191.32 | 190.07 | 189.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 09:15:00 | 179.35 | 187.93 | 188.80 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 193.03 | 188.34 | 187.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 12:15:00 | 193.85 | 190.02 | 188.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 13:15:00 | 208.41 | 209.65 | 206.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 13:45:00 | 208.39 | 209.65 | 206.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 215.57 | 218.40 | 216.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:00:00 | 215.57 | 218.40 | 216.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 216.94 | 218.11 | 216.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 11:30:00 | 217.52 | 218.65 | 217.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 12:15:00 | 223.21 | 224.93 | 225.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 12:15:00 | 223.21 | 224.93 | 225.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 14:15:00 | 222.40 | 224.38 | 224.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 10:15:00 | 216.47 | 216.32 | 219.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 10:30:00 | 216.81 | 216.32 | 219.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 216.92 | 215.20 | 217.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:30:00 | 216.20 | 215.20 | 217.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 216.82 | 215.52 | 217.20 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 220.94 | 217.64 | 217.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 224.17 | 220.77 | 219.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 14:15:00 | 221.80 | 222.05 | 220.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 15:00:00 | 221.80 | 222.05 | 220.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 221.64 | 221.97 | 220.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:15:00 | 223.09 | 221.97 | 220.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 09:15:00 | 222.83 | 222.27 | 221.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 09:45:00 | 222.26 | 222.32 | 221.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 10:15:00 | 218.97 | 221.65 | 221.47 | SL hit (close<static) qty=1.00 sl=220.20 alert=retest2 |

### Cycle 11 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 218.65 | 221.05 | 221.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 14:15:00 | 218.47 | 220.26 | 220.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 15:15:00 | 220.36 | 220.28 | 220.74 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 09:15:00 | 215.76 | 220.28 | 220.74 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 10:45:00 | 216.45 | 218.58 | 219.84 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 229.80 | 220.02 | 220.22 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-09 12:15:00 | 229.80 | 220.02 | 220.22 | SL hit (close>ema400) qty=1.00 sl=220.22 alert=retest1 |

### Cycle 12 — BUY (started 2024-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 13:15:00 | 233.38 | 222.69 | 221.42 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 11:15:00 | 225.00 | 227.20 | 227.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 12:15:00 | 224.94 | 226.75 | 227.12 | Break + close below crossover candle low |

### Cycle 14 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 237.60 | 227.82 | 227.36 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 11:15:00 | 226.21 | 228.63 | 228.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 218.81 | 225.31 | 227.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 214.69 | 214.30 | 216.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-24 09:30:00 | 214.54 | 214.30 | 216.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 212.29 | 212.07 | 213.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:15:00 | 212.64 | 212.07 | 213.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 212.21 | 212.09 | 213.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:30:00 | 212.92 | 212.09 | 213.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 212.32 | 211.55 | 212.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 10:45:00 | 210.35 | 211.36 | 212.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 11:15:00 | 209.71 | 211.36 | 212.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-29 09:15:00 | 213.51 | 210.26 | 211.27 | SL hit (close>static) qty=1.00 sl=213.24 alert=retest2 |

### Cycle 16 — BUY (started 2024-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 13:15:00 | 214.01 | 211.67 | 211.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 14:15:00 | 219.85 | 213.31 | 212.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 11:15:00 | 221.40 | 221.58 | 218.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 11:30:00 | 221.17 | 221.58 | 218.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 222.63 | 224.11 | 222.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 10:30:00 | 224.47 | 224.03 | 222.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 14:00:00 | 223.74 | 223.40 | 222.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 15:15:00 | 224.00 | 223.34 | 222.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 216.61 | 222.10 | 222.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 216.61 | 222.10 | 222.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 210.00 | 216.82 | 219.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 15:15:00 | 216.00 | 215.98 | 218.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 09:15:00 | 224.06 | 215.98 | 218.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 224.72 | 217.73 | 218.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 224.72 | 217.73 | 218.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 224.40 | 219.07 | 219.48 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 224.18 | 220.09 | 219.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 12:15:00 | 226.41 | 223.72 | 222.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 12:15:00 | 226.83 | 228.00 | 225.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 13:00:00 | 226.83 | 228.00 | 225.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 225.53 | 227.29 | 225.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 225.53 | 227.29 | 225.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 225.80 | 226.99 | 225.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:15:00 | 224.38 | 226.99 | 225.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 224.82 | 226.56 | 225.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 10:15:00 | 223.32 | 226.56 | 225.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 224.12 | 226.07 | 225.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 10:30:00 | 223.38 | 226.07 | 225.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2024-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 12:15:00 | 221.60 | 224.87 | 225.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 15:15:00 | 220.78 | 223.08 | 224.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 11:15:00 | 198.52 | 197.33 | 202.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 11:30:00 | 199.16 | 197.33 | 202.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 200.00 | 198.45 | 201.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 10:45:00 | 197.70 | 198.31 | 200.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 13:30:00 | 198.58 | 198.62 | 200.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 14:30:00 | 198.67 | 198.64 | 200.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 09:15:00 | 194.64 | 198.71 | 200.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 194.51 | 197.87 | 199.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 14:00:00 | 193.20 | 195.46 | 197.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 12:30:00 | 193.20 | 192.53 | 193.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 11:15:00 | 197.16 | 193.98 | 193.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 11:15:00 | 197.16 | 193.98 | 193.81 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 09:15:00 | 192.17 | 194.39 | 194.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 10:15:00 | 191.54 | 193.82 | 194.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 10:15:00 | 189.18 | 186.45 | 187.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 10:15:00 | 189.18 | 186.45 | 187.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 189.18 | 186.45 | 187.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:45:00 | 189.27 | 186.45 | 187.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 190.02 | 187.16 | 187.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:30:00 | 190.98 | 187.16 | 187.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 13:15:00 | 190.89 | 188.51 | 188.44 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 09:15:00 | 186.16 | 188.39 | 188.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 13:15:00 | 185.01 | 186.62 | 187.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 188.64 | 186.63 | 187.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 188.64 | 186.63 | 187.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 188.64 | 186.63 | 187.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 10:00:00 | 188.64 | 186.63 | 187.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 188.13 | 186.93 | 187.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 11:30:00 | 187.40 | 187.03 | 187.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 10:15:00 | 187.64 | 186.59 | 186.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 11:15:00 | 189.60 | 187.51 | 187.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2024-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 11:15:00 | 189.60 | 187.51 | 187.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 14:15:00 | 190.72 | 188.77 | 188.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 10:15:00 | 187.81 | 189.06 | 188.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 10:15:00 | 187.81 | 189.06 | 188.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 187.81 | 189.06 | 188.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 10:45:00 | 187.30 | 189.06 | 188.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 187.39 | 188.73 | 188.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 11:45:00 | 187.05 | 188.73 | 188.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 12:15:00 | 186.96 | 188.37 | 188.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 12:30:00 | 186.76 | 188.37 | 188.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2024-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 14:15:00 | 186.57 | 187.79 | 187.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 185.21 | 187.08 | 187.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 11:15:00 | 186.95 | 186.95 | 187.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-06 11:45:00 | 186.77 | 186.95 | 187.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 186.97 | 186.95 | 187.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 12:30:00 | 187.56 | 186.95 | 187.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 13:15:00 | 187.56 | 187.07 | 187.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:00:00 | 187.56 | 187.07 | 187.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 186.15 | 186.89 | 187.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 183.60 | 186.71 | 187.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 09:15:00 | 188.55 | 183.53 | 183.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 188.55 | 183.53 | 183.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 09:15:00 | 190.80 | 188.28 | 186.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 12:15:00 | 188.45 | 188.86 | 187.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 12:15:00 | 188.45 | 188.86 | 187.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 188.45 | 188.86 | 187.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 13:00:00 | 188.45 | 188.86 | 187.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 188.85 | 188.75 | 187.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 11:15:00 | 189.95 | 188.57 | 187.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 14:45:00 | 189.61 | 188.28 | 187.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 10:45:00 | 189.29 | 188.52 | 187.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 13:15:00 | 186.18 | 187.73 | 187.72 | SL hit (close<static) qty=1.00 sl=187.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 14:15:00 | 185.13 | 187.21 | 187.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 10:15:00 | 183.48 | 185.80 | 186.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 186.80 | 185.16 | 186.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 14:15:00 | 186.80 | 185.16 | 186.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 186.80 | 185.16 | 186.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 186.80 | 185.16 | 186.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 186.71 | 185.47 | 186.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 187.93 | 185.47 | 186.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 187.47 | 185.87 | 186.21 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 189.60 | 186.61 | 186.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 09:15:00 | 191.50 | 188.18 | 187.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 208.21 | 210.57 | 206.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-26 09:45:00 | 208.06 | 210.57 | 206.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 210.80 | 213.38 | 211.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 210.80 | 213.38 | 211.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 210.60 | 212.82 | 211.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 216.09 | 212.82 | 211.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 09:15:00 | 212.00 | 215.80 | 215.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 212.00 | 215.80 | 215.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 209.60 | 212.94 | 214.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 15:15:00 | 192.10 | 191.83 | 196.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-09 09:15:00 | 191.65 | 191.83 | 196.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 193.10 | 192.09 | 196.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:45:00 | 195.05 | 192.09 | 196.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 195.95 | 193.36 | 196.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 12:15:00 | 198.80 | 193.36 | 196.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 12:15:00 | 196.95 | 194.08 | 196.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 12:30:00 | 197.15 | 194.08 | 196.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 13:15:00 | 195.05 | 194.27 | 196.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 14:15:00 | 194.00 | 194.27 | 196.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:15:00 | 194.05 | 194.62 | 195.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 12:15:00 | 194.10 | 194.72 | 195.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 12:45:00 | 194.50 | 194.64 | 195.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 195.40 | 194.34 | 195.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:00:00 | 195.40 | 194.34 | 195.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 192.05 | 193.88 | 194.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 11:15:00 | 191.50 | 193.88 | 194.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 184.30 | 187.50 | 188.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 184.35 | 187.50 | 188.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 184.39 | 187.50 | 188.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 184.77 | 187.50 | 188.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-18 11:15:00 | 187.20 | 187.11 | 187.90 | SL hit (close>ema200) qty=0.50 sl=187.11 alert=retest2 |

### Cycle 30 — BUY (started 2024-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 13:15:00 | 193.00 | 187.67 | 187.67 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-10-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 11:15:00 | 185.30 | 187.80 | 187.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 09:15:00 | 183.70 | 186.69 | 187.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 12:15:00 | 186.20 | 185.65 | 186.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 12:15:00 | 186.20 | 185.65 | 186.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 186.20 | 185.65 | 186.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:30:00 | 186.15 | 185.65 | 186.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 14:15:00 | 186.05 | 185.79 | 186.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 15:00:00 | 186.05 | 185.79 | 186.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 173.90 | 172.79 | 174.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:15:00 | 175.85 | 172.79 | 174.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 177.50 | 173.73 | 174.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 10:00:00 | 177.50 | 173.73 | 174.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 176.65 | 174.31 | 174.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 10:45:00 | 177.85 | 174.31 | 174.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 12:15:00 | 178.50 | 175.45 | 175.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 14:15:00 | 181.55 | 177.14 | 176.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 192.76 | 193.39 | 187.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 10:00:00 | 192.76 | 193.39 | 187.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 201.66 | 203.08 | 200.89 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 12:15:00 | 198.67 | 200.50 | 200.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 197.25 | 199.85 | 200.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 12:15:00 | 198.50 | 198.47 | 199.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 12:15:00 | 198.50 | 198.47 | 199.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 198.50 | 198.47 | 199.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:45:00 | 196.12 | 198.47 | 199.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 196.37 | 198.05 | 198.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 15:00:00 | 195.26 | 197.49 | 198.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 13:15:00 | 185.50 | 191.13 | 194.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 12:15:00 | 188.74 | 188.02 | 191.16 | SL hit (close>ema200) qty=0.50 sl=188.02 alert=retest2 |

### Cycle 34 — BUY (started 2024-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 14:15:00 | 185.65 | 184.19 | 184.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 14:15:00 | 187.21 | 185.94 | 185.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 14:15:00 | 189.56 | 189.59 | 188.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 14:30:00 | 190.67 | 189.59 | 188.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 188.11 | 189.37 | 188.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:00:00 | 188.11 | 189.37 | 188.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 188.90 | 189.27 | 188.53 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 10:15:00 | 187.51 | 188.35 | 188.40 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2024-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 14:15:00 | 192.54 | 189.06 | 188.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 198.17 | 191.35 | 189.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 14:15:00 | 235.44 | 237.34 | 233.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-12 15:00:00 | 235.44 | 237.34 | 233.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 235.50 | 236.97 | 233.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:15:00 | 232.62 | 236.97 | 233.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 230.02 | 235.58 | 233.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:00:00 | 230.02 | 235.58 | 233.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 230.04 | 234.47 | 232.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:45:00 | 229.47 | 234.47 | 232.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 232.46 | 233.29 | 232.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 09:15:00 | 236.27 | 233.30 | 232.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 13:15:00 | 231.00 | 232.67 | 232.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2024-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 13:15:00 | 231.00 | 232.67 | 232.72 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 14:15:00 | 233.17 | 232.77 | 232.76 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 231.13 | 232.48 | 232.64 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 10:15:00 | 235.01 | 232.99 | 232.85 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 12:15:00 | 232.16 | 232.75 | 232.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 230.63 | 232.32 | 232.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 15:15:00 | 223.03 | 222.31 | 224.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 09:15:00 | 224.14 | 222.31 | 224.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 222.89 | 222.42 | 224.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:45:00 | 221.40 | 222.42 | 224.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 14:00:00 | 221.15 | 222.16 | 223.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 14:45:00 | 221.35 | 221.47 | 223.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 13:15:00 | 210.33 | 214.60 | 217.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 13:15:00 | 210.09 | 214.60 | 217.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 13:15:00 | 210.28 | 214.60 | 217.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-30 14:15:00 | 199.26 | 204.29 | 207.29 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 42 — BUY (started 2025-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 13:15:00 | 211.05 | 205.69 | 205.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 14:15:00 | 213.25 | 207.20 | 206.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 11:15:00 | 208.67 | 208.99 | 207.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 12:00:00 | 208.67 | 208.99 | 207.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 209.63 | 209.12 | 207.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 11:45:00 | 210.79 | 208.90 | 208.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 14:15:00 | 210.11 | 209.47 | 208.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 204.00 | 208.03 | 208.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 204.00 | 208.03 | 208.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 200.48 | 206.52 | 207.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 205.11 | 203.44 | 205.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 205.11 | 203.44 | 205.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 205.11 | 203.44 | 205.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:45:00 | 202.41 | 203.44 | 205.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 204.23 | 203.60 | 204.98 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2025-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 15:15:00 | 207.00 | 205.61 | 205.55 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 203.70 | 205.23 | 205.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 09:15:00 | 199.64 | 202.44 | 203.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 15:15:00 | 199.00 | 199.00 | 201.11 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:15:00 | 194.60 | 199.00 | 201.11 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 12:45:00 | 196.00 | 196.81 | 199.20 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 13:15:00 | 194.24 | 196.30 | 198.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 13:45:00 | 196.49 | 196.30 | 198.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:15:00 | 186.20 | 191.35 | 195.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 12:15:00 | 184.87 | 189.98 | 194.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-01-14 14:15:00 | 185.09 | 185.05 | 188.38 | SL hit (close>ema200) qty=0.50 sl=185.05 alert=retest1 |

### Cycle 46 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 188.47 | 186.83 | 186.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 12:15:00 | 190.80 | 187.85 | 187.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 188.01 | 188.75 | 188.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 10:15:00 | 188.01 | 188.75 | 188.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 188.01 | 188.75 | 188.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 188.01 | 188.75 | 188.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 189.29 | 188.86 | 188.13 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 183.06 | 187.43 | 187.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 180.99 | 185.38 | 186.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 183.72 | 183.34 | 184.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 183.72 | 183.34 | 184.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 184.74 | 183.62 | 184.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 185.33 | 183.62 | 184.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 184.19 | 183.73 | 184.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 12:30:00 | 183.55 | 183.75 | 184.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 13:15:00 | 183.14 | 183.75 | 184.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 174.37 | 177.00 | 180.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 173.98 | 177.00 | 180.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 09:15:00 | 165.20 | 170.86 | 174.96 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 48 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 176.50 | 173.35 | 173.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 177.25 | 174.13 | 173.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 176.51 | 176.79 | 175.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 13:45:00 | 176.44 | 176.79 | 175.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 177.01 | 176.84 | 175.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 10:45:00 | 178.08 | 177.21 | 176.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 12:30:00 | 178.15 | 177.69 | 176.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 11:15:00 | 178.43 | 181.49 | 180.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 14:15:00 | 178.04 | 179.36 | 179.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2025-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 14:15:00 | 178.04 | 179.36 | 179.50 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2025-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 14:15:00 | 181.26 | 179.28 | 179.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 184.61 | 180.80 | 179.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 09:15:00 | 184.36 | 185.91 | 184.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 09:15:00 | 184.36 | 185.91 | 184.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 184.36 | 185.91 | 184.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 184.36 | 185.91 | 184.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 184.71 | 185.67 | 184.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:30:00 | 182.90 | 185.67 | 184.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 186.63 | 185.86 | 184.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 12:30:00 | 188.22 | 186.29 | 184.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 13:00:00 | 188.01 | 186.29 | 184.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 183.72 | 186.05 | 185.29 | SL hit (close<static) qty=1.00 sl=184.50 alert=retest2 |

### Cycle 51 — SELL (started 2025-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 13:15:00 | 184.21 | 184.82 | 184.87 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2025-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 15:15:00 | 185.24 | 184.93 | 184.92 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 180.84 | 184.11 | 184.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 10:15:00 | 180.10 | 183.31 | 184.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 177.97 | 177.40 | 179.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:30:00 | 176.92 | 177.40 | 179.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 15:15:00 | 173.80 | 172.59 | 174.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 09:15:00 | 168.70 | 172.59 | 174.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 166.11 | 171.29 | 174.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:15:00 | 165.71 | 171.29 | 174.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 12:45:00 | 166.00 | 168.58 | 172.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 13:45:00 | 165.45 | 168.10 | 171.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 14:15:00 | 166.08 | 168.10 | 171.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 170.47 | 169.05 | 171.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:00:00 | 170.47 | 169.05 | 171.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 169.62 | 169.16 | 171.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:30:00 | 173.03 | 169.16 | 171.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 170.12 | 169.35 | 170.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 11:30:00 | 169.62 | 169.35 | 170.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 171.44 | 169.77 | 170.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:45:00 | 171.25 | 169.77 | 170.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 174.05 | 170.63 | 171.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 14:00:00 | 174.05 | 170.63 | 171.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 175.08 | 171.52 | 171.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 175.08 | 171.52 | 171.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-17 15:15:00 | 175.00 | 172.21 | 171.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2025-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 15:15:00 | 175.00 | 172.21 | 171.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 11:15:00 | 175.83 | 173.50 | 172.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 10:15:00 | 175.79 | 176.74 | 174.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 10:15:00 | 175.79 | 176.74 | 174.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 175.79 | 176.74 | 174.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:45:00 | 175.27 | 176.74 | 174.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 175.63 | 176.52 | 175.04 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2025-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 11:15:00 | 171.59 | 174.09 | 174.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-20 12:15:00 | 171.00 | 173.47 | 174.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 11:15:00 | 172.38 | 171.92 | 172.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-21 12:00:00 | 172.38 | 171.92 | 172.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 12:15:00 | 174.00 | 172.34 | 173.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 12:45:00 | 174.50 | 172.34 | 173.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 172.45 | 172.36 | 172.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 09:15:00 | 170.40 | 172.61 | 173.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 13:15:00 | 171.35 | 172.25 | 172.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 14:15:00 | 171.03 | 172.13 | 172.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 09:45:00 | 171.14 | 171.48 | 172.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 13:15:00 | 162.78 | 166.25 | 168.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 13:15:00 | 162.48 | 166.25 | 168.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 13:15:00 | 162.58 | 166.25 | 168.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 161.88 | 164.13 | 167.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-03-03 09:15:00 | 153.36 | 156.55 | 161.22 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 56 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 160.24 | 156.89 | 156.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 163.41 | 159.29 | 157.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 15:15:00 | 178.20 | 178.23 | 173.20 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 09:15:00 | 180.77 | 178.23 | 173.20 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 09:45:00 | 184.51 | 179.73 | 174.34 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 176.12 | 180.01 | 177.43 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 176.12 | 180.01 | 177.43 | SL hit (close<ema400) qty=1.00 sl=177.43 alert=retest1 |

### Cycle 57 — SELL (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 09:15:00 | 171.56 | 176.67 | 176.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 10:15:00 | 170.36 | 175.41 | 176.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 12:15:00 | 170.70 | 170.42 | 172.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 13:00:00 | 170.70 | 170.42 | 172.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 170.84 | 170.50 | 172.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 13:45:00 | 172.20 | 170.50 | 172.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 172.85 | 170.97 | 172.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:00:00 | 172.85 | 170.97 | 172.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 172.60 | 171.30 | 172.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:15:00 | 172.22 | 171.30 | 172.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 174.16 | 172.68 | 172.84 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 174.22 | 172.98 | 172.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 14:15:00 | 174.80 | 173.62 | 173.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 178.92 | 179.87 | 178.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 178.92 | 179.87 | 178.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 184.30 | 186.14 | 184.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:15:00 | 183.50 | 186.14 | 184.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 182.76 | 185.47 | 184.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:30:00 | 183.03 | 185.47 | 184.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 183.07 | 184.99 | 184.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 182.20 | 184.99 | 184.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 182.64 | 184.29 | 184.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 15:00:00 | 182.64 | 184.29 | 184.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 182.33 | 183.90 | 184.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 11:15:00 | 181.59 | 183.12 | 183.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 181.76 | 179.74 | 180.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 181.76 | 179.74 | 180.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 181.76 | 179.74 | 180.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 181.76 | 179.74 | 180.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 182.25 | 180.24 | 180.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 180.96 | 180.24 | 180.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 180.01 | 180.25 | 180.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:45:00 | 178.97 | 179.91 | 180.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 09:15:00 | 183.07 | 180.61 | 180.69 | SL hit (close>static) qty=1.00 sl=180.94 alert=retest2 |

### Cycle 60 — BUY (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 10:15:00 | 185.78 | 181.64 | 181.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 11:15:00 | 187.24 | 182.76 | 181.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 198.10 | 202.40 | 197.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 10:00:00 | 198.10 | 202.40 | 197.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 198.79 | 201.68 | 197.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 198.22 | 201.68 | 197.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 11:15:00 | 196.92 | 200.72 | 197.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:00:00 | 196.92 | 200.72 | 197.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 198.34 | 200.25 | 197.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 15:00:00 | 201.14 | 200.22 | 198.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 181.11 | 196.84 | 197.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 181.11 | 196.84 | 197.11 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 191.35 | 187.06 | 186.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 10:15:00 | 193.40 | 189.67 | 188.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 14:15:00 | 193.27 | 193.51 | 191.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 14:45:00 | 193.51 | 193.51 | 191.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 193.40 | 193.36 | 192.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 194.10 | 193.36 | 192.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-17 13:15:00 | 191.53 | 192.74 | 192.26 | SL hit (close<static) qty=1.00 sl=192.25 alert=retest2 |

### Cycle 63 — SELL (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-21 09:15:00 | 191.33 | 191.94 | 191.97 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 10:15:00 | 193.40 | 192.23 | 192.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 09:15:00 | 194.26 | 192.97 | 192.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 194.21 | 200.36 | 199.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 194.21 | 200.36 | 199.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 194.21 | 200.36 | 199.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 194.21 | 200.36 | 199.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 192.04 | 198.70 | 198.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 11:15:00 | 189.79 | 193.13 | 194.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 184.46 | 184.40 | 187.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 15:15:00 | 183.00 | 183.09 | 185.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 183.00 | 183.09 | 185.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 09:15:00 | 180.15 | 183.46 | 184.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 11:45:00 | 179.79 | 181.54 | 183.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 12:00:00 | 180.20 | 180.35 | 181.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 171.14 | 177.43 | 179.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 170.80 | 177.43 | 179.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 171.19 | 177.43 | 179.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 14:15:00 | 178.00 | 176.67 | 178.21 | SL hit (close>ema200) qty=0.50 sl=176.67 alert=retest2 |

### Cycle 66 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 185.44 | 180.09 | 179.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 190.10 | 184.67 | 182.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 200.78 | 202.04 | 197.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 10:00:00 | 200.78 | 202.04 | 197.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 202.08 | 201.84 | 200.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 14:45:00 | 200.90 | 201.84 | 200.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 201.76 | 201.82 | 200.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:15:00 | 203.25 | 201.82 | 200.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 204.22 | 202.30 | 200.76 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 198.53 | 201.16 | 201.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 15:15:00 | 197.90 | 200.05 | 200.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 11:15:00 | 198.00 | 196.03 | 197.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 11:15:00 | 198.00 | 196.03 | 197.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 198.00 | 196.03 | 197.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:45:00 | 198.48 | 196.03 | 197.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 197.88 | 196.40 | 197.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:30:00 | 198.02 | 196.40 | 197.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 198.51 | 196.82 | 197.56 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 198.74 | 197.95 | 197.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 11:15:00 | 204.59 | 199.28 | 198.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 13:15:00 | 201.94 | 202.32 | 201.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 14:00:00 | 201.94 | 202.32 | 201.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 201.13 | 202.09 | 201.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:30:00 | 201.31 | 202.09 | 201.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 200.90 | 201.85 | 201.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 200.27 | 201.85 | 201.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 200.18 | 201.51 | 200.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 200.18 | 201.51 | 200.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 201.10 | 201.43 | 201.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:00:00 | 201.10 | 201.43 | 201.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 200.12 | 201.17 | 200.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:45:00 | 200.17 | 201.17 | 200.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 199.20 | 200.78 | 200.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:30:00 | 199.88 | 200.78 | 200.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 196.75 | 199.97 | 200.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 11:15:00 | 196.62 | 198.06 | 199.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 10:15:00 | 195.17 | 193.46 | 194.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 10:15:00 | 195.17 | 193.46 | 194.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 195.17 | 193.46 | 194.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:30:00 | 195.57 | 193.46 | 194.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 193.81 | 193.53 | 194.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 12:45:00 | 193.23 | 193.61 | 194.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 14:30:00 | 193.37 | 193.72 | 194.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 15:15:00 | 193.45 | 193.72 | 194.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 10:45:00 | 192.68 | 193.58 | 194.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 190.67 | 192.06 | 193.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 14:45:00 | 190.10 | 191.18 | 192.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:30:00 | 189.87 | 190.94 | 191.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 14:15:00 | 183.57 | 188.29 | 189.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 14:15:00 | 183.70 | 188.29 | 189.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 14:15:00 | 183.78 | 188.29 | 189.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 14:15:00 | 183.05 | 188.29 | 189.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-09 11:15:00 | 188.88 | 188.07 | 189.11 | SL hit (close>ema200) qty=0.50 sl=188.07 alert=retest2 |

### Cycle 70 — BUY (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 10:15:00 | 195.69 | 189.00 | 188.45 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 186.29 | 188.62 | 188.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 185.72 | 187.59 | 188.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 184.53 | 184.33 | 185.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 184.53 | 184.33 | 185.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 185.35 | 184.53 | 185.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 185.35 | 184.53 | 185.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 185.58 | 184.74 | 185.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 185.30 | 184.74 | 185.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 186.21 | 185.04 | 185.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 186.89 | 185.04 | 185.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 185.24 | 185.08 | 185.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 184.88 | 185.08 | 185.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 184.18 | 184.87 | 185.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 09:15:00 | 175.64 | 178.55 | 180.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 180.11 | 178.86 | 180.34 | SL hit (close>ema200) qty=0.50 sl=178.86 alert=retest2 |

### Cycle 72 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 181.10 | 179.29 | 179.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 15:15:00 | 182.01 | 180.29 | 179.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 188.52 | 188.73 | 186.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 10:00:00 | 188.52 | 188.73 | 186.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 187.43 | 188.46 | 187.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:00:00 | 187.43 | 188.46 | 187.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 189.08 | 188.59 | 187.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 15:15:00 | 189.90 | 188.59 | 187.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 185.72 | 188.22 | 187.55 | SL hit (close<static) qty=1.00 sl=186.94 alert=retest2 |

### Cycle 73 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 186.00 | 187.02 | 187.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 14:15:00 | 183.99 | 186.16 | 186.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 184.93 | 184.51 | 185.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 14:15:00 | 184.93 | 184.51 | 185.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 184.93 | 184.51 | 185.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:00:00 | 184.93 | 184.51 | 185.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 184.61 | 184.57 | 185.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 10:45:00 | 183.86 | 184.40 | 185.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:45:00 | 183.64 | 184.40 | 184.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 09:30:00 | 183.85 | 184.20 | 184.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 10:30:00 | 183.66 | 184.02 | 184.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 183.75 | 183.04 | 183.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 184.55 | 183.04 | 183.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 183.77 | 183.19 | 183.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 183.77 | 183.19 | 183.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 183.00 | 183.15 | 183.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 12:45:00 | 182.39 | 183.02 | 183.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 14:30:00 | 182.54 | 183.04 | 183.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:15:00 | 182.70 | 183.04 | 183.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:45:00 | 182.51 | 182.81 | 183.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 181.94 | 180.90 | 181.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:00:00 | 181.94 | 180.90 | 181.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 182.00 | 181.12 | 181.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 15:15:00 | 181.29 | 181.12 | 181.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 12:15:00 | 182.50 | 181.93 | 181.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 12:15:00 | 182.50 | 181.93 | 181.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 13:15:00 | 183.39 | 182.22 | 182.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 10:15:00 | 185.74 | 185.82 | 184.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 11:00:00 | 185.74 | 185.82 | 184.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 184.25 | 185.50 | 184.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:30:00 | 184.39 | 185.50 | 184.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 183.41 | 185.08 | 184.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:00:00 | 183.41 | 185.08 | 184.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 182.99 | 184.67 | 184.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:45:00 | 182.91 | 184.67 | 184.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 15:15:00 | 182.35 | 183.98 | 184.08 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 185.23 | 184.23 | 184.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 186.37 | 184.66 | 184.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 186.90 | 186.92 | 185.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 09:45:00 | 187.03 | 186.92 | 185.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 186.00 | 186.74 | 185.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:30:00 | 186.09 | 186.74 | 185.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 186.68 | 186.73 | 186.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:30:00 | 186.35 | 186.73 | 186.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 186.67 | 186.72 | 186.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:45:00 | 186.30 | 186.72 | 186.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 185.61 | 186.50 | 186.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 15:00:00 | 185.61 | 186.50 | 186.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 186.50 | 186.50 | 186.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 196.73 | 186.50 | 186.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 190.63 | 192.47 | 192.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 190.63 | 192.47 | 192.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 189.15 | 191.18 | 191.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 189.76 | 189.06 | 189.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 11:15:00 | 189.76 | 189.06 | 189.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 189.76 | 189.06 | 189.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:00:00 | 189.76 | 189.06 | 189.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 189.66 | 189.18 | 189.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 189.66 | 189.18 | 189.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 194.10 | 190.17 | 190.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 194.10 | 190.17 | 190.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 194.12 | 190.96 | 190.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 10:15:00 | 195.46 | 192.81 | 191.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 194.18 | 195.24 | 194.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 14:15:00 | 194.18 | 195.24 | 194.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 194.18 | 195.24 | 194.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 194.18 | 195.24 | 194.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 194.00 | 194.99 | 194.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 193.82 | 194.99 | 194.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 192.32 | 194.46 | 194.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 192.32 | 194.46 | 194.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 191.70 | 193.91 | 193.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 191.70 | 193.91 | 193.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 190.05 | 193.13 | 193.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 189.34 | 191.65 | 192.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 190.99 | 190.98 | 192.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 190.99 | 190.98 | 192.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 190.99 | 190.98 | 192.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 193.84 | 190.98 | 192.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 191.03 | 190.99 | 192.02 | EMA400 retest candle locked (from downside) |

### Cycle 80 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 195.76 | 192.84 | 192.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 14:15:00 | 196.18 | 193.50 | 193.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 190.43 | 196.84 | 195.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 190.43 | 196.84 | 195.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 190.43 | 196.84 | 195.75 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 189.68 | 194.09 | 194.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 186.22 | 190.86 | 192.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 13:15:00 | 189.75 | 189.26 | 191.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:00:00 | 189.75 | 189.26 | 191.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 190.86 | 189.58 | 191.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:30:00 | 191.45 | 189.58 | 191.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 190.90 | 189.84 | 191.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 192.58 | 189.84 | 191.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 190.36 | 189.94 | 191.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:00:00 | 189.50 | 189.91 | 190.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:45:00 | 189.17 | 189.62 | 190.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 12:15:00 | 196.20 | 190.74 | 190.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 196.20 | 190.74 | 190.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 11:15:00 | 197.58 | 195.03 | 193.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 15:15:00 | 202.25 | 202.70 | 200.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 199.95 | 202.15 | 200.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 199.95 | 202.15 | 200.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 199.95 | 202.15 | 200.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 199.50 | 201.62 | 200.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:30:00 | 198.81 | 201.62 | 200.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 202.84 | 201.65 | 200.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 14:30:00 | 203.44 | 201.65 | 200.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 239.50 | 240.62 | 237.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 235.01 | 240.62 | 237.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 239.20 | 240.34 | 237.91 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 232.08 | 236.81 | 236.99 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 247.80 | 236.66 | 236.22 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 243.37 | 243.58 | 243.58 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 244.31 | 243.73 | 243.65 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 14:15:00 | 242.12 | 243.40 | 243.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 10:15:00 | 241.58 | 242.99 | 243.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 11:15:00 | 243.16 | 243.03 | 243.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 11:15:00 | 243.16 | 243.03 | 243.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 243.16 | 243.03 | 243.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:00:00 | 243.16 | 243.03 | 243.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 242.39 | 242.90 | 243.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:30:00 | 243.53 | 242.90 | 243.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 239.99 | 240.13 | 241.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:45:00 | 239.10 | 239.99 | 241.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 15:15:00 | 238.77 | 239.97 | 241.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 242.29 | 240.24 | 240.99 | SL hit (close>static) qty=1.00 sl=241.40 alert=retest2 |

### Cycle 88 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 243.03 | 241.58 | 241.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 13:15:00 | 243.76 | 242.02 | 241.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 240.69 | 242.08 | 241.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 240.69 | 242.08 | 241.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 240.69 | 242.08 | 241.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 241.12 | 242.08 | 241.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 241.53 | 241.97 | 241.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 242.85 | 242.01 | 241.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 13:15:00 | 241.41 | 241.71 | 241.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 241.41 | 241.71 | 241.72 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 244.21 | 242.07 | 241.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 249.27 | 244.77 | 243.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 12:15:00 | 264.90 | 265.32 | 258.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 12:45:00 | 263.93 | 265.32 | 258.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 263.90 | 264.96 | 259.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:00:00 | 263.90 | 264.96 | 259.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 260.05 | 263.92 | 262.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:15:00 | 258.49 | 263.92 | 262.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 259.30 | 262.99 | 261.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:30:00 | 258.14 | 262.99 | 261.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 13:15:00 | 260.56 | 261.21 | 261.26 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 270.75 | 262.47 | 261.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 10:15:00 | 274.32 | 269.94 | 267.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 13:15:00 | 268.34 | 270.48 | 268.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 13:15:00 | 268.34 | 270.48 | 268.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 268.34 | 270.48 | 268.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:45:00 | 267.65 | 270.48 | 268.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 265.45 | 269.48 | 267.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 265.45 | 269.48 | 267.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 266.99 | 268.98 | 267.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 261.60 | 268.98 | 267.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 10:15:00 | 263.51 | 267.20 | 267.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 11:15:00 | 262.04 | 266.17 | 266.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 258.00 | 257.77 | 260.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 258.00 | 257.77 | 260.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 247.70 | 246.14 | 248.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 11:15:00 | 245.26 | 246.92 | 247.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 09:30:00 | 245.27 | 245.78 | 246.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 10:30:00 | 244.89 | 245.59 | 246.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 15:15:00 | 243.80 | 241.53 | 241.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 15:15:00 | 243.80 | 241.53 | 241.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 09:15:00 | 246.32 | 242.49 | 241.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 10:15:00 | 246.05 | 246.23 | 244.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 11:00:00 | 246.05 | 246.23 | 244.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 244.52 | 245.71 | 244.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:45:00 | 243.07 | 245.71 | 244.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 243.10 | 245.19 | 244.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 243.10 | 245.19 | 244.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 242.90 | 244.73 | 244.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:30:00 | 242.75 | 244.73 | 244.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 277.31 | 280.10 | 277.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:45:00 | 276.09 | 280.10 | 277.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 277.70 | 279.62 | 277.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:00:00 | 277.70 | 279.62 | 277.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 279.75 | 279.65 | 277.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 13:45:00 | 281.20 | 280.18 | 278.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 13:30:00 | 281.20 | 280.51 | 279.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 14:00:00 | 281.05 | 280.51 | 279.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 10:15:00 | 282.60 | 280.76 | 279.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 281.50 | 280.91 | 280.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 272.60 | 278.53 | 279.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 272.60 | 278.53 | 279.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 271.10 | 276.03 | 277.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 271.45 | 271.08 | 273.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 11:15:00 | 271.45 | 271.08 | 273.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 271.45 | 271.08 | 273.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:30:00 | 272.95 | 271.08 | 273.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 275.80 | 272.03 | 274.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 275.80 | 272.03 | 274.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 276.70 | 272.96 | 274.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 276.75 | 272.96 | 274.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 274.00 | 273.49 | 274.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 275.05 | 273.49 | 274.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 274.60 | 273.72 | 274.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 277.20 | 273.72 | 274.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 272.50 | 273.47 | 274.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 11:30:00 | 271.75 | 273.12 | 273.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:30:00 | 271.50 | 272.75 | 273.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:15:00 | 271.30 | 272.49 | 273.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 12:45:00 | 271.85 | 272.17 | 272.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 274.10 | 272.56 | 272.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:00:00 | 274.10 | 272.56 | 272.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 275.25 | 273.10 | 273.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:30:00 | 277.30 | 273.10 | 273.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 274.00 | 273.28 | 273.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 274.00 | 273.28 | 273.23 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 09:15:00 | 271.75 | 272.97 | 273.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 12:15:00 | 270.65 | 272.07 | 272.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 11:15:00 | 270.00 | 268.84 | 270.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:00:00 | 270.00 | 268.84 | 270.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 267.45 | 268.56 | 270.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 14:00:00 | 265.35 | 267.92 | 269.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 14:45:00 | 263.00 | 267.03 | 269.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 09:15:00 | 274.50 | 267.97 | 269.20 | SL hit (close>static) qty=1.00 sl=270.50 alert=retest2 |

### Cycle 98 — BUY (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 10:15:00 | 278.40 | 270.06 | 270.04 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 269.75 | 272.61 | 272.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 266.15 | 270.81 | 272.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 240.20 | 238.13 | 240.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 240.20 | 238.13 | 240.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 240.20 | 238.13 | 240.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:30:00 | 234.55 | 237.29 | 239.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:30:00 | 236.24 | 235.71 | 236.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 12:15:00 | 238.09 | 236.86 | 236.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 12:15:00 | 238.09 | 236.86 | 236.76 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 13:15:00 | 236.25 | 236.90 | 236.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 233.08 | 235.94 | 236.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 11:15:00 | 235.57 | 235.30 | 236.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 11:45:00 | 235.29 | 235.30 | 236.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 237.93 | 235.82 | 236.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:00:00 | 237.93 | 235.82 | 236.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 238.40 | 236.34 | 236.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:00:00 | 238.40 | 236.34 | 236.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 14:15:00 | 238.40 | 236.75 | 236.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 09:15:00 | 239.76 | 237.55 | 237.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 11:15:00 | 236.60 | 237.44 | 237.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 11:15:00 | 236.60 | 237.44 | 237.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 236.60 | 237.44 | 237.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:45:00 | 236.94 | 237.44 | 237.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 236.25 | 237.20 | 236.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:45:00 | 235.85 | 237.20 | 236.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 14:15:00 | 235.92 | 236.75 | 236.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 10:15:00 | 232.15 | 235.48 | 236.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 228.56 | 227.46 | 229.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 228.56 | 227.46 | 229.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 230.50 | 228.07 | 229.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 230.50 | 228.07 | 229.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 230.28 | 228.51 | 229.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:15:00 | 229.67 | 228.51 | 229.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 10:15:00 | 233.84 | 230.55 | 230.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 233.84 | 230.55 | 230.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 13:15:00 | 234.05 | 231.67 | 231.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 11:15:00 | 233.63 | 233.68 | 232.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-11 11:45:00 | 233.22 | 233.68 | 232.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 232.34 | 233.41 | 232.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 232.10 | 233.41 | 232.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 232.23 | 233.18 | 232.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 234.62 | 233.24 | 232.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 238.21 | 241.81 | 241.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 238.21 | 241.81 | 241.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 237.68 | 240.41 | 241.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 238.80 | 236.81 | 237.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 238.80 | 236.81 | 237.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 238.80 | 236.81 | 237.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:30:00 | 238.95 | 236.81 | 237.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 239.06 | 237.26 | 237.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:00:00 | 238.36 | 237.48 | 237.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 13:15:00 | 239.27 | 238.12 | 238.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 239.27 | 238.12 | 238.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 241.51 | 238.80 | 238.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 252.96 | 254.32 | 249.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:00:00 | 252.96 | 254.32 | 249.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 251.27 | 253.23 | 250.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:45:00 | 251.06 | 253.23 | 250.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 251.15 | 252.82 | 250.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 251.15 | 252.82 | 250.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 253.51 | 252.95 | 250.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:45:00 | 251.68 | 252.95 | 250.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 252.74 | 253.11 | 251.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 252.25 | 253.11 | 251.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 261.00 | 257.67 | 255.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:45:00 | 265.21 | 258.71 | 256.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 13:15:00 | 261.80 | 259.72 | 256.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 13:15:00 | 271.05 | 273.16 | 273.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 271.05 | 273.16 | 273.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 268.75 | 272.27 | 272.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 256.85 | 256.78 | 261.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 10:15:00 | 261.00 | 256.78 | 261.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 259.75 | 257.37 | 261.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:30:00 | 259.05 | 257.37 | 261.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 261.75 | 258.72 | 260.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 261.75 | 258.72 | 260.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 265.55 | 260.09 | 260.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 265.55 | 260.09 | 260.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 266.30 | 261.71 | 261.27 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2026-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 14:15:00 | 258.10 | 261.49 | 261.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 15:15:00 | 257.55 | 260.70 | 261.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 11:15:00 | 243.05 | 242.81 | 247.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:00:00 | 243.05 | 242.81 | 247.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 250.25 | 244.26 | 246.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 251.00 | 244.26 | 246.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 248.10 | 245.03 | 246.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 247.50 | 245.03 | 246.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:15:00 | 248.00 | 246.14 | 246.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 250.15 | 247.35 | 247.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 250.15 | 247.35 | 247.26 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 245.65 | 247.24 | 247.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 243.30 | 246.45 | 247.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 11:15:00 | 245.45 | 243.50 | 245.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 11:15:00 | 245.45 | 243.50 | 245.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 245.45 | 243.50 | 245.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:00:00 | 245.45 | 243.50 | 245.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 12:15:00 | 245.10 | 243.82 | 245.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:30:00 | 245.15 | 243.82 | 245.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 245.30 | 244.12 | 245.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 14:15:00 | 246.45 | 244.12 | 245.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 247.25 | 244.75 | 245.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 14:30:00 | 247.35 | 244.75 | 245.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 249.40 | 245.68 | 245.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 09:15:00 | 253.70 | 247.28 | 246.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 15:15:00 | 256.00 | 257.29 | 254.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 09:15:00 | 246.20 | 257.29 | 254.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 248.25 | 255.49 | 253.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:30:00 | 249.70 | 255.49 | 253.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 249.00 | 254.19 | 253.46 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 247.95 | 252.94 | 252.96 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 258.40 | 252.67 | 252.44 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 246.98 | 251.99 | 252.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 243.87 | 250.37 | 251.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 248.68 | 245.32 | 247.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 248.68 | 245.32 | 247.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 248.68 | 245.32 | 247.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 248.68 | 245.32 | 247.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 251.00 | 246.46 | 247.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 251.00 | 246.46 | 247.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 252.52 | 248.24 | 248.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 11:00:00 | 252.52 | 248.24 | 248.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 252.87 | 249.16 | 248.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 254.33 | 250.81 | 249.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 255.94 | 257.20 | 254.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 255.94 | 257.20 | 254.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 255.94 | 257.20 | 254.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 255.00 | 257.20 | 254.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 256.76 | 257.11 | 255.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 15:15:00 | 258.00 | 257.11 | 255.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 252.57 | 256.34 | 255.53 | SL hit (close<static) qty=1.00 sl=255.25 alert=retest2 |

### Cycle 117 — SELL (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 13:15:00 | 252.77 | 254.82 | 255.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 14:15:00 | 251.76 | 254.21 | 254.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 252.89 | 252.12 | 253.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 11:00:00 | 252.89 | 252.12 | 253.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 252.72 | 252.24 | 253.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:15:00 | 253.57 | 252.24 | 253.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 255.94 | 252.98 | 253.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:00:00 | 255.94 | 252.98 | 253.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 256.37 | 253.66 | 253.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:45:00 | 256.12 | 253.66 | 253.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 257.48 | 254.42 | 254.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 261.44 | 256.24 | 255.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 258.22 | 259.51 | 257.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 258.22 | 259.51 | 257.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 258.22 | 259.51 | 257.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 257.95 | 259.51 | 257.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 257.20 | 258.81 | 257.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:45:00 | 256.65 | 258.81 | 257.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 257.54 | 258.56 | 257.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:30:00 | 260.45 | 259.61 | 258.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 10:30:00 | 260.38 | 261.96 | 260.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 14:00:00 | 258.84 | 260.73 | 260.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 14:15:00 | 258.20 | 260.22 | 260.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 258.20 | 260.22 | 260.36 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 09:15:00 | 263.09 | 260.57 | 260.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 10:15:00 | 266.79 | 261.81 | 261.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 13:15:00 | 271.00 | 272.97 | 270.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 14:00:00 | 271.00 | 272.97 | 270.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 269.81 | 272.34 | 270.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:30:00 | 268.73 | 272.34 | 270.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 269.90 | 271.85 | 270.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 268.91 | 271.85 | 270.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 266.78 | 270.84 | 269.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 266.78 | 270.84 | 269.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 265.47 | 269.77 | 269.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 265.47 | 269.77 | 269.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 263.61 | 268.53 | 268.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 262.09 | 265.90 | 267.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 12:15:00 | 264.38 | 263.85 | 265.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 12:45:00 | 264.00 | 263.85 | 265.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 265.00 | 263.79 | 265.04 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 269.06 | 266.22 | 265.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 270.65 | 268.05 | 267.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 267.28 | 268.43 | 267.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 12:15:00 | 267.28 | 268.43 | 267.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 267.28 | 268.43 | 267.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:45:00 | 267.50 | 268.43 | 267.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 268.99 | 268.54 | 267.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 14:45:00 | 271.03 | 268.79 | 267.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 14:15:00 | 266.32 | 269.30 | 269.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 266.32 | 269.30 | 269.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 15:15:00 | 265.50 | 268.54 | 269.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 246.40 | 245.78 | 250.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 246.40 | 245.78 | 250.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 253.00 | 247.22 | 250.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 239.90 | 248.85 | 250.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 10:15:00 | 245.95 | 248.34 | 249.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 12:00:00 | 245.85 | 247.24 | 249.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 252.75 | 249.36 | 249.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 252.75 | 249.36 | 249.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 253.90 | 251.37 | 250.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 13:15:00 | 254.45 | 254.55 | 252.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 14:00:00 | 254.45 | 254.55 | 252.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 247.80 | 253.00 | 252.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:15:00 | 250.30 | 253.00 | 252.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 255.60 | 253.52 | 252.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:30:00 | 257.75 | 255.54 | 253.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 250.70 | 255.32 | 255.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 250.70 | 255.32 | 255.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 250.10 | 252.45 | 253.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 11:15:00 | 252.60 | 252.48 | 253.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 12:00:00 | 252.60 | 252.48 | 253.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 252.35 | 250.72 | 252.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 252.35 | 250.72 | 252.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 257.00 | 251.97 | 252.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 257.55 | 251.97 | 252.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 256.85 | 253.49 | 253.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 13:15:00 | 258.35 | 254.46 | 253.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 15:15:00 | 259.55 | 260.02 | 257.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 09:15:00 | 257.15 | 260.02 | 257.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 255.50 | 259.11 | 257.66 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 253.15 | 256.51 | 256.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 251.45 | 255.50 | 256.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 261.50 | 255.39 | 255.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 261.50 | 255.39 | 255.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 261.50 | 255.39 | 255.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 261.50 | 255.39 | 255.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 266.10 | 257.54 | 256.81 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 11:15:00 | 252.10 | 256.98 | 257.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 250.55 | 255.69 | 256.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 255.65 | 254.34 | 255.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 255.65 | 254.34 | 255.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 255.65 | 254.34 | 255.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:45:00 | 256.15 | 254.34 | 255.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 260.95 | 255.66 | 256.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 260.95 | 255.66 | 256.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 263.15 | 257.16 | 256.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 14:15:00 | 263.85 | 258.50 | 257.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 11:15:00 | 273.55 | 273.98 | 269.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 11:30:00 | 274.40 | 273.98 | 269.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 272.45 | 273.39 | 270.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 278.55 | 271.08 | 270.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:45:00 | 274.90 | 274.92 | 274.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 12:15:00 | 274.95 | 275.41 | 275.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:00:00 | 276.20 | 275.57 | 275.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 274.90 | 275.43 | 275.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 13:30:00 | 274.25 | 275.43 | 275.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 275.70 | 275.49 | 275.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:45:00 | 275.50 | 275.49 | 275.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 15:15:00 | 275.10 | 275.41 | 275.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:15:00 | 278.65 | 275.41 | 275.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 281.35 | 276.60 | 275.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:15:00 | 284.55 | 276.60 | 275.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:00:00 | 282.65 | 278.94 | 277.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 10:15:00 | 302.39 | 293.03 | 289.95 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 12:15:00 | 297.55 | 299.94 | 300.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 13:15:00 | 295.50 | 299.06 | 299.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 299.60 | 298.21 | 299.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 299.60 | 298.21 | 299.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 299.60 | 298.21 | 299.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 299.60 | 298.21 | 299.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 299.20 | 298.41 | 299.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:45:00 | 300.15 | 298.41 | 299.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 297.85 | 298.30 | 298.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:30:00 | 296.55 | 297.96 | 298.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 296.75 | 294.73 | 295.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:15:00 | 296.55 | 295.76 | 295.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 13:15:00 | 298.85 | 296.37 | 296.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 298.85 | 296.37 | 296.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 310.70 | 299.74 | 297.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 304.10 | 305.30 | 303.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 14:00:00 | 304.10 | 305.30 | 303.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 302.90 | 304.82 | 303.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:15:00 | 304.25 | 304.82 | 303.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 304.25 | 304.71 | 303.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 299.40 | 304.71 | 303.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 297.15 | 303.20 | 302.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 297.15 | 303.20 | 302.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 293.20 | 301.20 | 301.83 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2026-05-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 13:15:00 | 304.45 | 300.71 | 300.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 305.80 | 302.42 | 301.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 302.00 | 302.33 | 301.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 302.00 | 302.33 | 301.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 302.00 | 302.33 | 301.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:30:00 | 302.80 | 302.33 | 301.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 302.95 | 302.46 | 301.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:30:00 | 301.40 | 302.46 | 301.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 302.75 | 302.51 | 301.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 12:15:00 | 303.95 | 302.51 | 301.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 12:45:00 | 303.15 | 303.41 | 302.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:30:00 | 304.60 | 303.79 | 303.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 09:15:00 | 297.80 | 302.85 | 303.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 297.80 | 302.85 | 303.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 13:15:00 | 295.70 | 299.49 | 301.33 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-14 09:15:00 | 180.19 | 2024-05-16 15:15:00 | 180.23 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest1 | 2024-05-22 09:15:00 | 189.66 | 2024-05-24 09:15:00 | 187.54 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest1 | 2024-05-22 10:30:00 | 187.66 | 2024-05-24 09:15:00 | 187.54 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2024-05-23 12:15:00 | 189.14 | 2024-05-24 12:15:00 | 187.03 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-06-19 11:30:00 | 217.52 | 2024-06-26 12:15:00 | 223.21 | STOP_HIT | 1.00 | 2.62% |
| BUY | retest2 | 2024-07-05 09:15:00 | 223.09 | 2024-07-08 10:15:00 | 218.97 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-07-08 09:15:00 | 222.83 | 2024-07-08 10:15:00 | 218.97 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-07-08 09:45:00 | 222.26 | 2024-07-08 10:15:00 | 218.97 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest1 | 2024-07-09 09:15:00 | 215.76 | 2024-07-09 12:15:00 | 229.80 | STOP_HIT | 1.00 | -6.51% |
| SELL | retest1 | 2024-07-09 10:45:00 | 216.45 | 2024-07-09 12:15:00 | 229.80 | STOP_HIT | 1.00 | -6.17% |
| SELL | retest2 | 2024-07-26 10:45:00 | 210.35 | 2024-07-29 09:15:00 | 213.51 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-07-26 11:15:00 | 209.71 | 2024-07-29 09:15:00 | 213.51 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-07-29 12:00:00 | 210.21 | 2024-07-29 13:15:00 | 214.01 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-08-02 10:30:00 | 224.47 | 2024-08-05 09:15:00 | 216.61 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest2 | 2024-08-02 14:00:00 | 223.74 | 2024-08-05 09:15:00 | 216.61 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2024-08-02 15:15:00 | 224.00 | 2024-08-05 09:15:00 | 216.61 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2024-08-19 10:45:00 | 197.70 | 2024-08-23 11:15:00 | 197.16 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2024-08-19 13:30:00 | 198.58 | 2024-08-23 11:15:00 | 197.16 | STOP_HIT | 1.00 | 0.72% |
| SELL | retest2 | 2024-08-19 14:30:00 | 198.67 | 2024-08-23 11:15:00 | 197.16 | STOP_HIT | 1.00 | 0.76% |
| SELL | retest2 | 2024-08-20 09:15:00 | 194.64 | 2024-08-23 11:15:00 | 197.16 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-08-20 14:00:00 | 193.20 | 2024-08-23 11:15:00 | 197.16 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-08-22 12:30:00 | 193.20 | 2024-08-23 11:15:00 | 197.16 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-09-03 11:30:00 | 187.40 | 2024-09-04 11:15:00 | 189.60 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-09-04 10:15:00 | 187.64 | 2024-09-04 11:15:00 | 189.60 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-09-09 09:15:00 | 183.60 | 2024-09-13 09:15:00 | 188.55 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2024-09-17 11:15:00 | 189.95 | 2024-09-18 13:15:00 | 186.18 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-09-17 14:45:00 | 189.61 | 2024-09-18 13:15:00 | 186.18 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-09-18 10:45:00 | 189.29 | 2024-09-18 13:15:00 | 186.18 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-09-30 09:15:00 | 216.09 | 2024-10-03 09:15:00 | 212.00 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-10-09 14:15:00 | 194.00 | 2024-10-18 09:15:00 | 184.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-10 11:15:00 | 194.05 | 2024-10-18 09:15:00 | 184.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-10 12:15:00 | 194.10 | 2024-10-18 09:15:00 | 184.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-10 12:45:00 | 194.50 | 2024-10-18 09:15:00 | 184.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-09 14:15:00 | 194.00 | 2024-10-18 11:15:00 | 187.20 | STOP_HIT | 0.50 | 3.51% |
| SELL | retest2 | 2024-10-10 11:15:00 | 194.05 | 2024-10-18 11:15:00 | 187.20 | STOP_HIT | 0.50 | 3.53% |
| SELL | retest2 | 2024-10-10 12:15:00 | 194.10 | 2024-10-18 11:15:00 | 187.20 | STOP_HIT | 0.50 | 3.55% |
| SELL | retest2 | 2024-10-10 12:45:00 | 194.50 | 2024-10-18 11:15:00 | 187.20 | STOP_HIT | 0.50 | 3.75% |
| SELL | retest2 | 2024-10-11 11:15:00 | 191.50 | 2024-10-21 13:15:00 | 193.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-11-12 15:00:00 | 195.26 | 2024-11-13 13:15:00 | 185.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 15:00:00 | 195.26 | 2024-11-14 12:15:00 | 188.74 | STOP_HIT | 0.50 | 3.34% |
| BUY | retest2 | 2024-12-16 09:15:00 | 236.27 | 2024-12-16 13:15:00 | 231.00 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2024-12-20 12:45:00 | 221.40 | 2024-12-24 13:15:00 | 210.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 14:00:00 | 221.15 | 2024-12-24 13:15:00 | 210.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 14:45:00 | 221.35 | 2024-12-24 13:15:00 | 210.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 12:45:00 | 221.40 | 2024-12-30 14:15:00 | 199.26 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-20 14:00:00 | 221.15 | 2024-12-30 14:15:00 | 199.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-20 14:45:00 | 221.35 | 2024-12-30 14:15:00 | 199.22 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-01-03 11:45:00 | 210.79 | 2025-01-06 09:15:00 | 204.00 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2025-01-03 14:15:00 | 210.11 | 2025-01-06 09:15:00 | 204.00 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest1 | 2025-01-10 09:15:00 | 194.60 | 2025-01-13 11:15:00 | 186.20 | PARTIAL | 0.50 | 4.32% |
| SELL | retest1 | 2025-01-10 12:45:00 | 196.00 | 2025-01-13 12:15:00 | 184.87 | PARTIAL | 0.50 | 5.68% |
| SELL | retest1 | 2025-01-10 09:15:00 | 194.60 | 2025-01-14 14:15:00 | 185.09 | STOP_HIT | 0.50 | 4.89% |
| SELL | retest1 | 2025-01-10 12:45:00 | 196.00 | 2025-01-14 14:15:00 | 185.09 | STOP_HIT | 0.50 | 5.57% |
| SELL | retest2 | 2025-01-23 12:30:00 | 183.55 | 2025-01-27 09:15:00 | 174.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 13:15:00 | 183.14 | 2025-01-27 09:15:00 | 173.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 12:30:00 | 183.55 | 2025-01-28 09:15:00 | 165.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-23 13:15:00 | 183.14 | 2025-01-28 10:15:00 | 164.83 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-01-31 10:45:00 | 178.08 | 2025-02-03 14:15:00 | 178.04 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-01-31 12:30:00 | 178.15 | 2025-02-03 14:15:00 | 178.04 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-02-03 11:15:00 | 178.43 | 2025-02-03 14:15:00 | 178.04 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-02-07 12:30:00 | 188.22 | 2025-02-10 09:15:00 | 183.72 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-02-07 13:00:00 | 188.01 | 2025-02-10 09:15:00 | 183.72 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-02-14 10:15:00 | 165.71 | 2025-02-17 15:15:00 | 175.00 | STOP_HIT | 1.00 | -5.61% |
| SELL | retest2 | 2025-02-14 12:45:00 | 166.00 | 2025-02-17 15:15:00 | 175.00 | STOP_HIT | 1.00 | -5.42% |
| SELL | retest2 | 2025-02-14 13:45:00 | 165.45 | 2025-02-17 15:15:00 | 175.00 | STOP_HIT | 1.00 | -5.77% |
| SELL | retest2 | 2025-02-14 14:15:00 | 166.08 | 2025-02-17 15:15:00 | 175.00 | STOP_HIT | 1.00 | -5.37% |
| SELL | retest2 | 2025-02-24 09:15:00 | 170.40 | 2025-02-27 13:15:00 | 162.78 | PARTIAL | 0.50 | 4.47% |
| SELL | retest2 | 2025-02-24 13:15:00 | 171.35 | 2025-02-27 13:15:00 | 162.48 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2025-02-24 14:15:00 | 171.03 | 2025-02-27 13:15:00 | 162.58 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2025-02-25 09:45:00 | 171.14 | 2025-02-28 09:15:00 | 161.88 | PARTIAL | 0.50 | 5.41% |
| SELL | retest2 | 2025-02-24 09:15:00 | 170.40 | 2025-03-03 09:15:00 | 153.36 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-24 13:15:00 | 171.35 | 2025-03-03 09:15:00 | 154.22 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-24 14:15:00 | 171.03 | 2025-03-03 09:15:00 | 153.93 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-25 09:45:00 | 171.14 | 2025-03-03 09:15:00 | 154.03 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-03-10 09:15:00 | 180.77 | 2025-03-11 09:15:00 | 176.12 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest1 | 2025-03-10 09:45:00 | 184.51 | 2025-03-11 09:15:00 | 176.12 | STOP_HIT | 1.00 | -4.55% |
| SELL | retest2 | 2025-03-28 12:45:00 | 178.97 | 2025-04-01 09:15:00 | 183.07 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-04-04 15:00:00 | 201.14 | 2025-04-07 09:15:00 | 181.11 | STOP_HIT | 1.00 | -9.96% |
| BUY | retest2 | 2025-04-17 11:15:00 | 194.10 | 2025-04-17 13:15:00 | 191.53 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-05-07 09:15:00 | 180.15 | 2025-05-09 09:15:00 | 171.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-07 11:45:00 | 179.79 | 2025-05-09 09:15:00 | 170.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 12:00:00 | 180.20 | 2025-05-09 09:15:00 | 171.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-07 09:15:00 | 180.15 | 2025-05-09 14:15:00 | 178.00 | STOP_HIT | 0.50 | 1.19% |
| SELL | retest2 | 2025-05-07 11:45:00 | 179.79 | 2025-05-09 14:15:00 | 178.00 | STOP_HIT | 0.50 | 1.00% |
| SELL | retest2 | 2025-05-08 12:00:00 | 180.20 | 2025-05-09 14:15:00 | 178.00 | STOP_HIT | 0.50 | 1.22% |
| SELL | retest2 | 2025-06-02 12:45:00 | 193.23 | 2025-06-06 14:15:00 | 183.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-02 14:30:00 | 193.37 | 2025-06-06 14:15:00 | 183.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-02 15:15:00 | 193.45 | 2025-06-06 14:15:00 | 183.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-03 10:45:00 | 192.68 | 2025-06-06 14:15:00 | 183.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-02 12:45:00 | 193.23 | 2025-06-09 11:15:00 | 188.88 | STOP_HIT | 0.50 | 2.25% |
| SELL | retest2 | 2025-06-02 14:30:00 | 193.37 | 2025-06-09 11:15:00 | 188.88 | STOP_HIT | 0.50 | 2.32% |
| SELL | retest2 | 2025-06-02 15:15:00 | 193.45 | 2025-06-09 11:15:00 | 188.88 | STOP_HIT | 0.50 | 2.36% |
| SELL | retest2 | 2025-06-03 10:45:00 | 192.68 | 2025-06-09 11:15:00 | 188.88 | STOP_HIT | 0.50 | 1.97% |
| SELL | retest2 | 2025-06-04 14:45:00 | 190.10 | 2025-06-11 10:15:00 | 195.69 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2025-06-05 13:30:00 | 189.87 | 2025-06-11 10:15:00 | 195.69 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-06-17 11:15:00 | 184.88 | 2025-06-20 09:15:00 | 175.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:15:00 | 184.88 | 2025-06-20 10:15:00 | 180.11 | STOP_HIT | 0.50 | 2.58% |
| SELL | retest2 | 2025-06-17 11:45:00 | 184.18 | 2025-06-24 12:15:00 | 181.10 | STOP_HIT | 1.00 | 1.67% |
| BUY | retest2 | 2025-06-30 15:15:00 | 189.90 | 2025-07-01 09:15:00 | 185.72 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-07-03 10:45:00 | 183.86 | 2025-07-10 12:15:00 | 182.50 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2025-07-03 13:45:00 | 183.64 | 2025-07-10 12:15:00 | 182.50 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2025-07-04 09:30:00 | 183.85 | 2025-07-10 12:15:00 | 182.50 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2025-07-04 10:30:00 | 183.66 | 2025-07-10 12:15:00 | 182.50 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2025-07-07 12:45:00 | 182.39 | 2025-07-10 12:15:00 | 182.50 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-07-07 14:30:00 | 182.54 | 2025-07-10 12:15:00 | 182.50 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2025-07-07 15:15:00 | 182.70 | 2025-07-10 12:15:00 | 182.50 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-07-08 09:45:00 | 182.51 | 2025-07-10 12:15:00 | 182.50 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-07-09 15:15:00 | 181.29 | 2025-07-10 12:15:00 | 182.50 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-07-17 09:15:00 | 196.73 | 2025-07-25 10:15:00 | 190.63 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-08-08 14:00:00 | 189.50 | 2025-08-11 12:15:00 | 196.20 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2025-08-08 14:45:00 | 189.17 | 2025-08-11 12:15:00 | 196.20 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2025-09-09 13:45:00 | 239.10 | 2025-09-10 09:15:00 | 242.29 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-09-09 15:15:00 | 238.77 | 2025-09-10 09:15:00 | 242.29 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-09-11 11:30:00 | 242.85 | 2025-09-11 13:15:00 | 241.41 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-10-06 11:15:00 | 245.26 | 2025-10-14 15:15:00 | 243.80 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2025-10-07 09:30:00 | 245.27 | 2025-10-14 15:15:00 | 243.80 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2025-10-07 10:30:00 | 244.89 | 2025-10-14 15:15:00 | 243.80 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2025-10-31 13:45:00 | 281.20 | 2025-11-06 09:15:00 | 272.60 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2025-11-03 13:30:00 | 281.20 | 2025-11-06 09:15:00 | 272.60 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2025-11-03 14:00:00 | 281.05 | 2025-11-06 09:15:00 | 272.60 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2025-11-04 10:15:00 | 282.60 | 2025-11-06 09:15:00 | 272.60 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2025-11-10 11:30:00 | 271.75 | 2025-11-11 15:15:00 | 274.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-11-10 12:30:00 | 271.50 | 2025-11-11 15:15:00 | 274.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-11-10 15:15:00 | 271.30 | 2025-11-11 15:15:00 | 274.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-11-11 12:45:00 | 271.85 | 2025-11-11 15:15:00 | 274.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-11-13 14:00:00 | 265.35 | 2025-11-14 09:15:00 | 274.50 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2025-11-13 14:45:00 | 263.00 | 2025-11-14 09:15:00 | 274.50 | STOP_HIT | 1.00 | -4.37% |
| SELL | retest2 | 2025-11-27 09:30:00 | 234.55 | 2025-12-01 12:15:00 | 238.09 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-12-01 09:30:00 | 236.24 | 2025-12-01 12:15:00 | 238.09 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-12-09 14:15:00 | 229.67 | 2025-12-10 10:15:00 | 233.84 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-12-12 09:15:00 | 234.62 | 2025-12-17 11:15:00 | 238.21 | STOP_HIT | 1.00 | 1.53% |
| SELL | retest2 | 2025-12-22 12:00:00 | 238.36 | 2025-12-22 13:15:00 | 239.27 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-12-30 10:45:00 | 265.21 | 2026-01-08 13:15:00 | 271.05 | STOP_HIT | 1.00 | 2.20% |
| BUY | retest2 | 2025-12-30 13:15:00 | 261.80 | 2026-01-08 13:15:00 | 271.05 | STOP_HIT | 1.00 | 3.53% |
| SELL | retest2 | 2026-01-22 11:15:00 | 247.50 | 2026-01-22 14:15:00 | 250.15 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-01-22 13:15:00 | 248.00 | 2026-01-22 14:15:00 | 250.15 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-02-05 15:15:00 | 258.00 | 2026-02-06 09:15:00 | 252.57 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2026-02-12 09:30:00 | 260.45 | 2026-02-13 14:15:00 | 258.20 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-02-13 10:30:00 | 260.38 | 2026-02-13 14:15:00 | 258.20 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2026-02-13 14:00:00 | 258.84 | 2026-02-13 14:15:00 | 258.20 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2026-02-25 14:45:00 | 271.03 | 2026-02-27 14:15:00 | 266.32 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-03-09 09:15:00 | 239.90 | 2026-03-10 10:15:00 | 252.75 | STOP_HIT | 1.00 | -5.36% |
| SELL | retest2 | 2026-03-09 10:15:00 | 245.95 | 2026-03-10 10:15:00 | 252.75 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2026-03-09 12:00:00 | 245.85 | 2026-03-10 10:15:00 | 252.75 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2026-03-12 11:30:00 | 257.75 | 2026-03-13 12:15:00 | 250.70 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2026-04-01 09:15:00 | 278.55 | 2026-04-15 10:15:00 | 302.39 | TARGET_HIT | 1.00 | 8.56% |
| BUY | retest2 | 2026-04-02 12:45:00 | 274.90 | 2026-04-15 10:15:00 | 302.44 | TARGET_HIT | 1.00 | 10.02% |
| BUY | retest2 | 2026-04-06 12:15:00 | 274.95 | 2026-04-15 15:15:00 | 303.82 | TARGET_HIT | 1.00 | 10.50% |
| BUY | retest2 | 2026-04-06 13:00:00 | 276.20 | 2026-04-16 09:15:00 | 306.41 | TARGET_HIT | 1.00 | 10.94% |
| BUY | retest2 | 2026-04-07 10:15:00 | 284.55 | 2026-04-21 12:15:00 | 297.55 | STOP_HIT | 1.00 | 4.57% |
| BUY | retest2 | 2026-04-07 13:00:00 | 282.65 | 2026-04-21 12:15:00 | 297.55 | STOP_HIT | 1.00 | 5.27% |
| SELL | retest2 | 2026-04-23 10:30:00 | 296.55 | 2026-04-27 13:15:00 | 298.85 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-04-27 11:15:00 | 296.75 | 2026-04-27 13:15:00 | 298.85 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-04-27 13:15:00 | 296.55 | 2026-04-27 13:15:00 | 298.85 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-05-05 12:15:00 | 303.95 | 2026-05-08 09:15:00 | 297.80 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2026-05-06 12:45:00 | 303.15 | 2026-05-08 09:15:00 | 297.80 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-05-06 13:30:00 | 304.60 | 2026-05-08 09:15:00 | 297.80 | STOP_HIT | 1.00 | -2.23% |
