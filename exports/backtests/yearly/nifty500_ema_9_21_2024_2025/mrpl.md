# Mangalore Refinery & Petrochemicals Ltd. (MRPL)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 167.97
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 151 |
| ALERT1 | 105 |
| ALERT2 | 101 |
| ALERT2_SKIP | 53 |
| ALERT3 | 235 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 124 |
| PARTIAL | 40 |
| TARGET_HIT | 13 |
| STOP_HIT | 115 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 168 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 99 / 69
- **Target hits / Stop hits / Partials:** 13 / 115 / 40
- **Avg / median % per leg:** 2.24% / 2.65%
- **Sum % (uncompounded):** 376.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 49 | 14 | 28.6% | 9 | 40 | 0 | 0.50% | 24.3% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.30% | -13.2% |
| BUY @ 3rd Alert (retest2) | 45 | 14 | 31.1% | 9 | 36 | 0 | 0.83% | 37.5% |
| SELL (all) | 119 | 85 | 71.4% | 4 | 75 | 40 | 2.96% | 351.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 119 | 85 | 71.4% | 4 | 75 | 40 | 2.96% | 351.8% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.30% | -13.2% |
| retest2 (combined) | 164 | 99 | 60.4% | 13 | 111 | 40 | 2.37% | 389.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 12:15:00 | 211.50 | 209.84 | 209.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 213.00 | 210.71 | 210.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 209.45 | 211.08 | 210.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 13:15:00 | 209.45 | 211.08 | 210.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 209.45 | 211.08 | 210.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:00:00 | 209.45 | 211.08 | 210.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 208.05 | 210.48 | 210.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:45:00 | 209.30 | 210.48 | 210.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 15:15:00 | 208.95 | 210.17 | 210.29 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 09:15:00 | 212.80 | 209.97 | 209.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 11:15:00 | 214.00 | 210.77 | 210.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 13:15:00 | 212.30 | 212.43 | 211.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 14:00:00 | 212.30 | 212.43 | 211.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 211.65 | 212.27 | 211.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 15:00:00 | 211.65 | 212.27 | 211.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 212.35 | 212.29 | 211.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 214.15 | 212.29 | 211.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-22 09:15:00 | 210.85 | 212.00 | 211.54 | SL hit (close<static) qty=1.00 sl=211.25 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 11:15:00 | 209.55 | 211.16 | 211.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 12:15:00 | 208.90 | 210.71 | 211.00 | Break + close below crossover candle low |

### Cycle 5 — BUY (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 09:15:00 | 213.70 | 211.05 | 211.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 10:15:00 | 220.70 | 214.18 | 212.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 14:15:00 | 217.90 | 218.39 | 216.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-27 15:00:00 | 217.90 | 218.39 | 216.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 213.50 | 217.45 | 216.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:00:00 | 213.50 | 217.45 | 216.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 214.95 | 216.95 | 216.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:45:00 | 215.70 | 216.95 | 216.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 214.80 | 216.07 | 216.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 210.65 | 214.28 | 214.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 214.00 | 206.71 | 208.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 214.00 | 206.71 | 208.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 214.00 | 206.71 | 208.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:00:00 | 214.00 | 206.71 | 208.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 213.75 | 208.12 | 209.08 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 212.95 | 209.95 | 209.80 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 195.85 | 207.18 | 208.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 185.50 | 202.84 | 206.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 200.85 | 198.75 | 201.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 13:00:00 | 200.85 | 198.75 | 201.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 200.50 | 199.10 | 201.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:00:00 | 200.50 | 199.10 | 201.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 207.25 | 200.62 | 201.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 206.55 | 200.62 | 201.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 206.75 | 201.85 | 202.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:45:00 | 206.95 | 201.85 | 202.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 202.40 | 202.05 | 202.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:30:00 | 203.00 | 202.05 | 202.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 202.00 | 202.04 | 202.25 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 204.55 | 202.50 | 202.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 09:15:00 | 208.49 | 206.44 | 205.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 14:15:00 | 213.70 | 213.96 | 211.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 15:00:00 | 213.70 | 213.96 | 211.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 215.00 | 216.20 | 214.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 15:00:00 | 215.00 | 216.20 | 214.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 215.00 | 215.96 | 214.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 09:15:00 | 218.85 | 215.96 | 214.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 212.70 | 214.92 | 214.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 09:15:00 | 212.70 | 214.92 | 214.98 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 10:15:00 | 215.50 | 214.91 | 214.90 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 12:15:00 | 213.50 | 214.73 | 214.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 13:15:00 | 213.26 | 214.44 | 214.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 214.15 | 213.93 | 214.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 214.15 | 213.93 | 214.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 214.15 | 213.93 | 214.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:45:00 | 216.65 | 213.93 | 214.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 213.96 | 213.94 | 214.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 12:30:00 | 212.20 | 213.56 | 214.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 15:15:00 | 212.51 | 213.30 | 213.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 09:15:00 | 216.69 | 213.85 | 214.01 | SL hit (close>static) qty=1.00 sl=214.68 alert=retest2 |

### Cycle 13 — BUY (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 10:15:00 | 223.19 | 215.72 | 214.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 11:15:00 | 226.30 | 217.83 | 215.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 09:15:00 | 219.57 | 221.40 | 218.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-25 10:00:00 | 219.57 | 221.40 | 218.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 218.13 | 220.75 | 218.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:00:00 | 218.13 | 220.75 | 218.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 217.96 | 220.19 | 218.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:15:00 | 218.22 | 220.19 | 218.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 217.97 | 219.75 | 218.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 13:15:00 | 217.60 | 219.75 | 218.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 09:15:00 | 214.08 | 217.27 | 217.67 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 15:15:00 | 218.20 | 217.74 | 217.70 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 09:15:00 | 216.93 | 217.57 | 217.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 10:15:00 | 216.30 | 217.32 | 217.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 216.97 | 215.75 | 216.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 09:15:00 | 216.97 | 215.75 | 216.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 216.97 | 215.75 | 216.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:30:00 | 218.49 | 215.75 | 216.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 216.72 | 215.95 | 216.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:30:00 | 216.87 | 215.95 | 216.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 215.78 | 215.91 | 216.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 12:30:00 | 215.16 | 215.93 | 216.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 14:15:00 | 215.00 | 215.83 | 216.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 13:45:00 | 215.27 | 215.75 | 215.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 09:15:00 | 218.08 | 216.34 | 216.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 218.08 | 216.34 | 216.19 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 214.50 | 215.96 | 216.05 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 11:15:00 | 218.82 | 215.82 | 215.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 09:15:00 | 220.00 | 218.21 | 217.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 14:15:00 | 218.10 | 218.59 | 218.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 14:15:00 | 218.10 | 218.59 | 218.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 218.10 | 218.59 | 218.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:45:00 | 218.01 | 218.59 | 218.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 15:15:00 | 217.95 | 218.46 | 218.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 09:15:00 | 223.40 | 218.46 | 218.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-11 13:15:00 | 245.74 | 236.48 | 232.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 230.41 | 238.39 | 239.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 14:15:00 | 228.79 | 233.74 | 236.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 210.30 | 207.86 | 213.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-24 10:00:00 | 210.30 | 207.86 | 213.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 212.00 | 208.69 | 213.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:45:00 | 212.85 | 208.69 | 213.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 212.37 | 209.43 | 213.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 12:15:00 | 210.95 | 209.43 | 213.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 13:15:00 | 216.04 | 211.30 | 213.32 | SL hit (close>static) qty=1.00 sl=214.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 11:15:00 | 221.61 | 215.60 | 214.89 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 14:15:00 | 214.74 | 215.61 | 215.63 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-07-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 15:15:00 | 215.79 | 215.65 | 215.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 10:15:00 | 216.38 | 215.85 | 215.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 11:15:00 | 215.46 | 215.77 | 215.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 11:15:00 | 215.46 | 215.77 | 215.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 11:15:00 | 215.46 | 215.77 | 215.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 12:00:00 | 215.46 | 215.77 | 215.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 12:15:00 | 217.25 | 216.07 | 215.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 14:45:00 | 220.11 | 217.08 | 216.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 09:15:00 | 214.90 | 219.91 | 219.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 214.90 | 219.91 | 219.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 209.60 | 216.64 | 218.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 203.58 | 203.42 | 207.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 09:15:00 | 204.05 | 203.25 | 204.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 204.05 | 203.25 | 204.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:30:00 | 205.00 | 203.25 | 204.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 207.80 | 204.16 | 204.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 11:00:00 | 207.80 | 204.16 | 204.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 11:15:00 | 205.80 | 204.49 | 204.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 12:15:00 | 205.21 | 204.49 | 204.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 14:15:00 | 207.78 | 205.05 | 204.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 14:15:00 | 207.78 | 205.05 | 204.84 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 12:15:00 | 204.04 | 204.98 | 205.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 15:15:00 | 203.89 | 204.57 | 204.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 10:15:00 | 204.66 | 204.56 | 204.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 10:15:00 | 204.66 | 204.56 | 204.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 204.66 | 204.56 | 204.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:45:00 | 204.60 | 204.56 | 204.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 204.74 | 204.60 | 204.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:30:00 | 205.27 | 204.60 | 204.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 204.98 | 204.68 | 204.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:30:00 | 205.61 | 204.68 | 204.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 205.05 | 204.75 | 204.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:00:00 | 205.05 | 204.75 | 204.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 203.24 | 204.45 | 204.69 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 212.00 | 205.89 | 205.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-23 09:15:00 | 217.90 | 213.80 | 212.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 14:15:00 | 214.19 | 214.65 | 213.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 15:00:00 | 214.19 | 214.65 | 213.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 214.00 | 214.52 | 213.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 213.95 | 214.52 | 213.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 212.37 | 214.09 | 213.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:00:00 | 212.37 | 214.09 | 213.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 211.48 | 213.57 | 213.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:30:00 | 211.00 | 213.57 | 213.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 11:15:00 | 211.40 | 213.13 | 213.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 09:15:00 | 209.07 | 211.43 | 212.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 13:15:00 | 209.10 | 208.43 | 209.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-28 14:00:00 | 209.10 | 208.43 | 209.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 209.09 | 208.56 | 209.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 15:15:00 | 209.04 | 208.56 | 209.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 209.04 | 208.66 | 209.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 210.60 | 208.66 | 209.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 209.92 | 208.91 | 209.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 12:00:00 | 207.24 | 208.57 | 209.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 14:45:00 | 207.67 | 208.43 | 208.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 09:30:00 | 207.68 | 207.98 | 208.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 11:15:00 | 206.30 | 208.05 | 208.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 203.75 | 207.19 | 208.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 13:15:00 | 202.87 | 206.50 | 207.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 10:30:00 | 203.32 | 204.47 | 206.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 10:15:00 | 202.94 | 204.10 | 205.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 11:30:00 | 202.75 | 202.84 | 203.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 198.46 | 201.70 | 202.78 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:15:00 | 197.29 | 201.70 | 202.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:15:00 | 197.30 | 201.70 | 202.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 14:15:00 | 196.88 | 199.16 | 200.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 192.93 | 198.76 | 200.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 195.99 | 197.45 | 199.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 192.73 | 197.45 | 199.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 193.15 | 197.45 | 199.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 192.79 | 197.45 | 199.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 192.61 | 197.45 | 199.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-10 12:15:00 | 192.55 | 192.02 | 194.53 | SL hit (close>ema200) qty=0.50 sl=192.02 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 15:15:00 | 191.80 | 190.46 | 190.41 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 11:15:00 | 188.94 | 190.09 | 190.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 09:15:00 | 187.68 | 189.48 | 189.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 09:15:00 | 192.20 | 188.66 | 189.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 09:15:00 | 192.20 | 188.66 | 189.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 192.20 | 188.66 | 189.02 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 10:15:00 | 192.65 | 189.46 | 189.35 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 09:15:00 | 185.10 | 188.80 | 189.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 10:15:00 | 184.12 | 187.86 | 188.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 09:15:00 | 184.75 | 183.24 | 184.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 09:15:00 | 184.75 | 183.24 | 184.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 184.75 | 183.24 | 184.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 12:30:00 | 182.01 | 183.70 | 184.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 09:45:00 | 182.00 | 178.58 | 179.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 10:15:00 | 181.43 | 180.19 | 180.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 10:15:00 | 181.43 | 180.19 | 180.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 14:15:00 | 182.48 | 181.38 | 180.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 13:15:00 | 182.14 | 182.19 | 181.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 14:15:00 | 181.98 | 182.19 | 181.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 180.65 | 181.88 | 181.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 180.65 | 181.88 | 181.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 181.00 | 181.71 | 181.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:15:00 | 179.63 | 181.71 | 181.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 178.90 | 181.14 | 181.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 178.00 | 180.21 | 180.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 10:15:00 | 179.35 | 178.75 | 179.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 10:15:00 | 179.35 | 178.75 | 179.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 179.35 | 178.75 | 179.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:30:00 | 179.91 | 178.75 | 179.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 178.68 | 178.73 | 179.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 12:30:00 | 177.20 | 178.38 | 179.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 09:30:00 | 177.69 | 177.92 | 178.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 13:15:00 | 168.34 | 173.79 | 176.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 13:15:00 | 168.81 | 173.79 | 176.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-07 15:15:00 | 173.33 | 173.18 | 175.58 | SL hit (close>ema200) qty=0.50 sl=173.18 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 177.90 | 175.34 | 175.24 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 11:15:00 | 175.01 | 175.99 | 176.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 12:15:00 | 174.97 | 175.79 | 175.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 09:15:00 | 170.95 | 170.32 | 171.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 170.95 | 170.32 | 171.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 170.95 | 170.32 | 171.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:45:00 | 173.31 | 170.32 | 171.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 14:15:00 | 153.50 | 150.26 | 153.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 15:15:00 | 156.95 | 150.26 | 153.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 15:15:00 | 156.95 | 151.60 | 153.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:15:00 | 151.03 | 151.60 | 153.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:15:00 | 151.65 | 154.32 | 154.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 14:15:00 | 143.48 | 148.32 | 150.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 14:15:00 | 144.07 | 148.32 | 150.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 147.03 | 146.85 | 149.45 | SL hit (close>ema200) qty=0.50 sl=146.85 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 14:15:00 | 148.83 | 147.35 | 147.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 150.86 | 148.22 | 147.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 145.59 | 148.30 | 147.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 145.59 | 148.30 | 147.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 145.59 | 148.30 | 147.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 145.59 | 148.30 | 147.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 145.87 | 147.81 | 147.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 145.87 | 147.81 | 147.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 146.33 | 147.52 | 147.60 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 09:15:00 | 149.98 | 147.95 | 147.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 11:15:00 | 153.36 | 149.42 | 148.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 13:15:00 | 167.03 | 167.56 | 163.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 14:00:00 | 167.03 | 167.56 | 163.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 161.40 | 165.91 | 163.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:45:00 | 160.73 | 165.91 | 163.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 160.37 | 164.80 | 163.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:00:00 | 160.37 | 164.80 | 163.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 158.82 | 162.24 | 162.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 157.09 | 159.28 | 160.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 12:15:00 | 154.22 | 153.46 | 155.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-13 13:00:00 | 154.22 | 153.46 | 155.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 151.50 | 152.92 | 154.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 12:30:00 | 150.54 | 152.58 | 154.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 13:30:00 | 150.62 | 152.11 | 153.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 12:15:00 | 150.85 | 149.93 | 150.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 10:15:00 | 153.83 | 148.88 | 148.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 10:15:00 | 153.83 | 148.88 | 148.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 11:15:00 | 156.00 | 150.30 | 149.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 155.93 | 156.24 | 154.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 15:00:00 | 155.93 | 156.24 | 154.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 155.36 | 155.87 | 154.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 10:00:00 | 155.36 | 155.87 | 154.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 154.13 | 155.52 | 154.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 11:15:00 | 154.20 | 155.52 | 154.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 11:15:00 | 154.02 | 155.22 | 154.25 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2024-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 11:15:00 | 153.28 | 153.89 | 153.93 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 15:15:00 | 154.40 | 153.95 | 153.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 155.09 | 154.18 | 154.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 153.68 | 154.37 | 154.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 09:15:00 | 153.68 | 154.37 | 154.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 153.68 | 154.37 | 154.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:30:00 | 153.40 | 154.37 | 154.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 10:15:00 | 153.37 | 154.17 | 154.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 11:15:00 | 152.95 | 153.93 | 154.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 13:15:00 | 154.68 | 153.91 | 154.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 13:15:00 | 154.68 | 153.91 | 154.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 154.68 | 153.91 | 154.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:00:00 | 154.68 | 153.91 | 154.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 154.70 | 154.07 | 154.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:30:00 | 154.79 | 154.07 | 154.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2024-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 15:15:00 | 154.40 | 154.14 | 154.12 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 09:15:00 | 153.07 | 153.92 | 154.02 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 14:15:00 | 155.50 | 154.15 | 154.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 15:15:00 | 155.60 | 154.44 | 154.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 15:15:00 | 155.92 | 156.03 | 155.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 15:15:00 | 155.92 | 156.03 | 155.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 155.92 | 156.03 | 155.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 159.36 | 156.03 | 155.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 09:15:00 | 154.93 | 156.69 | 156.28 | SL hit (close<static) qty=1.00 sl=155.05 alert=retest2 |

### Cycle 48 — SELL (started 2024-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 12:15:00 | 155.00 | 155.86 | 155.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 14:15:00 | 154.72 | 155.50 | 155.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 10:15:00 | 156.40 | 155.49 | 155.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 10:15:00 | 156.40 | 155.49 | 155.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 156.40 | 155.49 | 155.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:00:00 | 156.40 | 155.49 | 155.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 11:15:00 | 155.21 | 155.43 | 155.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 14:15:00 | 154.80 | 155.27 | 155.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 09:45:00 | 154.80 | 155.03 | 155.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 11:30:00 | 154.78 | 154.99 | 155.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 12:30:00 | 154.55 | 154.93 | 155.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 154.61 | 154.87 | 155.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 13:45:00 | 154.55 | 154.87 | 155.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 153.75 | 154.52 | 154.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 12:15:00 | 153.58 | 154.27 | 154.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 15:00:00 | 151.80 | 153.50 | 154.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 09:30:00 | 152.81 | 153.14 | 153.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 09:15:00 | 147.06 | 148.54 | 149.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 09:15:00 | 147.06 | 148.54 | 149.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 09:15:00 | 147.04 | 148.54 | 149.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-17 10:15:00 | 148.66 | 148.57 | 149.42 | SL hit (close>ema200) qty=0.50 sl=148.57 alert=retest2 |

### Cycle 49 — BUY (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 11:15:00 | 148.70 | 144.74 | 144.60 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 15:15:00 | 145.25 | 146.01 | 146.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 09:15:00 | 144.28 | 145.66 | 145.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 10:15:00 | 143.62 | 143.38 | 144.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 10:15:00 | 143.62 | 143.38 | 144.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 143.62 | 143.38 | 144.37 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2024-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 12:15:00 | 152.01 | 145.57 | 145.23 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 143.70 | 147.93 | 148.33 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 10:15:00 | 149.72 | 146.81 | 146.57 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 15:15:00 | 145.50 | 146.75 | 146.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 142.62 | 145.93 | 146.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 15:15:00 | 137.34 | 136.72 | 138.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 09:15:00 | 140.80 | 136.72 | 138.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 138.33 | 137.05 | 138.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 13:15:00 | 137.35 | 137.73 | 138.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 15:00:00 | 137.36 | 137.53 | 138.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 14:15:00 | 138.85 | 138.55 | 138.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 14:15:00 | 138.85 | 138.55 | 138.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 143.01 | 139.50 | 138.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 15:15:00 | 140.27 | 140.41 | 139.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-20 09:15:00 | 141.07 | 140.41 | 139.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 141.49 | 142.71 | 141.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 141.49 | 142.71 | 141.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 141.91 | 142.55 | 141.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 142.45 | 142.55 | 141.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 142.00 | 142.44 | 141.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:15:00 | 142.11 | 142.44 | 141.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 142.05 | 142.36 | 141.84 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 137.05 | 140.90 | 141.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 135.90 | 139.23 | 140.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 138.49 | 138.14 | 139.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 138.49 | 138.14 | 139.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 138.49 | 138.14 | 139.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 138.38 | 138.14 | 139.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 134.88 | 137.06 | 138.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 12:45:00 | 133.67 | 135.94 | 137.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 126.99 | 131.98 | 134.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 125.73 | 124.73 | 127.50 | SL hit (close>ema200) qty=0.50 sl=124.73 alert=retest2 |

### Cycle 57 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 128.40 | 127.16 | 127.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 12:15:00 | 128.95 | 127.52 | 127.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 129.24 | 129.33 | 128.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 129.24 | 129.33 | 128.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 129.24 | 129.33 | 128.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 128.71 | 129.33 | 128.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 127.50 | 128.97 | 128.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 127.50 | 128.97 | 128.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 129.37 | 129.05 | 128.44 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 122.83 | 127.53 | 127.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 121.39 | 125.57 | 126.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 123.60 | 123.35 | 125.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 09:45:00 | 124.40 | 123.35 | 125.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 127.34 | 124.23 | 124.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:30:00 | 127.80 | 124.23 | 124.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 11:15:00 | 126.78 | 125.17 | 124.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 12:15:00 | 127.51 | 125.64 | 125.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 11:15:00 | 126.60 | 126.76 | 126.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 12:00:00 | 126.60 | 126.76 | 126.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 125.50 | 126.46 | 126.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:00:00 | 125.50 | 126.46 | 126.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 126.00 | 126.37 | 126.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:45:00 | 125.50 | 126.37 | 126.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 125.99 | 126.29 | 126.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 125.05 | 126.29 | 126.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 124.91 | 126.02 | 125.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 124.91 | 126.02 | 125.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 10:15:00 | 124.89 | 125.79 | 125.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 123.48 | 124.85 | 125.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 15:15:00 | 115.70 | 115.63 | 117.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 09:15:00 | 115.62 | 115.63 | 117.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 116.85 | 115.88 | 117.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 116.93 | 115.88 | 117.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 108.81 | 110.19 | 111.70 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 113.18 | 111.33 | 111.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 114.38 | 112.40 | 111.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 115.76 | 116.35 | 114.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 115.76 | 116.35 | 114.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 115.70 | 116.00 | 115.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:45:00 | 115.29 | 116.00 | 115.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 115.59 | 115.83 | 115.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 113.60 | 115.83 | 115.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 114.00 | 115.47 | 115.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:30:00 | 112.97 | 115.47 | 115.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 114.64 | 115.06 | 114.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:15:00 | 114.37 | 115.06 | 114.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 114.28 | 114.90 | 114.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 113.69 | 114.57 | 114.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 14:15:00 | 114.42 | 114.12 | 114.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 14:15:00 | 114.42 | 114.12 | 114.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 114.42 | 114.12 | 114.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 15:00:00 | 114.42 | 114.12 | 114.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 15:15:00 | 114.19 | 114.14 | 114.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 09:15:00 | 112.75 | 114.14 | 114.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 107.11 | 110.42 | 112.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-03-03 09:15:00 | 101.48 | 105.63 | 108.52 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 63 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 105.95 | 104.85 | 104.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 110.65 | 106.52 | 105.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 13:15:00 | 112.63 | 112.83 | 110.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 13:45:00 | 112.55 | 112.83 | 110.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 111.74 | 112.61 | 111.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:30:00 | 111.26 | 112.61 | 111.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 110.80 | 112.24 | 111.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 111.70 | 112.24 | 111.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 113.44 | 112.48 | 111.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 11:15:00 | 114.20 | 112.54 | 111.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 14:15:00 | 113.81 | 112.90 | 112.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 09:15:00 | 113.82 | 112.87 | 112.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 15:15:00 | 111.75 | 112.14 | 112.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 15:15:00 | 111.75 | 112.14 | 112.16 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 09:15:00 | 113.25 | 112.36 | 112.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 10:15:00 | 117.30 | 113.35 | 112.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-17 12:15:00 | 117.16 | 117.69 | 116.10 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 13:30:00 | 118.10 | 117.75 | 116.27 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 14:45:00 | 117.90 | 117.76 | 116.41 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 116.40 | 117.49 | 116.41 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-17 15:15:00 | 116.40 | 117.49 | 116.41 | SL hit (close<ema400) qty=1.00 sl=116.41 alert=retest1 |

### Cycle 66 — SELL (started 2025-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-18 14:15:00 | 114.82 | 115.87 | 115.95 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 117.53 | 116.05 | 116.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 09:15:00 | 121.58 | 118.37 | 117.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 137.68 | 138.15 | 131.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:00:00 | 137.68 | 138.15 | 131.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 137.60 | 139.58 | 137.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:30:00 | 136.62 | 139.58 | 137.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 138.04 | 139.27 | 137.61 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 12:15:00 | 135.40 | 137.24 | 137.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 15:15:00 | 133.79 | 135.82 | 136.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 133.97 | 133.80 | 134.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 133.97 | 133.80 | 134.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 133.97 | 133.80 | 134.83 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 137.47 | 135.49 | 135.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 15:15:00 | 138.84 | 136.16 | 135.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 12:15:00 | 136.26 | 136.51 | 136.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 12:15:00 | 136.26 | 136.51 | 136.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 136.26 | 136.51 | 136.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 12:45:00 | 136.10 | 136.51 | 136.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 14:15:00 | 137.27 | 136.66 | 136.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 15:00:00 | 137.27 | 136.66 | 136.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 133.98 | 136.19 | 136.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 133.98 | 136.19 | 136.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 134.46 | 135.85 | 135.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 121.53 | 132.06 | 133.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 129.10 | 128.07 | 130.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 14:15:00 | 129.10 | 128.07 | 130.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 129.10 | 128.07 | 130.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 14:45:00 | 130.12 | 128.07 | 130.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 130.97 | 128.80 | 130.71 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2025-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 14:15:00 | 134.87 | 131.57 | 131.45 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-11 14:15:00 | 131.52 | 132.18 | 132.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-11 15:15:00 | 129.85 | 131.72 | 131.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 131.98 | 131.77 | 131.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-15 09:15:00 | 131.98 | 131.77 | 131.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 131.98 | 131.77 | 131.98 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2025-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 12:15:00 | 132.62 | 132.16 | 132.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 13:15:00 | 133.01 | 132.33 | 132.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 136.34 | 136.52 | 135.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-21 09:15:00 | 140.70 | 137.44 | 136.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 140.70 | 137.44 | 136.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 11:00:00 | 143.30 | 138.61 | 136.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 14:15:00 | 141.48 | 140.12 | 138.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 14:45:00 | 141.72 | 140.27 | 138.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 09:15:00 | 142.66 | 140.45 | 138.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 138.56 | 141.41 | 140.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 138.56 | 141.41 | 140.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 139.89 | 141.11 | 140.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 11:15:00 | 140.18 | 141.11 | 140.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 136.25 | 140.30 | 140.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 136.25 | 140.30 | 140.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 09:15:00 | 132.58 | 136.51 | 138.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 135.69 | 135.55 | 136.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 135.69 | 135.55 | 136.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 135.69 | 135.55 | 136.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:00:00 | 133.97 | 135.24 | 136.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 13:15:00 | 134.35 | 135.04 | 136.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 10:30:00 | 134.98 | 135.04 | 135.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 12:15:00 | 135.13 | 135.11 | 135.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 135.36 | 134.74 | 135.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:45:00 | 135.55 | 134.74 | 135.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 135.00 | 134.79 | 135.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:30:00 | 135.47 | 134.79 | 135.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 133.92 | 134.35 | 134.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 15:00:00 | 133.92 | 134.35 | 134.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 138.06 | 135.01 | 135.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:45:00 | 138.12 | 135.01 | 135.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 135.37 | 135.08 | 135.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 11:15:00 | 134.89 | 135.08 | 135.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 14:15:00 | 128.23 | 130.75 | 132.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 14:15:00 | 128.37 | 130.75 | 132.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 14:15:00 | 128.15 | 130.75 | 132.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 15:15:00 | 127.63 | 130.11 | 132.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 127.27 | 129.33 | 131.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 13:15:00 | 129.00 | 128.71 | 130.47 | SL hit (close>ema200) qty=0.50 sl=128.71 alert=retest2 |

### Cycle 75 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 131.23 | 128.49 | 128.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 131.99 | 130.42 | 129.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 12:15:00 | 138.32 | 138.36 | 136.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 13:00:00 | 138.32 | 138.36 | 136.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 141.42 | 138.75 | 137.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 137.70 | 138.75 | 137.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 139.67 | 139.69 | 138.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 139.62 | 139.69 | 138.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 138.30 | 139.42 | 138.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 138.30 | 139.42 | 138.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 138.99 | 139.33 | 138.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:45:00 | 139.37 | 139.34 | 138.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 10:15:00 | 139.17 | 139.32 | 138.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 12:15:00 | 137.69 | 138.70 | 138.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 137.69 | 138.70 | 138.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 13:15:00 | 136.71 | 138.30 | 138.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 138.18 | 137.93 | 138.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 10:15:00 | 138.18 | 137.93 | 138.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 138.18 | 137.93 | 138.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 138.18 | 137.93 | 138.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 138.88 | 138.12 | 138.33 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 139.89 | 138.47 | 138.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 13:15:00 | 142.50 | 139.28 | 138.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 140.06 | 141.12 | 140.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 140.06 | 141.12 | 140.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 140.06 | 141.12 | 140.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:45:00 | 139.39 | 141.12 | 140.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 140.58 | 141.02 | 140.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 140.82 | 141.02 | 140.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 12:30:00 | 140.85 | 140.91 | 140.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 15:00:00 | 140.74 | 140.76 | 140.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 15:15:00 | 143.82 | 145.64 | 145.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 143.82 | 145.64 | 145.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 12:15:00 | 143.03 | 144.33 | 145.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 09:15:00 | 143.88 | 143.84 | 144.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 09:15:00 | 143.88 | 143.84 | 144.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 143.88 | 143.84 | 144.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:30:00 | 146.94 | 143.84 | 144.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 142.65 | 143.61 | 144.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:15:00 | 141.90 | 143.61 | 144.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 09:30:00 | 142.04 | 141.65 | 142.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 10:15:00 | 143.24 | 141.97 | 141.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 143.24 | 141.97 | 141.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 11:15:00 | 144.25 | 142.43 | 142.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 142.46 | 142.89 | 142.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 142.46 | 142.89 | 142.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 142.46 | 142.89 | 142.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:00:00 | 142.46 | 142.89 | 142.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 142.35 | 142.78 | 142.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 142.35 | 142.78 | 142.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 143.30 | 142.88 | 142.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 147.35 | 143.29 | 142.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 12:15:00 | 142.75 | 144.27 | 144.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 142.75 | 144.27 | 144.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 141.06 | 143.63 | 144.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 09:15:00 | 138.87 | 137.38 | 138.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 138.87 | 137.38 | 138.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 138.87 | 137.38 | 138.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 139.65 | 137.38 | 138.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 138.17 | 137.54 | 138.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 137.58 | 137.54 | 138.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 15:15:00 | 137.29 | 137.45 | 138.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 138.76 | 135.54 | 135.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 138.76 | 135.54 | 135.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 10:15:00 | 143.92 | 137.22 | 136.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 10:15:00 | 142.51 | 142.54 | 140.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 10:45:00 | 142.50 | 142.54 | 140.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 141.05 | 141.91 | 140.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:00:00 | 141.05 | 141.91 | 140.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 140.50 | 141.63 | 140.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:45:00 | 140.65 | 141.63 | 140.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 140.24 | 141.35 | 140.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:00:00 | 140.24 | 141.35 | 140.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 140.45 | 141.17 | 140.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:30:00 | 140.45 | 141.17 | 140.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 141.85 | 141.22 | 140.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 142.94 | 141.22 | 140.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 15:15:00 | 145.70 | 147.58 | 147.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 15:15:00 | 145.70 | 147.58 | 147.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 143.86 | 146.47 | 147.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 10:15:00 | 145.00 | 144.97 | 145.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 10:45:00 | 145.08 | 144.97 | 145.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 146.38 | 145.25 | 145.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:00:00 | 146.38 | 145.25 | 145.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 150.07 | 146.22 | 146.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:00:00 | 150.07 | 146.22 | 146.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 13:15:00 | 148.05 | 146.58 | 146.45 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 145.56 | 146.33 | 146.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 143.29 | 145.30 | 145.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 143.05 | 142.38 | 143.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 143.05 | 142.38 | 143.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 143.05 | 142.38 | 143.22 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 13:15:00 | 145.77 | 144.02 | 143.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 10:15:00 | 146.56 | 145.06 | 144.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 15:15:00 | 147.00 | 147.05 | 146.27 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 09:15:00 | 148.15 | 147.05 | 146.27 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 10:45:00 | 147.83 | 147.41 | 146.58 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 140.22 | 146.77 | 146.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest1 |

### Cycle 86 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 140.22 | 146.77 | 146.77 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 09:15:00 | 148.75 | 144.30 | 144.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 10:15:00 | 152.18 | 145.88 | 144.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 12:15:00 | 153.04 | 153.82 | 150.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 13:00:00 | 153.04 | 153.82 | 150.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 151.08 | 153.04 | 150.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:45:00 | 150.84 | 153.04 | 150.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 147.10 | 151.62 | 150.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 147.10 | 151.62 | 150.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 145.04 | 150.30 | 150.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 145.10 | 150.30 | 150.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 144.84 | 149.21 | 149.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 142.20 | 147.10 | 148.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 125.39 | 124.90 | 126.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:45:00 | 125.40 | 124.90 | 126.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 126.88 | 125.52 | 126.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 126.88 | 125.52 | 126.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 126.95 | 125.81 | 126.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 127.99 | 125.81 | 126.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 126.62 | 125.97 | 126.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 11:00:00 | 126.23 | 126.02 | 126.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 14:15:00 | 126.20 | 126.33 | 126.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:30:00 | 124.88 | 126.09 | 126.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:45:00 | 126.11 | 124.02 | 124.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 126.81 | 124.58 | 124.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:00:00 | 126.81 | 124.58 | 124.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-08 11:15:00 | 126.55 | 124.97 | 124.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 11:15:00 | 126.55 | 124.97 | 124.89 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 10:15:00 | 124.29 | 125.04 | 125.05 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 125.93 | 125.01 | 125.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 15:15:00 | 126.40 | 125.29 | 125.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 14:15:00 | 126.10 | 126.52 | 125.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 15:00:00 | 126.10 | 126.52 | 125.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 126.80 | 126.57 | 126.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:15:00 | 125.60 | 126.57 | 126.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 124.83 | 126.23 | 125.93 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 11:15:00 | 124.70 | 125.68 | 125.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 13:15:00 | 124.50 | 125.29 | 125.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 14:15:00 | 124.00 | 123.79 | 124.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-14 15:00:00 | 124.00 | 123.79 | 124.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 124.21 | 123.78 | 124.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:00:00 | 123.05 | 123.71 | 124.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:30:00 | 123.25 | 123.51 | 124.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 14:45:00 | 123.12 | 123.49 | 123.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 11:15:00 | 125.41 | 124.20 | 124.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 11:15:00 | 125.41 | 124.20 | 124.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 126.20 | 125.02 | 124.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 14:15:00 | 125.38 | 125.43 | 125.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 15:00:00 | 125.38 | 125.43 | 125.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 125.08 | 125.36 | 125.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 127.70 | 125.36 | 125.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 10:00:00 | 125.60 | 126.44 | 125.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:15:00 | 125.44 | 126.22 | 125.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 14:15:00 | 125.24 | 125.73 | 125.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 125.24 | 125.73 | 125.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 15:15:00 | 124.50 | 125.48 | 125.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 12:15:00 | 123.79 | 123.58 | 124.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-26 13:15:00 | 123.70 | 123.58 | 124.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 123.96 | 123.79 | 124.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 122.43 | 123.79 | 124.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 10:15:00 | 123.37 | 123.75 | 124.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 11:15:00 | 125.02 | 123.08 | 123.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 125.02 | 123.08 | 123.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 15:15:00 | 125.96 | 124.44 | 123.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 126.53 | 126.93 | 126.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 10:15:00 | 126.04 | 126.75 | 126.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 126.04 | 126.75 | 126.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 126.04 | 126.75 | 126.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 125.90 | 126.58 | 126.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:30:00 | 125.98 | 126.58 | 126.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 125.05 | 125.97 | 125.94 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 125.00 | 125.78 | 125.86 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 127.13 | 126.03 | 125.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 127.65 | 126.53 | 126.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 14:15:00 | 126.72 | 126.92 | 126.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 14:15:00 | 126.72 | 126.92 | 126.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 126.72 | 126.92 | 126.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:00:00 | 126.72 | 126.92 | 126.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 126.81 | 126.90 | 126.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 126.06 | 126.90 | 126.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 127.52 | 127.02 | 126.66 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 13:15:00 | 125.76 | 126.40 | 126.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 15:15:00 | 125.12 | 126.01 | 126.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 127.75 | 126.36 | 126.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 127.75 | 126.36 | 126.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 127.75 | 126.36 | 126.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 127.85 | 126.36 | 126.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 127.26 | 126.54 | 126.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 128.85 | 127.25 | 126.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 13:15:00 | 128.12 | 128.23 | 127.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 14:00:00 | 128.12 | 128.23 | 127.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 127.65 | 128.11 | 127.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 127.65 | 128.11 | 127.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 127.65 | 128.02 | 127.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 126.85 | 128.02 | 127.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 126.89 | 127.79 | 127.51 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 126.31 | 127.19 | 127.28 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 127.48 | 127.24 | 127.23 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 14:15:00 | 126.95 | 127.17 | 127.20 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 128.00 | 127.33 | 127.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 129.25 | 128.01 | 127.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 10:15:00 | 130.85 | 130.86 | 129.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 11:00:00 | 130.85 | 130.86 | 129.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 130.02 | 130.59 | 130.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 130.02 | 130.59 | 130.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 129.75 | 130.42 | 130.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 130.90 | 130.42 | 130.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 130.29 | 130.69 | 130.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:00:00 | 130.29 | 130.69 | 130.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 129.95 | 130.54 | 130.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 129.94 | 130.54 | 130.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 129.50 | 130.21 | 130.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 128.96 | 129.96 | 130.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 128.98 | 128.40 | 128.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 128.98 | 128.40 | 128.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 128.98 | 128.40 | 128.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:30:00 | 129.70 | 128.40 | 128.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 128.63 | 128.45 | 128.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:45:00 | 128.35 | 128.42 | 128.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 13:15:00 | 129.17 | 128.57 | 128.85 | SL hit (close>static) qty=1.00 sl=128.98 alert=retest2 |

### Cycle 105 — BUY (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 11:15:00 | 129.32 | 129.02 | 129.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-26 12:15:00 | 129.80 | 129.18 | 129.07 | Break + close above crossover candle high |

### Cycle 106 — SELL (started 2025-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 13:15:00 | 127.53 | 128.85 | 128.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 15:15:00 | 127.30 | 128.32 | 128.67 | Break + close below crossover candle low |

### Cycle 107 — BUY (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 09:15:00 | 132.57 | 129.17 | 129.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 10:15:00 | 134.66 | 130.27 | 129.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 10:15:00 | 132.85 | 132.92 | 131.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-30 10:30:00 | 132.71 | 132.92 | 131.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 131.95 | 132.63 | 131.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:00:00 | 131.95 | 132.63 | 131.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 132.58 | 132.62 | 131.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 134.00 | 132.57 | 131.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 15:00:00 | 133.26 | 133.33 | 132.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 12:15:00 | 133.37 | 132.93 | 132.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-03 14:15:00 | 146.59 | 136.38 | 134.43 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 09:15:00 | 145.28 | 145.79 | 145.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 11:15:00 | 143.84 | 145.19 | 145.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 141.11 | 140.76 | 142.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 14:15:00 | 141.11 | 140.76 | 142.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 141.11 | 140.76 | 142.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:30:00 | 141.99 | 140.76 | 142.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 141.90 | 140.82 | 141.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:45:00 | 142.20 | 140.82 | 141.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 143.85 | 141.43 | 141.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 143.85 | 141.43 | 141.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 142.27 | 141.60 | 141.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 15:00:00 | 141.77 | 141.63 | 141.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 145.97 | 142.56 | 142.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 145.97 | 142.56 | 142.30 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 140.95 | 142.59 | 142.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 140.20 | 142.11 | 142.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 10:15:00 | 143.23 | 141.79 | 142.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 10:15:00 | 143.23 | 141.79 | 142.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 143.23 | 141.79 | 142.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:00:00 | 143.23 | 141.79 | 142.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 145.65 | 142.56 | 142.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 13:15:00 | 151.00 | 145.05 | 143.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 145.87 | 147.89 | 145.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 145.87 | 147.89 | 145.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 145.87 | 147.89 | 145.90 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 15:15:00 | 145.25 | 145.63 | 145.66 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 150.40 | 146.58 | 146.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 12:15:00 | 150.80 | 148.48 | 147.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 15:15:00 | 162.00 | 162.36 | 159.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 09:15:00 | 162.44 | 162.36 | 159.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 167.70 | 163.43 | 159.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 11:15:00 | 171.85 | 166.70 | 163.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 13:30:00 | 169.50 | 168.27 | 165.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 171.72 | 168.22 | 165.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 13:15:00 | 169.48 | 173.02 | 171.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 170.85 | 172.58 | 171.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 14:15:00 | 171.29 | 172.58 | 171.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 10:45:00 | 171.25 | 171.21 | 170.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 169.13 | 171.26 | 171.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 169.13 | 171.26 | 171.27 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 171.67 | 171.27 | 171.27 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 14:15:00 | 170.83 | 171.21 | 171.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 15:15:00 | 170.00 | 170.97 | 171.13 | Break + close below crossover candle low |

### Cycle 117 — BUY (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 09:15:00 | 172.30 | 171.24 | 171.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 176.35 | 173.05 | 172.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 12:15:00 | 174.86 | 175.29 | 173.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 12:45:00 | 174.50 | 175.29 | 173.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 173.36 | 174.82 | 174.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 176.65 | 174.82 | 174.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:00:00 | 176.24 | 175.10 | 174.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 09:15:00 | 172.17 | 174.24 | 174.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 172.17 | 174.24 | 174.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 11:15:00 | 171.70 | 173.37 | 173.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 175.76 | 173.20 | 173.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 175.76 | 173.20 | 173.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 175.76 | 173.20 | 173.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 175.76 | 173.20 | 173.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 175.00 | 173.56 | 173.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:30:00 | 175.14 | 173.56 | 173.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 174.55 | 173.76 | 173.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 13:15:00 | 180.49 | 175.42 | 174.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 11:15:00 | 178.46 | 178.79 | 176.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 12:00:00 | 178.46 | 178.79 | 176.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 176.10 | 178.28 | 177.40 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 12:15:00 | 175.39 | 176.74 | 176.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 14:15:00 | 174.53 | 176.04 | 176.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 178.56 | 176.07 | 176.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 178.56 | 176.07 | 176.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 178.56 | 176.07 | 176.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 178.56 | 176.07 | 176.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 179.55 | 176.77 | 176.68 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 14:15:00 | 174.32 | 176.62 | 176.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 173.51 | 175.65 | 176.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 11:15:00 | 161.60 | 161.51 | 163.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:30:00 | 161.52 | 161.51 | 163.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 162.51 | 159.71 | 160.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 13:30:00 | 158.63 | 159.85 | 160.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 15:15:00 | 158.74 | 159.73 | 160.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 158.31 | 158.98 | 159.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 11:30:00 | 158.80 | 159.17 | 159.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 150.80 | 153.64 | 155.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 150.86 | 153.64 | 155.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 150.70 | 152.99 | 154.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 150.39 | 152.99 | 154.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 151.46 | 151.25 | 153.19 | SL hit (close>ema200) qty=0.50 sl=151.25 alert=retest2 |

### Cycle 123 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 154.70 | 153.86 | 153.77 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 153.05 | 153.71 | 153.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 151.38 | 153.24 | 153.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 15:15:00 | 149.61 | 149.50 | 150.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-15 09:15:00 | 150.20 | 149.50 | 150.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 151.38 | 149.87 | 150.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 151.81 | 149.87 | 150.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 150.95 | 150.09 | 150.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 11:30:00 | 150.48 | 150.15 | 150.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 10:15:00 | 150.00 | 150.22 | 150.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 148.98 | 147.27 | 147.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 148.98 | 147.27 | 147.19 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 13:15:00 | 146.59 | 147.54 | 147.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 15:15:00 | 146.45 | 147.20 | 147.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 146.36 | 146.35 | 146.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 10:15:00 | 146.36 | 146.35 | 146.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 146.36 | 146.35 | 146.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:45:00 | 146.59 | 146.35 | 146.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 146.77 | 146.44 | 146.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 146.77 | 146.44 | 146.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 145.61 | 146.27 | 146.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 13:30:00 | 145.49 | 146.13 | 146.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:00:00 | 145.53 | 145.69 | 146.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:00:00 | 145.60 | 145.67 | 146.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 146.81 | 143.76 | 144.35 | SL hit (close>static) qty=1.00 sl=146.77 alert=retest2 |

### Cycle 127 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 152.25 | 146.16 | 145.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 155.48 | 148.02 | 146.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 09:15:00 | 149.76 | 152.10 | 150.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 149.76 | 152.10 | 150.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 149.76 | 152.10 | 150.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 149.76 | 152.10 | 150.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 151.16 | 151.91 | 150.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 13:00:00 | 151.33 | 151.68 | 150.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 15:15:00 | 151.40 | 151.47 | 150.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:00:00 | 151.61 | 151.30 | 150.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 149.31 | 150.60 | 150.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 149.31 | 150.60 | 150.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 15:15:00 | 148.51 | 150.00 | 150.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 147.99 | 147.74 | 148.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 14:15:00 | 147.99 | 147.74 | 148.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 147.99 | 147.74 | 148.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 148.72 | 147.74 | 148.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 148.28 | 147.86 | 148.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:45:00 | 147.15 | 147.78 | 148.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 14:00:00 | 147.15 | 147.65 | 148.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 15:00:00 | 146.77 | 147.48 | 148.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 139.79 | 143.22 | 145.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 139.79 | 143.22 | 145.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 139.43 | 143.22 | 145.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 12:15:00 | 139.89 | 139.18 | 141.29 | SL hit (close>ema200) qty=0.50 sl=139.18 alert=retest2 |

### Cycle 129 — BUY (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 11:15:00 | 145.10 | 142.43 | 142.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 09:15:00 | 147.14 | 144.58 | 143.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 152.30 | 152.37 | 149.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 13:30:00 | 152.35 | 152.37 | 149.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 149.96 | 151.79 | 149.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 146.73 | 151.79 | 149.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 145.75 | 150.58 | 149.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 145.75 | 150.58 | 149.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 145.60 | 149.59 | 149.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 145.13 | 149.59 | 149.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 144.46 | 148.56 | 148.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 14:15:00 | 143.51 | 146.27 | 147.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 146.17 | 142.57 | 144.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 146.17 | 142.57 | 144.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 146.17 | 142.57 | 144.26 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 12:15:00 | 153.07 | 145.76 | 145.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 09:15:00 | 156.24 | 150.31 | 147.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 153.63 | 154.00 | 151.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 11:45:00 | 153.69 | 154.00 | 151.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 155.46 | 154.51 | 152.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 10:00:00 | 156.16 | 154.84 | 153.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 11:45:00 | 156.71 | 155.28 | 153.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 13:00:00 | 156.05 | 155.44 | 153.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 14:30:00 | 156.03 | 155.68 | 154.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 166.20 | 157.87 | 155.46 | EMA400 retest candle locked (from upside) |
| Target hit | 2026-01-29 09:15:00 | 171.78 | 164.45 | 160.58 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 165.12 | 169.29 | 169.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 162.88 | 168.01 | 169.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 169.06 | 168.22 | 169.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 169.06 | 168.22 | 169.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 169.06 | 168.22 | 169.06 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 13:15:00 | 172.69 | 169.91 | 169.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 14:15:00 | 177.26 | 171.38 | 170.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 15:15:00 | 182.49 | 182.69 | 179.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:15:00 | 187.77 | 182.69 | 179.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 182.00 | 183.56 | 182.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:30:00 | 182.55 | 183.56 | 182.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 183.27 | 183.50 | 182.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:30:00 | 184.12 | 183.84 | 182.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 185.69 | 190.67 | 191.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 185.69 | 190.67 | 191.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 10:15:00 | 183.51 | 186.38 | 188.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 185.93 | 185.69 | 187.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 185.93 | 185.69 | 187.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 187.14 | 186.02 | 187.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 11:00:00 | 185.14 | 185.84 | 187.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 191.33 | 187.00 | 187.37 | SL hit (close>static) qty=1.00 sl=188.35 alert=retest2 |

### Cycle 135 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 193.20 | 188.24 | 187.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 193.60 | 189.31 | 188.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 11:15:00 | 191.43 | 191.51 | 189.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 12:00:00 | 191.43 | 191.51 | 189.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 190.20 | 191.35 | 190.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:00:00 | 190.20 | 191.35 | 190.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 189.93 | 191.06 | 190.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:15:00 | 189.90 | 191.06 | 190.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 189.90 | 190.83 | 190.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 192.27 | 190.83 | 190.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 15:15:00 | 191.00 | 191.96 | 192.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 15:15:00 | 191.00 | 191.96 | 192.01 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 192.42 | 192.05 | 192.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 10:15:00 | 195.81 | 193.48 | 192.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 09:15:00 | 198.09 | 198.78 | 197.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 198.09 | 198.78 | 197.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 198.09 | 198.78 | 197.01 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 194.03 | 196.42 | 196.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 191.73 | 194.99 | 195.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 10:15:00 | 189.47 | 189.45 | 191.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 10:45:00 | 188.90 | 189.45 | 191.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 191.52 | 189.86 | 191.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:00:00 | 191.52 | 189.86 | 191.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 191.79 | 190.25 | 191.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 13:00:00 | 191.79 | 190.25 | 191.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 191.47 | 190.49 | 191.75 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 10:15:00 | 197.23 | 193.09 | 192.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 201.00 | 196.12 | 194.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 192.30 | 201.73 | 199.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 192.30 | 201.73 | 199.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 192.30 | 201.73 | 199.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:45:00 | 193.12 | 201.73 | 199.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 189.65 | 199.31 | 198.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:45:00 | 191.28 | 199.31 | 198.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 190.40 | 197.53 | 197.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 189.07 | 194.65 | 196.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 10:15:00 | 194.20 | 192.82 | 194.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 11:00:00 | 194.20 | 192.82 | 194.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 194.18 | 193.09 | 194.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:45:00 | 195.37 | 193.09 | 194.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 194.00 | 193.28 | 194.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:45:00 | 195.75 | 193.28 | 194.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 195.02 | 193.62 | 194.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:30:00 | 194.90 | 193.62 | 194.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 193.00 | 193.50 | 194.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:15:00 | 191.52 | 193.78 | 194.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 181.94 | 188.65 | 191.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 10:15:00 | 189.01 | 188.72 | 191.03 | SL hit (close>ema200) qty=0.50 sl=188.72 alert=retest2 |

### Cycle 141 — BUY (started 2026-03-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 12:15:00 | 195.81 | 188.09 | 187.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-16 13:15:00 | 200.55 | 190.58 | 188.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 09:15:00 | 195.59 | 196.24 | 192.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-17 09:30:00 | 195.93 | 196.24 | 192.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 202.50 | 197.49 | 193.20 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2026-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 15:15:00 | 190.81 | 194.18 | 194.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 188.29 | 193.00 | 194.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 194.75 | 188.77 | 190.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 194.75 | 188.77 | 190.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 194.75 | 188.77 | 190.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 197.37 | 188.77 | 190.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 196.90 | 190.40 | 191.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 196.90 | 190.40 | 191.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 192.30 | 191.46 | 191.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 15:00:00 | 191.76 | 191.52 | 191.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 15:15:00 | 182.17 | 185.53 | 188.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 15:15:00 | 181.65 | 181.37 | 184.04 | SL hit (close>ema200) qty=0.50 sl=181.37 alert=retest2 |

### Cycle 143 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 187.22 | 185.21 | 185.18 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 179.10 | 184.33 | 184.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 177.47 | 180.52 | 182.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 09:15:00 | 182.33 | 180.25 | 182.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 182.33 | 180.25 | 182.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 182.33 | 180.25 | 182.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:30:00 | 183.17 | 180.25 | 182.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 181.82 | 180.57 | 182.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:15:00 | 181.37 | 180.57 | 182.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 183.60 | 181.17 | 182.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:30:00 | 182.76 | 181.17 | 182.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 12:15:00 | 182.36 | 181.41 | 182.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 13:45:00 | 182.09 | 181.56 | 182.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 14:15:00 | 181.89 | 181.56 | 182.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 11:15:00 | 184.20 | 182.14 | 182.15 | SL hit (close>static) qty=1.00 sl=184.16 alert=retest2 |

### Cycle 145 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 185.87 | 182.89 | 182.49 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 176.93 | 182.10 | 182.32 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 182.65 | 180.68 | 180.52 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 15:15:00 | 180.00 | 180.54 | 180.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-09 09:15:00 | 178.09 | 180.05 | 180.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 172.88 | 172.26 | 174.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 172.88 | 172.26 | 174.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 172.88 | 172.26 | 174.12 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2026-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 15:15:00 | 176.20 | 174.97 | 174.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 178.19 | 175.62 | 175.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 14:15:00 | 175.25 | 176.24 | 175.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 14:15:00 | 175.25 | 176.24 | 175.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 175.25 | 176.24 | 175.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 15:00:00 | 175.25 | 176.24 | 175.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 15:15:00 | 175.00 | 175.99 | 175.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 175.84 | 175.99 | 175.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-24 09:15:00 | 193.42 | 188.80 | 186.62 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 09:15:00 | 173.27 | 184.63 | 185.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 14:15:00 | 172.32 | 176.92 | 180.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-28 15:15:00 | 174.31 | 174.15 | 177.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 09:15:00 | 173.87 | 174.15 | 177.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 174.17 | 174.15 | 176.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:45:00 | 172.61 | 174.32 | 176.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:45:00 | 173.38 | 174.06 | 175.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:30:00 | 172.89 | 172.93 | 174.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 09:15:00 | 163.98 | 167.81 | 170.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 09:15:00 | 164.71 | 167.81 | 170.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 09:15:00 | 164.25 | 167.81 | 170.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-05-05 10:15:00 | 156.04 | 160.29 | 164.73 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 151 — BUY (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 09:15:00 | 167.88 | 158.10 | 157.94 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-22 09:15:00 | 214.15 | 2024-05-22 09:15:00 | 210.85 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-06-18 09:15:00 | 218.85 | 2024-06-19 09:15:00 | 212.70 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2024-06-21 12:30:00 | 212.20 | 2024-06-24 09:15:00 | 216.69 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2024-06-21 15:15:00 | 212.51 | 2024-06-24 09:15:00 | 216.69 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2024-06-28 12:30:00 | 215.16 | 2024-07-02 09:15:00 | 218.08 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-06-28 14:15:00 | 215.00 | 2024-07-02 09:15:00 | 218.08 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-07-01 13:45:00 | 215.27 | 2024-07-02 09:15:00 | 218.08 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-07-08 09:15:00 | 223.40 | 2024-07-11 13:15:00 | 245.74 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-24 12:15:00 | 210.95 | 2024-07-24 13:15:00 | 216.04 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2024-07-29 14:45:00 | 220.11 | 2024-08-02 09:15:00 | 214.90 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2024-08-09 12:15:00 | 205.21 | 2024-08-12 14:15:00 | 207.78 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-08-29 12:00:00 | 207.24 | 2024-09-06 09:15:00 | 197.29 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2024-08-30 14:45:00 | 207.67 | 2024-09-06 09:15:00 | 197.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-02 09:30:00 | 207.68 | 2024-09-06 14:15:00 | 196.88 | PARTIAL | 0.50 | 5.20% |
| SELL | retest2 | 2024-09-02 11:15:00 | 206.30 | 2024-09-09 09:15:00 | 195.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-02 13:15:00 | 202.87 | 2024-09-09 09:15:00 | 192.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-03 10:30:00 | 203.32 | 2024-09-09 09:15:00 | 193.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-04 10:15:00 | 202.94 | 2024-09-09 09:15:00 | 192.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 11:30:00 | 202.75 | 2024-09-09 09:15:00 | 192.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-29 12:00:00 | 207.24 | 2024-09-10 12:15:00 | 192.55 | STOP_HIT | 0.50 | 7.09% |
| SELL | retest2 | 2024-08-30 14:45:00 | 207.67 | 2024-09-10 12:15:00 | 192.55 | STOP_HIT | 0.50 | 7.28% |
| SELL | retest2 | 2024-09-02 09:30:00 | 207.68 | 2024-09-10 12:15:00 | 192.55 | STOP_HIT | 0.50 | 7.29% |
| SELL | retest2 | 2024-09-02 11:15:00 | 206.30 | 2024-09-10 12:15:00 | 192.55 | STOP_HIT | 0.50 | 6.67% |
| SELL | retest2 | 2024-09-02 13:15:00 | 202.87 | 2024-09-10 12:15:00 | 192.55 | STOP_HIT | 0.50 | 5.09% |
| SELL | retest2 | 2024-09-03 10:30:00 | 203.32 | 2024-09-10 12:15:00 | 192.55 | STOP_HIT | 0.50 | 5.30% |
| SELL | retest2 | 2024-09-04 10:15:00 | 202.94 | 2024-09-10 12:15:00 | 192.55 | STOP_HIT | 0.50 | 5.12% |
| SELL | retest2 | 2024-09-05 11:30:00 | 202.75 | 2024-09-10 12:15:00 | 192.55 | STOP_HIT | 0.50 | 5.03% |
| SELL | retest2 | 2024-09-09 09:15:00 | 192.93 | 2024-09-13 15:15:00 | 191.80 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2024-09-24 12:30:00 | 182.01 | 2024-09-30 10:15:00 | 181.43 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2024-09-27 09:45:00 | 182.00 | 2024-09-30 10:15:00 | 181.43 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2024-10-04 12:30:00 | 177.20 | 2024-10-07 13:15:00 | 168.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-07 09:30:00 | 177.69 | 2024-10-07 13:15:00 | 168.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 12:30:00 | 177.20 | 2024-10-07 15:15:00 | 173.33 | STOP_HIT | 0.50 | 2.18% |
| SELL | retest2 | 2024-10-07 09:30:00 | 177.69 | 2024-10-07 15:15:00 | 173.33 | STOP_HIT | 0.50 | 2.45% |
| SELL | retest2 | 2024-10-24 09:15:00 | 151.03 | 2024-10-25 14:15:00 | 143.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-25 09:15:00 | 151.65 | 2024-10-25 14:15:00 | 144.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 09:15:00 | 151.03 | 2024-10-28 10:15:00 | 147.03 | STOP_HIT | 0.50 | 2.65% |
| SELL | retest2 | 2024-10-25 09:15:00 | 151.65 | 2024-10-28 10:15:00 | 147.03 | STOP_HIT | 0.50 | 3.05% |
| SELL | retest2 | 2024-11-14 12:30:00 | 150.54 | 2024-11-22 10:15:00 | 153.83 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-11-14 13:30:00 | 150.62 | 2024-11-22 10:15:00 | 153.83 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-11-19 12:15:00 | 150.85 | 2024-11-22 10:15:00 | 153.83 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-12-04 09:15:00 | 159.36 | 2024-12-05 09:15:00 | 154.93 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2024-12-06 14:15:00 | 154.80 | 2024-12-17 09:15:00 | 147.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-09 09:45:00 | 154.80 | 2024-12-17 09:15:00 | 147.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-09 11:30:00 | 154.78 | 2024-12-17 09:15:00 | 147.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-06 14:15:00 | 154.80 | 2024-12-17 10:15:00 | 148.66 | STOP_HIT | 0.50 | 3.97% |
| SELL | retest2 | 2024-12-09 09:45:00 | 154.80 | 2024-12-17 10:15:00 | 148.66 | STOP_HIT | 0.50 | 3.97% |
| SELL | retest2 | 2024-12-09 11:30:00 | 154.78 | 2024-12-17 10:15:00 | 148.66 | STOP_HIT | 0.50 | 3.95% |
| SELL | retest2 | 2024-12-09 12:30:00 | 154.55 | 2024-12-17 12:15:00 | 146.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 12:15:00 | 153.58 | 2024-12-18 09:15:00 | 145.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 15:00:00 | 151.80 | 2024-12-18 09:15:00 | 145.17 | PARTIAL | 0.50 | 4.37% |
| SELL | retest2 | 2024-12-09 12:30:00 | 154.55 | 2024-12-18 12:15:00 | 146.49 | STOP_HIT | 0.50 | 5.22% |
| SELL | retest2 | 2024-12-10 12:15:00 | 153.58 | 2024-12-18 12:15:00 | 146.49 | STOP_HIT | 0.50 | 4.62% |
| SELL | retest2 | 2024-12-10 15:00:00 | 151.80 | 2024-12-18 12:15:00 | 146.49 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2024-12-11 09:30:00 | 152.81 | 2024-12-19 09:15:00 | 144.21 | PARTIAL | 0.50 | 5.63% |
| SELL | retest2 | 2024-12-11 09:30:00 | 152.81 | 2024-12-19 11:15:00 | 147.75 | STOP_HIT | 0.50 | 3.31% |
| SELL | retest2 | 2025-01-15 13:15:00 | 137.35 | 2025-01-16 14:15:00 | 138.85 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-01-15 15:00:00 | 137.36 | 2025-01-16 14:15:00 | 138.85 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-01-24 12:45:00 | 133.67 | 2025-01-27 10:15:00 | 126.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 12:45:00 | 133.67 | 2025-01-29 09:15:00 | 125.73 | STOP_HIT | 0.50 | 5.94% |
| SELL | retest2 | 2025-02-27 09:15:00 | 112.75 | 2025-02-28 09:15:00 | 107.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 09:15:00 | 112.75 | 2025-03-03 09:15:00 | 101.48 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-11 11:15:00 | 114.20 | 2025-03-12 15:15:00 | 111.75 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-03-11 14:15:00 | 113.81 | 2025-03-12 15:15:00 | 111.75 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-03-12 09:15:00 | 113.82 | 2025-03-12 15:15:00 | 111.75 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest1 | 2025-03-17 13:30:00 | 118.10 | 2025-03-17 15:15:00 | 116.40 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest1 | 2025-03-17 14:45:00 | 117.90 | 2025-03-17 15:15:00 | 116.40 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-04-21 11:00:00 | 143.30 | 2025-04-25 09:15:00 | 136.25 | STOP_HIT | 1.00 | -4.92% |
| BUY | retest2 | 2025-04-21 14:15:00 | 141.48 | 2025-04-25 09:15:00 | 136.25 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2025-04-21 14:45:00 | 141.72 | 2025-04-25 09:15:00 | 136.25 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest2 | 2025-04-22 09:15:00 | 142.66 | 2025-04-25 09:15:00 | 136.25 | STOP_HIT | 1.00 | -4.49% |
| BUY | retest2 | 2025-04-23 11:15:00 | 140.18 | 2025-04-25 09:15:00 | 136.25 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-04-29 11:00:00 | 133.97 | 2025-05-06 14:15:00 | 128.23 | PARTIAL | 0.50 | 4.28% |
| SELL | retest2 | 2025-04-29 13:15:00 | 134.35 | 2025-05-06 14:15:00 | 128.37 | PARTIAL | 0.50 | 4.45% |
| SELL | retest2 | 2025-04-30 10:30:00 | 134.98 | 2025-05-06 14:15:00 | 128.15 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2025-04-30 12:15:00 | 135.13 | 2025-05-06 15:15:00 | 127.63 | PARTIAL | 0.50 | 5.55% |
| SELL | retest2 | 2025-05-05 11:15:00 | 134.89 | 2025-05-07 09:15:00 | 127.27 | PARTIAL | 0.50 | 5.65% |
| SELL | retest2 | 2025-04-29 11:00:00 | 133.97 | 2025-05-07 13:15:00 | 129.00 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2025-04-29 13:15:00 | 134.35 | 2025-05-07 13:15:00 | 129.00 | STOP_HIT | 0.50 | 3.98% |
| SELL | retest2 | 2025-04-30 10:30:00 | 134.98 | 2025-05-07 13:15:00 | 129.00 | STOP_HIT | 0.50 | 4.43% |
| SELL | retest2 | 2025-04-30 12:15:00 | 135.13 | 2025-05-07 13:15:00 | 129.00 | STOP_HIT | 0.50 | 4.54% |
| SELL | retest2 | 2025-05-05 11:15:00 | 134.89 | 2025-05-07 13:15:00 | 129.00 | STOP_HIT | 0.50 | 4.37% |
| BUY | retest2 | 2025-05-21 13:45:00 | 139.37 | 2025-05-22 12:15:00 | 137.69 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-05-22 10:15:00 | 139.17 | 2025-05-22 12:15:00 | 137.69 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-05-27 11:15:00 | 140.82 | 2025-05-30 15:15:00 | 143.82 | STOP_HIT | 1.00 | 2.13% |
| BUY | retest2 | 2025-05-27 12:30:00 | 140.85 | 2025-05-30 15:15:00 | 143.82 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2025-05-27 15:00:00 | 140.74 | 2025-05-30 15:15:00 | 143.82 | STOP_HIT | 1.00 | 2.19% |
| SELL | retest2 | 2025-06-03 11:15:00 | 141.90 | 2025-06-09 10:15:00 | 143.24 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-06-05 09:30:00 | 142.04 | 2025-06-09 10:15:00 | 143.24 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-06-11 09:15:00 | 147.35 | 2025-06-12 12:15:00 | 142.75 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-06-17 11:45:00 | 137.58 | 2025-06-23 09:15:00 | 138.76 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-06-17 15:15:00 | 137.29 | 2025-06-23 09:15:00 | 138.76 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-06-26 09:15:00 | 142.94 | 2025-07-07 15:15:00 | 145.70 | STOP_HIT | 1.00 | 1.93% |
| BUY | retest1 | 2025-07-18 09:15:00 | 148.15 | 2025-07-21 09:15:00 | 140.22 | STOP_HIT | 1.00 | -5.35% |
| BUY | retest1 | 2025-07-18 10:45:00 | 147.83 | 2025-07-21 09:15:00 | 140.22 | STOP_HIT | 1.00 | -5.15% |
| SELL | retest2 | 2025-08-05 11:00:00 | 126.23 | 2025-08-08 11:15:00 | 126.55 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-08-05 14:15:00 | 126.20 | 2025-08-08 11:15:00 | 126.55 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-08-06 09:30:00 | 124.88 | 2025-08-08 11:15:00 | 126.55 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-08-08 09:45:00 | 126.11 | 2025-08-08 11:15:00 | 126.55 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-08-18 12:00:00 | 123.05 | 2025-08-19 11:15:00 | 125.41 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-08-18 13:30:00 | 123.25 | 2025-08-19 11:15:00 | 125.41 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-08-18 14:45:00 | 123.12 | 2025-08-19 11:15:00 | 125.41 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-08-21 09:15:00 | 127.70 | 2025-08-22 14:15:00 | 125.24 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-08-22 10:00:00 | 125.60 | 2025-08-22 14:15:00 | 125.24 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-08-22 11:15:00 | 125.44 | 2025-08-22 14:15:00 | 125.24 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-08-28 09:15:00 | 122.43 | 2025-09-01 11:15:00 | 125.02 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-08-28 10:15:00 | 123.37 | 2025-09-01 11:15:00 | 125.02 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-09-25 11:45:00 | 128.35 | 2025-09-25 13:15:00 | 129.17 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-09-26 09:15:00 | 126.70 | 2025-09-26 10:15:00 | 129.45 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-10-01 09:15:00 | 134.00 | 2025-10-03 14:15:00 | 146.59 | TARGET_HIT | 1.00 | 9.39% |
| BUY | retest2 | 2025-10-01 15:00:00 | 133.26 | 2025-10-03 14:15:00 | 146.71 | TARGET_HIT | 1.00 | 10.09% |
| BUY | retest2 | 2025-10-03 12:15:00 | 133.37 | 2025-10-06 09:15:00 | 147.40 | TARGET_HIT | 1.00 | 10.52% |
| SELL | retest2 | 2025-10-15 15:00:00 | 141.77 | 2025-10-16 09:15:00 | 145.97 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-11-03 11:15:00 | 171.85 | 2025-11-10 10:15:00 | 169.13 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-11-03 13:30:00 | 169.50 | 2025-11-10 10:15:00 | 169.13 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-11-04 09:15:00 | 171.72 | 2025-11-10 10:15:00 | 169.13 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-11-06 13:15:00 | 169.48 | 2025-11-10 10:15:00 | 169.13 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-11-06 14:15:00 | 171.29 | 2025-11-10 10:15:00 | 169.13 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-11-07 10:45:00 | 171.25 | 2025-11-10 10:15:00 | 169.13 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-11-13 09:15:00 | 176.65 | 2025-11-14 09:15:00 | 172.17 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-11-13 10:00:00 | 176.24 | 2025-11-14 09:15:00 | 172.17 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-12-02 13:30:00 | 158.63 | 2025-12-08 12:15:00 | 150.80 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2025-12-02 15:15:00 | 158.74 | 2025-12-08 12:15:00 | 150.86 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2025-12-04 09:15:00 | 158.31 | 2025-12-08 13:15:00 | 150.70 | PARTIAL | 0.50 | 4.81% |
| SELL | retest2 | 2025-12-04 11:30:00 | 158.80 | 2025-12-08 13:15:00 | 150.39 | PARTIAL | 0.50 | 5.29% |
| SELL | retest2 | 2025-12-02 13:30:00 | 158.63 | 2025-12-09 11:15:00 | 151.46 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2025-12-02 15:15:00 | 158.74 | 2025-12-09 11:15:00 | 151.46 | STOP_HIT | 0.50 | 4.59% |
| SELL | retest2 | 2025-12-04 09:15:00 | 158.31 | 2025-12-09 11:15:00 | 151.46 | STOP_HIT | 0.50 | 4.33% |
| SELL | retest2 | 2025-12-04 11:30:00 | 158.80 | 2025-12-09 11:15:00 | 151.46 | STOP_HIT | 0.50 | 4.62% |
| SELL | retest2 | 2025-12-15 11:30:00 | 150.48 | 2025-12-22 09:15:00 | 148.98 | STOP_HIT | 1.00 | 1.00% |
| SELL | retest2 | 2025-12-16 10:15:00 | 150.00 | 2025-12-22 09:15:00 | 148.98 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest2 | 2025-12-26 13:30:00 | 145.49 | 2025-12-31 09:15:00 | 146.81 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-29 10:00:00 | 145.53 | 2025-12-31 09:15:00 | 146.81 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-12-29 11:00:00 | 145.60 | 2025-12-31 09:15:00 | 146.81 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2026-01-02 13:00:00 | 151.33 | 2026-01-05 13:15:00 | 149.31 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-01-02 15:15:00 | 151.40 | 2026-01-05 13:15:00 | 149.31 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-01-05 11:00:00 | 151.61 | 2026-01-05 13:15:00 | 149.31 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-01-07 12:45:00 | 147.15 | 2026-01-09 09:15:00 | 139.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 14:00:00 | 147.15 | 2026-01-09 09:15:00 | 139.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 15:00:00 | 146.77 | 2026-01-09 09:15:00 | 139.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 12:45:00 | 147.15 | 2026-01-12 12:15:00 | 139.89 | STOP_HIT | 0.50 | 4.93% |
| SELL | retest2 | 2026-01-07 14:00:00 | 147.15 | 2026-01-12 12:15:00 | 139.89 | STOP_HIT | 0.50 | 4.93% |
| SELL | retest2 | 2026-01-07 15:00:00 | 146.77 | 2026-01-12 12:15:00 | 139.89 | STOP_HIT | 0.50 | 4.69% |
| BUY | retest2 | 2026-01-27 10:00:00 | 156.16 | 2026-01-29 09:15:00 | 171.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-27 11:45:00 | 156.71 | 2026-01-29 09:15:00 | 171.66 | TARGET_HIT | 1.00 | 9.54% |
| BUY | retest2 | 2026-01-27 13:00:00 | 156.05 | 2026-01-29 09:15:00 | 171.63 | TARGET_HIT | 1.00 | 9.99% |
| BUY | retest2 | 2026-01-27 14:30:00 | 156.03 | 2026-01-29 10:15:00 | 172.38 | TARGET_HIT | 1.00 | 10.48% |
| BUY | retest2 | 2026-01-29 09:45:00 | 170.50 | 2026-02-01 14:15:00 | 165.12 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2026-02-06 14:30:00 | 184.12 | 2026-02-13 09:15:00 | 185.69 | STOP_HIT | 1.00 | 0.85% |
| SELL | retest2 | 2026-02-17 11:00:00 | 185.14 | 2026-02-17 12:15:00 | 191.33 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest2 | 2026-02-19 09:15:00 | 192.27 | 2026-02-20 15:15:00 | 191.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-03-11 10:15:00 | 191.52 | 2026-03-12 09:15:00 | 181.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:15:00 | 191.52 | 2026-03-12 10:15:00 | 189.01 | STOP_HIT | 0.50 | 1.31% |
| SELL | retest2 | 2026-03-12 12:15:00 | 192.19 | 2026-03-13 09:15:00 | 182.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 14:30:00 | 190.85 | 2026-03-13 10:15:00 | 181.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 12:15:00 | 192.19 | 2026-03-16 09:15:00 | 187.23 | STOP_HIT | 0.50 | 2.58% |
| SELL | retest2 | 2026-03-12 14:30:00 | 190.85 | 2026-03-16 09:15:00 | 187.23 | STOP_HIT | 0.50 | 1.90% |
| SELL | retest2 | 2026-03-20 15:00:00 | 191.76 | 2026-03-23 15:15:00 | 182.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 15:00:00 | 191.76 | 2026-03-24 15:15:00 | 181.65 | STOP_HIT | 0.50 | 5.27% |
| SELL | retest2 | 2026-03-30 13:45:00 | 182.09 | 2026-04-01 11:15:00 | 184.20 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-03-30 14:15:00 | 181.89 | 2026-04-01 11:15:00 | 184.20 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-04-17 09:15:00 | 175.84 | 2026-04-24 09:15:00 | 193.42 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-29 13:45:00 | 172.61 | 2026-05-04 09:15:00 | 163.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-29 14:45:00 | 173.38 | 2026-05-04 09:15:00 | 164.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-30 09:30:00 | 172.89 | 2026-05-04 09:15:00 | 164.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-29 13:45:00 | 172.61 | 2026-05-05 10:15:00 | 156.04 | TARGET_HIT | 0.50 | 9.60% |
| SELL | retest2 | 2026-04-29 14:45:00 | 173.38 | 2026-05-05 14:15:00 | 155.60 | TARGET_HIT | 0.50 | 10.25% |
| SELL | retest2 | 2026-04-30 09:30:00 | 172.89 | 2026-05-06 09:15:00 | 155.35 | TARGET_HIT | 0.50 | 10.15% |
