# EIH Ltd. (EIHOTEL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 336.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 226 |
| ALERT1 | 145 |
| ALERT2 | 143 |
| ALERT2_SKIP | 88 |
| ALERT3 | 385 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 203 |
| PARTIAL | 16 |
| TARGET_HIT | 13 |
| STOP_HIT | 190 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 219 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 82 / 137
- **Target hits / Stop hits / Partials:** 13 / 190 / 16
- **Avg / median % per leg:** 0.17% / -0.70%
- **Sum % (uncompounded):** 38.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 86 | 25 | 29.1% | 8 | 78 | 0 | 0.44% | 37.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 86 | 25 | 29.1% | 8 | 78 | 0 | 0.44% | 37.4% |
| SELL (all) | 133 | 57 | 42.9% | 5 | 112 | 16 | 0.00% | 0.6% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | 1.17% | 4.7% |
| SELL @ 3rd Alert (retest2) | 129 | 55 | 42.6% | 5 | 109 | 15 | -0.03% | -4.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | 1.17% | 4.7% |
| retest2 (combined) | 215 | 80 | 37.2% | 13 | 187 | 15 | 0.16% | 33.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-22 15:15:00 | 201.00 | 202.03 | 202.09 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 09:15:00 | 203.90 | 202.40 | 202.26 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 14:15:00 | 201.50 | 202.12 | 202.17 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 09:15:00 | 204.60 | 202.44 | 202.30 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 12:15:00 | 201.35 | 202.21 | 202.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 15:15:00 | 199.70 | 201.49 | 201.89 | Break + close below crossover candle low |

### Cycle 6 — BUY (started 2023-05-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 09:15:00 | 210.10 | 203.21 | 202.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 10:15:00 | 216.45 | 209.94 | 208.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 13:15:00 | 213.85 | 214.82 | 212.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 13:15:00 | 213.85 | 214.82 | 212.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 13:15:00 | 213.85 | 214.82 | 212.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 14:00:00 | 213.85 | 214.82 | 212.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 09:15:00 | 217.90 | 216.85 | 215.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 10:00:00 | 217.90 | 216.85 | 215.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 13:15:00 | 215.80 | 216.63 | 215.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 13:30:00 | 215.85 | 216.63 | 215.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 14:15:00 | 214.95 | 216.30 | 215.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 15:00:00 | 214.95 | 216.30 | 215.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 15:15:00 | 215.25 | 216.09 | 215.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-07 09:15:00 | 216.60 | 216.09 | 215.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-07 13:30:00 | 215.60 | 215.65 | 215.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-07 14:15:00 | 214.15 | 215.35 | 215.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2023-06-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-07 14:15:00 | 214.15 | 215.35 | 215.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-07 15:15:00 | 212.90 | 214.86 | 215.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 09:15:00 | 210.05 | 209.28 | 210.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 09:15:00 | 210.05 | 209.28 | 210.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 210.05 | 209.28 | 210.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 09:45:00 | 209.90 | 209.28 | 210.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 13:15:00 | 210.75 | 209.57 | 210.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 14:00:00 | 210.75 | 209.57 | 210.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 14:15:00 | 210.15 | 209.69 | 210.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 14:30:00 | 211.20 | 209.69 | 210.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 15:15:00 | 211.00 | 209.95 | 210.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 09:15:00 | 211.90 | 209.95 | 210.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 211.30 | 210.22 | 210.59 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 11:15:00 | 213.00 | 211.05 | 210.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-15 09:15:00 | 214.80 | 212.91 | 212.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 13:15:00 | 212.65 | 213.22 | 212.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-15 13:15:00 | 212.65 | 213.22 | 212.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 13:15:00 | 212.65 | 213.22 | 212.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 13:30:00 | 213.00 | 213.22 | 212.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 14:15:00 | 213.05 | 213.19 | 212.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 14:45:00 | 212.50 | 213.19 | 212.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 15:15:00 | 212.00 | 212.95 | 212.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 09:15:00 | 213.65 | 212.95 | 212.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-16 13:15:00 | 211.10 | 212.29 | 212.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2023-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-16 13:15:00 | 211.10 | 212.29 | 212.43 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-19 09:15:00 | 215.70 | 213.15 | 212.80 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 09:15:00 | 211.55 | 212.67 | 212.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-20 12:15:00 | 210.35 | 211.86 | 212.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 14:15:00 | 212.85 | 211.96 | 212.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 14:15:00 | 212.85 | 211.96 | 212.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 14:15:00 | 212.85 | 211.96 | 212.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 15:00:00 | 212.85 | 211.96 | 212.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 15:15:00 | 211.00 | 211.77 | 212.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:15:00 | 215.70 | 211.77 | 212.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2023-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 09:15:00 | 216.10 | 212.64 | 212.54 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-06-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 11:15:00 | 210.60 | 212.71 | 212.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 12:15:00 | 209.45 | 212.06 | 212.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 09:15:00 | 211.00 | 209.10 | 210.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 09:15:00 | 211.00 | 209.10 | 210.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 211.00 | 209.10 | 210.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 10:00:00 | 211.00 | 209.10 | 210.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 211.00 | 209.48 | 210.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 13:00:00 | 209.75 | 209.74 | 210.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 14:15:00 | 210.00 | 209.84 | 210.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-27 09:15:00 | 211.15 | 210.42 | 210.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2023-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 09:15:00 | 211.15 | 210.42 | 210.40 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 11:15:00 | 208.00 | 209.98 | 210.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-28 13:15:00 | 207.30 | 208.33 | 209.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-30 09:15:00 | 212.85 | 209.14 | 209.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 09:15:00 | 212.85 | 209.14 | 209.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 09:15:00 | 212.85 | 209.14 | 209.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-30 09:45:00 | 213.10 | 209.14 | 209.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2023-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 10:15:00 | 211.90 | 209.69 | 209.46 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 09:15:00 | 209.65 | 210.01 | 210.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-06 15:15:00 | 207.00 | 208.32 | 208.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-07 15:15:00 | 207.90 | 206.61 | 207.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 15:15:00 | 207.90 | 206.61 | 207.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 15:15:00 | 207.90 | 206.61 | 207.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:15:00 | 207.50 | 206.61 | 207.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 207.55 | 206.80 | 207.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:45:00 | 207.65 | 206.80 | 207.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 206.75 | 206.79 | 207.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 10:30:00 | 208.00 | 206.79 | 207.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 11:15:00 | 205.75 | 206.58 | 207.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-10 15:15:00 | 205.05 | 206.16 | 206.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-11 09:15:00 | 207.90 | 206.33 | 206.82 | SL hit (close>static) qty=1.00 sl=207.35 alert=retest2 |

### Cycle 18 — BUY (started 2023-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 12:15:00 | 211.45 | 207.84 | 207.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 09:15:00 | 219.15 | 211.66 | 209.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 13:15:00 | 213.50 | 214.43 | 211.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-12 13:45:00 | 213.65 | 214.43 | 211.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 10:15:00 | 215.25 | 214.42 | 212.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 10:30:00 | 213.10 | 214.42 | 212.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 13:15:00 | 210.55 | 213.53 | 212.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 14:00:00 | 210.55 | 213.53 | 212.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 14:15:00 | 211.65 | 213.15 | 212.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 14:45:00 | 209.80 | 213.15 | 212.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 11:15:00 | 213.00 | 213.30 | 212.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-14 12:00:00 | 213.00 | 213.30 | 212.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 12:15:00 | 213.00 | 213.24 | 212.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-14 12:45:00 | 212.70 | 213.24 | 212.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 13:15:00 | 212.05 | 213.00 | 212.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-14 13:30:00 | 211.70 | 213.00 | 212.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 14:15:00 | 212.75 | 212.95 | 212.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-14 14:30:00 | 211.90 | 212.95 | 212.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 15:15:00 | 212.10 | 212.78 | 212.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-17 09:15:00 | 215.50 | 212.78 | 212.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 14:00:00 | 213.70 | 214.36 | 214.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-18 14:15:00 | 214.15 | 214.32 | 214.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2023-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 14:15:00 | 214.15 | 214.32 | 214.33 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 15:15:00 | 214.40 | 214.33 | 214.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 13:15:00 | 215.15 | 214.55 | 214.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 11:15:00 | 217.75 | 219.01 | 217.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 11:15:00 | 217.75 | 219.01 | 217.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 11:15:00 | 217.75 | 219.01 | 217.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 11:30:00 | 217.90 | 219.01 | 217.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 12:15:00 | 218.45 | 218.90 | 217.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 12:30:00 | 217.85 | 218.90 | 217.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 218.00 | 218.58 | 217.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 09:45:00 | 217.50 | 218.58 | 217.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 10:15:00 | 217.70 | 218.40 | 217.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 10:45:00 | 217.80 | 218.40 | 217.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 11:15:00 | 216.00 | 217.92 | 217.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 11:45:00 | 215.35 | 217.92 | 217.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 12:15:00 | 218.00 | 217.94 | 217.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-24 15:15:00 | 218.80 | 217.81 | 217.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-25 10:15:00 | 215.85 | 217.53 | 217.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 10:15:00 | 215.85 | 217.53 | 217.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-25 13:15:00 | 214.70 | 216.45 | 217.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-27 09:15:00 | 215.00 | 214.41 | 215.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 09:15:00 | 215.00 | 214.41 | 215.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 215.00 | 214.41 | 215.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 09:30:00 | 216.45 | 214.41 | 215.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 10:15:00 | 214.95 | 214.52 | 215.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 10:45:00 | 215.65 | 214.52 | 215.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 14:15:00 | 212.30 | 213.99 | 214.80 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 10:15:00 | 216.80 | 214.25 | 214.09 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-31 12:15:00 | 212.10 | 213.93 | 213.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-31 13:15:00 | 210.60 | 213.26 | 213.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 10:15:00 | 209.45 | 208.62 | 209.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-03 10:15:00 | 209.45 | 208.62 | 209.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 10:15:00 | 209.45 | 208.62 | 209.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 10:45:00 | 210.15 | 208.62 | 209.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 11:15:00 | 208.85 | 208.66 | 209.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 11:45:00 | 210.25 | 208.66 | 209.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 14:15:00 | 211.00 | 209.06 | 209.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 15:00:00 | 211.00 | 209.06 | 209.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 15:15:00 | 210.45 | 209.34 | 209.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 09:15:00 | 211.10 | 209.34 | 209.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2023-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 09:15:00 | 212.35 | 209.94 | 209.91 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-04 14:15:00 | 209.55 | 209.89 | 209.93 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 09:15:00 | 213.60 | 210.43 | 210.16 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 10:15:00 | 209.00 | 210.41 | 210.49 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 11:15:00 | 211.35 | 210.60 | 210.57 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 15:15:00 | 209.50 | 210.44 | 210.53 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 09:15:00 | 213.00 | 210.95 | 210.76 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-08-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 11:15:00 | 208.50 | 210.89 | 211.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 12:15:00 | 207.95 | 209.28 | 210.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-11 13:15:00 | 210.00 | 209.42 | 210.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 13:15:00 | 210.00 | 209.42 | 210.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 13:15:00 | 210.00 | 209.42 | 210.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 14:00:00 | 210.00 | 209.42 | 210.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 14:15:00 | 210.05 | 209.55 | 210.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 15:00:00 | 210.05 | 209.55 | 210.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 15:15:00 | 210.10 | 209.66 | 210.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 09:15:00 | 207.60 | 209.66 | 210.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 207.75 | 209.28 | 209.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-14 10:45:00 | 206.25 | 208.81 | 209.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-14 11:45:00 | 206.35 | 208.28 | 209.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-14 15:15:00 | 206.40 | 207.31 | 208.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-16 14:15:00 | 211.20 | 208.33 | 208.34 | SL hit (close>static) qty=1.00 sl=210.40 alert=retest2 |

### Cycle 32 — BUY (started 2023-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 15:15:00 | 211.40 | 208.94 | 208.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 10:15:00 | 212.45 | 210.04 | 209.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 09:15:00 | 238.10 | 239.43 | 236.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-24 09:30:00 | 239.35 | 239.43 | 236.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 10:15:00 | 237.10 | 238.96 | 236.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 10:45:00 | 237.40 | 238.96 | 236.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 11:15:00 | 237.15 | 238.60 | 236.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 14:00:00 | 237.80 | 238.23 | 236.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 15:00:00 | 237.80 | 238.15 | 236.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-25 11:45:00 | 238.35 | 238.33 | 237.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-28 09:15:00 | 235.80 | 236.78 | 236.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-28 09:15:00 | 235.80 | 236.78 | 236.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-28 14:15:00 | 232.40 | 234.88 | 235.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 10:15:00 | 233.95 | 233.86 | 235.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-29 10:30:00 | 234.70 | 233.86 | 235.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 11:15:00 | 240.95 | 234.00 | 234.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 12:00:00 | 240.95 | 234.00 | 234.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2023-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 12:15:00 | 242.90 | 235.78 | 234.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 09:15:00 | 245.50 | 240.18 | 237.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-04 09:15:00 | 259.15 | 262.23 | 254.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-04 12:15:00 | 255.15 | 259.20 | 254.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 12:15:00 | 255.15 | 259.20 | 254.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 13:00:00 | 255.15 | 259.20 | 254.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 13:15:00 | 254.00 | 258.16 | 254.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 14:15:00 | 252.50 | 258.16 | 254.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 14:15:00 | 252.50 | 257.03 | 254.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 14:30:00 | 251.65 | 257.03 | 254.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 10:15:00 | 253.20 | 254.93 | 254.06 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-05 12:15:00 | 249.40 | 253.43 | 253.51 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 10:15:00 | 257.45 | 253.70 | 253.41 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-09-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 11:15:00 | 250.75 | 253.84 | 253.95 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 10:15:00 | 254.60 | 253.87 | 253.83 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 11:15:00 | 250.95 | 253.29 | 253.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 09:15:00 | 241.55 | 250.14 | 251.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 245.30 | 241.84 | 243.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 245.30 | 241.84 | 243.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 245.30 | 241.84 | 243.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:30:00 | 247.00 | 241.84 | 243.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 10:15:00 | 245.00 | 242.47 | 244.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 10:30:00 | 244.50 | 242.47 | 244.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 14:15:00 | 245.15 | 243.40 | 244.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 15:00:00 | 245.15 | 243.40 | 244.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 15:15:00 | 245.00 | 243.72 | 244.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-15 09:15:00 | 246.65 | 243.72 | 244.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 09:15:00 | 245.00 | 243.98 | 244.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 11:00:00 | 243.30 | 243.84 | 244.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 11:45:00 | 243.50 | 243.85 | 244.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 12:15:00 | 242.45 | 243.85 | 244.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 09:15:00 | 231.13 | 234.34 | 237.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 09:15:00 | 231.32 | 234.34 | 237.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 09:15:00 | 230.33 | 234.34 | 237.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-09-21 09:15:00 | 239.65 | 232.13 | 234.33 | SL hit (close>ema200) qty=0.50 sl=232.13 alert=retest2 |

### Cycle 40 — BUY (started 2023-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 10:15:00 | 232.75 | 226.59 | 225.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-04 09:15:00 | 235.40 | 231.07 | 228.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 15:15:00 | 232.50 | 232.97 | 230.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 15:15:00 | 232.50 | 232.97 | 230.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 15:15:00 | 232.50 | 232.97 | 230.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-06 09:15:00 | 235.20 | 232.48 | 231.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-09 12:15:00 | 229.55 | 233.55 | 233.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2023-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 12:15:00 | 229.55 | 233.55 | 233.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 14:15:00 | 229.00 | 232.17 | 232.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 233.50 | 231.85 | 232.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 233.50 | 231.85 | 232.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 233.50 | 231.85 | 232.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 09:45:00 | 233.90 | 231.85 | 232.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 232.90 | 232.06 | 232.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 10:30:00 | 232.65 | 232.06 | 232.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 11:15:00 | 233.10 | 232.27 | 232.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 11:45:00 | 233.40 | 232.27 | 232.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 12:15:00 | 231.35 | 232.08 | 232.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 10:45:00 | 230.55 | 232.07 | 232.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 13:30:00 | 231.05 | 231.81 | 232.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-12 11:15:00 | 231.20 | 231.69 | 231.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-13 09:15:00 | 232.30 | 232.13 | 232.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2023-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-13 09:15:00 | 232.30 | 232.13 | 232.13 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 10:15:00 | 232.00 | 232.11 | 232.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 11:15:00 | 231.50 | 231.99 | 232.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 231.05 | 229.83 | 230.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 09:15:00 | 231.05 | 229.83 | 230.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 231.05 | 229.83 | 230.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 09:30:00 | 232.00 | 229.83 | 230.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 10:15:00 | 231.50 | 230.17 | 230.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 10:45:00 | 232.25 | 230.17 | 230.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 13:15:00 | 228.20 | 229.90 | 230.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 12:15:00 | 227.70 | 229.01 | 229.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 15:15:00 | 228.00 | 228.52 | 229.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 09:15:00 | 216.60 | 222.53 | 224.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 10:15:00 | 216.31 | 221.72 | 224.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-26 14:15:00 | 213.00 | 212.51 | 214.84 | SL hit (close>ema200) qty=0.50 sl=212.51 alert=retest2 |

### Cycle 44 — BUY (started 2023-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 10:15:00 | 220.05 | 214.78 | 214.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 11:15:00 | 224.65 | 216.75 | 215.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 14:15:00 | 225.10 | 225.64 | 222.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-31 15:00:00 | 225.10 | 225.64 | 222.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 224.60 | 225.19 | 222.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-01 15:00:00 | 226.05 | 224.89 | 223.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 13:00:00 | 225.60 | 225.97 | 224.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-03 13:15:00 | 222.90 | 224.26 | 224.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2023-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 13:15:00 | 222.90 | 224.26 | 224.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 15:15:00 | 222.50 | 223.67 | 224.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 09:15:00 | 225.00 | 223.94 | 224.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 09:15:00 | 225.00 | 223.94 | 224.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 225.00 | 223.94 | 224.13 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2023-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 10:15:00 | 230.15 | 225.18 | 224.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 11:15:00 | 232.85 | 226.72 | 225.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 09:15:00 | 238.90 | 239.72 | 236.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-09 09:45:00 | 238.75 | 239.72 | 236.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 236.65 | 239.11 | 236.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 11:00:00 | 236.65 | 239.11 | 236.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 11:15:00 | 235.60 | 238.40 | 236.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 12:00:00 | 235.60 | 238.40 | 236.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 12:15:00 | 235.25 | 237.77 | 236.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 13:15:00 | 234.85 | 237.77 | 236.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2023-11-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 15:15:00 | 233.35 | 235.69 | 235.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 10:15:00 | 231.85 | 234.73 | 235.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-12 18:15:00 | 234.90 | 233.05 | 234.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-12 18:15:00 | 234.90 | 233.05 | 234.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 234.90 | 233.05 | 234.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-12 18:45:00 | 234.75 | 233.05 | 234.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 09:15:00 | 232.45 | 232.93 | 233.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 10:30:00 | 231.30 | 232.60 | 233.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 11:00:00 | 231.25 | 232.60 | 233.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 13:00:00 | 231.00 | 232.17 | 233.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 14:30:00 | 230.00 | 231.87 | 232.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 236.50 | 232.64 | 233.09 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-11-15 09:15:00 | 236.50 | 232.64 | 233.09 | SL hit (close>static) qty=1.00 sl=235.65 alert=retest2 |

### Cycle 48 — BUY (started 2023-11-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 11:15:00 | 234.90 | 233.42 | 233.38 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2023-11-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 12:15:00 | 232.00 | 233.51 | 233.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-16 14:15:00 | 231.05 | 232.86 | 233.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 15:15:00 | 233.00 | 232.88 | 233.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 15:15:00 | 233.00 | 232.88 | 233.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 15:15:00 | 233.00 | 232.88 | 233.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-17 10:45:00 | 231.00 | 232.42 | 232.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-17 11:30:00 | 230.85 | 232.04 | 232.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-17 14:45:00 | 230.75 | 231.64 | 232.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-20 09:15:00 | 245.80 | 234.53 | 233.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2023-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 09:15:00 | 245.80 | 234.53 | 233.54 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 11:15:00 | 235.10 | 238.24 | 238.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 09:15:00 | 234.00 | 235.59 | 236.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 10:15:00 | 240.00 | 236.47 | 236.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 10:15:00 | 240.00 | 236.47 | 236.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 240.00 | 236.47 | 236.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 11:00:00 | 240.00 | 236.47 | 236.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2023-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 11:15:00 | 241.00 | 237.38 | 237.17 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2023-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-29 11:15:00 | 235.35 | 237.30 | 237.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-29 13:15:00 | 235.05 | 236.53 | 237.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 09:15:00 | 236.25 | 236.19 | 236.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 09:15:00 | 236.25 | 236.19 | 236.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 236.25 | 236.19 | 236.72 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2023-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 09:15:00 | 239.15 | 236.70 | 236.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 12:15:00 | 242.60 | 238.49 | 237.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 15:15:00 | 238.50 | 238.68 | 237.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-04 15:15:00 | 238.50 | 238.68 | 237.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 15:15:00 | 238.50 | 238.68 | 237.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 09:15:00 | 242.70 | 238.68 | 237.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 12:45:00 | 239.45 | 239.28 | 238.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 13:15:00 | 239.60 | 239.28 | 238.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-06 09:45:00 | 239.40 | 239.60 | 238.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 11:15:00 | 239.05 | 239.44 | 238.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 11:45:00 | 238.85 | 239.44 | 238.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 12:15:00 | 238.70 | 239.29 | 238.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 13:00:00 | 238.70 | 239.29 | 238.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 13:15:00 | 239.00 | 239.24 | 238.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-06 15:00:00 | 239.50 | 239.29 | 239.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 09:30:00 | 240.25 | 240.81 | 239.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-12 14:15:00 | 244.70 | 246.64 | 246.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2023-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 14:15:00 | 244.70 | 246.64 | 246.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 09:15:00 | 242.30 | 245.46 | 246.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 09:15:00 | 242.80 | 242.43 | 243.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 242.80 | 242.43 | 243.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 242.80 | 242.43 | 243.89 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2023-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 09:15:00 | 244.60 | 243.50 | 243.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-18 11:15:00 | 247.65 | 244.57 | 243.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 10:15:00 | 245.45 | 245.52 | 244.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-19 11:00:00 | 245.45 | 245.52 | 244.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 13:15:00 | 244.70 | 245.36 | 244.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 13:45:00 | 244.50 | 245.36 | 244.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 14:15:00 | 242.80 | 244.85 | 244.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 15:00:00 | 242.80 | 244.85 | 244.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 15:15:00 | 244.30 | 244.74 | 244.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 09:15:00 | 243.40 | 244.74 | 244.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 244.50 | 244.77 | 244.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 10:45:00 | 244.00 | 244.77 | 244.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2023-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 11:15:00 | 242.85 | 244.39 | 244.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 12:15:00 | 239.25 | 243.36 | 244.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 234.60 | 233.61 | 236.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 15:00:00 | 234.60 | 233.61 | 236.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 235.10 | 234.26 | 236.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 11:15:00 | 237.75 | 234.26 | 236.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 11:15:00 | 237.00 | 234.81 | 236.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 12:45:00 | 235.00 | 235.04 | 236.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-26 10:15:00 | 239.40 | 237.28 | 237.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2023-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 10:15:00 | 239.40 | 237.28 | 237.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 11:15:00 | 241.45 | 238.11 | 237.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 15:15:00 | 277.55 | 277.75 | 273.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-08 09:15:00 | 274.75 | 277.75 | 273.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 274.20 | 277.04 | 273.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 09:30:00 | 275.00 | 277.04 | 273.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 10:15:00 | 274.75 | 276.58 | 273.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:00:00 | 274.75 | 276.58 | 273.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 12:15:00 | 273.80 | 275.77 | 273.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 13:00:00 | 273.80 | 275.77 | 273.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 13:15:00 | 272.50 | 275.12 | 273.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 13:30:00 | 272.35 | 275.12 | 273.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 14:15:00 | 271.50 | 274.39 | 273.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 15:00:00 | 271.50 | 274.39 | 273.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 14:15:00 | 275.10 | 275.65 | 274.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 14:30:00 | 276.10 | 275.65 | 274.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 15:15:00 | 275.80 | 275.68 | 274.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 09:15:00 | 275.50 | 275.68 | 274.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 278.60 | 276.26 | 275.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 10:30:00 | 281.00 | 276.71 | 275.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 09:45:00 | 281.00 | 276.74 | 275.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 11:45:00 | 281.90 | 277.91 | 276.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 10:00:00 | 280.80 | 279.08 | 277.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 14:15:00 | 277.75 | 279.68 | 278.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-12 15:00:00 | 277.75 | 279.68 | 278.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 15:15:00 | 276.55 | 279.05 | 278.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 09:15:00 | 280.40 | 279.05 | 278.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 11:30:00 | 281.00 | 279.76 | 278.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 278.80 | 284.04 | 284.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 09:15:00 | 278.80 | 284.04 | 284.64 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 09:15:00 | 287.60 | 284.78 | 284.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 15:15:00 | 292.50 | 286.62 | 285.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 13:15:00 | 295.95 | 297.19 | 293.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-23 14:00:00 | 295.95 | 297.19 | 293.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 13:15:00 | 319.60 | 317.85 | 314.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-30 15:15:00 | 323.10 | 317.80 | 314.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-06 12:15:00 | 355.41 | 347.18 | 340.52 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2024-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 13:15:00 | 392.90 | 397.38 | 397.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-13 11:15:00 | 388.30 | 393.54 | 395.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 09:15:00 | 397.20 | 391.90 | 393.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 09:15:00 | 397.20 | 391.90 | 393.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 397.20 | 391.90 | 393.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 10:00:00 | 397.20 | 391.90 | 393.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 394.95 | 392.51 | 393.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 12:15:00 | 392.00 | 392.81 | 393.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 14:15:00 | 392.45 | 393.12 | 393.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-15 09:15:00 | 403.80 | 394.56 | 394.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 403.80 | 394.56 | 394.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 12:15:00 | 408.90 | 399.38 | 396.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 11:15:00 | 403.90 | 405.81 | 401.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-16 11:45:00 | 402.20 | 405.81 | 401.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 12:15:00 | 401.35 | 404.92 | 401.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 12:30:00 | 403.00 | 404.92 | 401.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 13:15:00 | 399.60 | 403.85 | 401.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 13:30:00 | 400.00 | 403.85 | 401.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 14:15:00 | 394.00 | 401.88 | 400.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 15:00:00 | 394.00 | 401.88 | 400.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 396.00 | 400.73 | 400.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 10:00:00 | 396.00 | 400.73 | 400.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 10:15:00 | 399.95 | 400.57 | 400.47 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 12:15:00 | 397.75 | 399.90 | 400.18 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 13:15:00 | 407.00 | 401.32 | 400.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-20 09:15:00 | 430.00 | 407.93 | 404.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 12:15:00 | 409.80 | 412.03 | 407.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-20 12:45:00 | 409.85 | 412.03 | 407.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 13:15:00 | 408.50 | 411.32 | 407.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 10:30:00 | 416.60 | 408.93 | 407.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 12:45:00 | 414.50 | 409.60 | 407.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 13:15:00 | 402.35 | 408.15 | 407.24 | SL hit (close<static) qty=1.00 sl=405.60 alert=retest2 |

### Cycle 65 — SELL (started 2024-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 14:15:00 | 400.10 | 406.54 | 406.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 15:15:00 | 397.15 | 402.00 | 403.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 09:15:00 | 403.00 | 402.20 | 403.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 09:15:00 | 403.00 | 402.20 | 403.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 403.00 | 402.20 | 403.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:30:00 | 403.00 | 402.20 | 403.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 14:15:00 | 405.35 | 401.78 | 402.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 14:45:00 | 405.80 | 401.78 | 402.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 15:15:00 | 407.10 | 402.85 | 403.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 09:15:00 | 409.85 | 402.85 | 403.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2024-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 09:15:00 | 410.60 | 404.40 | 403.95 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 12:15:00 | 400.10 | 403.38 | 403.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 13:15:00 | 399.60 | 402.62 | 403.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-26 14:15:00 | 405.10 | 403.12 | 403.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 14:15:00 | 405.10 | 403.12 | 403.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 14:15:00 | 405.10 | 403.12 | 403.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 15:00:00 | 405.10 | 403.12 | 403.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 15:15:00 | 401.45 | 402.78 | 403.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 12:00:00 | 400.85 | 402.65 | 403.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 12:45:00 | 399.75 | 402.11 | 402.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 13:30:00 | 400.30 | 401.79 | 402.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 14:45:00 | 400.70 | 401.47 | 402.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 13:15:00 | 398.50 | 397.10 | 399.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 13:30:00 | 399.20 | 397.10 | 399.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 14:15:00 | 393.25 | 396.33 | 398.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-29 09:15:00 | 389.30 | 395.67 | 398.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-29 11:15:00 | 402.20 | 396.62 | 398.05 | SL hit (close>static) qty=1.00 sl=400.20 alert=retest2 |

### Cycle 68 — BUY (started 2024-02-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 15:15:00 | 404.65 | 399.73 | 399.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 13:15:00 | 407.80 | 403.96 | 401.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-02 11:15:00 | 404.90 | 404.96 | 402.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-02 11:45:00 | 405.00 | 404.96 | 402.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 12:15:00 | 404.00 | 404.77 | 403.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-04 09:15:00 | 423.80 | 404.77 | 403.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-11 11:15:00 | 415.60 | 420.95 | 421.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 11:15:00 | 415.60 | 420.95 | 421.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 10:15:00 | 406.00 | 413.82 | 417.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 15:15:00 | 391.20 | 390.47 | 399.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 09:15:00 | 400.10 | 392.40 | 399.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 400.10 | 392.40 | 399.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 10:00:00 | 400.10 | 392.40 | 399.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 407.65 | 395.45 | 400.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 11:00:00 | 407.65 | 395.45 | 400.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 411.00 | 398.56 | 401.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 12:00:00 | 411.00 | 398.56 | 401.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2024-03-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 14:15:00 | 422.60 | 406.47 | 404.51 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 15:15:00 | 404.00 | 407.03 | 407.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 09:15:00 | 403.00 | 406.22 | 406.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 10:15:00 | 403.50 | 401.43 | 403.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 10:15:00 | 403.50 | 401.43 | 403.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 10:15:00 | 403.50 | 401.43 | 403.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 11:00:00 | 403.50 | 401.43 | 403.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 11:15:00 | 403.10 | 401.76 | 403.37 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 13:15:00 | 410.55 | 404.78 | 404.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-20 14:15:00 | 417.10 | 407.25 | 405.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 14:15:00 | 439.60 | 443.82 | 436.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 14:15:00 | 439.60 | 443.82 | 436.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 439.60 | 443.82 | 436.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 15:00:00 | 439.60 | 443.82 | 436.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 436.80 | 442.42 | 436.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 09:15:00 | 455.10 | 442.42 | 436.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 15:15:00 | 455.00 | 451.61 | 444.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-10 09:15:00 | 500.61 | 486.05 | 481.77 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-04-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 14:15:00 | 474.75 | 483.70 | 483.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 15:15:00 | 472.00 | 481.36 | 482.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 473.40 | 468.94 | 474.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 473.40 | 468.94 | 474.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 473.40 | 468.94 | 474.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:45:00 | 472.85 | 468.94 | 474.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 459.90 | 466.80 | 470.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 14:30:00 | 457.75 | 460.45 | 465.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 09:30:00 | 454.60 | 453.21 | 457.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 13:15:00 | 469.85 | 461.44 | 460.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2024-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 13:15:00 | 469.85 | 461.44 | 460.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 14:15:00 | 477.60 | 464.67 | 462.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 10:15:00 | 465.00 | 467.33 | 464.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 10:15:00 | 465.00 | 467.33 | 464.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 10:15:00 | 465.00 | 467.33 | 464.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 10:45:00 | 465.95 | 467.33 | 464.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 13:15:00 | 470.00 | 467.19 | 464.80 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 14:15:00 | 454.00 | 463.14 | 463.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-25 11:15:00 | 450.00 | 459.51 | 461.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 15:15:00 | 464.05 | 458.71 | 460.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 15:15:00 | 464.05 | 458.71 | 460.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 15:15:00 | 464.05 | 458.71 | 460.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-26 09:15:00 | 468.60 | 458.71 | 460.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 470.60 | 461.09 | 461.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-26 09:45:00 | 470.10 | 461.09 | 461.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-04-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 10:15:00 | 475.90 | 464.05 | 462.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 11:15:00 | 479.40 | 467.12 | 464.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-29 14:15:00 | 474.30 | 475.12 | 471.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-29 14:45:00 | 474.15 | 475.12 | 471.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 478.95 | 476.70 | 474.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:15:00 | 472.90 | 476.70 | 474.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 478.10 | 476.98 | 474.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:30:00 | 473.45 | 476.98 | 474.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 11:15:00 | 466.55 | 474.92 | 474.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 12:00:00 | 466.55 | 474.92 | 474.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 12:15:00 | 471.95 | 474.33 | 474.02 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2024-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 13:15:00 | 470.65 | 473.59 | 473.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 14:15:00 | 467.35 | 472.34 | 473.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 10:15:00 | 473.10 | 471.30 | 472.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 10:15:00 | 473.10 | 471.30 | 472.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 10:15:00 | 473.10 | 471.30 | 472.35 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2024-05-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 14:15:00 | 476.60 | 472.75 | 472.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-06 09:15:00 | 484.90 | 475.88 | 474.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-06 15:15:00 | 481.00 | 481.74 | 478.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-07 09:15:00 | 476.45 | 481.74 | 478.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 480.20 | 481.43 | 478.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-07 10:45:00 | 486.35 | 482.95 | 479.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 09:15:00 | 494.30 | 489.04 | 486.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 13:00:00 | 485.45 | 489.53 | 488.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-09 14:15:00 | 481.45 | 487.14 | 487.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-05-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 14:15:00 | 481.45 | 487.14 | 487.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 15:15:00 | 472.00 | 484.11 | 485.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 10:15:00 | 486.60 | 484.19 | 485.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 10:15:00 | 486.60 | 484.19 | 485.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 486.60 | 484.19 | 485.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 11:00:00 | 486.60 | 484.19 | 485.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 485.95 | 484.54 | 485.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 12:15:00 | 484.10 | 484.54 | 485.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 09:15:00 | 482.80 | 484.63 | 485.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-15 09:30:00 | 483.05 | 476.88 | 478.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 09:15:00 | 482.05 | 479.76 | 479.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 09:15:00 | 482.05 | 479.76 | 479.68 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 12:15:00 | 479.20 | 480.24 | 480.38 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-05-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 14:15:00 | 481.45 | 480.54 | 480.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 09:15:00 | 487.95 | 482.49 | 481.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 13:15:00 | 482.05 | 483.55 | 482.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 13:15:00 | 482.05 | 483.55 | 482.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 482.05 | 483.55 | 482.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:00:00 | 482.05 | 483.55 | 482.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 480.25 | 482.89 | 482.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 15:00:00 | 480.25 | 482.89 | 482.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 479.70 | 482.25 | 482.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 489.35 | 482.25 | 482.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 09:15:00 | 484.15 | 483.19 | 483.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 10:15:00 | 476.35 | 481.75 | 482.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 10:15:00 | 476.35 | 481.75 | 482.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 12:15:00 | 474.20 | 479.29 | 481.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 15:15:00 | 480.15 | 479.15 | 480.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 15:15:00 | 480.15 | 479.15 | 480.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 480.15 | 479.15 | 480.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 10:45:00 | 476.90 | 478.30 | 479.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 13:00:00 | 471.75 | 476.86 | 479.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 09:15:00 | 453.05 | 469.62 | 474.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 09:15:00 | 448.16 | 469.62 | 474.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-05-28 14:15:00 | 429.21 | 437.28 | 449.19 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 84 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 424.75 | 413.98 | 413.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 436.60 | 422.86 | 418.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 433.05 | 436.31 | 430.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 15:00:00 | 433.05 | 436.31 | 430.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 439.20 | 436.89 | 431.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:30:00 | 429.75 | 435.27 | 431.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 428.05 | 433.83 | 431.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 10:30:00 | 428.60 | 433.83 | 431.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 09:15:00 | 424.50 | 429.50 | 429.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 10:15:00 | 423.80 | 428.36 | 429.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-13 14:15:00 | 427.00 | 423.56 | 425.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-13 14:15:00 | 427.00 | 423.56 | 425.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 427.00 | 423.56 | 425.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 15:00:00 | 427.00 | 423.56 | 425.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 429.90 | 424.83 | 425.62 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2024-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 09:15:00 | 432.60 | 426.38 | 426.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 10:15:00 | 442.50 | 429.61 | 427.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 14:15:00 | 446.45 | 447.71 | 443.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 15:00:00 | 446.45 | 447.71 | 443.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 445.05 | 447.18 | 443.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 446.10 | 447.18 | 443.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 448.15 | 447.38 | 444.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 10:45:00 | 450.40 | 448.22 | 444.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 09:45:00 | 449.50 | 451.31 | 450.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 10:45:00 | 449.10 | 451.29 | 450.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 13:00:00 | 449.00 | 450.36 | 450.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 14:15:00 | 447.50 | 449.54 | 449.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 14:15:00 | 447.50 | 449.54 | 449.71 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 454.80 | 450.50 | 450.11 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 11:15:00 | 444.20 | 449.13 | 449.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 12:15:00 | 441.20 | 447.54 | 448.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 13:15:00 | 441.45 | 441.23 | 444.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-26 14:00:00 | 441.45 | 441.23 | 444.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 430.80 | 433.57 | 437.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 11:00:00 | 428.80 | 432.61 | 436.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 14:30:00 | 427.80 | 431.98 | 435.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 09:15:00 | 420.10 | 431.70 | 434.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 12:00:00 | 429.25 | 429.48 | 432.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 429.05 | 428.88 | 431.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 11:30:00 | 426.95 | 428.41 | 430.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 09:45:00 | 425.05 | 425.31 | 427.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 12:15:00 | 425.00 | 426.43 | 427.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 10:15:00 | 427.30 | 425.80 | 426.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 428.75 | 426.39 | 427.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:45:00 | 428.00 | 426.39 | 427.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 427.40 | 426.59 | 427.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 15:00:00 | 425.00 | 426.27 | 426.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 09:15:00 | 434.90 | 427.79 | 427.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 09:15:00 | 434.90 | 427.79 | 427.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 10:15:00 | 435.60 | 432.93 | 431.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 11:15:00 | 431.60 | 432.66 | 431.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 11:15:00 | 431.60 | 432.66 | 431.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 431.60 | 432.66 | 431.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 12:00:00 | 431.60 | 432.66 | 431.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 431.80 | 432.49 | 431.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 12:30:00 | 430.60 | 432.49 | 431.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 430.60 | 432.11 | 431.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:00:00 | 430.60 | 432.11 | 431.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 432.05 | 432.10 | 431.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:30:00 | 430.55 | 432.10 | 431.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 429.70 | 431.62 | 431.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:15:00 | 427.65 | 431.62 | 431.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 421.90 | 429.67 | 430.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 09:15:00 | 420.15 | 423.51 | 424.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 15:15:00 | 420.20 | 416.97 | 419.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 15:15:00 | 420.20 | 416.97 | 419.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 420.20 | 416.97 | 419.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 10:45:00 | 411.60 | 415.81 | 418.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 12:00:00 | 411.60 | 405.24 | 407.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 11:15:00 | 408.90 | 406.38 | 406.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 408.90 | 406.38 | 406.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 12:15:00 | 412.20 | 407.55 | 406.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 11:15:00 | 412.35 | 412.46 | 410.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 11:30:00 | 413.00 | 412.46 | 410.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 411.00 | 412.00 | 410.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 13:45:00 | 410.30 | 412.00 | 410.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 417.80 | 413.16 | 410.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 15:15:00 | 419.10 | 413.16 | 410.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 10:30:00 | 418.50 | 415.51 | 412.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 11:00:00 | 418.85 | 415.51 | 412.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 12:00:00 | 418.40 | 416.09 | 413.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 425.70 | 436.74 | 434.52 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-01 11:15:00 | 426.80 | 433.12 | 433.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 11:15:00 | 426.80 | 433.12 | 433.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 10:15:00 | 422.70 | 428.23 | 430.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 15:15:00 | 414.85 | 414.78 | 419.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 15:15:00 | 414.85 | 414.78 | 419.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 15:15:00 | 414.85 | 414.78 | 419.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 09:30:00 | 408.50 | 414.14 | 418.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:30:00 | 410.55 | 413.71 | 418.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:45:00 | 409.75 | 412.50 | 416.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-07 09:15:00 | 367.65 | 396.01 | 407.24 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2024-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 15:15:00 | 392.00 | 387.20 | 387.06 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 09:15:00 | 384.40 | 386.64 | 386.81 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 11:15:00 | 389.20 | 387.00 | 386.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 14:15:00 | 392.65 | 388.73 | 387.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 385.55 | 388.70 | 387.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 385.55 | 388.70 | 387.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 385.55 | 388.70 | 387.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:00:00 | 385.55 | 388.70 | 387.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 383.55 | 387.67 | 387.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 11:00:00 | 383.55 | 387.67 | 387.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 381.80 | 386.49 | 387.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 378.25 | 384.84 | 386.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 13:15:00 | 376.35 | 376.22 | 378.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 14:00:00 | 376.35 | 376.22 | 378.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 374.30 | 375.84 | 378.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 15:00:00 | 374.30 | 375.84 | 378.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 379.00 | 376.47 | 378.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:15:00 | 380.00 | 376.47 | 378.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 377.00 | 376.58 | 378.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 10:15:00 | 373.75 | 376.58 | 378.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 09:15:00 | 377.80 | 372.63 | 372.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 09:15:00 | 377.80 | 372.63 | 372.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 10:15:00 | 379.25 | 373.95 | 372.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 10:15:00 | 378.20 | 378.36 | 376.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 10:30:00 | 378.30 | 378.36 | 376.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 373.45 | 377.22 | 375.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:00:00 | 373.45 | 377.22 | 375.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 378.50 | 377.47 | 376.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 15:15:00 | 382.00 | 377.87 | 376.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:45:00 | 387.65 | 380.85 | 378.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 14:15:00 | 380.70 | 383.15 | 383.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 14:15:00 | 380.70 | 383.15 | 383.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 10:15:00 | 378.00 | 381.54 | 382.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 15:15:00 | 381.00 | 379.29 | 380.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 15:15:00 | 381.00 | 379.29 | 380.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 381.00 | 379.29 | 380.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 10:00:00 | 376.00 | 378.63 | 380.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 12:45:00 | 376.00 | 378.70 | 379.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 13:45:00 | 375.95 | 378.32 | 379.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 10:15:00 | 375.90 | 377.41 | 378.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 377.95 | 376.39 | 377.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 14:45:00 | 378.65 | 376.39 | 377.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 378.95 | 376.91 | 377.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 09:30:00 | 377.00 | 377.00 | 377.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 10:15:00 | 375.30 | 377.00 | 377.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 09:15:00 | 384.35 | 376.55 | 376.78 | SL hit (close>static) qty=1.00 sl=380.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 10:15:00 | 384.25 | 378.09 | 377.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 11:15:00 | 386.85 | 379.84 | 378.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 13:15:00 | 387.15 | 390.26 | 386.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 14:00:00 | 387.15 | 390.26 | 386.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 385.80 | 389.37 | 386.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:45:00 | 387.20 | 389.37 | 386.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 387.25 | 388.95 | 386.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 381.90 | 388.95 | 386.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 382.10 | 387.58 | 385.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:30:00 | 382.90 | 387.58 | 385.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 12:15:00 | 380.15 | 384.52 | 384.79 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 11:15:00 | 392.75 | 385.13 | 384.67 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 380.45 | 385.45 | 386.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 09:15:00 | 380.00 | 384.29 | 385.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 15:15:00 | 383.95 | 383.32 | 384.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 15:15:00 | 383.95 | 383.32 | 384.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 383.95 | 383.32 | 384.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 09:30:00 | 378.85 | 383.05 | 384.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 10:00:00 | 380.50 | 381.07 | 382.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 14:00:00 | 382.00 | 381.25 | 382.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 14:45:00 | 382.00 | 381.57 | 382.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 382.95 | 381.84 | 382.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 09:15:00 | 377.35 | 381.84 | 382.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 13:15:00 | 377.00 | 371.93 | 371.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2024-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 13:15:00 | 377.00 | 371.93 | 371.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 14:15:00 | 380.00 | 373.54 | 372.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 09:15:00 | 374.90 | 377.78 | 375.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 374.90 | 377.78 | 375.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 374.90 | 377.78 | 375.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:00:00 | 374.90 | 377.78 | 375.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 374.90 | 377.20 | 375.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:30:00 | 375.00 | 377.20 | 375.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 375.55 | 376.42 | 375.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:45:00 | 375.00 | 376.42 | 375.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 376.20 | 376.38 | 375.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:45:00 | 375.90 | 376.38 | 375.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 381.15 | 377.33 | 376.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:30:00 | 375.95 | 377.33 | 376.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 374.45 | 377.36 | 376.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 09:15:00 | 383.05 | 379.10 | 377.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-10 11:15:00 | 421.36 | 411.26 | 402.27 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 13:15:00 | 420.80 | 422.54 | 422.63 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 14:15:00 | 423.70 | 422.77 | 422.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 09:15:00 | 429.80 | 424.36 | 423.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 14:15:00 | 424.30 | 425.00 | 424.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 14:15:00 | 424.30 | 425.00 | 424.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 424.30 | 425.00 | 424.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 14:45:00 | 423.85 | 425.00 | 424.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 421.30 | 424.26 | 423.93 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 416.00 | 422.61 | 423.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 12:15:00 | 409.30 | 417.64 | 420.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 15:15:00 | 398.05 | 397.90 | 403.26 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 09:15:00 | 390.55 | 397.90 | 403.26 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 371.02 | 380.69 | 389.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-23 09:15:00 | 380.70 | 380.69 | 389.62 | SL hit (close>ema400) qty=0.50 sl=380.69 alert=retest1 |

### Cycle 108 — BUY (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 11:15:00 | 365.30 | 359.59 | 358.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 12:15:00 | 366.45 | 360.96 | 359.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 13:15:00 | 364.10 | 367.54 | 365.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 13:15:00 | 364.10 | 367.54 | 365.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 13:15:00 | 364.10 | 367.54 | 365.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 14:00:00 | 364.10 | 367.54 | 365.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 363.45 | 366.72 | 364.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 15:00:00 | 363.45 | 366.72 | 364.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 365.90 | 366.56 | 365.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:15:00 | 365.95 | 366.56 | 365.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 364.80 | 366.21 | 364.99 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2024-11-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 11:15:00 | 360.00 | 364.08 | 364.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 13:15:00 | 356.85 | 361.82 | 363.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 361.05 | 360.18 | 361.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 09:15:00 | 361.05 | 360.18 | 361.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 361.05 | 360.18 | 361.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:00:00 | 361.05 | 360.18 | 361.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 359.80 | 360.02 | 361.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:45:00 | 359.95 | 360.02 | 361.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 360.00 | 359.96 | 361.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:30:00 | 360.00 | 359.96 | 361.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 361.80 | 360.33 | 361.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:15:00 | 362.90 | 360.33 | 361.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 362.75 | 360.82 | 361.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:30:00 | 364.40 | 360.82 | 361.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 359.70 | 360.59 | 361.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 13:15:00 | 358.00 | 360.37 | 360.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 14:30:00 | 357.60 | 359.96 | 360.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 11:00:00 | 357.80 | 359.41 | 360.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 15:15:00 | 357.95 | 359.17 | 359.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 357.95 | 358.93 | 359.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 375.80 | 358.93 | 359.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-11 09:15:00 | 373.15 | 361.77 | 360.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 09:15:00 | 373.15 | 361.77 | 360.88 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 357.45 | 364.86 | 365.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 10:15:00 | 354.40 | 362.77 | 364.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 362.65 | 357.77 | 360.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 362.65 | 357.77 | 360.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 362.65 | 357.77 | 360.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 362.65 | 357.77 | 360.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 357.50 | 357.71 | 360.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 355.35 | 357.78 | 359.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 10:15:00 | 356.00 | 357.91 | 359.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 15:15:00 | 356.00 | 358.01 | 358.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 10:15:00 | 362.00 | 359.27 | 359.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 362.00 | 359.27 | 359.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 12:15:00 | 364.85 | 360.65 | 359.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 358.65 | 360.28 | 359.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 14:15:00 | 358.65 | 360.28 | 359.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 358.65 | 360.28 | 359.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:00:00 | 358.65 | 360.28 | 359.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 356.10 | 359.45 | 359.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 352.50 | 358.06 | 358.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 358.05 | 356.62 | 357.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 358.05 | 356.62 | 357.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 358.05 | 356.62 | 357.47 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 11:15:00 | 365.85 | 359.51 | 358.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 14:15:00 | 368.50 | 362.96 | 360.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 12:15:00 | 365.40 | 365.60 | 363.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 13:00:00 | 365.40 | 365.60 | 363.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 363.70 | 365.01 | 363.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 363.70 | 365.01 | 363.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 363.80 | 364.76 | 363.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 09:15:00 | 364.85 | 364.76 | 363.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 12:15:00 | 362.30 | 363.49 | 363.06 | SL hit (close<static) qty=1.00 sl=362.55 alert=retest2 |

### Cycle 115 — SELL (started 2024-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 15:15:00 | 422.65 | 423.60 | 423.69 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 10:15:00 | 425.15 | 423.73 | 423.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 12:15:00 | 427.60 | 425.07 | 424.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 09:15:00 | 422.95 | 425.17 | 424.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 422.95 | 425.17 | 424.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 422.95 | 425.17 | 424.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:00:00 | 422.95 | 425.17 | 424.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 424.20 | 424.98 | 424.66 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 418.10 | 423.33 | 423.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 13:15:00 | 416.00 | 421.86 | 423.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 410.20 | 409.56 | 414.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 10:00:00 | 410.20 | 409.56 | 414.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 406.85 | 409.09 | 411.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 10:30:00 | 406.00 | 408.44 | 411.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 12:00:00 | 405.70 | 407.89 | 410.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 09:15:00 | 417.40 | 411.50 | 411.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 417.40 | 411.50 | 411.43 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 10:15:00 | 410.00 | 411.67 | 411.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 11:15:00 | 405.80 | 410.50 | 411.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 12:15:00 | 407.75 | 403.53 | 406.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 12:15:00 | 407.75 | 403.53 | 406.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 407.75 | 403.53 | 406.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 13:00:00 | 407.75 | 403.53 | 406.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 415.40 | 405.90 | 407.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 13:30:00 | 415.15 | 405.90 | 407.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 14:15:00 | 418.00 | 408.32 | 408.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 15:15:00 | 421.00 | 410.86 | 409.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 13:15:00 | 429.00 | 429.03 | 425.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 13:45:00 | 429.30 | 429.03 | 425.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 421.50 | 427.30 | 425.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 426.00 | 427.30 | 425.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 412.55 | 424.35 | 424.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 12:15:00 | 411.60 | 419.97 | 422.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 416.00 | 414.67 | 418.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 11:00:00 | 416.00 | 414.67 | 418.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 422.65 | 416.39 | 417.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:00:00 | 422.65 | 416.39 | 417.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 425.25 | 418.16 | 418.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 419.90 | 418.16 | 418.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 10:15:00 | 420.40 | 418.96 | 418.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 10:15:00 | 420.40 | 418.96 | 418.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 13:15:00 | 422.00 | 419.93 | 419.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 09:15:00 | 419.55 | 420.06 | 419.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 09:15:00 | 419.55 | 420.06 | 419.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 419.55 | 420.06 | 419.61 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2025-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 10:15:00 | 415.30 | 419.11 | 419.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 12:15:00 | 411.50 | 416.79 | 418.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 11:15:00 | 410.25 | 410.19 | 413.59 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 13:15:00 | 407.55 | 410.14 | 413.26 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 14:00:00 | 407.40 | 409.59 | 412.72 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 14:15:00 | 413.30 | 410.33 | 412.78 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-10 14:15:00 | 413.30 | 410.33 | 412.78 | SL hit (close>ema400) qty=1.00 sl=412.78 alert=retest1 |

### Cycle 124 — BUY (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 09:15:00 | 401.40 | 398.99 | 398.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 14:15:00 | 405.00 | 402.44 | 400.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 14:15:00 | 403.15 | 403.25 | 402.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 14:15:00 | 403.15 | 403.25 | 402.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 403.15 | 403.25 | 402.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:45:00 | 402.45 | 403.25 | 402.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 395.40 | 401.80 | 401.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 394.00 | 401.80 | 401.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 396.35 | 400.71 | 401.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 09:15:00 | 392.40 | 397.36 | 399.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 10:15:00 | 386.25 | 385.92 | 390.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-24 11:00:00 | 386.25 | 385.92 | 390.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 361.20 | 355.20 | 361.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 362.45 | 355.20 | 361.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 362.00 | 356.56 | 361.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 362.00 | 356.56 | 361.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 364.60 | 358.17 | 362.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:00:00 | 364.60 | 358.17 | 362.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 361.95 | 358.93 | 362.20 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 370.60 | 363.61 | 363.56 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 15:15:00 | 362.75 | 363.78 | 363.91 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 09:15:00 | 365.65 | 364.16 | 364.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 11:15:00 | 368.90 | 365.59 | 364.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 10:15:00 | 363.15 | 367.14 | 366.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 10:15:00 | 363.15 | 367.14 | 366.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 363.15 | 367.14 | 366.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:00:00 | 363.15 | 367.14 | 366.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 362.15 | 366.14 | 365.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 360.75 | 366.14 | 365.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 362.60 | 365.43 | 365.50 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 14:15:00 | 372.30 | 366.66 | 366.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 09:15:00 | 377.95 | 369.58 | 367.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 12:15:00 | 376.40 | 376.56 | 373.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-04 13:00:00 | 376.40 | 376.56 | 373.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 377.30 | 380.30 | 378.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 15:00:00 | 377.30 | 380.30 | 378.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 15:15:00 | 378.20 | 379.88 | 378.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 09:45:00 | 379.10 | 379.41 | 377.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 11:45:00 | 379.20 | 378.83 | 377.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 12:15:00 | 380.60 | 378.83 | 377.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 373.80 | 380.90 | 380.40 | SL hit (close<static) qty=1.00 sl=375.50 alert=retest2 |

### Cycle 131 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 373.95 | 379.51 | 379.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 372.10 | 378.03 | 379.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 379.45 | 376.36 | 377.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 14:15:00 | 379.45 | 376.36 | 377.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 379.45 | 376.36 | 377.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 14:45:00 | 382.80 | 376.36 | 377.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 377.75 | 376.64 | 377.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:15:00 | 350.50 | 376.64 | 377.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 09:15:00 | 332.97 | 337.89 | 345.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-17 09:15:00 | 315.45 | 331.04 | 338.28 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 132 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 326.55 | 324.19 | 323.94 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 09:15:00 | 319.70 | 323.72 | 323.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 13:15:00 | 318.65 | 321.50 | 322.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 12:15:00 | 315.85 | 315.39 | 317.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 12:30:00 | 317.00 | 315.39 | 317.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 318.70 | 316.03 | 317.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 15:00:00 | 318.70 | 316.03 | 317.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 15:15:00 | 320.00 | 316.82 | 317.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 09:15:00 | 314.70 | 316.82 | 317.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 318.25 | 317.11 | 317.80 | EMA400 retest candle locked (from downside) |

### Cycle 134 — BUY (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 11:15:00 | 334.30 | 321.01 | 319.48 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-02-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 12:15:00 | 315.00 | 320.56 | 321.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 09:15:00 | 314.25 | 319.18 | 320.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 317.80 | 317.20 | 318.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 317.80 | 317.20 | 318.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 321.45 | 318.05 | 319.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:00:00 | 321.45 | 318.05 | 319.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 322.40 | 318.92 | 319.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:45:00 | 323.45 | 318.92 | 319.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 325.85 | 320.80 | 320.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 13:15:00 | 330.55 | 324.28 | 322.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 339.25 | 340.09 | 334.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 12:00:00 | 339.25 | 340.09 | 334.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 339.75 | 341.29 | 339.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 338.80 | 341.29 | 339.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 336.10 | 340.25 | 338.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 336.10 | 340.25 | 338.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 336.00 | 339.40 | 338.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 336.00 | 339.40 | 338.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 338.00 | 338.94 | 338.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 14:30:00 | 341.30 | 339.21 | 338.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 336.15 | 339.02 | 338.73 | SL hit (close<static) qty=1.00 sl=337.85 alert=retest2 |

### Cycle 137 — SELL (started 2025-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-20 15:15:00 | 358.50 | 361.89 | 362.00 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 09:15:00 | 367.55 | 363.02 | 362.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 14:15:00 | 387.30 | 369.72 | 366.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 14:15:00 | 376.90 | 380.03 | 374.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 15:00:00 | 376.90 | 380.03 | 374.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 371.55 | 377.87 | 374.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 371.55 | 377.87 | 374.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 371.25 | 376.55 | 374.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 371.25 | 376.55 | 374.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 373.65 | 374.06 | 373.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 14:30:00 | 374.15 | 374.06 | 373.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 378.00 | 374.85 | 374.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 371.85 | 374.85 | 374.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 371.25 | 374.13 | 373.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 368.90 | 374.13 | 373.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 10:15:00 | 368.50 | 373.00 | 373.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 11:15:00 | 367.00 | 371.80 | 372.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 367.80 | 367.09 | 369.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 367.80 | 367.09 | 369.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 367.80 | 367.09 | 369.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:30:00 | 369.25 | 367.09 | 369.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 380.30 | 369.39 | 369.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 380.30 | 369.39 | 369.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 365.50 | 368.61 | 369.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 14:00:00 | 352.60 | 361.28 | 365.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 15:15:00 | 352.05 | 359.68 | 364.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 09:15:00 | 380.00 | 364.99 | 363.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 09:15:00 | 380.00 | 364.99 | 363.70 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 13:15:00 | 365.50 | 371.02 | 371.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 342.05 | 363.75 | 367.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 352.20 | 351.55 | 359.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 352.20 | 351.55 | 359.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 355.30 | 352.91 | 358.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 353.30 | 358.69 | 359.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:45:00 | 353.15 | 357.42 | 358.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 13:45:00 | 353.50 | 354.47 | 356.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 15:00:00 | 353.10 | 354.19 | 356.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 361.80 | 355.68 | 356.70 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-11 11:15:00 | 364.05 | 358.42 | 357.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 364.05 | 358.42 | 357.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 368.00 | 361.45 | 359.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 14:15:00 | 369.65 | 370.51 | 367.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 15:00:00 | 369.65 | 370.51 | 367.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 379.85 | 380.40 | 378.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 378.60 | 380.40 | 378.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 385.80 | 381.48 | 378.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 11:30:00 | 380.20 | 381.48 | 378.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 374.85 | 386.81 | 385.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 374.85 | 386.81 | 385.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 370.50 | 383.55 | 384.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 14:15:00 | 368.00 | 372.92 | 376.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-06 10:15:00 | 365.60 | 364.96 | 367.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-06 11:00:00 | 365.60 | 364.96 | 367.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 362.20 | 358.19 | 362.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 362.20 | 358.19 | 362.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 367.40 | 360.03 | 362.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:15:00 | 365.65 | 360.03 | 362.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 362.65 | 360.56 | 362.71 | EMA400 retest candle locked (from downside) |

### Cycle 144 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 368.80 | 364.35 | 364.08 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 12:15:00 | 361.30 | 363.76 | 363.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 358.90 | 362.79 | 363.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 15:15:00 | 364.70 | 362.10 | 362.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 15:15:00 | 364.70 | 362.10 | 362.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 364.70 | 362.10 | 362.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 09:15:00 | 340.45 | 362.10 | 362.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 366.20 | 352.31 | 355.21 | SL hit (close>static) qty=1.00 sl=364.70 alert=retest2 |

### Cycle 146 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 367.70 | 357.29 | 357.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 369.30 | 364.35 | 361.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 363.35 | 365.14 | 362.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 363.35 | 365.14 | 362.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 363.35 | 365.14 | 362.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:45:00 | 362.85 | 365.14 | 362.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 15:15:00 | 366.00 | 366.46 | 365.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:30:00 | 365.75 | 366.23 | 365.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 363.85 | 365.75 | 365.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:00:00 | 363.85 | 365.75 | 365.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 363.65 | 365.33 | 364.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:00:00 | 363.65 | 365.33 | 364.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 364.90 | 365.11 | 364.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:15:00 | 366.00 | 365.11 | 364.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 366.60 | 364.76 | 364.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:45:00 | 365.50 | 365.94 | 365.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 13:15:00 | 371.40 | 374.39 | 374.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 13:15:00 | 371.40 | 374.39 | 374.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 14:15:00 | 370.80 | 373.67 | 374.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 375.95 | 373.70 | 374.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 375.95 | 373.70 | 374.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 375.95 | 373.70 | 374.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:30:00 | 375.90 | 373.70 | 374.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 375.65 | 374.09 | 374.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:45:00 | 376.85 | 374.09 | 374.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 375.55 | 374.38 | 374.37 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 372.65 | 374.04 | 374.21 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 377.00 | 374.46 | 374.29 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 11:15:00 | 373.50 | 374.48 | 374.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 12:15:00 | 371.60 | 373.91 | 374.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 365.45 | 365.32 | 366.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 365.45 | 365.32 | 366.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 365.45 | 365.32 | 366.96 | EMA400 retest candle locked (from downside) |

### Cycle 152 — BUY (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 13:15:00 | 372.25 | 367.92 | 367.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 09:15:00 | 376.00 | 370.20 | 368.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 376.00 | 376.16 | 373.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 13:00:00 | 376.00 | 376.16 | 373.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 374.40 | 375.78 | 374.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 374.40 | 375.78 | 374.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 374.00 | 375.42 | 374.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 10:00:00 | 377.00 | 375.74 | 374.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 10:30:00 | 376.95 | 376.09 | 374.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 13:45:00 | 377.00 | 376.39 | 375.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 377.20 | 375.65 | 375.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 376.75 | 375.87 | 375.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:30:00 | 376.90 | 375.87 | 375.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 374.90 | 375.68 | 375.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:45:00 | 375.00 | 375.68 | 375.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 374.35 | 375.41 | 375.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:00:00 | 374.35 | 375.41 | 375.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-05 14:15:00 | 374.45 | 374.80 | 374.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 14:15:00 | 374.45 | 374.80 | 374.84 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 15:15:00 | 375.40 | 374.92 | 374.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 377.45 | 375.42 | 375.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 13:15:00 | 375.80 | 376.15 | 375.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 13:15:00 | 375.80 | 376.15 | 375.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 375.80 | 376.15 | 375.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 13:45:00 | 375.40 | 376.15 | 375.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 374.80 | 375.88 | 375.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 374.80 | 375.88 | 375.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 374.80 | 375.67 | 375.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 375.45 | 375.67 | 375.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 10:45:00 | 377.00 | 375.88 | 375.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 373.15 | 378.34 | 378.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 373.15 | 378.34 | 378.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 369.65 | 373.85 | 376.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 15:15:00 | 359.20 | 358.77 | 362.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-17 09:15:00 | 358.70 | 358.77 | 362.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 361.00 | 359.22 | 362.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 362.40 | 359.22 | 362.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 361.80 | 359.73 | 362.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 362.00 | 359.73 | 362.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 360.05 | 359.80 | 361.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:30:00 | 358.50 | 359.47 | 361.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 09:15:00 | 340.57 | 347.59 | 351.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 13:15:00 | 346.90 | 346.81 | 349.91 | SL hit (close>ema200) qty=0.50 sl=346.81 alert=retest2 |

### Cycle 156 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 352.45 | 349.18 | 348.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 354.25 | 350.13 | 349.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 360.95 | 363.74 | 360.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 12:15:00 | 360.95 | 363.74 | 360.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 360.95 | 363.74 | 360.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:00:00 | 360.95 | 363.74 | 360.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 361.40 | 363.27 | 361.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:30:00 | 360.90 | 363.27 | 361.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 357.65 | 362.15 | 360.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 357.65 | 362.15 | 360.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 358.35 | 361.39 | 360.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 364.15 | 361.39 | 360.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 13:15:00 | 363.70 | 364.89 | 364.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 363.70 | 364.89 | 364.94 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 366.95 | 365.25 | 365.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 11:15:00 | 369.40 | 366.08 | 365.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 10:15:00 | 367.85 | 367.95 | 366.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:00:00 | 367.85 | 367.95 | 366.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 365.65 | 367.49 | 366.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:00:00 | 365.65 | 367.49 | 366.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 365.50 | 367.09 | 366.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:45:00 | 365.25 | 367.09 | 366.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 365.50 | 366.77 | 366.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:00:00 | 365.50 | 366.77 | 366.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 366.10 | 366.59 | 366.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 368.55 | 366.59 | 366.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 367.75 | 366.82 | 366.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 370.55 | 367.72 | 367.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:45:00 | 370.85 | 368.91 | 367.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 379.25 | 381.39 | 381.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 379.25 | 381.39 | 381.59 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 12:15:00 | 383.45 | 381.93 | 381.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 09:15:00 | 385.45 | 382.55 | 382.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 382.40 | 382.52 | 382.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 10:15:00 | 382.40 | 382.52 | 382.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 382.40 | 382.52 | 382.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 382.40 | 382.52 | 382.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 381.10 | 382.24 | 382.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:30:00 | 381.25 | 382.24 | 382.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — SELL (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 12:15:00 | 380.45 | 381.88 | 381.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 12:15:00 | 379.10 | 380.58 | 381.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 13:15:00 | 382.85 | 381.03 | 381.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 13:15:00 | 382.85 | 381.03 | 381.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 382.85 | 381.03 | 381.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:00:00 | 382.85 | 381.03 | 381.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 14:15:00 | 383.25 | 381.47 | 381.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 388.85 | 383.43 | 382.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 379.00 | 384.89 | 384.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 379.00 | 384.89 | 384.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 379.00 | 384.89 | 384.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 379.00 | 384.89 | 384.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 381.00 | 384.11 | 383.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 11:15:00 | 381.45 | 384.11 | 383.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 377.40 | 382.77 | 383.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 377.40 | 382.77 | 383.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 12:15:00 | 376.45 | 381.51 | 382.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 10:15:00 | 379.35 | 378.31 | 380.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 10:45:00 | 380.00 | 378.31 | 380.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 378.60 | 376.96 | 378.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:45:00 | 376.20 | 377.10 | 378.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 383.50 | 379.20 | 378.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 383.50 | 379.20 | 378.95 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 378.45 | 379.07 | 379.08 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 11:15:00 | 380.40 | 379.33 | 379.20 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 374.35 | 378.40 | 378.83 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 10:15:00 | 381.30 | 379.34 | 379.16 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 376.25 | 378.74 | 378.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 372.95 | 377.27 | 378.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 14:15:00 | 376.65 | 375.55 | 376.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 14:15:00 | 376.65 | 375.55 | 376.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 376.65 | 375.55 | 376.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 379.00 | 375.55 | 376.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 377.00 | 375.84 | 376.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 09:45:00 | 374.90 | 375.51 | 376.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 10:15:00 | 356.15 | 363.99 | 368.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 15:15:00 | 361.00 | 360.50 | 364.59 | SL hit (close>ema200) qty=0.50 sl=360.50 alert=retest2 |

### Cycle 170 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 363.55 | 361.23 | 361.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 13:15:00 | 364.00 | 361.78 | 361.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 13:15:00 | 393.80 | 394.26 | 384.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 14:00:00 | 393.80 | 394.26 | 384.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 13:15:00 | 386.35 | 389.83 | 386.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 13:45:00 | 386.60 | 389.83 | 386.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 384.80 | 388.83 | 386.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 15:00:00 | 384.80 | 388.83 | 386.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 387.65 | 388.59 | 386.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 389.30 | 388.59 | 386.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 386.05 | 388.08 | 386.71 | EMA400 retest candle locked (from upside) |

### Cycle 171 — SELL (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 15:15:00 | 385.55 | 386.33 | 386.41 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 394.85 | 388.04 | 387.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 10:15:00 | 398.90 | 390.21 | 388.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 14:15:00 | 406.15 | 406.82 | 402.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 15:00:00 | 406.15 | 406.82 | 402.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 406.20 | 408.43 | 405.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 403.75 | 408.43 | 405.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 403.60 | 407.47 | 405.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 404.10 | 407.47 | 405.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 402.00 | 406.37 | 405.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:00:00 | 402.00 | 406.37 | 405.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 402.10 | 405.18 | 405.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 14:00:00 | 402.10 | 405.18 | 405.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 402.40 | 404.53 | 404.73 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 10:15:00 | 405.90 | 404.98 | 404.91 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 403.30 | 404.67 | 404.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 401.95 | 403.99 | 404.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 404.65 | 403.40 | 404.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 404.65 | 403.40 | 404.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 404.65 | 403.40 | 404.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 404.65 | 403.40 | 404.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 405.30 | 403.78 | 404.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 407.35 | 403.78 | 404.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 400.75 | 401.29 | 402.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 14:15:00 | 396.55 | 400.69 | 401.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 403.45 | 400.73 | 401.30 | SL hit (close>static) qty=1.00 sl=402.70 alert=retest2 |

### Cycle 176 — BUY (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 12:15:00 | 405.55 | 401.70 | 401.69 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 13:15:00 | 399.80 | 401.32 | 401.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 14:15:00 | 397.95 | 400.64 | 401.19 | Break + close below crossover candle low |

### Cycle 178 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 415.55 | 403.20 | 402.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 10:15:00 | 418.40 | 413.75 | 410.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 13:15:00 | 411.50 | 413.54 | 411.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 411.50 | 413.54 | 411.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 411.50 | 413.54 | 411.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 12:30:00 | 415.35 | 413.93 | 412.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 13:15:00 | 407.60 | 413.85 | 413.72 | SL hit (close<static) qty=1.00 sl=409.70 alert=retest2 |

### Cycle 179 — SELL (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 14:15:00 | 408.90 | 412.86 | 413.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 12:15:00 | 405.50 | 409.47 | 411.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 401.00 | 398.30 | 400.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 12:15:00 | 401.00 | 398.30 | 400.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 401.00 | 398.30 | 400.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:45:00 | 401.10 | 398.30 | 400.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 400.00 | 398.64 | 400.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 14:45:00 | 399.00 | 399.03 | 400.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 15:15:00 | 403.00 | 399.83 | 400.80 | SL hit (close>static) qty=1.00 sl=401.30 alert=retest2 |

### Cycle 180 — BUY (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 13:15:00 | 402.25 | 401.45 | 401.35 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 09:15:00 | 397.50 | 400.68 | 401.03 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 15:15:00 | 403.05 | 401.35 | 401.14 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 09:15:00 | 397.80 | 400.83 | 401.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 13:15:00 | 395.40 | 398.48 | 399.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 389.90 | 389.07 | 391.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 10:00:00 | 389.90 | 389.07 | 391.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 364.50 | 363.78 | 366.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 368.00 | 363.78 | 366.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 371.70 | 365.36 | 366.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:00:00 | 371.70 | 365.36 | 366.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 373.60 | 367.01 | 367.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:30:00 | 370.30 | 367.01 | 367.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 380.20 | 369.65 | 368.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 404.00 | 376.52 | 371.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 384.00 | 384.87 | 378.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 11:00:00 | 384.00 | 384.87 | 378.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 380.85 | 382.44 | 379.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:45:00 | 381.35 | 382.44 | 379.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 384.35 | 382.43 | 380.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:30:00 | 380.75 | 382.43 | 380.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 390.45 | 391.10 | 388.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:15:00 | 390.65 | 391.10 | 388.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:45:00 | 390.90 | 391.05 | 388.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 13:00:00 | 390.75 | 390.99 | 388.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 14:30:00 | 390.65 | 390.89 | 388.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 392.95 | 391.25 | 389.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-10 14:15:00 | 384.95 | 388.80 | 388.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 14:15:00 | 384.95 | 388.80 | 388.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 382.05 | 386.56 | 387.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 380.75 | 378.90 | 381.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 380.75 | 378.90 | 381.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 380.75 | 378.90 | 381.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 381.00 | 378.90 | 381.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 382.10 | 379.54 | 381.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:30:00 | 380.45 | 380.28 | 381.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 14:15:00 | 386.45 | 381.77 | 381.93 | SL hit (close>static) qty=1.00 sl=384.10 alert=retest2 |

### Cycle 186 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 385.00 | 382.41 | 382.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 13:15:00 | 387.15 | 385.04 | 383.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 382.60 | 385.42 | 384.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 382.60 | 385.42 | 384.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 382.60 | 385.42 | 384.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 382.60 | 385.42 | 384.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 385.00 | 385.34 | 384.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 390.70 | 386.33 | 385.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:30:00 | 387.35 | 386.76 | 386.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 10:00:00 | 387.10 | 386.76 | 386.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 11:45:00 | 387.90 | 386.83 | 386.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 387.10 | 386.89 | 386.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 387.10 | 386.89 | 386.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 385.00 | 386.51 | 386.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 384.35 | 386.51 | 386.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 386.15 | 386.44 | 386.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:45:00 | 384.45 | 386.44 | 386.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-24 15:15:00 | 384.70 | 386.09 | 386.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — SELL (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 15:15:00 | 384.70 | 386.09 | 386.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 13:15:00 | 384.00 | 385.07 | 385.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 386.80 | 385.24 | 385.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 386.80 | 385.24 | 385.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 386.80 | 385.24 | 385.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 388.40 | 385.24 | 385.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 390.60 | 386.31 | 386.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 14:15:00 | 393.30 | 389.39 | 387.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 389.65 | 390.20 | 388.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 10:00:00 | 389.65 | 390.20 | 388.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 389.00 | 389.86 | 388.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:45:00 | 389.30 | 389.86 | 388.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 389.95 | 389.88 | 388.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:00:00 | 392.50 | 390.09 | 389.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 10:15:00 | 389.10 | 390.60 | 390.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 389.10 | 390.60 | 390.71 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 393.00 | 391.08 | 390.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 12:15:00 | 394.85 | 391.83 | 391.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 391.00 | 396.07 | 394.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 391.00 | 396.07 | 394.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 391.00 | 396.07 | 394.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:45:00 | 390.95 | 396.07 | 394.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 390.60 | 394.98 | 394.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 387.25 | 394.98 | 394.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 392.00 | 393.93 | 394.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 385.50 | 390.72 | 392.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 390.25 | 389.94 | 391.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:45:00 | 390.40 | 389.94 | 391.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 390.50 | 390.05 | 391.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 391.40 | 390.05 | 391.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 393.10 | 390.66 | 391.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 393.10 | 390.66 | 391.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 390.25 | 390.58 | 391.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 388.80 | 390.58 | 391.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 10:45:00 | 388.50 | 389.57 | 390.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 12:30:00 | 388.20 | 387.40 | 388.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 10:15:00 | 384.85 | 379.00 | 378.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 10:15:00 | 384.85 | 379.00 | 378.37 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 12:15:00 | 376.90 | 379.40 | 379.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 375.50 | 377.43 | 378.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 375.85 | 374.55 | 375.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 375.85 | 374.55 | 375.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 375.85 | 374.55 | 375.69 | EMA400 retest candle locked (from downside) |

### Cycle 194 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 381.30 | 376.84 | 376.40 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 375.40 | 376.48 | 376.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 14:15:00 | 372.85 | 375.75 | 376.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 09:15:00 | 375.55 | 375.27 | 375.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 375.55 | 375.27 | 375.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 375.55 | 375.27 | 375.93 | EMA400 retest candle locked (from downside) |

### Cycle 196 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 11:15:00 | 377.45 | 375.91 | 375.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 12:15:00 | 378.60 | 376.45 | 376.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 14:15:00 | 376.00 | 376.49 | 376.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 14:15:00 | 376.00 | 376.49 | 376.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 376.00 | 376.49 | 376.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 15:00:00 | 376.00 | 376.49 | 376.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 376.60 | 376.51 | 376.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 380.80 | 376.51 | 376.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 10:00:00 | 378.80 | 376.97 | 376.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 12:15:00 | 377.65 | 377.18 | 376.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:30:00 | 378.05 | 378.39 | 377.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 378.05 | 378.33 | 377.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 378.05 | 378.33 | 377.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 376.50 | 377.96 | 377.77 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-04 13:15:00 | 376.75 | 377.52 | 377.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — SELL (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 13:15:00 | 376.75 | 377.52 | 377.59 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 14:15:00 | 378.10 | 377.64 | 377.64 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 11:15:00 | 375.10 | 377.18 | 377.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 14:15:00 | 374.15 | 376.15 | 376.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 09:15:00 | 375.80 | 373.93 | 374.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 09:15:00 | 375.80 | 373.93 | 374.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 375.80 | 373.93 | 374.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:00:00 | 375.80 | 373.93 | 374.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 377.10 | 374.56 | 375.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 377.10 | 374.56 | 375.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 379.50 | 375.55 | 375.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 14:15:00 | 380.20 | 376.88 | 376.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 13:15:00 | 381.10 | 381.63 | 379.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 14:00:00 | 381.10 | 381.63 | 379.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 379.00 | 381.10 | 379.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 15:00:00 | 379.00 | 381.10 | 379.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 379.35 | 380.75 | 379.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 376.95 | 380.75 | 379.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 375.00 | 379.60 | 378.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:45:00 | 374.20 | 379.60 | 378.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 376.00 | 378.88 | 378.68 | EMA400 retest candle locked (from upside) |

### Cycle 201 — SELL (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 11:15:00 | 375.90 | 378.29 | 378.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 11:15:00 | 374.25 | 376.09 | 376.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 12:15:00 | 365.85 | 363.15 | 365.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 12:15:00 | 365.85 | 363.15 | 365.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 365.85 | 363.15 | 365.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:45:00 | 365.25 | 363.15 | 365.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 367.05 | 363.93 | 365.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:45:00 | 367.35 | 363.93 | 365.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 364.90 | 364.13 | 365.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:30:00 | 367.10 | 364.13 | 365.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 365.50 | 364.40 | 365.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 365.45 | 364.40 | 365.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 368.45 | 365.21 | 365.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:00:00 | 368.45 | 365.21 | 365.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 368.00 | 365.77 | 365.81 | EMA400 retest candle locked (from downside) |

### Cycle 202 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 368.00 | 366.22 | 366.01 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2025-12-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 14:15:00 | 366.10 | 366.36 | 366.38 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 09:15:00 | 367.75 | 366.48 | 366.42 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 12:15:00 | 365.30 | 366.36 | 366.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 13:15:00 | 364.00 | 365.89 | 366.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 366.20 | 365.50 | 365.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 366.20 | 365.50 | 365.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 366.20 | 365.50 | 365.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 366.20 | 365.50 | 365.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 366.00 | 365.60 | 365.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:15:00 | 367.20 | 365.60 | 365.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 367.10 | 365.90 | 366.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 367.10 | 365.90 | 366.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — BUY (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 12:15:00 | 367.65 | 366.25 | 366.15 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 365.35 | 366.31 | 366.36 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 367.30 | 365.58 | 365.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 368.40 | 366.49 | 366.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 10:15:00 | 366.35 | 366.85 | 366.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 10:15:00 | 366.35 | 366.85 | 366.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 366.35 | 366.85 | 366.39 | EMA400 retest candle locked (from upside) |

### Cycle 209 — SELL (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 10:15:00 | 364.20 | 365.79 | 366.01 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 12:15:00 | 367.70 | 366.29 | 366.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 369.00 | 367.15 | 366.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 367.40 | 368.08 | 367.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 13:15:00 | 367.40 | 368.08 | 367.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 367.40 | 368.08 | 367.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 367.40 | 368.08 | 367.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 365.30 | 367.53 | 367.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:30:00 | 364.50 | 367.53 | 367.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — SELL (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 15:15:00 | 363.15 | 366.65 | 366.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 358.80 | 365.08 | 366.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 346.20 | 346.10 | 348.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 344.90 | 346.10 | 348.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 346.55 | 345.74 | 347.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 346.55 | 345.74 | 347.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 345.25 | 345.84 | 347.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:30:00 | 344.40 | 345.50 | 346.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 343.85 | 345.20 | 346.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 327.18 | 331.44 | 334.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 326.66 | 330.89 | 334.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 12:15:00 | 333.60 | 330.73 | 333.33 | SL hit (close>ema200) qty=0.50 sl=330.73 alert=retest2 |

### Cycle 212 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 317.00 | 314.09 | 313.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 13:15:00 | 320.45 | 316.54 | 315.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 328.05 | 329.69 | 325.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 11:00:00 | 328.05 | 329.69 | 325.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 327.40 | 329.51 | 327.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 327.40 | 329.51 | 327.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 348.15 | 333.24 | 329.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 12:15:00 | 350.60 | 333.24 | 329.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 352.25 | 342.00 | 335.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 13:30:00 | 350.35 | 348.79 | 342.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 338.45 | 342.91 | 343.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 338.45 | 342.91 | 343.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 12:15:00 | 337.05 | 340.92 | 342.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 331.30 | 330.82 | 332.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 331.30 | 330.82 | 332.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 333.95 | 331.16 | 332.45 | EMA400 retest candle locked (from downside) |

### Cycle 214 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 335.50 | 333.06 | 333.01 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 331.15 | 332.94 | 333.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 330.10 | 332.37 | 332.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 15:15:00 | 321.50 | 321.29 | 323.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 09:15:00 | 320.80 | 321.29 | 323.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 318.50 | 320.74 | 323.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:00:00 | 317.60 | 320.11 | 322.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:45:00 | 317.35 | 319.53 | 322.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 315.95 | 318.27 | 320.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 316.75 | 318.60 | 319.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 315.70 | 317.36 | 318.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 13:15:00 | 315.00 | 317.36 | 318.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 15:00:00 | 311.60 | 316.48 | 317.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 301.72 | 314.81 | 316.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 301.48 | 314.81 | 316.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 300.91 | 314.81 | 316.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 14:15:00 | 312.40 | 312.36 | 314.66 | SL hit (close>ema200) qty=0.50 sl=312.36 alert=retest2 |

### Cycle 216 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 11:15:00 | 317.75 | 312.71 | 312.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 12:15:00 | 328.25 | 315.81 | 313.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 15:15:00 | 328.50 | 328.92 | 323.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 09:15:00 | 316.40 | 328.92 | 323.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 316.40 | 326.41 | 323.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:00:00 | 316.40 | 326.41 | 323.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 314.55 | 324.04 | 322.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 11:00:00 | 314.55 | 324.04 | 322.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — SELL (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 12:15:00 | 313.40 | 320.31 | 320.93 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 12:15:00 | 320.30 | 318.41 | 318.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 09:15:00 | 325.30 | 320.48 | 319.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 323.00 | 326.99 | 324.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 323.00 | 326.99 | 324.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 323.00 | 326.99 | 324.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 323.00 | 326.99 | 324.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 321.15 | 325.82 | 323.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:00:00 | 321.15 | 325.82 | 323.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 320.85 | 324.82 | 323.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:00:00 | 320.85 | 324.82 | 323.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 219 — SELL (started 2026-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 13:15:00 | 317.70 | 322.11 | 322.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 12:15:00 | 313.55 | 317.84 | 319.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 11:15:00 | 315.85 | 315.58 | 317.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 12:00:00 | 315.85 | 315.58 | 317.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 318.00 | 316.14 | 317.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 318.00 | 316.14 | 317.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 316.30 | 316.17 | 317.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:45:00 | 318.30 | 316.17 | 317.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 318.35 | 316.39 | 317.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 318.50 | 316.39 | 317.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 322.45 | 317.60 | 317.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 322.45 | 317.60 | 317.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 220 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 322.00 | 318.48 | 318.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 324.00 | 319.58 | 318.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 319.10 | 320.96 | 319.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 319.10 | 320.96 | 319.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 319.10 | 320.96 | 319.77 | EMA400 retest candle locked (from upside) |

### Cycle 221 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 316.00 | 319.13 | 319.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 313.90 | 318.08 | 318.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 297.85 | 295.83 | 302.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 297.85 | 295.83 | 302.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 302.00 | 297.93 | 301.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 302.00 | 297.93 | 301.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 300.65 | 298.47 | 301.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 307.60 | 298.47 | 301.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 307.90 | 300.36 | 302.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 306.50 | 300.36 | 302.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 306.30 | 301.55 | 302.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 11:15:00 | 305.50 | 301.55 | 302.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 12:45:00 | 305.15 | 302.89 | 303.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 13:15:00 | 304.40 | 303.19 | 303.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 222 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 304.40 | 303.19 | 303.16 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 14:15:00 | 299.90 | 302.53 | 302.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 09:15:00 | 292.85 | 300.19 | 301.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 288.55 | 280.49 | 286.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 288.55 | 280.49 | 286.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 288.55 | 280.49 | 286.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 288.55 | 280.49 | 286.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 280.90 | 280.57 | 285.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:30:00 | 280.55 | 280.66 | 285.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 280.65 | 280.67 | 284.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:00:00 | 280.00 | 280.67 | 284.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 15:15:00 | 289.90 | 284.40 | 283.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 224 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 289.90 | 284.40 | 283.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 10:15:00 | 300.00 | 287.54 | 285.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 14:15:00 | 302.00 | 302.12 | 294.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-06 15:00:00 | 302.00 | 302.12 | 294.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 297.85 | 300.00 | 295.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:30:00 | 296.05 | 300.00 | 295.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 297.90 | 298.91 | 296.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:45:00 | 297.20 | 298.91 | 296.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 297.80 | 298.69 | 296.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 304.00 | 298.69 | 296.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-20 15:15:00 | 334.40 | 327.58 | 323.06 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 225 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 327.60 | 332.80 | 333.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 326.75 | 331.59 | 332.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 329.80 | 328.54 | 330.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 329.80 | 328.54 | 330.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 329.80 | 328.54 | 330.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 330.95 | 328.54 | 330.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 329.55 | 328.74 | 330.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 328.50 | 328.74 | 330.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:45:00 | 328.50 | 328.99 | 330.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:30:00 | 328.45 | 328.91 | 329.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:15:00 | 328.60 | 328.91 | 329.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 327.55 | 326.93 | 328.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:30:00 | 325.85 | 327.10 | 327.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:45:00 | 325.85 | 324.41 | 324.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 324.65 | 324.41 | 324.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 13:15:00 | 325.25 | 324.58 | 324.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 226 — BUY (started 2026-05-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 13:15:00 | 325.25 | 324.58 | 324.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 327.55 | 325.49 | 325.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 10:15:00 | 324.50 | 325.49 | 325.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 10:15:00 | 324.50 | 325.49 | 325.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 324.50 | 325.49 | 325.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 323.75 | 325.49 | 325.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 324.95 | 325.38 | 325.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 324.35 | 325.38 | 325.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 326.00 | 325.51 | 325.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 14:15:00 | 327.20 | 325.61 | 325.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:30:00 | 328.20 | 326.86 | 326.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:00:00 | 327.55 | 327.25 | 326.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-18 09:15:00 | 197.30 | 2023-05-18 12:15:00 | 217.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-07 09:15:00 | 216.60 | 2023-06-07 14:15:00 | 214.15 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2023-06-07 13:30:00 | 215.60 | 2023-06-07 14:15:00 | 214.15 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2023-06-16 09:15:00 | 213.65 | 2023-06-16 13:15:00 | 211.10 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2023-06-26 13:00:00 | 209.75 | 2023-06-27 09:15:00 | 211.15 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2023-06-26 14:15:00 | 210.00 | 2023-06-27 09:15:00 | 211.15 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2023-07-10 15:15:00 | 205.05 | 2023-07-11 09:15:00 | 207.90 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2023-07-17 09:15:00 | 215.50 | 2023-07-18 14:15:00 | 214.15 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2023-07-18 14:00:00 | 213.70 | 2023-07-18 14:15:00 | 214.15 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest2 | 2023-07-24 15:15:00 | 218.80 | 2023-07-25 10:15:00 | 215.85 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2023-08-14 10:45:00 | 206.25 | 2023-08-16 14:15:00 | 211.20 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2023-08-14 11:45:00 | 206.35 | 2023-08-16 14:15:00 | 211.20 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2023-08-14 15:15:00 | 206.40 | 2023-08-16 14:15:00 | 211.20 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2023-08-24 14:00:00 | 237.80 | 2023-08-28 09:15:00 | 235.80 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2023-08-24 15:00:00 | 237.80 | 2023-08-28 09:15:00 | 235.80 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2023-08-25 11:45:00 | 238.35 | 2023-08-28 09:15:00 | 235.80 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2023-09-15 11:00:00 | 243.30 | 2023-09-20 09:15:00 | 231.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-15 11:45:00 | 243.50 | 2023-09-20 09:15:00 | 231.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-15 12:15:00 | 242.45 | 2023-09-20 09:15:00 | 230.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-15 11:00:00 | 243.30 | 2023-09-21 09:15:00 | 239.65 | STOP_HIT | 0.50 | 1.50% |
| SELL | retest2 | 2023-09-15 11:45:00 | 243.50 | 2023-09-21 09:15:00 | 239.65 | STOP_HIT | 0.50 | 1.58% |
| SELL | retest2 | 2023-09-15 12:15:00 | 242.45 | 2023-09-21 09:15:00 | 239.65 | STOP_HIT | 0.50 | 1.15% |
| BUY | retest2 | 2023-10-06 09:15:00 | 235.20 | 2023-10-09 12:15:00 | 229.55 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2023-10-11 10:45:00 | 230.55 | 2023-10-13 09:15:00 | 232.30 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2023-10-11 13:30:00 | 231.05 | 2023-10-13 09:15:00 | 232.30 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2023-10-12 11:15:00 | 231.20 | 2023-10-13 09:15:00 | 232.30 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2023-10-18 12:15:00 | 227.70 | 2023-10-23 09:15:00 | 216.60 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2023-10-18 15:15:00 | 228.00 | 2023-10-23 10:15:00 | 216.31 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2023-10-18 12:15:00 | 227.70 | 2023-10-26 14:15:00 | 213.00 | STOP_HIT | 0.50 | 6.46% |
| SELL | retest2 | 2023-10-18 15:15:00 | 228.00 | 2023-10-26 14:15:00 | 213.00 | STOP_HIT | 0.50 | 6.58% |
| BUY | retest2 | 2023-11-01 15:00:00 | 226.05 | 2023-11-03 13:15:00 | 222.90 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2023-11-02 13:00:00 | 225.60 | 2023-11-03 13:15:00 | 222.90 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2023-11-13 10:30:00 | 231.30 | 2023-11-15 09:15:00 | 236.50 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2023-11-13 11:00:00 | 231.25 | 2023-11-15 09:15:00 | 236.50 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2023-11-13 13:00:00 | 231.00 | 2023-11-15 09:15:00 | 236.50 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2023-11-13 14:30:00 | 230.00 | 2023-11-15 09:15:00 | 236.50 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2023-11-17 10:45:00 | 231.00 | 2023-11-20 09:15:00 | 245.80 | STOP_HIT | 1.00 | -6.41% |
| SELL | retest2 | 2023-11-17 11:30:00 | 230.85 | 2023-11-20 09:15:00 | 245.80 | STOP_HIT | 1.00 | -6.48% |
| SELL | retest2 | 2023-11-17 14:45:00 | 230.75 | 2023-11-20 09:15:00 | 245.80 | STOP_HIT | 1.00 | -6.52% |
| BUY | retest2 | 2023-12-05 09:15:00 | 242.70 | 2023-12-12 14:15:00 | 244.70 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2023-12-05 12:45:00 | 239.45 | 2023-12-12 14:15:00 | 244.70 | STOP_HIT | 1.00 | 2.19% |
| BUY | retest2 | 2023-12-05 13:15:00 | 239.60 | 2023-12-12 14:15:00 | 244.70 | STOP_HIT | 1.00 | 2.13% |
| BUY | retest2 | 2023-12-06 09:45:00 | 239.40 | 2023-12-12 14:15:00 | 244.70 | STOP_HIT | 1.00 | 2.21% |
| BUY | retest2 | 2023-12-06 15:00:00 | 239.50 | 2023-12-12 14:15:00 | 244.70 | STOP_HIT | 1.00 | 2.17% |
| BUY | retest2 | 2023-12-07 09:30:00 | 240.25 | 2023-12-12 14:15:00 | 244.70 | STOP_HIT | 1.00 | 1.85% |
| SELL | retest2 | 2023-12-22 12:45:00 | 235.00 | 2023-12-26 10:15:00 | 239.40 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-01-10 10:30:00 | 281.00 | 2024-01-18 09:15:00 | 278.80 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-01-11 09:45:00 | 281.00 | 2024-01-18 09:15:00 | 278.80 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-01-11 11:45:00 | 281.90 | 2024-01-18 09:15:00 | 278.80 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-01-12 10:00:00 | 280.80 | 2024-01-18 09:15:00 | 278.80 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-01-15 09:15:00 | 280.40 | 2024-01-18 09:15:00 | 278.80 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-01-15 11:30:00 | 281.00 | 2024-01-18 09:15:00 | 278.80 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-01-30 15:15:00 | 323.10 | 2024-02-06 12:15:00 | 355.41 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-02-14 12:15:00 | 392.00 | 2024-02-15 09:15:00 | 403.80 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2024-02-14 14:15:00 | 392.45 | 2024-02-15 09:15:00 | 403.80 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2024-02-21 10:30:00 | 416.60 | 2024-02-21 13:15:00 | 402.35 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2024-02-21 12:45:00 | 414.50 | 2024-02-21 13:15:00 | 402.35 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2024-02-27 12:00:00 | 400.85 | 2024-02-29 11:15:00 | 402.20 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2024-02-27 12:45:00 | 399.75 | 2024-02-29 15:15:00 | 404.65 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-02-27 13:30:00 | 400.30 | 2024-02-29 15:15:00 | 404.65 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-02-27 14:45:00 | 400.70 | 2024-02-29 15:15:00 | 404.65 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-02-29 09:15:00 | 389.30 | 2024-02-29 15:15:00 | 404.65 | STOP_HIT | 1.00 | -3.94% |
| BUY | retest2 | 2024-03-04 09:15:00 | 423.80 | 2024-03-11 11:15:00 | 415.60 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-03-28 09:15:00 | 455.10 | 2024-04-10 09:15:00 | 500.61 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-28 15:15:00 | 455.00 | 2024-04-10 09:15:00 | 500.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-04-18 14:30:00 | 457.75 | 2024-04-22 13:15:00 | 469.85 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2024-04-22 09:30:00 | 454.60 | 2024-04-22 13:15:00 | 469.85 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2024-05-07 10:45:00 | 486.35 | 2024-05-09 14:15:00 | 481.45 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-05-09 09:15:00 | 494.30 | 2024-05-09 14:15:00 | 481.45 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2024-05-09 13:00:00 | 485.45 | 2024-05-09 14:15:00 | 481.45 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-05-10 12:15:00 | 484.10 | 2024-05-16 09:15:00 | 482.05 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2024-05-13 09:15:00 | 482.80 | 2024-05-16 09:15:00 | 482.05 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2024-05-15 09:30:00 | 483.05 | 2024-05-16 09:15:00 | 482.05 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest2 | 2024-05-22 09:15:00 | 489.35 | 2024-05-23 10:15:00 | 476.35 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2024-05-23 09:15:00 | 484.15 | 2024-05-23 10:15:00 | 476.35 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-05-24 10:45:00 | 476.90 | 2024-05-27 09:15:00 | 453.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 13:00:00 | 471.75 | 2024-05-27 09:15:00 | 448.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 10:45:00 | 476.90 | 2024-05-28 14:15:00 | 429.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-24 13:00:00 | 471.75 | 2024-05-29 09:15:00 | 440.60 | STOP_HIT | 0.50 | 6.60% |
| BUY | retest2 | 2024-06-20 10:45:00 | 450.40 | 2024-06-24 14:15:00 | 447.50 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-06-24 09:45:00 | 449.50 | 2024-06-24 14:15:00 | 447.50 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2024-06-24 10:45:00 | 449.10 | 2024-06-24 14:15:00 | 447.50 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-06-24 13:00:00 | 449.00 | 2024-06-24 14:15:00 | 447.50 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-06-28 11:00:00 | 428.80 | 2024-07-05 09:15:00 | 434.90 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-06-28 14:30:00 | 427.80 | 2024-07-05 09:15:00 | 434.90 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-07-01 09:15:00 | 420.10 | 2024-07-05 09:15:00 | 434.90 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2024-07-01 12:00:00 | 429.25 | 2024-07-05 09:15:00 | 434.90 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-07-02 11:30:00 | 426.95 | 2024-07-05 09:15:00 | 434.90 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-07-03 09:45:00 | 425.05 | 2024-07-05 09:15:00 | 434.90 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2024-07-03 12:15:00 | 425.00 | 2024-07-05 09:15:00 | 434.90 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2024-07-04 10:15:00 | 427.30 | 2024-07-05 09:15:00 | 434.90 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-07-04 15:00:00 | 425.00 | 2024-07-05 09:15:00 | 434.90 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2024-07-18 10:45:00 | 411.60 | 2024-07-24 11:15:00 | 408.90 | STOP_HIT | 1.00 | 0.66% |
| SELL | retest2 | 2024-07-22 12:00:00 | 411.60 | 2024-07-24 11:15:00 | 408.90 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2024-07-25 15:15:00 | 419.10 | 2024-08-01 11:15:00 | 426.80 | STOP_HIT | 1.00 | 1.84% |
| BUY | retest2 | 2024-07-26 10:30:00 | 418.50 | 2024-08-01 11:15:00 | 426.80 | STOP_HIT | 1.00 | 1.98% |
| BUY | retest2 | 2024-07-26 11:00:00 | 418.85 | 2024-08-01 11:15:00 | 426.80 | STOP_HIT | 1.00 | 1.90% |
| BUY | retest2 | 2024-07-26 12:00:00 | 418.40 | 2024-08-01 11:15:00 | 426.80 | STOP_HIT | 1.00 | 2.01% |
| SELL | retest2 | 2024-08-06 09:30:00 | 408.50 | 2024-08-07 09:15:00 | 367.65 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-06 10:30:00 | 410.55 | 2024-08-07 09:15:00 | 369.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-06 12:45:00 | 409.75 | 2024-08-07 09:15:00 | 368.78 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-19 10:15:00 | 373.75 | 2024-08-22 09:15:00 | 377.80 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-08-23 15:15:00 | 382.00 | 2024-08-28 14:15:00 | 380.70 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2024-08-26 09:45:00 | 387.65 | 2024-08-28 14:15:00 | 380.70 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-08-30 10:00:00 | 376.00 | 2024-09-05 09:15:00 | 384.35 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2024-09-02 12:45:00 | 376.00 | 2024-09-05 09:15:00 | 384.35 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2024-09-02 13:45:00 | 375.95 | 2024-09-05 10:15:00 | 384.25 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-09-03 10:15:00 | 375.90 | 2024-09-05 10:15:00 | 384.25 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2024-09-04 09:30:00 | 377.00 | 2024-09-05 10:15:00 | 384.25 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-09-04 10:15:00 | 375.30 | 2024-09-05 10:15:00 | 384.25 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2024-09-13 09:30:00 | 378.85 | 2024-09-27 13:15:00 | 377.00 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2024-09-16 10:00:00 | 380.50 | 2024-09-27 13:15:00 | 377.00 | STOP_HIT | 1.00 | 0.92% |
| SELL | retest2 | 2024-09-16 14:00:00 | 382.00 | 2024-09-27 13:15:00 | 377.00 | STOP_HIT | 1.00 | 1.31% |
| SELL | retest2 | 2024-09-16 14:45:00 | 382.00 | 2024-09-27 13:15:00 | 377.00 | STOP_HIT | 1.00 | 1.31% |
| SELL | retest2 | 2024-09-17 09:15:00 | 377.35 | 2024-09-27 13:15:00 | 377.00 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2024-10-04 09:15:00 | 383.05 | 2024-10-10 11:15:00 | 421.36 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2024-10-22 09:15:00 | 390.55 | 2024-10-23 09:15:00 | 371.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-10-22 09:15:00 | 390.55 | 2024-10-23 09:15:00 | 380.70 | STOP_HIT | 0.50 | 2.52% |
| SELL | retest2 | 2024-10-30 13:45:00 | 357.15 | 2024-10-31 11:15:00 | 365.30 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-10-31 09:15:00 | 355.65 | 2024-10-31 11:15:00 | 365.30 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2024-11-07 13:15:00 | 358.00 | 2024-11-11 09:15:00 | 373.15 | STOP_HIT | 1.00 | -4.23% |
| SELL | retest2 | 2024-11-07 14:30:00 | 357.60 | 2024-11-11 09:15:00 | 373.15 | STOP_HIT | 1.00 | -4.35% |
| SELL | retest2 | 2024-11-08 11:00:00 | 357.80 | 2024-11-11 09:15:00 | 373.15 | STOP_HIT | 1.00 | -4.29% |
| SELL | retest2 | 2024-11-08 15:15:00 | 357.95 | 2024-11-11 09:15:00 | 373.15 | STOP_HIT | 1.00 | -4.25% |
| SELL | retest2 | 2024-11-18 09:15:00 | 355.35 | 2024-11-19 10:15:00 | 362.00 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-11-18 10:15:00 | 356.00 | 2024-11-19 10:15:00 | 362.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-11-18 15:15:00 | 356.00 | 2024-11-19 10:15:00 | 362.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-11-26 09:15:00 | 364.85 | 2024-11-26 12:15:00 | 362.30 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-11-26 14:30:00 | 364.30 | 2024-12-05 12:15:00 | 400.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-27 09:30:00 | 364.35 | 2024-12-05 12:15:00 | 400.79 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-26 10:30:00 | 406.00 | 2024-12-27 09:15:00 | 417.40 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2024-12-26 12:00:00 | 405.70 | 2024-12-27 09:15:00 | 417.40 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-01-08 09:15:00 | 419.90 | 2025-01-08 10:15:00 | 420.40 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2025-01-10 13:15:00 | 407.55 | 2025-01-10 14:15:00 | 413.30 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest1 | 2025-01-10 14:00:00 | 407.40 | 2025-01-10 14:15:00 | 413.30 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-01-13 09:15:00 | 400.80 | 2025-01-20 09:15:00 | 401.40 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-01-13 11:45:00 | 401.60 | 2025-01-20 09:15:00 | 401.40 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-02-06 09:45:00 | 379.10 | 2025-02-10 09:15:00 | 373.80 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-02-06 11:45:00 | 379.20 | 2025-02-10 09:15:00 | 373.80 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-02-06 12:15:00 | 380.60 | 2025-02-10 09:15:00 | 373.80 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-02-11 09:15:00 | 350.50 | 2025-02-14 09:15:00 | 332.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 350.50 | 2025-02-17 09:15:00 | 315.45 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-10 14:30:00 | 341.30 | 2025-03-11 09:15:00 | 336.15 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-03-11 10:15:00 | 341.00 | 2025-03-20 15:15:00 | 358.50 | STOP_HIT | 1.00 | 5.13% |
| SELL | retest2 | 2025-03-28 14:00:00 | 352.60 | 2025-04-02 09:15:00 | 380.00 | STOP_HIT | 1.00 | -7.77% |
| SELL | retest2 | 2025-03-28 15:15:00 | 352.05 | 2025-04-02 09:15:00 | 380.00 | STOP_HIT | 1.00 | -7.94% |
| SELL | retest2 | 2025-04-09 09:15:00 | 353.30 | 2025-04-11 11:15:00 | 364.05 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-04-09 09:45:00 | 353.15 | 2025-04-11 11:15:00 | 364.05 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-04-09 13:45:00 | 353.50 | 2025-04-11 11:15:00 | 364.05 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2025-04-09 15:00:00 | 353.10 | 2025-04-11 11:15:00 | 364.05 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-05-09 09:15:00 | 340.45 | 2025-05-12 09:15:00 | 366.20 | STOP_HIT | 1.00 | -7.56% |
| BUY | retest2 | 2025-05-15 14:15:00 | 366.00 | 2025-05-21 13:15:00 | 371.40 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest2 | 2025-05-16 09:15:00 | 366.60 | 2025-05-21 13:15:00 | 371.40 | STOP_HIT | 1.00 | 1.31% |
| BUY | retest2 | 2025-05-16 10:45:00 | 365.50 | 2025-05-21 13:15:00 | 371.40 | STOP_HIT | 1.00 | 1.61% |
| BUY | retest2 | 2025-06-04 10:00:00 | 377.00 | 2025-06-05 14:15:00 | 374.45 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-06-04 10:30:00 | 376.95 | 2025-06-05 14:15:00 | 374.45 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-06-04 13:45:00 | 377.00 | 2025-06-05 14:15:00 | 374.45 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-06-05 09:15:00 | 377.20 | 2025-06-05 14:15:00 | 374.45 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-06-09 09:15:00 | 375.45 | 2025-06-11 13:15:00 | 373.15 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-06-09 10:45:00 | 377.00 | 2025-06-11 13:15:00 | 373.15 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-06-17 12:30:00 | 358.50 | 2025-06-20 09:15:00 | 340.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 12:30:00 | 358.50 | 2025-06-20 13:15:00 | 346.90 | STOP_HIT | 0.50 | 3.24% |
| BUY | retest2 | 2025-06-30 09:15:00 | 364.15 | 2025-07-02 13:15:00 | 363.70 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-07-08 09:15:00 | 370.55 | 2025-07-21 09:15:00 | 379.25 | STOP_HIT | 1.00 | 2.35% |
| BUY | retest2 | 2025-07-08 09:45:00 | 370.85 | 2025-07-21 09:15:00 | 379.25 | STOP_HIT | 1.00 | 2.27% |
| BUY | retest2 | 2025-07-25 11:15:00 | 381.45 | 2025-07-25 11:15:00 | 377.40 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-07-29 10:45:00 | 376.20 | 2025-07-30 09:15:00 | 383.50 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-08-05 09:45:00 | 374.90 | 2025-08-07 10:15:00 | 356.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 09:45:00 | 374.90 | 2025-08-07 15:15:00 | 361.00 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2025-09-01 14:15:00 | 396.55 | 2025-09-02 11:15:00 | 403.45 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-09-08 12:30:00 | 415.35 | 2025-09-09 13:15:00 | 407.60 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-09-15 14:45:00 | 399.00 | 2025-09-15 15:15:00 | 403.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-16 12:15:00 | 399.45 | 2025-09-16 12:15:00 | 402.55 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-10-09 11:15:00 | 390.65 | 2025-10-10 14:15:00 | 384.95 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-10-09 11:45:00 | 390.90 | 2025-10-10 14:15:00 | 384.95 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-10-09 13:00:00 | 390.75 | 2025-10-10 14:15:00 | 384.95 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-10-09 14:30:00 | 390.65 | 2025-10-10 14:15:00 | 384.95 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-10-15 12:30:00 | 380.45 | 2025-10-15 14:15:00 | 386.45 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-10-21 13:45:00 | 390.70 | 2025-10-24 15:15:00 | 384.70 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-10-24 09:30:00 | 387.35 | 2025-10-24 15:15:00 | 384.70 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-10-24 10:00:00 | 387.10 | 2025-10-24 15:15:00 | 384.70 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-10-24 11:45:00 | 387.90 | 2025-10-24 15:15:00 | 384.70 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-10-30 11:00:00 | 392.50 | 2025-11-03 10:15:00 | 389.10 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-11-10 09:15:00 | 388.80 | 2025-11-18 10:15:00 | 384.85 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2025-11-10 10:45:00 | 388.50 | 2025-11-18 10:15:00 | 384.85 | STOP_HIT | 1.00 | 0.94% |
| SELL | retest2 | 2025-11-11 12:30:00 | 388.20 | 2025-11-18 10:15:00 | 384.85 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2025-12-02 09:15:00 | 380.80 | 2025-12-04 13:15:00 | 376.75 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-12-02 10:00:00 | 378.80 | 2025-12-04 13:15:00 | 376.75 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-12-02 12:15:00 | 377.65 | 2025-12-04 13:15:00 | 376.75 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-12-04 09:30:00 | 378.05 | 2025-12-04 13:15:00 | 376.75 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2026-01-14 13:30:00 | 344.40 | 2026-01-20 15:15:00 | 327.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:15:00 | 343.85 | 2026-01-21 09:15:00 | 326.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 13:30:00 | 344.40 | 2026-01-21 12:15:00 | 333.60 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2026-01-16 09:15:00 | 343.85 | 2026-01-21 12:15:00 | 333.60 | STOP_HIT | 0.50 | 2.98% |
| BUY | retest2 | 2026-02-05 12:15:00 | 350.60 | 2026-02-11 10:15:00 | 338.45 | STOP_HIT | 1.00 | -3.47% |
| BUY | retest2 | 2026-02-06 09:15:00 | 352.25 | 2026-02-11 10:15:00 | 338.45 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2026-02-06 13:30:00 | 350.35 | 2026-02-11 10:15:00 | 338.45 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2026-02-25 11:00:00 | 317.60 | 2026-03-02 09:15:00 | 301.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:45:00 | 317.35 | 2026-03-02 09:15:00 | 301.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 09:15:00 | 315.95 | 2026-03-02 09:15:00 | 300.91 | PARTIAL | 0.50 | 4.76% |
| SELL | retest2 | 2026-02-25 11:00:00 | 317.60 | 2026-03-02 14:15:00 | 312.40 | STOP_HIT | 0.50 | 1.64% |
| SELL | retest2 | 2026-02-25 11:45:00 | 317.35 | 2026-03-02 14:15:00 | 312.40 | STOP_HIT | 0.50 | 1.56% |
| SELL | retest2 | 2026-02-26 09:15:00 | 315.95 | 2026-03-02 14:15:00 | 312.40 | STOP_HIT | 0.50 | 1.12% |
| SELL | retest2 | 2026-02-27 09:15:00 | 316.75 | 2026-03-05 11:15:00 | 317.75 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2026-02-27 13:15:00 | 315.00 | 2026-03-05 11:15:00 | 317.75 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-02-27 15:00:00 | 311.60 | 2026-03-05 11:15:00 | 317.75 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-03-25 11:15:00 | 305.50 | 2026-03-25 13:15:00 | 304.40 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest2 | 2026-03-25 12:45:00 | 305.15 | 2026-03-25 13:15:00 | 304.40 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2026-04-01 11:30:00 | 280.55 | 2026-04-02 15:15:00 | 289.90 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2026-04-01 13:30:00 | 280.65 | 2026-04-02 15:15:00 | 289.90 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2026-04-01 14:00:00 | 280.00 | 2026-04-02 15:15:00 | 289.90 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest2 | 2026-04-08 09:15:00 | 304.00 | 2026-04-20 15:15:00 | 334.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-27 11:15:00 | 328.50 | 2026-05-04 13:15:00 | 325.25 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2026-04-27 14:45:00 | 328.50 | 2026-05-04 13:15:00 | 325.25 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2026-04-28 09:30:00 | 328.45 | 2026-05-04 13:15:00 | 325.25 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2026-04-28 10:15:00 | 328.60 | 2026-05-04 13:15:00 | 325.25 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2026-04-29 13:30:00 | 325.85 | 2026-05-04 13:15:00 | 325.25 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2026-05-04 12:45:00 | 325.85 | 2026-05-04 13:15:00 | 325.25 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2026-05-04 13:15:00 | 324.65 | 2026-05-04 13:15:00 | 325.25 | STOP_HIT | 1.00 | -0.18% |
