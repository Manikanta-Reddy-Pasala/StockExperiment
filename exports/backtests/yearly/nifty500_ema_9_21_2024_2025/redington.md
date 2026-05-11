# Redington Ltd. (REDINGTON)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 223.29
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 131 |
| ALERT1 | 84 |
| ALERT2 | 84 |
| ALERT2_SKIP | 43 |
| ALERT3 | 209 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 110 |
| PARTIAL | 26 |
| TARGET_HIT | 7 |
| STOP_HIT | 107 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 137 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 75 / 62
- **Target hits / Stop hits / Partials:** 7 / 104 / 26
- **Avg / median % per leg:** 1.28% / 0.47%
- **Sum % (uncompounded):** 174.93%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 49 | 17 | 34.7% | 6 | 43 | 0 | 0.54% | 26.5% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.14% | -0.1% |
| BUY @ 3rd Alert (retest2) | 48 | 17 | 35.4% | 6 | 42 | 0 | 0.56% | 26.7% |
| SELL (all) | 88 | 58 | 65.9% | 1 | 61 | 26 | 1.69% | 148.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 88 | 58 | 65.9% | 1 | 61 | 26 | 1.69% | 148.4% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.14% | -0.1% |
| retest2 (combined) | 136 | 75 | 55.1% | 7 | 103 | 26 | 1.29% | 175.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 11:15:00 | 210.95 | 207.21 | 206.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 14:15:00 | 211.65 | 209.11 | 207.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 09:15:00 | 208.40 | 209.20 | 208.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 09:15:00 | 208.40 | 209.20 | 208.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 208.40 | 209.20 | 208.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:00:00 | 208.40 | 209.20 | 208.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 210.35 | 209.43 | 208.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-14 13:30:00 | 212.65 | 210.52 | 209.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 09:15:00 | 206.95 | 213.29 | 212.23 | SL hit (close<static) qty=1.00 sl=208.20 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 10:15:00 | 208.65 | 211.72 | 212.09 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 14:15:00 | 212.70 | 211.60 | 211.56 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 09:15:00 | 210.05 | 211.36 | 211.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 09:15:00 | 208.10 | 209.27 | 210.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 13:15:00 | 206.80 | 206.78 | 207.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-27 14:00:00 | 206.80 | 206.78 | 207.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 206.45 | 206.33 | 207.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 10:30:00 | 204.65 | 206.19 | 207.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 11:30:00 | 205.25 | 206.06 | 207.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 12:15:00 | 204.70 | 206.06 | 207.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 11:15:00 | 203.55 | 200.18 | 200.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 203.55 | 200.18 | 200.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 13:15:00 | 204.95 | 201.63 | 200.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 197.95 | 201.58 | 200.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 197.95 | 201.58 | 200.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 197.95 | 201.58 | 200.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 193.95 | 201.58 | 200.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 195.00 | 200.26 | 200.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 190.85 | 198.38 | 199.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 14:15:00 | 199.25 | 197.74 | 198.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 14:15:00 | 199.25 | 197.74 | 198.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 199.25 | 197.74 | 198.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 15:00:00 | 199.25 | 197.74 | 198.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 200.30 | 198.25 | 199.02 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 203.65 | 200.07 | 199.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 14:15:00 | 208.85 | 202.41 | 200.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 13:15:00 | 205.05 | 205.32 | 203.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 14:00:00 | 205.05 | 205.32 | 203.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 219.73 | 221.33 | 219.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:15:00 | 214.65 | 221.33 | 219.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 215.11 | 220.08 | 219.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:30:00 | 214.35 | 220.08 | 219.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 216.00 | 219.27 | 218.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 11:00:00 | 216.00 | 219.27 | 218.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 12:15:00 | 216.15 | 218.14 | 218.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 13:15:00 | 215.20 | 216.37 | 217.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 14:15:00 | 216.51 | 216.40 | 217.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-18 15:00:00 | 216.51 | 216.40 | 217.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 15:15:00 | 216.50 | 216.42 | 217.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:15:00 | 214.10 | 216.42 | 217.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 213.50 | 215.84 | 216.73 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 15:15:00 | 216.44 | 215.87 | 215.81 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 215.29 | 215.75 | 215.77 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 11:15:00 | 217.54 | 216.06 | 215.90 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 10:15:00 | 215.55 | 215.87 | 215.89 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 13:15:00 | 216.60 | 216.03 | 215.96 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 12:15:00 | 215.80 | 216.40 | 216.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 13:15:00 | 215.06 | 216.13 | 216.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 11:15:00 | 215.45 | 215.30 | 215.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-27 11:30:00 | 215.40 | 215.30 | 215.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 214.20 | 213.28 | 214.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:45:00 | 214.03 | 213.28 | 214.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 214.79 | 213.58 | 214.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:30:00 | 214.87 | 213.58 | 214.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 214.00 | 213.66 | 214.29 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 220.50 | 215.20 | 214.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 10:15:00 | 223.61 | 216.88 | 215.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 10:15:00 | 219.95 | 220.55 | 218.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 10:45:00 | 219.85 | 220.55 | 218.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 218.66 | 219.99 | 218.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 218.66 | 219.99 | 218.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 219.40 | 219.87 | 218.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 222.33 | 219.38 | 218.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 11:00:00 | 220.08 | 219.62 | 218.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 12:30:00 | 220.17 | 219.78 | 219.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 13:00:00 | 220.04 | 219.78 | 219.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 219.84 | 221.24 | 220.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 13:45:00 | 219.77 | 221.24 | 220.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 219.99 | 220.99 | 220.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 219.99 | 220.99 | 220.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 215.85 | 220.15 | 220.09 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-05 09:15:00 | 215.85 | 220.15 | 220.09 | SL hit (close<static) qty=1.00 sl=218.51 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 10:15:00 | 215.27 | 219.17 | 219.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 09:15:00 | 211.73 | 215.69 | 217.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 10:15:00 | 209.73 | 209.38 | 211.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-10 11:00:00 | 209.73 | 209.38 | 211.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 207.99 | 209.14 | 210.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:15:00 | 207.00 | 209.14 | 210.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 15:15:00 | 207.70 | 209.32 | 209.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 09:30:00 | 207.05 | 208.79 | 209.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 15:15:00 | 207.00 | 208.61 | 209.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 208.00 | 208.23 | 208.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 09:30:00 | 206.80 | 207.92 | 208.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 196.65 | 200.83 | 203.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 197.31 | 200.83 | 203.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 196.70 | 200.83 | 203.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 196.65 | 200.83 | 203.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 196.46 | 200.83 | 203.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-22 12:15:00 | 200.62 | 200.60 | 202.61 | SL hit (close>ema200) qty=0.50 sl=200.60 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 205.24 | 202.77 | 202.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 11:15:00 | 208.10 | 204.33 | 203.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 11:15:00 | 213.05 | 213.36 | 211.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 12:00:00 | 213.05 | 213.36 | 211.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 15:15:00 | 214.00 | 213.37 | 212.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 09:15:00 | 215.37 | 213.37 | 212.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 15:15:00 | 211.94 | 213.45 | 213.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 15:15:00 | 211.94 | 213.45 | 213.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 09:15:00 | 206.00 | 211.96 | 212.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 13:15:00 | 191.96 | 190.75 | 193.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 14:00:00 | 191.96 | 190.75 | 193.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 192.45 | 191.42 | 193.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:15:00 | 194.43 | 191.42 | 193.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 191.91 | 191.52 | 193.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 10:45:00 | 191.49 | 191.50 | 192.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 12:30:00 | 191.56 | 191.47 | 192.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 10:15:00 | 196.95 | 191.92 | 191.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 10:15:00 | 196.95 | 191.92 | 191.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 12:15:00 | 197.89 | 193.72 | 192.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 12:15:00 | 197.99 | 198.16 | 195.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 12:45:00 | 198.00 | 198.16 | 195.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 195.81 | 197.63 | 196.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 195.81 | 197.63 | 196.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 195.10 | 197.13 | 195.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 09:15:00 | 195.95 | 197.13 | 195.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 09:45:00 | 197.00 | 197.35 | 196.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 11:15:00 | 204.39 | 207.18 | 207.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 11:15:00 | 204.39 | 207.18 | 207.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 13:15:00 | 202.80 | 205.79 | 206.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 09:15:00 | 203.63 | 203.51 | 204.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 203.63 | 203.51 | 204.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 203.63 | 203.51 | 204.54 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 13:15:00 | 207.11 | 205.03 | 204.99 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 15:15:00 | 204.30 | 204.89 | 204.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 203.85 | 204.68 | 204.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 202.69 | 202.40 | 203.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 202.69 | 202.40 | 203.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 202.69 | 202.40 | 203.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:30:00 | 203.28 | 202.40 | 203.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 203.72 | 202.66 | 203.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 203.72 | 202.66 | 203.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 203.52 | 202.84 | 203.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:45:00 | 204.50 | 202.84 | 203.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 12:15:00 | 202.92 | 202.85 | 203.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 12:45:00 | 203.55 | 202.85 | 203.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 203.39 | 202.96 | 203.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 14:00:00 | 203.39 | 202.96 | 203.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 201.86 | 202.74 | 203.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 14:45:00 | 203.64 | 202.74 | 203.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 200.16 | 202.04 | 202.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 10:30:00 | 199.11 | 201.44 | 202.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 09:45:00 | 198.85 | 199.56 | 200.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 14:30:00 | 199.32 | 199.04 | 200.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 09:15:00 | 198.62 | 199.30 | 200.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 199.43 | 199.04 | 199.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 13:30:00 | 199.29 | 199.04 | 199.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 198.71 | 198.97 | 199.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 11:00:00 | 197.99 | 198.63 | 198.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 12:00:00 | 198.00 | 198.50 | 198.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 12:30:00 | 197.95 | 198.51 | 198.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 13:15:00 | 201.00 | 199.01 | 199.06 | SL hit (close>static) qty=1.00 sl=199.81 alert=retest2 |

### Cycle 23 — BUY (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 10:15:00 | 192.02 | 190.72 | 190.61 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 190.15 | 190.59 | 190.61 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 09:15:00 | 192.05 | 190.88 | 190.74 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 10:15:00 | 189.67 | 190.64 | 190.64 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 11:15:00 | 191.78 | 190.87 | 190.74 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 13:15:00 | 189.05 | 190.49 | 190.60 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 15:15:00 | 191.70 | 190.83 | 190.74 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 11:15:00 | 189.95 | 190.64 | 190.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-20 15:15:00 | 189.25 | 190.10 | 190.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 12:15:00 | 189.98 | 189.34 | 189.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 12:15:00 | 189.98 | 189.34 | 189.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 12:15:00 | 189.98 | 189.34 | 189.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 12:30:00 | 190.12 | 189.34 | 189.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 13:15:00 | 190.34 | 189.54 | 189.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 13:15:00 | 187.84 | 189.04 | 189.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 13:45:00 | 187.87 | 188.79 | 189.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 15:15:00 | 187.00 | 185.90 | 186.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 10:00:00 | 187.76 | 185.47 | 185.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 10:15:00 | 187.18 | 185.82 | 185.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 10:15:00 | 187.18 | 185.82 | 185.75 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 184.14 | 185.44 | 185.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 181.86 | 184.08 | 184.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 184.00 | 183.41 | 184.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 184.00 | 183.41 | 184.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 184.00 | 183.41 | 184.23 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2024-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-07 15:15:00 | 185.89 | 184.19 | 184.16 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 11:15:00 | 183.43 | 184.61 | 184.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 182.94 | 184.10 | 184.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 09:15:00 | 184.19 | 183.58 | 184.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 09:15:00 | 184.19 | 183.58 | 184.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 184.19 | 183.58 | 184.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:00:00 | 184.19 | 183.58 | 184.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 183.54 | 183.57 | 184.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:15:00 | 183.87 | 183.57 | 184.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 184.22 | 183.70 | 184.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 12:00:00 | 184.22 | 183.70 | 184.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 184.70 | 183.90 | 184.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 12:30:00 | 184.62 | 183.90 | 184.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 183.83 | 183.89 | 184.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 183.29 | 184.06 | 184.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 14:15:00 | 174.13 | 176.75 | 178.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-18 14:15:00 | 175.01 | 174.88 | 176.59 | SL hit (close>ema200) qty=0.50 sl=174.88 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 14:15:00 | 168.51 | 166.21 | 165.98 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 165.17 | 165.88 | 165.93 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 166.95 | 166.10 | 166.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 179.00 | 168.97 | 167.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 18:15:00 | 191.65 | 191.81 | 185.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:45:00 | 191.65 | 191.81 | 185.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 13:15:00 | 188.29 | 190.24 | 187.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 14:00:00 | 188.29 | 190.24 | 187.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 194.40 | 190.90 | 188.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 10:15:00 | 195.73 | 190.90 | 188.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 11:15:00 | 190.65 | 194.48 | 194.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 190.65 | 194.48 | 194.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 189.44 | 191.38 | 192.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 193.00 | 188.22 | 189.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 193.00 | 188.22 | 189.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 193.00 | 188.22 | 189.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 09:30:00 | 193.15 | 188.22 | 189.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 193.27 | 189.23 | 189.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 11:00:00 | 193.27 | 189.23 | 189.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2024-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 11:15:00 | 193.99 | 190.18 | 190.08 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 15:15:00 | 191.50 | 192.70 | 192.79 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 09:15:00 | 196.17 | 193.40 | 193.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 12:15:00 | 197.65 | 195.38 | 194.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 14:15:00 | 194.46 | 195.58 | 194.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 14:15:00 | 194.46 | 195.58 | 194.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 194.46 | 195.58 | 194.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 15:00:00 | 194.46 | 195.58 | 194.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 195.00 | 195.47 | 194.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 09:15:00 | 196.39 | 195.47 | 194.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 11:15:00 | 193.48 | 194.97 | 194.54 | SL hit (close<static) qty=1.00 sl=194.20 alert=retest2 |

### Cycle 42 — SELL (started 2024-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 13:15:00 | 192.28 | 193.95 | 194.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-25 14:15:00 | 190.96 | 193.35 | 193.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 09:15:00 | 194.72 | 193.16 | 193.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 09:15:00 | 194.72 | 193.16 | 193.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 194.72 | 193.16 | 193.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:30:00 | 195.44 | 193.16 | 193.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 10:15:00 | 197.13 | 193.95 | 193.95 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 09:15:00 | 193.51 | 196.12 | 196.25 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 13:15:00 | 197.30 | 196.40 | 196.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 202.17 | 197.77 | 196.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 203.39 | 205.27 | 203.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 09:15:00 | 203.39 | 205.27 | 203.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 203.39 | 205.27 | 203.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:00:00 | 203.39 | 205.27 | 203.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 204.18 | 205.05 | 203.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 11:15:00 | 204.24 | 205.05 | 203.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 09:30:00 | 204.68 | 205.02 | 204.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 10:45:00 | 204.28 | 204.85 | 204.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 11:30:00 | 204.25 | 204.71 | 204.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 12:15:00 | 204.74 | 204.71 | 204.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 12:30:00 | 204.76 | 204.71 | 204.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 205.25 | 205.60 | 204.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:30:00 | 204.88 | 205.60 | 204.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 206.98 | 205.87 | 205.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 11:45:00 | 207.53 | 206.02 | 205.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 12:45:00 | 207.58 | 206.31 | 205.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 209.08 | 206.71 | 205.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 10:30:00 | 207.52 | 206.99 | 206.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 206.06 | 207.12 | 206.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 15:00:00 | 206.06 | 207.12 | 206.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 206.60 | 207.02 | 206.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 09:15:00 | 211.00 | 207.02 | 206.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 10:15:00 | 208.49 | 209.71 | 209.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 208.49 | 209.71 | 209.78 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 15:15:00 | 210.90 | 209.96 | 209.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 09:15:00 | 214.00 | 210.77 | 210.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 09:15:00 | 212.35 | 212.63 | 211.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 10:00:00 | 212.35 | 212.63 | 211.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 210.50 | 212.21 | 211.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:30:00 | 210.92 | 212.21 | 211.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 211.59 | 212.08 | 211.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 11:30:00 | 209.77 | 212.08 | 211.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 210.97 | 211.86 | 211.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:45:00 | 211.05 | 211.86 | 211.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 209.26 | 211.34 | 211.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:00:00 | 209.26 | 211.34 | 211.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2024-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 14:15:00 | 209.37 | 210.95 | 211.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 09:15:00 | 207.90 | 210.10 | 210.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 10:15:00 | 208.70 | 208.23 | 209.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 10:45:00 | 208.39 | 208.23 | 209.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 209.18 | 208.42 | 209.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:30:00 | 208.99 | 208.42 | 209.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 208.94 | 208.53 | 209.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 12:30:00 | 209.00 | 208.53 | 209.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 207.63 | 208.35 | 209.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 14:15:00 | 207.32 | 208.35 | 209.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 15:00:00 | 207.24 | 208.13 | 208.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 10:00:00 | 207.08 | 207.90 | 208.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 13:00:00 | 206.58 | 207.77 | 208.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 206.03 | 207.21 | 208.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 14:30:00 | 208.37 | 207.21 | 208.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 207.48 | 207.26 | 207.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 10:00:00 | 207.40 | 207.29 | 207.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 206.02 | 206.96 | 207.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:30:00 | 206.67 | 206.96 | 207.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 205.78 | 206.17 | 206.93 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-24 14:15:00 | 208.40 | 207.23 | 207.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 14:15:00 | 208.40 | 207.23 | 207.21 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 14:15:00 | 207.11 | 207.26 | 207.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 10:15:00 | 206.09 | 206.84 | 207.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 12:15:00 | 197.34 | 196.30 | 199.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 13:00:00 | 197.34 | 196.30 | 199.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 200.49 | 197.14 | 199.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:15:00 | 206.27 | 197.14 | 199.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 199.18 | 197.55 | 199.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:30:00 | 208.35 | 197.55 | 199.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 199.60 | 197.96 | 199.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 09:30:00 | 198.08 | 197.96 | 199.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 10:00:00 | 197.98 | 197.96 | 199.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 09:15:00 | 202.47 | 199.68 | 199.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 09:15:00 | 202.47 | 199.68 | 199.57 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 12:15:00 | 199.01 | 199.50 | 199.52 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 14:15:00 | 200.68 | 199.65 | 199.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 202.00 | 200.29 | 199.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 201.39 | 201.71 | 200.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 15:00:00 | 201.39 | 201.71 | 200.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 201.00 | 201.57 | 200.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 199.47 | 201.57 | 200.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 199.75 | 201.21 | 200.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:15:00 | 199.68 | 201.21 | 200.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 196.89 | 200.34 | 200.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 194.80 | 198.25 | 199.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 201.00 | 198.44 | 199.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 201.00 | 198.44 | 199.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 201.00 | 198.44 | 199.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 201.32 | 198.44 | 199.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 200.90 | 198.93 | 199.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:45:00 | 201.60 | 198.93 | 199.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 12:15:00 | 201.59 | 199.63 | 199.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 13:15:00 | 202.63 | 200.23 | 199.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 11:15:00 | 201.31 | 202.41 | 201.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 11:15:00 | 201.31 | 202.41 | 201.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 11:15:00 | 201.31 | 202.41 | 201.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 12:00:00 | 201.31 | 202.41 | 201.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 12:15:00 | 201.38 | 202.20 | 201.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 12:45:00 | 201.10 | 202.20 | 201.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 202.02 | 202.17 | 201.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 14:30:00 | 202.59 | 202.08 | 201.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 204.56 | 202.09 | 201.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 202.70 | 203.01 | 202.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 10:00:00 | 202.74 | 202.96 | 202.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 205.44 | 203.46 | 202.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:30:00 | 202.36 | 203.46 | 202.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 13:15:00 | 204.00 | 204.18 | 203.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 14:00:00 | 204.00 | 204.18 | 203.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 14:15:00 | 204.01 | 204.15 | 203.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 14:30:00 | 204.37 | 204.15 | 203.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 205.54 | 204.45 | 203.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-14 14:15:00 | 201.41 | 204.08 | 204.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 14:15:00 | 201.41 | 204.08 | 204.17 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 09:15:00 | 208.03 | 204.49 | 204.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 12:15:00 | 209.37 | 207.05 | 205.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 12:15:00 | 210.00 | 211.23 | 209.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 12:15:00 | 210.00 | 211.23 | 209.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 12:15:00 | 210.00 | 211.23 | 209.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:00:00 | 210.00 | 211.23 | 209.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 209.09 | 210.80 | 209.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:00:00 | 209.09 | 210.80 | 209.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 209.01 | 210.44 | 209.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 09:15:00 | 224.22 | 210.27 | 209.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 13:15:00 | 220.95 | 222.68 | 222.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 13:15:00 | 220.95 | 222.68 | 222.78 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-27 14:15:00 | 224.81 | 223.10 | 222.96 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 09:15:00 | 207.22 | 219.85 | 221.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 13:15:00 | 201.24 | 204.25 | 205.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 204.38 | 204.03 | 205.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 204.38 | 204.03 | 205.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 204.38 | 204.03 | 205.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 11:15:00 | 203.94 | 204.13 | 205.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 12:00:00 | 203.63 | 204.03 | 205.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 11:15:00 | 207.14 | 205.00 | 204.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 11:15:00 | 207.14 | 205.00 | 204.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 13:15:00 | 210.40 | 206.51 | 205.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 09:15:00 | 234.67 | 237.11 | 230.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-11 09:30:00 | 234.40 | 237.11 | 230.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 12:15:00 | 230.10 | 234.57 | 231.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 13:00:00 | 230.10 | 234.57 | 231.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 13:15:00 | 231.82 | 234.02 | 231.25 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 10:15:00 | 223.36 | 229.92 | 230.07 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 10:15:00 | 238.60 | 231.00 | 230.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 11:15:00 | 244.30 | 233.66 | 231.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-17 10:15:00 | 239.17 | 242.41 | 239.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 10:15:00 | 239.17 | 242.41 | 239.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 239.17 | 242.41 | 239.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-17 11:00:00 | 239.17 | 242.41 | 239.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 237.68 | 241.47 | 239.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-17 11:30:00 | 238.19 | 241.47 | 239.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 234.73 | 240.12 | 238.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:00:00 | 234.73 | 240.12 | 238.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 237.80 | 239.65 | 238.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-17 14:45:00 | 241.05 | 240.43 | 239.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-18 10:15:00 | 233.26 | 238.41 | 238.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 10:15:00 | 233.26 | 238.41 | 238.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 11:15:00 | 231.67 | 237.06 | 238.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 09:15:00 | 236.93 | 232.65 | 235.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 09:15:00 | 236.93 | 232.65 | 235.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 236.93 | 232.65 | 235.01 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 12:15:00 | 244.06 | 237.78 | 237.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 248.55 | 241.09 | 238.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 10:15:00 | 250.63 | 251.06 | 248.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 11:00:00 | 250.63 | 251.06 | 248.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 246.90 | 250.27 | 249.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:15:00 | 246.00 | 250.27 | 249.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 250.90 | 250.39 | 249.27 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 246.10 | 248.49 | 248.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 234.20 | 245.52 | 247.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 229.81 | 228.99 | 234.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 229.81 | 228.99 | 234.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 233.30 | 230.32 | 234.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 229.11 | 230.32 | 234.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 10:15:00 | 229.93 | 230.42 | 233.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:30:00 | 230.15 | 229.80 | 232.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 249.09 | 233.16 | 233.18 | SL hit (close>static) qty=1.00 sl=235.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 248.31 | 236.19 | 234.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 15:15:00 | 249.67 | 246.37 | 242.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 245.43 | 249.66 | 247.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 245.43 | 249.66 | 247.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 245.43 | 249.66 | 247.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 245.43 | 249.66 | 247.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 246.30 | 248.99 | 247.09 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 239.19 | 245.03 | 245.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 09:15:00 | 232.35 | 237.94 | 241.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 15:15:00 | 234.90 | 234.76 | 237.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 09:15:00 | 230.07 | 234.76 | 237.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 227.65 | 226.43 | 228.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:30:00 | 228.05 | 226.43 | 228.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 225.30 | 226.51 | 227.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-19 11:45:00 | 223.96 | 225.69 | 227.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-20 14:15:00 | 227.92 | 227.21 | 227.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 14:15:00 | 227.92 | 227.21 | 227.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 09:15:00 | 240.01 | 229.77 | 228.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 11:15:00 | 243.99 | 244.75 | 239.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 12:00:00 | 243.99 | 244.75 | 239.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 240.39 | 243.45 | 240.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 14:15:00 | 247.35 | 241.96 | 240.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 14:15:00 | 238.27 | 241.04 | 241.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 238.27 | 241.04 | 241.11 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 243.81 | 241.11 | 240.99 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 10:15:00 | 239.27 | 240.82 | 240.94 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 15:15:00 | 243.00 | 241.28 | 241.07 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 09:15:00 | 235.36 | 240.10 | 240.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 233.94 | 238.87 | 239.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 15:15:00 | 229.00 | 228.91 | 230.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-04 09:15:00 | 224.84 | 228.91 | 230.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 12:15:00 | 207.48 | 203.64 | 206.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 12:45:00 | 208.89 | 203.64 | 206.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 13:15:00 | 204.68 | 203.85 | 206.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 14:45:00 | 203.61 | 204.04 | 206.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 213.87 | 206.03 | 206.61 | SL hit (close>static) qty=1.00 sl=208.75 alert=retest2 |

### Cycle 75 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 212.96 | 208.11 | 207.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 215.08 | 209.50 | 208.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 224.30 | 224.90 | 221.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 15:15:00 | 223.69 | 224.71 | 223.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 223.69 | 224.71 | 223.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:15:00 | 225.13 | 224.71 | 223.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 226.59 | 225.09 | 223.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 11:00:00 | 227.63 | 225.60 | 223.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 11:30:00 | 229.66 | 226.42 | 224.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 230.99 | 226.70 | 226.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-25 09:15:00 | 250.39 | 243.15 | 238.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 246.49 | 249.96 | 250.23 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 252.70 | 250.56 | 250.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 254.26 | 251.30 | 250.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 251.12 | 252.38 | 251.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 251.12 | 252.38 | 251.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 251.12 | 252.38 | 251.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 251.12 | 252.38 | 251.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 248.63 | 251.63 | 251.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 248.63 | 251.63 | 251.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 246.80 | 250.67 | 250.82 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 10:15:00 | 254.40 | 251.53 | 251.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-09 14:15:00 | 258.00 | 254.01 | 252.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 277.12 | 278.21 | 275.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 10:00:00 | 277.12 | 278.21 | 275.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 279.44 | 280.53 | 279.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 09:15:00 | 292.42 | 280.53 | 279.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 11:15:00 | 276.99 | 283.57 | 283.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 11:15:00 | 276.99 | 283.57 | 283.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 275.85 | 282.03 | 283.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 09:15:00 | 265.52 | 264.42 | 267.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 265.52 | 264.42 | 267.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 265.52 | 264.42 | 267.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:30:00 | 267.56 | 264.42 | 267.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 269.22 | 265.38 | 268.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:00:00 | 269.22 | 265.38 | 268.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 269.84 | 266.27 | 268.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:30:00 | 269.52 | 266.27 | 268.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 267.40 | 266.50 | 268.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 11:00:00 | 266.10 | 266.78 | 267.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 12:45:00 | 266.46 | 266.41 | 267.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 13:15:00 | 265.98 | 266.41 | 267.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 12:15:00 | 269.50 | 265.06 | 264.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 12:15:00 | 269.50 | 265.06 | 264.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 14:15:00 | 270.10 | 266.67 | 265.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 268.60 | 270.36 | 268.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 14:15:00 | 268.60 | 270.36 | 268.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 268.60 | 270.36 | 268.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 268.60 | 270.36 | 268.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 268.65 | 270.02 | 268.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 273.25 | 270.02 | 268.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 11:45:00 | 270.05 | 269.90 | 268.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 12:30:00 | 270.05 | 270.11 | 268.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 13:00:00 | 270.95 | 270.11 | 268.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 268.50 | 269.79 | 268.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:30:00 | 267.85 | 269.79 | 268.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 267.70 | 269.37 | 268.83 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-04 14:15:00 | 267.70 | 269.37 | 268.83 | SL hit (close<static) qty=1.00 sl=267.80 alert=retest2 |

### Cycle 82 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 11:15:00 | 293.00 | 294.63 | 294.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 12:15:00 | 291.55 | 294.01 | 294.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 13:15:00 | 294.05 | 294.02 | 294.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 13:15:00 | 294.05 | 294.02 | 294.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 294.05 | 294.02 | 294.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 14:00:00 | 294.05 | 294.02 | 294.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 294.95 | 294.21 | 294.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 15:00:00 | 294.95 | 294.21 | 294.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 294.45 | 294.25 | 294.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 301.80 | 294.25 | 294.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 09:15:00 | 302.25 | 295.85 | 295.11 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 293.85 | 296.66 | 296.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 292.95 | 295.91 | 296.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 301.95 | 292.53 | 293.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 301.95 | 292.53 | 293.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 301.95 | 292.53 | 293.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 301.95 | 292.53 | 293.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 300.15 | 294.06 | 293.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 305.70 | 299.27 | 296.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 301.35 | 304.00 | 301.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 301.35 | 304.00 | 301.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 301.35 | 304.00 | 301.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:45:00 | 301.40 | 304.00 | 301.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 297.70 | 302.74 | 300.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 297.70 | 302.74 | 300.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 297.40 | 301.67 | 300.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:15:00 | 298.45 | 301.67 | 300.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:00:00 | 298.55 | 300.47 | 300.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 14:15:00 | 297.15 | 299.80 | 299.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 14:15:00 | 297.15 | 299.80 | 299.83 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 10:15:00 | 302.45 | 300.11 | 299.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 12:15:00 | 304.25 | 300.85 | 300.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 315.20 | 325.70 | 321.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 09:15:00 | 315.20 | 325.70 | 321.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 315.20 | 325.70 | 321.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 315.20 | 325.70 | 321.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 316.20 | 323.80 | 321.31 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 314.25 | 319.50 | 319.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 14:15:00 | 312.20 | 318.04 | 319.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 311.65 | 311.57 | 313.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 15:00:00 | 311.65 | 311.57 | 313.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 312.60 | 311.92 | 313.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:15:00 | 309.45 | 311.86 | 312.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 306.40 | 301.70 | 301.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 306.40 | 301.70 | 301.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 14:15:00 | 311.15 | 308.71 | 307.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 09:15:00 | 303.70 | 307.90 | 307.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 303.70 | 307.90 | 307.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 303.70 | 307.90 | 307.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 302.40 | 307.90 | 307.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 303.70 | 307.06 | 307.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 13:15:00 | 301.70 | 304.83 | 306.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 306.55 | 304.81 | 305.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 306.55 | 304.81 | 305.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 306.55 | 304.81 | 305.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:45:00 | 307.00 | 304.81 | 305.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 306.30 | 305.11 | 305.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:15:00 | 307.25 | 305.11 | 305.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 308.00 | 305.69 | 306.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:00:00 | 308.00 | 305.69 | 306.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 13:15:00 | 308.75 | 306.71 | 306.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 14:15:00 | 311.35 | 307.64 | 306.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 11:15:00 | 308.30 | 308.64 | 307.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 11:15:00 | 308.30 | 308.64 | 307.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 308.30 | 308.64 | 307.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:45:00 | 307.60 | 308.64 | 307.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 304.40 | 308.58 | 308.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 304.40 | 308.58 | 308.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 303.90 | 307.64 | 307.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 15:15:00 | 303.00 | 304.80 | 306.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 15:15:00 | 288.30 | 287.59 | 290.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-30 09:15:00 | 283.50 | 287.59 | 290.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 283.40 | 286.75 | 290.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:15:00 | 267.85 | 284.18 | 288.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 09:15:00 | 254.46 | 271.05 | 280.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-04 09:15:00 | 241.07 | 251.80 | 259.34 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 93 — BUY (started 2025-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 15:15:00 | 241.00 | 238.76 | 238.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 10:15:00 | 242.45 | 239.94 | 239.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 244.80 | 245.04 | 243.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 11:30:00 | 245.30 | 245.04 | 243.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 243.60 | 244.76 | 243.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 243.60 | 244.76 | 243.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 242.10 | 244.22 | 243.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:00:00 | 242.10 | 244.22 | 243.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 242.20 | 243.82 | 243.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:30:00 | 242.15 | 243.82 | 243.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 243.75 | 243.46 | 243.22 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 11:15:00 | 241.90 | 243.04 | 243.16 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 13:15:00 | 244.30 | 243.39 | 243.30 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 239.40 | 242.54 | 242.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 11:15:00 | 237.10 | 241.45 | 242.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 13:15:00 | 236.35 | 236.01 | 238.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 14:00:00 | 236.35 | 236.01 | 238.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 240.55 | 236.60 | 237.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 10:45:00 | 240.35 | 236.60 | 237.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 241.20 | 237.52 | 238.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 241.50 | 237.52 | 238.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 13:15:00 | 240.95 | 238.77 | 238.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 241.95 | 240.42 | 239.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 12:15:00 | 243.07 | 243.24 | 241.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 13:00:00 | 243.07 | 243.24 | 241.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 241.81 | 242.95 | 241.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 241.81 | 242.95 | 241.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 241.20 | 242.60 | 241.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 241.20 | 242.60 | 241.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 240.55 | 242.19 | 241.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 241.07 | 242.19 | 241.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 240.50 | 241.85 | 241.50 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 12:15:00 | 238.47 | 240.74 | 241.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 13:15:00 | 236.49 | 239.89 | 240.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 09:15:00 | 240.50 | 239.35 | 240.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 240.50 | 239.35 | 240.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 240.50 | 239.35 | 240.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:45:00 | 241.69 | 239.35 | 240.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 240.08 | 239.49 | 240.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:15:00 | 240.27 | 239.49 | 240.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 241.00 | 239.80 | 240.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:45:00 | 241.24 | 239.80 | 240.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 242.84 | 240.40 | 240.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 242.84 | 240.40 | 240.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 13:15:00 | 242.05 | 240.73 | 240.59 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 238.80 | 240.61 | 240.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 13:15:00 | 237.80 | 240.05 | 240.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 241.32 | 239.52 | 240.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 241.32 | 239.52 | 240.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 241.32 | 239.52 | 240.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 244.73 | 239.52 | 240.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 243.01 | 240.22 | 240.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:30:00 | 243.46 | 240.22 | 240.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 242.46 | 240.67 | 240.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 15:15:00 | 243.49 | 241.92 | 241.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 13:15:00 | 243.93 | 244.21 | 243.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 13:30:00 | 243.66 | 244.21 | 243.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 244.00 | 244.17 | 243.14 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 14:15:00 | 242.00 | 242.82 | 242.87 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 244.58 | 243.08 | 242.98 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 10:15:00 | 241.25 | 242.99 | 243.15 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 266.00 | 246.92 | 244.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 10:15:00 | 275.70 | 252.68 | 247.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 287.20 | 287.28 | 279.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 12:30:00 | 287.04 | 287.28 | 279.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 293.95 | 296.86 | 293.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:45:00 | 293.81 | 296.86 | 293.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 292.06 | 295.90 | 293.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:45:00 | 292.41 | 295.90 | 293.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 290.05 | 294.73 | 292.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 15:00:00 | 290.05 | 294.73 | 292.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 285.07 | 290.57 | 291.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 280.34 | 285.82 | 288.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 270.51 | 266.62 | 270.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 270.51 | 266.62 | 270.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 270.51 | 266.62 | 270.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 264.67 | 266.18 | 269.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 293.01 | 273.40 | 272.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 14:15:00 | 293.01 | 273.40 | 272.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 15:15:00 | 295.00 | 277.72 | 274.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 09:15:00 | 279.25 | 281.89 | 279.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 279.25 | 281.89 | 279.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 279.25 | 281.89 | 279.02 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 10:15:00 | 276.15 | 278.43 | 278.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 14:15:00 | 275.15 | 276.53 | 277.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 277.75 | 276.51 | 277.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 277.75 | 276.51 | 277.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 277.75 | 276.51 | 277.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:30:00 | 277.15 | 276.51 | 277.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 277.85 | 276.78 | 277.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:45:00 | 277.20 | 276.78 | 277.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 276.75 | 276.77 | 277.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:45:00 | 277.45 | 276.77 | 277.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 277.30 | 276.88 | 277.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:45:00 | 277.35 | 276.88 | 277.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 276.80 | 276.86 | 277.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:30:00 | 277.95 | 276.86 | 277.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 274.00 | 276.29 | 276.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 15:15:00 | 273.95 | 276.29 | 276.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 285.00 | 272.82 | 273.59 | SL hit (close>static) qty=1.00 sl=277.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 286.55 | 275.57 | 274.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 291.25 | 280.68 | 277.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 280.40 | 284.02 | 280.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 280.40 | 284.02 | 280.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 280.40 | 284.02 | 280.37 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 274.00 | 279.42 | 279.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 271.65 | 277.87 | 278.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 273.70 | 273.59 | 275.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 273.70 | 273.59 | 275.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 273.70 | 273.59 | 275.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:15:00 | 272.70 | 275.22 | 275.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 12:15:00 | 272.60 | 274.44 | 275.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:30:00 | 272.20 | 273.66 | 274.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 11:15:00 | 259.06 | 261.43 | 262.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 12:15:00 | 258.97 | 260.88 | 261.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 12:15:00 | 258.59 | 260.88 | 261.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 256.30 | 255.86 | 257.90 | SL hit (close>ema200) qty=0.50 sl=255.86 alert=retest2 |

### Cycle 111 — BUY (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 09:15:00 | 277.45 | 256.63 | 255.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 10:15:00 | 282.30 | 261.77 | 258.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 11:15:00 | 289.70 | 290.40 | 282.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 12:00:00 | 289.70 | 290.40 | 282.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 291.15 | 294.93 | 294.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 291.15 | 294.93 | 294.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 293.80 | 294.70 | 294.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 14:30:00 | 294.30 | 294.77 | 294.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:00:00 | 295.05 | 294.77 | 294.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 289.90 | 294.80 | 294.76 | SL hit (close<static) qty=1.00 sl=290.50 alert=retest2 |

### Cycle 112 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 290.50 | 293.94 | 294.38 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 297.60 | 294.28 | 293.83 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 286.80 | 293.51 | 293.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 285.30 | 289.62 | 291.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 290.60 | 288.97 | 290.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 290.60 | 288.97 | 290.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 290.60 | 288.97 | 290.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:30:00 | 297.10 | 288.97 | 290.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 290.20 | 289.21 | 290.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 290.20 | 289.21 | 290.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 293.50 | 290.07 | 291.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:30:00 | 295.60 | 290.07 | 291.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 290.60 | 290.18 | 291.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 13:15:00 | 289.75 | 290.18 | 291.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:15:00 | 289.80 | 290.44 | 291.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:15:00 | 289.20 | 290.63 | 291.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:30:00 | 288.65 | 289.50 | 290.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 288.65 | 289.11 | 289.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:15:00 | 285.75 | 287.84 | 288.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 12:15:00 | 275.26 | 278.51 | 280.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 12:15:00 | 275.31 | 278.51 | 280.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 15:15:00 | 274.74 | 277.34 | 279.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 15:15:00 | 274.22 | 277.34 | 279.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 279.60 | 277.80 | 279.49 | SL hit (close>ema200) qty=0.50 sl=277.80 alert=retest2 |

### Cycle 115 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 276.40 | 270.84 | 270.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 12:15:00 | 278.00 | 275.19 | 273.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 15:15:00 | 281.00 | 281.09 | 279.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:15:00 | 276.60 | 281.09 | 279.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 275.75 | 280.02 | 279.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 275.75 | 280.02 | 279.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 275.15 | 279.04 | 278.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 275.15 | 279.04 | 278.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 275.00 | 278.24 | 278.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 272.60 | 277.11 | 277.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 275.35 | 274.71 | 276.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 10:00:00 | 275.35 | 274.71 | 276.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 271.95 | 269.86 | 271.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 271.95 | 269.86 | 271.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 272.35 | 270.36 | 271.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 272.45 | 270.36 | 271.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 272.70 | 270.83 | 271.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 274.45 | 270.83 | 271.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 276.85 | 272.03 | 272.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:00:00 | 276.85 | 272.03 | 272.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 275.55 | 272.74 | 272.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 10:15:00 | 277.70 | 275.19 | 274.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 11:15:00 | 274.95 | 275.14 | 274.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 12:00:00 | 274.95 | 275.14 | 274.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 273.85 | 274.88 | 274.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:00:00 | 273.85 | 274.88 | 274.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 274.20 | 274.75 | 274.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 273.75 | 274.75 | 274.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 274.10 | 274.62 | 274.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:30:00 | 273.95 | 274.62 | 274.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 274.65 | 274.62 | 274.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 274.05 | 274.62 | 274.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 273.65 | 274.43 | 274.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:15:00 | 273.65 | 274.43 | 274.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 273.20 | 274.18 | 274.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:15:00 | 273.00 | 274.18 | 274.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 272.35 | 273.82 | 274.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 13:15:00 | 271.65 | 273.09 | 273.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 274.40 | 267.37 | 269.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 13:15:00 | 274.40 | 267.37 | 269.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 274.40 | 267.37 | 269.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 274.40 | 267.37 | 269.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 267.75 | 267.45 | 268.97 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 273.30 | 270.15 | 269.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 274.70 | 271.97 | 270.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 15:15:00 | 277.20 | 278.15 | 276.61 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 09:15:00 | 285.60 | 278.15 | 276.61 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 284.15 | 286.06 | 284.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 15:00:00 | 284.15 | 286.06 | 284.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 285.10 | 285.86 | 284.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 294.35 | 285.86 | 284.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 285.20 | 287.12 | 285.34 | SL hit (close<ema400) qty=1.00 sl=285.34 alert=retest1 |

### Cycle 120 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 279.95 | 283.69 | 284.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 275.25 | 282.00 | 283.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 11:15:00 | 272.60 | 271.26 | 274.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 12:00:00 | 272.60 | 271.26 | 274.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 273.70 | 271.75 | 274.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:30:00 | 273.95 | 271.75 | 274.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 273.95 | 272.59 | 273.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 268.70 | 273.77 | 274.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 10:15:00 | 255.26 | 259.96 | 261.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 252.90 | 251.70 | 254.46 | SL hit (close>ema200) qty=0.50 sl=251.70 alert=retest2 |

### Cycle 121 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 261.00 | 256.17 | 255.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 11:15:00 | 261.85 | 257.68 | 256.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 266.75 | 267.87 | 264.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 266.75 | 267.87 | 264.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 266.75 | 267.87 | 264.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 266.00 | 267.87 | 264.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 266.00 | 267.08 | 265.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 266.00 | 267.08 | 265.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 262.50 | 266.16 | 265.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 262.50 | 266.16 | 265.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 263.00 | 265.53 | 264.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:00:00 | 267.15 | 265.41 | 264.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 12:15:00 | 268.00 | 272.42 | 272.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 12:15:00 | 268.00 | 272.42 | 272.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 264.60 | 268.78 | 270.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 268.60 | 268.06 | 269.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 14:00:00 | 268.60 | 268.06 | 269.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 270.25 | 268.70 | 269.67 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 272.15 | 270.49 | 270.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 274.40 | 271.33 | 270.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 272.65 | 273.24 | 271.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:00:00 | 272.65 | 273.24 | 271.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 270.35 | 272.66 | 271.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 270.35 | 272.66 | 271.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 270.70 | 272.27 | 271.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 273.00 | 272.27 | 271.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:45:00 | 271.70 | 271.68 | 271.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:45:00 | 271.50 | 271.71 | 271.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 12:15:00 | 270.15 | 271.40 | 271.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 270.15 | 271.40 | 271.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 267.35 | 269.70 | 270.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 10:15:00 | 259.45 | 258.40 | 261.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:30:00 | 260.00 | 258.40 | 261.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 259.75 | 258.83 | 260.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:45:00 | 259.75 | 258.83 | 260.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 260.90 | 259.43 | 260.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 260.90 | 259.43 | 260.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 261.00 | 259.75 | 260.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:15:00 | 259.40 | 259.75 | 260.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 12:15:00 | 246.43 | 249.88 | 252.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 244.10 | 242.46 | 245.79 | SL hit (close>ema200) qty=0.50 sl=242.46 alert=retest2 |

### Cycle 125 — BUY (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 09:15:00 | 270.15 | 248.41 | 245.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 10:15:00 | 274.00 | 253.53 | 248.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 268.80 | 270.83 | 261.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 268.80 | 270.83 | 261.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 268.80 | 270.83 | 261.30 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2026-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 11:15:00 | 248.65 | 258.36 | 259.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 11:15:00 | 246.80 | 250.68 | 254.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 15:15:00 | 250.00 | 249.13 | 252.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 09:15:00 | 251.60 | 249.13 | 252.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 249.20 | 249.14 | 252.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:45:00 | 247.35 | 248.84 | 251.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:15:00 | 247.60 | 248.39 | 250.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:30:00 | 247.30 | 247.98 | 250.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 234.98 | 245.01 | 248.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 235.22 | 245.01 | 248.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 234.94 | 245.01 | 248.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 261.85 | 240.58 | 243.03 | SL hit (close>ema200) qty=0.50 sl=240.58 alert=retest2 |

### Cycle 127 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 262.60 | 244.98 | 244.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 11:15:00 | 266.25 | 249.24 | 246.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 09:15:00 | 251.10 | 254.22 | 250.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 09:15:00 | 251.10 | 254.22 | 250.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 251.10 | 254.22 | 250.75 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 242.50 | 248.41 | 249.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 235.30 | 239.87 | 243.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 233.05 | 232.43 | 236.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 233.50 | 232.43 | 236.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 229.90 | 229.09 | 231.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 224.60 | 230.25 | 231.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 213.37 | 216.45 | 220.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 209.85 | 208.58 | 212.53 | SL hit (close>ema200) qty=0.50 sl=208.58 alert=retest2 |

### Cycle 129 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 205.79 | 201.20 | 201.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 208.25 | 202.61 | 201.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 14:15:00 | 218.39 | 218.61 | 215.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 15:00:00 | 218.39 | 218.61 | 215.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 218.40 | 220.52 | 218.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:30:00 | 218.83 | 220.32 | 218.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:45:00 | 218.82 | 219.61 | 218.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 223.51 | 219.16 | 218.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 09:15:00 | 240.71 | 227.88 | 224.96 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 15:15:00 | 228.00 | 228.50 | 228.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 226.20 | 228.04 | 228.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 221.01 | 218.26 | 220.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 221.01 | 218.26 | 220.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 221.01 | 218.26 | 220.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 217.50 | 219.07 | 219.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:45:00 | 216.58 | 218.48 | 219.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 221.17 | 216.63 | 216.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 221.17 | 216.63 | 216.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 223.34 | 220.64 | 219.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 222.36 | 223.30 | 221.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 10:00:00 | 222.36 | 223.30 | 221.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 221.80 | 223.00 | 221.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:30:00 | 222.00 | 223.00 | 221.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 222.96 | 222.99 | 221.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:30:00 | 222.26 | 222.99 | 221.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-14 13:30:00 | 212.65 | 2024-05-16 09:15:00 | 206.95 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2024-05-16 11:15:00 | 212.65 | 2024-05-17 09:15:00 | 208.05 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-05-28 10:30:00 | 204.65 | 2024-06-03 11:15:00 | 203.55 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2024-05-28 11:30:00 | 205.25 | 2024-06-03 11:15:00 | 203.55 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2024-05-28 12:15:00 | 204.70 | 2024-06-03 11:15:00 | 203.55 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2024-07-03 09:15:00 | 222.33 | 2024-07-05 09:15:00 | 215.85 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest2 | 2024-07-03 11:00:00 | 220.08 | 2024-07-05 09:15:00 | 215.85 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-07-03 12:30:00 | 220.17 | 2024-07-05 09:15:00 | 215.85 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-07-03 13:00:00 | 220.04 | 2024-07-05 09:15:00 | 215.85 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-07-11 10:15:00 | 207.00 | 2024-07-22 09:15:00 | 196.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 15:15:00 | 207.70 | 2024-07-22 09:15:00 | 197.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-15 09:30:00 | 207.05 | 2024-07-22 09:15:00 | 196.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-15 15:15:00 | 207.00 | 2024-07-22 09:15:00 | 196.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-18 09:30:00 | 206.80 | 2024-07-22 09:15:00 | 196.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-11 10:15:00 | 207.00 | 2024-07-22 12:15:00 | 200.62 | STOP_HIT | 0.50 | 3.08% |
| SELL | retest2 | 2024-07-12 15:15:00 | 207.70 | 2024-07-22 12:15:00 | 200.62 | STOP_HIT | 0.50 | 3.41% |
| SELL | retest2 | 2024-07-15 09:30:00 | 207.05 | 2024-07-22 12:15:00 | 200.62 | STOP_HIT | 0.50 | 3.11% |
| SELL | retest2 | 2024-07-15 15:15:00 | 207.00 | 2024-07-22 12:15:00 | 200.62 | STOP_HIT | 0.50 | 3.08% |
| SELL | retest2 | 2024-07-18 09:30:00 | 206.80 | 2024-07-22 12:15:00 | 200.62 | STOP_HIT | 0.50 | 2.99% |
| BUY | retest2 | 2024-07-30 09:15:00 | 215.37 | 2024-07-31 15:15:00 | 211.94 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-08-08 10:45:00 | 191.49 | 2024-08-12 10:15:00 | 196.95 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2024-08-08 12:30:00 | 191.56 | 2024-08-12 10:15:00 | 196.95 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2024-08-14 09:15:00 | 195.95 | 2024-08-26 11:15:00 | 204.39 | STOP_HIT | 1.00 | 4.31% |
| BUY | retest2 | 2024-08-14 09:45:00 | 197.00 | 2024-08-26 11:15:00 | 204.39 | STOP_HIT | 1.00 | 3.75% |
| SELL | retest2 | 2024-09-02 10:30:00 | 199.11 | 2024-09-06 13:15:00 | 201.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-09-03 09:45:00 | 198.85 | 2024-09-06 13:15:00 | 201.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-09-03 14:30:00 | 199.32 | 2024-09-06 13:15:00 | 201.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-09-04 09:15:00 | 198.62 | 2024-09-16 11:15:00 | 189.15 | PARTIAL | 0.50 | 4.77% |
| SELL | retest2 | 2024-09-06 11:00:00 | 197.99 | 2024-09-16 11:15:00 | 189.35 | PARTIAL | 0.50 | 4.36% |
| SELL | retest2 | 2024-09-06 12:00:00 | 198.00 | 2024-09-17 09:15:00 | 188.91 | PARTIAL | 0.50 | 4.59% |
| SELL | retest2 | 2024-09-06 12:30:00 | 197.95 | 2024-09-17 09:15:00 | 188.69 | PARTIAL | 0.50 | 4.68% |
| SELL | retest2 | 2024-09-06 15:15:00 | 197.50 | 2024-09-17 10:15:00 | 187.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-04 09:15:00 | 198.62 | 2024-09-17 13:15:00 | 190.20 | STOP_HIT | 0.50 | 4.24% |
| SELL | retest2 | 2024-09-06 11:00:00 | 197.99 | 2024-09-17 13:15:00 | 190.20 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2024-09-06 12:00:00 | 198.00 | 2024-09-17 13:15:00 | 190.20 | STOP_HIT | 0.50 | 3.94% |
| SELL | retest2 | 2024-09-06 12:30:00 | 197.95 | 2024-09-17 13:15:00 | 190.20 | STOP_HIT | 0.50 | 3.92% |
| SELL | retest2 | 2024-09-06 15:15:00 | 197.50 | 2024-09-17 13:15:00 | 190.20 | STOP_HIT | 0.50 | 3.70% |
| SELL | retest2 | 2024-09-09 11:45:00 | 192.94 | 2024-09-18 10:15:00 | 192.02 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2024-09-09 14:15:00 | 192.92 | 2024-09-18 10:15:00 | 192.02 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2024-09-11 13:30:00 | 191.99 | 2024-09-18 10:15:00 | 192.02 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2024-09-24 13:15:00 | 187.84 | 2024-10-01 10:15:00 | 187.18 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2024-09-24 13:45:00 | 187.87 | 2024-10-01 10:15:00 | 187.18 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2024-09-27 15:15:00 | 187.00 | 2024-10-01 10:15:00 | 187.18 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2024-10-01 10:00:00 | 187.76 | 2024-10-01 10:15:00 | 187.18 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2024-10-14 09:15:00 | 183.29 | 2024-10-17 14:15:00 | 174.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-14 09:15:00 | 183.29 | 2024-10-18 14:15:00 | 175.01 | STOP_HIT | 0.50 | 4.52% |
| BUY | retest2 | 2024-11-05 10:15:00 | 195.73 | 2024-11-08 11:15:00 | 190.65 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2024-11-25 09:15:00 | 196.39 | 2024-11-25 11:15:00 | 193.48 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-12-05 11:15:00 | 204.24 | 2024-12-13 10:15:00 | 208.49 | STOP_HIT | 1.00 | 2.08% |
| BUY | retest2 | 2024-12-06 09:30:00 | 204.68 | 2024-12-13 10:15:00 | 208.49 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2024-12-06 10:45:00 | 204.28 | 2024-12-13 10:15:00 | 208.49 | STOP_HIT | 1.00 | 2.06% |
| BUY | retest2 | 2024-12-06 11:30:00 | 204.25 | 2024-12-13 10:15:00 | 208.49 | STOP_HIT | 1.00 | 2.08% |
| BUY | retest2 | 2024-12-09 11:45:00 | 207.53 | 2024-12-13 10:15:00 | 208.49 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2024-12-09 12:45:00 | 207.58 | 2024-12-13 10:15:00 | 208.49 | STOP_HIT | 1.00 | 0.44% |
| BUY | retest2 | 2024-12-10 09:15:00 | 209.08 | 2024-12-13 10:15:00 | 208.49 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2024-12-10 10:30:00 | 207.52 | 2024-12-13 10:15:00 | 208.49 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2024-12-11 09:15:00 | 211.00 | 2024-12-13 10:15:00 | 208.49 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-12-19 14:15:00 | 207.32 | 2024-12-24 14:15:00 | 208.40 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-12-19 15:00:00 | 207.24 | 2024-12-24 14:15:00 | 208.40 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-12-20 10:00:00 | 207.08 | 2024-12-24 14:15:00 | 208.40 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-12-20 13:00:00 | 206.58 | 2024-12-24 14:15:00 | 208.40 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-01-01 09:30:00 | 198.08 | 2025-01-02 09:15:00 | 202.47 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-01-01 10:00:00 | 197.98 | 2025-01-02 09:15:00 | 202.47 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-01-08 14:30:00 | 202.59 | 2025-01-14 14:15:00 | 201.41 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-01-09 09:15:00 | 204.56 | 2025-01-14 14:15:00 | 201.41 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-01-10 09:15:00 | 202.70 | 2025-01-14 14:15:00 | 201.41 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-01-10 10:00:00 | 202.74 | 2025-01-14 14:15:00 | 201.41 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-01-20 09:15:00 | 224.22 | 2025-01-27 13:15:00 | 220.95 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-02-04 11:15:00 | 203.94 | 2025-02-05 11:15:00 | 207.14 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-02-04 12:00:00 | 203.63 | 2025-02-05 11:15:00 | 207.14 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-02-17 14:45:00 | 241.05 | 2025-02-18 10:15:00 | 233.26 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2025-03-04 09:15:00 | 229.11 | 2025-03-05 09:15:00 | 249.09 | STOP_HIT | 1.00 | -8.72% |
| SELL | retest2 | 2025-03-04 10:15:00 | 229.93 | 2025-03-05 09:15:00 | 249.09 | STOP_HIT | 1.00 | -8.33% |
| SELL | retest2 | 2025-03-04 12:30:00 | 230.15 | 2025-03-05 09:15:00 | 249.09 | STOP_HIT | 1.00 | -8.23% |
| SELL | retest2 | 2025-03-19 11:45:00 | 223.96 | 2025-03-20 14:15:00 | 227.92 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-03-25 14:15:00 | 247.35 | 2025-03-26 14:15:00 | 238.27 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2025-04-09 14:45:00 | 203.61 | 2025-04-11 09:15:00 | 213.87 | STOP_HIT | 1.00 | -5.04% |
| BUY | retest2 | 2025-04-21 11:00:00 | 227.63 | 2025-04-25 09:15:00 | 250.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-21 11:30:00 | 229.66 | 2025-04-29 09:15:00 | 252.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-23 09:15:00 | 230.99 | 2025-05-02 09:15:00 | 254.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-20 09:15:00 | 292.42 | 2025-05-22 11:15:00 | 276.99 | STOP_HIT | 1.00 | -5.28% |
| SELL | retest2 | 2025-05-29 11:00:00 | 266.10 | 2025-06-02 12:15:00 | 269.50 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-05-29 12:45:00 | 266.46 | 2025-06-02 12:15:00 | 269.50 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-05-29 13:15:00 | 265.98 | 2025-06-02 12:15:00 | 269.50 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-06-04 09:15:00 | 273.25 | 2025-06-04 14:15:00 | 267.70 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-06-04 11:45:00 | 270.05 | 2025-06-04 14:15:00 | 267.70 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-06-04 12:30:00 | 270.05 | 2025-06-04 14:15:00 | 267.70 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-06-04 13:00:00 | 270.95 | 2025-06-04 14:15:00 | 267.70 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-06-05 09:15:00 | 268.75 | 2025-06-11 09:15:00 | 295.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-26 12:15:00 | 298.45 | 2025-06-26 14:15:00 | 297.15 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-06-26 14:00:00 | 298.55 | 2025-06-26 14:15:00 | 297.15 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-07-07 15:15:00 | 309.45 | 2025-07-14 10:15:00 | 306.40 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2025-07-30 13:15:00 | 267.85 | 2025-07-31 09:15:00 | 254.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 13:15:00 | 267.85 | 2025-08-04 09:15:00 | 241.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-29 11:30:00 | 264.67 | 2025-09-29 14:15:00 | 293.01 | STOP_HIT | 1.00 | -10.71% |
| SELL | retest2 | 2025-10-08 15:15:00 | 273.95 | 2025-10-10 09:15:00 | 285.00 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2025-10-16 10:15:00 | 272.70 | 2025-10-30 11:15:00 | 259.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 12:15:00 | 272.60 | 2025-10-30 12:15:00 | 258.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 13:30:00 | 272.20 | 2025-10-30 12:15:00 | 258.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 10:15:00 | 272.70 | 2025-11-03 09:15:00 | 256.30 | STOP_HIT | 0.50 | 6.01% |
| SELL | retest2 | 2025-10-16 12:15:00 | 272.60 | 2025-11-03 09:15:00 | 256.30 | STOP_HIT | 0.50 | 5.98% |
| SELL | retest2 | 2025-10-16 13:30:00 | 272.20 | 2025-11-03 09:15:00 | 256.30 | STOP_HIT | 0.50 | 5.84% |
| BUY | retest2 | 2025-11-14 14:30:00 | 294.30 | 2025-11-18 09:15:00 | 289.90 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-11-14 15:00:00 | 295.05 | 2025-11-18 09:15:00 | 289.90 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-11-24 13:15:00 | 289.75 | 2025-12-03 12:15:00 | 275.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 14:15:00 | 289.80 | 2025-12-03 12:15:00 | 275.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 15:15:00 | 289.20 | 2025-12-03 15:15:00 | 274.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-25 14:30:00 | 288.65 | 2025-12-03 15:15:00 | 274.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 13:15:00 | 289.75 | 2025-12-04 09:15:00 | 279.60 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2025-11-24 14:15:00 | 289.80 | 2025-12-04 09:15:00 | 279.60 | STOP_HIT | 0.50 | 3.52% |
| SELL | retest2 | 2025-11-24 15:15:00 | 289.20 | 2025-12-04 09:15:00 | 279.60 | STOP_HIT | 0.50 | 3.32% |
| SELL | retest2 | 2025-11-25 14:30:00 | 288.65 | 2025-12-04 09:15:00 | 279.60 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2025-11-27 13:15:00 | 285.75 | 2025-12-05 09:15:00 | 271.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 13:15:00 | 285.75 | 2025-12-08 14:15:00 | 267.30 | STOP_HIT | 0.50 | 6.46% |
| BUY | retest1 | 2026-01-06 09:15:00 | 285.60 | 2026-01-08 11:15:00 | 285.20 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2026-01-08 09:15:00 | 294.35 | 2026-01-08 13:15:00 | 282.00 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2026-01-16 09:15:00 | 268.70 | 2026-01-23 10:15:00 | 255.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:15:00 | 268.70 | 2026-01-27 15:15:00 | 252.90 | STOP_HIT | 0.50 | 5.88% |
| BUY | retest2 | 2026-02-02 14:00:00 | 267.15 | 2026-02-05 12:15:00 | 268.00 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2026-02-11 09:15:00 | 273.00 | 2026-02-11 12:15:00 | 270.15 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2026-02-11 10:45:00 | 271.70 | 2026-02-11 12:15:00 | 270.15 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-02-11 11:45:00 | 271.50 | 2026-02-11 12:15:00 | 270.15 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-02-18 09:15:00 | 259.40 | 2026-02-23 12:15:00 | 246.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 09:15:00 | 259.40 | 2026-02-25 09:15:00 | 244.10 | STOP_HIT | 0.50 | 5.90% |
| SELL | retest2 | 2026-03-06 10:45:00 | 247.35 | 2026-03-09 09:15:00 | 234.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:15:00 | 247.60 | 2026-03-09 09:15:00 | 235.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:30:00 | 247.30 | 2026-03-09 09:15:00 | 234.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 10:45:00 | 247.35 | 2026-03-10 09:15:00 | 261.85 | STOP_HIT | 0.50 | -5.86% |
| SELL | retest2 | 2026-03-06 13:15:00 | 247.60 | 2026-03-10 09:15:00 | 261.85 | STOP_HIT | 0.50 | -5.76% |
| SELL | retest2 | 2026-03-06 14:30:00 | 247.30 | 2026-03-10 09:15:00 | 261.85 | STOP_HIT | 0.50 | -5.88% |
| SELL | retest2 | 2026-03-19 09:15:00 | 224.60 | 2026-03-23 09:15:00 | 213.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 224.60 | 2026-03-24 12:15:00 | 209.85 | STOP_HIT | 0.50 | 6.57% |
| BUY | retest2 | 2026-04-13 10:30:00 | 218.83 | 2026-04-17 09:15:00 | 240.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 13:45:00 | 218.82 | 2026-04-17 09:15:00 | 240.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 09:15:00 | 223.51 | 2026-04-21 15:15:00 | 228.00 | STOP_HIT | 1.00 | 2.01% |
| SELL | retest2 | 2026-04-28 11:15:00 | 217.50 | 2026-05-04 10:15:00 | 221.17 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2026-04-28 11:45:00 | 216.58 | 2026-05-04 10:15:00 | 221.17 | STOP_HIT | 1.00 | -2.12% |
