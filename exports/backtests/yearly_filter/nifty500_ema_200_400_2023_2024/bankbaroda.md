# Bank of Baroda (BANKBARODA)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 263.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 3 |
| ALERT3 | 66 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 62 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 57 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 64 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 11 / 53
- **Target hits / Stop hits / Partials:** 3 / 57 / 4
- **Avg / median % per leg:** -0.90% / -1.69%
- **Sum % (uncompounded):** -57.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 5 | 10.0% | 3 | 47 | 0 | -1.48% | -73.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 50 | 5 | 10.0% | 3 | 47 | 0 | -1.48% | -73.8% |
| SELL (all) | 14 | 6 | 42.9% | 0 | 10 | 4 | 1.16% | 16.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 6 | 42.9% | 0 | 10 | 4 | 1.16% | 16.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 64 | 11 | 17.2% | 3 | 57 | 4 | -0.90% | -57.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 13:15:00 | 192.30 | 200.90 | 200.93 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 11:15:00 | 208.90 | 199.99 | 199.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-06 13:15:00 | 209.80 | 200.17 | 200.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 09:15:00 | 221.80 | 222.14 | 214.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-10 09:30:00 | 222.85 | 222.14 | 214.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 10:15:00 | 251.90 | 264.33 | 252.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-15 11:00:00 | 251.90 | 264.33 | 252.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 11:15:00 | 251.35 | 264.20 | 252.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-15 12:00:00 | 251.35 | 264.20 | 252.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 12:15:00 | 252.15 | 264.08 | 252.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-15 13:30:00 | 254.10 | 263.97 | 252.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-15 14:45:00 | 254.30 | 263.86 | 252.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-15 15:15:00 | 254.90 | 263.86 | 252.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-18 12:00:00 | 254.90 | 263.48 | 252.12 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 252.55 | 263.03 | 252.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:00:00 | 252.55 | 263.03 | 252.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 250.75 | 262.90 | 252.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:45:00 | 250.35 | 262.90 | 252.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 11:15:00 | 250.40 | 262.78 | 252.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 11:30:00 | 249.95 | 262.78 | 252.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-03-19 13:15:00 | 249.45 | 262.54 | 252.13 | SL hit (close<static) qty=1.00 sl=250.15 alert=retest2 |

### Cycle 3 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 256.25 | 267.57 | 267.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 14:15:00 | 255.35 | 267.11 | 267.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 09:15:00 | 252.25 | 250.69 | 256.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-20 10:00:00 | 252.25 | 250.69 | 256.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 256.60 | 251.07 | 256.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 11:00:00 | 256.60 | 251.07 | 256.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 11:15:00 | 255.80 | 251.12 | 256.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 12:15:00 | 255.35 | 251.12 | 256.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 14:15:00 | 242.58 | 250.93 | 254.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-23 10:15:00 | 244.20 | 243.26 | 248.72 | SL hit (close>ema200) qty=0.50 sl=243.26 alert=retest2 |

### Cycle 4 — BUY (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 10:15:00 | 263.10 | 247.98 | 247.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 12:15:00 | 265.10 | 248.30 | 248.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 247.30 | 250.23 | 249.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 09:15:00 | 247.30 | 250.23 | 249.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 247.30 | 250.23 | 249.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 10:00:00 | 247.30 | 250.23 | 249.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 249.65 | 250.22 | 249.19 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 224.55 | 248.15 | 248.23 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 09:15:00 | 259.55 | 247.88 | 247.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 10:15:00 | 264.56 | 248.86 | 248.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 10:15:00 | 252.12 | 252.63 | 250.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-13 10:45:00 | 252.10 | 252.63 | 250.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 252.18 | 253.33 | 251.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 12:45:00 | 251.34 | 253.33 | 251.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 13:15:00 | 250.50 | 253.30 | 251.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 14:00:00 | 250.50 | 253.30 | 251.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 250.37 | 253.27 | 251.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 14:45:00 | 249.97 | 253.27 | 251.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 247.45 | 253.19 | 251.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 11:00:00 | 249.10 | 253.15 | 251.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 11:00:00 | 249.03 | 252.83 | 250.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 11:30:00 | 248.80 | 252.79 | 250.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 13:15:00 | 243.19 | 252.62 | 250.91 | SL hit (close<static) qty=1.00 sl=244.85 alert=retest2 |

### Cycle 7 — SELL (started 2024-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 15:15:00 | 240.30 | 249.56 | 249.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-01 09:15:00 | 240.05 | 249.47 | 249.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 11:15:00 | 209.27 | 208.74 | 217.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 12:00:00 | 209.27 | 208.74 | 217.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 215.15 | 209.09 | 216.84 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 13:15:00 | 229.87 | 221.03 | 221.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 10:15:00 | 233.73 | 221.44 | 221.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 14:15:00 | 223.66 | 238.17 | 231.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 14:15:00 | 223.66 | 238.17 | 231.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 223.66 | 238.17 | 231.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 15:00:00 | 223.66 | 238.17 | 231.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 221.95 | 238.01 | 231.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 09:15:00 | 226.62 | 238.01 | 231.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:15:00 | 224.93 | 237.71 | 231.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 15:00:00 | 224.51 | 237.15 | 231.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 10:15:00 | 224.67 | 236.90 | 231.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 11:15:00 | 221.53 | 236.62 | 231.18 | SL hit (close<static) qty=1.00 sl=221.60 alert=retest2 |

### Cycle 9 — SELL (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 15:15:00 | 234.50 | 240.31 | 240.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 233.89 | 239.71 | 240.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 11:15:00 | 239.75 | 238.66 | 239.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 11:15:00 | 239.75 | 238.66 | 239.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 239.75 | 238.66 | 239.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:45:00 | 239.50 | 238.66 | 239.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 238.35 | 238.65 | 239.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:15:00 | 238.34 | 238.65 | 239.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:45:00 | 238.17 | 238.65 | 239.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 10:15:00 | 240.09 | 238.65 | 239.39 | SL hit (close>static) qty=1.00 sl=240.05 alert=retest2 |

### Cycle 10 — BUY (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 10:15:00 | 252.50 | 239.90 | 239.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 09:15:00 | 257.25 | 242.24 | 241.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 284.40 | 284.57 | 274.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 12:45:00 | 284.00 | 284.57 | 274.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 293.25 | 299.16 | 292.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 10:30:00 | 297.05 | 293.05 | 290.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 282.05 | 302.05 | 297.22 | SL hit (close<static) qty=1.00 sl=288.25 alert=retest2 |

### Cycle 11 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 273.60 | 293.85 | 293.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 272.15 | 293.64 | 293.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-09 09:15:00 | 277.86 | 276.51 | 283.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 277.86 | 276.51 | 283.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 279.80 | 276.26 | 282.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 11:00:00 | 278.06 | 276.27 | 282.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 11:45:00 | 278.92 | 276.30 | 282.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 15:15:00 | 278.60 | 276.40 | 282.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:30:00 | 278.95 | 276.52 | 282.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 280.20 | 276.88 | 282.44 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-20 10:15:00 | 284.05 | 276.95 | 282.45 | SL hit (close>static) qty=1.00 sl=283.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-08-04 11:15:00 | 195.70 | 2023-08-04 13:15:00 | 192.40 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2023-08-04 12:45:00 | 195.55 | 2023-08-04 13:15:00 | 192.40 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2023-08-08 10:30:00 | 196.00 | 2023-08-10 12:15:00 | 191.95 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2023-09-04 14:30:00 | 195.35 | 2023-09-18 09:15:00 | 214.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-13 11:15:00 | 203.55 | 2023-10-20 12:15:00 | 201.95 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2023-10-13 11:45:00 | 203.50 | 2023-10-20 12:15:00 | 201.95 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2023-10-18 15:15:00 | 204.25 | 2023-10-20 12:15:00 | 201.95 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2023-10-19 10:45:00 | 203.75 | 2023-10-20 12:15:00 | 201.95 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-03-15 13:30:00 | 254.10 | 2024-03-19 13:15:00 | 249.45 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2024-03-15 14:45:00 | 254.30 | 2024-03-19 13:15:00 | 249.45 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-03-15 15:15:00 | 254.90 | 2024-03-19 13:15:00 | 249.45 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-03-18 12:00:00 | 254.90 | 2024-03-19 13:15:00 | 249.45 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-04-18 09:15:00 | 260.55 | 2024-04-19 09:15:00 | 254.85 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-04-18 15:15:00 | 260.20 | 2024-04-19 09:15:00 | 254.85 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-04-22 09:45:00 | 259.95 | 2024-05-10 09:15:00 | 260.35 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2024-04-22 13:15:00 | 260.00 | 2024-05-10 09:15:00 | 260.35 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2024-05-09 09:15:00 | 268.05 | 2024-05-10 09:15:00 | 260.35 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2024-05-09 13:00:00 | 264.70 | 2024-05-10 14:15:00 | 253.00 | STOP_HIT | 1.00 | -4.42% |
| BUY | retest2 | 2024-05-09 14:00:00 | 265.60 | 2024-05-10 14:15:00 | 253.00 | STOP_HIT | 1.00 | -4.74% |
| BUY | retest2 | 2024-05-10 11:15:00 | 263.70 | 2024-05-10 14:15:00 | 253.00 | STOP_HIT | 1.00 | -4.06% |
| BUY | retest2 | 2024-05-13 09:15:00 | 257.00 | 2024-06-03 09:15:00 | 282.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-13 11:15:00 | 256.60 | 2024-06-03 09:15:00 | 282.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 11:15:00 | 258.15 | 2024-06-04 11:15:00 | 239.85 | STOP_HIT | 1.00 | -7.09% |
| BUY | retest2 | 2024-06-04 13:45:00 | 258.80 | 2024-06-04 14:15:00 | 249.85 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2024-07-05 11:15:00 | 272.25 | 2024-07-08 09:15:00 | 266.05 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-07-05 12:45:00 | 271.85 | 2024-07-08 09:15:00 | 266.05 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-08-22 12:15:00 | 255.35 | 2024-09-04 14:15:00 | 242.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-22 12:15:00 | 255.35 | 2024-09-23 10:15:00 | 244.20 | STOP_HIT | 0.50 | 4.37% |
| SELL | retest2 | 2024-10-04 12:30:00 | 254.20 | 2024-10-07 13:15:00 | 241.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 12:30:00 | 254.20 | 2024-10-08 10:15:00 | 246.79 | STOP_HIT | 0.50 | 2.92% |
| BUY | retest2 | 2024-12-19 11:00:00 | 249.10 | 2024-12-20 13:15:00 | 243.19 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2024-12-20 11:00:00 | 249.03 | 2024-12-20 13:15:00 | 243.19 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-12-20 11:30:00 | 248.80 | 2024-12-20 13:15:00 | 243.19 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-05-07 09:15:00 | 226.62 | 2025-05-08 11:15:00 | 221.53 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-05-07 11:15:00 | 224.93 | 2025-05-08 11:15:00 | 221.53 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-05-07 15:00:00 | 224.51 | 2025-05-08 11:15:00 | 221.53 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-05-08 10:15:00 | 224.67 | 2025-05-08 11:15:00 | 221.53 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-06-16 13:30:00 | 240.25 | 2025-06-19 09:15:00 | 233.26 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest2 | 2025-06-25 12:30:00 | 240.08 | 2025-07-11 09:15:00 | 237.71 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-06-27 09:15:00 | 243.00 | 2025-07-30 15:15:00 | 239.53 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-07-04 12:30:00 | 240.32 | 2025-07-30 15:15:00 | 239.53 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-07-09 09:15:00 | 240.82 | 2025-07-31 09:15:00 | 236.93 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-07-14 09:30:00 | 240.95 | 2025-07-31 09:15:00 | 236.93 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-07-14 12:00:00 | 240.36 | 2025-07-31 09:15:00 | 236.93 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-07-23 09:15:00 | 240.85 | 2025-07-31 09:15:00 | 236.93 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-07-23 13:00:00 | 241.47 | 2025-07-31 09:15:00 | 236.93 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-07-23 13:45:00 | 241.57 | 2025-07-31 09:15:00 | 236.93 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-07-24 13:15:00 | 242.50 | 2025-07-31 09:15:00 | 236.93 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-07-28 14:30:00 | 241.62 | 2025-08-26 09:15:00 | 238.71 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-07-29 12:45:00 | 243.07 | 2025-08-26 09:15:00 | 238.71 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-07-29 14:15:00 | 242.97 | 2025-08-28 09:15:00 | 233.08 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest2 | 2025-08-11 09:45:00 | 243.10 | 2025-08-28 09:15:00 | 233.08 | STOP_HIT | 1.00 | -4.12% |
| BUY | retest2 | 2025-08-11 12:00:00 | 243.07 | 2025-08-28 09:15:00 | 233.08 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2025-09-10 13:15:00 | 238.34 | 2025-09-11 10:15:00 | 240.09 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-09-10 13:45:00 | 238.17 | 2025-09-11 10:15:00 | 240.09 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-09-11 14:45:00 | 238.32 | 2025-09-16 13:15:00 | 240.34 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-09-12 09:45:00 | 238.33 | 2025-09-16 13:15:00 | 240.34 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2026-02-17 10:30:00 | 297.05 | 2026-03-09 09:15:00 | 282.05 | STOP_HIT | 1.00 | -5.05% |
| SELL | retest2 | 2026-04-15 11:00:00 | 278.06 | 2026-04-20 10:15:00 | 284.05 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2026-04-15 11:45:00 | 278.92 | 2026-04-20 10:15:00 | 284.05 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-04-15 15:15:00 | 278.60 | 2026-04-20 10:15:00 | 284.05 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-04-16 11:30:00 | 278.95 | 2026-04-20 10:15:00 | 284.05 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-04-23 12:45:00 | 274.65 | 2026-04-30 09:15:00 | 263.62 | PARTIAL | 0.50 | 4.01% |
| SELL | retest2 | 2026-04-23 13:30:00 | 277.50 | 2026-04-30 11:15:00 | 260.92 | PARTIAL | 0.50 | 5.98% |
