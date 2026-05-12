# FSN E-Commerce Ventures Ltd. (NYKAA)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 273.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 83 |
| ALERT1 | 54 |
| ALERT2 | 53 |
| ALERT2_SKIP | 22 |
| ALERT3 | 132 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 64 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 18 / 48
- **Target hits / Stop hits / Partials:** 0 / 63 / 3
- **Avg / median % per leg:** -0.00% / -0.88%
- **Sum % (uncompounded):** -0.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 11 | 28.9% | 0 | 38 | 0 | 0.13% | 5.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 38 | 11 | 28.9% | 0 | 38 | 0 | 0.13% | 5.0% |
| SELL (all) | 28 | 7 | 25.0% | 0 | 25 | 3 | -0.19% | -5.3% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.74% | -2.7% |
| SELL @ 3rd Alert (retest2) | 27 | 7 | 25.9% | 0 | 24 | 3 | -0.09% | -2.5% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.74% | -2.7% |
| retest2 (combined) | 65 | 18 | 27.7% | 0 | 62 | 3 | 0.04% | 2.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 198.08 | 194.74 | 194.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 198.40 | 195.47 | 194.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 196.16 | 197.70 | 196.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 196.16 | 197.70 | 196.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 196.16 | 197.70 | 196.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 196.16 | 197.70 | 196.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 197.94 | 197.74 | 196.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 14:15:00 | 198.58 | 197.74 | 196.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 200.79 | 197.62 | 196.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 11:00:00 | 198.00 | 197.89 | 197.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 13:00:00 | 198.01 | 197.91 | 197.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 197.31 | 197.84 | 197.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:45:00 | 197.50 | 197.84 | 197.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 15:15:00 | 197.15 | 197.70 | 197.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:15:00 | 198.81 | 197.72 | 197.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 199.84 | 197.46 | 197.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 15:15:00 | 197.06 | 199.37 | 199.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 15:15:00 | 197.06 | 199.37 | 199.48 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 09:15:00 | 200.87 | 199.67 | 199.61 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 199.11 | 199.56 | 199.56 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 12:15:00 | 200.55 | 199.75 | 199.65 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 198.58 | 199.46 | 199.54 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 201.00 | 199.77 | 199.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 10:15:00 | 202.99 | 200.42 | 199.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 13:15:00 | 200.39 | 200.61 | 200.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 13:15:00 | 200.39 | 200.61 | 200.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 200.39 | 200.61 | 200.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 13:45:00 | 199.96 | 200.61 | 200.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 200.46 | 200.58 | 200.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:15:00 | 201.30 | 200.58 | 200.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 201.30 | 200.72 | 200.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 201.29 | 200.72 | 200.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 202.88 | 201.16 | 200.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:45:00 | 204.00 | 202.14 | 201.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:30:00 | 203.96 | 202.43 | 201.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 204.36 | 202.79 | 202.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 14:15:00 | 200.17 | 201.97 | 202.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 14:15:00 | 200.17 | 201.97 | 202.02 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 204.10 | 201.64 | 201.60 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 200.93 | 201.62 | 201.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 09:15:00 | 200.00 | 201.20 | 201.44 | Break + close below crossover candle low |

### Cycle 11 — BUY (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 14:15:00 | 205.10 | 201.46 | 201.39 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 09:15:00 | 200.08 | 202.03 | 202.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 11:15:00 | 195.00 | 199.75 | 200.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 13:15:00 | 195.77 | 195.44 | 197.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 14:00:00 | 195.77 | 195.44 | 197.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 196.50 | 195.40 | 196.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 10:30:00 | 195.20 | 195.54 | 196.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 12:15:00 | 199.17 | 196.50 | 196.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 199.17 | 196.50 | 196.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 10:15:00 | 201.08 | 198.00 | 197.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 200.00 | 200.67 | 199.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 10:00:00 | 200.00 | 200.67 | 199.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 199.21 | 200.27 | 199.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:00:00 | 199.21 | 200.27 | 199.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 198.90 | 199.99 | 199.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:30:00 | 198.82 | 199.99 | 199.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 199.15 | 199.82 | 199.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 199.98 | 199.65 | 199.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 11:00:00 | 199.96 | 199.74 | 199.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 198.27 | 199.29 | 199.27 | SL hit (close<static) qty=1.00 sl=198.80 alert=retest2 |

### Cycle 14 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 198.45 | 199.13 | 199.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 198.28 | 198.96 | 199.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 194.45 | 194.39 | 195.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 13:45:00 | 194.12 | 194.39 | 195.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 196.18 | 194.71 | 195.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 196.18 | 194.71 | 195.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 195.34 | 194.84 | 195.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 15:15:00 | 194.57 | 194.84 | 195.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 10:15:00 | 196.89 | 195.06 | 194.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 196.89 | 195.06 | 194.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 11:15:00 | 197.80 | 195.61 | 195.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 13:15:00 | 195.21 | 195.62 | 195.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 13:15:00 | 195.21 | 195.62 | 195.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 195.21 | 195.62 | 195.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:00:00 | 195.21 | 195.62 | 195.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 196.31 | 195.75 | 195.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 09:15:00 | 196.83 | 195.81 | 195.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 11:15:00 | 194.73 | 195.74 | 195.52 | SL hit (close<static) qty=1.00 sl=194.78 alert=retest2 |

### Cycle 16 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 193.46 | 195.28 | 195.34 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 197.11 | 195.26 | 195.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 10:15:00 | 200.17 | 196.77 | 195.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 198.29 | 200.67 | 198.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 198.29 | 200.67 | 198.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 198.29 | 200.67 | 198.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:45:00 | 196.64 | 200.67 | 198.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 198.82 | 200.30 | 198.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 11:15:00 | 199.47 | 200.30 | 198.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 12:30:00 | 199.34 | 199.85 | 198.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 201.03 | 199.01 | 198.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 204.95 | 207.07 | 207.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 204.95 | 207.07 | 207.08 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 10:15:00 | 207.90 | 207.24 | 207.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 11:15:00 | 209.07 | 207.60 | 207.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 09:15:00 | 204.08 | 208.46 | 208.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 204.08 | 208.46 | 208.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 204.08 | 208.46 | 208.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:45:00 | 202.43 | 208.46 | 208.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 10:15:00 | 203.13 | 207.40 | 207.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 14:15:00 | 202.51 | 204.84 | 206.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 202.46 | 200.33 | 202.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 202.46 | 200.33 | 202.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 202.46 | 200.33 | 202.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 202.46 | 200.33 | 202.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 202.33 | 200.73 | 202.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:30:00 | 201.61 | 200.59 | 202.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:00:00 | 201.81 | 201.28 | 201.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 205.85 | 202.83 | 202.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 205.85 | 202.83 | 202.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 11:15:00 | 209.70 | 204.57 | 203.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 09:15:00 | 218.40 | 218.84 | 216.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 09:45:00 | 218.40 | 218.84 | 216.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 217.02 | 218.31 | 216.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:30:00 | 216.21 | 218.31 | 216.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 217.66 | 218.18 | 217.04 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 10:15:00 | 214.67 | 216.43 | 216.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 11:15:00 | 214.25 | 216.00 | 216.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 14:15:00 | 216.00 | 215.74 | 216.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 14:15:00 | 216.00 | 215.74 | 216.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 216.00 | 215.74 | 216.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 15:00:00 | 216.00 | 215.74 | 216.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 216.00 | 215.79 | 216.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 217.10 | 215.79 | 216.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 216.58 | 215.95 | 216.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 218.28 | 215.95 | 216.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 10:15:00 | 218.71 | 216.50 | 216.36 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 214.88 | 216.30 | 216.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 213.32 | 215.71 | 216.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 11:15:00 | 214.38 | 213.93 | 214.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 12:00:00 | 214.38 | 213.93 | 214.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 214.58 | 214.15 | 214.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:30:00 | 214.66 | 214.15 | 214.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 215.04 | 214.33 | 214.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 215.04 | 214.33 | 214.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 216.00 | 214.66 | 214.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 216.90 | 214.66 | 214.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 219.17 | 215.56 | 215.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 12:15:00 | 219.81 | 217.26 | 216.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 217.98 | 218.58 | 217.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 10:00:00 | 217.98 | 218.58 | 217.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 216.83 | 218.23 | 217.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 216.83 | 218.23 | 217.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 217.40 | 218.07 | 217.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:45:00 | 216.60 | 218.07 | 217.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 216.08 | 217.67 | 217.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:00:00 | 216.08 | 217.67 | 217.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 216.10 | 217.36 | 217.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:00:00 | 216.10 | 217.36 | 217.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 216.75 | 217.71 | 217.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 216.75 | 217.71 | 217.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 216.38 | 217.44 | 217.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:00:00 | 216.38 | 217.44 | 217.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 216.21 | 217.19 | 217.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:30:00 | 216.48 | 217.19 | 217.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 215.93 | 216.94 | 217.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 13:15:00 | 215.60 | 216.67 | 216.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 214.20 | 213.51 | 214.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:30:00 | 213.71 | 213.51 | 214.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 212.49 | 213.30 | 214.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:30:00 | 211.85 | 213.11 | 214.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:15:00 | 212.20 | 213.11 | 214.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 12:00:00 | 212.07 | 210.76 | 211.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 12:45:00 | 212.07 | 211.02 | 211.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 14:15:00 | 214.00 | 211.99 | 211.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 214.00 | 211.99 | 211.86 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 210.30 | 211.76 | 211.80 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 212.11 | 211.82 | 211.82 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 13:15:00 | 210.78 | 211.61 | 211.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 209.42 | 211.17 | 211.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 14:15:00 | 210.46 | 210.28 | 210.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-01 15:00:00 | 210.46 | 210.28 | 210.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 210.00 | 210.02 | 210.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 210.52 | 210.02 | 210.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 209.46 | 209.91 | 210.46 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 12:15:00 | 213.79 | 210.82 | 210.79 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 208.95 | 210.97 | 211.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 207.60 | 210.29 | 210.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 209.50 | 207.74 | 208.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 209.50 | 207.74 | 208.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 209.50 | 207.74 | 208.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 209.50 | 207.74 | 208.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 209.52 | 208.10 | 208.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 206.89 | 208.10 | 208.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 211.10 | 205.73 | 205.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 211.10 | 205.73 | 205.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 11:15:00 | 219.69 | 209.77 | 207.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 10:15:00 | 211.75 | 212.94 | 210.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 11:00:00 | 211.75 | 212.94 | 210.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 219.55 | 218.35 | 215.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 10:45:00 | 221.36 | 219.10 | 216.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 12:15:00 | 228.44 | 230.09 | 230.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 12:15:00 | 228.44 | 230.09 | 230.13 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 15:15:00 | 230.80 | 230.22 | 230.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 231.80 | 230.54 | 230.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 232.50 | 233.07 | 232.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 232.50 | 233.07 | 232.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 232.50 | 233.07 | 232.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:45:00 | 232.24 | 233.07 | 232.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 232.55 | 232.97 | 232.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:30:00 | 232.90 | 232.97 | 232.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 231.34 | 232.64 | 232.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 233.78 | 232.64 | 232.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 15:15:00 | 241.00 | 241.87 | 241.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 15:15:00 | 241.00 | 241.87 | 241.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 09:15:00 | 240.67 | 241.63 | 241.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 15:15:00 | 238.22 | 238.08 | 239.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 09:15:00 | 237.80 | 238.08 | 239.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 237.79 | 238.02 | 238.94 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 241.80 | 239.46 | 239.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 13:15:00 | 243.00 | 240.54 | 239.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 242.66 | 242.68 | 241.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 09:45:00 | 242.50 | 242.68 | 241.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 241.45 | 242.39 | 241.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:00:00 | 241.45 | 242.39 | 241.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 241.21 | 242.16 | 241.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:45:00 | 241.06 | 242.16 | 241.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 241.16 | 241.96 | 241.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 13:45:00 | 241.63 | 241.96 | 241.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 242.64 | 242.09 | 241.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 15:15:00 | 243.25 | 242.09 | 241.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 13:15:00 | 242.91 | 242.88 | 242.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 10:15:00 | 243.45 | 243.84 | 243.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 15:15:00 | 242.59 | 243.49 | 243.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 15:15:00 | 242.59 | 243.49 | 243.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 238.30 | 242.45 | 243.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 15:15:00 | 238.55 | 238.37 | 239.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 09:15:00 | 237.72 | 238.37 | 239.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 232.32 | 232.04 | 233.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:00:00 | 232.32 | 232.04 | 233.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 232.70 | 231.88 | 232.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:45:00 | 232.31 | 231.88 | 232.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 234.30 | 232.36 | 233.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 235.59 | 232.36 | 233.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 237.00 | 233.29 | 233.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:45:00 | 237.11 | 233.29 | 233.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 10:15:00 | 237.89 | 234.21 | 233.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 11:15:00 | 238.31 | 235.03 | 234.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 13:15:00 | 238.40 | 238.64 | 237.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 14:00:00 | 238.40 | 238.64 | 237.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 263.00 | 264.46 | 262.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:45:00 | 262.45 | 264.46 | 262.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 261.36 | 263.84 | 262.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:00:00 | 261.36 | 263.84 | 262.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 262.22 | 263.52 | 262.38 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 259.18 | 261.66 | 261.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 257.54 | 260.84 | 261.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 259.71 | 258.31 | 259.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 259.71 | 258.31 | 259.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 259.71 | 258.31 | 259.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 259.71 | 258.31 | 259.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 257.93 | 258.23 | 259.54 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 264.62 | 260.90 | 260.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 265.15 | 262.10 | 261.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 15:15:00 | 262.68 | 263.04 | 261.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 09:15:00 | 263.56 | 263.04 | 261.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 263.96 | 263.23 | 262.13 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 15:15:00 | 258.93 | 261.65 | 261.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 13:15:00 | 257.91 | 259.96 | 260.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 259.19 | 259.12 | 260.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 259.19 | 259.12 | 260.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 259.19 | 259.12 | 260.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 259.11 | 259.12 | 260.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 257.39 | 258.43 | 259.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:30:00 | 258.90 | 258.43 | 259.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 253.99 | 252.30 | 254.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 253.99 | 252.30 | 254.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 253.38 | 252.65 | 254.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:45:00 | 253.51 | 252.65 | 254.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 254.87 | 253.24 | 254.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:00:00 | 254.87 | 253.24 | 254.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 255.16 | 253.63 | 254.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:45:00 | 254.87 | 253.63 | 254.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 254.92 | 253.89 | 254.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 255.66 | 253.89 | 254.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 256.35 | 254.63 | 254.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 15:15:00 | 258.77 | 256.30 | 255.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 14:15:00 | 257.61 | 257.79 | 256.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 15:00:00 | 257.61 | 257.79 | 256.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 255.71 | 257.57 | 256.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 255.71 | 257.57 | 256.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 256.80 | 257.41 | 256.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 254.24 | 257.41 | 256.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 256.00 | 257.13 | 256.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:00:00 | 256.00 | 257.13 | 256.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 257.60 | 257.22 | 256.86 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 253.93 | 256.34 | 256.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 12:15:00 | 253.00 | 255.21 | 255.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 12:15:00 | 250.51 | 250.21 | 252.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 13:00:00 | 250.51 | 250.21 | 252.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 251.31 | 250.16 | 251.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:15:00 | 251.91 | 250.16 | 251.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 250.87 | 250.30 | 251.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 14:15:00 | 250.68 | 250.30 | 251.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 15:00:00 | 250.28 | 250.30 | 251.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 249.66 | 250.42 | 251.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 256.68 | 248.33 | 248.52 | SL hit (close>static) qty=1.00 sl=252.68 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 260.44 | 250.76 | 249.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 263.05 | 260.04 | 259.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 15:15:00 | 268.50 | 268.64 | 266.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 09:15:00 | 268.86 | 268.64 | 266.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 267.07 | 268.08 | 266.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:45:00 | 267.76 | 268.08 | 266.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 268.45 | 268.54 | 267.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 12:00:00 | 269.25 | 268.68 | 267.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 12:45:00 | 269.00 | 268.88 | 267.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 270.04 | 269.03 | 268.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 12:15:00 | 269.47 | 269.29 | 268.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 269.33 | 269.30 | 268.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-24 11:15:00 | 266.94 | 268.52 | 268.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 11:15:00 | 266.94 | 268.52 | 268.53 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 12:15:00 | 269.07 | 268.63 | 268.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 13:15:00 | 270.83 | 269.07 | 268.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 11:15:00 | 270.00 | 270.22 | 269.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-25 11:45:00 | 270.10 | 270.22 | 269.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 269.79 | 270.14 | 269.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 10:15:00 | 271.30 | 270.30 | 269.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 12:15:00 | 265.06 | 269.16 | 269.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 12:15:00 | 265.06 | 269.16 | 269.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-26 14:15:00 | 264.51 | 267.75 | 268.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 09:15:00 | 268.17 | 266.01 | 266.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 268.17 | 266.01 | 266.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 268.17 | 266.01 | 266.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:30:00 | 268.10 | 266.01 | 266.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 267.99 | 266.41 | 266.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:30:00 | 268.19 | 266.41 | 266.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 267.42 | 267.29 | 267.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 13:45:00 | 267.50 | 267.29 | 267.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 14:15:00 | 267.46 | 267.32 | 267.31 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 15:15:00 | 267.02 | 267.26 | 267.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 09:15:00 | 264.20 | 266.65 | 267.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 265.55 | 265.13 | 265.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 15:00:00 | 265.55 | 265.13 | 265.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 264.60 | 265.03 | 265.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 263.90 | 265.03 | 265.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 250.70 | 253.64 | 255.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 15:15:00 | 253.65 | 253.05 | 254.65 | SL hit (close>ema200) qty=0.50 sl=253.05 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 252.80 | 250.74 | 250.64 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 247.35 | 250.42 | 250.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 246.05 | 248.68 | 249.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 14:15:00 | 248.00 | 247.98 | 249.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 15:00:00 | 248.00 | 247.98 | 249.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 242.25 | 244.89 | 246.56 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 249.20 | 246.12 | 245.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 253.65 | 248.19 | 246.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 258.05 | 258.11 | 255.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 13:45:00 | 258.00 | 258.11 | 255.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 258.65 | 259.08 | 257.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:30:00 | 260.40 | 259.12 | 258.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:00:00 | 260.00 | 259.12 | 258.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 11:15:00 | 260.15 | 259.22 | 258.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 14:15:00 | 266.00 | 267.36 | 267.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 14:15:00 | 266.00 | 267.36 | 267.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 264.05 | 266.53 | 267.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 252.70 | 252.64 | 255.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 253.15 | 252.64 | 255.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 254.10 | 252.04 | 253.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:45:00 | 253.95 | 252.04 | 253.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 255.95 | 252.82 | 254.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 255.60 | 252.82 | 254.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 252.25 | 252.71 | 253.85 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 254.80 | 253.88 | 253.88 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 252.80 | 253.80 | 253.86 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 255.90 | 253.95 | 253.89 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 252.00 | 253.84 | 253.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 13:15:00 | 251.35 | 253.34 | 253.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 244.75 | 242.71 | 245.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 244.75 | 242.71 | 245.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 244.75 | 242.71 | 245.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 246.65 | 242.71 | 245.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 241.80 | 242.53 | 244.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 240.90 | 242.53 | 244.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 11:15:00 | 237.05 | 236.73 | 236.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 237.05 | 236.73 | 236.70 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 12:15:00 | 236.00 | 236.59 | 236.63 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 14:15:00 | 237.25 | 236.75 | 236.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 239.71 | 237.34 | 236.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 237.57 | 239.27 | 238.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 15:15:00 | 237.57 | 239.27 | 238.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 237.57 | 239.27 | 238.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 234.78 | 239.27 | 238.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 236.86 | 238.79 | 238.25 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 236.26 | 237.91 | 237.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 13:15:00 | 235.35 | 237.09 | 237.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 238.05 | 237.29 | 237.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 238.05 | 237.29 | 237.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 238.05 | 237.29 | 237.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 238.05 | 237.29 | 237.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 235.79 | 236.99 | 237.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 244.50 | 236.99 | 237.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 244.58 | 238.50 | 238.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 247.15 | 240.23 | 238.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 10:15:00 | 279.81 | 280.16 | 275.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 11:00:00 | 279.81 | 280.16 | 275.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 276.95 | 278.79 | 276.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:45:00 | 276.88 | 278.79 | 276.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 277.72 | 278.58 | 276.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 11:00:00 | 278.37 | 277.99 | 276.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 09:30:00 | 278.53 | 278.92 | 277.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 274.12 | 277.96 | 277.54 | SL hit (close<static) qty=1.00 sl=276.22 alert=retest2 |

### Cycle 64 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 274.40 | 276.86 | 277.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 271.16 | 275.21 | 276.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 274.52 | 274.40 | 275.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 10:00:00 | 274.52 | 274.40 | 275.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 276.49 | 274.82 | 275.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:00:00 | 276.49 | 274.82 | 275.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 275.17 | 274.89 | 275.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 12:15:00 | 274.68 | 274.89 | 275.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 274.22 | 274.74 | 275.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:15:00 | 260.95 | 264.48 | 266.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:15:00 | 260.51 | 264.48 | 266.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 15:15:00 | 262.10 | 262.01 | 264.15 | SL hit (close>ema200) qty=0.50 sl=262.01 alert=retest2 |

### Cycle 65 — BUY (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 13:15:00 | 266.03 | 265.18 | 265.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 268.33 | 266.25 | 265.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 265.79 | 266.65 | 265.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 11:15:00 | 265.79 | 266.65 | 265.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 265.79 | 266.65 | 265.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:00:00 | 265.79 | 266.65 | 265.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 264.99 | 266.32 | 265.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:45:00 | 264.76 | 266.32 | 265.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 266.03 | 266.26 | 265.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 14:15:00 | 266.34 | 266.26 | 265.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 263.90 | 266.23 | 266.02 | SL hit (close<static) qty=1.00 sl=264.06 alert=retest2 |

### Cycle 66 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 265.10 | 265.93 | 265.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 261.75 | 264.93 | 265.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 261.00 | 260.41 | 262.48 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:15:00 | 253.50 | 260.41 | 262.48 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 256.65 | 255.60 | 258.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:15:00 | 258.70 | 255.60 | 258.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 256.10 | 255.70 | 258.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:15:00 | 255.90 | 255.70 | 258.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:00:00 | 255.60 | 255.68 | 257.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:45:00 | 256.00 | 255.69 | 257.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 260.45 | 256.64 | 257.62 | SL hit (close>ema400) qty=1.00 sl=257.62 alert=retest1 |

### Cycle 67 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 261.60 | 258.27 | 258.23 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 12:15:00 | 256.45 | 257.93 | 258.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 14:15:00 | 254.85 | 257.05 | 257.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 253.05 | 250.56 | 252.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 253.05 | 250.56 | 252.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 253.05 | 250.56 | 252.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:00:00 | 253.05 | 250.56 | 252.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 255.75 | 251.59 | 253.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:00:00 | 255.75 | 251.59 | 253.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 255.30 | 252.34 | 253.26 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 257.35 | 253.79 | 253.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 258.70 | 255.59 | 254.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 11:15:00 | 255.00 | 255.61 | 254.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 12:00:00 | 255.00 | 255.61 | 254.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 253.95 | 255.28 | 254.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 253.95 | 255.28 | 254.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 253.95 | 255.01 | 254.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 253.95 | 255.01 | 254.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 251.70 | 254.35 | 254.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 251.00 | 253.68 | 254.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 10:15:00 | 237.10 | 237.00 | 239.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:45:00 | 237.10 | 237.00 | 239.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 240.45 | 238.02 | 239.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 240.45 | 238.02 | 239.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 240.20 | 238.45 | 239.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:30:00 | 240.45 | 238.45 | 239.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 238.15 | 238.39 | 239.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 242.30 | 238.39 | 239.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 242.70 | 239.25 | 239.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 242.80 | 239.25 | 239.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 241.85 | 239.77 | 239.97 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 242.30 | 240.28 | 240.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 242.80 | 240.78 | 240.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 238.30 | 241.12 | 240.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 238.30 | 241.12 | 240.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 238.30 | 241.12 | 240.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:45:00 | 240.30 | 240.88 | 240.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 11:15:00 | 239.50 | 240.61 | 240.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 239.50 | 240.61 | 240.61 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 242.65 | 240.68 | 240.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 11:15:00 | 243.20 | 241.19 | 240.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 14:15:00 | 240.80 | 241.48 | 241.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 14:15:00 | 240.80 | 241.48 | 241.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 240.80 | 241.48 | 241.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 240.80 | 241.48 | 241.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 239.80 | 241.15 | 240.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 234.20 | 241.15 | 240.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 234.70 | 239.86 | 240.38 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 240.00 | 238.05 | 238.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 247.40 | 240.23 | 239.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 243.50 | 243.68 | 241.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 243.00 | 243.68 | 241.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 240.95 | 243.13 | 241.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 240.95 | 243.13 | 241.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 239.50 | 242.41 | 241.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 239.50 | 242.41 | 241.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 238.55 | 241.64 | 241.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:15:00 | 238.90 | 241.64 | 241.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 239.75 | 240.88 | 240.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 233.70 | 238.92 | 239.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 15:15:00 | 236.75 | 236.09 | 237.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 09:15:00 | 242.62 | 236.09 | 237.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 240.19 | 236.91 | 237.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 237.88 | 236.93 | 237.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 14:15:00 | 240.08 | 238.60 | 238.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 240.08 | 238.60 | 238.44 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 235.36 | 238.06 | 238.23 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 11:15:00 | 238.81 | 238.38 | 238.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 246.66 | 240.33 | 239.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 247.35 | 249.61 | 246.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 247.35 | 249.61 | 246.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 247.35 | 249.61 | 246.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:30:00 | 253.46 | 248.96 | 247.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 10:00:00 | 253.59 | 248.96 | 247.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 13:00:00 | 253.40 | 251.35 | 248.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 263.67 | 264.81 | 264.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 263.67 | 264.81 | 264.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 11:15:00 | 261.69 | 264.19 | 264.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 263.51 | 262.17 | 263.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 263.51 | 262.17 | 263.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 263.51 | 262.17 | 263.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:00:00 | 263.51 | 262.17 | 263.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 263.70 | 262.48 | 263.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:00:00 | 263.70 | 262.48 | 263.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 262.88 | 262.56 | 263.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 12:45:00 | 261.74 | 262.39 | 263.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 14:15:00 | 261.92 | 262.49 | 263.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:15:00 | 261.09 | 262.36 | 262.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 11:00:00 | 261.71 | 262.39 | 262.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 262.64 | 262.10 | 262.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:45:00 | 263.67 | 262.10 | 262.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 262.90 | 262.26 | 262.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 264.89 | 262.26 | 262.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 264.80 | 262.77 | 262.77 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 264.80 | 262.77 | 262.77 | SL hit (close>static) qty=1.00 sl=264.19 alert=retest2 |

### Cycle 81 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 267.12 | 263.64 | 263.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 11:15:00 | 269.70 | 264.85 | 263.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 268.50 | 268.73 | 267.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 13:30:00 | 268.55 | 268.73 | 267.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 267.50 | 268.85 | 267.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 267.43 | 268.85 | 267.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 268.33 | 268.74 | 267.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 12:00:00 | 270.00 | 268.99 | 267.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 13:15:00 | 267.25 | 268.73 | 267.93 | SL hit (close<static) qty=1.00 sl=267.40 alert=retest2 |

### Cycle 82 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 264.68 | 267.01 | 267.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 12:15:00 | 263.05 | 265.66 | 266.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 11:15:00 | 266.80 | 265.43 | 265.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 11:15:00 | 266.80 | 265.43 | 265.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 266.80 | 265.43 | 265.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:00:00 | 266.80 | 265.43 | 265.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 268.65 | 266.07 | 266.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 269.50 | 266.86 | 266.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 10:15:00 | 271.25 | 271.62 | 269.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 11:00:00 | 271.25 | 271.62 | 269.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 271.25 | 271.52 | 270.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 12:45:00 | 270.10 | 271.52 | 270.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 269.75 | 271.20 | 270.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 15:00:00 | 269.75 | 271.20 | 270.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 269.70 | 270.90 | 270.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 271.45 | 270.90 | 270.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:45:00 | 272.40 | 271.06 | 270.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-13 14:15:00 | 198.58 | 2025-05-19 15:15:00 | 197.06 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-05-14 09:15:00 | 200.79 | 2025-05-19 15:15:00 | 197.06 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-05-14 11:00:00 | 198.00 | 2025-05-19 15:15:00 | 197.06 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-05-14 13:00:00 | 198.01 | 2025-05-19 15:15:00 | 197.06 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-05-15 10:15:00 | 198.81 | 2025-05-19 15:15:00 | 197.06 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-05-16 09:15:00 | 199.84 | 2025-05-19 15:15:00 | 197.06 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-05-23 09:45:00 | 204.00 | 2025-05-26 14:15:00 | 200.17 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-05-23 11:30:00 | 203.96 | 2025-05-26 14:15:00 | 200.17 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-05-26 09:15:00 | 204.36 | 2025-05-26 14:15:00 | 200.17 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-06-05 10:30:00 | 195.20 | 2025-06-05 12:15:00 | 199.17 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-06-11 09:15:00 | 199.98 | 2025-06-11 13:15:00 | 198.27 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-06-11 11:00:00 | 199.96 | 2025-06-11 13:15:00 | 198.27 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-06-16 15:15:00 | 194.57 | 2025-06-18 10:15:00 | 196.89 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-06-19 09:15:00 | 196.83 | 2025-06-19 11:15:00 | 194.73 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-06-24 11:15:00 | 199.47 | 2025-07-02 09:15:00 | 204.95 | STOP_HIT | 1.00 | 2.75% |
| BUY | retest2 | 2025-06-24 12:30:00 | 199.34 | 2025-07-02 09:15:00 | 204.95 | STOP_HIT | 1.00 | 2.81% |
| BUY | retest2 | 2025-06-25 09:15:00 | 201.03 | 2025-07-02 09:15:00 | 204.95 | STOP_HIT | 1.00 | 1.95% |
| SELL | retest2 | 2025-07-07 11:30:00 | 201.61 | 2025-07-09 09:15:00 | 205.85 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-07-08 11:00:00 | 201.81 | 2025-07-09 09:15:00 | 205.85 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-07-28 11:30:00 | 211.85 | 2025-07-30 14:15:00 | 214.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-07-28 12:15:00 | 212.20 | 2025-07-30 14:15:00 | 214.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-07-30 12:00:00 | 212.07 | 2025-07-30 14:15:00 | 214.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-07-30 12:45:00 | 212.07 | 2025-07-30 14:15:00 | 214.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-08-08 09:15:00 | 206.89 | 2025-08-13 09:15:00 | 211.10 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-08-19 10:45:00 | 221.36 | 2025-08-29 12:15:00 | 228.44 | STOP_HIT | 1.00 | 3.20% |
| BUY | retest2 | 2025-09-03 09:15:00 | 233.78 | 2025-09-09 15:15:00 | 241.00 | STOP_HIT | 1.00 | 3.09% |
| BUY | retest2 | 2025-09-17 15:15:00 | 243.25 | 2025-09-22 15:15:00 | 242.59 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-09-18 13:15:00 | 242.91 | 2025-09-22 15:15:00 | 242.59 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-09-22 10:15:00 | 243.45 | 2025-09-22 15:15:00 | 242.59 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-11-04 14:15:00 | 250.68 | 2025-11-10 09:15:00 | 256.68 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-11-04 15:00:00 | 250.28 | 2025-11-10 09:15:00 | 256.68 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-11-06 09:15:00 | 249.66 | 2025-11-10 09:15:00 | 256.68 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2025-11-20 12:00:00 | 269.25 | 2025-11-24 11:15:00 | 266.94 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-11-20 12:45:00 | 269.00 | 2025-11-24 11:15:00 | 266.94 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-11-21 09:15:00 | 270.04 | 2025-11-24 11:15:00 | 266.94 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-11-21 12:15:00 | 269.47 | 2025-11-24 11:15:00 | 266.94 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-11-26 10:15:00 | 271.30 | 2025-11-26 12:15:00 | 265.06 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-12-02 09:15:00 | 263.90 | 2025-12-08 12:15:00 | 250.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 09:15:00 | 263.90 | 2025-12-08 15:15:00 | 253.65 | STOP_HIT | 0.50 | 3.88% |
| BUY | retest2 | 2025-12-30 09:30:00 | 260.40 | 2026-01-07 14:15:00 | 266.00 | STOP_HIT | 1.00 | 2.15% |
| BUY | retest2 | 2025-12-30 10:00:00 | 260.00 | 2026-01-07 14:15:00 | 266.00 | STOP_HIT | 1.00 | 2.31% |
| BUY | retest2 | 2025-12-30 11:15:00 | 260.15 | 2026-01-07 14:15:00 | 266.00 | STOP_HIT | 1.00 | 2.25% |
| SELL | retest2 | 2026-01-22 11:15:00 | 240.90 | 2026-01-30 11:15:00 | 237.05 | STOP_HIT | 1.00 | 1.60% |
| BUY | retest2 | 2026-02-12 11:00:00 | 278.37 | 2026-02-13 10:15:00 | 274.12 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-02-13 09:30:00 | 278.53 | 2026-02-13 10:15:00 | 274.12 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2026-02-16 12:15:00 | 274.68 | 2026-02-24 10:15:00 | 260.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 09:15:00 | 274.22 | 2026-02-24 10:15:00 | 260.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-16 12:15:00 | 274.68 | 2026-02-24 15:15:00 | 262.10 | STOP_HIT | 0.50 | 4.58% |
| SELL | retest2 | 2026-02-17 09:15:00 | 274.22 | 2026-02-24 15:15:00 | 262.10 | STOP_HIT | 0.50 | 4.42% |
| BUY | retest2 | 2026-02-26 14:15:00 | 266.34 | 2026-02-27 09:15:00 | 263.90 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-02-27 12:30:00 | 266.60 | 2026-02-27 14:15:00 | 265.10 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2026-02-27 13:00:00 | 266.42 | 2026-02-27 14:15:00 | 265.10 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-03-04 09:15:00 | 253.50 | 2026-03-05 14:15:00 | 260.45 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2026-03-05 10:15:00 | 255.90 | 2026-03-05 14:15:00 | 260.45 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-03-05 11:00:00 | 255.60 | 2026-03-05 14:15:00 | 260.45 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-03-05 13:45:00 | 256.00 | 2026-03-05 14:15:00 | 260.45 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-03-19 10:45:00 | 240.30 | 2026-03-19 11:15:00 | 239.50 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2026-04-01 10:45:00 | 237.88 | 2026-04-01 14:15:00 | 240.08 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-04-08 09:30:00 | 253.46 | 2026-04-22 10:15:00 | 263.67 | STOP_HIT | 1.00 | 4.03% |
| BUY | retest2 | 2026-04-08 10:00:00 | 253.59 | 2026-04-22 10:15:00 | 263.67 | STOP_HIT | 1.00 | 3.97% |
| BUY | retest2 | 2026-04-08 13:00:00 | 253.40 | 2026-04-22 10:15:00 | 263.67 | STOP_HIT | 1.00 | 4.05% |
| SELL | retest2 | 2026-04-23 12:45:00 | 261.74 | 2026-04-27 09:15:00 | 264.80 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-04-23 14:15:00 | 261.92 | 2026-04-27 09:15:00 | 264.80 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-04-24 09:15:00 | 261.09 | 2026-04-27 09:15:00 | 264.80 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-04-24 11:00:00 | 261.71 | 2026-04-27 09:15:00 | 264.80 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2026-04-29 12:00:00 | 270.00 | 2026-04-29 13:15:00 | 267.25 | STOP_HIT | 1.00 | -1.02% |
