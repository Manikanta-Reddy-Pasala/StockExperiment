# Bank of India (BANKINDIA)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 139.85
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 222 |
| ALERT1 | 147 |
| ALERT2 | 144 |
| ALERT2_SKIP | 69 |
| ALERT3 | 371 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 161 |
| PARTIAL | 26 |
| TARGET_HIT | 13 |
| STOP_HIT | 153 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 192 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 98 / 94
- **Target hits / Stop hits / Partials:** 13 / 153 / 26
- **Avg / median % per leg:** 1.17% / 0.06%
- **Sum % (uncompounded):** 223.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 85 | 30 | 35.3% | 7 | 78 | 0 | 0.06% | 5.3% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -0.69% | -2.1% |
| BUY @ 3rd Alert (retest2) | 82 | 29 | 35.4% | 7 | 75 | 0 | 0.09% | 7.4% |
| SELL (all) | 107 | 68 | 63.6% | 6 | 75 | 26 | 2.04% | 218.4% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 1 | 1 | 2 | 6.49% | 25.9% |
| SELL @ 3rd Alert (retest2) | 103 | 64 | 62.1% | 5 | 74 | 24 | 1.87% | 192.4% |
| retest1 (combined) | 7 | 5 | 71.4% | 1 | 4 | 2 | 3.41% | 23.9% |
| retest2 (combined) | 185 | 93 | 50.3% | 12 | 149 | 24 | 1.08% | 199.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 09:15:00 | 75.10 | 74.02 | 73.89 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 11:15:00 | 73.85 | 74.52 | 74.57 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-06-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 14:15:00 | 74.65 | 74.44 | 74.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 09:15:00 | 74.90 | 74.52 | 74.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 14:15:00 | 74.70 | 74.71 | 74.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 14:15:00 | 74.70 | 74.71 | 74.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 14:15:00 | 74.70 | 74.71 | 74.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-05 09:15:00 | 75.25 | 74.68 | 74.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-05 15:15:00 | 74.30 | 74.55 | 74.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 15:15:00 | 74.30 | 74.55 | 74.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 10:15:00 | 73.95 | 74.36 | 74.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-07 09:15:00 | 74.70 | 74.23 | 74.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-07 09:15:00 | 74.70 | 74.23 | 74.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 09:15:00 | 74.70 | 74.23 | 74.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 09:45:00 | 74.60 | 74.23 | 74.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 10:15:00 | 74.65 | 74.32 | 74.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 10:30:00 | 74.70 | 74.32 | 74.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2023-06-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 12:15:00 | 74.50 | 74.39 | 74.39 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-06-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-07 13:15:00 | 74.25 | 74.36 | 74.38 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-06-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-08 09:15:00 | 74.90 | 74.45 | 74.41 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 13:15:00 | 74.20 | 74.41 | 74.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 09:15:00 | 74.00 | 74.27 | 74.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 11:15:00 | 73.35 | 73.34 | 73.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-12 12:00:00 | 73.35 | 73.34 | 73.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 73.55 | 73.31 | 73.54 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 09:15:00 | 74.05 | 73.69 | 73.65 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-06-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 14:15:00 | 73.30 | 73.85 | 73.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-15 15:15:00 | 73.00 | 73.68 | 73.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 09:15:00 | 73.75 | 73.69 | 73.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 09:15:00 | 73.75 | 73.69 | 73.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 73.75 | 73.69 | 73.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 09:30:00 | 73.90 | 73.69 | 73.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 10:15:00 | 73.40 | 73.64 | 73.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 10:30:00 | 73.60 | 73.64 | 73.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 14:15:00 | 73.20 | 73.40 | 73.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 14:45:00 | 73.65 | 73.40 | 73.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 09:15:00 | 74.30 | 73.53 | 73.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-19 09:45:00 | 74.35 | 73.53 | 73.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2023-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-19 10:15:00 | 74.30 | 73.69 | 73.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 11:15:00 | 75.45 | 74.04 | 73.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 09:15:00 | 73.90 | 74.53 | 74.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 09:15:00 | 73.90 | 74.53 | 74.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 73.90 | 74.53 | 74.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 10:00:00 | 73.90 | 74.53 | 74.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 10:15:00 | 74.25 | 74.47 | 74.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 10:30:00 | 74.00 | 74.47 | 74.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 11:15:00 | 74.35 | 74.45 | 74.23 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 11:15:00 | 73.90 | 74.20 | 74.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-21 13:15:00 | 73.65 | 74.06 | 74.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 13:15:00 | 70.85 | 70.76 | 71.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 14:00:00 | 70.85 | 70.76 | 71.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 71.50 | 70.97 | 71.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:30:00 | 71.65 | 70.97 | 71.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 71.40 | 71.05 | 71.52 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 15:15:00 | 72.40 | 71.66 | 71.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 09:15:00 | 72.70 | 71.86 | 71.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 15:15:00 | 71.90 | 72.22 | 72.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 15:15:00 | 71.90 | 72.22 | 72.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 15:15:00 | 71.90 | 72.22 | 72.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 14:15:00 | 72.95 | 72.21 | 72.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-04 11:15:00 | 80.25 | 77.40 | 75.43 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2023-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 10:15:00 | 78.75 | 79.29 | 79.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 11:15:00 | 78.05 | 79.05 | 79.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-12 09:15:00 | 79.15 | 78.25 | 78.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 09:15:00 | 79.15 | 78.25 | 78.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 79.15 | 78.25 | 78.50 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 11:15:00 | 79.40 | 78.71 | 78.68 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 13:15:00 | 77.25 | 78.75 | 78.83 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 14:15:00 | 79.90 | 78.39 | 78.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-18 09:15:00 | 80.20 | 78.98 | 78.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 11:15:00 | 79.10 | 79.18 | 78.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-18 12:00:00 | 79.10 | 79.18 | 78.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 09:15:00 | 79.05 | 79.23 | 78.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-19 09:30:00 | 79.20 | 79.23 | 78.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 10:15:00 | 78.75 | 79.13 | 78.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-19 10:30:00 | 78.85 | 79.13 | 78.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 11:15:00 | 79.10 | 79.13 | 78.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-19 12:30:00 | 79.80 | 79.28 | 79.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-25 15:15:00 | 82.80 | 83.54 | 83.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2023-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 15:15:00 | 82.80 | 83.54 | 83.56 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 09:15:00 | 83.90 | 83.61 | 83.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-26 11:15:00 | 84.55 | 83.82 | 83.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-27 15:15:00 | 84.40 | 84.60 | 84.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 15:15:00 | 84.40 | 84.60 | 84.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 15:15:00 | 84.40 | 84.60 | 84.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-28 09:30:00 | 85.20 | 84.68 | 84.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-28 10:00:00 | 85.00 | 84.68 | 84.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-28 13:15:00 | 83.80 | 84.38 | 84.32 | SL hit (close<static) qty=1.00 sl=84.25 alert=retest2 |

### Cycle 20 — SELL (started 2023-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 15:15:00 | 83.50 | 84.17 | 84.24 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 09:15:00 | 86.40 | 84.62 | 84.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 09:15:00 | 87.20 | 85.74 | 85.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 12:15:00 | 86.00 | 86.28 | 85.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-01 12:45:00 | 85.85 | 86.28 | 85.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 15:15:00 | 85.80 | 86.08 | 85.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 09:15:00 | 85.80 | 86.08 | 85.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 86.20 | 86.11 | 85.73 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 12:15:00 | 84.35 | 85.31 | 85.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 82.80 | 84.81 | 85.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 14:15:00 | 83.85 | 83.70 | 84.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-03 15:00:00 | 83.85 | 83.70 | 84.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 84.25 | 83.77 | 84.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 10:45:00 | 83.65 | 84.12 | 84.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 13:45:00 | 83.60 | 83.89 | 84.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 14:30:00 | 83.65 | 83.80 | 84.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-08 09:15:00 | 85.20 | 83.97 | 84.06 | SL hit (close>static) qty=1.00 sl=84.40 alert=retest2 |

### Cycle 23 — BUY (started 2023-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 10:15:00 | 84.85 | 84.15 | 84.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-08 14:15:00 | 86.00 | 84.93 | 84.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 10:15:00 | 86.85 | 86.96 | 86.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-10 10:30:00 | 86.80 | 86.96 | 86.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 13:15:00 | 85.95 | 86.70 | 86.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 13:45:00 | 85.80 | 86.70 | 86.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 14:15:00 | 86.25 | 86.61 | 86.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 14:30:00 | 85.95 | 86.61 | 86.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 15:15:00 | 86.50 | 86.59 | 86.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-11 09:15:00 | 86.55 | 86.59 | 86.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-14 10:00:00 | 86.55 | 87.38 | 87.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-14 10:45:00 | 87.90 | 87.44 | 87.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-22 10:15:00 | 89.00 | 89.71 | 89.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2023-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-22 10:15:00 | 89.00 | 89.71 | 89.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-22 14:15:00 | 88.60 | 89.20 | 89.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-23 09:15:00 | 89.30 | 89.11 | 89.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 09:15:00 | 89.30 | 89.11 | 89.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 89.30 | 89.11 | 89.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 09:30:00 | 89.15 | 89.11 | 89.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 10:15:00 | 89.65 | 89.22 | 89.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 11:00:00 | 89.65 | 89.22 | 89.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 11:15:00 | 89.60 | 89.29 | 89.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 12:00:00 | 89.60 | 89.29 | 89.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 12:15:00 | 89.55 | 89.34 | 89.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 13:00:00 | 89.55 | 89.34 | 89.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2023-08-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 13:15:00 | 90.60 | 89.60 | 89.54 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-08-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 13:15:00 | 88.80 | 89.44 | 89.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 14:15:00 | 88.15 | 89.18 | 89.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 10:15:00 | 87.20 | 86.84 | 87.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-28 11:00:00 | 87.20 | 86.84 | 87.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 11:15:00 | 87.50 | 86.98 | 87.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 11:30:00 | 87.70 | 86.98 | 87.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 12:15:00 | 87.45 | 87.07 | 87.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 12:30:00 | 87.70 | 87.07 | 87.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 13:15:00 | 87.60 | 87.18 | 87.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 14:00:00 | 87.60 | 87.18 | 87.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 14:15:00 | 87.00 | 87.14 | 87.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-28 15:15:00 | 86.60 | 87.14 | 87.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 10:15:00 | 86.60 | 87.03 | 87.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 14:15:00 | 86.30 | 86.98 | 87.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-30 10:15:00 | 88.05 | 87.09 | 87.20 | SL hit (close>static) qty=1.00 sl=87.65 alert=retest2 |

### Cycle 27 — BUY (started 2023-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 12:15:00 | 87.60 | 87.28 | 87.27 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-30 13:15:00 | 87.10 | 87.24 | 87.26 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 14:15:00 | 87.55 | 87.30 | 87.28 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-08-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 09:15:00 | 86.80 | 87.24 | 87.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 10:15:00 | 86.50 | 87.09 | 87.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 10:15:00 | 87.40 | 86.46 | 86.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 10:15:00 | 87.40 | 86.46 | 86.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 10:15:00 | 87.40 | 86.46 | 86.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 11:00:00 | 87.40 | 86.46 | 86.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 11:15:00 | 87.40 | 86.65 | 86.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 11:30:00 | 87.40 | 86.65 | 86.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2023-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 12:15:00 | 88.25 | 86.97 | 86.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 09:15:00 | 88.75 | 87.76 | 87.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 12:15:00 | 92.05 | 92.49 | 91.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-06 12:45:00 | 91.85 | 92.49 | 91.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 13:15:00 | 93.10 | 93.11 | 92.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-08 14:15:00 | 93.40 | 93.11 | 92.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 09:15:00 | 94.65 | 93.09 | 92.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-13 13:15:00 | 102.74 | 99.62 | 98.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2023-09-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 13:15:00 | 106.45 | 107.84 | 107.86 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 09:15:00 | 108.60 | 107.88 | 107.86 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-25 11:15:00 | 106.85 | 108.07 | 108.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-25 14:15:00 | 106.50 | 107.61 | 107.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 09:15:00 | 108.80 | 107.76 | 107.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 09:15:00 | 108.80 | 107.76 | 107.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 108.80 | 107.76 | 107.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:30:00 | 108.70 | 107.76 | 107.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 108.95 | 107.99 | 108.02 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 11:15:00 | 109.05 | 108.21 | 108.11 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 09:15:00 | 106.80 | 107.92 | 108.02 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 14:15:00 | 108.45 | 108.02 | 107.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 09:15:00 | 109.60 | 108.33 | 108.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 12:15:00 | 107.35 | 108.57 | 108.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 12:15:00 | 107.35 | 108.57 | 108.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 12:15:00 | 107.35 | 108.57 | 108.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 13:00:00 | 107.35 | 108.57 | 108.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 107.10 | 108.28 | 108.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 14:00:00 | 107.10 | 108.28 | 108.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 104.45 | 107.51 | 107.88 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 14:15:00 | 109.15 | 107.95 | 107.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 09:15:00 | 110.80 | 108.66 | 108.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 11:15:00 | 110.35 | 110.92 | 109.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 11:15:00 | 110.35 | 110.92 | 109.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 11:15:00 | 110.35 | 110.92 | 109.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 12:00:00 | 110.35 | 110.92 | 109.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 12:15:00 | 109.00 | 110.54 | 109.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 13:00:00 | 109.00 | 110.54 | 109.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 13:15:00 | 109.25 | 110.28 | 109.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 14:00:00 | 109.25 | 110.28 | 109.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 15:15:00 | 110.00 | 110.22 | 109.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 09:15:00 | 111.60 | 110.22 | 109.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 11:00:00 | 110.65 | 110.40 | 110.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-05 11:15:00 | 109.45 | 110.21 | 109.96 | SL hit (close<static) qty=1.00 sl=109.50 alert=retest2 |

### Cycle 40 — SELL (started 2023-10-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 15:15:00 | 109.50 | 109.83 | 109.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-06 09:15:00 | 108.75 | 109.61 | 109.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 10:15:00 | 105.80 | 105.70 | 106.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 12:15:00 | 106.80 | 106.07 | 106.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 12:15:00 | 106.80 | 106.07 | 106.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 12:30:00 | 106.90 | 106.07 | 106.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 13:15:00 | 107.10 | 106.28 | 106.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 14:00:00 | 107.10 | 106.28 | 106.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 14:15:00 | 107.25 | 106.47 | 106.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 14:45:00 | 107.45 | 106.47 | 106.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 15:15:00 | 107.50 | 106.68 | 107.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 09:15:00 | 108.35 | 106.68 | 107.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 108.45 | 107.03 | 107.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 09:30:00 | 108.95 | 107.03 | 107.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2023-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 10:15:00 | 108.10 | 107.25 | 107.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 11:15:00 | 108.60 | 107.52 | 107.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 13:15:00 | 106.95 | 107.56 | 107.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 13:15:00 | 106.95 | 107.56 | 107.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 13:15:00 | 106.95 | 107.56 | 107.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 14:00:00 | 106.95 | 107.56 | 107.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 14:15:00 | 106.90 | 107.43 | 107.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 14:30:00 | 106.95 | 107.43 | 107.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 10:15:00 | 107.20 | 107.35 | 107.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 10:30:00 | 107.35 | 107.35 | 107.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2023-10-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 11:15:00 | 107.10 | 107.30 | 107.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-12 12:15:00 | 106.75 | 107.19 | 107.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-12 13:15:00 | 107.20 | 107.19 | 107.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 13:15:00 | 107.20 | 107.19 | 107.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 13:15:00 | 107.20 | 107.19 | 107.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-12 13:45:00 | 107.25 | 107.19 | 107.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 14:15:00 | 107.75 | 107.30 | 107.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-12 15:00:00 | 107.75 | 107.30 | 107.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2023-10-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 15:15:00 | 107.50 | 107.34 | 107.33 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2023-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 09:15:00 | 106.60 | 107.19 | 107.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 14:15:00 | 106.20 | 106.73 | 106.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 106.60 | 105.86 | 106.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 09:15:00 | 106.60 | 105.86 | 106.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 106.60 | 105.86 | 106.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 13:15:00 | 105.80 | 106.03 | 106.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 10:30:00 | 105.60 | 105.76 | 106.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-20 12:15:00 | 100.51 | 102.25 | 103.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-20 12:15:00 | 100.32 | 102.25 | 103.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2023-10-23 11:15:00 | 95.22 | 98.63 | 101.04 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 45 — BUY (started 2023-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 12:15:00 | 96.00 | 93.32 | 93.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 15:15:00 | 97.00 | 94.84 | 93.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 14:15:00 | 95.95 | 96.69 | 95.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 14:15:00 | 95.95 | 96.69 | 95.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 14:15:00 | 95.95 | 96.69 | 95.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 15:00:00 | 95.95 | 96.69 | 95.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 15:15:00 | 95.50 | 96.45 | 95.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 09:15:00 | 96.45 | 96.45 | 95.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 97.20 | 96.60 | 95.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-01 10:45:00 | 97.70 | 96.88 | 96.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-01 13:30:00 | 97.90 | 97.38 | 96.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-10 15:15:00 | 103.05 | 103.27 | 103.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2023-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 15:15:00 | 103.05 | 103.27 | 103.29 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-11-12 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-12 18:15:00 | 104.20 | 103.45 | 103.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-13 11:15:00 | 106.25 | 103.95 | 103.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 09:15:00 | 107.10 | 107.70 | 106.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-16 10:00:00 | 107.10 | 107.70 | 106.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 11:15:00 | 106.55 | 107.37 | 106.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-16 12:00:00 | 106.55 | 107.37 | 106.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 12:15:00 | 107.05 | 107.31 | 106.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-16 14:00:00 | 107.30 | 107.31 | 106.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-17 09:15:00 | 103.90 | 106.48 | 106.48 | SL hit (close<static) qty=1.00 sl=106.50 alert=retest2 |

### Cycle 48 — SELL (started 2023-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 10:15:00 | 104.20 | 106.02 | 106.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 15:15:00 | 103.40 | 103.85 | 104.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 10:15:00 | 103.80 | 103.73 | 104.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-28 10:30:00 | 103.80 | 103.73 | 104.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 12:15:00 | 105.60 | 104.10 | 104.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 12:45:00 | 106.45 | 104.10 | 104.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2023-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 13:15:00 | 105.35 | 104.35 | 104.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 09:15:00 | 106.35 | 104.98 | 104.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 15:15:00 | 105.45 | 105.59 | 105.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 09:15:00 | 104.75 | 105.43 | 105.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 104.75 | 105.43 | 105.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 09:45:00 | 104.65 | 105.43 | 105.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 10:15:00 | 104.65 | 105.27 | 105.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 10:45:00 | 104.65 | 105.27 | 105.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2023-11-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 13:15:00 | 104.45 | 104.92 | 104.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-30 14:15:00 | 103.75 | 104.69 | 104.82 | Break + close below crossover candle low |

### Cycle 51 — BUY (started 2023-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 09:15:00 | 108.40 | 105.28 | 105.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 110.55 | 107.61 | 106.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 11:15:00 | 109.30 | 109.46 | 108.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-05 11:45:00 | 109.30 | 109.46 | 108.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 09:15:00 | 114.90 | 114.47 | 113.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-08 10:30:00 | 115.90 | 114.55 | 113.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-08 12:00:00 | 115.35 | 114.71 | 113.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 09:15:00 | 116.00 | 114.22 | 113.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 15:00:00 | 115.90 | 116.66 | 116.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-13 09:15:00 | 109.85 | 115.18 | 115.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2023-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 09:15:00 | 109.85 | 115.18 | 115.59 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2023-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 13:15:00 | 113.95 | 111.95 | 111.89 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 109.15 | 111.91 | 112.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 108.00 | 111.13 | 111.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 109.00 | 108.91 | 110.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 14:45:00 | 109.05 | 108.91 | 110.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 15:15:00 | 109.70 | 109.06 | 110.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:15:00 | 111.20 | 109.06 | 110.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 112.15 | 109.68 | 110.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:00:00 | 112.15 | 109.68 | 110.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 112.70 | 110.29 | 110.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 11:00:00 | 112.70 | 110.29 | 110.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2023-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 11:15:00 | 112.85 | 110.80 | 110.65 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2023-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-26 13:15:00 | 109.75 | 110.66 | 110.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-26 14:15:00 | 109.40 | 110.41 | 110.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-27 09:15:00 | 112.00 | 110.62 | 110.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 09:15:00 | 112.00 | 110.62 | 110.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 09:15:00 | 112.00 | 110.62 | 110.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 10:00:00 | 112.00 | 110.62 | 110.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 10:15:00 | 111.20 | 110.74 | 110.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-27 11:15:00 | 111.00 | 110.74 | 110.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-27 11:15:00 | 111.35 | 110.86 | 110.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2023-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 11:15:00 | 111.35 | 110.86 | 110.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 14:15:00 | 113.90 | 111.71 | 111.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 13:15:00 | 112.20 | 112.34 | 111.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-29 13:30:00 | 112.35 | 112.34 | 111.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 14:15:00 | 112.60 | 112.40 | 111.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 09:15:00 | 113.05 | 112.44 | 111.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 09:15:00 | 112.95 | 112.94 | 112.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 11:15:00 | 113.20 | 112.89 | 112.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-10 11:15:00 | 117.30 | 118.19 | 118.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 11:15:00 | 117.30 | 118.19 | 118.28 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-01-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 09:15:00 | 120.80 | 118.63 | 118.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-12 09:15:00 | 124.80 | 121.29 | 120.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 11:15:00 | 133.80 | 133.93 | 131.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-17 12:00:00 | 133.80 | 133.93 | 131.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 14:15:00 | 133.30 | 133.77 | 132.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 14:30:00 | 131.70 | 133.77 | 132.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 131.20 | 133.05 | 132.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 09:45:00 | 128.65 | 133.05 | 132.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 10:15:00 | 132.20 | 132.88 | 132.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 11:15:00 | 132.60 | 132.88 | 132.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 13:00:00 | 134.00 | 133.23 | 132.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-23 13:15:00 | 132.00 | 134.54 | 134.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 13:15:00 | 132.00 | 134.54 | 134.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 15:15:00 | 131.70 | 133.55 | 134.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 10:15:00 | 134.30 | 133.18 | 133.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 10:15:00 | 134.30 | 133.18 | 133.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 10:15:00 | 134.30 | 133.18 | 133.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 11:00:00 | 134.30 | 133.18 | 133.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 11:15:00 | 132.30 | 133.00 | 133.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 14:30:00 | 130.65 | 132.86 | 133.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 15:15:00 | 131.25 | 132.86 | 133.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-29 10:15:00 | 136.10 | 133.50 | 133.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2024-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 10:15:00 | 136.10 | 133.50 | 133.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-30 11:15:00 | 139.30 | 136.02 | 134.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 14:15:00 | 138.75 | 138.78 | 137.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-31 15:00:00 | 138.75 | 138.78 | 137.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 142.20 | 139.48 | 138.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 12:30:00 | 144.40 | 140.51 | 138.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-05 14:15:00 | 138.60 | 143.29 | 143.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2024-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 14:15:00 | 138.60 | 143.29 | 143.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-06 09:15:00 | 136.20 | 141.17 | 142.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 09:15:00 | 138.95 | 136.00 | 138.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 09:15:00 | 138.95 | 136.00 | 138.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 138.95 | 136.00 | 138.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 09:45:00 | 138.80 | 136.00 | 138.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 10:15:00 | 137.20 | 136.24 | 138.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 10:30:00 | 138.20 | 136.24 | 138.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 15:15:00 | 138.70 | 137.04 | 138.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 09:15:00 | 142.70 | 137.04 | 138.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 143.65 | 138.36 | 138.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 10:00:00 | 143.65 | 138.36 | 138.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2024-02-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 10:15:00 | 142.45 | 139.18 | 138.89 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 135.40 | 139.60 | 139.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 09:15:00 | 132.70 | 136.96 | 138.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 09:15:00 | 131.95 | 131.94 | 134.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 10:15:00 | 132.45 | 131.94 | 134.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 11:15:00 | 132.05 | 131.61 | 132.76 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 15:15:00 | 135.80 | 133.57 | 133.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 138.45 | 134.54 | 133.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 14:15:00 | 144.15 | 144.61 | 141.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-16 15:00:00 | 144.15 | 144.61 | 141.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 15:15:00 | 142.00 | 143.81 | 142.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 10:00:00 | 144.15 | 143.88 | 142.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 10:45:00 | 143.80 | 143.85 | 142.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 12:15:00 | 143.35 | 143.73 | 142.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 09:15:00 | 144.60 | 142.78 | 142.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 13:15:00 | 143.35 | 143.88 | 143.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 14:00:00 | 143.35 | 143.88 | 143.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 14:15:00 | 141.45 | 143.39 | 143.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 14:45:00 | 141.05 | 143.39 | 143.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-21 15:15:00 | 141.55 | 143.03 | 143.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 141.55 | 143.03 | 143.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 09:15:00 | 137.35 | 141.89 | 142.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 15:15:00 | 139.50 | 139.41 | 140.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-23 09:15:00 | 140.15 | 139.41 | 140.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 138.90 | 139.31 | 140.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 10:15:00 | 137.80 | 139.31 | 140.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 12:15:00 | 130.91 | 133.98 | 135.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-29 15:15:00 | 133.00 | 131.39 | 132.86 | SL hit (close>ema200) qty=0.50 sl=131.39 alert=retest2 |

### Cycle 67 — BUY (started 2024-03-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 12:15:00 | 136.10 | 133.55 | 133.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 09:15:00 | 136.85 | 135.25 | 134.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-07 15:15:00 | 145.60 | 145.98 | 144.32 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 09:15:00 | 147.35 | 145.98 | 144.32 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 12:30:00 | 146.20 | 146.49 | 145.18 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 13:15:00 | 144.10 | 146.01 | 145.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-03-11 13:15:00 | 144.10 | 146.01 | 145.08 | SL hit (close<ema400) qty=1.00 sl=145.08 alert=retest1 |

### Cycle 68 — SELL (started 2024-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 10:15:00 | 140.80 | 143.97 | 144.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 11:15:00 | 139.75 | 143.13 | 143.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-15 13:15:00 | 129.30 | 128.62 | 131.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-15 14:00:00 | 129.30 | 128.62 | 131.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 14:15:00 | 135.35 | 129.96 | 131.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 15:00:00 | 135.35 | 129.96 | 131.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 15:15:00 | 135.20 | 131.01 | 132.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 09:15:00 | 133.70 | 131.01 | 132.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-18 15:15:00 | 133.00 | 132.42 | 132.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2024-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 15:15:00 | 133.00 | 132.42 | 132.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-19 09:15:00 | 133.90 | 132.72 | 132.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 12:15:00 | 132.85 | 132.94 | 132.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 12:15:00 | 132.85 | 132.94 | 132.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 12:15:00 | 132.85 | 132.94 | 132.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 13:00:00 | 132.85 | 132.94 | 132.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 13:15:00 | 132.00 | 132.75 | 132.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 14:00:00 | 132.00 | 132.75 | 132.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 14:15:00 | 134.05 | 133.01 | 132.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 09:15:00 | 134.60 | 133.24 | 132.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-20 13:15:00 | 131.90 | 132.69 | 132.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 13:15:00 | 131.90 | 132.69 | 132.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 15:15:00 | 130.00 | 132.03 | 132.41 | Break + close below crossover candle low |

### Cycle 71 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 135.50 | 132.73 | 132.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 09:15:00 | 136.35 | 134.72 | 133.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 135.80 | 135.92 | 134.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-22 14:45:00 | 135.60 | 135.92 | 134.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 134.90 | 135.69 | 134.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:30:00 | 134.25 | 135.69 | 134.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 10:15:00 | 133.90 | 135.33 | 134.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 11:00:00 | 133.90 | 135.33 | 134.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 11:15:00 | 134.00 | 135.07 | 134.80 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 14:15:00 | 133.60 | 134.52 | 134.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 14:15:00 | 132.20 | 133.73 | 134.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 09:15:00 | 134.30 | 133.59 | 133.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 09:15:00 | 134.30 | 133.59 | 133.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 134.30 | 133.59 | 133.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 09:45:00 | 134.45 | 133.59 | 133.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 10:15:00 | 135.90 | 134.06 | 134.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 10:45:00 | 135.70 | 134.06 | 134.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 11:15:00 | 135.55 | 134.35 | 134.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 12:15:00 | 137.30 | 134.94 | 134.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-02 13:15:00 | 140.40 | 140.58 | 139.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-02 14:00:00 | 140.40 | 140.58 | 139.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 138.75 | 140.29 | 139.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 09:30:00 | 138.30 | 140.29 | 139.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 10:15:00 | 138.65 | 139.96 | 139.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 10:30:00 | 138.75 | 139.96 | 139.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 13:15:00 | 145.60 | 146.96 | 146.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 14:00:00 | 145.60 | 146.96 | 146.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 14:15:00 | 144.15 | 146.40 | 146.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 15:00:00 | 144.15 | 146.40 | 146.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2024-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 10:15:00 | 144.45 | 145.66 | 145.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 12:15:00 | 143.10 | 144.96 | 145.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 09:15:00 | 143.80 | 143.48 | 144.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-10 09:45:00 | 143.40 | 143.48 | 144.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 10:15:00 | 144.85 | 143.75 | 144.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 10:45:00 | 145.05 | 143.75 | 144.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 11:15:00 | 143.95 | 143.79 | 144.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 15:15:00 | 143.10 | 144.12 | 144.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-19 09:15:00 | 135.94 | 138.47 | 139.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-22 09:15:00 | 138.45 | 137.14 | 138.08 | SL hit (close>ema200) qty=0.50 sl=137.14 alert=retest2 |

### Cycle 75 — BUY (started 2024-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 12:15:00 | 140.10 | 138.72 | 138.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 13:15:00 | 140.40 | 139.06 | 138.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 14:15:00 | 155.00 | 155.34 | 153.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-30 15:00:00 | 155.00 | 155.34 | 153.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 10:15:00 | 153.20 | 154.84 | 153.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 11:00:00 | 153.20 | 154.84 | 153.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 11:15:00 | 152.40 | 154.35 | 153.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 11:30:00 | 152.55 | 154.35 | 153.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 14:15:00 | 151.55 | 153.22 | 153.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 15:00:00 | 151.55 | 153.22 | 153.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2024-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 15:15:00 | 151.60 | 152.89 | 152.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 09:15:00 | 150.30 | 152.37 | 152.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 15:15:00 | 146.50 | 146.40 | 148.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-07 09:15:00 | 146.90 | 146.40 | 148.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 143.85 | 142.70 | 144.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:30:00 | 145.00 | 142.70 | 144.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 142.15 | 142.59 | 144.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 14:30:00 | 141.75 | 142.49 | 143.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:30:00 | 141.30 | 142.24 | 143.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 14:15:00 | 141.55 | 141.95 | 142.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-13 09:15:00 | 127.58 | 136.23 | 139.08 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 11:15:00 | 125.05 | 124.27 | 124.25 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 14:15:00 | 123.15 | 124.11 | 124.19 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 09:15:00 | 125.70 | 124.35 | 124.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 13:15:00 | 127.05 | 125.67 | 125.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 12:15:00 | 131.05 | 131.78 | 130.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 12:15:00 | 131.05 | 131.78 | 130.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 131.05 | 131.78 | 130.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:30:00 | 130.30 | 131.78 | 130.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 130.50 | 131.53 | 130.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:00:00 | 130.50 | 131.53 | 130.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 130.65 | 131.35 | 130.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:30:00 | 130.65 | 131.35 | 130.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 129.65 | 131.01 | 130.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 130.20 | 131.01 | 130.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 129.75 | 130.76 | 130.54 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 12:15:00 | 130.00 | 130.34 | 130.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 15:15:00 | 129.40 | 129.99 | 130.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 129.20 | 127.85 | 128.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 13:15:00 | 129.20 | 127.85 | 128.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 129.20 | 127.85 | 128.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 129.20 | 127.85 | 128.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 128.65 | 128.01 | 128.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:15:00 | 128.75 | 128.01 | 128.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 128.75 | 128.16 | 128.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 134.00 | 128.16 | 128.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 135.60 | 129.64 | 129.26 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 118.45 | 130.29 | 130.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 113.35 | 126.90 | 129.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 117.65 | 117.49 | 121.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:00:00 | 117.65 | 117.49 | 121.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 120.45 | 117.83 | 120.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 120.55 | 117.83 | 120.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 122.00 | 118.67 | 120.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 122.00 | 118.67 | 120.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 11:15:00 | 119.15 | 118.76 | 120.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 12:15:00 | 119.00 | 118.76 | 120.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-10 09:15:00 | 125.21 | 121.19 | 120.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2024-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 09:15:00 | 125.21 | 121.19 | 120.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 09:15:00 | 125.66 | 124.31 | 123.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 12:15:00 | 124.29 | 124.34 | 123.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 13:00:00 | 124.29 | 124.34 | 123.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 125.60 | 124.65 | 124.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 10:15:00 | 125.84 | 124.65 | 124.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 11:30:00 | 126.08 | 125.15 | 124.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 15:15:00 | 125.99 | 125.48 | 124.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 09:15:00 | 123.05 | 125.08 | 124.74 | SL hit (close<static) qty=1.00 sl=124.11 alert=retest2 |

### Cycle 84 — SELL (started 2024-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 11:15:00 | 123.05 | 124.35 | 124.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 12:15:00 | 122.81 | 124.04 | 124.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 09:15:00 | 123.26 | 123.24 | 123.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 123.26 | 123.24 | 123.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 123.26 | 123.24 | 123.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:00:00 | 123.26 | 123.24 | 123.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 123.40 | 123.34 | 123.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 15:00:00 | 123.05 | 123.33 | 123.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 11:45:00 | 123.05 | 123.17 | 123.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 13:45:00 | 123.00 | 123.06 | 123.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 15:15:00 | 122.93 | 121.70 | 121.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 15:15:00 | 122.93 | 121.70 | 121.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 14:15:00 | 124.00 | 122.82 | 122.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 10:15:00 | 122.18 | 122.87 | 122.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 10:15:00 | 122.18 | 122.87 | 122.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 122.18 | 122.87 | 122.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:45:00 | 122.17 | 122.87 | 122.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 121.83 | 122.66 | 122.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:00:00 | 121.83 | 122.66 | 122.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 120.32 | 122.17 | 122.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 11:15:00 | 119.02 | 120.44 | 120.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 09:15:00 | 118.47 | 118.43 | 119.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 09:15:00 | 118.47 | 118.43 | 119.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 118.47 | 118.43 | 119.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 10:15:00 | 117.86 | 118.43 | 119.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 09:15:00 | 119.69 | 118.63 | 118.82 | SL hit (close>static) qty=1.00 sl=119.25 alert=retest2 |

### Cycle 87 — BUY (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 11:15:00 | 120.26 | 119.14 | 119.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 12:15:00 | 120.55 | 119.42 | 119.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 11:15:00 | 119.81 | 120.09 | 119.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 12:00:00 | 119.81 | 120.09 | 119.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 120.89 | 120.25 | 119.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:45:00 | 119.56 | 120.25 | 119.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 120.49 | 122.77 | 121.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 120.49 | 122.77 | 121.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 120.73 | 122.36 | 121.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 120.07 | 122.36 | 121.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 121.33 | 121.70 | 121.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:45:00 | 121.26 | 121.70 | 121.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2024-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 11:15:00 | 120.62 | 121.48 | 121.53 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 14:15:00 | 122.07 | 121.66 | 121.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 122.71 | 121.86 | 121.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 11:15:00 | 121.42 | 121.88 | 121.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 11:15:00 | 121.42 | 121.88 | 121.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 121.42 | 121.88 | 121.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:45:00 | 121.37 | 121.88 | 121.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 121.20 | 121.75 | 121.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:30:00 | 120.90 | 121.75 | 121.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2024-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 13:15:00 | 121.00 | 121.60 | 121.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 14:15:00 | 120.61 | 121.40 | 121.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 11:15:00 | 121.50 | 120.98 | 121.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 11:15:00 | 121.50 | 120.98 | 121.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 121.50 | 120.98 | 121.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 12:00:00 | 121.50 | 120.98 | 121.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 122.82 | 121.35 | 121.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 12:45:00 | 122.50 | 121.35 | 121.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 13:15:00 | 124.17 | 121.91 | 121.65 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 120.89 | 122.63 | 122.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 15:15:00 | 120.51 | 121.49 | 122.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 14:15:00 | 122.05 | 121.37 | 121.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 14:15:00 | 122.05 | 121.37 | 121.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 122.05 | 121.37 | 121.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 122.05 | 121.37 | 121.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 121.50 | 121.40 | 121.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 121.84 | 121.40 | 121.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 121.32 | 121.38 | 121.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:30:00 | 120.87 | 121.42 | 121.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 118.22 | 121.42 | 121.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 13:45:00 | 120.95 | 120.36 | 120.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-29 09:15:00 | 123.95 | 120.19 | 120.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 123.95 | 120.19 | 120.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 11:15:00 | 125.95 | 122.01 | 120.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 15:15:00 | 125.40 | 125.45 | 124.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 09:15:00 | 125.75 | 125.45 | 124.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 123.60 | 125.38 | 125.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:45:00 | 123.47 | 125.38 | 125.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 125.33 | 125.37 | 125.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 14:30:00 | 127.41 | 125.57 | 125.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 120.20 | 124.51 | 124.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 120.20 | 124.51 | 124.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 119.03 | 121.11 | 122.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 120.45 | 120.42 | 121.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 10:30:00 | 120.64 | 120.42 | 121.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 120.90 | 120.13 | 120.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 11:00:00 | 120.90 | 120.13 | 120.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 11:15:00 | 119.98 | 120.10 | 120.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 15:15:00 | 117.68 | 120.05 | 120.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 09:45:00 | 119.45 | 119.54 | 120.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 14:15:00 | 119.39 | 119.09 | 119.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 10:15:00 | 117.65 | 116.57 | 116.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 117.65 | 116.57 | 116.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 118.46 | 117.46 | 117.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 10:15:00 | 118.88 | 118.91 | 118.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 11:00:00 | 118.88 | 118.91 | 118.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 119.68 | 120.44 | 120.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:00:00 | 119.68 | 120.44 | 120.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 118.65 | 120.08 | 119.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:45:00 | 117.91 | 120.08 | 119.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2024-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 12:15:00 | 118.26 | 119.72 | 119.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 13:15:00 | 118.00 | 119.37 | 119.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 11:15:00 | 118.51 | 118.48 | 119.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-27 11:45:00 | 118.56 | 118.48 | 119.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 118.49 | 118.50 | 118.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 13:45:00 | 118.26 | 118.42 | 118.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 10:15:00 | 118.30 | 117.70 | 117.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 10:15:00 | 118.30 | 117.70 | 117.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 12:15:00 | 118.70 | 117.98 | 117.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 09:15:00 | 117.48 | 118.18 | 118.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 117.48 | 118.18 | 118.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 117.48 | 118.18 | 118.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:15:00 | 117.19 | 118.18 | 118.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 11:15:00 | 118.10 | 118.14 | 118.01 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2024-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 15:15:00 | 117.70 | 117.95 | 117.95 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 09:15:00 | 118.06 | 117.97 | 117.96 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 10:15:00 | 117.88 | 117.95 | 117.95 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 11:15:00 | 118.03 | 117.97 | 117.96 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 13:15:00 | 117.81 | 117.94 | 117.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 14:15:00 | 117.30 | 117.81 | 117.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 15:15:00 | 117.95 | 117.84 | 117.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 15:15:00 | 117.95 | 117.84 | 117.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 117.95 | 117.84 | 117.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 09:15:00 | 117.40 | 117.84 | 117.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 116.09 | 117.49 | 117.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 12:00:00 | 115.49 | 116.83 | 117.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 113.45 | 116.23 | 116.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 13:15:00 | 112.93 | 112.29 | 112.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 13:15:00 | 112.93 | 112.29 | 112.28 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 12:15:00 | 112.27 | 112.50 | 112.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 09:15:00 | 111.93 | 112.30 | 112.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 110.42 | 110.29 | 110.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 10:45:00 | 110.58 | 110.29 | 110.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 110.38 | 110.30 | 110.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 110.38 | 110.30 | 110.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 110.77 | 110.42 | 110.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:30:00 | 111.06 | 110.42 | 110.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 113.29 | 111.00 | 110.90 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 10:15:00 | 110.40 | 111.54 | 111.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 109.97 | 110.58 | 110.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 14:15:00 | 110.78 | 110.34 | 110.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 14:15:00 | 110.78 | 110.34 | 110.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 110.78 | 110.34 | 110.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:00:00 | 110.78 | 110.34 | 110.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 110.41 | 110.36 | 110.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:15:00 | 111.27 | 110.36 | 110.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 111.01 | 110.49 | 110.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:45:00 | 111.50 | 110.49 | 110.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2024-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 11:15:00 | 111.10 | 110.69 | 110.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 13:15:00 | 111.34 | 110.89 | 110.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 15:15:00 | 110.89 | 110.89 | 110.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 15:15:00 | 110.89 | 110.89 | 110.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 110.89 | 110.89 | 110.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:15:00 | 109.01 | 110.89 | 110.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 109.20 | 110.55 | 110.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 09:15:00 | 108.42 | 109.37 | 109.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 11:15:00 | 106.55 | 105.82 | 106.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 11:15:00 | 106.55 | 105.82 | 106.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 106.55 | 105.82 | 106.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:15:00 | 105.20 | 105.90 | 106.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 14:00:00 | 105.22 | 105.51 | 105.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 10:45:00 | 105.13 | 105.36 | 105.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 15:00:00 | 104.89 | 104.49 | 104.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 104.67 | 104.53 | 104.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 105.00 | 104.53 | 104.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 104.05 | 104.43 | 104.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:30:00 | 103.47 | 104.13 | 104.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 09:15:00 | 102.30 | 104.05 | 104.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 99.94 | 102.00 | 103.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 99.96 | 102.00 | 103.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 99.87 | 102.00 | 103.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 15:15:00 | 99.65 | 101.52 | 102.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 98.30 | 101.11 | 102.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 97.18 | 101.11 | 102.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 09:15:00 | 98.09 | 97.44 | 98.47 | SL hit (close>ema200) qty=0.50 sl=97.44 alert=retest2 |

### Cycle 109 — BUY (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 14:15:00 | 100.34 | 99.10 | 98.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 11:15:00 | 101.02 | 99.90 | 99.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 107.63 | 108.47 | 106.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 10:00:00 | 107.63 | 108.47 | 106.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 110.30 | 108.53 | 107.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 13:45:00 | 110.40 | 109.17 | 108.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 111.58 | 109.61 | 108.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 15:15:00 | 111.25 | 111.68 | 111.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 15:15:00 | 111.25 | 111.68 | 111.69 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 09:15:00 | 112.81 | 111.90 | 111.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-11 10:15:00 | 114.94 | 112.51 | 112.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-11 13:15:00 | 112.75 | 112.90 | 112.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-11 13:30:00 | 112.73 | 112.90 | 112.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 14:15:00 | 112.31 | 112.78 | 112.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 15:15:00 | 112.00 | 112.78 | 112.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 15:15:00 | 112.00 | 112.62 | 112.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:15:00 | 112.85 | 112.62 | 112.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 111.76 | 112.45 | 112.30 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 11:15:00 | 111.15 | 112.02 | 112.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 12:15:00 | 110.00 | 111.61 | 111.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 106.80 | 104.36 | 105.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 09:15:00 | 106.80 | 104.36 | 105.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 106.80 | 104.36 | 105.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 106.80 | 104.36 | 105.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 107.38 | 104.96 | 105.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 106.93 | 104.96 | 105.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2024-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 13:15:00 | 106.17 | 105.69 | 105.68 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 104.45 | 105.44 | 105.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 101.01 | 104.39 | 105.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 102.66 | 102.54 | 103.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 102.66 | 102.54 | 103.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 102.66 | 102.54 | 103.58 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 106.83 | 103.57 | 103.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 108.00 | 104.46 | 103.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 11:15:00 | 110.20 | 110.84 | 109.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 12:00:00 | 110.20 | 110.84 | 109.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 109.48 | 110.95 | 110.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:45:00 | 109.07 | 110.95 | 110.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 109.79 | 110.72 | 110.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:45:00 | 109.52 | 110.72 | 110.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 110.35 | 110.65 | 110.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 109.10 | 110.65 | 110.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 109.07 | 110.33 | 110.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:00:00 | 109.07 | 110.33 | 110.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 10:15:00 | 108.16 | 109.90 | 109.92 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 113.31 | 110.52 | 110.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 10:15:00 | 113.86 | 111.18 | 110.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 116.51 | 116.64 | 114.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 10:00:00 | 116.51 | 116.64 | 114.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 117.07 | 117.79 | 117.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 117.07 | 117.79 | 117.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 116.95 | 117.62 | 117.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:00:00 | 116.95 | 117.62 | 117.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 116.86 | 117.47 | 117.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 12:00:00 | 116.86 | 117.47 | 117.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 116.87 | 117.35 | 117.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 12:30:00 | 116.70 | 117.35 | 117.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 116.01 | 117.08 | 116.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 14:00:00 | 116.01 | 117.08 | 116.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2024-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 14:15:00 | 115.93 | 116.85 | 116.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 13:15:00 | 115.18 | 115.95 | 116.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 112.86 | 112.42 | 113.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 14:00:00 | 112.86 | 112.42 | 113.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 112.98 | 112.53 | 113.33 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2024-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 13:15:00 | 114.19 | 113.67 | 113.64 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 112.83 | 113.64 | 113.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 111.70 | 113.11 | 113.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 09:15:00 | 102.92 | 102.59 | 103.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 09:15:00 | 102.92 | 102.59 | 103.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 102.92 | 102.59 | 103.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:30:00 | 103.45 | 102.59 | 103.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 103.33 | 102.67 | 103.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 103.33 | 102.67 | 103.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 102.85 | 102.71 | 103.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:45:00 | 103.62 | 102.94 | 103.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 103.91 | 103.14 | 103.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:30:00 | 103.79 | 103.14 | 103.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 103.09 | 103.15 | 103.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 14:45:00 | 102.72 | 103.09 | 103.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:15:00 | 102.43 | 103.06 | 103.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 12:15:00 | 102.39 | 102.92 | 103.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 15:15:00 | 102.80 | 102.14 | 102.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 15:15:00 | 102.80 | 102.14 | 102.09 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 10:15:00 | 101.78 | 102.06 | 102.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 11:15:00 | 101.37 | 101.92 | 102.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 12:15:00 | 102.37 | 102.01 | 102.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 12:15:00 | 102.37 | 102.01 | 102.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 102.37 | 102.01 | 102.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 13:00:00 | 102.37 | 102.01 | 102.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 13:15:00 | 103.61 | 102.33 | 102.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 106.05 | 103.49 | 102.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 103.40 | 104.22 | 103.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 103.40 | 104.22 | 103.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 103.40 | 104.22 | 103.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 102.70 | 104.22 | 103.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 101.09 | 103.59 | 103.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 101.09 | 103.59 | 103.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 100.99 | 103.07 | 103.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 99.74 | 102.03 | 102.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 100.19 | 99.87 | 100.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 14:00:00 | 100.19 | 99.87 | 100.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 94.07 | 92.94 | 94.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:00:00 | 94.07 | 92.94 | 94.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 94.37 | 93.23 | 94.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:45:00 | 94.64 | 93.23 | 94.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 95.10 | 93.60 | 94.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 94.37 | 93.60 | 94.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 95.40 | 93.96 | 94.41 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 96.51 | 94.81 | 94.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 98.72 | 96.14 | 95.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 101.10 | 101.33 | 100.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 101.10 | 101.33 | 100.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 100.80 | 101.04 | 100.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:30:00 | 100.56 | 101.04 | 100.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 100.70 | 100.97 | 100.39 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 98.93 | 100.04 | 100.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 97.78 | 99.59 | 99.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 99.23 | 98.93 | 99.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 99.23 | 98.93 | 99.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 99.33 | 99.00 | 99.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 99.33 | 99.00 | 99.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 99.78 | 99.16 | 99.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 99.79 | 99.16 | 99.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 99.46 | 99.22 | 99.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:45:00 | 98.98 | 99.39 | 99.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 13:30:00 | 98.98 | 99.34 | 99.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 09:15:00 | 102.65 | 99.70 | 99.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-27 09:15:00 | 102.65 | 99.70 | 99.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-27 13:15:00 | 104.10 | 102.01 | 100.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 09:15:00 | 111.96 | 111.99 | 110.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 110.47 | 111.67 | 110.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 110.47 | 111.67 | 110.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 110.26 | 111.67 | 110.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 106.95 | 110.72 | 110.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 106.95 | 110.72 | 110.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 13:15:00 | 109.18 | 110.41 | 110.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 105.50 | 108.85 | 109.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 108.86 | 106.16 | 107.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 108.86 | 106.16 | 107.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 108.86 | 106.16 | 107.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 108.86 | 106.16 | 107.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 107.86 | 106.50 | 107.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 11:45:00 | 107.09 | 106.58 | 107.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 11:15:00 | 108.16 | 107.64 | 107.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 11:15:00 | 108.16 | 107.64 | 107.60 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 14:15:00 | 107.55 | 107.73 | 107.74 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 09:15:00 | 108.50 | 107.85 | 107.79 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 10:15:00 | 107.15 | 107.71 | 107.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 105.62 | 106.95 | 107.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 101.71 | 101.23 | 102.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:45:00 | 101.55 | 101.23 | 102.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 102.12 | 101.62 | 102.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 102.63 | 101.62 | 102.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 102.60 | 101.95 | 102.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:15:00 | 102.67 | 101.95 | 102.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 103.15 | 102.19 | 102.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 11:00:00 | 102.21 | 102.19 | 102.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 97.10 | 98.83 | 100.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 99.05 | 98.28 | 99.32 | SL hit (close>ema200) qty=0.50 sl=98.28 alert=retest2 |

### Cycle 133 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 99.60 | 98.34 | 98.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 101.10 | 99.20 | 98.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 102.28 | 102.78 | 101.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 102.28 | 102.78 | 101.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 101.51 | 102.53 | 101.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 101.51 | 102.53 | 101.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 100.80 | 102.18 | 101.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 12:00:00 | 100.80 | 102.18 | 101.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 12:15:00 | 101.24 | 101.99 | 101.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 15:00:00 | 101.68 | 101.78 | 101.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 100.68 | 101.50 | 101.18 | SL hit (close<static) qty=1.00 sl=100.80 alert=retest2 |

### Cycle 134 — SELL (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 14:15:00 | 100.90 | 101.03 | 101.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 99.38 | 100.61 | 100.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 94.81 | 94.78 | 96.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 94.81 | 94.78 | 96.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 95.76 | 94.96 | 95.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:45:00 | 95.82 | 94.96 | 95.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 96.03 | 95.24 | 95.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 96.91 | 95.24 | 95.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 96.89 | 95.57 | 96.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 96.89 | 95.57 | 96.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 96.37 | 95.88 | 96.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:00:00 | 96.37 | 95.88 | 96.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 96.00 | 95.91 | 96.05 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 97.15 | 96.14 | 96.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 98.13 | 96.54 | 96.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 12:15:00 | 97.95 | 97.95 | 97.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 12:45:00 | 97.94 | 97.95 | 97.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 98.20 | 97.97 | 97.56 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 09:15:00 | 96.70 | 97.33 | 97.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 11:15:00 | 95.71 | 96.87 | 97.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 94.89 | 94.77 | 95.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 14:30:00 | 94.50 | 94.77 | 95.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 94.54 | 94.81 | 95.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:15:00 | 94.45 | 94.81 | 95.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:45:00 | 94.25 | 94.70 | 95.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:15:00 | 94.29 | 94.36 | 94.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 13:00:00 | 94.46 | 94.37 | 94.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 94.66 | 94.36 | 94.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 94.78 | 94.36 | 94.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 94.59 | 94.41 | 94.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:00:00 | 94.59 | 94.41 | 94.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 94.86 | 94.50 | 94.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:30:00 | 94.95 | 94.50 | 94.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 94.75 | 94.55 | 94.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:15:00 | 95.13 | 94.55 | 94.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-17 13:15:00 | 95.22 | 94.68 | 94.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 13:15:00 | 95.22 | 94.68 | 94.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 97.10 | 95.20 | 94.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 106.42 | 106.89 | 105.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:00:00 | 106.42 | 106.89 | 105.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 105.70 | 106.50 | 105.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 105.12 | 106.50 | 105.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 105.32 | 106.26 | 105.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:00:00 | 105.32 | 106.26 | 105.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 105.98 | 106.21 | 105.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 14:45:00 | 106.07 | 106.16 | 105.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 106.44 | 106.09 | 105.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 09:30:00 | 106.93 | 106.26 | 105.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 14:15:00 | 106.03 | 106.31 | 106.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 107.59 | 106.57 | 106.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:30:00 | 106.00 | 106.57 | 106.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 107.52 | 107.67 | 107.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:00:00 | 107.52 | 107.67 | 107.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 107.12 | 107.56 | 107.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:00:00 | 107.12 | 107.56 | 107.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 106.90 | 107.43 | 107.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 108.01 | 107.43 | 107.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 109.30 | 107.80 | 107.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 12:00:00 | 110.59 | 108.55 | 107.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 10:45:00 | 110.36 | 109.64 | 108.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-03 13:15:00 | 116.68 | 112.95 | 111.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 105.01 | 110.79 | 111.28 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 109.85 | 108.28 | 108.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 13:15:00 | 111.10 | 109.48 | 108.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 13:15:00 | 119.65 | 119.92 | 118.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 13:45:00 | 119.67 | 119.92 | 118.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 115.80 | 118.77 | 118.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 115.80 | 118.77 | 118.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 116.00 | 118.22 | 117.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 115.52 | 118.22 | 117.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 12:15:00 | 116.91 | 117.65 | 117.73 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 09:15:00 | 118.38 | 117.80 | 117.78 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 115.74 | 117.79 | 117.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 114.53 | 117.14 | 117.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 12:15:00 | 117.00 | 116.98 | 117.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-25 12:45:00 | 117.10 | 116.98 | 117.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 13:15:00 | 116.83 | 116.95 | 117.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 13:45:00 | 117.70 | 116.95 | 117.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 115.53 | 116.66 | 117.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 15:15:00 | 115.24 | 116.66 | 117.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-28 09:15:00 | 117.82 | 116.67 | 117.08 | SL hit (close>static) qty=1.00 sl=117.30 alert=retest2 |

### Cycle 143 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 119.14 | 117.45 | 117.36 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 116.68 | 117.68 | 117.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 13:15:00 | 115.65 | 116.80 | 117.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 116.56 | 116.25 | 116.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 116.56 | 116.25 | 116.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 116.56 | 116.25 | 116.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:30:00 | 116.75 | 116.25 | 116.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 115.75 | 116.15 | 116.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 12:00:00 | 115.59 | 116.04 | 116.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 14:15:00 | 115.36 | 116.01 | 116.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 10:00:00 | 115.65 | 115.73 | 116.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 10:30:00 | 115.65 | 115.78 | 116.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 116.33 | 115.89 | 116.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 116.33 | 115.89 | 116.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 116.14 | 115.94 | 116.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:30:00 | 115.92 | 115.94 | 116.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 116.40 | 116.03 | 116.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:00:00 | 116.40 | 116.03 | 116.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 116.34 | 116.09 | 116.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:30:00 | 116.45 | 116.09 | 116.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 116.30 | 116.13 | 116.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 115.08 | 116.13 | 116.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 113.84 | 115.67 | 116.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:00:00 | 113.11 | 115.16 | 115.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 14:15:00 | 109.81 | 112.74 | 114.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 14:15:00 | 109.59 | 112.74 | 114.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 14:15:00 | 109.87 | 112.74 | 114.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 14:15:00 | 109.87 | 112.74 | 114.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 107.45 | 111.28 | 113.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-08 10:15:00 | 111.31 | 110.31 | 111.57 | SL hit (close>ema200) qty=0.50 sl=110.31 alert=retest2 |

### Cycle 145 — BUY (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 09:15:00 | 114.25 | 110.79 | 110.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 114.52 | 111.53 | 110.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 15:15:00 | 112.49 | 112.57 | 111.74 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:15:00 | 114.40 | 112.57 | 111.74 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 116.20 | 117.07 | 116.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 116.20 | 117.07 | 116.30 | SL hit (close<ema400) qty=1.00 sl=116.30 alert=retest1 |

### Cycle 146 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 115.68 | 116.69 | 116.76 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 117.17 | 116.75 | 116.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 117.53 | 117.00 | 116.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 11:15:00 | 116.85 | 117.03 | 116.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 11:15:00 | 116.85 | 117.03 | 116.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 116.85 | 117.03 | 116.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:30:00 | 116.70 | 117.03 | 116.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 116.88 | 117.00 | 116.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:45:00 | 116.60 | 117.00 | 116.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 116.70 | 117.07 | 116.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 116.70 | 117.07 | 116.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 116.75 | 117.01 | 116.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:30:00 | 116.61 | 117.01 | 116.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 116.95 | 117.00 | 116.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:30:00 | 116.86 | 117.00 | 116.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 116.34 | 116.87 | 116.91 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 118.88 | 117.23 | 117.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 14:15:00 | 119.90 | 118.08 | 117.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 13:15:00 | 118.65 | 118.72 | 118.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 14:00:00 | 118.65 | 118.72 | 118.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 118.22 | 118.76 | 118.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 118.22 | 118.76 | 118.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 118.32 | 118.67 | 118.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 117.87 | 118.67 | 118.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 119.38 | 118.81 | 118.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 13:00:00 | 120.55 | 119.16 | 118.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 10:15:00 | 125.36 | 126.68 | 126.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 10:15:00 | 125.36 | 126.68 | 126.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 12:15:00 | 125.26 | 126.24 | 126.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 121.32 | 120.91 | 121.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 121.32 | 120.91 | 121.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 121.80 | 121.30 | 121.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 120.70 | 121.16 | 121.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 114.66 | 117.38 | 118.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 13:15:00 | 114.15 | 113.59 | 114.88 | SL hit (close>ema200) qty=0.50 sl=113.59 alert=retest2 |

### Cycle 151 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 116.21 | 115.42 | 115.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 14:15:00 | 116.99 | 116.39 | 116.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 116.72 | 117.41 | 116.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 116.72 | 117.41 | 116.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 116.72 | 117.41 | 116.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 116.72 | 117.41 | 116.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 116.95 | 117.32 | 116.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 119.19 | 117.32 | 116.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 118.06 | 118.86 | 118.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 118.06 | 118.86 | 118.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 117.23 | 118.27 | 118.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 118.12 | 118.00 | 118.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 118.12 | 118.00 | 118.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 118.12 | 118.00 | 118.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:15:00 | 117.49 | 118.08 | 118.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 117.65 | 117.89 | 118.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 115.86 | 115.27 | 115.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 115.86 | 115.27 | 115.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 11:15:00 | 116.62 | 115.96 | 115.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 116.33 | 116.96 | 116.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 116.33 | 116.96 | 116.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 116.33 | 116.96 | 116.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 116.43 | 116.96 | 116.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 116.15 | 116.80 | 116.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:30:00 | 116.02 | 116.80 | 116.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 116.18 | 116.67 | 116.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:30:00 | 116.23 | 116.67 | 116.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 116.09 | 116.56 | 116.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 116.09 | 116.56 | 116.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 116.08 | 116.46 | 116.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 116.40 | 116.31 | 116.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 115.55 | 116.16 | 116.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 115.55 | 116.16 | 116.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 10:15:00 | 115.33 | 116.00 | 116.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 115.30 | 115.20 | 115.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 12:15:00 | 115.30 | 115.20 | 115.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 115.30 | 115.20 | 115.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:45:00 | 115.35 | 115.20 | 115.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 113.00 | 113.11 | 113.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 11:15:00 | 112.70 | 113.09 | 113.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 13:15:00 | 114.18 | 113.14 | 113.46 | SL hit (close>static) qty=1.00 sl=113.70 alert=retest2 |

### Cycle 155 — BUY (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 15:15:00 | 115.10 | 113.76 | 113.70 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 113.17 | 113.70 | 113.73 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 09:15:00 | 115.00 | 113.78 | 113.74 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 11:15:00 | 112.96 | 113.59 | 113.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 112.84 | 113.44 | 113.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 112.08 | 111.92 | 112.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 112.08 | 111.92 | 112.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 115.39 | 112.72 | 112.74 | EMA400 retest candle locked (from downside) |

### Cycle 159 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 115.79 | 113.33 | 113.02 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 111.95 | 113.31 | 113.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 110.45 | 111.96 | 112.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 110.93 | 110.37 | 111.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 11:15:00 | 110.93 | 110.37 | 111.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 110.93 | 110.37 | 111.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:45:00 | 110.71 | 110.37 | 111.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 110.95 | 110.49 | 111.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 110.95 | 110.49 | 111.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 111.58 | 110.71 | 111.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 111.58 | 110.71 | 111.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 111.83 | 110.93 | 111.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 111.69 | 110.93 | 111.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 111.38 | 111.19 | 111.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 111.43 | 111.19 | 111.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 111.80 | 111.31 | 111.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:00:00 | 111.80 | 111.31 | 111.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 111.07 | 111.26 | 111.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 13:15:00 | 110.74 | 111.26 | 111.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:00:00 | 110.37 | 110.88 | 111.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 112.09 | 111.17 | 111.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 12:15:00 | 112.09 | 111.17 | 111.17 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 110.84 | 111.23 | 111.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 109.88 | 110.96 | 111.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 111.81 | 111.08 | 111.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 111.81 | 111.08 | 111.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 111.81 | 111.08 | 111.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 111.81 | 111.08 | 111.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 111.84 | 111.23 | 111.20 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 14:15:00 | 110.81 | 111.13 | 111.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 110.39 | 110.98 | 111.10 | Break + close below crossover candle low |

### Cycle 165 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 112.23 | 111.23 | 111.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 113.04 | 111.99 | 111.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 113.24 | 113.47 | 112.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 11:15:00 | 113.24 | 113.47 | 112.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 113.24 | 113.47 | 112.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 113.24 | 113.47 | 112.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 113.25 | 113.45 | 113.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 113.44 | 113.45 | 113.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 113.38 | 113.44 | 113.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 114.35 | 113.27 | 113.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 15:15:00 | 115.40 | 115.89 | 115.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 115.40 | 115.89 | 115.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 115.01 | 115.51 | 115.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 111.47 | 111.22 | 112.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:45:00 | 111.45 | 111.22 | 112.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 111.18 | 110.86 | 111.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 111.91 | 110.86 | 111.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 111.70 | 111.15 | 111.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 111.70 | 111.15 | 111.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 111.86 | 111.29 | 111.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 111.86 | 111.29 | 111.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 111.96 | 111.42 | 111.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 111.60 | 111.42 | 111.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 113.29 | 111.80 | 111.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 14:15:00 | 113.73 | 113.18 | 112.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 113.30 | 113.35 | 112.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:30:00 | 113.20 | 113.35 | 112.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 113.56 | 113.39 | 113.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:15:00 | 113.00 | 113.39 | 113.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 113.08 | 113.33 | 113.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:45:00 | 113.08 | 113.33 | 113.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 112.51 | 113.16 | 112.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 112.51 | 113.16 | 112.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 112.15 | 112.96 | 112.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 112.15 | 112.96 | 112.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 112.17 | 112.80 | 112.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 111.58 | 112.37 | 112.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 112.77 | 112.38 | 112.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 112.77 | 112.38 | 112.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 112.77 | 112.38 | 112.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 112.77 | 112.38 | 112.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 112.53 | 112.41 | 112.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:30:00 | 112.48 | 112.41 | 112.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 112.95 | 112.52 | 112.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 112.81 | 112.52 | 112.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 113.36 | 112.69 | 112.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 11:15:00 | 113.73 | 113.00 | 112.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 14:15:00 | 113.15 | 113.21 | 112.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 15:00:00 | 113.15 | 113.21 | 112.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 112.83 | 113.14 | 112.99 | EMA400 retest candle locked (from upside) |

### Cycle 170 — SELL (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 14:15:00 | 112.71 | 112.88 | 112.91 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 114.75 | 113.23 | 113.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 116.55 | 113.89 | 113.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 10:15:00 | 117.50 | 118.02 | 116.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 11:00:00 | 117.50 | 118.02 | 116.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 116.59 | 117.74 | 116.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:45:00 | 116.69 | 117.74 | 116.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 116.73 | 117.53 | 116.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:45:00 | 116.56 | 117.53 | 116.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 117.37 | 117.49 | 116.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:45:00 | 117.21 | 117.49 | 116.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 117.43 | 117.53 | 117.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:30:00 | 117.51 | 117.53 | 117.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 117.10 | 117.44 | 117.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 117.10 | 117.44 | 117.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 118.00 | 117.55 | 117.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:15:00 | 119.06 | 117.51 | 117.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:45:00 | 119.09 | 117.68 | 117.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 13:30:00 | 119.18 | 118.21 | 117.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 12:45:00 | 118.95 | 118.97 | 118.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 118.65 | 118.91 | 118.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:45:00 | 118.93 | 118.93 | 118.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 11:15:00 | 120.11 | 120.99 | 121.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 11:15:00 | 120.11 | 120.99 | 121.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 119.62 | 120.60 | 120.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 118.98 | 118.15 | 119.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 118.98 | 118.15 | 119.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 118.91 | 118.30 | 119.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 118.50 | 118.34 | 119.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 118.71 | 118.76 | 119.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 122.01 | 119.45 | 119.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — BUY (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 09:15:00 | 122.01 | 119.45 | 119.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 14:15:00 | 123.05 | 121.53 | 120.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 10:15:00 | 123.88 | 124.03 | 122.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 10:45:00 | 124.05 | 124.03 | 122.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 125.01 | 126.04 | 125.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:45:00 | 124.76 | 126.04 | 125.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 123.94 | 125.62 | 125.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 123.94 | 125.62 | 125.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 124.75 | 125.14 | 125.17 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 125.80 | 125.26 | 125.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 127.31 | 125.94 | 125.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 14:15:00 | 126.56 | 126.61 | 126.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 15:00:00 | 126.56 | 126.61 | 126.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 126.05 | 126.51 | 126.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 125.93 | 126.51 | 126.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 126.45 | 126.50 | 126.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:45:00 | 125.81 | 126.50 | 126.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 126.02 | 126.40 | 126.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:30:00 | 125.94 | 126.40 | 126.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 126.08 | 126.34 | 126.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:30:00 | 126.08 | 126.34 | 126.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 126.45 | 126.36 | 126.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 14:15:00 | 127.19 | 126.36 | 126.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 125.63 | 126.34 | 126.23 | SL hit (close<static) qty=1.00 sl=126.01 alert=retest2 |

### Cycle 176 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 124.61 | 125.99 | 126.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 124.07 | 125.61 | 125.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 125.43 | 124.90 | 125.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 125.43 | 124.90 | 125.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 125.43 | 124.90 | 125.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 125.43 | 124.90 | 125.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 126.15 | 125.15 | 125.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 126.15 | 125.15 | 125.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 126.42 | 125.40 | 125.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 126.42 | 125.40 | 125.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 126.55 | 125.63 | 125.61 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 15:15:00 | 125.21 | 125.63 | 125.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 09:15:00 | 124.75 | 125.45 | 125.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 10:15:00 | 126.01 | 125.57 | 125.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 10:15:00 | 126.01 | 125.57 | 125.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 126.01 | 125.57 | 125.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 126.01 | 125.57 | 125.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 125.50 | 125.55 | 125.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 12:30:00 | 124.55 | 125.02 | 125.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 129.49 | 125.49 | 125.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 129.49 | 125.49 | 125.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 129.90 | 128.37 | 127.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 12:15:00 | 134.00 | 134.35 | 132.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 13:00:00 | 134.00 | 134.35 | 132.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 140.29 | 140.56 | 139.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:15:00 | 139.71 | 140.56 | 139.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 138.86 | 140.22 | 139.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:00:00 | 138.86 | 140.22 | 139.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 138.91 | 139.96 | 139.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:45:00 | 138.70 | 139.96 | 139.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 140.45 | 140.76 | 140.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:45:00 | 140.31 | 140.76 | 140.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 139.89 | 140.59 | 140.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 139.89 | 140.59 | 140.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 140.00 | 140.47 | 140.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 142.08 | 140.47 | 140.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 140.12 | 141.33 | 141.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 140.12 | 141.33 | 141.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 139.36 | 140.50 | 140.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 141.46 | 140.69 | 140.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 10:15:00 | 141.46 | 140.69 | 140.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 141.46 | 140.69 | 140.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:45:00 | 141.01 | 140.69 | 140.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 141.85 | 140.93 | 141.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:45:00 | 141.81 | 140.93 | 141.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 143.08 | 141.36 | 141.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 13:15:00 | 143.92 | 141.87 | 141.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 143.35 | 144.60 | 143.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 143.35 | 144.60 | 143.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 143.35 | 144.60 | 143.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 143.35 | 144.60 | 143.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 143.65 | 144.41 | 143.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:30:00 | 142.40 | 144.41 | 143.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 144.84 | 144.50 | 143.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 13:00:00 | 145.44 | 144.69 | 143.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 145.35 | 145.49 | 145.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 10:15:00 | 145.47 | 147.33 | 147.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 145.47 | 147.33 | 147.50 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2025-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 15:15:00 | 147.21 | 146.85 | 146.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 150.85 | 147.65 | 147.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 147.53 | 148.45 | 147.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 147.53 | 148.45 | 147.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 147.53 | 148.45 | 147.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:45:00 | 147.26 | 148.45 | 147.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 147.54 | 148.27 | 147.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:15:00 | 147.28 | 148.27 | 147.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 146.23 | 147.86 | 147.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 146.23 | 147.86 | 147.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 147.25 | 147.74 | 147.74 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 10:15:00 | 148.00 | 147.53 | 147.47 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 145.87 | 147.19 | 147.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 142.55 | 146.01 | 146.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 10:15:00 | 142.07 | 141.81 | 143.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:45:00 | 142.01 | 141.81 | 143.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 142.64 | 141.76 | 142.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 142.64 | 141.76 | 142.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 143.12 | 142.03 | 142.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:15:00 | 143.47 | 142.03 | 142.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 142.28 | 142.08 | 142.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 141.60 | 142.58 | 142.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 10:15:00 | 141.05 | 140.88 | 140.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 141.05 | 140.88 | 140.88 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 139.66 | 140.63 | 140.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 138.47 | 139.84 | 140.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 140.55 | 139.67 | 140.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 140.55 | 139.67 | 140.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 140.55 | 139.67 | 140.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 140.01 | 139.67 | 140.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 141.05 | 139.95 | 140.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 11:30:00 | 140.36 | 140.23 | 140.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 12:15:00 | 141.20 | 140.43 | 140.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 141.20 | 140.43 | 140.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 14:15:00 | 141.94 | 141.38 | 141.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 140.83 | 141.30 | 141.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 140.83 | 141.30 | 141.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 140.83 | 141.30 | 141.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:45:00 | 140.45 | 141.30 | 141.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 140.39 | 141.12 | 141.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 140.39 | 141.12 | 141.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 139.82 | 140.86 | 140.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 139.59 | 140.60 | 140.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 141.68 | 140.31 | 140.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 141.68 | 140.31 | 140.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 141.68 | 140.31 | 140.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:30:00 | 142.27 | 140.31 | 140.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 141.93 | 140.63 | 140.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 141.93 | 140.63 | 140.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — BUY (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 11:15:00 | 142.18 | 140.94 | 140.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 11:15:00 | 143.11 | 142.07 | 141.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 13:15:00 | 142.03 | 142.26 | 141.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-18 14:00:00 | 142.03 | 142.26 | 141.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 142.34 | 142.28 | 141.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:45:00 | 141.87 | 142.28 | 141.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 142.00 | 142.22 | 141.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 143.27 | 142.22 | 141.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 10:30:00 | 142.51 | 142.20 | 141.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 12:30:00 | 142.51 | 142.25 | 141.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 13:15:00 | 142.62 | 142.25 | 141.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 12:15:00 | 142.58 | 142.65 | 142.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 13:00:00 | 142.58 | 142.65 | 142.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 142.71 | 142.65 | 142.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 14:45:00 | 142.58 | 142.65 | 142.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 142.65 | 142.65 | 142.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 143.03 | 142.65 | 142.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:45:00 | 142.85 | 142.68 | 142.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 12:15:00 | 141.93 | 142.45 | 142.40 | SL hit (close<static) qty=1.00 sl=142.26 alert=retest2 |

### Cycle 192 — SELL (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 13:15:00 | 141.84 | 142.33 | 142.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 14:15:00 | 140.97 | 142.05 | 142.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 13:15:00 | 140.10 | 140.06 | 140.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 13:45:00 | 140.05 | 140.06 | 140.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 140.18 | 139.59 | 139.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 138.40 | 139.59 | 139.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 140.53 | 139.91 | 140.02 | SL hit (close>static) qty=1.00 sl=140.36 alert=retest2 |

### Cycle 193 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 141.97 | 140.44 | 140.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 14:15:00 | 142.15 | 140.78 | 140.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 148.54 | 149.22 | 147.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 148.54 | 149.22 | 147.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 150.80 | 150.96 | 149.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 150.43 | 150.96 | 149.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 150.32 | 150.81 | 149.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 151.21 | 150.79 | 149.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:00:00 | 151.05 | 150.96 | 150.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 147.33 | 150.18 | 150.09 | SL hit (close<static) qty=1.00 sl=149.71 alert=retest2 |

### Cycle 194 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 148.12 | 149.77 | 149.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 145.75 | 147.55 | 148.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 145.59 | 145.42 | 146.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 145.91 | 145.42 | 146.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 146.49 | 145.64 | 146.59 | EMA400 retest candle locked (from downside) |

### Cycle 195 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 148.72 | 147.11 | 146.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 151.70 | 148.08 | 147.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 10:15:00 | 160.85 | 161.17 | 158.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 11:00:00 | 160.85 | 161.17 | 158.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 158.24 | 159.94 | 158.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:45:00 | 158.13 | 159.94 | 158.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 158.49 | 159.65 | 158.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 11:45:00 | 157.25 | 159.65 | 158.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 156.65 | 159.03 | 158.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 14:00:00 | 156.65 | 159.03 | 158.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 157.37 | 158.70 | 158.59 | EMA400 retest candle locked (from upside) |

### Cycle 196 — SELL (started 2026-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 15:15:00 | 157.45 | 158.45 | 158.49 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 165.10 | 159.78 | 159.09 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 15:15:00 | 159.64 | 161.62 | 161.63 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 161.90 | 161.68 | 161.65 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 161.09 | 161.56 | 161.60 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 162.18 | 161.69 | 161.65 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2026-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 12:15:00 | 160.60 | 161.47 | 161.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 160.24 | 161.22 | 161.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 162.98 | 161.57 | 161.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 162.98 | 161.57 | 161.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 162.98 | 161.57 | 161.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 162.98 | 161.57 | 161.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 163.45 | 161.95 | 161.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 164.33 | 162.54 | 162.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 11:15:00 | 164.84 | 166.03 | 164.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 12:00:00 | 164.84 | 166.03 | 164.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 163.60 | 165.54 | 164.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 13:00:00 | 163.60 | 165.54 | 164.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 164.91 | 165.42 | 164.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:45:00 | 165.76 | 165.33 | 164.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 165.84 | 165.43 | 164.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 09:15:00 | 159.78 | 163.83 | 164.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 159.78 | 163.83 | 164.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 154.52 | 161.38 | 163.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 152.00 | 151.68 | 155.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 09:15:00 | 156.65 | 151.68 | 155.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 156.65 | 152.67 | 155.23 | EMA400 retest candle locked (from downside) |

### Cycle 205 — BUY (started 2026-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 15:15:00 | 157.70 | 156.33 | 156.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 158.56 | 156.78 | 156.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 12:15:00 | 161.70 | 162.24 | 161.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 13:00:00 | 161.70 | 162.24 | 161.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 163.93 | 166.60 | 165.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:45:00 | 163.45 | 166.60 | 165.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 165.87 | 166.46 | 165.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:00:00 | 166.35 | 166.32 | 165.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 11:15:00 | 163.74 | 165.68 | 165.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 163.74 | 165.68 | 165.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 161.79 | 164.66 | 165.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 11:15:00 | 163.91 | 162.94 | 163.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 11:15:00 | 163.91 | 162.94 | 163.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 163.91 | 162.94 | 163.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:00:00 | 163.91 | 162.94 | 163.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 164.93 | 163.34 | 163.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:00:00 | 164.93 | 163.34 | 163.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 165.71 | 163.81 | 164.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:00:00 | 165.71 | 163.81 | 164.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 165.71 | 164.46 | 164.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 166.91 | 164.95 | 164.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 171.68 | 171.81 | 170.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 171.68 | 171.81 | 170.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 169.08 | 171.16 | 170.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 169.08 | 171.16 | 170.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 168.89 | 170.71 | 170.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 170.08 | 170.71 | 170.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 170.64 | 170.69 | 170.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 11:15:00 | 171.80 | 170.69 | 170.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 13:15:00 | 171.50 | 171.00 | 170.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 14:45:00 | 171.59 | 171.23 | 170.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 174.23 | 171.18 | 170.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 174.15 | 174.79 | 173.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:00:00 | 174.15 | 174.79 | 173.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 174.91 | 174.82 | 174.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 12:15:00 | 175.27 | 174.82 | 174.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 15:00:00 | 175.50 | 174.81 | 174.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 176.71 | 174.89 | 174.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 10:30:00 | 175.36 | 175.11 | 174.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 176.16 | 176.48 | 175.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 176.16 | 176.48 | 175.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 176.02 | 176.39 | 175.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 173.23 | 176.39 | 175.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 172.84 | 175.68 | 175.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 172.84 | 175.68 | 175.56 | SL hit (close<static) qty=1.00 sl=173.72 alert=retest2 |

### Cycle 208 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 173.11 | 175.17 | 175.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 163.94 | 171.43 | 173.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 165.57 | 165.28 | 168.46 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:30:00 | 164.46 | 164.98 | 167.77 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 15:00:00 | 163.43 | 164.11 | 166.64 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 156.24 | 159.39 | 162.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 155.26 | 159.39 | 162.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-03-09 10:15:00 | 148.01 | 157.50 | 161.57 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 209 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 152.89 | 151.53 | 151.34 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 148.15 | 151.34 | 151.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 145.85 | 148.99 | 150.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 149.59 | 148.28 | 149.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 149.59 | 148.28 | 149.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 149.59 | 148.28 | 149.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 151.22 | 148.28 | 149.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 151.66 | 148.96 | 149.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 151.44 | 148.96 | 149.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 150.53 | 149.27 | 149.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 149.85 | 149.27 | 149.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 150.99 | 150.04 | 150.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — BUY (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 13:15:00 | 150.99 | 150.04 | 150.04 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 144.43 | 149.02 | 149.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 143.81 | 147.18 | 148.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 145.75 | 145.31 | 146.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 145.75 | 145.31 | 146.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 145.75 | 145.31 | 146.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 145.11 | 145.37 | 146.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 147.68 | 145.91 | 146.84 | SL hit (close>static) qty=1.00 sl=147.60 alert=retest2 |

### Cycle 213 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 150.15 | 147.59 | 147.34 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 143.73 | 147.37 | 147.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 141.73 | 144.82 | 146.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 140.51 | 139.97 | 142.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 140.51 | 139.97 | 142.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 140.51 | 139.97 | 142.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 140.31 | 139.97 | 142.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:45:00 | 140.34 | 140.50 | 142.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 143.40 | 141.08 | 142.42 | SL hit (close>static) qty=1.00 sl=142.81 alert=retest2 |

### Cycle 215 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 142.75 | 140.95 | 140.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 143.02 | 141.37 | 140.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 138.08 | 141.00 | 140.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 138.08 | 141.00 | 140.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 138.08 | 141.00 | 140.88 | EMA400 retest candle locked (from upside) |

### Cycle 216 — SELL (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 10:15:00 | 137.54 | 140.31 | 140.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 11:15:00 | 137.24 | 139.70 | 140.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 145.86 | 139.80 | 139.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 145.86 | 139.80 | 139.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 145.86 | 139.80 | 139.91 | EMA400 retest candle locked (from downside) |

### Cycle 217 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 147.25 | 141.29 | 140.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 147.99 | 145.45 | 144.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 142.17 | 145.97 | 145.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 142.17 | 145.97 | 145.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 142.17 | 145.97 | 145.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:15:00 | 144.62 | 145.03 | 144.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 13:15:00 | 144.50 | 144.84 | 144.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 218 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 144.50 | 144.84 | 144.86 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 15:15:00 | 145.35 | 144.96 | 144.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 148.02 | 145.57 | 145.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 12:15:00 | 148.41 | 148.54 | 147.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 13:00:00 | 148.41 | 148.54 | 147.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 149.20 | 148.80 | 147.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 10:30:00 | 149.89 | 148.57 | 148.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 149.77 | 148.79 | 148.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 13:00:00 | 150.72 | 151.85 | 151.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 148.38 | 150.77 | 150.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 220 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 148.38 | 150.77 | 150.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 146.76 | 149.97 | 150.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 148.22 | 147.93 | 148.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 13:00:00 | 148.22 | 147.93 | 148.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 145.55 | 145.06 | 146.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:15:00 | 144.41 | 145.50 | 146.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:15:00 | 137.19 | 138.98 | 140.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 141.06 | 139.19 | 140.00 | SL hit (close>ema200) qty=0.50 sl=139.19 alert=retest2 |

### Cycle 221 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 142.35 | 140.37 | 140.25 | EMA200 above EMA400 |

### Cycle 222 — SELL (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 10:15:00 | 139.95 | 140.32 | 140.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 11:15:00 | 139.17 | 140.09 | 140.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 15:15:00 | 139.85 | 139.76 | 140.01 | EMA200 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-17 11:45:00 | 76.90 | 2023-05-25 09:15:00 | 73.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-05-18 09:45:00 | 76.90 | 2023-05-25 09:15:00 | 73.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-05-17 11:45:00 | 76.90 | 2023-05-26 09:15:00 | 73.35 | STOP_HIT | 0.50 | 4.62% |
| SELL | retest2 | 2023-05-18 09:45:00 | 76.90 | 2023-05-26 09:15:00 | 73.35 | STOP_HIT | 0.50 | 4.62% |
| BUY | retest2 | 2023-06-05 09:15:00 | 75.25 | 2023-06-05 15:15:00 | 74.30 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2023-06-30 14:15:00 | 72.95 | 2023-07-04 11:15:00 | 80.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-19 12:30:00 | 79.80 | 2023-07-25 15:15:00 | 82.80 | STOP_HIT | 1.00 | 3.76% |
| BUY | retest2 | 2023-07-28 09:30:00 | 85.20 | 2023-07-28 13:15:00 | 83.80 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2023-07-28 10:00:00 | 85.00 | 2023-07-28 13:15:00 | 83.80 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2023-08-07 10:45:00 | 83.65 | 2023-08-08 09:15:00 | 85.20 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2023-08-07 13:45:00 | 83.60 | 2023-08-08 09:15:00 | 85.20 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2023-08-07 14:30:00 | 83.65 | 2023-08-08 09:15:00 | 85.20 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2023-08-11 09:15:00 | 86.55 | 2023-08-22 10:15:00 | 89.00 | STOP_HIT | 1.00 | 2.83% |
| BUY | retest2 | 2023-08-14 10:00:00 | 86.55 | 2023-08-22 10:15:00 | 89.00 | STOP_HIT | 1.00 | 2.83% |
| BUY | retest2 | 2023-08-14 10:45:00 | 87.90 | 2023-08-22 10:15:00 | 89.00 | STOP_HIT | 1.00 | 1.25% |
| SELL | retest2 | 2023-08-28 15:15:00 | 86.60 | 2023-08-30 10:15:00 | 88.05 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2023-08-29 10:15:00 | 86.60 | 2023-08-30 10:15:00 | 88.05 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2023-08-29 14:15:00 | 86.30 | 2023-08-30 10:15:00 | 88.05 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2023-09-08 14:15:00 | 93.40 | 2023-09-13 13:15:00 | 102.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-11 09:15:00 | 94.65 | 2023-09-13 15:15:00 | 104.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-05 09:15:00 | 111.60 | 2023-10-05 11:15:00 | 109.45 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2023-10-05 11:00:00 | 110.65 | 2023-10-05 11:15:00 | 109.45 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2023-10-17 13:15:00 | 105.80 | 2023-10-20 12:15:00 | 100.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-18 10:30:00 | 105.60 | 2023-10-20 12:15:00 | 100.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-17 13:15:00 | 105.80 | 2023-10-23 11:15:00 | 95.22 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-18 10:30:00 | 105.60 | 2023-10-23 11:15:00 | 95.04 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2023-11-01 10:45:00 | 97.70 | 2023-11-10 15:15:00 | 103.05 | STOP_HIT | 1.00 | 5.48% |
| BUY | retest2 | 2023-11-01 13:30:00 | 97.90 | 2023-11-10 15:15:00 | 103.05 | STOP_HIT | 1.00 | 5.26% |
| BUY | retest2 | 2023-11-16 14:00:00 | 107.30 | 2023-11-17 09:15:00 | 103.90 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2023-12-08 10:30:00 | 115.90 | 2023-12-13 09:15:00 | 109.85 | STOP_HIT | 1.00 | -5.22% |
| BUY | retest2 | 2023-12-08 12:00:00 | 115.35 | 2023-12-13 09:15:00 | 109.85 | STOP_HIT | 1.00 | -4.77% |
| BUY | retest2 | 2023-12-11 09:15:00 | 116.00 | 2023-12-13 09:15:00 | 109.85 | STOP_HIT | 1.00 | -5.30% |
| BUY | retest2 | 2023-12-12 15:00:00 | 115.90 | 2023-12-13 09:15:00 | 109.85 | STOP_HIT | 1.00 | -5.22% |
| SELL | retest2 | 2023-12-27 11:15:00 | 111.00 | 2023-12-27 11:15:00 | 111.35 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-01-01 09:15:00 | 113.05 | 2024-01-10 11:15:00 | 117.30 | STOP_HIT | 1.00 | 3.76% |
| BUY | retest2 | 2024-01-02 09:15:00 | 112.95 | 2024-01-10 11:15:00 | 117.30 | STOP_HIT | 1.00 | 3.85% |
| BUY | retest2 | 2024-01-02 11:15:00 | 113.20 | 2024-01-10 11:15:00 | 117.30 | STOP_HIT | 1.00 | 3.62% |
| BUY | retest2 | 2024-01-18 11:15:00 | 132.60 | 2024-01-23 13:15:00 | 132.00 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-01-18 13:00:00 | 134.00 | 2024-01-23 13:15:00 | 132.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-01-25 14:30:00 | 130.65 | 2024-01-29 10:15:00 | 136.10 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2024-01-25 15:15:00 | 131.25 | 2024-01-29 10:15:00 | 136.10 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2024-02-01 12:30:00 | 144.40 | 2024-02-05 14:15:00 | 138.60 | STOP_HIT | 1.00 | -4.02% |
| BUY | retest2 | 2024-02-20 10:00:00 | 144.15 | 2024-02-21 15:15:00 | 141.55 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-02-20 10:45:00 | 143.80 | 2024-02-21 15:15:00 | 141.55 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-02-20 12:15:00 | 143.35 | 2024-02-21 15:15:00 | 141.55 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-02-21 09:15:00 | 144.60 | 2024-02-21 15:15:00 | 141.55 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-02-23 10:15:00 | 137.80 | 2024-02-28 12:15:00 | 130.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-23 10:15:00 | 137.80 | 2024-02-29 15:15:00 | 133.00 | STOP_HIT | 0.50 | 3.48% |
| BUY | retest1 | 2024-03-11 09:15:00 | 147.35 | 2024-03-11 13:15:00 | 144.10 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest1 | 2024-03-11 12:30:00 | 146.20 | 2024-03-11 13:15:00 | 144.10 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-03-18 09:15:00 | 133.70 | 2024-03-18 15:15:00 | 133.00 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2024-03-20 09:15:00 | 134.60 | 2024-03-20 13:15:00 | 131.90 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2024-04-12 15:15:00 | 143.10 | 2024-04-19 09:15:00 | 135.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 15:15:00 | 143.10 | 2024-04-22 09:15:00 | 138.45 | STOP_HIT | 0.50 | 3.25% |
| SELL | retest2 | 2024-05-08 14:30:00 | 141.75 | 2024-05-13 09:15:00 | 127.58 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-09 10:30:00 | 141.30 | 2024-05-13 09:15:00 | 127.17 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-09 14:15:00 | 141.55 | 2024-05-13 09:15:00 | 127.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-06 12:15:00 | 119.00 | 2024-06-10 09:15:00 | 125.21 | STOP_HIT | 1.00 | -5.22% |
| BUY | retest2 | 2024-06-14 10:15:00 | 125.84 | 2024-06-18 09:15:00 | 123.05 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-06-14 11:30:00 | 126.08 | 2024-06-18 09:15:00 | 123.05 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2024-06-14 15:15:00 | 125.99 | 2024-06-18 09:15:00 | 123.05 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2024-06-19 15:00:00 | 123.05 | 2024-06-25 15:15:00 | 122.93 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2024-06-20 11:45:00 | 123.05 | 2024-06-25 15:15:00 | 122.93 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2024-06-20 13:45:00 | 123.00 | 2024-06-25 15:15:00 | 122.93 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2024-07-04 10:15:00 | 117.86 | 2024-07-05 09:15:00 | 119.69 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-07-23 11:30:00 | 120.87 | 2024-07-29 09:15:00 | 123.95 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2024-07-23 12:15:00 | 118.22 | 2024-07-29 09:15:00 | 123.95 | STOP_HIT | 1.00 | -4.85% |
| SELL | retest2 | 2024-07-24 13:45:00 | 120.95 | 2024-07-29 09:15:00 | 123.95 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2024-08-02 14:30:00 | 127.41 | 2024-08-05 09:15:00 | 120.20 | STOP_HIT | 1.00 | -5.66% |
| SELL | retest2 | 2024-08-08 15:15:00 | 117.68 | 2024-08-19 10:15:00 | 117.65 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2024-08-09 09:45:00 | 119.45 | 2024-08-19 10:15:00 | 117.65 | STOP_HIT | 1.00 | 1.51% |
| SELL | retest2 | 2024-08-09 14:15:00 | 119.39 | 2024-08-19 10:15:00 | 117.65 | STOP_HIT | 1.00 | 1.46% |
| SELL | retest2 | 2024-08-28 13:45:00 | 118.26 | 2024-09-03 10:15:00 | 118.30 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2024-09-06 12:00:00 | 115.49 | 2024-09-13 13:15:00 | 112.93 | STOP_HIT | 1.00 | 2.22% |
| SELL | retest2 | 2024-09-09 09:15:00 | 113.45 | 2024-09-13 13:15:00 | 112.93 | STOP_HIT | 1.00 | 0.46% |
| SELL | retest2 | 2024-10-10 11:15:00 | 105.20 | 2024-10-22 14:15:00 | 99.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-11 14:00:00 | 105.22 | 2024-10-22 14:15:00 | 99.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-14 10:45:00 | 105.13 | 2024-10-22 14:15:00 | 99.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 15:00:00 | 104.89 | 2024-10-22 15:15:00 | 99.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 11:30:00 | 103.47 | 2024-10-23 09:15:00 | 98.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 09:15:00 | 102.30 | 2024-10-23 09:15:00 | 97.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-10 11:15:00 | 105.20 | 2024-10-28 09:15:00 | 98.09 | STOP_HIT | 0.50 | 6.76% |
| SELL | retest2 | 2024-10-11 14:00:00 | 105.22 | 2024-10-28 09:15:00 | 98.09 | STOP_HIT | 0.50 | 6.78% |
| SELL | retest2 | 2024-10-14 10:45:00 | 105.13 | 2024-10-28 09:15:00 | 98.09 | STOP_HIT | 0.50 | 6.70% |
| SELL | retest2 | 2024-10-18 15:00:00 | 104.89 | 2024-10-28 09:15:00 | 98.09 | STOP_HIT | 0.50 | 6.48% |
| SELL | retest2 | 2024-10-21 11:30:00 | 103.47 | 2024-10-28 09:15:00 | 98.09 | STOP_HIT | 0.50 | 5.20% |
| SELL | retest2 | 2024-10-22 09:15:00 | 102.30 | 2024-10-28 09:15:00 | 98.09 | STOP_HIT | 0.50 | 4.12% |
| BUY | retest2 | 2024-11-05 13:45:00 | 110.40 | 2024-11-08 15:15:00 | 111.25 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest2 | 2024-11-06 09:15:00 | 111.58 | 2024-11-08 15:15:00 | 111.25 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2024-12-27 14:45:00 | 102.72 | 2025-01-01 15:15:00 | 102.80 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2024-12-30 09:15:00 | 102.43 | 2025-01-01 15:15:00 | 102.80 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2024-12-30 12:15:00 | 102.39 | 2025-01-01 15:15:00 | 102.80 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-01-24 09:45:00 | 98.98 | 2025-01-27 09:15:00 | 102.65 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-01-24 13:30:00 | 98.98 | 2025-01-27 09:15:00 | 102.65 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-02-04 11:45:00 | 107.09 | 2025-02-05 11:15:00 | 108.16 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-02-13 11:00:00 | 102.21 | 2025-02-17 09:15:00 | 97.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 11:00:00 | 102.21 | 2025-02-17 14:15:00 | 99.05 | STOP_HIT | 0.50 | 3.09% |
| BUY | retest2 | 2025-02-21 15:00:00 | 101.68 | 2025-02-24 09:15:00 | 100.68 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-03-12 10:15:00 | 94.45 | 2025-03-17 13:15:00 | 95.22 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-03-12 10:45:00 | 94.25 | 2025-03-17 13:15:00 | 95.22 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-03-13 11:15:00 | 94.29 | 2025-03-17 13:15:00 | 95.22 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-03-13 13:00:00 | 94.46 | 2025-03-17 13:15:00 | 95.22 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-03-25 14:45:00 | 106.07 | 2025-04-03 13:15:00 | 116.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-26 09:15:00 | 106.44 | 2025-04-03 13:15:00 | 117.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-27 09:30:00 | 106.93 | 2025-04-03 13:15:00 | 117.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-27 14:15:00 | 106.03 | 2025-04-03 13:15:00 | 116.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-01 12:00:00 | 110.59 | 2025-04-07 09:15:00 | 105.01 | STOP_HIT | 1.00 | -5.05% |
| BUY | retest2 | 2025-04-02 10:45:00 | 110.36 | 2025-04-07 09:15:00 | 105.01 | STOP_HIT | 1.00 | -4.85% |
| SELL | retest2 | 2025-04-25 15:15:00 | 115.24 | 2025-04-28 09:15:00 | 117.82 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-05-02 12:00:00 | 115.59 | 2025-05-06 14:15:00 | 109.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 14:15:00 | 115.36 | 2025-05-06 14:15:00 | 109.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-05 10:00:00 | 115.65 | 2025-05-06 14:15:00 | 109.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-05 10:30:00 | 115.65 | 2025-05-06 14:15:00 | 109.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 11:00:00 | 113.11 | 2025-05-07 09:15:00 | 107.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 12:00:00 | 115.59 | 2025-05-08 10:15:00 | 111.31 | STOP_HIT | 0.50 | 3.70% |
| SELL | retest2 | 2025-05-02 14:15:00 | 115.36 | 2025-05-08 10:15:00 | 111.31 | STOP_HIT | 0.50 | 3.51% |
| SELL | retest2 | 2025-05-05 10:00:00 | 115.65 | 2025-05-08 10:15:00 | 111.31 | STOP_HIT | 0.50 | 3.75% |
| SELL | retest2 | 2025-05-05 10:30:00 | 115.65 | 2025-05-08 10:15:00 | 111.31 | STOP_HIT | 0.50 | 3.75% |
| SELL | retest2 | 2025-05-06 11:00:00 | 113.11 | 2025-05-08 10:15:00 | 111.31 | STOP_HIT | 0.50 | 1.59% |
| BUY | retest1 | 2025-05-14 09:15:00 | 114.40 | 2025-05-20 13:15:00 | 116.20 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2025-05-21 10:15:00 | 117.96 | 2025-05-22 13:15:00 | 115.68 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-05-21 13:00:00 | 118.00 | 2025-05-22 13:15:00 | 115.68 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-05-21 13:45:00 | 117.91 | 2025-05-22 13:15:00 | 115.68 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-05-30 13:00:00 | 120.55 | 2025-06-11 10:15:00 | 125.36 | STOP_HIT | 1.00 | 3.99% |
| SELL | retest2 | 2025-06-17 11:45:00 | 120.70 | 2025-06-19 12:15:00 | 114.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:45:00 | 120.70 | 2025-06-23 13:15:00 | 114.15 | STOP_HIT | 0.50 | 5.43% |
| BUY | retest2 | 2025-06-30 09:15:00 | 119.19 | 2025-07-03 12:15:00 | 118.06 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-07-07 15:15:00 | 117.49 | 2025-07-15 10:15:00 | 115.86 | STOP_HIT | 1.00 | 1.39% |
| SELL | retest2 | 2025-07-08 09:30:00 | 117.65 | 2025-07-15 10:15:00 | 115.86 | STOP_HIT | 1.00 | 1.52% |
| BUY | retest2 | 2025-07-18 09:15:00 | 116.40 | 2025-07-18 09:15:00 | 115.55 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-07-24 11:15:00 | 112.70 | 2025-07-24 13:15:00 | 114.18 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-08-05 13:15:00 | 110.74 | 2025-08-06 12:15:00 | 112.09 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-08-06 10:00:00 | 110.37 | 2025-08-06 12:15:00 | 112.09 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-08-18 09:15:00 | 114.35 | 2025-08-22 15:15:00 | 115.40 | STOP_HIT | 1.00 | 0.92% |
| BUY | retest2 | 2025-09-17 10:15:00 | 119.06 | 2025-09-25 11:15:00 | 120.11 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest2 | 2025-09-17 10:45:00 | 119.09 | 2025-09-25 11:15:00 | 120.11 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2025-09-17 13:30:00 | 119.18 | 2025-09-25 11:15:00 | 120.11 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2025-09-18 12:45:00 | 118.95 | 2025-09-25 11:15:00 | 120.11 | STOP_HIT | 1.00 | 0.98% |
| BUY | retest2 | 2025-09-18 14:45:00 | 118.93 | 2025-09-25 11:15:00 | 120.11 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2025-09-29 11:30:00 | 118.50 | 2025-09-30 09:15:00 | 122.01 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-09-29 15:00:00 | 118.71 | 2025-09-30 09:15:00 | 122.01 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-10-13 14:15:00 | 127.19 | 2025-10-14 09:15:00 | 125.63 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-10-17 12:30:00 | 124.55 | 2025-10-20 10:15:00 | 129.49 | STOP_HIT | 1.00 | -3.97% |
| BUY | retest2 | 2025-11-03 09:15:00 | 142.08 | 2025-11-06 11:15:00 | 140.12 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-11-11 13:00:00 | 145.44 | 2025-11-21 10:15:00 | 145.47 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-11-14 09:15:00 | 145.35 | 2025-11-21 10:15:00 | 145.47 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-12-08 09:15:00 | 141.60 | 2025-12-10 10:15:00 | 141.05 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2025-12-11 11:30:00 | 140.36 | 2025-12-11 12:15:00 | 141.20 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-12-19 09:15:00 | 143.27 | 2025-12-23 12:15:00 | 141.93 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-12-19 10:30:00 | 142.51 | 2025-12-23 12:15:00 | 141.93 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-12-19 12:30:00 | 142.51 | 2025-12-23 13:15:00 | 141.84 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-12-19 13:15:00 | 142.62 | 2025-12-23 13:15:00 | 141.84 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-12-23 09:15:00 | 143.03 | 2025-12-23 13:15:00 | 141.84 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-12-23 09:45:00 | 142.85 | 2025-12-23 13:15:00 | 141.84 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-12-30 09:15:00 | 138.40 | 2025-12-30 11:15:00 | 140.53 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-01-07 11:15:00 | 151.21 | 2026-01-08 10:15:00 | 147.33 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2026-01-07 14:00:00 | 151.05 | 2026-01-08 10:15:00 | 147.33 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2026-01-30 09:45:00 | 165.76 | 2026-02-01 09:15:00 | 159.78 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2026-01-30 10:30:00 | 165.84 | 2026-02-01 09:15:00 | 159.78 | STOP_HIT | 1.00 | -3.65% |
| BUY | retest2 | 2026-02-11 13:00:00 | 166.35 | 2026-02-12 11:15:00 | 163.74 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2026-02-20 11:15:00 | 171.80 | 2026-03-02 09:15:00 | 172.84 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2026-02-20 13:15:00 | 171.50 | 2026-03-02 09:15:00 | 172.84 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2026-02-20 14:45:00 | 171.59 | 2026-03-02 09:15:00 | 172.84 | STOP_HIT | 1.00 | 0.73% |
| BUY | retest2 | 2026-02-23 09:15:00 | 174.23 | 2026-03-02 09:15:00 | 172.84 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-02-25 12:15:00 | 175.27 | 2026-03-02 10:15:00 | 173.11 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2026-02-25 15:00:00 | 175.50 | 2026-03-02 10:15:00 | 173.11 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-02-26 09:15:00 | 176.71 | 2026-03-02 10:15:00 | 173.11 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-02-26 10:30:00 | 175.36 | 2026-03-02 10:15:00 | 173.11 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest1 | 2026-03-05 11:30:00 | 164.46 | 2026-03-09 09:15:00 | 156.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-05 15:00:00 | 163.43 | 2026-03-09 09:15:00 | 155.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-05 11:30:00 | 164.46 | 2026-03-09 10:15:00 | 148.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2026-03-05 15:00:00 | 163.43 | 2026-03-10 10:15:00 | 153.71 | STOP_HIT | 0.50 | 5.95% |
| SELL | retest2 | 2026-03-11 10:15:00 | 155.14 | 2026-03-16 10:15:00 | 147.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 11:30:00 | 155.00 | 2026-03-16 10:15:00 | 147.58 | PARTIAL | 0.50 | 4.79% |
| SELL | retest2 | 2026-03-11 13:30:00 | 155.35 | 2026-03-16 10:15:00 | 147.37 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2026-03-11 14:30:00 | 155.13 | 2026-03-16 12:15:00 | 147.25 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2026-03-11 10:15:00 | 155.14 | 2026-03-16 14:15:00 | 150.73 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest2 | 2026-03-11 11:30:00 | 155.00 | 2026-03-16 14:15:00 | 150.73 | STOP_HIT | 0.50 | 2.75% |
| SELL | retest2 | 2026-03-11 13:30:00 | 155.35 | 2026-03-16 14:15:00 | 150.73 | STOP_HIT | 0.50 | 2.97% |
| SELL | retest2 | 2026-03-11 14:30:00 | 155.13 | 2026-03-16 14:15:00 | 150.73 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest2 | 2026-03-13 09:15:00 | 153.54 | 2026-03-18 11:15:00 | 152.89 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2026-03-20 12:15:00 | 149.85 | 2026-03-20 13:15:00 | 150.99 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-03-24 10:30:00 | 145.11 | 2026-03-24 12:15:00 | 147.68 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2026-04-01 10:15:00 | 140.31 | 2026-04-01 12:15:00 | 143.40 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2026-04-01 11:45:00 | 140.34 | 2026-04-01 12:15:00 | 143.40 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2026-04-02 09:15:00 | 136.65 | 2026-04-06 13:15:00 | 142.75 | STOP_HIT | 1.00 | -4.46% |
| BUY | retest2 | 2026-04-13 12:15:00 | 144.62 | 2026-04-13 13:15:00 | 144.50 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2026-04-20 10:30:00 | 149.89 | 2026-04-24 09:15:00 | 148.38 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-04-21 09:15:00 | 149.77 | 2026-04-24 09:15:00 | 148.38 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-04-23 13:00:00 | 150.72 | 2026-04-24 09:15:00 | 148.38 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-04-29 14:15:00 | 144.41 | 2026-05-05 11:15:00 | 137.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-29 14:15:00 | 144.41 | 2026-05-06 09:15:00 | 141.06 | STOP_HIT | 0.50 | 2.32% |
