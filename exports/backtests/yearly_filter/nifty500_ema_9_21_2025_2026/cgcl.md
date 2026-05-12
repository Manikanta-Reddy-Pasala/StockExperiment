# Capri Global Capital Ltd. (CGCL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 197.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 89 |
| ALERT1 | 55 |
| ALERT2 | 55 |
| ALERT2_SKIP | 25 |
| ALERT3 | 140 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 74 |
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 75 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 81 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 58
- **Target hits / Stop hits / Partials:** 1 / 74 / 6
- **Avg / median % per leg:** -0.11% / -0.89%
- **Sum % (uncompounded):** -9.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 6 | 15.0% | 0 | 39 | 1 | -0.62% | -24.7% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.31% | 6.6% |
| BUY @ 3rd Alert (retest2) | 38 | 4 | 10.5% | 0 | 38 | 0 | -0.83% | -31.4% |
| SELL (all) | 41 | 17 | 41.5% | 1 | 35 | 5 | 0.38% | 15.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 41 | 17 | 41.5% | 1 | 35 | 5 | 0.38% | 15.7% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.31% | 6.6% |
| retest2 (combined) | 79 | 21 | 26.6% | 1 | 73 | 5 | -0.20% | -15.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

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

### Cycle 2 — SELL (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 13:15:00 | 167.07 | 167.69 | 167.69 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 168.63 | 167.83 | 167.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 10:15:00 | 169.74 | 168.21 | 167.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 12:15:00 | 168.25 | 168.28 | 168.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 12:45:00 | 168.32 | 168.28 | 168.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 166.72 | 167.97 | 167.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 166.72 | 167.97 | 167.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-19 14:15:00)

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

### Cycle 5 — BUY (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 12:15:00 | 156.13 | 154.47 | 154.42 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 14:15:00 | 150.85 | 153.91 | 154.18 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 157.31 | 154.66 | 154.41 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 15:15:00 | 153.00 | 154.86 | 154.93 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-09 09:15:00)

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

### Cycle 10 — SELL (started 2025-06-13 10:15:00)

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

### Cycle 11 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 177.35 | 172.90 | 172.87 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-18 10:15:00)

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

### Cycle 13 — BUY (started 2025-06-24 09:15:00)

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

### Cycle 14 — SELL (started 2025-06-26 12:15:00)

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

### Cycle 15 — BUY (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 13:15:00 | 173.71 | 171.29 | 171.26 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-01 09:15:00)

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

### Cycle 17 — BUY (started 2025-07-08 11:15:00)

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

### Cycle 18 — SELL (started 2025-07-25 12:15:00)

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

### Cycle 19 — BUY (started 2025-07-30 11:15:00)

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

### Cycle 20 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 175.20 | 182.75 | 183.77 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 12:15:00 | 189.04 | 183.65 | 183.52 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 184.19 | 186.25 | 186.51 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 187.79 | 186.54 | 186.50 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-08 10:15:00)

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

### Cycle 25 — BUY (started 2025-08-13 11:15:00)

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

### Cycle 26 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 187.71 | 188.99 | 189.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 15:15:00 | 187.60 | 188.71 | 188.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 11:15:00 | 189.32 | 188.78 | 188.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 11:15:00 | 189.32 | 188.78 | 188.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 189.32 | 188.78 | 188.86 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 13:15:00 | 189.29 | 188.93 | 188.92 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 187.70 | 188.68 | 188.81 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-08-25 09:15:00)

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

### Cycle 30 — SELL (started 2025-08-28 10:15:00)

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

### Cycle 31 — BUY (started 2025-09-01 14:15:00)

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

### Cycle 32 — SELL (started 2025-09-04 10:15:00)

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

### Cycle 33 — BUY (started 2025-09-09 13:15:00)

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

### Cycle 34 — SELL (started 2025-09-11 12:15:00)

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

### Cycle 35 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 188.75 | 186.27 | 186.17 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-09-18 11:15:00)

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

### Cycle 37 — BUY (started 2025-09-19 14:15:00)

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

### Cycle 38 — SELL (started 2025-09-25 11:15:00)

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

### Cycle 39 — BUY (started 2025-10-01 14:15:00)

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

### Cycle 40 — SELL (started 2025-10-08 09:15:00)

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

### Cycle 41 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 189.38 | 188.36 | 188.30 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 12:15:00 | 187.40 | 188.66 | 188.70 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 189.14 | 188.76 | 188.74 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-10-13 10:15:00)

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

### Cycle 45 — BUY (started 2025-10-15 11:15:00)

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

### Cycle 46 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 202.51 | 204.77 | 205.04 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 15:15:00 | 206.50 | 205.25 | 205.22 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-10-31 15:15:00)

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

### Cycle 49 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 201.00 | 198.01 | 197.85 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 197.35 | 197.91 | 197.92 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 199.92 | 198.10 | 197.89 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-11-13 15:15:00)

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

### Cycle 53 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 192.59 | 189.81 | 189.47 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-12-02 11:15:00)

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

### Cycle 55 — BUY (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 14:15:00 | 184.00 | 183.19 | 183.16 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-12-15 11:15:00)

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

### Cycle 57 — BUY (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 11:15:00 | 179.22 | 178.54 | 178.49 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-12-29 10:15:00)

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

### Cycle 59 — BUY (started 2025-12-30 14:15:00)

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

### Cycle 60 — SELL (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 15:15:00 | 184.90 | 186.72 | 186.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 181.37 | 183.82 | 184.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 10:15:00 | 181.48 | 181.01 | 181.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 11:00:00 | 181.48 | 181.01 | 181.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 181.03 | 181.02 | 181.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:45:00 | 180.70 | 180.97 | 181.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 13:15:00 | 182.40 | 181.26 | 181.80 | SL hit (close>static) qty=1.00 sl=181.87 alert=retest2 |

### Cycle 61 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 174.79 | 169.50 | 169.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 176.40 | 170.88 | 169.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 172.05 | 176.60 | 174.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 172.05 | 176.60 | 174.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 172.05 | 176.60 | 174.89 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-02-01 10:15:00)

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

### Cycle 63 — BUY (started 2026-02-03 13:15:00)

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

### Cycle 64 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 169.25 | 170.49 | 170.59 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-02-06 12:15:00)

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

### Cycle 66 — SELL (started 2026-02-16 11:15:00)

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

### Cycle 67 — BUY (started 2026-02-18 13:15:00)

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

### Cycle 68 — SELL (started 2026-02-19 13:15:00)

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

### Cycle 69 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 160.69 | 157.82 | 157.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 13:15:00 | 161.41 | 159.03 | 158.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 155.20 | 158.83 | 158.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 155.20 | 158.83 | 158.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 155.20 | 158.83 | 158.34 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 153.19 | 157.70 | 157.87 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-03-10 09:15:00)

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

### Cycle 72 — SELL (started 2026-03-19 12:15:00)

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

### Cycle 73 — BUY (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 15:15:00 | 169.90 | 169.20 | 169.18 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 162.05 | 167.77 | 168.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 161.95 | 166.60 | 167.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 161.37 | 160.99 | 163.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 161.90 | 160.99 | 163.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 171.50 | 163.05 | 163.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 171.50 | 163.05 | 163.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 171.18 | 164.68 | 164.60 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 164.00 | 165.85 | 166.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 161.95 | 164.47 | 165.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 170.04 | 164.43 | 164.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 170.04 | 164.43 | 164.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 170.04 | 164.43 | 164.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 168.43 | 164.43 | 164.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 172.80 | 166.10 | 165.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 174.90 | 167.86 | 166.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 164.48 | 170.58 | 168.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 164.48 | 170.58 | 168.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 164.48 | 170.58 | 168.69 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 12:15:00 | 164.80 | 167.62 | 167.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 10:15:00 | 163.69 | 166.00 | 166.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 165.50 | 165.40 | 166.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 166.00 | 165.56 | 166.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 166.00 | 165.56 | 166.15 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2026-04-07 13:15:00)

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

### Cycle 80 — SELL (started 2026-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 13:15:00 | 179.77 | 182.39 | 182.50 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-04-16 14:15:00)

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

### Cycle 82 — SELL (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 14:15:00 | 181.95 | 182.82 | 182.90 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 09:15:00 | 185.50 | 183.23 | 183.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 12:15:00 | 185.90 | 184.36 | 183.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 09:15:00 | 181.92 | 184.14 | 183.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 181.92 | 184.14 | 183.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 181.92 | 184.14 | 183.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:00:00 | 181.92 | 184.14 | 183.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 10:15:00 | 181.00 | 183.51 | 183.57 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2026-04-22 13:15:00)

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

### Cycle 86 — SELL (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 15:15:00 | 181.99 | 184.17 | 184.38 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2026-04-27 10:15:00)

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

### Cycle 88 — SELL (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 10:15:00 | 183.95 | 185.00 | 185.06 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2026-04-30 13:15:00)

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
