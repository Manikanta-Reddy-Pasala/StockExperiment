# GMR Airports Ltd. (GMRAIRPORT)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 101.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 233 |
| ALERT1 | 150 |
| ALERT2 | 149 |
| ALERT2_SKIP | 73 |
| ALERT3 | 382 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 144 |
| PARTIAL | 22 |
| TARGET_HIT | 13 |
| STOP_HIT | 137 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 172 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 99 / 73
- **Target hits / Stop hits / Partials:** 13 / 137 / 22
- **Avg / median % per leg:** 1.71% / 0.64%
- **Sum % (uncompounded):** 293.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 80 | 50 | 62.5% | 7 | 71 | 2 | 1.72% | 137.8% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 0 | 5 | 2 | 2.13% | 14.9% |
| BUY @ 3rd Alert (retest2) | 73 | 46 | 63.0% | 7 | 66 | 0 | 1.68% | 122.9% |
| SELL (all) | 92 | 49 | 53.3% | 6 | 66 | 20 | 1.70% | 156.0% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.28% | -4.3% |
| SELL @ 3rd Alert (retest2) | 91 | 49 | 53.8% | 6 | 65 | 20 | 1.76% | 160.3% |
| retest1 (combined) | 8 | 4 | 50.0% | 0 | 6 | 2 | 1.33% | 10.6% |
| retest2 (combined) | 164 | 95 | 57.9% | 13 | 131 | 20 | 1.73% | 283.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 11:15:00 | 45.50 | 45.27 | 45.27 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 10:15:00 | 45.15 | 45.28 | 45.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 11:15:00 | 45.00 | 45.23 | 45.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 09:15:00 | 45.05 | 45.01 | 45.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 09:15:00 | 45.05 | 45.01 | 45.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 45.05 | 45.01 | 45.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 09:45:00 | 45.10 | 45.01 | 45.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 10:15:00 | 45.00 | 45.01 | 45.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 11:00:00 | 45.00 | 45.01 | 45.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 15:15:00 | 45.10 | 44.93 | 45.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 09:15:00 | 45.55 | 44.93 | 45.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 45.10 | 44.96 | 45.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-26 10:15:00 | 44.85 | 44.96 | 45.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-29 09:15:00 | 42.45 | 44.89 | 44.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-29 09:15:00 | 42.61 | 44.13 | 44.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2023-05-29 11:15:00 | 40.37 | 42.80 | 43.88 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2023-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 14:15:00 | 41.35 | 41.14 | 41.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 09:15:00 | 41.70 | 41.27 | 41.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 15:15:00 | 41.45 | 41.57 | 41.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 15:15:00 | 41.45 | 41.57 | 41.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 15:15:00 | 41.45 | 41.57 | 41.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 09:15:00 | 41.90 | 41.57 | 41.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 15:15:00 | 41.75 | 41.54 | 41.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-13 13:15:00 | 42.10 | 42.20 | 42.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-06-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 13:15:00 | 42.10 | 42.20 | 42.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-13 15:15:00 | 42.00 | 42.15 | 42.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-14 09:15:00 | 42.40 | 42.20 | 42.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-14 09:15:00 | 42.40 | 42.20 | 42.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 09:15:00 | 42.40 | 42.20 | 42.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-14 10:00:00 | 42.40 | 42.20 | 42.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2023-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 10:15:00 | 42.50 | 42.26 | 42.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 11:15:00 | 42.70 | 42.35 | 42.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 15:15:00 | 43.20 | 43.20 | 42.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-16 09:15:00 | 43.25 | 43.20 | 42.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 15:15:00 | 43.05 | 43.17 | 43.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 09:15:00 | 43.20 | 43.17 | 43.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 10:00:00 | 43.25 | 43.19 | 43.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-19 10:15:00 | 42.95 | 43.14 | 43.05 | SL hit (close<static) qty=1.00 sl=43.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 13:15:00 | 42.70 | 42.98 | 42.99 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 14:15:00 | 43.20 | 42.96 | 42.94 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 11:15:00 | 42.85 | 42.93 | 42.94 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 12:15:00 | 43.05 | 42.96 | 42.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 13:15:00 | 43.25 | 43.02 | 42.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 09:15:00 | 42.85 | 43.09 | 43.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 09:15:00 | 42.85 | 43.09 | 43.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 42.85 | 43.09 | 43.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 10:00:00 | 42.85 | 43.09 | 43.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 42.65 | 43.00 | 42.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 10:30:00 | 42.70 | 43.00 | 42.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2023-06-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 11:15:00 | 42.30 | 42.86 | 42.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 12:15:00 | 42.15 | 42.72 | 42.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 11:15:00 | 42.50 | 42.44 | 42.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-23 12:00:00 | 42.50 | 42.44 | 42.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 42.65 | 42.40 | 42.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 11:00:00 | 42.65 | 42.40 | 42.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 11:15:00 | 42.70 | 42.46 | 42.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 11:30:00 | 42.70 | 42.46 | 42.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 42.40 | 42.47 | 42.53 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 12:15:00 | 42.85 | 42.58 | 42.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 11:15:00 | 43.10 | 42.87 | 42.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 09:15:00 | 43.75 | 43.78 | 43.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 09:15:00 | 43.75 | 43.78 | 43.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 43.75 | 43.78 | 43.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 09:30:00 | 43.25 | 43.78 | 43.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 13:15:00 | 43.55 | 43.82 | 43.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 14:00:00 | 43.55 | 43.82 | 43.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 14:15:00 | 43.55 | 43.77 | 43.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 14:45:00 | 43.50 | 43.77 | 43.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 15:15:00 | 43.60 | 43.73 | 43.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 09:15:00 | 43.75 | 43.73 | 43.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 43.80 | 43.75 | 43.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 10:15:00 | 43.95 | 43.75 | 43.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-06 09:15:00 | 44.20 | 43.79 | 43.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-06 10:15:00 | 43.95 | 43.81 | 43.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-06 10:45:00 | 44.20 | 43.90 | 43.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 44.65 | 44.84 | 44.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 11:15:00 | 44.55 | 44.84 | 44.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 11:15:00 | 44.30 | 44.73 | 44.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 12:00:00 | 44.30 | 44.73 | 44.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 12:15:00 | 44.25 | 44.64 | 44.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 13:00:00 | 44.25 | 44.64 | 44.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 14:15:00 | 44.45 | 44.57 | 44.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 14:45:00 | 44.45 | 44.57 | 44.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 44.60 | 44.68 | 44.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-11 15:00:00 | 44.60 | 44.68 | 44.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 15:15:00 | 44.65 | 44.68 | 44.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-12 09:15:00 | 44.75 | 44.68 | 44.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-13 13:15:00 | 44.10 | 44.88 | 44.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-07-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 13:15:00 | 44.10 | 44.88 | 44.89 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 10:15:00 | 45.05 | 44.77 | 44.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 11:15:00 | 45.20 | 44.86 | 44.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 09:15:00 | 44.90 | 44.94 | 44.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 09:15:00 | 44.90 | 44.94 | 44.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 44.90 | 44.94 | 44.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 09:30:00 | 44.85 | 44.94 | 44.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 10:15:00 | 44.85 | 44.92 | 44.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 11:00:00 | 44.85 | 44.92 | 44.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 11:15:00 | 44.50 | 44.84 | 44.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 12:00:00 | 44.50 | 44.84 | 44.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2023-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 12:15:00 | 44.60 | 44.79 | 44.80 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-07-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 10:15:00 | 44.95 | 44.77 | 44.77 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 13:15:00 | 44.45 | 44.75 | 44.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 10:15:00 | 44.25 | 44.51 | 44.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 09:15:00 | 45.50 | 44.58 | 44.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 09:15:00 | 45.50 | 44.58 | 44.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 45.50 | 44.58 | 44.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 09:30:00 | 45.90 | 44.58 | 44.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2023-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 10:15:00 | 46.10 | 44.88 | 44.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-24 13:15:00 | 46.45 | 45.55 | 45.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 11:15:00 | 51.20 | 51.32 | 50.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-01 11:45:00 | 51.25 | 51.32 | 50.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 51.50 | 51.37 | 50.93 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2023-08-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 12:15:00 | 50.60 | 50.86 | 50.87 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 10:15:00 | 51.05 | 50.88 | 50.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 11:15:00 | 51.40 | 50.98 | 50.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 09:15:00 | 51.20 | 51.23 | 51.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 09:15:00 | 51.20 | 51.23 | 51.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 51.20 | 51.23 | 51.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 09:30:00 | 51.05 | 51.23 | 51.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 10:15:00 | 51.10 | 51.20 | 51.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 11:00:00 | 51.10 | 51.20 | 51.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 11:15:00 | 51.55 | 51.27 | 51.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-08 09:15:00 | 52.25 | 51.40 | 51.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-08 11:45:00 | 51.75 | 51.69 | 51.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-14 15:15:00 | 53.00 | 53.07 | 53.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2023-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 15:15:00 | 53.00 | 53.07 | 53.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-16 10:15:00 | 51.95 | 52.84 | 52.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 13:15:00 | 52.60 | 52.56 | 52.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-16 14:00:00 | 52.60 | 52.56 | 52.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 14:15:00 | 52.75 | 52.60 | 52.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 15:00:00 | 52.75 | 52.60 | 52.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 15:15:00 | 52.85 | 52.65 | 52.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 09:15:00 | 53.25 | 52.65 | 52.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 53.05 | 52.73 | 52.81 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-08-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 11:15:00 | 53.20 | 52.89 | 52.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 13:15:00 | 53.45 | 53.07 | 52.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-18 09:15:00 | 52.95 | 53.21 | 53.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 09:15:00 | 52.95 | 53.21 | 53.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 09:15:00 | 52.95 | 53.21 | 53.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 10:00:00 | 52.95 | 53.21 | 53.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 10:15:00 | 52.90 | 53.15 | 53.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 10:45:00 | 52.85 | 53.15 | 53.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 11:15:00 | 53.20 | 53.16 | 53.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 09:15:00 | 54.05 | 53.16 | 53.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-08-23 09:15:00 | 59.45 | 57.35 | 55.94 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2023-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-28 14:15:00 | 60.80 | 61.63 | 61.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-28 15:15:00 | 60.45 | 61.39 | 61.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 09:15:00 | 62.00 | 61.51 | 61.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 09:15:00 | 62.00 | 61.51 | 61.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 62.00 | 61.51 | 61.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 09:45:00 | 62.45 | 61.51 | 61.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 10:15:00 | 61.90 | 61.59 | 61.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 10:30:00 | 61.90 | 61.59 | 61.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2023-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 11:15:00 | 61.75 | 61.62 | 61.62 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 12:15:00 | 61.55 | 61.61 | 61.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-29 13:15:00 | 60.75 | 61.44 | 61.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-31 09:15:00 | 61.80 | 60.76 | 60.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 09:15:00 | 61.80 | 60.76 | 60.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 61.80 | 60.76 | 60.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-31 10:00:00 | 61.80 | 60.76 | 60.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 62.15 | 61.04 | 61.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-31 11:15:00 | 62.45 | 61.04 | 61.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2023-08-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 11:15:00 | 61.90 | 61.21 | 61.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 14:15:00 | 62.95 | 62.15 | 61.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 12:15:00 | 62.85 | 63.22 | 62.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 12:15:00 | 62.85 | 63.22 | 62.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 12:15:00 | 62.85 | 63.22 | 62.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 13:00:00 | 62.85 | 63.22 | 62.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 13:15:00 | 63.25 | 63.23 | 62.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-05 14:15:00 | 63.35 | 63.23 | 62.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 10:15:00 | 63.35 | 63.45 | 63.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 11:00:00 | 63.35 | 63.43 | 63.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-06 13:15:00 | 62.50 | 63.05 | 62.99 | SL hit (close<static) qty=1.00 sl=62.65 alert=retest2 |

### Cycle 26 — SELL (started 2023-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 10:15:00 | 61.70 | 63.19 | 63.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 13:15:00 | 60.30 | 61.91 | 62.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 12:15:00 | 60.75 | 60.49 | 61.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 13:00:00 | 60.75 | 60.49 | 61.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 60.65 | 60.58 | 61.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:30:00 | 61.15 | 60.58 | 61.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 10:15:00 | 61.10 | 60.68 | 61.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 10:30:00 | 61.00 | 60.68 | 61.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 11:15:00 | 61.15 | 60.78 | 61.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 12:15:00 | 60.85 | 60.78 | 61.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 13:30:00 | 60.80 | 60.76 | 61.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 11:45:00 | 60.85 | 60.90 | 61.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-18 09:45:00 | 60.70 | 60.74 | 60.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 59.95 | 60.17 | 60.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-20 09:30:00 | 60.25 | 60.17 | 60.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 09:15:00 | 60.00 | 59.89 | 60.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 09:30:00 | 60.30 | 59.89 | 60.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 10:15:00 | 60.10 | 59.93 | 60.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-21 12:00:00 | 59.60 | 59.86 | 60.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-22 09:15:00 | 57.81 | 58.94 | 59.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-22 09:15:00 | 57.76 | 58.94 | 59.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-22 09:15:00 | 57.81 | 58.94 | 59.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-22 09:15:00 | 57.66 | 58.94 | 59.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-09-22 12:15:00 | 58.80 | 58.73 | 59.27 | SL hit (close>ema200) qty=0.50 sl=58.73 alert=retest2 |

### Cycle 27 — BUY (started 2023-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 10:15:00 | 59.35 | 58.79 | 58.74 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-09-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 13:15:00 | 58.05 | 58.69 | 58.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 14:15:00 | 57.95 | 58.55 | 58.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 09:15:00 | 59.30 | 58.64 | 58.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 09:15:00 | 59.30 | 58.64 | 58.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 59.30 | 58.64 | 58.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 09:45:00 | 59.40 | 58.64 | 58.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2023-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 10:15:00 | 59.35 | 58.78 | 58.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 12:15:00 | 59.65 | 59.06 | 58.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 10:15:00 | 59.20 | 59.32 | 59.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 10:15:00 | 59.20 | 59.32 | 59.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 10:15:00 | 59.20 | 59.32 | 59.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 11:00:00 | 59.20 | 59.32 | 59.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 11:15:00 | 59.30 | 59.31 | 59.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 11:45:00 | 59.15 | 59.31 | 59.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 13:15:00 | 59.25 | 59.31 | 59.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 14:00:00 | 59.25 | 59.31 | 59.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 14:15:00 | 59.55 | 59.35 | 59.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-04 11:00:00 | 59.65 | 59.32 | 59.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-04 11:15:00 | 58.55 | 59.16 | 59.15 | SL hit (close<static) qty=1.00 sl=59.15 alert=retest2 |

### Cycle 30 — SELL (started 2023-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 12:15:00 | 58.20 | 58.97 | 59.07 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-10-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 10:15:00 | 59.55 | 59.08 | 59.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 09:15:00 | 59.70 | 59.40 | 59.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 59.35 | 60.26 | 59.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 59.35 | 60.26 | 59.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 59.35 | 60.26 | 59.84 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 12:15:00 | 57.95 | 59.31 | 59.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 14:15:00 | 57.75 | 58.75 | 59.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 58.85 | 58.60 | 59.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 58.85 | 58.60 | 59.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 58.85 | 58.60 | 59.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 09:30:00 | 58.90 | 58.60 | 59.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 59.15 | 58.71 | 59.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 11:00:00 | 59.15 | 58.71 | 59.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 11:15:00 | 59.05 | 58.78 | 59.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 12:00:00 | 59.05 | 58.78 | 59.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 12:15:00 | 59.15 | 58.85 | 59.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 13:00:00 | 59.15 | 58.85 | 59.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 13:15:00 | 59.05 | 58.89 | 59.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 13:30:00 | 59.15 | 58.89 | 59.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 14:15:00 | 59.10 | 58.93 | 59.05 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 10:15:00 | 59.95 | 59.24 | 59.16 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-10-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 13:15:00 | 59.00 | 59.25 | 59.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-12 15:15:00 | 58.85 | 59.14 | 59.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-13 09:15:00 | 59.30 | 59.17 | 59.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 09:15:00 | 59.30 | 59.17 | 59.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 59.30 | 59.17 | 59.21 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-13 13:15:00 | 59.35 | 59.23 | 59.23 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 15:15:00 | 59.10 | 59.22 | 59.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-16 09:15:00 | 59.00 | 59.17 | 59.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 10:15:00 | 59.20 | 58.97 | 59.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 10:15:00 | 59.20 | 58.97 | 59.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 10:15:00 | 59.20 | 58.97 | 59.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 13:30:00 | 58.70 | 58.85 | 58.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 09:15:00 | 58.65 | 58.82 | 58.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-20 09:15:00 | 55.77 | 56.89 | 57.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-20 09:15:00 | 55.72 | 56.89 | 57.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2023-10-26 09:15:00 | 52.83 | 53.93 | 54.75 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 37 — BUY (started 2023-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 14:15:00 | 54.80 | 54.37 | 54.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 09:15:00 | 55.35 | 54.88 | 54.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 11:15:00 | 54.75 | 54.87 | 54.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 11:15:00 | 54.75 | 54.87 | 54.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 11:15:00 | 54.75 | 54.87 | 54.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 11:30:00 | 54.80 | 54.87 | 54.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 12:15:00 | 54.90 | 54.87 | 54.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 13:15:00 | 54.85 | 54.87 | 54.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 13:15:00 | 54.85 | 54.87 | 54.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 13:30:00 | 54.75 | 54.87 | 54.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 14:15:00 | 54.70 | 54.83 | 54.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 15:00:00 | 54.70 | 54.83 | 54.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 15:15:00 | 54.60 | 54.79 | 54.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-01 09:15:00 | 54.80 | 54.79 | 54.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-01 14:15:00 | 54.25 | 54.75 | 54.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2023-11-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 14:15:00 | 54.25 | 54.75 | 54.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 15:15:00 | 54.15 | 54.63 | 54.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 10:15:00 | 54.70 | 54.63 | 54.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 10:15:00 | 54.70 | 54.63 | 54.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 10:15:00 | 54.70 | 54.63 | 54.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:00:00 | 54.70 | 54.63 | 54.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 54.70 | 54.65 | 54.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:30:00 | 54.60 | 54.65 | 54.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 12:15:00 | 54.90 | 54.70 | 54.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 13:00:00 | 54.90 | 54.70 | 54.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2023-11-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 13:15:00 | 55.75 | 54.91 | 54.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 15:15:00 | 56.50 | 55.38 | 55.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 15:15:00 | 56.00 | 56.11 | 55.69 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 09:30:00 | 56.40 | 56.19 | 55.76 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 56.05 | 56.39 | 56.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-11-07 09:15:00 | 56.05 | 56.39 | 56.12 | SL hit (close<ema400) qty=1.00 sl=56.12 alert=retest1 |

### Cycle 40 — SELL (started 2023-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 13:15:00 | 57.60 | 58.03 | 58.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 11:15:00 | 57.25 | 57.58 | 57.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-22 14:15:00 | 57.35 | 57.33 | 57.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 14:15:00 | 57.35 | 57.33 | 57.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 14:15:00 | 57.35 | 57.33 | 57.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-22 15:00:00 | 57.35 | 57.33 | 57.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 57.50 | 57.37 | 57.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 14:30:00 | 57.15 | 57.37 | 57.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-24 10:15:00 | 57.20 | 57.34 | 57.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-24 11:15:00 | 58.05 | 57.48 | 57.49 | SL hit (close>static) qty=1.00 sl=57.85 alert=retest2 |

### Cycle 41 — BUY (started 2023-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 12:15:00 | 58.30 | 57.64 | 57.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 13:15:00 | 60.45 | 58.20 | 57.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 14:15:00 | 60.35 | 60.45 | 59.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-29 15:00:00 | 60.35 | 60.45 | 59.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 59.75 | 60.30 | 59.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 09:45:00 | 59.80 | 60.30 | 59.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 10:15:00 | 60.65 | 60.37 | 59.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-01 09:30:00 | 61.15 | 60.37 | 60.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-01 10:30:00 | 61.10 | 60.42 | 60.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 09:15:00 | 61.90 | 60.67 | 60.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-08 09:30:00 | 63.20 | 62.87 | 62.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-08 10:15:00 | 67.27 | 63.90 | 62.80 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 09:15:00 | 75.50 | 76.72 | 76.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 12:15:00 | 74.45 | 76.06 | 76.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 76.15 | 73.96 | 74.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 76.15 | 73.96 | 74.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 76.15 | 73.96 | 74.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:45:00 | 76.55 | 73.96 | 74.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 13:15:00 | 75.05 | 74.73 | 74.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 14:15:00 | 74.95 | 74.73 | 74.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 14:15:00 | 74.65 | 74.71 | 74.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 14:45:00 | 74.95 | 74.71 | 74.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 15:15:00 | 74.75 | 74.72 | 74.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 09:15:00 | 74.80 | 74.72 | 74.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2023-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 09:15:00 | 75.15 | 74.81 | 74.78 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2023-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-26 12:15:00 | 74.35 | 74.69 | 74.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-26 14:15:00 | 73.85 | 74.42 | 74.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-27 09:15:00 | 74.80 | 74.47 | 74.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 09:15:00 | 74.80 | 74.47 | 74.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 09:15:00 | 74.80 | 74.47 | 74.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 10:00:00 | 74.80 | 74.47 | 74.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 10:15:00 | 74.30 | 74.43 | 74.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-27 11:30:00 | 74.10 | 74.33 | 74.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-28 11:15:00 | 75.50 | 74.22 | 74.26 | SL hit (close>static) qty=1.00 sl=75.00 alert=retest2 |

### Cycle 45 — BUY (started 2023-12-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 12:15:00 | 76.85 | 74.74 | 74.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 12:15:00 | 77.45 | 76.39 | 75.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 15:15:00 | 80.20 | 80.25 | 78.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-02 09:15:00 | 79.95 | 80.25 | 78.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 09:15:00 | 78.90 | 79.98 | 78.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 09:45:00 | 78.80 | 79.98 | 78.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 77.90 | 79.57 | 78.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:30:00 | 77.80 | 79.57 | 78.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 11:15:00 | 78.45 | 79.34 | 78.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 12:30:00 | 78.80 | 79.20 | 78.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 13:30:00 | 78.85 | 79.16 | 78.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-08 09:15:00 | 86.68 | 84.48 | 83.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 09:15:00 | 84.60 | 85.40 | 85.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 14:15:00 | 84.20 | 84.81 | 85.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 09:15:00 | 86.00 | 84.12 | 84.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 09:15:00 | 86.00 | 84.12 | 84.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 86.00 | 84.12 | 84.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 09:45:00 | 86.00 | 84.12 | 84.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 10:15:00 | 85.20 | 84.33 | 84.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-16 12:45:00 | 84.20 | 84.23 | 84.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-17 10:30:00 | 84.65 | 84.42 | 84.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-17 11:30:00 | 84.35 | 84.33 | 84.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:15:00 | 80.42 | 83.60 | 83.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-18 11:15:00 | 83.55 | 83.49 | 83.86 | SL hit (close>ema200) qty=0.50 sl=83.49 alert=retest2 |

### Cycle 47 — BUY (started 2024-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 15:15:00 | 78.45 | 77.84 | 77.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-30 09:15:00 | 78.85 | 78.04 | 77.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 15:15:00 | 78.85 | 78.96 | 78.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 15:15:00 | 78.85 | 78.96 | 78.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 15:15:00 | 78.85 | 78.96 | 78.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 09:15:00 | 79.10 | 78.96 | 78.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 78.25 | 78.82 | 78.50 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-01-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 12:15:00 | 77.75 | 78.32 | 78.33 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 13:15:00 | 78.50 | 78.36 | 78.34 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-01-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 15:15:00 | 77.85 | 78.25 | 78.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 10:15:00 | 77.35 | 78.00 | 78.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-01 13:15:00 | 78.40 | 78.06 | 78.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 13:15:00 | 78.40 | 78.06 | 78.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 13:15:00 | 78.40 | 78.06 | 78.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-01 14:00:00 | 78.40 | 78.06 | 78.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-01 14:15:00 | 78.85 | 78.22 | 78.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-01 15:15:00 | 79.80 | 78.53 | 78.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-06 09:15:00 | 87.10 | 87.11 | 84.97 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-06 10:30:00 | 88.00 | 87.24 | 85.22 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-06 14:30:00 | 88.45 | 87.41 | 85.94 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 09:15:00 | 92.40 | 90.21 | 88.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-02-08 14:15:00 | 90.20 | 90.78 | 89.49 | SL hit (close<ema200) qty=0.50 sl=90.78 alert=retest1 |

### Cycle 52 — SELL (started 2024-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 11:15:00 | 87.60 | 88.89 | 88.93 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 12:15:00 | 89.40 | 88.14 | 87.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-13 14:15:00 | 89.80 | 88.64 | 88.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-14 12:15:00 | 89.30 | 89.41 | 88.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-14 13:00:00 | 89.30 | 89.41 | 88.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 09:15:00 | 89.70 | 89.69 | 89.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 09:45:00 | 89.25 | 89.69 | 89.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 14:15:00 | 90.70 | 90.00 | 89.53 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 15:15:00 | 88.75 | 89.55 | 89.55 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 09:15:00 | 89.75 | 89.59 | 89.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 10:15:00 | 90.10 | 89.69 | 89.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 13:15:00 | 91.05 | 91.65 | 91.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 13:15:00 | 91.05 | 91.65 | 91.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 13:15:00 | 91.05 | 91.65 | 91.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 14:00:00 | 91.05 | 91.65 | 91.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 14:15:00 | 90.80 | 91.48 | 91.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 14:45:00 | 90.45 | 91.48 | 91.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 15:15:00 | 90.15 | 91.21 | 90.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 09:15:00 | 87.65 | 91.21 | 90.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2024-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 09:15:00 | 88.70 | 90.71 | 90.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 13:15:00 | 86.40 | 88.68 | 89.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 11:15:00 | 87.70 | 87.38 | 88.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-22 12:00:00 | 87.70 | 87.38 | 88.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 89.00 | 87.85 | 88.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:30:00 | 88.50 | 87.85 | 88.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 10:15:00 | 88.70 | 88.02 | 88.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 11:00:00 | 88.70 | 88.02 | 88.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 11:15:00 | 88.60 | 88.14 | 88.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 11:45:00 | 88.60 | 88.14 | 88.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 14:15:00 | 88.15 | 88.24 | 88.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 14:45:00 | 88.70 | 88.24 | 88.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 15:15:00 | 87.85 | 88.16 | 88.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 09:15:00 | 88.65 | 88.16 | 88.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 88.35 | 88.20 | 88.33 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 12:15:00 | 89.15 | 88.49 | 88.44 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 14:15:00 | 87.80 | 88.32 | 88.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 15:15:00 | 87.45 | 88.15 | 88.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-28 09:15:00 | 87.25 | 86.59 | 87.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 09:15:00 | 87.25 | 86.59 | 87.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 87.25 | 86.59 | 87.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 10:00:00 | 87.25 | 86.59 | 87.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 10:15:00 | 85.60 | 86.39 | 87.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 12:45:00 | 84.70 | 85.90 | 86.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 14:15:00 | 84.95 | 85.83 | 86.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 10:00:00 | 85.15 | 84.30 | 85.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 15:00:00 | 85.10 | 84.97 | 85.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 15:15:00 | 85.95 | 85.16 | 85.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 09:15:00 | 86.20 | 85.16 | 85.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-03-02 09:15:00 | 85.95 | 85.32 | 85.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2024-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 09:15:00 | 85.95 | 85.32 | 85.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-04 10:15:00 | 86.95 | 85.94 | 85.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 14:15:00 | 86.60 | 86.68 | 86.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-05 14:45:00 | 86.60 | 86.68 | 86.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 60 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 84.15 | 86.16 | 86.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 81.70 | 84.81 | 85.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 10:15:00 | 83.70 | 83.53 | 84.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-07 11:00:00 | 83.70 | 83.53 | 84.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 12:15:00 | 84.35 | 83.79 | 84.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 13:00:00 | 84.35 | 83.79 | 84.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 13:15:00 | 84.45 | 83.92 | 84.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 13:30:00 | 84.50 | 83.92 | 84.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 14:15:00 | 84.45 | 84.03 | 84.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 15:15:00 | 84.25 | 84.03 | 84.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 09:15:00 | 80.04 | 81.01 | 82.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-13 12:15:00 | 75.83 | 78.85 | 80.80 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 61 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 78.10 | 76.06 | 76.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 13:15:00 | 78.85 | 77.03 | 76.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 78.55 | 78.70 | 77.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-22 15:00:00 | 78.55 | 78.70 | 77.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 78.70 | 78.66 | 78.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 14:15:00 | 79.65 | 78.85 | 78.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 09:15:00 | 80.55 | 78.88 | 78.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-09 15:15:00 | 84.80 | 85.19 | 85.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2024-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 15:15:00 | 84.80 | 85.19 | 85.21 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 15:15:00 | 85.50 | 85.21 | 85.20 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-04-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 09:15:00 | 84.95 | 85.16 | 85.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 13:15:00 | 83.90 | 84.58 | 84.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 09:15:00 | 81.55 | 80.91 | 81.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 09:15:00 | 81.55 | 80.91 | 81.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 81.55 | 80.91 | 81.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:30:00 | 81.35 | 80.91 | 81.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 81.60 | 81.05 | 81.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 10:45:00 | 81.75 | 81.05 | 81.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 11:15:00 | 81.55 | 81.15 | 81.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 11:45:00 | 81.60 | 81.15 | 81.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 12:15:00 | 82.25 | 81.37 | 81.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 13:00:00 | 82.25 | 81.37 | 81.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 13:15:00 | 81.25 | 81.35 | 81.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 14:30:00 | 80.60 | 81.02 | 81.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 14:00:00 | 80.75 | 80.69 | 81.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 11:00:00 | 81.00 | 80.79 | 81.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 11:30:00 | 80.85 | 80.87 | 81.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 12:15:00 | 80.90 | 80.88 | 81.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 14:15:00 | 80.75 | 80.87 | 81.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-23 10:15:00 | 82.15 | 81.24 | 81.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2024-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 10:15:00 | 82.15 | 81.24 | 81.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 11:15:00 | 82.45 | 81.48 | 81.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 14:15:00 | 82.40 | 82.54 | 82.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-24 14:15:00 | 82.40 | 82.54 | 82.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 14:15:00 | 82.40 | 82.54 | 82.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-24 14:30:00 | 82.40 | 82.54 | 82.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 84.25 | 82.90 | 82.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 13:00:00 | 84.80 | 83.63 | 82.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 13:30:00 | 84.90 | 83.81 | 83.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-30 12:15:00 | 85.75 | 87.01 | 87.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 12:15:00 | 85.75 | 87.01 | 87.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 14:15:00 | 85.10 | 86.45 | 86.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-02 11:15:00 | 87.10 | 85.72 | 86.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 11:15:00 | 87.10 | 85.72 | 86.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 11:15:00 | 87.10 | 85.72 | 86.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 12:00:00 | 87.10 | 85.72 | 86.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 12:15:00 | 87.70 | 86.12 | 86.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 13:00:00 | 87.70 | 86.12 | 86.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 13:15:00 | 88.15 | 86.53 | 86.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 13:45:00 | 88.25 | 86.53 | 86.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2024-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 14:15:00 | 87.85 | 86.79 | 86.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-02 15:15:00 | 88.40 | 87.11 | 86.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-03 09:15:00 | 86.65 | 87.02 | 86.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 09:15:00 | 86.65 | 87.02 | 86.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 09:15:00 | 86.65 | 87.02 | 86.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 10:00:00 | 86.65 | 87.02 | 86.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 10:15:00 | 86.00 | 86.82 | 86.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 11:00:00 | 86.00 | 86.82 | 86.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2024-05-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 11:15:00 | 85.45 | 86.54 | 86.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 09:15:00 | 82.10 | 85.19 | 85.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 10:15:00 | 81.00 | 80.39 | 81.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 10:45:00 | 80.85 | 80.39 | 81.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 81.85 | 80.85 | 81.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 13:00:00 | 81.85 | 80.85 | 81.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 81.00 | 80.88 | 81.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 14:15:00 | 80.60 | 80.88 | 81.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 09:30:00 | 80.70 | 80.79 | 81.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 12:45:00 | 79.95 | 80.61 | 81.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 13:15:00 | 81.85 | 80.74 | 80.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2024-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 13:15:00 | 81.85 | 80.74 | 80.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 14:15:00 | 82.20 | 81.04 | 80.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 11:15:00 | 83.10 | 83.12 | 82.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 11:45:00 | 83.20 | 83.12 | 82.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 83.50 | 83.36 | 82.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 83.65 | 83.11 | 82.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 10:15:00 | 83.65 | 83.16 | 82.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 10:15:00 | 86.60 | 87.34 | 87.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 10:15:00 | 86.60 | 87.34 | 87.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 85.35 | 86.84 | 87.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 09:15:00 | 83.90 | 83.73 | 85.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 14:15:00 | 84.85 | 84.27 | 84.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 84.85 | 84.27 | 84.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 84.85 | 84.27 | 84.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 84.30 | 84.27 | 84.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 87.30 | 84.27 | 84.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 88.35 | 85.09 | 85.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:30:00 | 88.65 | 85.09 | 85.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 88.30 | 85.73 | 85.45 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 79.85 | 85.44 | 85.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 75.10 | 83.37 | 84.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 79.35 | 78.49 | 80.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 13:00:00 | 79.35 | 78.49 | 80.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 80.90 | 78.97 | 80.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:00:00 | 80.90 | 78.97 | 80.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 81.95 | 79.57 | 80.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 81.95 | 79.57 | 80.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 81.70 | 79.99 | 80.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 84.35 | 79.99 | 80.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 85.90 | 81.99 | 81.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 13:15:00 | 86.60 | 85.16 | 83.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 86.70 | 86.83 | 85.76 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:15:00 | 90.18 | 86.83 | 85.76 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 89.73 | 90.47 | 89.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 12:30:00 | 89.55 | 90.47 | 89.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 91.20 | 90.39 | 89.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 13:00:00 | 92.30 | 90.77 | 89.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 15:00:00 | 91.85 | 91.13 | 90.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 91.85 | 91.22 | 90.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 09:15:00 | 94.69 | 93.73 | 92.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-06-19 10:15:00 | 96.14 | 96.16 | 94.64 | SL hit (close<ema200) qty=0.50 sl=96.16 alert=retest1 |

### Cycle 74 — SELL (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 10:15:00 | 96.00 | 97.16 | 97.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 12:15:00 | 95.63 | 96.66 | 97.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 09:15:00 | 97.33 | 96.22 | 96.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 09:15:00 | 97.33 | 96.22 | 96.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 97.33 | 96.22 | 96.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:30:00 | 97.92 | 96.22 | 96.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 97.90 | 96.56 | 96.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:30:00 | 98.87 | 96.56 | 96.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 11:15:00 | 98.24 | 96.89 | 96.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 13:15:00 | 98.75 | 97.41 | 97.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 13:15:00 | 98.07 | 98.50 | 97.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 13:15:00 | 98.07 | 98.50 | 97.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 98.07 | 98.50 | 97.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 98.07 | 98.50 | 97.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 99.35 | 98.67 | 98.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 98.20 | 98.67 | 98.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 97.29 | 98.49 | 98.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:30:00 | 98.00 | 98.49 | 98.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 97.11 | 98.21 | 98.02 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2024-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 11:15:00 | 96.57 | 97.88 | 97.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 12:15:00 | 96.03 | 97.51 | 97.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 10:15:00 | 96.99 | 96.98 | 97.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-01 10:30:00 | 97.15 | 96.98 | 97.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 11:15:00 | 97.02 | 96.99 | 97.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 12:00:00 | 97.02 | 96.99 | 97.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 12:15:00 | 97.07 | 97.00 | 97.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 09:30:00 | 96.89 | 97.25 | 97.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 10:15:00 | 97.50 | 97.30 | 97.35 | SL hit (close>static) qty=1.00 sl=97.45 alert=retest2 |

### Cycle 77 — BUY (started 2024-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 10:15:00 | 97.27 | 96.72 | 96.72 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 13:15:00 | 96.42 | 96.79 | 96.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 14:15:00 | 96.20 | 96.67 | 96.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 12:15:00 | 96.22 | 96.09 | 96.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-08 13:00:00 | 96.22 | 96.09 | 96.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 79 — BUY (started 2024-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 14:15:00 | 99.25 | 96.71 | 96.62 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 13:15:00 | 96.89 | 97.34 | 97.37 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 15:15:00 | 97.80 | 97.46 | 97.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 09:15:00 | 98.35 | 97.63 | 97.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 11:15:00 | 98.49 | 98.65 | 98.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-12 12:00:00 | 98.49 | 98.65 | 98.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 98.20 | 98.56 | 98.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:30:00 | 98.14 | 98.56 | 98.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 98.21 | 98.49 | 98.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 13:45:00 | 98.15 | 98.49 | 98.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 97.88 | 98.37 | 98.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 14:30:00 | 97.94 | 98.37 | 98.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 97.35 | 98.17 | 98.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 96.95 | 98.17 | 98.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 09:15:00 | 97.45 | 98.02 | 98.04 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 11:15:00 | 98.35 | 98.10 | 98.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 98.60 | 98.21 | 98.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 12:15:00 | 97.80 | 98.29 | 98.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 12:15:00 | 97.80 | 98.29 | 98.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 97.80 | 98.29 | 98.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:45:00 | 97.78 | 98.29 | 98.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2024-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 13:15:00 | 97.01 | 98.04 | 98.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 96.19 | 97.33 | 97.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 11:15:00 | 93.05 | 92.90 | 94.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 11:30:00 | 92.91 | 92.90 | 94.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 93.99 | 93.24 | 94.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 93.99 | 93.24 | 94.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 94.34 | 93.46 | 94.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 93.25 | 93.46 | 94.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 93.22 | 93.41 | 93.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 91.12 | 94.22 | 94.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 95.32 | 94.36 | 94.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 95.32 | 94.36 | 94.31 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 12:15:00 | 94.15 | 94.51 | 94.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 13:15:00 | 93.70 | 94.35 | 94.44 | Break + close below crossover candle low |

### Cycle 87 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 95.71 | 94.57 | 94.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 11:15:00 | 96.11 | 95.08 | 94.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 96.51 | 96.98 | 96.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:00:00 | 96.51 | 96.98 | 96.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 97.14 | 97.01 | 96.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 09:15:00 | 97.75 | 97.05 | 96.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 13:15:00 | 99.25 | 99.99 | 100.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 13:15:00 | 99.25 | 99.99 | 100.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 14:15:00 | 98.12 | 99.62 | 99.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 95.39 | 94.12 | 96.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 10:00:00 | 95.39 | 94.12 | 96.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 95.08 | 93.78 | 94.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 09:30:00 | 95.72 | 93.78 | 94.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 95.69 | 94.16 | 95.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:30:00 | 96.08 | 94.16 | 95.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 95.63 | 94.45 | 95.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:30:00 | 95.81 | 94.45 | 95.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 14:15:00 | 97.14 | 95.59 | 95.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 09:15:00 | 98.12 | 96.32 | 95.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 15:15:00 | 99.20 | 99.23 | 98.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 09:15:00 | 98.67 | 99.23 | 98.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 98.86 | 99.15 | 98.37 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 09:15:00 | 97.34 | 98.22 | 98.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 96.49 | 97.56 | 97.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 94.04 | 93.68 | 95.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 12:15:00 | 95.09 | 94.05 | 94.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 95.09 | 94.05 | 94.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:30:00 | 95.13 | 94.05 | 94.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 95.64 | 94.37 | 94.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:00:00 | 95.64 | 94.37 | 94.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 96.02 | 94.70 | 95.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 15:00:00 | 96.02 | 94.70 | 95.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 96.75 | 95.23 | 95.21 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 09:15:00 | 95.17 | 95.68 | 95.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-21 14:15:00 | 94.84 | 95.25 | 95.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 09:15:00 | 95.44 | 95.26 | 95.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 09:15:00 | 95.44 | 95.26 | 95.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 95.44 | 95.26 | 95.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 12:45:00 | 95.29 | 95.29 | 95.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 09:15:00 | 97.27 | 95.64 | 95.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 09:15:00 | 97.27 | 95.64 | 95.52 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 12:15:00 | 95.28 | 95.79 | 95.81 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 14:15:00 | 96.08 | 95.86 | 95.84 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 15:15:00 | 95.65 | 95.82 | 95.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 10:15:00 | 95.46 | 95.73 | 95.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 14:15:00 | 95.64 | 95.56 | 95.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 14:15:00 | 95.64 | 95.56 | 95.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 95.64 | 95.56 | 95.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:30:00 | 95.54 | 95.56 | 95.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 95.40 | 95.53 | 95.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:15:00 | 95.36 | 95.53 | 95.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 95.03 | 95.43 | 95.59 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 11:15:00 | 96.50 | 95.71 | 95.69 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 14:15:00 | 94.67 | 95.55 | 95.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 15:15:00 | 94.40 | 95.32 | 95.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 15:15:00 | 94.38 | 94.14 | 94.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 09:15:00 | 94.52 | 94.14 | 94.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 94.10 | 94.13 | 94.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 09:45:00 | 93.38 | 94.09 | 94.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 10:15:00 | 93.39 | 93.52 | 93.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 13:45:00 | 93.65 | 93.68 | 93.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 09:15:00 | 93.70 | 93.84 | 93.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 09:15:00 | 94.62 | 94.00 | 93.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 09:15:00 | 94.62 | 94.00 | 93.96 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 13:15:00 | 93.67 | 93.90 | 93.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 15:15:00 | 93.30 | 93.74 | 93.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 94.13 | 93.82 | 93.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 09:15:00 | 94.13 | 93.82 | 93.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 94.13 | 93.82 | 93.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:45:00 | 94.18 | 93.82 | 93.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 10:15:00 | 94.44 | 93.94 | 93.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 12:15:00 | 95.09 | 94.32 | 94.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 92.40 | 94.60 | 94.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 92.40 | 94.60 | 94.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 92.40 | 94.60 | 94.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 92.40 | 94.60 | 94.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 91.59 | 94.00 | 94.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 91.15 | 92.65 | 93.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 90.95 | 90.64 | 91.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 15:00:00 | 90.95 | 90.64 | 91.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 92.27 | 91.05 | 91.78 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2024-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 09:15:00 | 93.38 | 92.29 | 92.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 15:15:00 | 94.11 | 93.44 | 93.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 11:15:00 | 96.47 | 96.72 | 95.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 12:00:00 | 96.47 | 96.72 | 95.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 95.75 | 96.49 | 96.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 15:00:00 | 95.75 | 96.49 | 96.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 95.62 | 96.31 | 95.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 09:15:00 | 96.09 | 96.31 | 95.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 10:15:00 | 95.20 | 96.02 | 95.90 | SL hit (close<static) qty=1.00 sl=95.50 alert=retest2 |

### Cycle 104 — SELL (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 11:15:00 | 94.46 | 95.71 | 95.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 94.06 | 95.38 | 95.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 15:15:00 | 93.10 | 92.95 | 93.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 09:15:00 | 93.20 | 92.95 | 93.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 93.20 | 93.00 | 93.81 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 94.86 | 94.00 | 93.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 13:15:00 | 95.70 | 95.15 | 94.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 94.63 | 95.11 | 94.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 94.63 | 95.11 | 94.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 94.63 | 95.11 | 94.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 94.63 | 95.11 | 94.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 94.99 | 95.08 | 94.85 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2024-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 13:15:00 | 94.43 | 94.66 | 94.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 93.18 | 94.35 | 94.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 93.92 | 93.72 | 94.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 14:15:00 | 93.92 | 93.72 | 94.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 93.92 | 93.72 | 94.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 93.92 | 93.72 | 94.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 94.15 | 93.80 | 94.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 94.50 | 93.80 | 94.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 94.89 | 94.02 | 94.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:00:00 | 94.89 | 94.02 | 94.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 10:15:00 | 95.37 | 94.29 | 94.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 11:15:00 | 95.43 | 94.52 | 94.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 10:15:00 | 94.57 | 94.72 | 94.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 10:15:00 | 94.57 | 94.72 | 94.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 94.57 | 94.72 | 94.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:30:00 | 94.31 | 94.72 | 94.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 94.69 | 94.71 | 94.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:45:00 | 94.60 | 94.71 | 94.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 94.63 | 94.70 | 94.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:30:00 | 94.61 | 94.70 | 94.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 93.92 | 94.54 | 94.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:00:00 | 93.92 | 94.54 | 94.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2024-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 14:15:00 | 94.14 | 94.46 | 94.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 91.65 | 93.61 | 94.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 10:15:00 | 90.68 | 90.41 | 91.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 10:30:00 | 90.59 | 90.41 | 91.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 87.97 | 87.12 | 87.73 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 89.89 | 88.29 | 88.14 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2024-10-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 15:15:00 | 88.35 | 88.51 | 88.52 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 09:15:00 | 88.86 | 88.58 | 88.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 14:15:00 | 89.55 | 89.09 | 88.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 11:15:00 | 88.84 | 89.16 | 88.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 11:15:00 | 88.84 | 89.16 | 88.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 88.84 | 89.16 | 88.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:45:00 | 88.88 | 89.16 | 88.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 88.95 | 89.12 | 88.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 12:30:00 | 88.84 | 89.12 | 88.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 89.41 | 89.21 | 89.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 14:30:00 | 89.38 | 89.21 | 89.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 89.10 | 89.18 | 89.05 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 87.63 | 88.87 | 88.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 86.63 | 87.81 | 88.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 15:15:00 | 81.38 | 81.38 | 82.99 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-23 09:30:00 | 79.75 | 81.39 | 82.85 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 82.19 | 81.55 | 82.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:00:00 | 82.19 | 81.55 | 82.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 82.37 | 81.79 | 82.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 14:00:00 | 82.37 | 81.79 | 82.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 14:15:00 | 82.14 | 81.86 | 82.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 15:00:00 | 82.14 | 81.86 | 82.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 83.16 | 82.14 | 82.57 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-24 09:15:00 | 83.16 | 82.14 | 82.57 | SL hit (close>ema400) qty=1.00 sl=82.57 alert=retest1 |

### Cycle 113 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 81.32 | 79.57 | 79.37 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2024-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 14:15:00 | 79.41 | 79.71 | 79.73 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-11-01 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 18:15:00 | 79.95 | 79.77 | 79.75 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 77.94 | 79.41 | 79.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 76.99 | 78.92 | 79.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 77.73 | 77.31 | 77.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 14:00:00 | 77.73 | 77.31 | 77.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 78.56 | 77.56 | 78.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 78.56 | 77.56 | 78.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 78.85 | 77.82 | 78.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 79.42 | 77.82 | 78.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 79.50 | 78.15 | 78.24 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 80.12 | 78.55 | 78.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 11:15:00 | 80.83 | 79.00 | 78.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 14:15:00 | 80.88 | 81.15 | 80.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 15:00:00 | 80.88 | 81.15 | 80.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 81.58 | 81.20 | 80.52 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 79.09 | 80.24 | 80.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 13:15:00 | 78.67 | 79.42 | 79.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 77.14 | 76.91 | 77.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 77.14 | 76.91 | 77.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 77.07 | 76.87 | 77.37 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 13:15:00 | 78.44 | 77.61 | 77.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 81.71 | 78.56 | 78.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 15:15:00 | 80.15 | 80.22 | 79.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-21 09:15:00 | 77.45 | 80.22 | 79.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 76.07 | 79.39 | 79.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:00:00 | 76.07 | 79.39 | 79.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 76.25 | 78.76 | 78.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:45:00 | 75.80 | 78.76 | 78.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 11:15:00 | 77.23 | 78.45 | 78.62 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2024-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 12:15:00 | 79.26 | 78.34 | 78.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 13:15:00 | 79.47 | 78.56 | 78.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 11:15:00 | 80.19 | 80.30 | 79.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 12:00:00 | 80.19 | 80.30 | 79.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 13:15:00 | 80.34 | 80.26 | 79.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 13:30:00 | 80.01 | 80.26 | 79.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 80.62 | 80.31 | 79.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 80.26 | 80.31 | 79.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 83.69 | 83.60 | 83.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 11:15:00 | 84.07 | 83.60 | 83.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 10:30:00 | 83.75 | 83.72 | 83.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 12:00:00 | 83.80 | 83.74 | 83.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 12:30:00 | 83.75 | 83.77 | 83.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 83.99 | 83.84 | 83.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 83.56 | 83.84 | 83.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 85.46 | 85.80 | 85.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-12 10:15:00 | 84.62 | 85.13 | 85.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 84.62 | 85.13 | 85.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 82.76 | 84.39 | 84.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 83.88 | 83.80 | 84.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 14:00:00 | 83.88 | 83.80 | 84.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 84.00 | 83.87 | 84.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:30:00 | 84.26 | 83.96 | 84.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 83.94 | 83.95 | 84.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:45:00 | 83.79 | 83.90 | 84.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 13:45:00 | 83.84 | 83.92 | 84.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 14:15:00 | 83.80 | 83.92 | 84.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 09:15:00 | 84.59 | 84.07 | 84.16 | SL hit (close>static) qty=1.00 sl=84.40 alert=retest2 |

### Cycle 123 — BUY (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 15:15:00 | 78.70 | 78.54 | 78.54 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 11:15:00 | 78.15 | 78.47 | 78.51 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 14:15:00 | 78.66 | 78.54 | 78.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 79.11 | 78.68 | 78.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 13:15:00 | 78.71 | 78.92 | 78.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 13:15:00 | 78.71 | 78.92 | 78.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 78.71 | 78.92 | 78.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:00:00 | 78.71 | 78.92 | 78.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 78.70 | 78.87 | 78.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 78.70 | 78.87 | 78.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 78.64 | 78.83 | 78.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 77.59 | 78.83 | 78.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 77.79 | 78.62 | 78.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 76.30 | 78.16 | 78.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 13:15:00 | 76.95 | 76.86 | 77.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 13:15:00 | 76.95 | 76.86 | 77.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 76.95 | 76.86 | 77.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:30:00 | 77.31 | 76.86 | 77.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 76.87 | 76.55 | 76.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:45:00 | 76.40 | 76.59 | 76.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 12:15:00 | 76.54 | 76.64 | 76.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 14:15:00 | 76.55 | 76.61 | 76.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 72.71 | 74.91 | 75.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 72.72 | 74.91 | 75.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 10:15:00 | 72.58 | 74.39 | 75.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 72.24 | 71.78 | 73.27 | SL hit (close>ema200) qty=0.50 sl=71.78 alert=retest2 |

### Cycle 127 — BUY (started 2025-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 13:15:00 | 73.73 | 73.18 | 73.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 14:15:00 | 74.37 | 73.42 | 73.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 75.57 | 76.39 | 75.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 75.57 | 76.39 | 75.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 75.57 | 76.39 | 75.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 75.57 | 76.39 | 75.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 74.71 | 76.06 | 75.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 74.71 | 76.06 | 75.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 74.39 | 75.36 | 75.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 72.80 | 74.68 | 75.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 73.05 | 72.99 | 74.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 73.05 | 72.99 | 74.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 73.50 | 73.15 | 73.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 15:00:00 | 73.50 | 73.15 | 73.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 73.50 | 73.22 | 73.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 73.04 | 73.22 | 73.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 72.34 | 73.05 | 73.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 13:30:00 | 72.20 | 72.76 | 73.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 10:15:00 | 72.97 | 71.81 | 71.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 72.97 | 71.81 | 71.67 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 14:15:00 | 71.32 | 71.60 | 71.61 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 12:15:00 | 71.88 | 71.63 | 71.61 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 71.14 | 71.53 | 71.56 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 10:15:00 | 72.03 | 71.55 | 71.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 11:15:00 | 72.78 | 71.79 | 71.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 72.94 | 72.98 | 72.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 11:45:00 | 73.00 | 72.98 | 72.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 72.62 | 73.27 | 72.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 11:00:00 | 73.64 | 73.34 | 72.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 15:00:00 | 73.59 | 73.23 | 72.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:15:00 | 74.32 | 73.29 | 73.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 11:45:00 | 74.06 | 73.48 | 73.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 74.39 | 75.06 | 74.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 74.46 | 75.06 | 74.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 74.63 | 74.98 | 74.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-07 13:15:00 | 73.94 | 74.48 | 74.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 73.94 | 74.48 | 74.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 73.61 | 74.37 | 74.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 72.30 | 71.86 | 72.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:00:00 | 72.30 | 71.86 | 72.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 72.20 | 72.00 | 72.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:15:00 | 72.95 | 72.00 | 72.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 73.25 | 72.25 | 72.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 73.24 | 72.25 | 72.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2025-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 12:15:00 | 72.99 | 72.69 | 72.68 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 15:15:00 | 72.30 | 72.61 | 72.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 71.65 | 72.42 | 72.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 70.30 | 69.78 | 70.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 15:00:00 | 70.30 | 69.78 | 70.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 70.60 | 69.94 | 70.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 69.96 | 69.94 | 70.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 69.61 | 69.88 | 70.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 10:15:00 | 69.43 | 69.88 | 70.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:15:00 | 69.48 | 69.80 | 70.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:45:00 | 69.15 | 69.68 | 70.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 11:15:00 | 70.70 | 70.28 | 70.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 11:15:00 | 70.70 | 70.28 | 70.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 14:15:00 | 71.60 | 70.66 | 70.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 70.62 | 70.76 | 70.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:15:00 | 70.73 | 70.76 | 70.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 70.64 | 70.74 | 70.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:15:00 | 70.53 | 70.74 | 70.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 70.34 | 70.66 | 70.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 12:00:00 | 70.34 | 70.66 | 70.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 12:15:00 | 70.31 | 70.59 | 70.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:00:00 | 70.31 | 70.59 | 70.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 69.89 | 70.45 | 70.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:45:00 | 69.75 | 70.45 | 70.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2025-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 14:15:00 | 70.08 | 70.37 | 70.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 68.83 | 69.97 | 70.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 69.97 | 69.69 | 69.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 69.97 | 69.69 | 69.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 69.97 | 69.69 | 69.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:45:00 | 69.92 | 69.69 | 69.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 70.07 | 69.77 | 69.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:30:00 | 70.15 | 69.77 | 69.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 69.71 | 69.76 | 69.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 12:45:00 | 69.37 | 69.61 | 69.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 15:15:00 | 69.35 | 69.10 | 69.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-03 13:15:00 | 69.78 | 69.35 | 69.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2025-03-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 13:15:00 | 69.78 | 69.35 | 69.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 09:15:00 | 70.72 | 69.76 | 69.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 73.62 | 73.69 | 72.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 09:15:00 | 74.10 | 73.69 | 72.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 72.95 | 73.57 | 73.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:30:00 | 73.60 | 73.15 | 72.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 10:15:00 | 73.44 | 73.15 | 72.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 13:15:00 | 72.60 | 73.00 | 72.96 | SL hit (close<static) qty=1.00 sl=72.83 alert=retest2 |

### Cycle 140 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 71.91 | 72.78 | 72.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 71.50 | 72.52 | 72.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 10:15:00 | 73.50 | 72.71 | 72.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 10:15:00 | 73.50 | 72.71 | 72.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 73.50 | 72.71 | 72.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:00:00 | 73.50 | 72.71 | 72.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 11:15:00 | 73.76 | 72.92 | 72.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 13:15:00 | 74.22 | 73.30 | 73.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 10:15:00 | 73.30 | 73.56 | 73.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 10:15:00 | 73.30 | 73.56 | 73.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 73.30 | 73.56 | 73.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:45:00 | 73.13 | 73.56 | 73.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 72.85 | 73.42 | 73.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:00:00 | 72.85 | 73.42 | 73.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 73.12 | 73.36 | 73.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 13:15:00 | 73.42 | 73.36 | 73.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 14:00:00 | 73.37 | 73.36 | 73.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 09:30:00 | 73.20 | 73.46 | 73.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 12:15:00 | 76.38 | 77.10 | 77.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 76.38 | 77.10 | 77.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 14:15:00 | 75.65 | 76.69 | 76.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 10:15:00 | 76.75 | 76.41 | 76.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 10:15:00 | 76.75 | 76.41 | 76.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 76.75 | 76.41 | 76.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:45:00 | 76.72 | 76.41 | 76.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 77.08 | 76.55 | 76.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 77.08 | 76.55 | 76.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 76.83 | 76.60 | 76.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:30:00 | 77.10 | 76.60 | 76.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 76.36 | 76.55 | 76.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 14:00:00 | 76.36 | 76.55 | 76.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 76.41 | 76.53 | 76.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 14:30:00 | 76.27 | 76.53 | 76.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 76.40 | 76.46 | 76.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 12:30:00 | 75.39 | 76.10 | 76.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 10:15:00 | 76.83 | 76.19 | 76.31 | SL hit (close>static) qty=1.00 sl=76.74 alert=retest2 |

### Cycle 143 — BUY (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 11:15:00 | 76.60 | 76.33 | 76.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 12:15:00 | 77.09 | 76.48 | 76.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 11:15:00 | 81.69 | 81.85 | 80.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 12:00:00 | 81.69 | 81.85 | 80.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 13:15:00 | 80.88 | 81.57 | 80.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:45:00 | 80.72 | 81.57 | 80.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 80.75 | 81.41 | 80.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 15:00:00 | 80.75 | 81.41 | 80.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 79.49 | 80.96 | 80.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 11:00:00 | 81.05 | 80.98 | 80.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 12:15:00 | 86.31 | 86.53 | 86.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 12:15:00 | 86.31 | 86.53 | 86.56 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 14:15:00 | 87.79 | 86.76 | 86.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 88.10 | 87.15 | 86.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 86.13 | 87.94 | 87.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 86.13 | 87.94 | 87.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 86.13 | 87.94 | 87.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 86.13 | 87.94 | 87.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 84.46 | 87.25 | 87.29 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 11:15:00 | 88.18 | 87.14 | 87.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 89.71 | 88.12 | 87.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 12:15:00 | 88.80 | 88.97 | 88.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 13:00:00 | 88.80 | 88.97 | 88.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 13:15:00 | 88.66 | 88.91 | 88.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:00:00 | 88.66 | 88.91 | 88.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 87.24 | 88.58 | 88.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 15:00:00 | 87.24 | 88.58 | 88.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 86.90 | 88.24 | 88.28 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 88.44 | 88.10 | 88.07 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 87.00 | 87.88 | 87.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 10:15:00 | 86.69 | 87.64 | 87.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 86.75 | 86.62 | 87.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:30:00 | 86.81 | 86.62 | 87.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 87.07 | 86.53 | 86.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:30:00 | 87.04 | 86.53 | 86.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 86.94 | 86.61 | 86.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 86.43 | 86.61 | 86.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 87.26 | 86.74 | 86.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:30:00 | 86.07 | 86.61 | 86.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 81.77 | 85.48 | 86.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 84.30 | 84.20 | 85.10 | SL hit (close>ema200) qty=0.50 sl=84.20 alert=retest2 |

### Cycle 151 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 88.70 | 86.10 | 85.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 10:15:00 | 89.00 | 88.12 | 87.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 09:15:00 | 89.96 | 89.98 | 89.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 09:45:00 | 89.91 | 89.98 | 89.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 89.15 | 89.71 | 89.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:00:00 | 89.15 | 89.71 | 89.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 88.80 | 89.53 | 89.16 | EMA400 retest candle locked (from upside) |

### Cycle 152 — SELL (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 14:15:00 | 87.46 | 88.86 | 88.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 87.05 | 87.83 | 88.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 87.06 | 87.01 | 87.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:00:00 | 87.06 | 87.01 | 87.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 86.65 | 87.01 | 87.59 | EMA400 retest candle locked (from downside) |

### Cycle 153 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 88.30 | 87.75 | 87.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 14:15:00 | 88.69 | 88.10 | 87.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 09:15:00 | 86.93 | 88.04 | 87.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 86.93 | 88.04 | 87.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 86.93 | 88.04 | 87.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 86.97 | 88.04 | 87.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 87.16 | 87.87 | 87.86 | EMA400 retest candle locked (from upside) |

### Cycle 154 — SELL (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 11:15:00 | 87.28 | 87.75 | 87.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 12:15:00 | 86.97 | 87.59 | 87.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 09:15:00 | 86.89 | 86.80 | 87.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 86.89 | 86.80 | 87.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 86.89 | 86.80 | 87.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 10:00:00 | 85.85 | 86.33 | 86.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 11:00:00 | 85.95 | 86.26 | 86.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 12:00:00 | 86.05 | 86.22 | 86.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 12:30:00 | 85.96 | 86.26 | 86.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 86.15 | 86.24 | 86.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 85.76 | 86.30 | 86.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 85.50 | 84.90 | 84.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 09:15:00 | 85.50 | 84.90 | 84.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 86.27 | 85.35 | 85.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 15:15:00 | 86.35 | 86.35 | 85.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 09:15:00 | 85.91 | 86.35 | 85.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 86.10 | 86.30 | 85.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:30:00 | 85.98 | 86.30 | 85.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 85.78 | 86.20 | 85.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 85.78 | 86.20 | 85.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 85.60 | 86.08 | 85.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:00:00 | 85.60 | 86.08 | 85.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 85.48 | 85.92 | 85.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:00:00 | 85.48 | 85.92 | 85.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 14:15:00 | 85.11 | 85.76 | 85.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 84.27 | 85.24 | 85.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 81.96 | 81.55 | 82.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 10:30:00 | 81.90 | 81.55 | 82.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 82.32 | 81.70 | 82.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:45:00 | 82.36 | 81.70 | 82.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 82.75 | 81.91 | 82.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 82.75 | 81.91 | 82.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 83.55 | 82.24 | 82.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 83.55 | 82.24 | 82.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 15:15:00 | 83.80 | 82.75 | 82.75 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-06-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 15:15:00 | 82.64 | 82.82 | 82.82 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 09:15:00 | 83.24 | 82.90 | 82.86 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 82.23 | 82.74 | 82.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 81.61 | 82.52 | 82.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 81.49 | 81.30 | 81.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 11:00:00 | 81.49 | 81.30 | 81.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 81.42 | 81.17 | 81.52 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 83.17 | 81.75 | 81.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 12:15:00 | 83.60 | 82.54 | 82.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 84.88 | 84.90 | 84.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 84.88 | 84.90 | 84.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 84.88 | 84.90 | 84.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 84.88 | 84.90 | 84.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 84.64 | 84.86 | 84.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:45:00 | 84.70 | 84.86 | 84.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 84.73 | 84.84 | 84.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:45:00 | 84.51 | 84.84 | 84.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 92.10 | 92.24 | 91.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:15:00 | 92.39 | 92.24 | 91.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 91.45 | 92.08 | 91.80 | SL hit (close<static) qty=1.00 sl=91.65 alert=retest2 |

### Cycle 162 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 91.12 | 91.58 | 91.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 14:15:00 | 90.68 | 91.35 | 91.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 11:15:00 | 91.06 | 91.02 | 91.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 12:00:00 | 91.06 | 91.02 | 91.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 91.93 | 91.20 | 91.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:30:00 | 92.24 | 91.20 | 91.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 91.65 | 91.29 | 91.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:15:00 | 91.42 | 91.37 | 91.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 09:30:00 | 91.13 | 91.30 | 91.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 14:15:00 | 91.91 | 91.36 | 91.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 91.91 | 91.36 | 91.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 15:15:00 | 92.20 | 91.53 | 91.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 15:15:00 | 94.25 | 94.36 | 93.52 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 09:15:00 | 96.24 | 94.36 | 93.52 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 94.29 | 94.58 | 94.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 93.51 | 94.58 | 94.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 93.45 | 94.35 | 94.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 93.45 | 94.35 | 94.06 | SL hit (close<ema400) qty=1.00 sl=94.06 alert=retest1 |

### Cycle 164 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 93.34 | 93.86 | 93.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 10:15:00 | 92.81 | 93.47 | 93.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 91.76 | 91.56 | 92.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 09:30:00 | 91.86 | 91.56 | 92.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 91.88 | 91.62 | 92.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:45:00 | 92.10 | 91.62 | 92.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 91.97 | 91.65 | 92.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:00:00 | 91.97 | 91.65 | 92.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 91.90 | 91.70 | 91.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 91.15 | 91.54 | 91.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 90.73 | 90.27 | 90.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 90.73 | 90.27 | 90.26 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 90.00 | 90.21 | 90.23 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 10:15:00 | 90.59 | 90.31 | 90.28 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 89.27 | 90.10 | 90.19 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 11:15:00 | 90.54 | 90.19 | 90.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 12:15:00 | 91.15 | 90.39 | 90.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 13:15:00 | 91.98 | 92.09 | 91.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 14:00:00 | 91.98 | 92.09 | 91.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 91.00 | 91.87 | 91.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 15:00:00 | 91.00 | 91.87 | 91.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 91.00 | 91.70 | 91.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 90.70 | 91.70 | 91.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 89.42 | 91.02 | 91.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 89.15 | 89.86 | 90.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 10:15:00 | 89.75 | 89.59 | 90.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 11:00:00 | 89.75 | 89.59 | 90.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 88.81 | 88.66 | 89.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:00:00 | 88.81 | 88.66 | 89.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 89.03 | 88.73 | 89.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 14:15:00 | 88.76 | 88.73 | 89.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 15:00:00 | 88.64 | 88.71 | 88.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 89.98 | 88.98 | 89.06 | SL hit (close>static) qty=1.00 sl=89.32 alert=retest2 |

### Cycle 171 — BUY (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 10:15:00 | 89.80 | 89.14 | 89.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 14:15:00 | 90.88 | 89.93 | 89.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 11:15:00 | 89.85 | 90.12 | 89.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 12:00:00 | 89.85 | 90.12 | 89.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 90.25 | 90.15 | 89.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:30:00 | 89.67 | 90.15 | 89.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 90.35 | 90.39 | 90.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 90.36 | 90.39 | 90.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 91.59 | 90.63 | 90.34 | EMA400 retest candle locked (from upside) |

### Cycle 172 — SELL (started 2025-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 12:15:00 | 89.55 | 90.35 | 90.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 13:15:00 | 89.42 | 90.17 | 90.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 15:15:00 | 89.64 | 89.63 | 89.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 09:15:00 | 89.62 | 89.63 | 89.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 89.95 | 89.70 | 89.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:00:00 | 89.95 | 89.70 | 89.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 90.04 | 89.76 | 89.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:15:00 | 89.62 | 89.76 | 89.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 87.95 | 87.22 | 87.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 87.95 | 87.22 | 87.18 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 10:15:00 | 86.90 | 87.14 | 87.17 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 14:15:00 | 87.41 | 87.20 | 87.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 87.87 | 87.37 | 87.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 87.35 | 87.37 | 87.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 11:00:00 | 87.35 | 87.37 | 87.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 86.95 | 87.28 | 87.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 86.95 | 87.28 | 87.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 87.04 | 87.23 | 87.23 | EMA400 retest candle locked (from upside) |

### Cycle 176 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 86.60 | 87.11 | 87.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 86.52 | 86.99 | 87.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 86.30 | 86.20 | 86.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 14:00:00 | 86.30 | 86.20 | 86.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 86.49 | 86.20 | 86.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 86.47 | 86.20 | 86.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 86.40 | 86.24 | 86.49 | EMA400 retest candle locked (from downside) |

### Cycle 177 — BUY (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 09:15:00 | 88.04 | 86.72 | 86.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 88.74 | 88.22 | 87.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 13:15:00 | 88.13 | 88.50 | 88.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 13:15:00 | 88.13 | 88.50 | 88.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 88.13 | 88.50 | 88.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 88.13 | 88.50 | 88.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 88.77 | 88.55 | 88.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 15:15:00 | 89.78 | 88.55 | 88.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 15:15:00 | 91.00 | 92.01 | 92.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — SELL (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 15:15:00 | 91.00 | 92.01 | 92.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 90.56 | 91.61 | 91.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 89.90 | 89.88 | 90.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 10:15:00 | 89.90 | 89.88 | 90.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 87.11 | 86.91 | 87.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 88.46 | 86.91 | 87.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 87.96 | 87.12 | 87.44 | EMA400 retest candle locked (from downside) |

### Cycle 179 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 89.28 | 87.92 | 87.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 15:15:00 | 89.95 | 88.32 | 87.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 87.86 | 88.85 | 88.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 87.86 | 88.85 | 88.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 87.86 | 88.85 | 88.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:00:00 | 87.86 | 88.85 | 88.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 88.02 | 88.69 | 88.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 88.04 | 88.69 | 88.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 88.02 | 88.43 | 88.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:30:00 | 87.81 | 88.43 | 88.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2025-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 13:15:00 | 88.09 | 88.36 | 88.38 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 14:15:00 | 88.55 | 88.40 | 88.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 88.71 | 88.48 | 88.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 88.48 | 88.48 | 88.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 11:15:00 | 88.48 | 88.48 | 88.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 88.48 | 88.48 | 88.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:30:00 | 88.54 | 88.48 | 88.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 88.63 | 88.51 | 88.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:45:00 | 88.54 | 88.51 | 88.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 88.63 | 88.54 | 88.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:30:00 | 88.66 | 88.54 | 88.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 88.31 | 88.49 | 88.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 88.31 | 88.49 | 88.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 88.40 | 88.47 | 88.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 88.13 | 88.47 | 88.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 87.82 | 88.34 | 88.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 87.22 | 88.12 | 88.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 87.92 | 87.70 | 87.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 87.92 | 87.70 | 87.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 87.92 | 87.70 | 87.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 87.92 | 87.70 | 87.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 89.26 | 88.01 | 88.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 89.26 | 88.01 | 88.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 90.13 | 88.44 | 88.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 13:15:00 | 90.42 | 89.11 | 88.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 14:15:00 | 89.98 | 90.05 | 89.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 15:00:00 | 89.98 | 90.05 | 89.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 89.10 | 89.88 | 89.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:00:00 | 89.10 | 89.88 | 89.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 88.78 | 89.66 | 89.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 88.78 | 89.66 | 89.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 89.42 | 89.48 | 89.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:30:00 | 89.47 | 89.48 | 89.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 89.85 | 89.56 | 89.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:30:00 | 89.30 | 89.56 | 89.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 89.42 | 89.62 | 89.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 89.42 | 89.62 | 89.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 88.53 | 89.40 | 89.42 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 89.75 | 89.27 | 89.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 15:15:00 | 90.17 | 89.55 | 89.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 11:15:00 | 90.25 | 90.49 | 90.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 11:15:00 | 90.25 | 90.49 | 90.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 90.25 | 90.49 | 90.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:00:00 | 90.25 | 90.49 | 90.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 90.12 | 90.41 | 90.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:45:00 | 90.16 | 90.41 | 90.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 90.44 | 90.42 | 90.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 89.85 | 90.42 | 90.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 90.19 | 90.37 | 90.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 90.19 | 90.37 | 90.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 90.07 | 90.31 | 90.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 90.30 | 90.31 | 90.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 91.97 | 92.50 | 92.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 91.97 | 92.50 | 92.51 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 92.75 | 92.51 | 92.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 93.43 | 92.69 | 92.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 13:15:00 | 94.68 | 94.75 | 94.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 14:15:00 | 94.54 | 94.75 | 94.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 93.89 | 94.58 | 94.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 93.89 | 94.58 | 94.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 93.83 | 94.43 | 94.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 94.00 | 94.43 | 94.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 93.67 | 94.28 | 94.06 | EMA400 retest candle locked (from upside) |

### Cycle 188 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 93.11 | 93.89 | 93.91 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 95.84 | 94.07 | 93.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 96.32 | 95.64 | 95.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 12:15:00 | 95.75 | 95.77 | 95.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 13:00:00 | 95.75 | 95.77 | 95.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 95.65 | 95.75 | 95.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:45:00 | 95.49 | 95.75 | 95.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 95.45 | 95.69 | 95.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:15:00 | 95.51 | 95.69 | 95.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 94.51 | 95.45 | 95.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 94.51 | 95.45 | 95.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 94.66 | 95.29 | 95.31 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 95.97 | 95.40 | 95.33 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 13:15:00 | 95.23 | 95.30 | 95.30 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 95.59 | 95.35 | 95.32 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 09:15:00 | 95.17 | 95.30 | 95.30 | EMA200 below EMA400 |

### Cycle 195 — BUY (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 10:15:00 | 95.44 | 95.33 | 95.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 97.04 | 95.88 | 95.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 14:15:00 | 103.26 | 103.36 | 102.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 14:45:00 | 102.95 | 103.36 | 102.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 104.88 | 103.61 | 102.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 11:00:00 | 105.40 | 103.97 | 102.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 13:00:00 | 105.08 | 104.37 | 103.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 13:45:00 | 105.00 | 104.43 | 103.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 09:30:00 | 105.19 | 103.98 | 103.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 104.09 | 104.44 | 104.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 104.09 | 104.44 | 104.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 104.57 | 104.47 | 104.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 106.04 | 104.47 | 104.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 14:15:00 | 106.70 | 107.51 | 107.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 106.70 | 107.51 | 107.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 106.61 | 107.12 | 107.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 13:15:00 | 103.03 | 102.91 | 104.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 13:45:00 | 103.00 | 102.91 | 104.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 100.10 | 98.68 | 99.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 100.02 | 98.68 | 99.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 101.04 | 99.15 | 99.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:00:00 | 101.04 | 99.15 | 99.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 101.74 | 99.67 | 99.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 104.00 | 100.92 | 100.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 104.45 | 104.56 | 103.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 104.45 | 104.56 | 103.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 103.36 | 104.22 | 103.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 103.31 | 104.22 | 103.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 103.01 | 103.98 | 103.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 103.01 | 103.98 | 103.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 102.89 | 103.76 | 103.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 102.89 | 103.76 | 103.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 103.57 | 103.72 | 103.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:30:00 | 103.07 | 103.72 | 103.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 103.21 | 103.62 | 103.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 103.57 | 103.62 | 103.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 102.89 | 103.47 | 103.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 102.89 | 103.47 | 103.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 102.49 | 103.28 | 103.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 102.49 | 103.28 | 103.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 101.92 | 103.01 | 103.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 101.74 | 102.75 | 102.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 101.03 | 100.71 | 101.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 101.03 | 100.71 | 101.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 101.03 | 100.71 | 101.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 101.52 | 100.71 | 101.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 101.37 | 100.84 | 101.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 100.78 | 100.84 | 101.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 14:00:00 | 100.95 | 100.88 | 101.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 101.58 | 101.02 | 101.27 | SL hit (close>static) qty=1.00 sl=101.53 alert=retest2 |

### Cycle 199 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 102.72 | 101.59 | 101.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 13:15:00 | 104.15 | 102.53 | 102.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 12:15:00 | 103.83 | 104.01 | 103.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 13:00:00 | 103.83 | 104.01 | 103.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 103.15 | 103.77 | 103.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 103.15 | 103.77 | 103.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 103.17 | 103.65 | 103.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 103.99 | 103.65 | 103.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 103.58 | 103.63 | 103.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 104.20 | 103.63 | 103.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 14:15:00 | 102.45 | 103.26 | 103.21 | SL hit (close<static) qty=1.00 sl=102.70 alert=retest2 |

### Cycle 200 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 102.40 | 103.09 | 103.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 101.46 | 102.76 | 102.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 11:15:00 | 102.15 | 102.07 | 102.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 11:30:00 | 102.17 | 102.07 | 102.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 102.77 | 102.22 | 102.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 102.77 | 102.22 | 102.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 103.04 | 102.39 | 102.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:45:00 | 103.50 | 102.39 | 102.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — BUY (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 15:15:00 | 102.94 | 102.50 | 102.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 103.37 | 102.67 | 102.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 12:15:00 | 105.41 | 105.42 | 104.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 13:00:00 | 105.41 | 105.42 | 104.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 105.03 | 105.65 | 105.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:45:00 | 105.10 | 105.65 | 105.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 105.88 | 105.69 | 105.31 | EMA400 retest candle locked (from upside) |

### Cycle 202 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 104.78 | 105.08 | 105.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 103.93 | 104.85 | 105.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 10:15:00 | 104.83 | 104.60 | 104.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 10:15:00 | 104.83 | 104.60 | 104.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 104.83 | 104.60 | 104.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:45:00 | 104.93 | 104.60 | 104.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 105.25 | 104.73 | 104.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:45:00 | 104.96 | 104.73 | 104.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 104.93 | 104.77 | 104.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:00:00 | 103.80 | 104.51 | 104.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 98.61 | 100.73 | 102.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 100.07 | 99.84 | 100.92 | SL hit (close>ema200) qty=0.50 sl=99.84 alert=retest2 |

### Cycle 203 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 101.40 | 100.17 | 100.04 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 100.00 | 100.25 | 100.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 98.91 | 99.98 | 100.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 95.69 | 94.99 | 96.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 95.69 | 94.99 | 96.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 95.69 | 94.99 | 96.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:30:00 | 94.32 | 94.75 | 95.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 10:15:00 | 89.60 | 92.02 | 93.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 14:15:00 | 92.49 | 91.51 | 92.81 | SL hit (close>ema200) qty=0.50 sl=91.51 alert=retest2 |

### Cycle 205 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 94.20 | 93.31 | 93.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 14:15:00 | 94.25 | 93.59 | 93.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 10:15:00 | 93.66 | 93.74 | 93.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 11:00:00 | 93.66 | 93.74 | 93.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 93.86 | 93.84 | 93.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 15:15:00 | 93.99 | 93.83 | 93.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 10:30:00 | 94.00 | 93.86 | 93.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 11:15:00 | 94.00 | 93.86 | 93.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 92.72 | 93.63 | 93.61 | SL hit (close<static) qty=1.00 sl=93.60 alert=retest2 |

### Cycle 206 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 91.89 | 93.28 | 93.45 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 93.96 | 93.07 | 93.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 95.59 | 93.72 | 93.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 97.07 | 97.25 | 96.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 11:00:00 | 97.07 | 97.25 | 96.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 96.62 | 97.26 | 96.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 95.98 | 97.26 | 96.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 96.99 | 97.21 | 96.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:30:00 | 96.46 | 97.21 | 96.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 96.34 | 97.02 | 96.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 96.34 | 97.02 | 96.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 96.62 | 96.94 | 96.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:30:00 | 96.89 | 97.07 | 96.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:15:00 | 96.80 | 97.42 | 97.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 96.84 | 97.31 | 97.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 96.84 | 97.31 | 97.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 95.92 | 96.51 | 96.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 99.25 | 95.54 | 95.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 99.25 | 95.54 | 95.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 99.25 | 95.54 | 95.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 99.25 | 95.54 | 95.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 99.70 | 96.38 | 96.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 13:15:00 | 101.25 | 98.35 | 97.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 13:15:00 | 99.76 | 99.96 | 98.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-17 14:00:00 | 99.76 | 99.96 | 98.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 100.83 | 100.11 | 99.16 | EMA400 retest candle locked (from upside) |

### Cycle 210 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 98.80 | 99.66 | 99.68 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 99.74 | 99.68 | 99.68 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 15:15:00 | 99.57 | 99.67 | 99.68 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 100.89 | 99.92 | 99.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 15:15:00 | 102.10 | 100.82 | 100.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 101.56 | 101.97 | 101.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 12:00:00 | 101.56 | 101.97 | 101.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 102.00 | 101.97 | 101.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 13:30:00 | 102.10 | 102.01 | 101.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 14:00:00 | 102.13 | 102.01 | 101.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 14:30:00 | 102.28 | 102.04 | 101.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 100.97 | 101.88 | 101.64 | SL hit (close<static) qty=1.00 sl=101.51 alert=retest2 |

### Cycle 214 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 100.41 | 101.47 | 101.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 15:15:00 | 99.75 | 101.12 | 101.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 13:15:00 | 95.87 | 95.79 | 97.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 14:00:00 | 95.87 | 95.79 | 97.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 96.77 | 96.13 | 96.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:30:00 | 96.87 | 96.13 | 96.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 96.66 | 96.23 | 96.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:30:00 | 96.83 | 96.23 | 96.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 97.34 | 96.45 | 96.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 97.34 | 96.45 | 96.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 98.25 | 96.81 | 97.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 98.79 | 96.81 | 97.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 215 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 97.46 | 97.19 | 97.16 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 12:15:00 | 96.73 | 97.10 | 97.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 14:15:00 | 94.95 | 96.60 | 96.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 94.16 | 92.97 | 94.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 94.16 | 92.97 | 94.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 94.16 | 92.97 | 94.26 | EMA400 retest candle locked (from downside) |

### Cycle 217 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 96.35 | 94.83 | 94.74 | EMA200 above EMA400 |

### Cycle 218 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 93.58 | 94.70 | 94.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 93.08 | 94.24 | 94.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 94.47 | 94.28 | 94.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 94.47 | 94.28 | 94.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 94.47 | 94.28 | 94.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 94.47 | 94.28 | 94.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 95.00 | 94.43 | 94.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 95.00 | 94.43 | 94.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 94.14 | 94.37 | 94.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:30:00 | 94.23 | 94.37 | 94.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 94.50 | 94.39 | 94.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 94.83 | 94.39 | 94.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 93.28 | 94.17 | 94.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 91.85 | 94.00 | 94.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 13:15:00 | 92.12 | 91.18 | 91.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 219 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 92.12 | 91.18 | 91.07 | EMA200 above EMA400 |

### Cycle 220 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 89.19 | 90.77 | 90.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 88.90 | 90.39 | 90.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 92.18 | 89.90 | 90.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 92.18 | 89.90 | 90.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 92.18 | 89.90 | 90.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 92.19 | 89.90 | 90.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 92.64 | 90.45 | 90.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 92.75 | 90.45 | 90.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 221 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 11:15:00 | 91.95 | 90.75 | 90.60 | EMA200 above EMA400 |

### Cycle 222 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 86.59 | 90.06 | 90.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 85.99 | 89.25 | 89.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 87.59 | 86.55 | 88.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 87.59 | 86.55 | 88.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 87.59 | 86.55 | 88.00 | EMA400 retest candle locked (from downside) |

### Cycle 223 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 91.23 | 88.86 | 88.61 | EMA200 above EMA400 |

### Cycle 224 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 88.23 | 89.12 | 89.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 86.31 | 88.54 | 88.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 88.97 | 86.66 | 87.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 88.97 | 86.66 | 87.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 88.97 | 86.66 | 87.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 88.97 | 86.66 | 87.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 88.12 | 86.96 | 87.52 | EMA400 retest candle locked (from downside) |

### Cycle 225 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 90.76 | 88.24 | 88.04 | EMA200 above EMA400 |

### Cycle 226 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 86.92 | 88.05 | 88.10 | EMA200 below EMA400 |

### Cycle 227 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 88.99 | 88.19 | 88.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 89.23 | 88.46 | 88.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 94.46 | 94.68 | 93.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 94.40 | 94.68 | 93.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 94.53 | 95.05 | 94.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 94.89 | 95.05 | 94.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 94.93 | 95.05 | 94.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 14:15:00 | 97.41 | 97.52 | 97.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — SELL (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 14:15:00 | 97.41 | 97.52 | 97.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 09:15:00 | 96.36 | 97.25 | 97.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 95.52 | 95.22 | 95.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 95.52 | 95.22 | 95.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 95.52 | 95.22 | 95.87 | EMA400 retest candle locked (from downside) |

### Cycle 229 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 97.05 | 96.09 | 96.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 14:15:00 | 97.79 | 97.00 | 96.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 09:15:00 | 96.87 | 97.14 | 96.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 96.87 | 97.14 | 96.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 96.87 | 97.14 | 96.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 96.58 | 97.14 | 96.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 97.49 | 97.21 | 96.83 | EMA400 retest candle locked (from upside) |

### Cycle 230 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 94.22 | 96.29 | 96.52 | EMA200 below EMA400 |

### Cycle 231 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 99.29 | 96.80 | 96.52 | EMA200 above EMA400 |

### Cycle 232 — SELL (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 11:15:00 | 96.31 | 96.98 | 97.06 | EMA200 below EMA400 |

### Cycle 233 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 98.99 | 97.46 | 97.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 99.75 | 97.91 | 97.48 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-15 15:15:00 | 45.85 | 2023-05-23 11:15:00 | 45.50 | STOP_HIT | 1.00 | 0.76% |
| SELL | retest2 | 2023-05-26 10:15:00 | 44.85 | 2023-05-29 09:15:00 | 42.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-05-26 10:15:00 | 44.85 | 2023-05-29 11:15:00 | 40.37 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-05-29 09:15:00 | 42.45 | 2023-05-29 11:15:00 | 40.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-05-29 09:15:00 | 42.45 | 2023-06-01 09:15:00 | 40.70 | STOP_HIT | 0.50 | 4.12% |
| BUY | retest2 | 2023-06-06 09:15:00 | 41.90 | 2023-06-13 13:15:00 | 42.10 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2023-06-06 15:15:00 | 41.75 | 2023-06-13 13:15:00 | 42.10 | STOP_HIT | 1.00 | 0.84% |
| BUY | retest2 | 2023-06-19 09:15:00 | 43.20 | 2023-06-19 10:15:00 | 42.95 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2023-06-19 10:00:00 | 43.25 | 2023-06-19 10:15:00 | 42.95 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2023-07-05 10:15:00 | 43.95 | 2023-07-13 13:15:00 | 44.10 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2023-07-06 09:15:00 | 44.20 | 2023-07-13 13:15:00 | 44.10 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2023-07-06 10:15:00 | 43.95 | 2023-07-13 13:15:00 | 44.10 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2023-07-06 10:45:00 | 44.20 | 2023-07-13 13:15:00 | 44.10 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2023-07-12 09:15:00 | 44.75 | 2023-07-13 13:15:00 | 44.10 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2023-08-08 09:15:00 | 52.25 | 2023-08-14 15:15:00 | 53.00 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest2 | 2023-08-08 11:45:00 | 51.75 | 2023-08-14 15:15:00 | 53.00 | STOP_HIT | 1.00 | 2.42% |
| BUY | retest2 | 2023-08-21 09:15:00 | 54.05 | 2023-08-23 09:15:00 | 59.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-05 14:15:00 | 63.35 | 2023-09-06 13:15:00 | 62.50 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2023-09-06 10:15:00 | 63.35 | 2023-09-06 13:15:00 | 62.50 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2023-09-06 11:00:00 | 63.35 | 2023-09-06 13:15:00 | 62.50 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2023-09-07 09:30:00 | 63.75 | 2023-09-12 09:15:00 | 61.45 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2023-09-08 09:15:00 | 63.40 | 2023-09-12 09:15:00 | 61.45 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2023-09-14 12:15:00 | 60.85 | 2023-09-22 09:15:00 | 57.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-14 13:30:00 | 60.80 | 2023-09-22 09:15:00 | 57.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-15 11:45:00 | 60.85 | 2023-09-22 09:15:00 | 57.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-18 09:45:00 | 60.70 | 2023-09-22 09:15:00 | 57.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-14 12:15:00 | 60.85 | 2023-09-22 12:15:00 | 58.80 | STOP_HIT | 0.50 | 3.37% |
| SELL | retest2 | 2023-09-14 13:30:00 | 60.80 | 2023-09-22 12:15:00 | 58.80 | STOP_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2023-09-15 11:45:00 | 60.85 | 2023-09-22 12:15:00 | 58.80 | STOP_HIT | 0.50 | 3.37% |
| SELL | retest2 | 2023-09-18 09:45:00 | 60.70 | 2023-09-22 12:15:00 | 58.80 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2023-09-21 12:00:00 | 59.60 | 2023-09-27 10:15:00 | 59.35 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2023-10-04 11:00:00 | 59.65 | 2023-10-04 11:15:00 | 58.55 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2023-10-17 13:30:00 | 58.70 | 2023-10-20 09:15:00 | 55.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-18 09:15:00 | 58.65 | 2023-10-20 09:15:00 | 55.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-17 13:30:00 | 58.70 | 2023-10-26 09:15:00 | 52.83 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-18 09:15:00 | 58.65 | 2023-10-26 09:15:00 | 52.78 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2023-11-01 09:15:00 | 54.80 | 2023-11-01 14:15:00 | 54.25 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest1 | 2023-11-06 09:30:00 | 56.40 | 2023-11-07 09:15:00 | 56.05 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2023-11-08 09:15:00 | 56.65 | 2023-11-20 13:15:00 | 57.60 | STOP_HIT | 1.00 | 1.68% |
| BUY | retest2 | 2023-11-09 10:45:00 | 56.40 | 2023-11-20 13:15:00 | 57.60 | STOP_HIT | 1.00 | 2.13% |
| BUY | retest2 | 2023-11-10 10:00:00 | 56.45 | 2023-11-20 13:15:00 | 57.60 | STOP_HIT | 1.00 | 2.04% |
| SELL | retest2 | 2023-11-23 14:30:00 | 57.15 | 2023-11-24 11:15:00 | 58.05 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2023-11-24 10:15:00 | 57.20 | 2023-11-24 11:15:00 | 58.05 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2023-12-01 09:30:00 | 61.15 | 2023-12-08 10:15:00 | 67.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-01 10:30:00 | 61.10 | 2023-12-08 10:15:00 | 67.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-04 09:15:00 | 61.90 | 2023-12-08 10:15:00 | 68.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-08 09:30:00 | 63.20 | 2023-12-08 10:15:00 | 69.52 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-12-27 11:30:00 | 74.10 | 2023-12-28 11:15:00 | 75.50 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-01-02 12:30:00 | 78.80 | 2024-01-08 09:15:00 | 86.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-02 13:30:00 | 78.85 | 2024-01-08 09:15:00 | 86.73 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-01-16 12:45:00 | 84.20 | 2024-01-18 09:15:00 | 80.42 | PARTIAL | 0.50 | 4.49% |
| SELL | retest2 | 2024-01-16 12:45:00 | 84.20 | 2024-01-18 11:15:00 | 83.55 | STOP_HIT | 0.50 | 0.77% |
| SELL | retest2 | 2024-01-17 10:30:00 | 84.65 | 2024-01-23 10:15:00 | 79.99 | PARTIAL | 0.50 | 5.51% |
| SELL | retest2 | 2024-01-17 11:30:00 | 84.35 | 2024-01-23 10:15:00 | 80.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-17 10:30:00 | 84.65 | 2024-01-23 14:15:00 | 75.78 | TARGET_HIT | 0.50 | 10.48% |
| SELL | retest2 | 2024-01-17 11:30:00 | 84.35 | 2024-01-23 14:15:00 | 75.91 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2024-02-06 10:30:00 | 88.00 | 2024-02-08 09:15:00 | 92.40 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-02-06 10:30:00 | 88.00 | 2024-02-08 14:15:00 | 90.20 | STOP_HIT | 0.50 | 2.50% |
| BUY | retest1 | 2024-02-06 14:30:00 | 88.45 | 2024-02-09 09:15:00 | 87.85 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-02-28 12:45:00 | 84.70 | 2024-03-02 09:15:00 | 85.95 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-02-28 14:15:00 | 84.95 | 2024-03-02 09:15:00 | 85.95 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-03-01 10:00:00 | 85.15 | 2024-03-02 09:15:00 | 85.95 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-03-01 15:00:00 | 85.10 | 2024-03-02 09:15:00 | 85.95 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-03-07 15:15:00 | 84.25 | 2024-03-13 09:15:00 | 80.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 15:15:00 | 84.25 | 2024-03-13 12:15:00 | 75.83 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-03-26 14:15:00 | 79.65 | 2024-04-09 15:15:00 | 84.80 | STOP_HIT | 1.00 | 6.47% |
| BUY | retest2 | 2024-03-27 09:15:00 | 80.55 | 2024-04-09 15:15:00 | 84.80 | STOP_HIT | 1.00 | 5.28% |
| SELL | retest2 | 2024-04-18 14:30:00 | 80.60 | 2024-04-23 10:15:00 | 82.15 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-04-19 14:00:00 | 80.75 | 2024-04-23 10:15:00 | 82.15 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-04-22 11:00:00 | 81.00 | 2024-04-23 10:15:00 | 82.15 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-04-22 11:30:00 | 80.85 | 2024-04-23 10:15:00 | 82.15 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-04-22 14:15:00 | 80.75 | 2024-04-23 10:15:00 | 82.15 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-04-25 13:00:00 | 84.80 | 2024-04-30 12:15:00 | 85.75 | STOP_HIT | 1.00 | 1.12% |
| BUY | retest2 | 2024-04-25 13:30:00 | 84.90 | 2024-04-30 12:15:00 | 85.75 | STOP_HIT | 1.00 | 1.00% |
| SELL | retest2 | 2024-05-08 14:15:00 | 80.60 | 2024-05-13 13:15:00 | 81.85 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-05-09 09:30:00 | 80.70 | 2024-05-13 13:15:00 | 81.85 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-05-09 12:45:00 | 79.95 | 2024-05-13 13:15:00 | 81.85 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2024-05-17 09:15:00 | 83.65 | 2024-05-29 10:15:00 | 86.60 | STOP_HIT | 1.00 | 3.53% |
| BUY | retest2 | 2024-05-17 10:15:00 | 83.65 | 2024-05-29 10:15:00 | 86.60 | STOP_HIT | 1.00 | 3.53% |
| BUY | retest1 | 2024-06-11 09:15:00 | 90.18 | 2024-06-18 09:15:00 | 94.69 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-06-11 09:15:00 | 90.18 | 2024-06-19 10:15:00 | 96.14 | STOP_HIT | 0.50 | 6.61% |
| BUY | retest2 | 2024-06-13 13:00:00 | 92.30 | 2024-06-25 10:15:00 | 96.00 | STOP_HIT | 1.00 | 4.01% |
| BUY | retest2 | 2024-06-13 15:00:00 | 91.85 | 2024-06-25 10:15:00 | 96.00 | STOP_HIT | 1.00 | 4.52% |
| BUY | retest2 | 2024-06-14 09:15:00 | 91.85 | 2024-06-25 10:15:00 | 96.00 | STOP_HIT | 1.00 | 4.52% |
| SELL | retest2 | 2024-07-02 09:30:00 | 96.89 | 2024-07-02 10:15:00 | 97.50 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-07-02 11:30:00 | 97.01 | 2024-07-04 09:15:00 | 97.54 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-07-23 12:15:00 | 91.12 | 2024-07-24 09:15:00 | 95.32 | STOP_HIT | 1.00 | -4.61% |
| BUY | retest2 | 2024-07-30 09:15:00 | 97.75 | 2024-08-02 13:15:00 | 99.25 | STOP_HIT | 1.00 | 1.53% |
| SELL | retest2 | 2024-08-22 12:45:00 | 95.29 | 2024-08-23 09:15:00 | 97.27 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-09-02 09:45:00 | 93.38 | 2024-09-04 09:15:00 | 94.62 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-09-03 10:15:00 | 93.39 | 2024-09-04 09:15:00 | 94.62 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-09-03 13:45:00 | 93.65 | 2024-09-04 09:15:00 | 94.62 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-09-04 09:15:00 | 93.70 | 2024-09-04 09:15:00 | 94.62 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-09-18 09:15:00 | 96.09 | 2024-09-18 10:15:00 | 95.20 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest1 | 2024-10-23 09:30:00 | 79.75 | 2024-10-24 09:15:00 | 83.16 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2024-10-25 09:15:00 | 80.73 | 2024-10-28 09:15:00 | 76.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-25 09:15:00 | 80.73 | 2024-10-29 12:15:00 | 78.60 | STOP_HIT | 0.50 | 2.64% |
| BUY | retest2 | 2024-12-04 11:15:00 | 84.07 | 2024-12-12 10:15:00 | 84.62 | STOP_HIT | 1.00 | 0.65% |
| BUY | retest2 | 2024-12-05 10:30:00 | 83.75 | 2024-12-12 10:15:00 | 84.62 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2024-12-05 12:00:00 | 83.80 | 2024-12-12 10:15:00 | 84.62 | STOP_HIT | 1.00 | 0.98% |
| BUY | retest2 | 2024-12-05 12:30:00 | 83.75 | 2024-12-12 10:15:00 | 84.62 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2024-12-16 11:45:00 | 83.79 | 2024-12-17 09:15:00 | 84.59 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-12-16 13:45:00 | 83.84 | 2024-12-17 09:15:00 | 84.59 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-12-16 14:15:00 | 83.80 | 2024-12-17 09:15:00 | 84.59 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-12-17 14:00:00 | 83.61 | 2024-12-20 13:15:00 | 79.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 14:00:00 | 83.61 | 2024-12-23 09:15:00 | 80.32 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2024-12-30 12:00:00 | 78.57 | 2025-01-01 15:15:00 | 78.70 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-01-01 11:15:00 | 78.55 | 2025-01-01 15:15:00 | 78.70 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-01-09 10:45:00 | 76.40 | 2025-01-13 09:15:00 | 72.71 | PARTIAL | 0.50 | 4.83% |
| SELL | retest2 | 2025-01-09 12:15:00 | 76.54 | 2025-01-13 09:15:00 | 72.72 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2025-01-09 14:15:00 | 76.55 | 2025-01-13 10:15:00 | 72.58 | PARTIAL | 0.50 | 5.19% |
| SELL | retest2 | 2025-01-09 10:45:00 | 76.40 | 2025-01-14 10:15:00 | 72.24 | STOP_HIT | 0.50 | 5.45% |
| SELL | retest2 | 2025-01-09 12:15:00 | 76.54 | 2025-01-14 10:15:00 | 72.24 | STOP_HIT | 0.50 | 5.62% |
| SELL | retest2 | 2025-01-09 14:15:00 | 76.55 | 2025-01-14 10:15:00 | 72.24 | STOP_HIT | 0.50 | 5.63% |
| SELL | retest2 | 2025-01-24 13:30:00 | 72.20 | 2025-01-29 10:15:00 | 72.97 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-02-03 11:00:00 | 73.64 | 2025-02-07 13:15:00 | 73.94 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2025-02-03 15:00:00 | 73.59 | 2025-02-07 13:15:00 | 73.94 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-02-04 09:15:00 | 74.32 | 2025-02-07 13:15:00 | 73.94 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-02-04 11:45:00 | 74.06 | 2025-02-07 13:15:00 | 73.94 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-02-18 10:15:00 | 69.43 | 2025-02-20 11:15:00 | 70.70 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-02-18 11:15:00 | 69.48 | 2025-02-20 11:15:00 | 70.70 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-02-18 11:45:00 | 69.15 | 2025-02-20 11:15:00 | 70.70 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-02-27 12:45:00 | 69.37 | 2025-03-03 13:15:00 | 69.78 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-02-28 15:15:00 | 69.35 | 2025-03-03 13:15:00 | 69.78 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-03-10 09:30:00 | 73.60 | 2025-03-10 13:15:00 | 72.60 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-03-10 10:15:00 | 73.44 | 2025-03-10 13:15:00 | 72.60 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-03-12 13:15:00 | 73.42 | 2025-03-25 12:15:00 | 76.38 | STOP_HIT | 1.00 | 4.03% |
| BUY | retest2 | 2025-03-12 14:00:00 | 73.37 | 2025-03-25 12:15:00 | 76.38 | STOP_HIT | 1.00 | 4.10% |
| BUY | retest2 | 2025-03-13 09:30:00 | 73.20 | 2025-03-25 12:15:00 | 76.38 | STOP_HIT | 1.00 | 4.34% |
| SELL | retest2 | 2025-03-27 12:30:00 | 75.39 | 2025-03-28 10:15:00 | 76.83 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-04-07 11:00:00 | 81.05 | 2025-04-23 12:15:00 | 86.31 | STOP_HIT | 1.00 | 6.49% |
| SELL | retest2 | 2025-05-08 13:30:00 | 86.07 | 2025-05-09 09:15:00 | 81.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 13:30:00 | 86.07 | 2025-05-09 15:15:00 | 84.30 | STOP_HIT | 0.50 | 2.06% |
| SELL | retest2 | 2025-05-29 10:00:00 | 85.85 | 2025-06-06 09:15:00 | 85.50 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-05-29 11:00:00 | 85.95 | 2025-06-06 09:15:00 | 85.50 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2025-05-29 12:00:00 | 86.05 | 2025-06-06 09:15:00 | 85.50 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2025-05-29 12:30:00 | 85.96 | 2025-06-06 09:15:00 | 85.50 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2025-05-30 09:15:00 | 85.76 | 2025-06-06 09:15:00 | 85.50 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2025-07-10 14:15:00 | 92.39 | 2025-07-11 09:15:00 | 91.45 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-07-14 15:15:00 | 91.42 | 2025-07-15 14:15:00 | 91.91 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-07-15 09:30:00 | 91.13 | 2025-07-15 14:15:00 | 91.91 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest1 | 2025-07-18 09:15:00 | 96.24 | 2025-07-21 09:15:00 | 93.45 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-07-25 09:30:00 | 91.15 | 2025-07-31 12:15:00 | 90.73 | STOP_HIT | 1.00 | 0.46% |
| SELL | retest2 | 2025-08-13 14:15:00 | 88.76 | 2025-08-14 09:15:00 | 89.98 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-08-13 15:00:00 | 88.64 | 2025-08-14 09:15:00 | 89.98 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-08-25 11:15:00 | 89.62 | 2025-09-02 11:15:00 | 87.95 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2025-09-15 15:15:00 | 89.78 | 2025-09-22 15:15:00 | 91.00 | STOP_HIT | 1.00 | 1.36% |
| BUY | retest2 | 2025-10-20 09:15:00 | 90.30 | 2025-10-29 11:15:00 | 91.97 | STOP_HIT | 1.00 | 1.85% |
| BUY | retest2 | 2025-11-21 11:00:00 | 105.40 | 2025-12-02 14:15:00 | 106.70 | STOP_HIT | 1.00 | 1.23% |
| BUY | retest2 | 2025-11-21 13:00:00 | 105.08 | 2025-12-02 14:15:00 | 106.70 | STOP_HIT | 1.00 | 1.54% |
| BUY | retest2 | 2025-11-21 13:45:00 | 105.00 | 2025-12-02 14:15:00 | 106.70 | STOP_HIT | 1.00 | 1.62% |
| BUY | retest2 | 2025-11-25 09:30:00 | 105.19 | 2025-12-02 14:15:00 | 106.70 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest2 | 2025-11-26 09:15:00 | 106.04 | 2025-12-02 14:15:00 | 106.70 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2025-12-19 11:15:00 | 100.78 | 2025-12-19 14:15:00 | 101.58 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-12-19 14:00:00 | 100.95 | 2025-12-19 14:15:00 | 101.58 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-12-26 10:15:00 | 104.20 | 2025-12-26 14:15:00 | 102.45 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-01-08 10:00:00 | 103.80 | 2026-01-12 09:15:00 | 98.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:00:00 | 103.80 | 2026-01-12 15:15:00 | 100.07 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2026-01-23 11:30:00 | 94.32 | 2026-01-27 10:15:00 | 89.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 11:30:00 | 94.32 | 2026-01-27 14:15:00 | 92.49 | STOP_HIT | 0.50 | 1.94% |
| BUY | retest2 | 2026-01-30 15:15:00 | 93.99 | 2026-02-01 11:15:00 | 92.72 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-02-01 10:30:00 | 94.00 | 2026-02-01 11:15:00 | 92.72 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-02-01 11:15:00 | 94.00 | 2026-02-01 11:15:00 | 92.72 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-02-06 14:30:00 | 96.89 | 2026-02-11 10:15:00 | 96.84 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2026-02-11 10:15:00 | 96.80 | 2026-02-11 10:15:00 | 96.84 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2026-02-26 13:30:00 | 102.10 | 2026-02-27 09:15:00 | 100.97 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-02-26 14:00:00 | 102.13 | 2026-02-27 09:15:00 | 100.97 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-02-26 14:30:00 | 102.28 | 2026-02-27 09:15:00 | 100.97 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2026-02-27 13:45:00 | 102.13 | 2026-02-27 14:15:00 | 100.41 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-03-13 09:15:00 | 91.85 | 2026-03-18 13:15:00 | 92.12 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2026-04-13 10:15:00 | 94.89 | 2026-04-22 14:15:00 | 97.41 | STOP_HIT | 1.00 | 2.66% |
| BUY | retest2 | 2026-04-13 10:45:00 | 94.93 | 2026-04-22 14:15:00 | 97.41 | STOP_HIT | 1.00 | 2.61% |
