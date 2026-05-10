# Tata Teleservices (Maharashtra) Ltd. (TTML)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 44.09
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 69 |
| ALERT1 | 42 |
| ALERT2 | 40 |
| ALERT2_SKIP | 27 |
| ALERT3 | 104 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 65 |
| PARTIAL | 20 |
| TARGET_HIT | 7 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 86 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 49 / 37
- **Target hits / Stop hits / Partials:** 7 / 59 / 20
- **Avg / median % per leg:** 1.66% / 2.15%
- **Sum % (uncompounded):** 143.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 1 | 3 | 0 | 0.23% | 0.9% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -5.71% | -5.7% |
| BUY @ 3rd Alert (retest2) | 3 | 1 | 33.3% | 1 | 2 | 0 | 2.22% | 6.6% |
| SELL (all) | 82 | 48 | 58.5% | 6 | 56 | 20 | 1.73% | 142.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 82 | 48 | 58.5% | 6 | 56 | 20 | 1.73% | 142.2% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -5.71% | -5.7% |
| retest2 (combined) | 85 | 49 | 57.6% | 7 | 58 | 20 | 1.75% | 148.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 57.00 | 55.46 | 55.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 57.27 | 56.10 | 55.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 60.40 | 60.62 | 59.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 60.40 | 60.62 | 59.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 59.22 | 60.29 | 59.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 59.43 | 60.29 | 59.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 59.25 | 60.08 | 59.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:15:00 | 59.03 | 60.08 | 59.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 58.78 | 59.67 | 59.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 58.53 | 59.44 | 59.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 10:15:00 | 59.59 | 59.10 | 59.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 10:15:00 | 59.59 | 59.10 | 59.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 59.59 | 59.10 | 59.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:45:00 | 59.36 | 59.10 | 59.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 60.91 | 59.46 | 59.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:30:00 | 61.26 | 59.46 | 59.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 12:15:00 | 61.29 | 59.82 | 59.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 13:15:00 | 65.02 | 60.86 | 60.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 15:15:00 | 76.15 | 76.24 | 72.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 09:15:00 | 76.80 | 76.24 | 72.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 74.70 | 75.20 | 74.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:30:00 | 75.31 | 75.19 | 74.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 13:15:00 | 74.30 | 74.83 | 74.61 | SL hit (close<static) qty=1.00 sl=74.54 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 73.26 | 74.35 | 74.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 15:15:00 | 72.69 | 73.55 | 73.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 12:15:00 | 73.92 | 73.14 | 73.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 12:15:00 | 73.92 | 73.14 | 73.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 73.92 | 73.14 | 73.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:30:00 | 75.37 | 73.14 | 73.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 74.09 | 73.33 | 73.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:15:00 | 74.38 | 73.33 | 73.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 72.63 | 73.23 | 73.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 13:15:00 | 72.40 | 72.91 | 73.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 15:15:00 | 72.16 | 72.84 | 73.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 11:15:00 | 72.09 | 72.64 | 72.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 13:45:00 | 72.42 | 72.63 | 72.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 73.53 | 72.77 | 72.87 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-05 12:15:00 | 73.09 | 72.95 | 72.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-05 12:15:00 | 73.09 | 72.95 | 72.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-05 12:15:00 | 73.09 | 72.95 | 72.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-05 12:15:00 | 73.09 | 72.95 | 72.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 73.09 | 72.95 | 72.94 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 72.50 | 72.86 | 72.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 15:15:00 | 72.20 | 72.69 | 72.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 71.65 | 71.64 | 72.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 71.65 | 71.64 | 72.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 71.65 | 71.64 | 72.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:00:00 | 71.65 | 71.64 | 72.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 11:15:00 | 71.61 | 71.61 | 71.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 11:45:00 | 71.56 | 71.61 | 71.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 71.70 | 71.62 | 71.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:30:00 | 71.88 | 71.62 | 71.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 72.00 | 71.70 | 71.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:15:00 | 71.94 | 71.70 | 71.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 71.69 | 71.70 | 71.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 11:15:00 | 71.50 | 71.67 | 71.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 12:00:00 | 71.45 | 71.63 | 71.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 12:30:00 | 71.51 | 71.59 | 71.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 15:00:00 | 71.50 | 71.55 | 71.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 80.12 | 73.24 | 72.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 80.12 | 73.24 | 72.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 80.12 | 73.24 | 72.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 80.12 | 73.24 | 72.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 09:15:00 | 80.12 | 73.24 | 72.47 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 11:15:00 | 73.52 | 75.28 | 75.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 73.32 | 74.89 | 75.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 71.67 | 71.01 | 72.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:45:00 | 71.69 | 71.01 | 72.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 70.50 | 71.04 | 72.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:00:00 | 69.80 | 70.71 | 71.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-18 09:15:00 | 66.31 | 69.22 | 70.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-06-20 12:15:00 | 62.82 | 64.72 | 66.13 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 66.80 | 65.12 | 64.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 67.90 | 66.58 | 65.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 09:15:00 | 66.99 | 67.22 | 66.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 66.99 | 67.22 | 66.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 66.99 | 67.22 | 66.64 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 10:15:00 | 65.50 | 66.58 | 66.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 10:15:00 | 64.85 | 65.19 | 65.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 09:15:00 | 64.99 | 64.92 | 65.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 14:15:00 | 65.25 | 64.89 | 65.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 65.25 | 64.89 | 65.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 15:00:00 | 65.25 | 64.89 | 65.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 65.30 | 64.97 | 65.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 64.90 | 64.97 | 65.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 09:45:00 | 64.90 | 64.37 | 64.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 65.00 | 63.27 | 63.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 65.00 | 63.27 | 63.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 09:15:00 | 65.00 | 63.27 | 63.25 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 63.19 | 63.53 | 63.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 62.91 | 63.32 | 63.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 14:15:00 | 60.55 | 60.46 | 61.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 15:00:00 | 60.55 | 60.46 | 61.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 60.15 | 60.46 | 60.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 11:45:00 | 59.60 | 60.32 | 60.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 14:15:00 | 59.63 | 60.16 | 60.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 15:00:00 | 59.65 | 60.06 | 60.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 59.20 | 60.00 | 60.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 59.20 | 59.65 | 60.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 15:15:00 | 58.86 | 59.65 | 60.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 12:15:00 | 59.01 | 59.39 | 59.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 13:00:00 | 59.00 | 59.31 | 59.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 14:15:00 | 58.80 | 59.28 | 59.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 58.87 | 59.20 | 59.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 58.87 | 59.20 | 59.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 57.99 | 58.89 | 59.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:45:00 | 57.06 | 57.76 | 58.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 11:15:00 | 56.62 | 57.53 | 58.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 11:15:00 | 56.65 | 57.53 | 58.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 11:15:00 | 56.67 | 57.53 | 58.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 14:00:00 | 57.10 | 57.37 | 57.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 14:45:00 | 57.30 | 57.33 | 57.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:15:00 | 56.24 | 57.15 | 57.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 13:15:00 | 55.92 | 56.75 | 57.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 13:15:00 | 56.06 | 56.75 | 57.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 13:15:00 | 56.05 | 56.75 | 57.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 57.00 | 56.80 | 57.27 | SL hit (close>ema200) qty=0.50 sl=56.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 57.00 | 56.80 | 57.27 | SL hit (close>ema200) qty=0.50 sl=56.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 57.00 | 56.80 | 57.27 | SL hit (close>ema200) qty=0.50 sl=56.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 57.00 | 56.80 | 57.27 | SL hit (close>ema200) qty=0.50 sl=56.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 57.00 | 56.80 | 57.27 | SL hit (close>ema200) qty=0.50 sl=56.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 57.00 | 56.80 | 57.27 | SL hit (close>ema200) qty=0.50 sl=56.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 57.00 | 56.80 | 57.27 | SL hit (close>ema200) qty=0.50 sl=56.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 11:15:00 | 57.29 | 57.38 | 57.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 11:15:00 | 57.95 | 57.50 | 57.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-08 11:15:00 | 57.95 | 57.50 | 57.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-08 11:15:00 | 57.95 | 57.50 | 57.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-08 11:15:00 | 57.95 | 57.50 | 57.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-08 11:15:00 | 57.95 | 57.50 | 57.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 11:15:00 | 57.95 | 57.50 | 57.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 58.47 | 57.82 | 57.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 58.50 | 58.76 | 58.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 11:15:00 | 58.50 | 58.76 | 58.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 58.50 | 58.76 | 58.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:30:00 | 58.50 | 58.76 | 58.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 58.60 | 58.73 | 58.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:30:00 | 58.58 | 58.73 | 58.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 57.95 | 58.62 | 58.48 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 57.97 | 58.39 | 58.39 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 58.97 | 58.38 | 58.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 11:15:00 | 59.25 | 58.88 | 58.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 58.90 | 59.09 | 58.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 58.90 | 59.09 | 58.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 58.90 | 59.09 | 58.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 58.76 | 59.09 | 58.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 59.65 | 59.21 | 58.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:45:00 | 59.00 | 59.21 | 58.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 59.10 | 59.20 | 58.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:45:00 | 59.00 | 59.20 | 58.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 59.30 | 59.22 | 59.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:45:00 | 59.70 | 59.29 | 59.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 14:15:00 | 58.50 | 59.06 | 59.05 | SL hit (close<static) qty=1.00 sl=58.81 alert=retest2 |

### Cycle 16 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 58.70 | 58.99 | 59.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 58.42 | 58.87 | 58.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 13:15:00 | 59.22 | 58.69 | 58.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 13:15:00 | 59.22 | 58.69 | 58.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 59.22 | 58.69 | 58.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:45:00 | 58.91 | 58.69 | 58.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 59.08 | 58.77 | 58.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 58.70 | 58.82 | 58.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 14:15:00 | 55.77 | 56.20 | 56.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 57.15 | 56.38 | 56.55 | SL hit (close>ema200) qty=0.50 sl=56.38 alert=retest2 |

### Cycle 17 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 58.75 | 56.86 | 56.75 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 57.30 | 57.68 | 57.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 56.66 | 57.23 | 57.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 14:15:00 | 57.25 | 57.12 | 57.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 15:00:00 | 57.25 | 57.12 | 57.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 57.86 | 57.27 | 57.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 58.05 | 57.29 | 57.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 57.10 | 57.25 | 57.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:45:00 | 57.20 | 57.25 | 57.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 57.00 | 57.20 | 57.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:30:00 | 57.18 | 57.20 | 57.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 57.03 | 57.17 | 57.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:45:00 | 57.03 | 57.17 | 57.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 57.45 | 57.22 | 57.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:30:00 | 57.63 | 57.22 | 57.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 57.24 | 57.23 | 57.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:30:00 | 57.50 | 57.23 | 57.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 57.05 | 57.19 | 57.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 56.90 | 57.19 | 57.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 56.70 | 57.09 | 57.22 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 57.45 | 57.18 | 57.17 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 10:15:00 | 57.00 | 57.18 | 57.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 12:15:00 | 56.96 | 57.12 | 57.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 15:15:00 | 57.39 | 57.14 | 57.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 15:15:00 | 57.39 | 57.14 | 57.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 57.39 | 57.14 | 57.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 57.94 | 57.14 | 57.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 57.88 | 57.29 | 57.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 59.99 | 58.15 | 57.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 58.50 | 58.61 | 58.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 09:15:00 | 58.50 | 58.61 | 58.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 58.50 | 58.61 | 58.17 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 14:15:00 | 57.76 | 58.15 | 58.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 10:15:00 | 57.40 | 57.80 | 57.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 57.74 | 57.19 | 57.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 57.74 | 57.19 | 57.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 57.74 | 57.19 | 57.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 57.74 | 57.19 | 57.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 57.01 | 57.15 | 57.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:00:00 | 56.93 | 57.09 | 57.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:45:00 | 56.66 | 56.25 | 56.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 13:15:00 | 55.44 | 55.31 | 55.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 13:15:00 | 55.44 | 55.31 | 55.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 13:15:00 | 55.44 | 55.31 | 55.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 56.15 | 55.48 | 55.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 56.93 | 56.97 | 56.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 10:15:00 | 56.36 | 56.85 | 56.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 56.36 | 56.85 | 56.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 56.36 | 56.85 | 56.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 56.45 | 56.77 | 56.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:00:00 | 56.45 | 56.77 | 56.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 59.10 | 57.23 | 56.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:30:00 | 56.70 | 57.23 | 56.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 57.50 | 57.83 | 57.32 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 10:15:00 | 56.24 | 57.14 | 57.19 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 59.10 | 57.27 | 57.14 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 56.80 | 57.28 | 57.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 56.34 | 56.95 | 57.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 56.16 | 56.11 | 56.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 11:00:00 | 56.16 | 56.11 | 56.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 56.23 | 56.15 | 56.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 14:15:00 | 56.04 | 56.15 | 56.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:00:00 | 56.00 | 56.17 | 56.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 14:45:00 | 56.03 | 56.18 | 56.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 15:15:00 | 56.00 | 56.18 | 56.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 56.00 | 56.14 | 56.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:00:00 | 55.67 | 56.05 | 56.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 55.26 | 55.70 | 55.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 12:45:00 | 55.39 | 55.57 | 55.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 09:45:00 | 54.67 | 55.61 | 55.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 54.88 | 55.15 | 55.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-27 13:15:00 | 55.40 | 55.21 | 55.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 13:15:00 | 55.40 | 55.21 | 55.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 13:15:00 | 55.40 | 55.21 | 55.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 13:15:00 | 55.40 | 55.21 | 55.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 13:15:00 | 55.40 | 55.21 | 55.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 13:15:00 | 55.40 | 55.21 | 55.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 13:15:00 | 55.40 | 55.21 | 55.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 13:15:00 | 55.40 | 55.21 | 55.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 55.40 | 55.21 | 55.20 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 55.14 | 55.23 | 55.23 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 55.44 | 55.27 | 55.25 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 54.81 | 55.23 | 55.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 10:15:00 | 54.65 | 54.88 | 55.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 54.90 | 54.75 | 54.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 54.90 | 54.75 | 54.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 54.90 | 54.75 | 54.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 54.96 | 54.75 | 54.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 54.78 | 54.76 | 54.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 11:15:00 | 54.65 | 54.76 | 54.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 12:15:00 | 54.71 | 54.77 | 54.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 13:15:00 | 54.65 | 54.76 | 54.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 14:15:00 | 55.95 | 54.97 | 54.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 14:15:00 | 55.95 | 54.97 | 54.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 14:15:00 | 55.95 | 54.97 | 54.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 55.95 | 54.97 | 54.93 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 54.69 | 55.15 | 55.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 53.99 | 54.61 | 54.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 15:15:00 | 54.62 | 54.45 | 54.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 09:15:00 | 54.33 | 54.45 | 54.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 54.20 | 54.40 | 54.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 11:15:00 | 54.15 | 54.36 | 54.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 10:15:00 | 53.89 | 54.16 | 54.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 54.92 | 54.22 | 54.26 | SL hit (close>static) qty=1.00 sl=54.77 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 54.92 | 54.22 | 54.26 | SL hit (close>static) qty=1.00 sl=54.77 alert=retest2 |

### Cycle 33 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 55.12 | 54.40 | 54.34 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 54.23 | 54.37 | 54.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 54.02 | 54.23 | 54.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 10:15:00 | 54.39 | 54.21 | 54.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 10:15:00 | 54.39 | 54.21 | 54.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 54.39 | 54.21 | 54.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:00:00 | 54.39 | 54.21 | 54.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 54.13 | 54.19 | 54.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 14:30:00 | 53.99 | 54.12 | 54.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:30:00 | 53.92 | 54.08 | 54.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 12:30:00 | 53.84 | 54.04 | 54.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 09:15:00 | 51.29 | 51.58 | 52.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 09:15:00 | 51.22 | 51.58 | 52.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 09:15:00 | 51.15 | 51.58 | 52.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 51.20 | 51.18 | 51.58 | SL hit (close>ema200) qty=0.50 sl=51.18 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 51.20 | 51.18 | 51.58 | SL hit (close>ema200) qty=0.50 sl=51.18 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 51.20 | 51.18 | 51.58 | SL hit (close>ema200) qty=0.50 sl=51.18 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 53.45 | 51.39 | 51.20 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 51.56 | 51.69 | 51.70 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 14:15:00 | 52.49 | 51.83 | 51.76 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 51.11 | 51.69 | 51.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 50.93 | 51.54 | 51.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 09:15:00 | 47.52 | 46.77 | 48.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 10:00:00 | 47.52 | 46.77 | 48.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 48.22 | 47.06 | 48.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:30:00 | 48.67 | 47.06 | 48.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 51.48 | 47.95 | 48.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 51.48 | 47.95 | 48.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 50.98 | 48.55 | 48.60 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 53.11 | 49.46 | 49.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 14:15:00 | 54.08 | 50.39 | 49.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 50.69 | 51.17 | 50.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 15:15:00 | 50.69 | 51.17 | 50.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 50.69 | 51.17 | 50.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 49.62 | 51.17 | 50.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 50.01 | 50.94 | 50.46 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 09:15:00 | 49.38 | 50.33 | 50.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 10:15:00 | 49.13 | 50.09 | 50.23 | Break + close below crossover candle low |

### Cycle 41 — BUY (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 09:15:00 | 52.64 | 49.67 | 49.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 10:15:00 | 53.70 | 50.47 | 49.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 14:15:00 | 51.29 | 52.07 | 51.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 15:00:00 | 51.29 | 52.07 | 51.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 50.75 | 51.81 | 51.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 50.75 | 51.81 | 51.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 50.55 | 51.56 | 50.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 50.55 | 51.56 | 50.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 50.17 | 51.28 | 50.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 50.21 | 51.28 | 50.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 15:15:00 | 50.30 | 50.65 | 50.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 49.75 | 50.47 | 50.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 50.18 | 50.03 | 50.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 50.18 | 50.03 | 50.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 50.18 | 50.03 | 50.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 15:00:00 | 49.82 | 50.07 | 50.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 49.87 | 50.01 | 50.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 10:15:00 | 49.83 | 49.99 | 50.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 51.21 | 50.04 | 50.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 51.21 | 50.04 | 50.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 51.21 | 50.04 | 50.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 09:15:00 | 51.21 | 50.04 | 50.02 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 50.15 | 50.36 | 50.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 49.88 | 50.23 | 50.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 49.53 | 49.50 | 49.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 14:15:00 | 49.55 | 49.51 | 49.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 49.55 | 49.51 | 49.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 49.55 | 49.51 | 49.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 49.66 | 49.54 | 49.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 49.83 | 49.54 | 49.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 49.86 | 49.60 | 49.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:30:00 | 49.43 | 49.59 | 49.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:00:00 | 49.45 | 49.59 | 49.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 15:15:00 | 49.49 | 49.57 | 49.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:30:00 | 49.41 | 49.53 | 49.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 49.71 | 49.57 | 49.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 49.71 | 49.57 | 49.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 49.69 | 49.59 | 49.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:15:00 | 49.61 | 49.59 | 49.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 49.65 | 49.60 | 49.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 13:15:00 | 49.57 | 49.60 | 49.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 14:15:00 | 49.73 | 49.64 | 49.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 14:15:00 | 49.73 | 49.64 | 49.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 14:15:00 | 49.73 | 49.64 | 49.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 14:15:00 | 49.73 | 49.64 | 49.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 14:15:00 | 49.73 | 49.64 | 49.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 14:15:00 | 49.73 | 49.64 | 49.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 15:15:00 | 49.90 | 49.70 | 49.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 10:15:00 | 49.62 | 49.69 | 49.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 10:15:00 | 49.62 | 49.69 | 49.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 49.62 | 49.69 | 49.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 49.51 | 49.69 | 49.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 49.58 | 49.66 | 49.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:30:00 | 49.57 | 49.66 | 49.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 12:15:00 | 49.58 | 49.65 | 49.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 49.46 | 49.61 | 49.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 46.29 | 46.18 | 46.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 46.69 | 46.18 | 46.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 46.73 | 46.29 | 46.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 46.11 | 46.52 | 46.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 10:00:00 | 46.29 | 46.47 | 46.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:00:00 | 46.09 | 46.40 | 46.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 43.98 | 44.72 | 45.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 12:15:00 | 43.80 | 44.63 | 45.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 12:15:00 | 43.79 | 44.63 | 45.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 15:15:00 | 44.20 | 44.07 | 44.80 | SL hit (close>ema200) qty=0.50 sl=44.07 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 15:15:00 | 44.20 | 44.07 | 44.80 | SL hit (close>ema200) qty=0.50 sl=44.07 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 15:15:00 | 44.20 | 44.07 | 44.80 | SL hit (close>ema200) qty=0.50 sl=44.07 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 43.68 | 42.93 | 42.82 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 12:15:00 | 42.45 | 42.76 | 42.79 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 43.80 | 42.89 | 42.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 10:15:00 | 44.05 | 43.12 | 42.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 15:15:00 | 44.89 | 44.89 | 44.04 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 10:45:00 | 46.20 | 45.19 | 44.33 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 44.85 | 45.21 | 44.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 43.67 | 45.21 | 44.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 43.56 | 44.88 | 44.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-02 09:15:00 | 43.56 | 44.88 | 44.59 | SL hit (close<ema400) qty=1.00 sl=44.59 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 43.56 | 44.88 | 44.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 42.82 | 44.47 | 44.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 42.82 | 44.47 | 44.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 43.06 | 44.19 | 44.30 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 44.73 | 44.24 | 44.22 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 43.93 | 44.30 | 44.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 13:15:00 | 43.87 | 44.16 | 44.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 43.84 | 43.56 | 43.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 14:15:00 | 43.84 | 43.56 | 43.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 43.84 | 43.56 | 43.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 43.84 | 43.56 | 43.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 43.51 | 43.55 | 43.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 44.40 | 43.55 | 43.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 44.29 | 43.70 | 43.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 44.14 | 43.70 | 43.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 44.99 | 43.96 | 43.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 46.45 | 45.03 | 44.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 45.64 | 45.91 | 45.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 45.64 | 45.91 | 45.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 45.64 | 45.91 | 45.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 45.39 | 45.91 | 45.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 45.43 | 45.65 | 45.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:30:00 | 45.44 | 45.65 | 45.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 45.40 | 45.60 | 45.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 44.72 | 45.60 | 45.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 44.71 | 45.42 | 45.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:00:00 | 44.71 | 45.42 | 45.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 44.65 | 45.27 | 45.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 43.83 | 44.67 | 44.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 44.01 | 43.55 | 43.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 44.01 | 43.55 | 43.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 44.01 | 43.55 | 43.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:45:00 | 43.23 | 43.49 | 43.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 15:00:00 | 43.22 | 43.29 | 43.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:30:00 | 43.08 | 43.26 | 43.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 41.07 | 41.31 | 41.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 41.06 | 41.31 | 41.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 40.93 | 41.31 | 41.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 38.91 | 39.91 | 40.62 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 38.90 | 39.91 | 40.62 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 38.77 | 39.91 | 40.62 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 55 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 40.30 | 39.72 | 39.67 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 38.63 | 39.71 | 39.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 38.11 | 38.85 | 39.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 38.88 | 38.77 | 39.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 38.88 | 38.77 | 39.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 38.88 | 38.77 | 39.12 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 39.66 | 39.21 | 39.19 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 38.86 | 39.19 | 39.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 38.28 | 38.98 | 39.10 | Break + close below crossover candle low |

### Cycle 59 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 11:15:00 | 40.05 | 39.17 | 39.17 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 38.45 | 39.22 | 39.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 38.25 | 39.02 | 39.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 38.16 | 37.94 | 38.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 38.16 | 37.94 | 38.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 38.16 | 37.94 | 38.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 38.40 | 37.94 | 38.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 38.18 | 37.98 | 38.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 38.74 | 37.98 | 38.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 38.57 | 38.10 | 38.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 38.25 | 38.16 | 38.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 38.99 | 38.39 | 38.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 38.99 | 38.39 | 38.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 39.68 | 38.64 | 38.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 38.80 | 39.26 | 38.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 38.80 | 39.26 | 38.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 38.80 | 39.26 | 38.94 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 38.04 | 38.66 | 38.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 15:15:00 | 37.90 | 38.51 | 38.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 34.82 | 34.34 | 35.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 34.82 | 34.34 | 35.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 35.45 | 34.66 | 35.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:45:00 | 35.63 | 34.66 | 35.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 35.35 | 34.80 | 35.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:30:00 | 35.44 | 34.80 | 35.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 12:15:00 | 35.17 | 34.93 | 35.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 14:15:00 | 35.05 | 34.96 | 35.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 14:45:00 | 34.96 | 34.96 | 35.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 10:15:00 | 33.30 | 34.41 | 34.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 10:15:00 | 33.21 | 34.41 | 34.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-30 14:15:00 | 31.54 | 32.28 | 33.20 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-30 14:15:00 | 31.46 | 32.28 | 33.20 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 63 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 35.71 | 33.57 | 33.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 36.45 | 34.15 | 33.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-01 15:15:00 | 34.69 | 34.74 | 34.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 15:15:00 | 34.69 | 34.74 | 34.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 15:15:00 | 34.69 | 34.74 | 34.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:15:00 | 33.75 | 34.74 | 34.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 34.35 | 34.67 | 34.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 10:30:00 | 35.43 | 34.98 | 34.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 09:15:00 | 38.97 | 38.25 | 37.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 11:15:00 | 43.62 | 44.11 | 44.13 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 13:15:00 | 44.34 | 44.02 | 44.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 14:15:00 | 45.13 | 44.39 | 44.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 09:15:00 | 44.56 | 44.57 | 44.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 09:15:00 | 44.56 | 44.57 | 44.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 44.56 | 44.57 | 44.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:45:00 | 44.04 | 44.57 | 44.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 43.75 | 44.40 | 44.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:45:00 | 43.93 | 44.40 | 44.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 43.74 | 44.27 | 44.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:45:00 | 43.53 | 44.27 | 44.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 43.75 | 44.17 | 44.19 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 15:15:00 | 44.58 | 44.27 | 44.23 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 44.12 | 44.35 | 44.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 43.98 | 44.18 | 44.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 10:15:00 | 43.25 | 43.18 | 43.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 11:00:00 | 43.25 | 43.18 | 43.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 43.15 | 43.17 | 43.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:15:00 | 43.07 | 43.17 | 43.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:00:00 | 43.07 | 43.04 | 43.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 13:00:00 | 43.03 | 43.01 | 43.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 43.50 | 43.15 | 43.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 43.50 | 43.15 | 43.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 43.50 | 43.15 | 43.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 43.50 | 43.15 | 43.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 43.81 | 43.29 | 43.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 44.11 | 44.47 | 44.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 44.11 | 44.47 | 44.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 44.11 | 44.47 | 44.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 44.19 | 44.47 | 44.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 44.07 | 44.39 | 44.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:45:00 | 44.03 | 44.39 | 44.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 44.15 | 44.34 | 44.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:30:00 | 44.15 | 44.34 | 44.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 44.18 | 44.31 | 44.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 44.12 | 44.31 | 44.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 44.01 | 44.25 | 44.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 44.01 | 44.25 | 44.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 44.14 | 44.23 | 44.05 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-28 09:30:00 | 75.31 | 2025-05-28 13:15:00 | 74.30 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-06-03 13:15:00 | 72.40 | 2025-06-05 12:15:00 | 73.09 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-06-03 15:15:00 | 72.16 | 2025-06-05 12:15:00 | 73.09 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-06-04 11:15:00 | 72.09 | 2025-06-05 12:15:00 | 73.09 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-06-04 13:45:00 | 72.42 | 2025-06-05 12:15:00 | 73.09 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-06-10 11:15:00 | 71.50 | 2025-06-11 09:15:00 | 80.12 | STOP_HIT | 1.00 | -12.06% |
| SELL | retest2 | 2025-06-10 12:00:00 | 71.45 | 2025-06-11 09:15:00 | 80.12 | STOP_HIT | 1.00 | -12.13% |
| SELL | retest2 | 2025-06-10 12:30:00 | 71.51 | 2025-06-11 09:15:00 | 80.12 | STOP_HIT | 1.00 | -12.04% |
| SELL | retest2 | 2025-06-10 15:00:00 | 71.50 | 2025-06-11 09:15:00 | 80.12 | STOP_HIT | 1.00 | -12.06% |
| SELL | retest2 | 2025-06-17 12:00:00 | 69.80 | 2025-06-18 09:15:00 | 66.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 12:00:00 | 69.80 | 2025-06-20 12:15:00 | 62.82 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-11 09:15:00 | 64.90 | 2025-07-23 09:15:00 | 65.00 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-07-15 09:45:00 | 64.90 | 2025-07-23 09:15:00 | 65.00 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-07-30 11:45:00 | 59.60 | 2025-08-06 11:15:00 | 56.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 14:15:00 | 59.63 | 2025-08-06 11:15:00 | 56.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 15:00:00 | 59.65 | 2025-08-06 11:15:00 | 56.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 09:15:00 | 59.20 | 2025-08-07 09:15:00 | 56.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 15:15:00 | 58.86 | 2025-08-07 13:15:00 | 55.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-01 12:15:00 | 59.01 | 2025-08-07 13:15:00 | 56.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-01 13:00:00 | 59.00 | 2025-08-07 13:15:00 | 56.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 11:45:00 | 59.60 | 2025-08-07 14:15:00 | 57.00 | STOP_HIT | 0.50 | 4.36% |
| SELL | retest2 | 2025-07-30 14:15:00 | 59.63 | 2025-08-07 14:15:00 | 57.00 | STOP_HIT | 0.50 | 4.41% |
| SELL | retest2 | 2025-07-30 15:00:00 | 59.65 | 2025-08-07 14:15:00 | 57.00 | STOP_HIT | 0.50 | 4.44% |
| SELL | retest2 | 2025-07-31 09:15:00 | 59.20 | 2025-08-07 14:15:00 | 57.00 | STOP_HIT | 0.50 | 3.72% |
| SELL | retest2 | 2025-07-31 15:15:00 | 58.86 | 2025-08-07 14:15:00 | 57.00 | STOP_HIT | 0.50 | 3.16% |
| SELL | retest2 | 2025-08-01 12:15:00 | 59.01 | 2025-08-07 14:15:00 | 57.00 | STOP_HIT | 0.50 | 3.41% |
| SELL | retest2 | 2025-08-01 13:00:00 | 59.00 | 2025-08-07 14:15:00 | 57.00 | STOP_HIT | 0.50 | 3.39% |
| SELL | retest2 | 2025-08-01 14:15:00 | 58.80 | 2025-08-08 11:15:00 | 57.95 | STOP_HIT | 1.00 | 1.45% |
| SELL | retest2 | 2025-08-06 10:45:00 | 57.06 | 2025-08-08 11:15:00 | 57.95 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-08-06 14:00:00 | 57.10 | 2025-08-08 11:15:00 | 57.95 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-08-06 14:45:00 | 57.30 | 2025-08-08 11:15:00 | 57.95 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-08-08 11:15:00 | 57.29 | 2025-08-08 11:15:00 | 57.95 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-08-21 10:45:00 | 59.70 | 2025-08-21 14:15:00 | 58.50 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-08-25 09:15:00 | 58.70 | 2025-09-01 14:15:00 | 55.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 09:15:00 | 58.70 | 2025-09-02 09:15:00 | 57.15 | STOP_HIT | 0.50 | 2.64% |
| SELL | retest2 | 2025-09-24 14:00:00 | 56.93 | 2025-10-03 13:15:00 | 55.44 | STOP_HIT | 1.00 | 2.62% |
| SELL | retest2 | 2025-09-26 09:45:00 | 56.66 | 2025-10-03 13:15:00 | 55.44 | STOP_HIT | 1.00 | 2.15% |
| SELL | retest2 | 2025-10-15 14:15:00 | 56.04 | 2025-10-27 13:15:00 | 55.40 | STOP_HIT | 1.00 | 1.14% |
| SELL | retest2 | 2025-10-16 13:00:00 | 56.00 | 2025-10-27 13:15:00 | 55.40 | STOP_HIT | 1.00 | 1.07% |
| SELL | retest2 | 2025-10-16 14:45:00 | 56.03 | 2025-10-27 13:15:00 | 55.40 | STOP_HIT | 1.00 | 1.12% |
| SELL | retest2 | 2025-10-16 15:15:00 | 56.00 | 2025-10-27 13:15:00 | 55.40 | STOP_HIT | 1.00 | 1.07% |
| SELL | retest2 | 2025-10-17 10:00:00 | 55.67 | 2025-10-27 13:15:00 | 55.40 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2025-10-20 09:15:00 | 55.26 | 2025-10-27 13:15:00 | 55.40 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-10-20 12:45:00 | 55.39 | 2025-10-27 13:15:00 | 55.40 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-10-23 09:45:00 | 54.67 | 2025-10-27 13:15:00 | 55.40 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-11-03 11:15:00 | 54.65 | 2025-11-03 14:15:00 | 55.95 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2025-11-03 12:15:00 | 54.71 | 2025-11-03 14:15:00 | 55.95 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-11-03 13:15:00 | 54.65 | 2025-11-03 14:15:00 | 55.95 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2025-11-10 11:15:00 | 54.15 | 2025-11-12 09:15:00 | 54.92 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-11-11 10:15:00 | 53.89 | 2025-11-12 09:15:00 | 54.92 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-11-14 14:30:00 | 53.99 | 2025-11-21 09:15:00 | 51.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 11:30:00 | 53.92 | 2025-11-21 09:15:00 | 51.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 12:30:00 | 53.84 | 2025-11-21 09:15:00 | 51.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-14 14:30:00 | 53.99 | 2025-11-24 09:15:00 | 51.20 | STOP_HIT | 0.50 | 5.17% |
| SELL | retest2 | 2025-11-17 11:30:00 | 53.92 | 2025-11-24 09:15:00 | 51.20 | STOP_HIT | 0.50 | 5.04% |
| SELL | retest2 | 2025-11-17 12:30:00 | 53.84 | 2025-11-24 09:15:00 | 51.20 | STOP_HIT | 0.50 | 4.90% |
| SELL | retest2 | 2025-12-19 15:00:00 | 49.82 | 2025-12-24 09:15:00 | 51.21 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-12-23 09:15:00 | 49.87 | 2025-12-24 09:15:00 | 51.21 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-12-23 10:15:00 | 49.83 | 2025-12-24 09:15:00 | 51.21 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2026-01-01 12:30:00 | 49.43 | 2026-01-02 14:15:00 | 49.73 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-01-01 13:00:00 | 49.45 | 2026-01-02 14:15:00 | 49.73 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2026-01-01 15:15:00 | 49.49 | 2026-01-02 14:15:00 | 49.73 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2026-01-02 09:30:00 | 49.41 | 2026-01-02 14:15:00 | 49.73 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2026-01-02 13:15:00 | 49.57 | 2026-01-02 14:15:00 | 49.73 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2026-01-16 09:15:00 | 46.11 | 2026-01-20 11:15:00 | 43.98 | PARTIAL | 0.50 | 4.63% |
| SELL | retest2 | 2026-01-16 10:00:00 | 46.29 | 2026-01-20 12:15:00 | 43.80 | PARTIAL | 0.50 | 5.37% |
| SELL | retest2 | 2026-01-16 11:00:00 | 46.09 | 2026-01-20 12:15:00 | 43.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:15:00 | 46.11 | 2026-01-20 15:15:00 | 44.20 | STOP_HIT | 0.50 | 4.14% |
| SELL | retest2 | 2026-01-16 10:00:00 | 46.29 | 2026-01-20 15:15:00 | 44.20 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2026-01-16 11:00:00 | 46.09 | 2026-01-20 15:15:00 | 44.20 | STOP_HIT | 0.50 | 4.10% |
| BUY | retest1 | 2026-02-01 10:45:00 | 46.20 | 2026-02-02 09:15:00 | 43.56 | STOP_HIT | 1.00 | -5.71% |
| SELL | retest2 | 2026-02-18 09:45:00 | 43.23 | 2026-03-02 09:15:00 | 41.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 15:00:00 | 43.22 | 2026-03-02 09:15:00 | 41.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 09:30:00 | 43.08 | 2026-03-02 09:15:00 | 40.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 09:45:00 | 43.23 | 2026-03-04 09:15:00 | 38.91 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-18 15:00:00 | 43.22 | 2026-03-04 09:15:00 | 38.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-19 09:30:00 | 43.08 | 2026-03-04 09:15:00 | 38.77 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-17 11:15:00 | 38.25 | 2026-03-18 09:15:00 | 38.99 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-03-25 14:15:00 | 35.05 | 2026-03-27 10:15:00 | 33.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 14:45:00 | 34.96 | 2026-03-27 10:15:00 | 33.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 14:15:00 | 35.05 | 2026-03-30 14:15:00 | 31.54 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-25 14:45:00 | 34.96 | 2026-03-30 14:15:00 | 31.46 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-02 10:30:00 | 35.43 | 2026-04-08 09:15:00 | 38.97 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-04 12:15:00 | 43.07 | 2026-05-06 15:15:00 | 43.50 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-05-05 10:00:00 | 43.07 | 2026-05-06 15:15:00 | 43.50 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-05-05 13:00:00 | 43.03 | 2026-05-06 15:15:00 | 43.50 | STOP_HIT | 1.00 | -1.09% |
