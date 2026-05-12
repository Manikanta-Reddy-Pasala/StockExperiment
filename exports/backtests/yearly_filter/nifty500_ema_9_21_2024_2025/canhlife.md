# Canara HSBC Life Insurance Company Ltd. (CANHLIFE)

## Backtest Summary

- **Window:** 2025-10-17 09:15:00 → 2026-05-11 15:15:00 (954 bars)
- **Last close:** 141.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 47 |
| ALERT1 | 32 |
| ALERT2 | 31 |
| ALERT2_SKIP | 20 |
| ALERT3 | 85 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 45 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 46 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 36
- **Target hits / Stop hits / Partials:** 1 / 44 / 1
- **Avg / median % per leg:** -0.75% / -1.41%
- **Sum % (uncompounded):** -34.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 8 | 44.4% | 1 | 17 | 0 | 0.25% | 4.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 8 | 44.4% | 1 | 17 | 0 | 0.25% | 4.6% |
| SELL (all) | 28 | 2 | 7.1% | 0 | 27 | 1 | -1.39% | -38.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 2 | 7.1% | 0 | 27 | 1 | -1.39% | -38.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 46 | 10 | 21.7% | 1 | 44 | 1 | -0.75% | -34.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 14:15:00 | 117.65 | 118.82 | 118.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 15:15:00 | 116.99 | 118.46 | 118.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 12:15:00 | 118.18 | 118.14 | 118.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-30 12:45:00 | 118.06 | 118.14 | 118.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 119.25 | 118.36 | 118.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:15:00 | 119.67 | 118.36 | 118.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 118.05 | 118.30 | 118.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 15:15:00 | 117.90 | 118.30 | 118.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 14:45:00 | 117.43 | 118.11 | 118.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 12:45:00 | 117.52 | 117.29 | 117.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 14:45:00 | 117.94 | 117.54 | 117.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 118.26 | 117.68 | 117.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 120.99 | 117.68 | 117.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 120.27 | 118.20 | 118.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 120.27 | 118.20 | 118.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 13:15:00 | 123.91 | 122.25 | 120.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 09:15:00 | 122.20 | 122.38 | 121.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 10:00:00 | 122.20 | 122.38 | 121.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 121.72 | 122.25 | 121.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 121.00 | 122.25 | 121.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 121.45 | 122.09 | 121.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:45:00 | 121.31 | 122.09 | 121.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 121.29 | 121.93 | 121.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:45:00 | 121.25 | 121.93 | 121.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 121.50 | 121.84 | 121.40 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 14:15:00 | 121.12 | 121.23 | 121.24 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 121.93 | 121.33 | 121.28 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 121.00 | 121.27 | 121.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 119.19 | 120.75 | 121.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 13:15:00 | 120.70 | 120.17 | 120.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 13:15:00 | 120.70 | 120.17 | 120.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 120.70 | 120.17 | 120.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 14:00:00 | 120.70 | 120.17 | 120.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 120.14 | 120.16 | 120.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:15:00 | 120.50 | 120.16 | 120.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 120.50 | 120.23 | 120.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 10:30:00 | 119.55 | 120.22 | 120.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 119.69 | 120.29 | 120.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 119.26 | 119.70 | 119.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 11:15:00 | 121.86 | 120.10 | 120.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 121.86 | 120.10 | 120.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 125.25 | 123.35 | 122.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 124.62 | 124.79 | 124.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 13:00:00 | 124.62 | 124.79 | 124.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 124.25 | 124.68 | 124.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 122.98 | 124.68 | 124.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 122.66 | 124.28 | 124.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:30:00 | 122.08 | 124.28 | 124.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 10:15:00 | 123.14 | 124.05 | 124.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 120.19 | 122.58 | 123.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 12:15:00 | 122.99 | 122.46 | 122.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 12:15:00 | 122.99 | 122.46 | 122.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 122.99 | 122.46 | 122.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:30:00 | 122.99 | 122.46 | 122.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 122.78 | 122.53 | 122.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:00:00 | 122.78 | 122.53 | 122.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 123.00 | 122.62 | 122.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:45:00 | 123.00 | 122.62 | 122.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 123.00 | 122.70 | 122.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 121.50 | 122.70 | 122.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 11:00:00 | 122.41 | 122.50 | 122.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 12:30:00 | 122.36 | 122.54 | 122.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 14:45:00 | 122.38 | 122.46 | 122.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 121.85 | 121.41 | 121.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:45:00 | 121.80 | 121.41 | 121.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 122.09 | 121.55 | 121.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:30:00 | 122.30 | 121.55 | 121.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 122.70 | 121.78 | 121.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:30:00 | 122.96 | 121.78 | 121.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-05 14:15:00 | 122.64 | 122.07 | 122.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 14:15:00 | 122.64 | 122.07 | 122.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 09:15:00 | 122.96 | 122.32 | 122.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 122.34 | 122.64 | 122.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 11:15:00 | 122.34 | 122.64 | 122.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 122.34 | 122.64 | 122.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:45:00 | 122.87 | 122.64 | 122.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 122.40 | 122.59 | 122.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 15:15:00 | 122.89 | 122.50 | 122.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 120.25 | 122.11 | 122.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 120.25 | 122.11 | 122.21 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 123.65 | 122.44 | 122.32 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 10:15:00 | 121.69 | 122.29 | 122.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 119.64 | 121.71 | 122.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 120.50 | 120.00 | 120.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 10:00:00 | 120.50 | 120.00 | 120.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 121.54 | 120.31 | 120.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:00:00 | 121.54 | 120.31 | 120.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 121.65 | 120.58 | 120.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:00:00 | 121.65 | 120.58 | 120.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 121.68 | 120.80 | 121.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:00:00 | 121.68 | 120.80 | 121.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 123.00 | 121.50 | 121.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 10:15:00 | 123.90 | 122.00 | 121.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 10:15:00 | 122.67 | 123.28 | 122.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 10:15:00 | 122.67 | 123.28 | 122.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 122.67 | 123.28 | 122.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 122.67 | 123.28 | 122.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 123.92 | 123.41 | 122.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 12:15:00 | 124.05 | 123.41 | 122.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 122.34 | 123.60 | 123.13 | SL hit (close<static) qty=1.00 sl=122.60 alert=retest2 |

### Cycle 13 — SELL (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 14:15:00 | 148.17 | 148.87 | 148.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 09:15:00 | 147.07 | 148.44 | 148.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 10:15:00 | 148.80 | 147.47 | 147.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 10:15:00 | 148.80 | 147.47 | 147.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 148.80 | 147.47 | 147.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:45:00 | 149.50 | 147.47 | 147.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 150.39 | 148.05 | 148.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:00:00 | 150.39 | 148.05 | 148.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 12:15:00 | 149.14 | 148.27 | 148.21 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 14:15:00 | 148.04 | 148.37 | 148.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 15:15:00 | 148.00 | 148.29 | 148.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 11:15:00 | 148.95 | 148.17 | 148.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 11:15:00 | 148.95 | 148.17 | 148.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 148.95 | 148.17 | 148.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:00:00 | 148.95 | 148.17 | 148.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 12:15:00 | 150.80 | 148.70 | 148.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 15:15:00 | 155.00 | 150.85 | 149.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 150.00 | 150.68 | 149.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 150.00 | 150.68 | 149.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 150.00 | 150.68 | 149.81 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 148.12 | 149.50 | 149.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 146.38 | 148.87 | 149.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 10:15:00 | 149.58 | 149.01 | 149.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 10:15:00 | 149.58 | 149.01 | 149.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 149.58 | 149.01 | 149.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:00:00 | 149.58 | 149.01 | 149.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 148.98 | 149.01 | 149.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:30:00 | 149.32 | 149.01 | 149.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 148.19 | 148.15 | 148.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 15:15:00 | 144.92 | 146.79 | 147.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 13:15:00 | 137.67 | 138.61 | 139.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 146.04 | 139.53 | 139.64 | SL hit (close>ema200) qty=0.50 sl=139.53 alert=retest2 |

### Cycle 18 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 142.00 | 140.03 | 139.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 12:15:00 | 150.44 | 142.92 | 141.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 147.25 | 147.54 | 144.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 11:45:00 | 146.98 | 147.54 | 144.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 146.25 | 147.03 | 145.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 09:15:00 | 150.30 | 146.71 | 146.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 149.75 | 146.90 | 146.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 13:15:00 | 145.51 | 146.36 | 146.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 145.51 | 146.36 | 146.41 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 148.57 | 146.71 | 146.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 10:15:00 | 149.21 | 147.21 | 146.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 146.69 | 147.36 | 146.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 12:15:00 | 146.69 | 147.36 | 146.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 146.69 | 147.36 | 146.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 146.69 | 147.36 | 146.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 147.07 | 147.30 | 146.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 15:15:00 | 148.25 | 147.24 | 146.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 09:45:00 | 147.94 | 148.31 | 147.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 151.07 | 148.75 | 148.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 10:15:00 | 146.48 | 148.02 | 148.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 10:15:00 | 146.48 | 148.02 | 148.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-03 12:15:00 | 144.91 | 147.17 | 147.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 147.35 | 146.17 | 146.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 147.35 | 146.17 | 146.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 147.35 | 146.17 | 146.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:00:00 | 147.35 | 146.17 | 146.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 146.15 | 146.16 | 146.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 11:15:00 | 145.88 | 146.16 | 146.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 12:00:00 | 145.54 | 146.04 | 146.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 14:15:00 | 145.15 | 145.97 | 146.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 145.04 | 145.97 | 146.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 145.00 | 145.13 | 145.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 146.75 | 145.13 | 145.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 147.73 | 145.65 | 145.87 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 147.73 | 145.65 | 145.87 | SL hit (close>static) qty=1.00 sl=147.48 alert=retest2 |

### Cycle 22 — BUY (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 10:15:00 | 148.37 | 146.19 | 146.10 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 13:15:00 | 145.14 | 145.97 | 146.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 15:15:00 | 144.67 | 145.54 | 145.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 145.59 | 145.55 | 145.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 145.59 | 145.55 | 145.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 145.59 | 145.55 | 145.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 145.95 | 145.55 | 145.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 147.60 | 145.96 | 145.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 12:15:00 | 149.39 | 147.83 | 147.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 147.36 | 147.74 | 147.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:00:00 | 147.36 | 147.74 | 147.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 146.48 | 147.49 | 147.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 146.48 | 147.49 | 147.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 147.00 | 147.39 | 147.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 145.02 | 147.39 | 147.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 147.06 | 147.08 | 146.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 12:30:00 | 148.02 | 147.16 | 147.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 10:30:00 | 147.50 | 147.81 | 147.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 11:00:00 | 147.80 | 147.81 | 147.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:30:00 | 147.62 | 147.69 | 147.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 147.17 | 147.58 | 147.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:45:00 | 146.95 | 147.58 | 147.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 147.00 | 147.47 | 147.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:15:00 | 146.53 | 147.47 | 147.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 147.27 | 147.43 | 147.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 10:15:00 | 148.15 | 147.43 | 147.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 12:00:00 | 151.64 | 148.44 | 147.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 12:15:00 | 151.20 | 151.55 | 151.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 151.20 | 151.55 | 151.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 150.93 | 151.43 | 151.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 14:15:00 | 149.93 | 149.88 | 150.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 15:00:00 | 149.93 | 149.88 | 150.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 148.17 | 149.38 | 150.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:15:00 | 147.99 | 149.38 | 150.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:30:00 | 147.91 | 148.59 | 149.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:15:00 | 147.99 | 148.59 | 149.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 15:00:00 | 147.66 | 148.28 | 149.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 148.49 | 148.21 | 149.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 148.43 | 148.21 | 149.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 148.79 | 148.33 | 148.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 10:30:00 | 148.91 | 148.33 | 148.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 150.56 | 148.64 | 149.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:00:00 | 150.56 | 148.64 | 149.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 149.90 | 148.89 | 149.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 15:15:00 | 148.12 | 149.09 | 149.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 150.07 | 149.18 | 149.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 150.07 | 149.18 | 149.16 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 10:15:00 | 147.71 | 149.11 | 149.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 11:15:00 | 146.93 | 148.68 | 149.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 13:15:00 | 146.41 | 145.76 | 146.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 13:15:00 | 146.41 | 145.76 | 146.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 146.41 | 145.76 | 146.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:45:00 | 146.36 | 145.76 | 146.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 142.70 | 145.15 | 146.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 142.70 | 145.15 | 146.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 142.51 | 144.62 | 146.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 139.10 | 144.62 | 146.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 10:30:00 | 140.43 | 143.39 | 145.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 11:30:00 | 141.43 | 142.97 | 144.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 139.64 | 142.99 | 144.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 143.24 | 142.01 | 143.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:30:00 | 143.13 | 142.01 | 143.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 143.50 | 142.31 | 143.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:15:00 | 144.74 | 142.31 | 143.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 145.13 | 143.15 | 143.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 145.13 | 143.15 | 143.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 144.10 | 143.39 | 143.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:45:00 | 144.24 | 143.39 | 143.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 143.52 | 143.42 | 143.44 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 143.97 | 143.53 | 143.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 143.97 | 143.53 | 143.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 15:15:00 | 144.88 | 143.80 | 143.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 143.06 | 144.93 | 144.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 143.06 | 144.93 | 144.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 143.06 | 144.93 | 144.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 12:30:00 | 145.07 | 144.93 | 144.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 11:00:00 | 146.17 | 145.43 | 145.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 14:15:00 | 146.11 | 147.03 | 147.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 14:15:00 | 146.11 | 147.03 | 147.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 15:15:00 | 144.50 | 146.52 | 146.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 12:15:00 | 141.94 | 141.36 | 143.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 12:15:00 | 141.94 | 141.36 | 143.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 141.94 | 141.36 | 143.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:45:00 | 142.93 | 141.36 | 143.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 144.25 | 141.94 | 143.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:45:00 | 145.17 | 141.94 | 143.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 145.05 | 142.56 | 143.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 145.64 | 142.56 | 143.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 144.92 | 143.18 | 143.58 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 145.05 | 143.88 | 143.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 13:15:00 | 147.14 | 144.71 | 144.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 145.49 | 146.87 | 146.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 145.49 | 146.87 | 146.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 145.49 | 146.87 | 146.13 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 144.32 | 145.65 | 145.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 140.78 | 144.38 | 145.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 142.64 | 141.69 | 142.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 142.64 | 141.69 | 142.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 142.64 | 141.69 | 142.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 142.56 | 141.69 | 142.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 142.95 | 141.94 | 142.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 143.22 | 141.94 | 142.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 144.55 | 142.46 | 142.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:45:00 | 144.60 | 142.46 | 142.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 144.90 | 142.95 | 143.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 144.00 | 142.95 | 143.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 144.43 | 143.25 | 143.22 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 14:15:00 | 142.00 | 143.33 | 143.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-25 15:15:00 | 140.20 | 142.71 | 143.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 13:15:00 | 141.44 | 141.29 | 142.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-27 14:00:00 | 141.44 | 141.29 | 142.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 142.39 | 141.51 | 142.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 142.39 | 141.51 | 142.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 140.50 | 141.31 | 141.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 10:00:00 | 138.01 | 140.65 | 141.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 143.98 | 140.12 | 140.50 | SL hit (close>static) qty=1.00 sl=142.80 alert=retest2 |

### Cycle 34 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 144.08 | 140.91 | 140.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 144.94 | 141.72 | 141.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 141.58 | 142.91 | 142.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 141.58 | 142.91 | 142.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 141.58 | 142.91 | 142.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 142.47 | 142.58 | 142.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 15:15:00 | 144.80 | 145.62 | 145.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 15:15:00 | 144.80 | 145.62 | 145.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 09:15:00 | 143.09 | 145.11 | 145.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 14:15:00 | 144.29 | 144.28 | 144.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 14:15:00 | 144.29 | 144.28 | 144.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 144.29 | 144.28 | 144.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 15:00:00 | 144.29 | 144.28 | 144.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 143.60 | 144.15 | 144.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:15:00 | 145.00 | 144.15 | 144.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 145.17 | 144.35 | 144.76 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2026-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 13:15:00 | 146.32 | 145.21 | 145.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 147.32 | 145.96 | 145.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 11:15:00 | 149.09 | 149.38 | 148.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 11:45:00 | 149.02 | 149.38 | 148.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 13:15:00 | 148.46 | 149.10 | 148.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 14:00:00 | 148.46 | 149.10 | 148.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 148.43 | 149.11 | 148.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 148.10 | 149.11 | 148.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 148.68 | 149.02 | 148.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 12:30:00 | 150.20 | 149.09 | 148.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 09:15:00 | 144.00 | 148.60 | 148.55 | SL hit (close<static) qty=1.00 sl=147.96 alert=retest2 |

### Cycle 37 — SELL (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 10:15:00 | 145.23 | 147.93 | 148.25 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 11:15:00 | 149.16 | 145.81 | 145.80 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 144.20 | 146.08 | 146.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 142.25 | 145.31 | 145.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 146.70 | 138.96 | 140.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 146.70 | 138.96 | 140.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 146.70 | 138.96 | 140.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 147.55 | 138.96 | 140.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 145.78 | 141.47 | 141.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 147.25 | 142.62 | 141.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 143.62 | 144.33 | 142.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 143.62 | 144.33 | 142.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 143.62 | 144.33 | 142.90 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 15:15:00 | 140.82 | 142.55 | 142.58 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 143.57 | 142.75 | 142.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 144.41 | 143.30 | 142.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 14:15:00 | 144.37 | 144.42 | 143.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 14:45:00 | 144.98 | 144.42 | 143.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 143.10 | 144.16 | 143.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 144.16 | 144.16 | 143.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 143.27 | 143.98 | 143.66 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 11:15:00 | 141.35 | 143.24 | 143.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 12:15:00 | 140.20 | 142.63 | 143.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-07 09:15:00 | 141.99 | 141.79 | 142.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-07 09:30:00 | 142.17 | 141.79 | 142.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 143.65 | 142.17 | 142.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:45:00 | 143.90 | 142.17 | 142.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 143.73 | 142.48 | 142.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:30:00 | 143.40 | 142.48 | 142.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2026-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 13:15:00 | 143.41 | 142.80 | 142.80 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 14:15:00 | 142.49 | 142.74 | 142.77 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2026-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 15:15:00 | 143.68 | 142.93 | 142.85 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 141.85 | 142.80 | 142.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-11 14:15:00 | 141.01 | 142.10 | 142.42 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-10-30 15:15:00 | 117.90 | 2025-11-04 09:15:00 | 120.27 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-10-31 14:45:00 | 117.43 | 2025-11-04 09:15:00 | 120.27 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-11-03 12:45:00 | 117.52 | 2025-11-04 09:15:00 | 120.27 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-11-03 14:45:00 | 117.94 | 2025-11-04 09:15:00 | 120.27 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-11-17 10:30:00 | 119.55 | 2025-11-19 11:15:00 | 121.86 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-11-18 09:15:00 | 119.69 | 2025-11-19 11:15:00 | 121.86 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-11-19 09:15:00 | 119.26 | 2025-11-19 11:15:00 | 121.86 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-12-03 09:15:00 | 121.50 | 2025-12-05 14:15:00 | 122.64 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-12-03 11:00:00 | 122.41 | 2025-12-05 14:15:00 | 122.64 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-12-03 12:30:00 | 122.36 | 2025-12-05 14:15:00 | 122.64 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-12-03 14:45:00 | 122.38 | 2025-12-05 14:15:00 | 122.64 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-12-08 15:15:00 | 122.89 | 2025-12-09 09:15:00 | 120.25 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-12-16 12:15:00 | 124.05 | 2025-12-17 09:15:00 | 122.34 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-12-17 12:30:00 | 126.33 | 2025-12-23 09:15:00 | 138.96 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-12 15:15:00 | 144.92 | 2026-01-21 13:15:00 | 137.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-12 15:15:00 | 144.92 | 2026-01-22 09:15:00 | 146.04 | STOP_HIT | 0.50 | -0.77% |
| SELL | retest2 | 2026-01-22 09:45:00 | 144.87 | 2026-01-22 10:15:00 | 142.00 | STOP_HIT | 1.00 | 1.98% |
| BUY | retest2 | 2026-01-28 09:15:00 | 150.30 | 2026-01-29 13:15:00 | 145.51 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2026-01-29 09:15:00 | 149.75 | 2026-01-29 13:15:00 | 145.51 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-01-30 15:15:00 | 148.25 | 2026-02-03 10:15:00 | 146.48 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-02-01 09:45:00 | 147.94 | 2026-02-03 10:15:00 | 146.48 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2026-02-02 09:15:00 | 151.07 | 2026-02-03 10:15:00 | 146.48 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2026-02-04 11:15:00 | 145.88 | 2026-02-06 09:15:00 | 147.73 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-02-04 12:00:00 | 145.54 | 2026-02-06 09:15:00 | 147.73 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-02-04 14:15:00 | 145.15 | 2026-02-06 09:15:00 | 147.73 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-02-05 09:15:00 | 145.04 | 2026-02-06 09:15:00 | 147.73 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2026-02-11 12:30:00 | 148.02 | 2026-02-19 12:15:00 | 151.20 | STOP_HIT | 1.00 | 2.15% |
| BUY | retest2 | 2026-02-12 10:30:00 | 147.50 | 2026-02-19 12:15:00 | 151.20 | STOP_HIT | 1.00 | 2.51% |
| BUY | retest2 | 2026-02-12 11:00:00 | 147.80 | 2026-02-19 12:15:00 | 151.20 | STOP_HIT | 1.00 | 2.30% |
| BUY | retest2 | 2026-02-12 13:30:00 | 147.62 | 2026-02-19 12:15:00 | 151.20 | STOP_HIT | 1.00 | 2.43% |
| BUY | retest2 | 2026-02-13 10:15:00 | 148.15 | 2026-02-19 12:15:00 | 151.20 | STOP_HIT | 1.00 | 2.06% |
| BUY | retest2 | 2026-02-13 12:00:00 | 151.64 | 2026-02-19 12:15:00 | 151.20 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2026-02-23 10:15:00 | 147.99 | 2026-02-25 11:15:00 | 150.07 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2026-02-23 12:30:00 | 147.91 | 2026-02-25 11:15:00 | 150.07 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-02-23 13:15:00 | 147.99 | 2026-02-25 11:15:00 | 150.07 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2026-02-23 15:00:00 | 147.66 | 2026-02-25 11:15:00 | 150.07 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2026-02-24 15:15:00 | 148.12 | 2026-02-25 11:15:00 | 150.07 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-03-02 09:15:00 | 139.10 | 2026-03-05 14:15:00 | 143.97 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2026-03-02 10:30:00 | 140.43 | 2026-03-05 14:15:00 | 143.97 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2026-03-02 11:30:00 | 141.43 | 2026-03-05 14:15:00 | 143.97 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-03-04 09:15:00 | 139.64 | 2026-03-05 14:15:00 | 143.97 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2026-03-09 12:30:00 | 145.07 | 2026-03-12 14:15:00 | 146.11 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2026-03-10 11:00:00 | 146.17 | 2026-03-12 14:15:00 | 146.11 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2026-03-30 10:00:00 | 138.01 | 2026-04-01 09:15:00 | 143.98 | STOP_HIT | 1.00 | -4.33% |
| BUY | retest2 | 2026-04-02 11:15:00 | 142.47 | 2026-04-10 15:15:00 | 144.80 | STOP_HIT | 1.00 | 1.64% |
| BUY | retest2 | 2026-04-20 12:30:00 | 150.20 | 2026-04-21 09:15:00 | 144.00 | STOP_HIT | 1.00 | -4.13% |
