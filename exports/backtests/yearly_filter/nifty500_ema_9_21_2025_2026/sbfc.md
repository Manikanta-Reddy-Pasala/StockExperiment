# SBFC Finance Ltd. (SBFC)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-11 15:15:00 (1983 bars)
- **Last close:** 98.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 77 |
| ALERT1 | 47 |
| ALERT2 | 47 |
| ALERT2_SKIP | 25 |
| ALERT3 | 131 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 46 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 45 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 50 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 31
- **Target hits / Stop hits / Partials:** 1 / 45 / 4
- **Avg / median % per leg:** 0.16% / -0.80%
- **Sum % (uncompounded):** 7.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 5 | 25.0% | 0 | 20 | 0 | -0.77% | -15.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 5 | 25.0% | 0 | 20 | 0 | -0.77% | -15.5% |
| SELL (all) | 30 | 14 | 46.7% | 1 | 25 | 4 | 0.77% | 23.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 14 | 46.7% | 1 | 25 | 4 | 0.77% | 23.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 50 | 19 | 38.0% | 1 | 45 | 4 | 0.16% | 7.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 101.19 | 98.88 | 98.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 104.70 | 101.15 | 100.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 107.56 | 107.66 | 105.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 10:00:00 | 107.56 | 107.66 | 105.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 108.01 | 107.24 | 106.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 14:30:00 | 106.12 | 107.24 | 106.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 106.25 | 107.89 | 107.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 106.25 | 107.89 | 107.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 106.30 | 107.57 | 107.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:00:00 | 106.30 | 107.57 | 107.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 105.70 | 106.73 | 106.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 11:15:00 | 105.14 | 106.41 | 106.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 105.22 | 104.51 | 105.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 14:15:00 | 105.22 | 104.51 | 105.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 105.22 | 104.51 | 105.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:00:00 | 105.22 | 104.51 | 105.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 105.39 | 104.69 | 105.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 104.65 | 104.69 | 105.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 105.39 | 104.83 | 105.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 12:45:00 | 104.34 | 104.70 | 105.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:15:00 | 104.33 | 104.70 | 105.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 14:00:00 | 104.21 | 104.60 | 104.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 10:15:00 | 104.25 | 104.84 | 105.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 105.33 | 104.91 | 105.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:45:00 | 105.40 | 104.91 | 105.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 105.33 | 104.99 | 105.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:45:00 | 105.40 | 104.99 | 105.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-23 13:15:00 | 106.84 | 105.36 | 105.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 106.84 | 105.36 | 105.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 107.92 | 106.35 | 105.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 13:15:00 | 106.80 | 106.81 | 106.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 13:30:00 | 106.85 | 106.81 | 106.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 105.50 | 106.54 | 106.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 15:00:00 | 105.50 | 106.54 | 106.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 106.00 | 106.44 | 106.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 106.35 | 106.44 | 106.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 10:15:00 | 106.69 | 106.34 | 106.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 13:15:00 | 111.55 | 112.16 | 112.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 13:15:00 | 111.55 | 112.16 | 112.19 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 113.80 | 112.49 | 112.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 114.39 | 113.14 | 112.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 12:15:00 | 114.50 | 114.81 | 114.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 13:00:00 | 114.50 | 114.81 | 114.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 13:15:00 | 114.05 | 114.66 | 114.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:00:00 | 114.05 | 114.66 | 114.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 113.61 | 114.45 | 114.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 15:00:00 | 113.61 | 114.45 | 114.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 114.00 | 114.36 | 114.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:30:00 | 113.81 | 114.36 | 114.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 113.97 | 114.28 | 114.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 113.97 | 114.28 | 114.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 113.86 | 114.20 | 114.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:15:00 | 113.55 | 114.20 | 114.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 113.17 | 113.99 | 113.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:00:00 | 113.17 | 113.99 | 113.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 114.45 | 114.21 | 114.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 114.47 | 114.21 | 114.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 114.15 | 114.20 | 114.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 114.22 | 114.20 | 114.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 113.91 | 114.14 | 114.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 113.91 | 114.14 | 114.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 113.88 | 114.09 | 114.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:45:00 | 113.70 | 114.09 | 114.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 112.29 | 113.73 | 113.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 110.64 | 112.37 | 113.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 110.25 | 109.84 | 111.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 15:15:00 | 110.25 | 109.84 | 111.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 110.25 | 109.84 | 111.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 107.90 | 109.84 | 111.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 15:15:00 | 102.50 | 104.24 | 105.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 104.28 | 104.20 | 105.20 | SL hit (close>ema200) qty=0.50 sl=104.20 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 107.47 | 105.43 | 105.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 108.73 | 106.97 | 106.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 11:15:00 | 107.00 | 107.12 | 106.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 12:00:00 | 107.00 | 107.12 | 106.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 106.42 | 106.86 | 106.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:00:00 | 106.42 | 106.86 | 106.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 106.75 | 106.84 | 106.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 106.12 | 106.70 | 106.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 105.11 | 106.38 | 106.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 105.11 | 106.38 | 106.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 105.30 | 106.16 | 106.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 14:15:00 | 103.78 | 105.40 | 105.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 106.15 | 105.44 | 105.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 106.15 | 105.44 | 105.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 106.15 | 105.44 | 105.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:15:00 | 106.92 | 105.44 | 105.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 106.74 | 105.70 | 105.83 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 106.93 | 105.95 | 105.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 12:15:00 | 110.49 | 107.41 | 106.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 12:15:00 | 109.10 | 110.34 | 108.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 12:15:00 | 109.10 | 110.34 | 108.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 109.10 | 110.34 | 108.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:45:00 | 109.19 | 110.34 | 108.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 108.60 | 109.99 | 108.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:00:00 | 108.60 | 109.99 | 108.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 108.00 | 109.59 | 108.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:45:00 | 108.00 | 109.59 | 108.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 108.30 | 109.34 | 108.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 107.22 | 109.34 | 108.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 106.50 | 108.35 | 108.41 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 110.82 | 108.28 | 108.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 11:15:00 | 112.04 | 111.03 | 110.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 10:15:00 | 111.45 | 111.91 | 111.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 11:00:00 | 111.45 | 111.91 | 111.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 110.75 | 111.68 | 111.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 110.75 | 111.68 | 111.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 110.60 | 111.46 | 111.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:15:00 | 110.50 | 111.46 | 111.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 110.93 | 111.28 | 111.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 110.93 | 111.28 | 111.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 110.61 | 111.14 | 111.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 111.68 | 111.14 | 111.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 10:45:00 | 111.30 | 111.22 | 111.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 115.81 | 117.31 | 117.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 115.81 | 117.31 | 117.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 115.55 | 116.29 | 116.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 11:15:00 | 115.83 | 115.82 | 116.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 12:00:00 | 115.83 | 115.82 | 116.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 116.30 | 115.25 | 115.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 116.94 | 115.25 | 115.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 115.98 | 115.40 | 115.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:45:00 | 116.23 | 115.40 | 115.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 116.57 | 115.63 | 115.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:00:00 | 116.57 | 115.63 | 115.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 115.85 | 115.68 | 115.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:30:00 | 116.90 | 115.68 | 115.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 117.03 | 115.95 | 115.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:45:00 | 117.14 | 115.95 | 115.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 14:15:00 | 118.19 | 116.40 | 116.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 13:15:00 | 118.70 | 117.84 | 117.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 09:15:00 | 117.15 | 117.91 | 117.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 117.15 | 117.91 | 117.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 117.15 | 117.91 | 117.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 117.15 | 117.91 | 117.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 115.72 | 117.47 | 117.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:45:00 | 115.82 | 117.47 | 117.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 114.56 | 116.89 | 116.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 13:15:00 | 114.19 | 115.97 | 116.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 112.70 | 111.06 | 112.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 112.70 | 111.06 | 112.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 112.70 | 111.06 | 112.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:00:00 | 108.65 | 111.08 | 112.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 11:15:00 | 103.22 | 104.73 | 106.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 12:15:00 | 103.24 | 102.68 | 103.99 | SL hit (close>ema200) qty=0.50 sl=102.68 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 104.58 | 102.32 | 102.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 09:15:00 | 105.25 | 103.64 | 103.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 15:15:00 | 113.50 | 113.55 | 112.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 09:15:00 | 112.95 | 113.55 | 112.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 111.50 | 113.14 | 112.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 111.50 | 113.14 | 112.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 111.83 | 112.88 | 112.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:45:00 | 111.33 | 112.88 | 112.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 112.13 | 112.30 | 112.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:45:00 | 112.18 | 112.30 | 112.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 15:15:00 | 112.10 | 112.26 | 112.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 110.69 | 111.94 | 112.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 11:15:00 | 107.26 | 106.26 | 107.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 12:00:00 | 107.26 | 106.26 | 107.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 106.11 | 106.22 | 107.25 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 108.01 | 107.49 | 107.43 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 11:15:00 | 106.61 | 107.31 | 107.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 12:15:00 | 105.63 | 106.98 | 107.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 107.90 | 106.33 | 106.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 107.90 | 106.33 | 106.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 107.90 | 106.33 | 106.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:30:00 | 107.87 | 106.33 | 106.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 106.94 | 106.45 | 106.76 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 14:15:00 | 107.81 | 106.99 | 106.93 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 106.61 | 106.92 | 106.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 106.46 | 106.83 | 106.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 12:15:00 | 106.70 | 106.48 | 106.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 12:15:00 | 106.70 | 106.48 | 106.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 106.70 | 106.48 | 106.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:00:00 | 106.70 | 106.48 | 106.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 107.04 | 106.59 | 106.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:45:00 | 106.90 | 106.59 | 106.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 106.90 | 106.66 | 106.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:30:00 | 106.72 | 106.66 | 106.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 106.30 | 106.58 | 106.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 106.51 | 106.58 | 106.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 106.72 | 106.61 | 106.69 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 107.56 | 106.80 | 106.77 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 15:15:00 | 106.40 | 106.78 | 106.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 12:15:00 | 105.97 | 106.56 | 106.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 14:15:00 | 106.48 | 106.45 | 106.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 14:15:00 | 106.48 | 106.45 | 106.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 106.48 | 106.45 | 106.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 106.48 | 106.45 | 106.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 105.82 | 106.33 | 106.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 106.75 | 106.33 | 106.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 107.04 | 106.47 | 106.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 107.04 | 106.47 | 106.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 107.01 | 106.58 | 106.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 107.01 | 106.58 | 106.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 106.70 | 106.62 | 106.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:30:00 | 106.62 | 106.62 | 106.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 106.55 | 106.61 | 106.62 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 14:15:00 | 106.91 | 106.67 | 106.65 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 09:15:00 | 106.47 | 106.64 | 106.64 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 106.85 | 106.68 | 106.66 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 14:15:00 | 106.27 | 106.63 | 106.65 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 107.13 | 106.72 | 106.68 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 106.19 | 106.61 | 106.65 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 106.78 | 106.59 | 106.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 108.00 | 106.98 | 106.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 107.50 | 107.64 | 107.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 107.50 | 107.64 | 107.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 107.50 | 107.64 | 107.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 10:30:00 | 109.03 | 107.76 | 107.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 106.82 | 107.63 | 107.50 | SL hit (close<static) qty=1.00 sl=107.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 13:15:00 | 107.18 | 107.43 | 107.44 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 108.19 | 107.56 | 107.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 10:15:00 | 108.56 | 107.76 | 107.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 10:15:00 | 110.72 | 111.05 | 110.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-24 11:00:00 | 110.72 | 111.05 | 110.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 111.82 | 111.69 | 110.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:30:00 | 111.34 | 111.69 | 110.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 111.17 | 111.59 | 110.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:45:00 | 110.75 | 111.59 | 110.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 110.10 | 111.29 | 110.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:00:00 | 110.10 | 111.29 | 110.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 109.56 | 110.94 | 110.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:45:00 | 109.66 | 110.94 | 110.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 14:15:00 | 108.97 | 110.41 | 110.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 10:15:00 | 107.30 | 109.23 | 109.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 12:15:00 | 107.84 | 107.28 | 108.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 13:00:00 | 107.84 | 107.28 | 108.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 108.48 | 107.52 | 108.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 108.48 | 107.52 | 108.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 107.31 | 107.48 | 108.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:45:00 | 107.20 | 107.37 | 107.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:15:00 | 106.82 | 107.35 | 107.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:15:00 | 107.09 | 107.23 | 107.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:45:00 | 107.00 | 106.56 | 107.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 107.32 | 106.71 | 107.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 107.32 | 106.71 | 107.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 108.09 | 106.99 | 107.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 108.83 | 107.76 | 107.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 108.83 | 107.76 | 107.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 110.95 | 108.76 | 108.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 108.56 | 109.45 | 109.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 108.56 | 109.45 | 109.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 108.56 | 109.45 | 109.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:00:00 | 108.56 | 109.45 | 109.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 110.68 | 109.70 | 109.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 11:15:00 | 112.76 | 109.70 | 109.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 13:15:00 | 109.04 | 109.69 | 109.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 13:15:00 | 109.04 | 109.69 | 109.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 108.37 | 109.43 | 109.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 108.75 | 108.67 | 108.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 108.75 | 108.67 | 108.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 108.75 | 108.67 | 108.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 108.98 | 108.67 | 108.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 108.85 | 108.71 | 108.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:30:00 | 108.77 | 108.67 | 108.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 11:15:00 | 108.68 | 107.40 | 107.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 108.68 | 107.40 | 107.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 12:15:00 | 109.87 | 108.56 | 108.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 10:15:00 | 110.31 | 110.56 | 109.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 11:00:00 | 110.31 | 110.56 | 109.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 111.94 | 112.60 | 111.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 111.94 | 112.60 | 111.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 111.45 | 112.37 | 111.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 112.08 | 112.37 | 111.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 09:15:00 | 113.53 | 114.28 | 114.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 113.53 | 114.28 | 114.38 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 115.68 | 114.24 | 114.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 09:15:00 | 119.48 | 115.58 | 114.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 15:15:00 | 118.19 | 118.26 | 116.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 09:15:00 | 116.86 | 118.26 | 116.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 116.53 | 117.91 | 116.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:15:00 | 116.50 | 117.91 | 116.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 117.30 | 117.79 | 116.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 11:30:00 | 117.81 | 117.72 | 116.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 12:15:00 | 117.70 | 117.72 | 116.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 15:00:00 | 117.48 | 117.55 | 117.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 113.74 | 116.72 | 116.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 113.74 | 116.72 | 116.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 113.50 | 116.08 | 116.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 113.94 | 113.81 | 114.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:00:00 | 113.94 | 113.81 | 114.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 114.22 | 113.91 | 114.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 114.40 | 113.91 | 114.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 114.16 | 113.96 | 114.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 112.17 | 113.97 | 114.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 10:45:00 | 113.60 | 113.83 | 114.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 10:15:00 | 111.51 | 110.15 | 110.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 111.51 | 110.15 | 110.09 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 109.59 | 110.25 | 110.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 108.67 | 109.51 | 109.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 107.14 | 106.38 | 107.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 10:00:00 | 107.14 | 106.38 | 107.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 106.58 | 106.42 | 107.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 107.13 | 106.42 | 107.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 107.06 | 106.55 | 107.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:30:00 | 106.72 | 106.55 | 107.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 107.20 | 106.68 | 107.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:00:00 | 107.20 | 106.68 | 107.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 107.09 | 106.76 | 107.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:15:00 | 107.21 | 106.76 | 107.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 107.36 | 106.88 | 107.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 107.36 | 106.88 | 107.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 107.08 | 106.92 | 107.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 108.33 | 106.92 | 107.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 108.26 | 107.19 | 107.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:15:00 | 108.60 | 107.19 | 107.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 108.23 | 107.40 | 107.41 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 108.69 | 107.65 | 107.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 12:15:00 | 108.99 | 107.92 | 107.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 107.81 | 108.27 | 107.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 107.81 | 108.27 | 107.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 107.81 | 108.27 | 107.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:45:00 | 108.07 | 108.27 | 107.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 107.44 | 108.10 | 107.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 107.44 | 108.10 | 107.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 106.72 | 107.83 | 107.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 106.72 | 107.83 | 107.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 106.10 | 107.48 | 107.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 13:15:00 | 105.82 | 107.15 | 107.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 09:15:00 | 108.40 | 107.26 | 107.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 108.40 | 107.26 | 107.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 108.40 | 107.26 | 107.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 108.40 | 107.26 | 107.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 108.35 | 107.48 | 107.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:30:00 | 108.29 | 107.48 | 107.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 108.06 | 107.59 | 107.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 14:15:00 | 108.45 | 107.87 | 107.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 11:15:00 | 107.49 | 107.94 | 107.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 11:15:00 | 107.49 | 107.94 | 107.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 107.49 | 107.94 | 107.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 107.49 | 107.94 | 107.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 107.33 | 107.82 | 107.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:30:00 | 107.18 | 107.82 | 107.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 14:15:00 | 107.61 | 107.71 | 107.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 106.91 | 107.53 | 107.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 15:15:00 | 106.49 | 106.20 | 106.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 09:15:00 | 105.78 | 106.20 | 106.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 106.13 | 106.18 | 106.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 105.35 | 106.10 | 106.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:30:00 | 105.33 | 105.93 | 106.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 11:15:00 | 106.00 | 105.48 | 105.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 106.00 | 105.48 | 105.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 12:15:00 | 106.41 | 105.67 | 105.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 105.68 | 105.70 | 105.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 14:15:00 | 105.68 | 105.70 | 105.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 105.68 | 105.70 | 105.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:45:00 | 105.86 | 105.70 | 105.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 105.10 | 105.58 | 105.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 105.40 | 105.58 | 105.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 105.60 | 105.58 | 105.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:30:00 | 106.03 | 105.57 | 105.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 12:15:00 | 105.00 | 105.48 | 105.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 105.00 | 105.48 | 105.49 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 105.51 | 105.45 | 105.45 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 13:15:00 | 105.37 | 105.44 | 105.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 13:15:00 | 105.04 | 105.32 | 105.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 15:15:00 | 105.35 | 105.30 | 105.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 15:15:00 | 105.35 | 105.30 | 105.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 105.35 | 105.30 | 105.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 105.68 | 105.30 | 105.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 104.46 | 105.13 | 105.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 13:00:00 | 104.16 | 104.80 | 105.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 14:00:00 | 104.28 | 104.71 | 104.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 14:45:00 | 103.98 | 104.46 | 104.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:30:00 | 104.29 | 103.71 | 103.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 104.30 | 103.83 | 103.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:15:00 | 104.62 | 103.83 | 103.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 104.81 | 104.03 | 103.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 104.81 | 104.03 | 103.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 105.32 | 104.50 | 104.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 104.99 | 105.09 | 104.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 14:15:00 | 104.99 | 105.09 | 104.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 104.99 | 105.09 | 104.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 15:00:00 | 104.99 | 105.09 | 104.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 104.95 | 105.06 | 104.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 106.05 | 105.06 | 104.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 106.43 | 105.34 | 104.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 107.19 | 105.34 | 104.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 13:15:00 | 106.92 | 106.09 | 105.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 13:15:00 | 104.56 | 105.25 | 105.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 104.56 | 105.25 | 105.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 104.14 | 104.83 | 105.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 105.01 | 104.21 | 104.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 14:15:00 | 105.01 | 104.21 | 104.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 105.01 | 104.21 | 104.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 105.01 | 104.21 | 104.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 105.00 | 104.37 | 104.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 104.22 | 104.37 | 104.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 14:45:00 | 104.64 | 104.26 | 104.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 11:15:00 | 104.00 | 102.94 | 102.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 11:15:00 | 104.00 | 102.94 | 102.80 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 102.38 | 102.91 | 102.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 102.00 | 102.64 | 102.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 100.68 | 100.52 | 101.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 100.68 | 100.52 | 101.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 100.68 | 100.52 | 101.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:30:00 | 101.16 | 100.52 | 101.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 101.00 | 100.62 | 101.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:30:00 | 100.84 | 100.62 | 101.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 102.00 | 100.90 | 101.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 102.00 | 100.90 | 101.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 105.00 | 101.72 | 101.62 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 12:15:00 | 101.50 | 101.66 | 101.66 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 102.64 | 101.81 | 101.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 10:15:00 | 104.55 | 102.36 | 101.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 105.28 | 105.32 | 103.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 105.28 | 105.32 | 103.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 105.28 | 105.32 | 103.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 14:45:00 | 107.15 | 105.75 | 104.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 103.37 | 104.85 | 104.75 | SL hit (close<static) qty=1.00 sl=103.45 alert=retest2 |

### Cycle 56 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 103.74 | 104.62 | 104.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 102.10 | 104.12 | 104.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 102.35 | 101.77 | 102.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:30:00 | 102.50 | 101.77 | 102.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 102.56 | 101.93 | 102.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:45:00 | 102.96 | 101.93 | 102.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 102.76 | 102.10 | 102.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:00:00 | 102.76 | 102.10 | 102.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 102.55 | 102.19 | 102.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:15:00 | 102.33 | 102.19 | 102.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 103.59 | 102.53 | 102.80 | SL hit (close>static) qty=1.00 sl=102.95 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 88.95 | 86.46 | 86.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 89.20 | 87.01 | 86.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 12:15:00 | 88.83 | 88.87 | 88.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 12:30:00 | 88.88 | 88.87 | 88.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 97.30 | 97.69 | 96.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 14:45:00 | 97.58 | 97.61 | 96.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 15:15:00 | 97.48 | 97.61 | 96.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:45:00 | 97.48 | 97.63 | 96.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 96.19 | 97.69 | 97.39 | SL hit (close<static) qty=1.00 sl=96.74 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 11:15:00 | 95.73 | 96.97 | 97.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 10:15:00 | 95.27 | 96.42 | 96.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 96.86 | 95.82 | 96.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 96.86 | 95.82 | 96.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 96.86 | 95.82 | 96.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 96.86 | 95.82 | 96.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 97.00 | 96.06 | 96.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 97.08 | 96.06 | 96.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 97.24 | 96.51 | 96.47 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 95.93 | 96.63 | 96.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 95.49 | 96.31 | 96.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 95.75 | 95.59 | 96.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 11:45:00 | 95.80 | 95.59 | 96.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 95.73 | 95.62 | 96.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:45:00 | 95.88 | 95.62 | 96.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 95.94 | 95.72 | 96.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 95.94 | 95.72 | 96.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 96.55 | 95.85 | 96.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:15:00 | 97.00 | 95.85 | 96.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 96.68 | 96.02 | 96.07 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 97.46 | 96.31 | 96.19 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 13:15:00 | 95.87 | 96.58 | 96.62 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 97.01 | 96.69 | 96.65 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 94.80 | 96.35 | 96.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 10:15:00 | 94.57 | 96.00 | 96.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 11:15:00 | 92.52 | 92.34 | 93.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 12:00:00 | 92.52 | 92.34 | 93.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 92.62 | 92.43 | 93.02 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 94.79 | 93.52 | 93.36 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 91.54 | 93.20 | 93.34 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 93.85 | 93.09 | 93.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 94.97 | 93.47 | 93.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 93.45 | 93.71 | 93.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 12:15:00 | 93.45 | 93.71 | 93.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 93.45 | 93.71 | 93.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 93.45 | 93.71 | 93.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 93.51 | 93.67 | 93.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:15:00 | 93.45 | 93.67 | 93.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 92.52 | 93.44 | 93.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 92.52 | 93.44 | 93.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 92.66 | 93.28 | 93.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 91.54 | 93.28 | 93.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 91.68 | 92.96 | 93.13 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 13:15:00 | 93.62 | 93.24 | 93.22 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 14:15:00 | 92.80 | 93.15 | 93.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 15:15:00 | 92.50 | 93.02 | 93.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 91.56 | 91.00 | 91.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 91.56 | 91.00 | 91.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 91.56 | 91.00 | 91.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 91.56 | 91.00 | 91.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 91.51 | 91.10 | 91.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 09:15:00 | 90.10 | 91.10 | 91.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 14:15:00 | 85.59 | 87.11 | 88.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 14:15:00 | 85.52 | 85.50 | 86.75 | SL hit (close>ema200) qty=0.50 sl=85.50 alert=retest2 |

### Cycle 71 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 85.91 | 85.39 | 85.33 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 15:15:00 | 84.15 | 85.14 | 85.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 09:15:00 | 82.07 | 84.53 | 84.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 14:15:00 | 83.19 | 83.11 | 83.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-27 15:00:00 | 83.19 | 83.11 | 83.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 82.10 | 82.91 | 83.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 81.31 | 82.91 | 83.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 84.00 | 82.90 | 82.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 84.00 | 82.90 | 82.89 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 81.91 | 82.79 | 82.90 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 83.24 | 82.73 | 82.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 83.73 | 82.93 | 82.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 14:15:00 | 88.00 | 88.03 | 87.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 14:30:00 | 87.99 | 88.03 | 87.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 97.52 | 97.35 | 96.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 13:15:00 | 98.40 | 96.95 | 96.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 14:00:00 | 98.62 | 97.28 | 96.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:30:00 | 98.39 | 97.77 | 97.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 11:15:00 | 96.69 | 97.14 | 97.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 96.69 | 97.14 | 97.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 13:15:00 | 96.21 | 96.84 | 97.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 95.62 | 95.23 | 95.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 95.62 | 95.23 | 95.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 95.62 | 95.23 | 95.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:30:00 | 95.00 | 95.50 | 95.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:15:00 | 90.25 | 93.22 | 94.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 93.80 | 92.82 | 93.40 | SL hit (close>ema200) qty=0.50 sl=92.82 alert=retest2 |

### Cycle 77 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 94.50 | 93.72 | 93.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 94.99 | 93.98 | 93.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 13:15:00 | 94.27 | 94.33 | 94.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 14:00:00 | 94.27 | 94.33 | 94.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 09:15:00 | 98.04 | 98.48 | 97.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-11 09:45:00 | 97.41 | 98.48 | 97.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 12:15:00 | 97.93 | 98.22 | 97.61 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-22 12:45:00 | 104.34 | 2025-05-23 13:15:00 | 106.84 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-05-22 13:15:00 | 104.33 | 2025-05-23 13:15:00 | 106.84 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-05-22 14:00:00 | 104.21 | 2025-05-23 13:15:00 | 106.84 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-05-23 10:15:00 | 104.25 | 2025-05-23 13:15:00 | 106.84 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-05-27 09:15:00 | 106.35 | 2025-06-04 13:15:00 | 111.55 | STOP_HIT | 1.00 | 4.89% |
| BUY | retest2 | 2025-05-27 10:15:00 | 106.69 | 2025-06-04 13:15:00 | 111.55 | STOP_HIT | 1.00 | 4.56% |
| SELL | retest2 | 2025-06-16 09:15:00 | 107.90 | 2025-06-20 15:15:00 | 102.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-16 09:15:00 | 107.90 | 2025-06-23 10:15:00 | 104.28 | STOP_HIT | 0.50 | 3.35% |
| BUY | retest2 | 2025-07-09 09:15:00 | 111.68 | 2025-07-18 09:15:00 | 115.81 | STOP_HIT | 1.00 | 3.70% |
| BUY | retest2 | 2025-07-09 10:45:00 | 111.30 | 2025-07-18 09:15:00 | 115.81 | STOP_HIT | 1.00 | 4.05% |
| SELL | retest2 | 2025-07-29 10:00:00 | 108.65 | 2025-08-01 11:15:00 | 103.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-29 10:00:00 | 108.65 | 2025-08-04 12:15:00 | 103.24 | STOP_HIT | 0.50 | 4.98% |
| BUY | retest2 | 2025-09-18 10:30:00 | 109.03 | 2025-09-19 09:15:00 | 106.82 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-09-30 09:45:00 | 107.20 | 2025-10-01 14:15:00 | 108.83 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-09-30 11:15:00 | 106.82 | 2025-10-01 14:15:00 | 108.83 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-09-30 13:15:00 | 107.09 | 2025-10-01 14:15:00 | 108.83 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-10-01 09:45:00 | 107.00 | 2025-10-01 14:15:00 | 108.83 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-10-07 11:15:00 | 112.76 | 2025-10-08 13:15:00 | 109.04 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2025-10-10 11:30:00 | 108.77 | 2025-10-15 11:15:00 | 108.68 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2025-10-24 09:15:00 | 112.08 | 2025-10-30 09:15:00 | 113.53 | STOP_HIT | 1.00 | 1.29% |
| BUY | retest2 | 2025-11-04 11:30:00 | 117.81 | 2025-11-06 09:15:00 | 113.74 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2025-11-04 12:15:00 | 117.70 | 2025-11-06 09:15:00 | 113.74 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2025-11-04 15:00:00 | 117.48 | 2025-11-06 09:15:00 | 113.74 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-11-10 09:15:00 | 112.17 | 2025-11-17 10:15:00 | 111.51 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2025-11-10 10:45:00 | 113.60 | 2025-11-17 10:15:00 | 111.51 | STOP_HIT | 1.00 | 1.84% |
| SELL | retest2 | 2025-12-05 09:15:00 | 105.35 | 2025-12-10 11:15:00 | 106.00 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-12-08 09:30:00 | 105.33 | 2025-12-10 11:15:00 | 106.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-12-11 10:30:00 | 106.03 | 2025-12-11 12:15:00 | 105.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-12-16 13:00:00 | 104.16 | 2025-12-22 11:15:00 | 104.81 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-12-17 14:00:00 | 104.28 | 2025-12-22 11:15:00 | 104.81 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-12-17 14:45:00 | 103.98 | 2025-12-22 11:15:00 | 104.81 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-12-22 09:30:00 | 104.29 | 2025-12-22 11:15:00 | 104.81 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-12-24 10:15:00 | 107.19 | 2025-12-26 13:15:00 | 104.56 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-12-24 13:15:00 | 106.92 | 2025-12-26 13:15:00 | 104.56 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-12-30 09:15:00 | 104.22 | 2026-01-07 11:15:00 | 104.00 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-12-30 14:45:00 | 104.64 | 2026-01-07 11:15:00 | 104.00 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2026-01-19 14:45:00 | 107.15 | 2026-01-20 14:15:00 | 103.37 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2026-01-22 13:15:00 | 102.33 | 2026-01-22 14:15:00 | 103.59 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-01-23 11:00:00 | 102.20 | 2026-01-27 09:15:00 | 91.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-11 14:45:00 | 97.58 | 2026-02-13 09:15:00 | 96.19 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-02-11 15:15:00 | 97.48 | 2026-02-13 09:15:00 | 96.19 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2026-02-12 09:45:00 | 97.48 | 2026-02-13 09:15:00 | 96.19 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-03-17 09:15:00 | 90.10 | 2026-03-19 14:15:00 | 85.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 09:15:00 | 90.10 | 2026-03-20 14:15:00 | 85.52 | STOP_HIT | 0.50 | 5.08% |
| SELL | retest2 | 2026-03-30 09:15:00 | 81.31 | 2026-04-01 12:15:00 | 84.00 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2026-04-21 13:15:00 | 98.40 | 2026-04-23 11:15:00 | 96.69 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-04-21 14:00:00 | 98.62 | 2026-04-23 11:15:00 | 96.69 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2026-04-22 09:30:00 | 98.39 | 2026-04-23 11:15:00 | 96.69 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2026-04-27 14:30:00 | 95.00 | 2026-04-30 10:15:00 | 90.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 14:30:00 | 95.00 | 2026-05-04 09:15:00 | 93.80 | STOP_HIT | 0.50 | 1.26% |
