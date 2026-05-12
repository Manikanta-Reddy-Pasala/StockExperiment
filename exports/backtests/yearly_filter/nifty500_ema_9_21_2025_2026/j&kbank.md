# Jammu & Kashmir Bank Ltd. (J&KBANK)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 141.24
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 87 |
| ALERT1 | 58 |
| ALERT2 | 58 |
| ALERT2_SKIP | 29 |
| ALERT3 | 157 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 45 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 50 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 38
- **Target hits / Stop hits / Partials:** 1 / 47 / 2
- **Avg / median % per leg:** -0.36% / -1.07%
- **Sum % (uncompounded):** -18.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 8 | 47.1% | 1 | 16 | 0 | 0.37% | 6.4% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.81% | -1.6% |
| BUY @ 3rd Alert (retest2) | 15 | 8 | 53.3% | 1 | 14 | 0 | 0.53% | 8.0% |
| SELL (all) | 33 | 4 | 12.1% | 0 | 31 | 2 | -0.74% | -24.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.00% | -1.0% |
| SELL @ 3rd Alert (retest2) | 32 | 4 | 12.5% | 0 | 30 | 2 | -0.74% | -23.5% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.88% | -2.6% |
| retest2 (combined) | 47 | 12 | 25.5% | 1 | 44 | 2 | -0.33% | -15.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 95.15 | 92.37 | 92.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 96.60 | 93.22 | 92.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 11:15:00 | 100.86 | 101.01 | 99.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 12:00:00 | 100.86 | 101.01 | 99.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 104.07 | 103.95 | 103.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 103.54 | 103.95 | 103.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 102.80 | 103.62 | 103.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:00:00 | 102.80 | 103.62 | 103.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 102.00 | 103.29 | 103.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 102.00 | 103.29 | 103.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 101.53 | 102.94 | 102.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 101.53 | 102.94 | 102.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 101.75 | 102.70 | 102.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 10:15:00 | 100.87 | 101.69 | 102.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 100.88 | 100.83 | 101.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 11:00:00 | 100.88 | 100.83 | 101.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 101.27 | 100.94 | 101.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:30:00 | 101.46 | 100.94 | 101.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 101.30 | 101.01 | 101.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:30:00 | 101.15 | 101.01 | 101.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 101.30 | 101.07 | 101.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:45:00 | 101.40 | 101.07 | 101.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 101.10 | 101.07 | 101.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:15:00 | 101.90 | 101.07 | 101.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 101.32 | 101.12 | 101.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 10:30:00 | 101.12 | 101.15 | 101.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:15:00 | 101.05 | 101.15 | 101.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 12:15:00 | 102.20 | 100.46 | 100.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 12:15:00 | 102.20 | 100.46 | 100.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 13:15:00 | 103.23 | 101.01 | 100.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 11:15:00 | 104.65 | 105.03 | 103.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 11:30:00 | 104.56 | 105.03 | 103.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 104.12 | 104.59 | 103.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:15:00 | 103.59 | 104.59 | 103.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 103.59 | 104.39 | 103.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:15:00 | 104.08 | 104.39 | 103.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 104.66 | 104.44 | 103.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 10:30:00 | 105.10 | 104.59 | 104.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 11:30:00 | 105.16 | 104.72 | 104.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 12:00:00 | 105.25 | 104.72 | 104.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 14:45:00 | 105.20 | 104.87 | 104.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 104.32 | 104.79 | 104.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:00:00 | 104.32 | 104.79 | 104.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 104.24 | 104.68 | 104.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 10:15:00 | 105.27 | 104.54 | 104.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 107.17 | 109.04 | 109.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 107.17 | 109.04 | 109.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 105.50 | 107.83 | 108.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 105.30 | 105.15 | 106.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:30:00 | 105.41 | 105.15 | 106.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 105.16 | 105.07 | 105.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 105.96 | 105.07 | 105.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 104.45 | 104.01 | 104.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:45:00 | 104.42 | 104.01 | 104.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 103.00 | 102.27 | 103.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 103.00 | 102.27 | 103.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 102.95 | 102.39 | 103.02 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 104.24 | 103.26 | 103.26 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 09:15:00 | 101.93 | 103.20 | 103.24 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 106.66 | 103.73 | 103.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 108.75 | 104.73 | 103.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 108.68 | 108.75 | 107.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 11:00:00 | 108.68 | 108.75 | 107.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 107.90 | 108.58 | 107.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:30:00 | 107.14 | 108.58 | 107.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 113.95 | 114.51 | 113.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 113.60 | 114.51 | 113.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 114.30 | 114.47 | 113.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:15:00 | 113.99 | 114.47 | 113.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 113.99 | 114.37 | 113.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 113.34 | 114.37 | 113.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 113.59 | 114.22 | 113.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:45:00 | 112.95 | 114.22 | 113.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 113.22 | 114.02 | 113.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:00:00 | 113.22 | 114.02 | 113.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 13:15:00 | 112.55 | 113.40 | 113.51 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 14:15:00 | 114.19 | 113.46 | 113.43 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 10:15:00 | 112.91 | 113.40 | 113.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 11:15:00 | 111.67 | 113.05 | 113.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 09:15:00 | 112.41 | 112.20 | 112.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 09:15:00 | 112.41 | 112.20 | 112.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 112.41 | 112.20 | 112.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:45:00 | 112.35 | 112.15 | 112.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 15:00:00 | 112.03 | 112.01 | 112.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:30:00 | 112.26 | 112.06 | 112.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 113.40 | 112.41 | 112.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 113.40 | 112.41 | 112.37 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 12:15:00 | 111.34 | 112.29 | 112.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 111.11 | 111.77 | 112.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 111.81 | 110.26 | 110.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 111.81 | 110.26 | 110.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 111.81 | 110.26 | 110.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 111.40 | 110.26 | 110.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 112.02 | 110.61 | 111.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:00:00 | 112.02 | 110.61 | 111.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 110.41 | 110.63 | 110.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:45:00 | 110.30 | 110.58 | 110.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 111.66 | 110.80 | 110.98 | SL hit (close>static) qty=1.00 sl=111.24 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 114.49 | 111.77 | 111.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 15:15:00 | 115.50 | 114.05 | 113.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 10:15:00 | 114.10 | 114.13 | 113.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 10:30:00 | 113.91 | 114.13 | 113.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 114.01 | 114.08 | 113.45 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 112.35 | 113.19 | 113.26 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 13:15:00 | 114.01 | 113.34 | 113.32 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 112.43 | 113.31 | 113.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 111.54 | 112.45 | 112.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 13:15:00 | 110.20 | 110.10 | 110.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 14:00:00 | 110.20 | 110.10 | 110.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 112.01 | 110.48 | 111.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 112.01 | 110.48 | 111.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 111.84 | 110.75 | 111.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 111.61 | 110.75 | 111.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 112.19 | 111.19 | 111.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:45:00 | 111.85 | 111.19 | 111.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 14:15:00 | 111.70 | 111.30 | 111.27 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 110.55 | 111.14 | 111.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 11:15:00 | 110.06 | 110.93 | 111.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 105.64 | 105.54 | 107.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 105.64 | 105.54 | 107.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 106.20 | 105.78 | 107.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 106.15 | 105.78 | 107.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 105.60 | 104.97 | 105.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:45:00 | 105.62 | 104.97 | 105.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 104.80 | 104.93 | 105.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:30:00 | 105.25 | 104.93 | 105.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 103.25 | 102.74 | 103.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 103.40 | 102.74 | 103.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 102.79 | 102.75 | 103.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:45:00 | 102.37 | 102.67 | 103.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 13:15:00 | 102.25 | 102.62 | 103.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:00:00 | 102.50 | 101.79 | 101.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 10:15:00 | 102.73 | 101.98 | 101.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 10:15:00 | 102.73 | 101.98 | 101.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 12:15:00 | 102.87 | 102.29 | 102.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 14:15:00 | 102.05 | 102.31 | 102.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 14:15:00 | 102.05 | 102.31 | 102.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 102.05 | 102.31 | 102.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 102.05 | 102.31 | 102.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 101.51 | 102.15 | 102.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 102.62 | 102.15 | 102.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 12:15:00 | 104.34 | 104.60 | 104.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 12:15:00 | 104.34 | 104.60 | 104.63 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 11:15:00 | 104.71 | 104.62 | 104.61 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 104.20 | 104.53 | 104.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 103.60 | 104.29 | 104.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 100.36 | 99.90 | 100.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:45:00 | 100.53 | 99.90 | 100.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 100.35 | 99.99 | 100.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 100.69 | 99.99 | 100.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 100.12 | 99.61 | 100.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 100.12 | 99.61 | 100.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 100.30 | 99.75 | 100.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 100.30 | 99.75 | 100.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 100.02 | 99.80 | 100.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:15:00 | 100.49 | 99.80 | 100.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 100.49 | 99.94 | 100.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 100.51 | 99.94 | 100.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 102.20 | 100.39 | 100.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 103.02 | 100.92 | 100.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 101.62 | 102.36 | 101.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 101.62 | 102.36 | 101.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 101.62 | 102.36 | 101.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:45:00 | 101.92 | 102.36 | 101.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 101.64 | 102.21 | 101.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 101.67 | 102.21 | 101.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 100.88 | 101.66 | 101.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 100.00 | 101.33 | 101.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 100.94 | 100.86 | 101.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 100.94 | 100.86 | 101.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 100.94 | 100.86 | 101.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:45:00 | 101.00 | 100.86 | 101.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 101.15 | 100.92 | 101.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 101.15 | 100.92 | 101.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 101.24 | 100.98 | 101.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 100.69 | 100.98 | 101.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 101.30 | 101.04 | 101.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 101.30 | 101.04 | 101.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 100.90 | 101.02 | 101.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:45:00 | 100.50 | 100.80 | 101.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:45:00 | 100.49 | 100.67 | 100.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 101.49 | 100.50 | 100.61 | SL hit (close>static) qty=1.00 sl=101.35 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 101.87 | 100.77 | 100.73 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 13:15:00 | 101.08 | 101.23 | 101.24 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 14:15:00 | 101.28 | 101.24 | 101.24 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 15:15:00 | 100.95 | 101.18 | 101.22 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 101.79 | 101.31 | 101.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 102.45 | 101.99 | 101.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 13:15:00 | 102.00 | 102.03 | 101.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 14:15:00 | 102.00 | 102.03 | 101.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 101.90 | 102.00 | 101.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 102.50 | 102.01 | 101.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 103.50 | 104.43 | 104.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 103.50 | 104.43 | 104.47 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 13:15:00 | 105.38 | 104.51 | 104.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 09:15:00 | 106.11 | 105.07 | 104.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 10:15:00 | 104.88 | 105.03 | 104.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 10:15:00 | 104.88 | 105.03 | 104.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 104.88 | 105.03 | 104.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 104.88 | 105.03 | 104.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 104.46 | 104.92 | 104.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 104.46 | 104.92 | 104.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 104.46 | 104.82 | 104.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:00:00 | 104.46 | 104.82 | 104.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 104.38 | 104.74 | 104.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:45:00 | 104.37 | 104.74 | 104.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 14:15:00 | 104.33 | 104.65 | 104.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 103.98 | 104.45 | 104.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 101.05 | 100.73 | 101.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 101.05 | 100.73 | 101.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 101.41 | 100.87 | 101.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:00:00 | 100.78 | 100.85 | 101.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 100.16 | 100.91 | 101.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 103.74 | 101.49 | 101.76 | SL hit (close>static) qty=1.00 sl=102.01 alert=retest2 |

### Cycle 33 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 104.02 | 102.00 | 101.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 15:15:00 | 105.10 | 103.91 | 103.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 10:15:00 | 103.66 | 103.95 | 103.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 10:45:00 | 103.75 | 103.95 | 103.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 104.30 | 104.80 | 104.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 104.43 | 104.80 | 104.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 104.20 | 104.68 | 104.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:00:00 | 104.20 | 104.68 | 104.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 104.57 | 104.66 | 104.35 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 104.12 | 104.26 | 104.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 103.15 | 104.02 | 104.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 11:15:00 | 102.94 | 102.58 | 103.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 11:15:00 | 102.94 | 102.58 | 103.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 102.94 | 102.58 | 103.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:45:00 | 103.01 | 102.58 | 103.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 102.85 | 102.63 | 103.06 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 105.05 | 103.28 | 103.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 10:15:00 | 105.56 | 103.73 | 103.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 12:15:00 | 105.90 | 105.98 | 105.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 13:00:00 | 105.90 | 105.98 | 105.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 104.70 | 105.80 | 105.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 104.70 | 105.80 | 105.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 104.09 | 105.46 | 105.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 104.09 | 105.46 | 105.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 12:15:00 | 103.63 | 104.82 | 104.95 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 105.70 | 104.81 | 104.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 14:15:00 | 106.83 | 106.02 | 105.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 105.90 | 106.12 | 105.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 09:45:00 | 106.22 | 106.12 | 105.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 106.73 | 106.24 | 105.77 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 104.45 | 105.49 | 105.52 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 108.00 | 105.91 | 105.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 109.91 | 107.40 | 106.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 13:15:00 | 107.50 | 107.68 | 107.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 14:00:00 | 107.50 | 107.68 | 107.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 106.96 | 107.54 | 107.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 106.96 | 107.54 | 107.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 107.30 | 107.49 | 107.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 106.76 | 107.49 | 107.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 106.40 | 107.27 | 106.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:00:00 | 106.40 | 107.27 | 106.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 106.03 | 107.02 | 106.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:45:00 | 106.00 | 107.02 | 106.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 105.87 | 106.66 | 106.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 105.65 | 106.46 | 106.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 106.26 | 106.19 | 106.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 106.26 | 106.19 | 106.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 106.26 | 106.19 | 106.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:30:00 | 106.30 | 106.19 | 106.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 106.25 | 106.20 | 106.43 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 14:15:00 | 106.94 | 106.54 | 106.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 15:15:00 | 108.00 | 106.83 | 106.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 107.58 | 108.19 | 107.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 107.58 | 108.19 | 107.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 107.58 | 108.19 | 107.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 107.58 | 108.19 | 107.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 107.33 | 108.02 | 107.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 107.33 | 108.02 | 107.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 107.77 | 107.97 | 107.63 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 106.41 | 107.50 | 107.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 11:15:00 | 106.24 | 107.10 | 107.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 106.55 | 106.44 | 106.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 106.55 | 106.44 | 106.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 106.55 | 106.44 | 106.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:45:00 | 106.50 | 106.44 | 106.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 108.14 | 106.70 | 106.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:45:00 | 108.09 | 106.70 | 106.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 107.91 | 106.94 | 107.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:45:00 | 108.15 | 106.94 | 107.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 13:15:00 | 107.88 | 107.13 | 107.08 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 106.06 | 106.92 | 106.98 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 10:15:00 | 107.34 | 107.05 | 107.02 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 105.91 | 107.00 | 107.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 105.57 | 106.71 | 106.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 103.42 | 103.28 | 104.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:00:00 | 103.42 | 103.28 | 104.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 104.71 | 103.64 | 104.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 104.71 | 103.64 | 104.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 105.41 | 103.99 | 104.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:00:00 | 105.41 | 103.99 | 104.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 107.09 | 105.00 | 104.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 14:15:00 | 107.59 | 106.41 | 105.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 15:15:00 | 107.17 | 107.26 | 106.62 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:15:00 | 107.61 | 107.26 | 106.62 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 11:45:00 | 107.80 | 107.20 | 106.76 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 107.00 | 107.17 | 106.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 107.00 | 107.17 | 106.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 106.83 | 107.26 | 106.99 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-13 10:15:00 | 106.83 | 107.26 | 106.99 | SL hit (close<ema400) qty=1.00 sl=106.99 alert=retest1 |

### Cycle 48 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 106.25 | 106.85 | 106.85 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 14:15:00 | 107.39 | 106.78 | 106.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 109.02 | 107.30 | 107.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 15:15:00 | 107.90 | 108.16 | 107.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 09:15:00 | 107.91 | 108.16 | 107.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 107.28 | 107.98 | 107.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 107.28 | 107.98 | 107.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 108.04 | 107.99 | 107.68 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 107.00 | 107.57 | 107.58 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 10:15:00 | 107.82 | 107.62 | 107.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 12:15:00 | 108.62 | 107.92 | 107.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 09:15:00 | 107.99 | 108.37 | 108.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 107.99 | 108.37 | 108.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 107.99 | 108.37 | 108.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 107.99 | 108.37 | 108.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 107.62 | 108.22 | 108.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 107.62 | 108.22 | 108.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 107.40 | 108.06 | 107.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:45:00 | 107.50 | 108.06 | 107.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 12:15:00 | 106.86 | 107.82 | 107.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 105.88 | 107.30 | 107.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 105.98 | 105.95 | 106.63 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 14:00:00 | 104.62 | 105.46 | 106.17 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 105.67 | 104.95 | 105.44 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-25 13:15:00 | 105.67 | 104.95 | 105.44 | SL hit (close>ema400) qty=1.00 sl=105.44 alert=retest1 |

### Cycle 53 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 107.39 | 105.87 | 105.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 14:15:00 | 108.18 | 107.25 | 106.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 106.35 | 107.21 | 106.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 106.35 | 107.21 | 106.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 106.35 | 107.21 | 106.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 106.35 | 107.21 | 106.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 107.07 | 107.18 | 106.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 11:45:00 | 107.45 | 107.19 | 106.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:30:00 | 107.53 | 107.28 | 106.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 14:00:00 | 107.55 | 107.28 | 106.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 108.10 | 107.19 | 106.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 107.05 | 107.34 | 107.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 107.05 | 107.34 | 107.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 107.27 | 107.33 | 107.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 108.94 | 107.28 | 107.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 105.18 | 107.12 | 107.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 105.18 | 107.12 | 107.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 104.05 | 106.50 | 106.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 101.38 | 101.04 | 102.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:00:00 | 101.38 | 101.04 | 102.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 101.73 | 101.33 | 102.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:15:00 | 101.11 | 101.75 | 102.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:45:00 | 101.00 | 101.51 | 101.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 14:15:00 | 102.40 | 101.79 | 101.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 102.40 | 101.79 | 101.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 102.90 | 102.05 | 101.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 101.76 | 102.21 | 102.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 101.76 | 102.21 | 102.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 101.76 | 102.21 | 102.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 101.66 | 102.21 | 102.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 101.06 | 101.98 | 101.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 14:15:00 | 100.68 | 101.36 | 101.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 100.53 | 100.16 | 100.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 14:15:00 | 100.53 | 100.16 | 100.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 100.53 | 100.16 | 100.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 100.53 | 100.16 | 100.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 100.09 | 100.14 | 100.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 100.14 | 100.14 | 100.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 99.69 | 100.05 | 100.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:00:00 | 98.99 | 99.72 | 100.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 14:15:00 | 100.18 | 99.73 | 99.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 14:15:00 | 100.18 | 99.73 | 99.72 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 12:15:00 | 99.48 | 99.81 | 99.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 99.38 | 99.66 | 99.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 12:15:00 | 98.55 | 98.27 | 98.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 12:15:00 | 98.55 | 98.27 | 98.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 98.55 | 98.27 | 98.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:45:00 | 98.48 | 98.27 | 98.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 98.70 | 98.36 | 98.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:45:00 | 98.49 | 98.36 | 98.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 99.22 | 98.53 | 98.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 99.22 | 98.53 | 98.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 99.27 | 98.68 | 98.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 99.93 | 98.68 | 98.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 100.40 | 99.02 | 98.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 100.59 | 99.57 | 99.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 102.60 | 102.78 | 101.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 13:45:00 | 102.80 | 102.78 | 101.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 104.13 | 104.39 | 103.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:30:00 | 104.50 | 104.35 | 103.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 103.71 | 104.20 | 103.88 | SL hit (close<static) qty=1.00 sl=103.79 alert=retest2 |

### Cycle 60 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 102.02 | 103.47 | 103.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 101.47 | 102.31 | 102.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 100.40 | 100.27 | 101.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 100.93 | 100.27 | 101.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 100.61 | 100.34 | 101.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 100.12 | 100.22 | 100.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 101.60 | 100.58 | 100.74 | SL hit (close>static) qty=1.00 sl=101.46 alert=retest2 |

### Cycle 61 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 102.84 | 101.13 | 100.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 15:15:00 | 102.90 | 101.91 | 101.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 10:15:00 | 101.79 | 102.50 | 102.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 10:15:00 | 101.79 | 102.50 | 102.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 101.79 | 102.50 | 102.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:00:00 | 101.79 | 102.50 | 102.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 102.20 | 102.44 | 102.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:30:00 | 102.16 | 102.44 | 102.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 102.58 | 102.61 | 102.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:00:00 | 102.58 | 102.61 | 102.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 102.01 | 102.49 | 102.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 102.01 | 102.49 | 102.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 101.44 | 102.28 | 102.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 101.44 | 102.28 | 102.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 100.76 | 101.98 | 102.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 99.45 | 101.47 | 101.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 101.58 | 101.04 | 101.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 101.58 | 101.04 | 101.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 101.58 | 101.04 | 101.52 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 12:15:00 | 103.21 | 101.93 | 101.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 09:15:00 | 106.88 | 103.29 | 102.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 12:15:00 | 105.27 | 106.10 | 105.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 13:00:00 | 105.27 | 106.10 | 105.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 104.45 | 105.77 | 104.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 104.45 | 105.77 | 104.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 104.35 | 105.48 | 104.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:15:00 | 104.10 | 105.48 | 104.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 103.77 | 104.81 | 104.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:45:00 | 103.08 | 104.81 | 104.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 103.86 | 104.62 | 104.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 103.41 | 104.38 | 104.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 104.16 | 104.01 | 104.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 104.16 | 104.01 | 104.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 104.16 | 104.01 | 104.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 104.30 | 104.01 | 104.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 103.61 | 103.93 | 104.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 10:15:00 | 103.39 | 103.93 | 104.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 12:15:00 | 104.90 | 104.18 | 104.27 | SL hit (close>static) qty=1.00 sl=104.78 alert=retest2 |

### Cycle 65 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 105.15 | 104.43 | 104.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 10:15:00 | 105.48 | 104.84 | 104.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 11:15:00 | 104.70 | 104.81 | 104.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 12:00:00 | 104.70 | 104.81 | 104.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 104.49 | 104.75 | 104.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 13:00:00 | 104.49 | 104.75 | 104.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 104.42 | 104.68 | 104.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:00:00 | 104.42 | 104.68 | 104.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 104.27 | 104.60 | 104.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 15:00:00 | 104.27 | 104.60 | 104.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 104.28 | 104.54 | 104.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:15:00 | 102.89 | 104.54 | 104.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 104.49 | 104.62 | 104.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:30:00 | 104.57 | 104.62 | 104.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 105.00 | 104.69 | 104.61 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 10:15:00 | 104.18 | 104.56 | 104.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 103.00 | 104.05 | 104.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 101.80 | 101.19 | 102.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 101.80 | 101.19 | 102.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 102.29 | 101.41 | 102.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 102.69 | 101.41 | 102.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 102.46 | 101.62 | 102.32 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 103.17 | 102.63 | 102.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 103.65 | 102.91 | 102.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 104.34 | 104.49 | 103.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 104.34 | 104.49 | 103.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 104.34 | 104.49 | 103.79 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 102.42 | 103.67 | 103.73 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 105.25 | 103.85 | 103.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 10:15:00 | 105.83 | 105.18 | 104.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 105.37 | 105.44 | 104.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 104.61 | 105.44 | 104.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 103.80 | 105.11 | 104.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 103.80 | 105.11 | 104.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 104.38 | 104.96 | 104.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:15:00 | 104.76 | 104.80 | 104.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 104.10 | 104.66 | 104.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 104.10 | 104.66 | 104.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 103.77 | 104.48 | 104.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 13:15:00 | 104.43 | 104.38 | 104.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 14:00:00 | 104.43 | 104.38 | 104.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 103.49 | 104.20 | 104.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 102.35 | 104.09 | 104.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 10:15:00 | 104.70 | 103.01 | 102.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 104.70 | 103.01 | 102.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 105.76 | 104.45 | 103.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 105.00 | 105.08 | 104.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 10:15:00 | 105.00 | 105.08 | 104.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 105.00 | 105.08 | 104.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 104.79 | 105.08 | 104.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 104.55 | 104.98 | 104.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 104.55 | 104.98 | 104.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 104.61 | 104.90 | 104.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:30:00 | 104.50 | 104.90 | 104.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 103.86 | 104.69 | 104.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 103.86 | 104.69 | 104.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 103.58 | 104.47 | 104.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 103.58 | 104.47 | 104.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 103.05 | 104.19 | 104.31 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 106.45 | 104.55 | 104.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 15:15:00 | 106.80 | 105.70 | 105.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 119.71 | 120.23 | 117.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 119.71 | 120.23 | 117.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 119.71 | 120.23 | 117.47 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2026-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 11:15:00 | 113.55 | 117.07 | 117.38 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 10:15:00 | 119.02 | 117.26 | 117.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 12:15:00 | 119.14 | 117.85 | 117.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 118.79 | 119.83 | 119.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 14:15:00 | 118.79 | 119.83 | 119.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 118.79 | 119.83 | 119.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 118.79 | 119.83 | 119.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 118.40 | 119.54 | 119.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 110.86 | 119.54 | 119.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 109.87 | 117.61 | 118.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 10:15:00 | 109.35 | 115.96 | 117.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 115.02 | 113.38 | 115.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 115.02 | 113.38 | 115.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 115.02 | 113.38 | 115.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:00:00 | 115.02 | 113.38 | 115.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 117.62 | 114.23 | 115.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:00:00 | 117.62 | 114.23 | 115.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 117.00 | 114.78 | 115.48 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 120.63 | 116.48 | 116.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 120.80 | 117.35 | 116.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 122.75 | 125.79 | 123.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 122.75 | 125.79 | 123.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 122.75 | 125.79 | 123.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 122.75 | 125.79 | 123.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 122.14 | 125.06 | 123.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:45:00 | 122.40 | 125.06 | 123.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 120.65 | 123.69 | 123.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 13:00:00 | 120.65 | 123.69 | 123.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2026-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 14:15:00 | 120.76 | 122.92 | 122.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 120.28 | 122.02 | 122.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 13:15:00 | 121.20 | 120.13 | 121.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 13:15:00 | 121.20 | 120.13 | 121.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 121.20 | 120.13 | 121.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:00:00 | 121.20 | 120.13 | 121.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 123.75 | 120.85 | 121.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 09:15:00 | 120.63 | 121.17 | 121.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 10:15:00 | 120.98 | 121.18 | 121.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:00:00 | 120.90 | 121.12 | 121.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 120.78 | 120.80 | 121.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 121.38 | 120.91 | 121.24 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 123.67 | 121.83 | 121.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 123.67 | 121.83 | 121.60 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 120.51 | 121.98 | 121.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 119.47 | 121.48 | 121.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 121.58 | 120.70 | 121.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 121.58 | 120.70 | 121.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 121.58 | 120.70 | 121.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 122.76 | 120.70 | 121.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 121.78 | 120.92 | 121.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 121.70 | 120.92 | 121.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 119.80 | 120.69 | 121.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 119.40 | 120.69 | 121.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:30:00 | 119.25 | 120.13 | 120.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 113.43 | 117.92 | 119.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 113.29 | 117.92 | 119.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 10:15:00 | 112.10 | 111.92 | 114.79 | SL hit (close>ema200) qty=0.50 sl=111.92 alert=retest2 |

### Cycle 81 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 118.73 | 115.21 | 115.17 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 113.90 | 115.78 | 115.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 112.42 | 114.44 | 115.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 112.94 | 111.94 | 113.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 112.94 | 111.94 | 113.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 112.94 | 111.94 | 113.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 112.06 | 111.94 | 113.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 110.86 | 113.53 | 113.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 14:30:00 | 112.20 | 112.51 | 112.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:00:00 | 112.62 | 112.81 | 112.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 113.39 | 112.92 | 112.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:45:00 | 113.45 | 112.92 | 112.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 114.55 | 113.25 | 113.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 114.55 | 113.25 | 113.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 115.25 | 113.65 | 113.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 11:15:00 | 122.70 | 122.88 | 120.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 12:00:00 | 122.70 | 122.88 | 120.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 120.13 | 122.46 | 121.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 121.00 | 122.46 | 121.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 09:15:00 | 133.10 | 130.09 | 127.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 130.83 | 132.03 | 132.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 12:15:00 | 128.97 | 131.42 | 131.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 126.47 | 126.31 | 127.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 13:15:00 | 126.58 | 126.31 | 127.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 129.55 | 127.03 | 127.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 15:00:00 | 129.55 | 127.03 | 127.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 128.31 | 127.29 | 127.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:15:00 | 127.33 | 127.29 | 127.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 12:15:00 | 129.39 | 128.21 | 128.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 12:15:00 | 129.39 | 128.21 | 128.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 133.12 | 129.29 | 128.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 130.03 | 132.25 | 130.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 130.03 | 132.25 | 130.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 130.03 | 132.25 | 130.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 130.03 | 132.25 | 130.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 130.16 | 131.83 | 130.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:30:00 | 130.21 | 131.83 | 130.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 15:15:00 | 129.85 | 130.36 | 130.41 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 131.20 | 130.53 | 130.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 11:15:00 | 132.00 | 131.03 | 130.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 09:15:00 | 133.07 | 134.23 | 133.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 133.07 | 134.23 | 133.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 133.07 | 134.23 | 133.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:30:00 | 132.66 | 134.23 | 133.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 130.15 | 133.41 | 132.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:00:00 | 130.15 | 133.41 | 132.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 130.37 | 132.80 | 132.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:45:00 | 129.95 | 132.80 | 132.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-26 10:30:00 | 101.12 | 2025-05-30 12:15:00 | 102.20 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-05-26 11:15:00 | 101.05 | 2025-05-30 12:15:00 | 102.20 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-06-04 10:30:00 | 105.10 | 2025-06-12 13:15:00 | 107.17 | STOP_HIT | 1.00 | 1.97% |
| BUY | retest2 | 2025-06-04 11:30:00 | 105.16 | 2025-06-12 13:15:00 | 107.17 | STOP_HIT | 1.00 | 1.91% |
| BUY | retest2 | 2025-06-04 12:00:00 | 105.25 | 2025-06-12 13:15:00 | 107.17 | STOP_HIT | 1.00 | 1.82% |
| BUY | retest2 | 2025-06-04 14:45:00 | 105.20 | 2025-06-12 13:15:00 | 107.17 | STOP_HIT | 1.00 | 1.87% |
| BUY | retest2 | 2025-06-06 10:15:00 | 105.27 | 2025-06-12 13:15:00 | 107.17 | STOP_HIT | 1.00 | 1.80% |
| SELL | retest2 | 2025-07-08 10:45:00 | 112.35 | 2025-07-10 09:15:00 | 113.40 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-07-08 15:00:00 | 112.03 | 2025-07-10 09:15:00 | 113.40 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-07-09 09:30:00 | 112.26 | 2025-07-10 09:15:00 | 113.40 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-07-14 13:45:00 | 110.30 | 2025-07-14 14:15:00 | 111.66 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-08-05 10:45:00 | 102.37 | 2025-08-08 10:15:00 | 102.73 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-08-05 13:15:00 | 102.25 | 2025-08-08 10:15:00 | 102.73 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-08-08 10:00:00 | 102.50 | 2025-08-08 10:15:00 | 102.73 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-08-11 09:15:00 | 102.62 | 2025-08-20 12:15:00 | 104.34 | STOP_HIT | 1.00 | 1.68% |
| SELL | retest2 | 2025-09-08 13:45:00 | 100.50 | 2025-09-10 09:15:00 | 101.49 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-09-08 14:45:00 | 100.49 | 2025-09-10 09:15:00 | 101.49 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-09-17 09:15:00 | 102.50 | 2025-09-23 10:15:00 | 103.50 | STOP_HIT | 1.00 | 0.98% |
| SELL | retest2 | 2025-09-29 12:00:00 | 100.78 | 2025-09-30 09:15:00 | 103.74 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2025-09-29 15:00:00 | 100.16 | 2025-09-30 09:15:00 | 103.74 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest1 | 2025-11-12 09:15:00 | 107.61 | 2025-11-13 10:15:00 | 106.83 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest1 | 2025-11-12 11:45:00 | 107.80 | 2025-11-13 10:15:00 | 106.83 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest1 | 2025-11-24 14:00:00 | 104.62 | 2025-11-25 13:15:00 | 105.67 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-11-28 11:45:00 | 107.45 | 2025-12-03 09:15:00 | 105.18 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-11-28 13:30:00 | 107.53 | 2025-12-03 09:15:00 | 105.18 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-11-28 14:00:00 | 107.55 | 2025-12-03 09:15:00 | 105.18 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-12-01 09:15:00 | 108.10 | 2025-12-03 09:15:00 | 105.18 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-12-02 09:15:00 | 108.94 | 2025-12-03 09:15:00 | 105.18 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2025-12-10 12:15:00 | 101.11 | 2025-12-11 14:15:00 | 102.40 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-12-10 13:45:00 | 101.00 | 2025-12-11 14:15:00 | 102.40 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-12-19 11:00:00 | 98.99 | 2025-12-22 14:15:00 | 100.18 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-01-07 14:30:00 | 104.50 | 2026-01-08 09:15:00 | 103.71 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-01-13 11:30:00 | 100.12 | 2026-01-14 10:15:00 | 101.60 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-01-28 10:15:00 | 103.39 | 2026-01-28 12:15:00 | 104.90 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-02-11 13:15:00 | 104.76 | 2026-02-12 10:15:00 | 104.10 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2026-02-13 09:15:00 | 102.35 | 2026-02-17 10:15:00 | 104.70 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2026-03-17 09:15:00 | 120.63 | 2026-03-18 10:15:00 | 123.67 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2026-03-17 10:15:00 | 120.98 | 2026-03-18 10:15:00 | 123.67 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2026-03-17 11:00:00 | 120.90 | 2026-03-18 10:15:00 | 123.67 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-03-17 14:15:00 | 120.78 | 2026-03-18 10:15:00 | 123.67 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2026-03-20 12:15:00 | 119.40 | 2026-03-23 09:15:00 | 113.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:30:00 | 119.25 | 2026-03-23 09:15:00 | 113.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 119.40 | 2026-03-24 10:15:00 | 112.10 | STOP_HIT | 0.50 | 6.11% |
| SELL | retest2 | 2026-03-20 13:30:00 | 119.25 | 2026-03-24 10:15:00 | 112.10 | STOP_HIT | 0.50 | 6.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 112.06 | 2026-04-06 12:15:00 | 114.55 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2026-04-02 09:15:00 | 110.86 | 2026-04-06 12:15:00 | 114.55 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2026-04-02 14:30:00 | 112.20 | 2026-04-06 12:15:00 | 114.55 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2026-04-06 11:00:00 | 112.62 | 2026-04-06 12:15:00 | 114.55 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2026-04-13 10:15:00 | 121.00 | 2026-04-17 09:15:00 | 133.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-28 09:15:00 | 127.33 | 2026-04-28 12:15:00 | 129.39 | STOP_HIT | 1.00 | -1.62% |
